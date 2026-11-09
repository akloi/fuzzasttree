"""Inverse-generative anomaly matching (IGM): pattern -> host anchor alignment.

The matcher inverts each phase of the ast_shannon extractor so that pattern and
host live in one coordinate system (javac kinds, 18 typed relations, KL-weighted
features, extraction roles):

    extraction phase            matching stage (this module)
    -----------------           -------------------------------------------
    P1 sink location        <-  anchor proposal (core-kind occurrences,
                                Shannon-refined when oversized)
    P2 dependency closure   <-  region alignment (the SAME closure operator run
                                on the host; typed-edge cocktail comparison)
    P3 DIE guardrails       <-  feasibility gates (enclosing-decl containment,
                                span limits) + soft context support
    P4 information weights  <-  KL/IDF-weighted feature confirmation

Composite score for one (pattern, anchor) pair:

    S(p, a) = Gate(p, a) . Q(p) . ( l1*Cov + l2*Str + l3*Sem )
      Cov = 0.5 * role-weighted kind-IDF coverage       (prefilter signal 1)
        + 0.5 * IDF-weighted relation-vocabulary recall (prefilter signal 2)
      Str = role-weighted kind-multiset Jaccard        (structural alignment)
      Sem = 0.65*EdgeIDF-multiset Jaccard              (typed-relation cocktail)
          + 0.35*KL-weighted feature recall            (information-theoretic)
      l1/l2/l3 = 0.15 / 0.45 / 0.40

The prefilter screens on the library's two indexed signals (kinds, then typed
relations) -- filters, not evidence. Structure is necessary but not
discriminative (many patterns share kinds), and the typed-relation cocktail
is what actually distinguishes a Multi-Node Synergy pattern from a bag of kinds
-- hence the semantic term carries the largest single weight.
"""

from __future__ import annotations

import math
import re
from typing import Any

from .library import ShannonLibrary
from .models import CORE_TRIGGER, HostRegion, MatchResult, ShannonPattern
from .relations import HostAnalysis, SINK_CANDIDATE_SET, TYPE_DECL_KINDS as _TYPE_DECL_TUPLE

TYPE_DECL_KINDS = frozenset(_TYPE_DECL_TUPLE)


# Score composition weights (see module docstring).
SCORE_LAMBDAS = {"coverage": 0.15, "structure": 0.45, "semantics": 0.40}
SEMANTIC_MIX = {"edge_cocktail": 0.65, "feature_recall": 0.35}
CONTEXT_KIND_BETA = 0.5  # context kinds count half in coverage

EXCLUDED_ANCHOR_KINDS = {"COMPILATION_UNIT", "IMPORT", "PACKAGE", "MODIFIERS", "MODULE"}

# Kinds present in nearly every fragment; they never carry an anomaly mechanism
# on their own (used by retention validation in the pipeline as well).
TRIVIAL_KINDS = {
    "IDENTIFIER", "MODIFIERS", "BLOCK", "EXPRESSION_STATEMENT", "PARENTHESIZED",
    "METHOD", "CLASS", "METHOD_INVOCATION", "RETURN", "EMPTY_STATEMENT",
    "COMPILATION_UNIT", "IMPORT", "PACKAGE",
}

TOKEN_RE = re.compile(r"[A-Za-z_$][A-Za-z0-9_$-]{2,}")
STOP_WORDS = {"and", "are", "compiler", "error", "from", "java", "javac", "not", "the", "this", "with"}


class ShannonMatcher:
    def __init__(self, library: ShannonLibrary) -> None:
        self.library = library

    # -- stage 1: two-signal prefilter (kinds + typed relations) ----------------

    def prefilter(self, analysis: HostAnalysis, limit: int = 200) -> list[tuple[ShannonPattern, float]]:
        """Kind-IDF coverage x edge-IDF coverage: the cheap two-signal screen.

        The library indexes two orthogonal retrieval signals, and the prefilter
        uses both: role-weighted kind-IDF coverage (syntactic containment) and
        IDF-weighted typed-relation coverage (does the host realize the
        pattern's relation vocabulary, with multiplicity?). Patterns with no
        edges are neutral on the second signal (0.5), so Single-Point patterns
        survive on kinds alone while Multi-Node Synergy patterns whose exact
        relation cocktail is present rank at the top — a generic {CLASS,
        VARIABLE} pattern can no longer be crowded out by kind-only coverage,
        and an edge-rich pattern can no longer hide behind common kinds.
        """
        query_kinds = analysis.tree.kinds()
        query_edges = dict(analysis.relation_type_multiset)
        pool = self.library.candidates(query_kinds) | self.library.edge_candidates(query_edges)
        ranked: list[tuple[ShannonPattern, float]] = []
        for pattern_id in pool:
            pattern = self.library.patterns[pattern_id]
            kind_coverage = self._kind_coverage(pattern, query_kinds)
            edge_coverage = self._edge_coverage(pattern, query_edges)
            score = 0.5 * kind_coverage + 0.5 * edge_coverage
            if score > 0.0:
                ranked.append((pattern, score))
        ranked.sort(key=lambda item: (-item[1], -item[0].corpus_rarity, item[0].pattern_id))
        return ranked[:limit] if limit > 0 else ranked

    def _kind_coverage(self, pattern: ShannonPattern, query_kinds: set[str]) -> float:
        """Role-weighted, IDF-weighted coverage of the pattern's kinds in the query."""
        denominator = 0.0
        numerator = 0.0
        for kind in pattern.core_kinds:
            weight = self.library.kind_idf(kind)
            denominator += weight
            if kind in query_kinds:
                numerator += weight
        for kind in pattern.context_kinds:
            if kind in pattern.core_kinds:
                continue
            weight = CONTEXT_KIND_BETA * self.library.kind_idf(kind)
            denominator += weight
            if kind in query_kinds:
                numerator += weight
        return numerator / denominator if denominator else 0.0

    def _edge_coverage(self, pattern: ShannonPattern, query_edges: dict[str, int]) -> float:
        """IDF-weighted recall of the pattern's relation vocabulary (with multiplicity)."""
        denominator = 0.0
        numerator = 0.0
        for edge_type, count in pattern.edge_type_multiset.items():
            weight = count * self.library.edge_idf(edge_type)
            denominator += weight
            host_count = query_edges.get(edge_type, 0)
            if host_count:
                numerator += weight * min(1.0, host_count / count)
        return numerator / denominator if denominator else 0.5

    # -- stage 2+3: anchor proposal, region alignment, scoring ------------------

    def match(
        self,
        analysis: HostAnalysis,
        *,
        prefilter_limit: int = 200,
        top_k: int = 20,
        min_score: float = 0.0,
        candidates_per_pattern: int = 3,
        max_span_bytes: int = 0,
        max_span_fraction: float = 0.0,
        pattern_type: str = "",
    ) -> list[MatchResult]:
        results: list[MatchResult] = []
        for pattern, coverage in self.prefilter(analysis, prefilter_limit):
            if pattern_type and pattern.pattern_type != pattern_type:
                continue
            best: MatchResult | None = None
            for region in self.propose_regions(
                analysis, pattern, count=candidates_per_pattern,
                max_span_bytes=max_span_bytes, max_span_fraction=max_span_fraction,
            ):
                result = self.score(pattern, analysis, region, coverage)
                if not result.feasible:
                    continue  # keep probing anchors; DIE-infeasible sites cannot host the pattern
                if best is None or result.final_score > best.final_score:
                    best = result
            if best is not None and best.final_score >= min_score:
                results.append(best)
        results.sort(key=lambda item: (-item.final_score, item.pattern.pattern_id))
        return results[:top_k] if top_k > 0 else results

    def match_anchors(
        self,
        analysis: HostAnalysis,
        *,
        candidates_per_pattern: int = 3,
        prefilter_limit: int = 200,
        top_k: int = 20,
        min_score: float = 0.0,
        max_span_bytes: int = 0,
        max_span_fraction: float = 0.0,
        pattern_type: str = "",
    ) -> dict[int, list[MatchResult]]:
        """Per-anchor match groups for same-node and cross-node fusion."""
        groups: dict[int, list[MatchResult]] = {}
        for pattern, coverage in self.prefilter(analysis, prefilter_limit):
            if pattern_type and pattern.pattern_type != pattern_type:
                continue
            for region in self.propose_regions(
                analysis, pattern, count=candidates_per_pattern,
                max_span_bytes=max_span_bytes, max_span_fraction=max_span_fraction,
            ):
                result = self.score(pattern, analysis, region, coverage)
                if not result.feasible or result.final_score < min_score:
                    continue
                groups.setdefault(region.anchor_id, []).append(result)
        for group in groups.values():
            group.sort(key=lambda item: (-item.final_score, item.pattern.pattern_id))
        if top_k > 0:
            groups = {anchor: group[:top_k] for anchor, group in groups.items()}
        return groups

    # -- anchor proposal (inverts P1 sink location) ------------------------------

    def propose_regions(
        self,
        analysis: HostAnalysis,
        pattern: ShannonPattern,
        count: int = 1,
        max_span_bytes: int = 0,
        max_span_fraction: float = 0.0,
    ) -> list[HostRegion]:
        """Propose up to ``count`` anchor regions whose closure is then aligned.

        Seeds are host occurrences of the pattern's CORE kinds, lifted to
        statement anchors (the same anchoring the extractor used for relation
        endpoints). Oversized anchors descend to the highest-Shannon statement
        descendant, exactly like the extractor's sink refinement.
        """
        tree = analysis.tree
        core_kinds = pattern.core_kinds or pattern.required_kinds
        expected_size = sum(len(node.source_code.encode("utf-8")) for node in pattern.nodes
                            if node.role == CORE_TRIGGER) or 1

        seed_hits: dict[int, set[str]] = {}
        for kind in core_kinds:
            for occurrence in tree.occurrences(kind):
                anchor = analysis.graph.anchor_statement(occurrence)
                if anchor is None or tree.nodes[anchor].raw_kind in EXCLUDED_ANCHOR_KINDS:
                    continue
                anchor = refine_oversize_anchor(analysis, anchor)
                if anchor is None or tree.nodes[anchor].raw_kind in EXCLUDED_ANCHOR_KINDS:
                    continue
                # P3 guardrail as a proposal gate: an anchor outside the pattern's
                # required enclosing declarations can never host it, so it must
                # not consume one of the few candidate slots.
                feasible, _reasons = die_feasible(pattern, analysis, anchor)
                if not feasible:
                    continue
                seed_hits.setdefault(anchor, set()).add(kind)

        ranked: list[tuple[tuple, int]] = []
        for anchor, _kinds in seed_hits.items():
            node = tree.nodes[anchor]
            if not span_allowed(analysis, node, max_span_bytes, max_span_fraction):
                continue
            subtree_kinds = analysis.subtree_kind_counts(anchor).keys()
            core_hit = sum(1 for kind in core_kinds if kind in subtree_kinds)
            primary_match = node.raw_kind == pattern.primary_core_kind
            size = node.end_byte - node.start_byte
            key = (
                -core_hit,
                0 if primary_match else 1,
                abs(size - expected_size),
                size,
                node.start_byte,
                anchor,
            )
            ranked.append((key, anchor))
        ranked.sort(key=lambda item: item[0])

        regions: list[HostRegion] = []
        for _key, anchor in ranked:
            if len(regions) >= count:
                break
            region = build_region(analysis, anchor, pattern)
            if region is not None:
                regions.append(region)
        return regions

    # -- composite scoring (inverts P2 closure + P3 DIE + P4 weighting) ----------

    def score(
        self,
        pattern: ShannonPattern,
        analysis: HostAnalysis,
        region: HostRegion,
        coverage: float,
    ) -> MatchResult | None:
        feasible, reasons = die_feasible(pattern, analysis, region.anchor_id)
        if not feasible:
            return MatchResult(
                pattern=pattern,
                region=region,
                prefilter_score=coverage,
                structural_score=0.0,
                feature_score=0.0,
                quality_score=quality_factor(pattern),
                specificity_score=specificity_factor(self.library, pattern),
                final_score=0.0,
                feasible=False,
                feasibility_reasons=tuple(reasons),
                components={"coverage": coverage, "gate": 0.0},
            )

        kind_sim = weighted_jaccard(pattern.kind_role_multiset, region.kind_role_multiset)
        edge_sim = self.edge_cocktail_similarity(pattern, region)
        feat_sim, feat_evidence = self.feature_recall(pattern, region)
        semantics = SEMANTIC_MIX["edge_cocktail"] * edge_sim + SEMANTIC_MIX["feature_recall"] * feat_sim

        quality = quality_factor(pattern)
        specificity = specificity_factor(self.library, pattern)
        raw = (
            SCORE_LAMBDAS["coverage"] * coverage
            + SCORE_LAMBDAS["structure"] * kind_sim
            + SCORE_LAMBDAS["semantics"] * semantics
        )
        final = max(0.0, min(1.0, raw * quality * specificity))
        components = {
            "coverage": coverage,
            "kind_alignment": kind_sim,
            "edge_cocktail": edge_sim,
            "feature_recall": feat_sim,
            "semantics": semantics,
            "quality": quality,
            "specificity": specificity,
        }
        return MatchResult(
            pattern=pattern,
            region=region,
            prefilter_score=coverage,
            structural_score=kind_sim,
            feature_score=feat_sim,
            quality_score=quality,
            specificity_score=specificity,
            final_score=final,
            feasible=True,
            components=components,
            feature_evidence=feat_evidence,
        )

    def edge_cocktail_similarity(self, pattern: ShannonPattern, region: HostRegion) -> float:
        """IDF-weighted Jaccard over the typed-relation multiset.

        Rare relations (SWITCH_EXHAUSTIVENESS, RECORD_DECONSTRUCTION, ...) carry
        almost all of the anomaly identity; DEF_USE is common and cheap.
        """
        left = {
            edge_type: count * self.library.edge_idf(edge_type)
            for edge_type, count in pattern.edge_type_multiset.items()
        }
        right = {
            edge_type: count * self.library.edge_idf(edge_type)
            for edge_type, count in region.edge_type_multiset.items()
        }
        return weighted_jaccard(left, right)

    def feature_recall(self, pattern: ShannonPattern, region: HostRegion) -> tuple[float, dict[str, Any]]:
        """KL/IDF-weighted recall of the pattern's features in the region.

        Weights are squashed through w/(1+w) so IDF/KL magnitudes become
        comparable in [0, 1). A pattern with no features is neutral (0.5).
        """
        weights = {name: self.library.feature_weight(name) for name, _w in pattern.features}
        if not weights:
            return 0.5, {"neutral": True, "matched": [], "missing": []}
        unit = {name: weight / (1.0 + abs(weight)) for name, weight in weights.items()}
        total = sum(unit.values())
        region_features = set(region.features)
        matched = sorted(name for name in unit if name in region_features)
        missing = sorted(name for name in unit if name not in region_features)
        score = sum(unit[name] for name in matched) / total if total else 0.0
        evidence = {
            "score": round(score, 6),
            "matched": matched,
            "missing": missing,
            "region_features": sorted(region_features),
            "weighting": self.library.feature_weighting,
        }
        return score, evidence


# ---------------------------------------------------------------------------
# Region construction
# ---------------------------------------------------------------------------

def build_region(analysis: HostAnalysis, anchor_id: int, pattern: ShannonPattern) -> HostRegion | None:
    """Run the extractor's closure operator around the anchor and sign it."""
    tree = analysis.tree
    roles, edges = analysis.graph.region(anchor_id)
    if not roles:
        return None
    kind_multiset, edge_multiset, features = analysis.region_signature(roles, edges)
    anchor_kinds = analysis.subtree_kind_counts(anchor_id)
    covered = tuple(kind for kind in pattern.core_kinds if kind in anchor_kinds)
    missing = tuple(kind for kind in pattern.core_kinds if kind not in anchor_kinds)
    node = tree.nodes[anchor_id]
    return HostRegion(
        anchor_id=anchor_id,
        region_ids=tuple(sorted(roles)),
        roles=dict(roles),
        edges=tuple(edges),
        anchor_kind=node.raw_kind,
        start_byte=node.start_byte,
        end_byte=node.end_byte,
        kind_role_multiset=kind_multiset,
        edge_type_multiset=edge_multiset,
        features=tuple(sorted(features)),
        covered_core_kinds=covered,
        missing_core_kinds=missing,
    )


def refine_oversize_anchor(analysis: HostAnalysis, anchor_id: int) -> int:
    """Descend to the highest-Shannon statement descendant when the anchor is huge."""
    tree = analysis.tree
    limit = max(6, int(0.6 * len(tree.nodes)))
    if analysis.subtree_size(anchor_id) <= limit:
        return anchor_id
    best, best_se = anchor_id, -1.0
    descendants = tree.descendants(anchor_id)
    for nid in descendants:
        node = tree.nodes[nid]
        if node.raw_kind not in SINK_CANDIDATE_SET:
            continue
        size = analysis.subtree_size(nid)
        if not 3 <= size <= limit:
            continue
        se = analysis.subtree_entropy(nid)
        if se > best_se:
            best, best_se = nid, se
    return best


def span_allowed(analysis: HostAnalysis, node: Any, max_span_bytes: int, max_span_fraction: float) -> bool:
    span = node.end_byte - node.start_byte
    if max_span_bytes and span > max_span_bytes:
        return False
    if max_span_fraction:
        total = max(len(analysis.tree.source_bytes), 1)
        if span / total > max_span_fraction:
            return False
    return True


def die_feasible(pattern: ShannonPattern, analysis: HostAnalysis, anchor_id: int) -> tuple[bool, list[str]]:
    """Hard DIE gate: the pattern's structural_context containment requirements.

    The extractor recorded which enclosing declaration kinds the CORE nodes
    must stay inside (CONTEXT declaration nodes). A mutation anchored outside
    such a scope cannot satisfy the guardrail, so the pair is infeasible.
    """
    reasons: list[str] = []
    ancestor_kinds = {
        analysis.tree.nodes[node_id].raw_kind
        for node_id in analysis.tree.ancestors(anchor_id, include_self=False)
    }
    for kind in pattern.enclosing_decl_kinds:
        if kind in ancestor_kinds:
            continue
        if kind in TYPE_DECL_KINDS and (ancestor_kinds & TYPE_DECL_KINDS):
            continue  # any enclosing type declaration satisfies the containment
        reasons.append(
            f"pattern requires CORE inside an enclosing {kind}; "
            f"anchor ancestors are {sorted(ancestor_kinds) or ['<top level>']}"
        )
    return (not reasons), reasons


# ---------------------------------------------------------------------------
# Pattern-level factors
# ---------------------------------------------------------------------------

def quality_factor(pattern: ShannonPattern) -> float:
    """Extraction-confidence multiplier: synergy patterns and strongly weighted
    rare features are more trustworthy transplant sources."""
    multi_node = 1.0 if pattern.edge_type_count else 0.0
    top_weight = max((abs(weight) for _name, weight in pattern.features), default=0.0)
    weight_factor = min(top_weight / 6.0, 1.0)
    return max(0.5, min(1.0, 0.5 + 0.3 * multi_node + 0.2 * weight_factor))


def specificity_factor(library: ShannonLibrary, pattern: ShannonPattern) -> float:
    """Rarity multiplier: specific patterns (many nodes, many distinct relations,
    few signature collisions) transfer more information."""
    node_factor = min(pattern.node_count / 6.0, 1.0)
    edge_factor = min(pattern.edge_type_count / 3.0, 1.0)
    collision_factor = 1.0 / math.sqrt(max(pattern.collision_count, 1))
    score = 0.45 + 0.20 * node_factor + 0.20 * edge_factor + 0.15 * collision_factor
    if pattern.edge_type_count == 0:
        score = min(score, 0.68)  # single-point patterns are inherently generic
    return max(0.45, min(1.0, score))


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

def weighted_jaccard(left: dict[str, float], right: dict[str, float]) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 0.0
    numerator = sum(min(left.get(key, 0.0), right.get(key, 0.0)) for key in keys)
    denominator = sum(max(left.get(key, 0.0), right.get(key, 0.0)) for key in keys)
    return numerator / denominator if denominator else 0.0


def token_set(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text or "") if token.lower() not in STOP_WORDS}


def token_jaccard(left: str, right: str) -> float:
    a, b = token_set(left), token_set(right)
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)
