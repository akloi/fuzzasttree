"""Fusion planning: same-anchor pattern pairing and cross-anchor interaction graphs.

Two fusion regimes are supported, both built on the same matched anchors:

- ``same_node``  — ONE host anchor receives TWO patterns (single-node,
   multi-anomaly mutation). Pairs are typed by AST-slot operations and scored
   by match strength x structural non-conflict x mechanism orthogonality.
- ``cross_node`` — TWO anchors receive one pattern each (multi-node,
   multi-anomaly mutation). Anchors are coupled by an interaction graph whose
   edges come from host STRUCTURE (sibling slots, control-flow branches,
   ancestor-descendant) *and* from the host's own derived SEMANTIC relations
   (a DEF_USE / LAMBDA_CAPTURE / ... edge between the two anchor regions is
   direct evidence the two anomalies will interact after transplantation).

Orthogonality is measured in the pattern signal space: typed-edge multisets,
KL-weighted feature sets, and bug-report summary tokens. Maximizing it spreads
fused anomalies across mechanism families (diversity), which the selector then
enforces globally via decayed greedy b-matching.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations, product
import math
import re

from .matcher import token_set
from .models import AnchorEdge, FusionPlan, MatchResult
from .relations import HostAnalysis


EXPRESSION_KINDS = {
    "IDENTIFIER", "MEMBER_SELECT", "METHOD_INVOCATION", "MEMBER_REFERENCE", "LAMBDA_EXPRESSION",
    "TYPE_CAST", "PARENTHESIZED", "CONDITIONAL_EXPRESSION", "NEW_CLASS", "NEW_ARRAY",
    "ARRAY_ACCESS", "ASSIGNMENT", "INSTANCE_OF", "SWITCH_EXPRESSION",
    "INT_LITERAL", "LONG_LITERAL", "FLOAT_LITERAL", "DOUBLE_LITERAL", "BOOLEAN_LITERAL",
    "CHAR_LITERAL", "STRING_LITERAL", "NULL_LITERAL",
    "POSTFIX_INCREMENT", "POSTFIX_DECREMENT", "PREFIX_INCREMENT", "PREFIX_DECREMENT",
    "UNARY_PLUS", "UNARY_MINUS", "BITWISE_COMPLEMENT", "LOGICAL_COMPLEMENT",
    "MULTIPLY", "DIVIDE", "REMAINDER", "PLUS", "MINUS", "LEFT_SHIFT", "RIGHT_SHIFT",
    "UNSIGNED_RIGHT_SHIFT", "LESS_THAN", "GREATER_THAN", "LESS_THAN_EQUAL", "GREATER_THAN_EQUAL",
    "EQUAL_TO", "NOT_EQUAL_TO", "AND", "XOR", "OR", "CONDITIONAL_AND", "CONDITIONAL_OR",
    "MULTIPLY_ASSIGNMENT", "DIVIDE_ASSIGNMENT", "REMAINDER_ASSIGNMENT", "PLUS_ASSIGNMENT",
    "MINUS_ASSIGNMENT", "LEFT_SHIFT_ASSIGNMENT", "RIGHT_SHIFT_ASSIGNMENT",
    "UNSIGNED_RIGHT_SHIFT_ASSIGNMENT", "AND_ASSIGNMENT", "XOR_ASSIGNMENT", "OR_ASSIGNMENT",
}
STATEMENT_KINDS = {
    "BLOCK", "EXPRESSION_STATEMENT", "IF", "FOR_LOOP", "ENHANCED_FOR_LOOP", "WHILE_LOOP",
    "DO_WHILE_LOOP", "TRY", "SYNCHRONIZED", "SWITCH", "RETURN", "THROW", "ASSERT",
    "BREAK", "CONTINUE", "EMPTY_STATEMENT", "LABELED_STATEMENT", "YIELD",
}
EXCLUDED_ANCHORS = {
    "COMPILATION_UNIT", "IMPORT", "PACKAGE", "CLASS", "METHOD", "INTERFACE", "ENUM",
    "ANNOTATION_TYPE", "RECORD", "MODULE", "MODIFIERS",
}
WRAPPER_KINDS = {"TYPE_CAST", "PARENTHESIZED", "METHOD_INVOCATION"}
STATEMENT_WRAPPERS = {
    "BLOCK", "IF", "FOR_LOOP", "ENHANCED_FOR_LOOP", "WHILE_LOOP", "DO_WHILE_LOOP",
    "SYNCHRONIZED", "TRY",
}
REFERENCE_OPERANDS = {"IDENTIFIER", "MEMBER_SELECT", "METHOD_INVOCATION", "TYPE_CAST", "NEW_CLASS", "ARRAY_ACCESS"}
CONTROL_KINDS = {"IF", "TRY", "SWITCH", "SWITCH_EXPRESSION", "FOR_LOOP", "ENHANCED_FOR_LOOP", "WHILE_LOOP", "DO_WHILE_LOOP"}

OPERATION_INSTRUCTIONS = {
    "wrapper": "Use pattern 2's core as an outer AST wrapper around pattern 1's core while retaining both backbones.",
    "role_extension": "Extend the host expression to an operand or conditional position and preserve both mechanisms.",
    "constraint_conjunction": "Make both historical type/lambda constraints act on the same target expression; retain both constraint sources.",
    "branch_splitting": "Place the two mechanisms into distinct parts of one conditional expression under the existing host context.",
    "nested_expression": "Nest pattern 1's lambda/method-reference/expression into an argument slot of pattern 2's invocation.",
    "sibling": "Coordinate both sibling replacements using their shared operator, conditional, or argument context.",
    "control_flow": "Preserve the control-flow relationship and place the two mechanisms in their respective branch or loop slots.",
    "ancestor_descendant": "Preserve the enclosing/contained relationship of the two anomaly anchors.",
    "semantic_flow": "The two anchors are already coupled by the shown host relation; keep that relation intact while transplanting each mechanism.",
}


def syntactic_role(kind: str) -> str:
    if kind in EXPRESSION_KINDS:
        return "expression"
    if kind in STATEMENT_KINDS:
        return "statement"
    return kind


DECLARATION_ANCHOR_KINDS = {"VARIABLE", "CLASS", "METHOD", "INTERFACE", "ENUM", "RECORD", "ANNOTATION_TYPE"}


def slot_role(kind: str) -> str:
    """The composable slot an anchor offers: statement, declaration, or expression."""
    if kind in STATEMENT_KINDS or kind in DECLARATION_ANCHOR_KINDS:
        return "statement"
    if kind in EXPRESSION_KINDS:
        return "expression"
    return kind


def hosting_prior(match: MatchResult) -> float:
    """Can the anchor's slot host the pattern's full mechanism verbatim?

    A pattern with several distinct core kinds (or any declaration core kind)
    needs a statement/declaration slot: only there can supporting declarations
    be grafted next to the sink snippet. At an expression slot only the sink
    snippet fits, so the mechanism backbone is structurally at risk — the plan
    is not forbidden, but it cannot outrank a statement-slot plan.
    """
    pattern = match.pattern
    anchor_kind = match.region.anchor_kind
    needs_statement_slot = len(set(pattern.core_kinds)) > 1 or bool(
        set(pattern.core_kinds) & DECLARATION_ANCHOR_KINDS
    )
    role = slot_role(anchor_kind)
    if needs_statement_slot:
        return 1.0 if role == "statement" else 0.3
    return 1.0 if role in {"statement", "expression"} else 0.3


# ---------------------------------------------------------------------------
# Mechanism distance / orthogonality (pattern signal space)
# ---------------------------------------------------------------------------

def _multiset_distance(left: dict, right: dict) -> float:
    a, b = Counter(left), Counter(right)
    union = sum((a | b).values())
    return 1.0 - sum((a & b).values()) / union if union else 0.0


def mechanism_orthogonality(first: MatchResult, second: MatchResult) -> float:
    """0.5*edge-cocktail distance + 0.3*KL-feature distance + 0.2*summary distance."""
    one, two = first.pattern, second.pattern
    edge_distance = _multiset_distance(one.edge_type_multiset, two.edge_type_multiset)
    features_a = {name for name, _w in one.features}
    features_b = {name for name, _w in two.features}
    feature_distance = (
        1.0 - len(features_a & features_b) / len(features_a | features_b)
        if features_a or features_b
        else 0.0
    )
    summary_a = token_set(one.summary)
    summary_b = token_set(two.summary)
    summary_distance = (
        1.0 - len(summary_a & summary_b) / len(summary_a | summary_b)
        if summary_a or summary_b
        else 0.0
    )
    return 0.5 * edge_distance + 0.3 * feature_distance + 0.2 * summary_distance


def _pattern_features(match: MatchResult) -> set[str]:
    return {name for name, _weight in match.pattern.features}


# ---------------------------------------------------------------------------
# Same-anchor fusion (single node, multiple anomalies)
# ---------------------------------------------------------------------------

def _operations(analysis: HostAnalysis, first: MatchResult, second: MatchResult) -> list[tuple[str, float]]:
    anchor_kind = first.region.anchor_kind
    one = first.pattern.primary_core_kind
    two = second.pattern.primary_core_kind
    role = syntactic_role(anchor_kind)
    if syntactic_role(one) != role or syntactic_role(two) != role:
        return []
    operations: list[tuple[str, float]] = []
    if role == "statement":
        if two in STATEMENT_WRAPPERS:
            operations.append(("wrapper", 0.9))
        operations.append(("sibling", 0.6))
        return operations
    if role != "expression":
        return []
    if two in WRAPPER_KINDS:
        operations.append(("wrapper", 0.9))
    if one in {"MEMBER_REFERENCE", "LAMBDA_EXPRESSION"} and two == "METHOD_INVOCATION":
        operations.append(("nested_expression", 0.9))
    if one in REFERENCE_OPERANDS:
        operations.append(("role_extension", 0.8))
    constraint_families = {
        "Lambda_Capture", "Generic_Inference", "TypeCast_Relation",
        "Flow_Scoping", "Variable_Shadowing",
    }
    if _pattern_features(first) & constraint_families and _pattern_features(second) & constraint_families:
        operations.append(("constraint_conjunction", 0.7))
    if any(
        analysis.tree.nodes[node_id].raw_kind == "CONDITIONAL_EXPRESSION"
        for node_id in analysis.tree.ancestors(first.region.anchor_id)
    ):
        operations.append(("branch_splitting", 0.7))
    return operations


def same_node_plans(
    analysis: HostAnalysis,
    groups: dict[int, list[MatchResult]],
    pairs_per_anchor: int = 4,
) -> list[FusionPlan]:
    plans: list[FusionPlan] = []
    for anchor_id, group in groups.items():
        if analysis.tree.nodes[anchor_id].raw_kind in EXCLUDED_ANCHORS:
            continue
        candidates: list[FusionPlan] = []
        for left, right in combinations(group[:4], 2):
            if left.pattern.pattern_id == right.pattern.pattern_id:
                continue
            structure_distance = _multiset_distance(
                left.pattern.kind_role_multiset, right.pattern.kind_role_multiset
            )
            type_feasibility = math.sqrt(
                max(0.0, left.components.get("feature_recall", 0.0))
                * max(0.0, right.components.get("feature_recall", 0.0))
            )
            compatibility = structure_distance * type_feasibility
            orthogonality = mechanism_orthogonality(left, right)
            base_score = math.sqrt(max(0.0, left.final_score * right.final_score))
            for first, second in ((left, right), (right, left)):
                for operation, prior in _operations(analysis, first, second):
                    score = base_score * compatibility * orthogonality * prior
                    if score <= 0:
                        continue
                    candidates.append(
                        FusionPlan("same_node", operation, first, second, None, score, {
                            "match_geometric_mean": base_score,
                            "compatibility": compatibility,
                            "transform_non_conflict": structure_distance,
                            "type_constraint_feasible": type_feasibility,
                            "mechanism_orthogonality": orthogonality,
                            "transform_composability": prior,
                        })
                    )
        candidates.sort(key=plan_sort_key)
        plans.extend(candidates[:pairs_per_anchor])
    return sorted(plans, key=plan_sort_key)


# ---------------------------------------------------------------------------
# Anchor interaction graph (host structure + host semantics)
# ---------------------------------------------------------------------------

def lowest_common_ancestor(analysis: HostAnalysis, first: int, second: int) -> int:
    result = analysis.tree.root_id
    for a, b in zip(analysis.tree.ancestors(first), analysis.tree.ancestors(second)):
        if a != b:
            break
        result = a
    return result


def disjoint_spans(first: tuple[int, int], second: tuple[int, int]) -> bool:
    return first[1] <= second[0] or second[1] <= first[0]


def _span(match: MatchResult) -> tuple[int, int]:
    return match.region.start_byte, match.region.end_byte


def build_anchor_interaction_graph(
    analysis: HostAnalysis,
    groups: dict[int, list[MatchResult]],
    anchor_limit: int = 6,
) -> list[AnchorEdge]:
    """Edges coupling two anchors: structural slots plus host semantic relations."""
    tree = analysis.tree
    roots = sorted(
        (root for root in groups if tree.nodes[root].raw_kind not in EXCLUDED_ANCHORS),
        key=lambda root: (-groups[root][0].final_score, root),
    )[:anchor_limit]
    edges: list[AnchorEdge] = []
    for a, b in combinations(roots, 2):
        pa, pb = tree.ancestors(a), tree.ancestors(b)
        lca = lowest_common_ancestor(analysis, a, b)
        na, nb = tree.nodes[a], tree.nodes[b]
        nonoverlap = disjoint_spans((na.start_byte, na.end_byte), (nb.start_byte, nb.end_byte))
        evidence = {"lca_id": lca, "lca_kind": tree.nodes[lca].raw_kind, "span_non_conflict": nonoverlap}
        if a in pb or b in pa:
            outer, inner, path = (a, b, pb) if a in pb else (b, a, pa)
            distance = len(path) - 1 - path.index(outer)
            strength = min(
                1.0,
                1.0 / (1 + distance)
                + (0.1 if any(tree.nodes[n].raw_kind in {"METHOD", "BLOCK"} for n in tree.ancestors(outer)) else 0.0),
            )
            edges.append(
                AnchorEdge("ancestor_descendant", outer, inner, strength,
                           {**evidence, "ancestor_distance": distance, "execution_exclusion": "overlapping_anchor_spans"})
            )
            continue
        if na.parent_id == nb.parent_id and na.parent_id is not None:
            parent_kind = tree.nodes[na.parent_id].raw_kind
            slot_distance = abs(na.child_index - nb.child_index)
            semantic_slots = parent_kind in EXPRESSION_KINDS
            if slot_distance == 1 or semantic_slots:
                strength = 0.95 if parent_kind == "CONDITIONAL_EXPRESSION" else (0.8 if semantic_slots else 0.6)
                edges.append(
                    AnchorEdge("sibling", a, b, strength,
                               {**evidence, "parent_id": na.parent_id, "parent_kind": parent_kind,
                                "child_indices": [na.child_index, nb.child_index]})
                )
        if tree.nodes[lca].raw_kind in CONTROL_KINDS:
            branch_a = pa[pa.index(lca) + 1]
            branch_b = pb[pb.index(lca) + 1]
            if branch_a != branch_b:
                edges.append(
                    AnchorEdge("control_flow", a, b, 0.65, {**evidence, "branch_roots": [branch_a, branch_b]})
                )

    # Semantic interaction edges: host relations whose endpoints statement-anchor
    # into two DIFFERENT matched anchors. This is evidence the two anchor regions
    # already exchange data/control in the host, so both transplanted anomalies
    # will interact through it after mutation.
    anchor_set = set(roots)
    seen_semantic = set()
    for relation in analysis.relations:
        for source, target in ((relation["source"], relation["target"]), (relation["target"], relation["source"])):
            other = target if source in anchor_set else (source if target in anchor_set else None)
            anchor = source if source in anchor_set else (target if target in anchor_set else None)
            if anchor is None or other is None:
                continue
            other_anchor = analysis.graph.anchor_statement(other)
            if other_anchor in anchor_set and other_anchor != anchor:
                key = (min(anchor, other_anchor), max(anchor, other_anchor), relation["type"])
                if key in seen_semantic:
                    continue
                seen_semantic.add(key)
                a, b = key[0], key[1]
                lca = lowest_common_ancestor(analysis, a, b)
                na, nb = tree.nodes[a], tree.nodes[b]
                edges.append(
                    AnchorEdge("semantic_flow", a, b, 0.85,
                               {
                                   "lca_id": lca,
                                   "lca_kind": tree.nodes[lca].raw_kind,
                                   "span_non_conflict": disjoint_spans((na.start_byte, na.end_byte), (nb.start_byte, nb.end_byte)),
                                   "host_relation_endpoints": [relation["source"], relation["target"]],
                               },
                               host_relation=relation["type"])
                )
    return edges


def edge_compatibility(first: MatchResult, second: MatchResult, edge: AnchorEdge) -> float:
    """Do the fused patterns speak the same relation vocabulary as the interaction?"""
    one = _pattern_features(first)
    two = _pattern_features(second)
    if edge.edge_type == "semantic_flow":
        from .relations import RELATION_TO_FEATURE

        feature = RELATION_TO_FEATURE.get(edge.host_relation, "")
        return 1.0 if feature and (feature in one or feature in two) else 0.6
    if edge.edge_type == "control_flow":
        families = {"Switch_Exhaustiveness", "Flow_Scoping", "Plain_DefUse", "ForwardReference",
                    "Pattern_Variable_Flow", "Lambda_Capture"}
        return 1.0 if (one | two) & families else 0.5
    if edge.edge_type in {"sibling", "ancestor_descendant"}:
        interacting = {"Lambda_Capture", "Generic_Inference", "TypeCast_Relation",
                       "Variable_Shadowing", "Plain_DefUse", "Early_Construction_Reference"}
        return 1.0 if one & interacting and two & interacting else 0.5
    return 0.0


def distance_prior(analysis: HostAnalysis, first: MatchResult, second: MatchResult) -> float:
    path = analysis.tree.ancestors(
        lowest_common_ancestor(analysis, first.region.anchor_id, second.region.anchor_id)
    )
    kinds = {analysis.tree.nodes[n].raw_kind for n in path}
    scope = 1.0 if "METHOD" in kinds else (0.7 if kinds & {"CLASS", "INTERFACE", "ENUM", "RECORD"} else 0.4)
    a, b = _span(first), _span(second)
    gap = max(0, max(a[0], b[0]) - min(a[1], b[1]))
    return scope * math.exp(-gap / 2000.0)


def cross_node_plans(
    analysis: HostAnalysis,
    groups: dict[int, list[MatchResult]],
    edges: list[AnchorEdge],
    pairs_per_edge: int = 4,
    max_bridge_bytes: int = 8000,
) -> list[FusionPlan]:
    plans: list[FusionPlan] = []
    for edge in edges:
        if not edge.evidence["span_non_conflict"]:
            continue
        lca = analysis.tree.nodes[edge.evidence["lca_id"]]
        if lca.raw_kind == "COMPILATION_UNIT" or lca.end_byte - lca.start_byte > max_bridge_bytes:
            continue
        if analysis.tree.snippet(lca.start_byte, lca.end_byte).strip() == analysis.tree.source.strip():
            continue
        candidates: list[FusionPlan] = []
        for first, second in product(groups[edge.source_id][:4], groups[edge.target_id][:4]):
            if first.pattern.pattern_id == second.pattern.pattern_id:
                continue
            compatible = edge_compatibility(first, second, edge)
            orthogonality = mechanism_orthogonality(first, second)
            distance = distance_prior(analysis, first, second)
            hosting = hosting_prior(first) * hosting_prior(second)
            base_score = math.sqrt(max(0.0, first.final_score * second.final_score))
            score = base_score * edge.strength * compatible * orthogonality * distance * hosting
            if score > 0:
                candidates.append(
                    FusionPlan("cross_node", edge.edge_type, first, second, edge, score, {
                        "match_geometric_mean": base_score,
                        "interaction_strength": edge.strength,
                        "edge_compatibility": compatible,
                        "mechanism_orthogonality": orthogonality,
                        "distance_prior": distance,
                        "hosting_feasibility": hosting,
                    })
                )
        candidates.sort(key=plan_sort_key)
        plans.extend(candidates[:pairs_per_edge])
    return sorted(plans, key=plan_sort_key)


def plan_sort_key(plan: FusionPlan) -> tuple:
    return (
        -plan.score,
        plan.fusion_kind,
        plan.operation,
        plan.match_1.pattern.pattern_id,
        plan.match_2.pattern.pattern_id,
        plan.match_1.region.anchor_id,
        plan.match_2.region.anchor_id,
    )


def select_fusion_plans(
    plans: list[FusionPlan],
    *,
    usage: dict[str, int] | None = None,
    per_file_cap: int = 0,
    limit: int | None = None,
) -> list[FusionPlan]:
    """Decayed greedy b-matching over already-ranked plans.

    Run-level ``usage`` counts how often each pattern has been emitted; a plan's
    priority decays by 1/sqrt((1+u1)(1+u2)) so a repeatedly emitted generic pair
    cannot keep out-scoring fresh combinations. ``per_file_cap`` bounds how many
    selected plans one pattern may join within a single input.
    """
    books = dict(usage or {})

    def priority(plan: FusionPlan) -> float:
        first = plan.match_1.pattern.pattern_id
        second = plan.match_2.pattern.pattern_id
        return -plan.score / math.sqrt((1 + books.get(first, 0)) * (1 + books.get(second, 0)))

    if not per_file_cap and limit is None:
        return sorted(plans, key=priority)
    chosen: list[FusionPlan] = []
    in_file: Counter[str] = Counter()
    for plan in sorted(plans, key=priority):
        if limit is not None and len(chosen) >= limit:
            break
        first = plan.match_1.pattern.pattern_id
        second = plan.match_2.pattern.pattern_id
        if per_file_cap and (in_file[first] >= per_file_cap or in_file[second] >= per_file_cap):
            continue
        chosen.append(plan)
        in_file[first] += 1
        in_file[second] += 1
    return chosen
