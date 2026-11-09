"""Ground-truth retrieval evaluation for the inverse-generative matcher.

Two evaluation regimes:

- ``evaluate_manifest``      : explicit queries (host files + expected pattern ids).
- ``evaluate_self_retrieval``: each pattern's own ``original_code`` is used as a
  host; the matcher should retrieve the pattern itself at the top. Because host
  regions are built by the SAME closure operator and relation detectors that
  built the pattern, self-retrieval is the calibration anchor of the whole
  design (a bug's original reproducer must re-find its own pattern).
"""

from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any

from .library import ShannonLibrary
from .matcher import ShannonMatcher
from .parser import parse_java
from .relations import HostAnalysis


def load_queries(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        value = json.loads(text)
        records = value.get("queries", []) if isinstance(value, dict) else value
    if not isinstance(records, list):
        raise ValueError("evaluation manifest must contain a list of queries")
    return [record for record in records if isinstance(record, dict)]


def _rank_metrics(
    details: list[dict[str, Any]],
    prefilter_hits: int,
    prefilter_limit: int,
    top_k: int,
    library: ShannonLibrary,
    manifest: str,
) -> dict[str, Any]:
    reciprocal_rank_sum = 0.0
    hits = {1: 0, 5: 0, 10: 0}
    group_reciprocal = 0.0
    group_hits = {1: 0, 5: 0, 10: 0}
    group_retrieved = 0
    for detail in details:
        rank = detail["rank"]
        if rank is not None:
            reciprocal_rank_sum += 1.0 / rank
            for cutoff in hits:
                hits[cutoff] += int(rank <= cutoff)
        group_rank = detail.get("signature_group_rank")
        if group_rank is not None:
            group_retrieved += 1
            group_reciprocal += 1.0 / group_rank
            for cutoff in group_hits:
                group_hits[cutoff] += int(group_rank <= cutoff)
    total = len(details)
    return {
        "manifest": manifest,
        "query_count": total,
        "library_patterns": len(library.patterns),
        "prefilter_limit": prefilter_limit,
        "top_k": top_k,
        "metrics": {
            "prefilter_recall": prefilter_hits / total if total else 0.0,
            "recall_at_1": hits[1] / total if total else 0.0,
            "recall_at_5": hits[5] / total if total else 0.0,
            "recall_at_10": hits[10] / total if total else 0.0,
            "mrr": reciprocal_rank_sum / total if total else 0.0,
            # Identity up to signature equivalence: a hit is any pattern whose
            # kind/edge signature equals the expected one. Patterns inside one
            # collision group are indistinguishable by construction, so strict
            # identity retrieval is information-theoretically impossible there.
            # Misses count as zero, exactly as in the strict metrics.
            "signature_group_retrieved": group_retrieved,
            "signature_group_recall_at_1": group_hits[1] / total if total else 0.0,
            "signature_group_recall_at_5": group_hits[5] / total if total else 0.0,
            "signature_group_recall_at_10": group_hits[10] / total if total else 0.0,
            "signature_group_mrr": group_reciprocal / total if total else 0.0,
        },
        "queries": details,
    }


def evaluate_manifest(
    library: ShannonLibrary,
    manifest_path: Path,
    *,
    prefilter_limit: int = 200,
    top_k: int = 10,
    pattern_type: str = "",
) -> dict[str, Any]:
    matcher = ShannonMatcher(library)
    details: list[dict[str, Any]] = []
    prefilter_hits = 0

    for index, query in enumerate(load_queries(manifest_path), 1):
        raw_path = Path(str(query.get("query_file") or query.get("query") or ""))
        query_path = raw_path if raw_path.is_absolute() else manifest_path.parent / raw_path
        expected_value = query.get("expected_bug_ids") or query.get("expected_bug_id") or ()
        if isinstance(expected_value, str):
            expected = {expected_value}
        else:
            expected = {str(item) for item in expected_value}
        if not expected:
            raise ValueError(f"query {index} has no expected_bug_id(s)")
        source = query_path.read_text(encoding="utf-8", errors="replace")
        tree = parse_java(source, query_path.name)
        analysis = HostAnalysis(tree)
        prefiltered = matcher.prefilter(analysis, prefilter_limit)
        prefilter_ids = [pattern.pattern_id for pattern, _score in prefiltered]
        prefilter_hit = any(bug_id in expected for bug_id in prefilter_ids)
        prefilter_hits += int(prefilter_hit)
        matches = matcher.match(
            analysis,
            prefilter_limit=prefilter_limit,
            top_k=top_k,
            pattern_type=pattern_type or str(query.get("pattern_type") or ""),
        )
        ranked_ids = [match.pattern.pattern_id for match in matches]
        rank = next((position for position, bug_id in enumerate(ranked_ids, 1) if bug_id in expected), None)
        expected_signatures = {
            library.signature_hash(bug_id) for bug_id in expected if bug_id in library.patterns
        }
        group_rank = next(
            (
                position for position, bug_id in enumerate(ranked_ids, 1)
                if library.signature_hash(bug_id) in expected_signatures
            ),
            None,
        )
        details.append(
            {
                "query_file": str(query_path),
                "expected_bug_ids": sorted(expected),
                "prefilter_hit": prefilter_hit,
                "rank": rank,
                "signature_group_rank": group_rank,
                "top_bug_ids": ranked_ids,
                "top_scores": [round(match.final_score, 6) for match in matches],
            }
        )
    return _rank_metrics(details, prefilter_hits, prefilter_limit, top_k, library, str(manifest_path))


def _has_duplicated_nodes(pattern) -> bool:
    """Extraction artifact: the same snippet recorded as several pattern nodes.

    Some historical reproducers were duplicated inside ``extracted_code``; the
    extractor then recorded the duplicated copies as separate nodes, producing
    'double' patterns no single host anchor region can realize. Such patterns
    are excluded from clean-subset calibration and counted separately.
    """
    snippets = [node.source_code for node in pattern.nodes if node.source_code.strip()]
    return len(snippets) != len(set(snippets))


def evaluate_self_retrieval(
    library: ShannonLibrary,
    *,
    sample: int = 0,
    seed: int = 0,
    prefilter_limit: int = 200,
    top_k: int = 10,
    pattern_type: str = "",
    clean_only: bool = False,
) -> dict[str, Any]:
    """Feed every (or sampled) pattern's original reproducer back as the host."""
    matcher = ShannonMatcher(library)
    patterns = sorted(library.patterns.values(), key=lambda pattern: pattern.pattern_id)
    if pattern_type:
        patterns = [pattern for pattern in patterns if pattern.pattern_type == pattern_type]
    duplicated = [pattern for pattern in patterns if _has_duplicated_nodes(pattern)]
    if clean_only:
        patterns = [pattern for pattern in patterns if not _has_duplicated_nodes(pattern)]
    if sample and sample < len(patterns):
        rng = random.Random(seed)
        patterns = sorted(rng.sample(patterns, sample), key=lambda pattern: pattern.pattern_id)

    details: list[dict[str, Any]] = []
    prefilter_hits = 0
    skipped = 0
    score_sum = 0.0
    self_rank_sum = 0
    top1_component_sum: dict[str, float] = {}
    roundtrip_total = 0
    roundtrip_full = 0

    for pattern in patterns:
        source = pattern.original_code
        if not source or len(source.strip()) < 20:
            skipped += 1
            continue
        try:
            tree = parse_java(source, f"{pattern.pattern_id}.java")
        except (RuntimeError, OSError, ValueError):
            skipped += 1
            continue
        # Extraction roundtrip: does the published reproducer still contain
        # every recorded node snippet? Library artifacts (duplicated node
        # recordings, trimmed reproducers) bound what self-retrieval can reach.
        snippets = [node.source_code for node in pattern.nodes if node.source_code.strip()]
        if snippets:
            roundtrip_total += 1
            if all(snippet in source for snippet in snippets):
                roundtrip_full += 1
        analysis = HostAnalysis(tree)
        prefiltered = matcher.prefilter(analysis, prefilter_limit)
        prefilter_hit = pattern.pattern_id in {item.pattern_id for item, _score in prefiltered}
        prefilter_hits += int(prefilter_hit)
        matches = matcher.match(analysis, prefilter_limit=prefilter_limit, top_k=top_k)
        ranked_ids = [match.pattern.pattern_id for match in matches]
        rank = next(
            (position for position, bug_id in enumerate(ranked_ids, 1) if bug_id == pattern.pattern_id),
            None,
        )
        signature = library.signature_hash(pattern.pattern_id)
        group_rank = next(
            (
                position for position, bug_id in enumerate(ranked_ids, 1)
                if library.signature_hash(bug_id) == signature
            ),
            None,
        )
        self_match = next((match for match in matches if match.pattern.pattern_id == pattern.pattern_id), None)
        if self_match is not None:
            score_sum += self_match.final_score
            self_rank_sum += 1
            for key, value in self_match.components.items():
                top1_component_sum[key] = top1_component_sum.get(key, 0.0) + value
        details.append(
            {
                "bug_id": pattern.pattern_id,
                "pattern_type": pattern.pattern_type,
                "prefilter_hit": prefilter_hit,
                "rank": rank,
                "signature_group_rank": group_rank,
                "top_bug_ids": ranked_ids[:10],
                "top_scores": [round(match.final_score, 6) for match in matches[:10]],
                "self_match_score": round(self_match.final_score, 6) if self_match else None,
                "host_anchor_kind": self_match.region.anchor_kind if self_match else None,
            }
        )

    report = _rank_metrics(details, prefilter_hits, prefilter_limit, top_k, library, "<self-retrieval>")
    report["skipped"] = skipped
    report["clean_only"] = bool(clean_only)
    report["self_retrieval"] = {
        "matched_self_count": self_rank_sum,
        "mean_self_match_score": round(score_sum / self_rank_sum, 6) if self_rank_sum else None,
        "mean_score_components": {
            key: round(value / self_rank_sum, 6) for key, value in sorted(top1_component_sum.items())
        } if self_rank_sum else {},
        "extraction_roundtrip": {
            "checked": roundtrip_total,
            "all_node_snippets_in_reproducer": roundtrip_full,
            "fraction": round(roundtrip_full / roundtrip_total, 6) if roundtrip_total else None,
        },
        "duplicated_node_patterns_excluded": len(duplicated),
    }
    return report
