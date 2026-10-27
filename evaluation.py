"""Ground-truth retrieval evaluation for the two-stage matcher."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .library import HistoricalLibrary
from .matcher import SubtreeMatcher
from .parser import parse_java_file


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


def evaluate_manifest(
    library: HistoricalLibrary,
    manifest_path: Path,
    *,
    prefilter_limit: int = 200,
    top_k: int = 10,
) -> dict[str, Any]:
    matcher = SubtreeMatcher(library)
    details: list[dict[str, Any]] = []
    reciprocal_rank_sum = 0.0
    hits = {1: 0, 5: 0, 10: 0}
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
        tree = parse_java_file(query_path)
        prefiltered = matcher.prefilter(tree, prefilter_limit)
        prefilter_ids = [pattern.bug_id for pattern, _score in prefiltered]
        prefilter_hit = any(bug_id in expected for bug_id in prefilter_ids)
        prefilter_hits += int(prefilter_hit)
        phases = query.get("phases") or query.get("P") or ()
        if isinstance(phases, str):
            phases = [part.strip() for part in phases.split(",") if part.strip()]
        matches = matcher.match(
            tree,
            prefilter_limit=prefilter_limit,
            top_k=top_k,
            query_language_feature=str(query.get("language_feature") or query.get("L") or ""),
            query_phases=phases,
            query_diagnostic=str(query.get("diagnostic") or query.get("D") or ""),
        )
        ranked_ids = [match.pattern.bug_id for match in matches]
        rank = next((position for position, bug_id in enumerate(ranked_ids, 1) if bug_id in expected), None)
        if rank is not None:
            reciprocal_rank_sum += 1.0 / rank
            for cutoff in hits:
                hits[cutoff] += int(rank <= cutoff)
        details.append(
            {
                "query_file": str(query_path),
                "expected_bug_ids": sorted(expected),
                "prefilter_hit": prefilter_hit,
                "rank": rank,
                "top_bug_ids": ranked_ids,
                "top_scores": [round(match.final_score, 6) for match in matches],
            }
        )

    total = len(details)
    return {
        "manifest": str(manifest_path),
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
        },
        "queries": details,
    }
