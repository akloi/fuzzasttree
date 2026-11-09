"""Load and index the ast_shannon multi-dimensional anomaly pattern library."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from .models import CORE_TRIGGER, PatternEdge, PatternNode, ShannonPattern


DEFAULT_LIBRARY_DIR = Path(
    "/disk1/users/zengj/jdk-fuzzllm/javac_data/javac_data/javaccode_8635/ast_shannon/patterns"
)
DEFAULT_LIBRARY_FILE = Path(
    "/disk1/users/zengj/jdk-fuzzllm/javac_data/javac_data/javaccode_8635/ast_shannon/shannon_pattern_library.json"
)


class ShannonLibrary:
    """In-memory index over the SELFuzz anomaly pattern library.

    Indexes three orthogonal retrieval signals so the matcher can cascade:

    - kind inverted index + kind IDF       (syntactic prefilter)
    - edge-type document frequency         (typed-relation rerank weights)
    - corpus feature weights (IDF or KL)   (information-theoretic confirmation)
    """

    def __init__(
        self,
        library_dir: Path | None = None,
        library_file: Path | None = None,
        pattern_types: Iterable[str] = ("Single-Point", "Multi-Node Synergy"),
        min_nodes: int = 0,
    ) -> None:
        if library_dir is None and library_file is None:
            library_file = DEFAULT_LIBRARY_FILE
        self.library_dir = Path(library_dir) if library_dir else None
        self.library_file = Path(library_file) if library_file else None
        self.pattern_types = frozenset(pattern_types)
        self.min_nodes = min_nodes
        self.patterns: dict[str, ShannonPattern] = {}
        self.inverted_index: dict[str, set[str]] = defaultdict(set)
        self.edge_inverted_index: dict[str, set[str]] = defaultdict(set)
        self.kind_document_frequency: Counter[str] = Counter()
        self.edge_document_frequency: Counter[str] = Counter()
        self.feature_weights: dict[str, float] = {}
        self.feature_weighting: str = "IDF(bug_corpus)"
        self._load()
        self._collisions = self._signature_collisions()
        self._signature_of = {
            pattern.pattern_id: self._signature_key(pattern)
            for pattern in self.patterns.values()
        }

    def signature_hash(self, pattern_id: str) -> str:
        """Stable hash of the pattern's kind/edge signature (collision-group id)."""
        return self._signature_of[pattern_id]

    def same_signature(self, first: str, second: str) -> bool:
        """True when two patterns are indistinguishable in the matcher's evidence space."""
        return self._signature_of[first] == self._signature_of[second]

    # -- loading ---------------------------------------------------------------

    def _load(self) -> None:
        records: list[dict[str, Any]] = []
        if self.library_file is not None and self.library_file.exists():
            payload = json.loads(self.library_file.read_text(encoding="utf-8", errors="replace"))
            meta = payload.get("library_meta") or {}
            self.feature_weights = {
                str(name): float(weight)
                for name, weight in (meta.get("feature_weights") or {}).items()
            }
            self.feature_weighting = str(meta.get("feature_weighting") or self.feature_weighting)
            records = list(payload.get("patterns") or ())
        elif self.library_dir is not None:
            for path in sorted(self.library_dir.glob("JDK-*.json")):
                records.append(json.loads(path.read_text(encoding="utf-8", errors="replace")))
        else:
            raise ValueError("ShannonLibrary needs a pattern directory or a merged library file")

        for record in records:
            pattern = self._parse_record(record)
            if pattern is None:
                continue
            if pattern.pattern_id in self.patterns:
                raise ValueError(f"duplicate pattern id: {pattern.pattern_id}")
            self.patterns[pattern.pattern_id] = pattern

        if not self.patterns:
            raise ValueError("no patterns loaded from the ast_shannon library")

        for pattern in self.patterns.values():
            for kind in set(pattern.required_kinds):
                self.inverted_index[kind].add(pattern.pattern_id)
                self.kind_document_frequency[kind] += 1
            for edge_type in pattern.edge_type_multiset:
                self.edge_inverted_index[edge_type].add(pattern.pattern_id)
                self.edge_document_frequency[edge_type] += 1

        # Fall back to presence-based IDF when the merged file was not used.
        if not self.feature_weights:
            presence: Counter[str] = Counter()
            for pattern in self.patterns.values():
                for name, _weight in pattern.features:
                    presence[name] += 1
            total = len(self.patterns)
            self.feature_weights = {
                name: round(math.log(total / max(1, count)), 4) for name, count in presence.items()
            }

        for pattern in self.patterns.values():
            pattern.corpus_rarity = sum(self.kind_idf(kind) for kind in pattern.required_kinds) / max(
                1, len(pattern.required_kinds)
            )

    def _parse_record(self, record: dict[str, Any]) -> ShannonPattern | None:
        pattern_type = str(record.get("pattern_type") or "")
        if pattern_type not in self.pattern_types:
            return None
        nodes_raw = record.get("nodes")
        if not isinstance(nodes_raw, list) or not nodes_raw:
            return None
        nodes = [
            PatternNode(
                node_id=str(node.get("node_id") or f"N{index}"),
                kind=str(node.get("kind") or "UNKNOWN"),
                source_code=str(node.get("source_code") or ""),
                role=str(node.get("role") or "DEPENDENT"),
            )
            for index, node in enumerate(nodes_raw)
        ]
        if len(nodes) < self.min_nodes:
            return None
        edges = [
            PatternEdge(
                source=str(edge.get("source") or ""),
                target=str(edge.get("target") or ""),
                type=str(edge.get("type") or ""),
            )
            for edge in record.get("edges") or ()
            if edge.get("type")
        ]
        guardrails = record.get("die_guardrails") or {}
        constraints = guardrails.get("type_constraints") if isinstance(guardrails, dict) else {}
        features = []
        for item in record.get("semantic_features_with_KL") or ():
            if isinstance(item, dict) and item.get("feature") is not None:
                try:
                    features.append((str(item["feature"]), float(item.get("weight") or 0.0)))
                except (TypeError, ValueError):
                    continue
        return ShannonPattern(
            pattern_id=str(record.get("pattern_id") or ""),
            pattern_type=pattern_type,
            summary=str(record.get("original_summary") or ""),
            description=str(record.get("original_description") or ""),
            original_code=str(record.get("original_code") or ""),
            nodes=nodes,
            edges=edges,
            type_constraints=constraints if isinstance(constraints, dict) else {},
            structural_context=str(guardrails.get("structural_context") or "") if isinstance(guardrails, dict) else "",
            features=features,
        )

    def _signature_collisions(self) -> Counter[str]:
        counts: Counter[str] = Counter()
        for pattern in self.patterns.values():
            counts[self._signature_key(pattern)] += 1
        for pattern in self.patterns.values():
            pattern.collision_count = counts[self._signature_key(pattern)]
        return counts

    @staticmethod
    def _signature_key(pattern: ShannonPattern) -> str:
        payload = json.dumps(pattern.signature, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # -- retrieval signals -----------------------------------------------------

    def kind_idf(self, kind: str) -> float:
        return math.log((len(self.patterns) + 1.0) / (self.kind_document_frequency[kind] + 1.0)) + 1.0

    def edge_idf(self, edge_type: str) -> float:
        return math.log((len(self.patterns) + 1.0) / (self.edge_document_frequency[edge_type] + 1.0)) + 1.0

    def feature_weight(self, feature: str) -> float:
        return self.feature_weights.get(feature, 1.0)

    def candidates(self, query_kinds: Iterable[str]) -> set[str]:
        result: set[str] = set()
        for kind in query_kinds:
            result.update(self.inverted_index.get(kind, ()))
        return result

    def edge_candidates(self, query_edge_types: Iterable[str]) -> set[str]:
        result: set[str] = set()
        for edge_type in query_edge_types:
            result.update(self.edge_inverted_index.get(edge_type, ()))
        return result

    # -- mutation material (all embedded; no external join needed) -------------

    def pattern_payload(self, pattern: ShannonPattern) -> dict[str, Any]:
        """Pattern view for the snippet protocol, with hidden report fields."""
        payload = pattern.to_dict()
        payload["_summary"] = pattern.summary
        payload["_description"] = pattern.description
        payload["_original_code"] = pattern.original_code
        return payload

    def core_material(self, pattern: ShannonPattern) -> dict[str, str]:
        historical_subtree = pattern.primary_core_source
        if not historical_subtree:
            raise ValueError(f"{pattern.pattern_id}: pattern has no core snippet")
        return {
            "historical_subtree": historical_subtree,
            "Summary": pattern.summary,
            "Description": pattern.description,
            "extracted_code": pattern.original_code,
        }

    def stats(self) -> dict[str, Any]:
        types = Counter(pattern.pattern_type for pattern in self.patterns.values())
        edge_types = Counter()
        node_total = 0
        for pattern in self.patterns.values():
            edge_types.update(pattern.edge_type_multiset)
            node_total += pattern.node_count
        collisions = Counter(pattern.collision_count for pattern in self.patterns.values())
        return {
            "source": str(self.library_file) if self.library_file else str(self.library_dir),
            "schema": "selfuzz-anomaly-pattern-v1",
            "ast_kind_domain": "com.sun.source.tree.Tree.Kind",
            "relation_domain": "18 typed semantic relations (5 categories)",
            "patterns": len(self.patterns),
            "pattern_types": dict(types),
            "avg_nodes": round(node_total / max(1, len(self.patterns)), 3),
            "avg_edges": round(sum(sum(p.edge_type_multiset.values()) for p in self.patterns.values()) / max(1, len(self.patterns)), 3),
            "indexed_kinds": len(self.inverted_index),
            "indexed_edge_types": len(self.edge_inverted_index),
            "edge_type_distribution": dict(sorted(edge_types.items(), key=lambda kv: -kv[1])),
            "feature_weighting": self.feature_weighting,
            "records_in_collision_groups": sum(count for size, count in collisions.items() if size > 1),
        }


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
