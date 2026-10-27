"""Load and index extracted historical anomaly subtrees."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from .models import HistoricalPattern


DEFAULT_LIBRARY_DIR = Path(
    "/disk1/users/zengj/jdk-fuzzllm/javac_data/javac_data/javaccode_8635/astpatterns_minimal"
)
DEFAULT_HISTORICAL_SOURCE_DIR = Path(
    "/disk1/users/zengj/jdk-fuzzllm/javac_data/javac_data/javaccode_8635/astwithcontext"
)


class HistoricalLibrary:
    def __init__(
        self,
        library_dir: Path = DEFAULT_LIBRARY_DIR,
        source_dir: Path = DEFAULT_HISTORICAL_SOURCE_DIR,
        statuses: Iterable[str] = ("candidate", "candidate_partial"),
    ) -> None:
        self.library_dir = Path(library_dir)
        self.source_dir = Path(source_dir)
        self.statuses = frozenset(statuses)
        self.patterns: dict[str, HistoricalPattern] = {}
        self.inverted_index: dict[str, set[str]] = defaultdict(set)
        self.kind_document_frequency: Counter[str] = Counter()
        self._load()

    def _load(self) -> None:
        signature_counts: Counter[str] = Counter()
        pending: list[HistoricalPattern] = []
        for path in sorted(self.library_dir.glob("JDK-*.json")):
            record = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            relevance = record.get("ast_relevance") or {}
            status = str(relevance.get("status") or "")
            subtree = record.get("minimum_anomaly_subtree")
            if status not in self.statuses or not isinstance(subtree, dict):
                continue
            abstraction = record.get("abstraction") or {}
            native_nodes, signature, ancestor_path = self._restore_native_subtree(record, subtree, path)
            required = tuple(dict.fromkeys(str(node["kind"]) for node in native_nodes))
            if not required:
                raise ValueError(f"{path}: restored native subtree contains no node kinds")
            span = subtree.get("source_span_utf16") or {}
            start = span.get("start")
            end = span.get("end")
            source_span = (int(start), int(end)) if start is not None and end is not None else None
            pattern = HistoricalPattern(
                bug_id=str(record.get("bug_id") or path.stem),
                source_file=str(record.get("source_file") or path.name),
                status=status,
                reason=str(relevance.get("reason") or ""),
                language_feature=str(abstraction.get("L") or "Unknown"),
                required_kinds=required,
                abstract_raw_kinds=tuple(str(item) for item in abstraction.get("A_raw") or ()),
                abstract_canonical_kinds=tuple(str(item) for item in abstraction.get("A_canonical") or ()),
                phases=tuple(str(item) for item in abstraction.get("P") or ()),
                diagnostic=str(abstraction.get("D") or "Unknown"),
                confidence=_float(abstraction.get("confidence")),
                candidate_confidence=_float(relevance.get("candidate_confidence")),
                coverage=_float(subtree.get("coverage")),
                corpus_rarity=0.0,
                ancestor_path=ancestor_path,
                source_span_utf16=source_span,
                signature=signature,
                subtree_nodes=native_nodes,
            )
            pending.append(pattern)
            signature_counts[str(signature.get("sha256") or _signature_key(signature))] += 1

        for pattern in pending:
            signature_key = str(pattern.signature.get("sha256") or _signature_key(pattern.signature))
            pattern.collision_count = signature_counts[signature_key]
            self.patterns[pattern.bug_id] = pattern
            for kind in set(pattern.required_kinds):
                self.inverted_index[kind].add(pattern.bug_id)
                self.kind_document_frequency[kind] += 1
        for pattern in pending:
            pattern.corpus_rarity = sum(self.idf(kind) for kind in pattern.required_kinds) / len(
                pattern.required_kinds
            )

    def _restore_native_subtree(
        self,
        pattern_record: dict[str, Any],
        subtree: dict[str, Any],
        pattern_path: Path,
    ) -> tuple[tuple[dict[str, Any], ...], dict[str, Any], tuple[str, ...]]:
        """Restore exact javac kinds by joining minimum nodes to the full AST."""
        bug_id = str(pattern_record.get("bug_id") or pattern_path.stem)
        source_path = self.source_dir / str(pattern_record.get("source_file") or f"{bug_id}.json")
        if not source_path.exists():
            source_path = self.source_dir / f"{bug_id}.json"
        if not source_path.exists():
            raise ValueError(f"{bug_id}: historical AST record is missing: {source_path}")

        source_record = json.loads(source_path.read_text(encoding="utf-8", errors="replace"))
        raw_nodes = (source_record.get("ast") or {}).get("nodes")
        if not isinstance(raw_nodes, list) or not raw_nodes:
            raise ValueError(f"{bug_id}: historical record has no ast.nodes")
        by_original_id = {int(node["node_id"]): node for node in raw_nodes}

        minimum_nodes = subtree.get("nodes")
        if not isinstance(minimum_nodes, list) or not minimum_nodes:
            raise ValueError(f"{bug_id}: minimum anomaly subtree has no nodes")
        original_to_relative: dict[int, int] = {}
        for node in minimum_nodes:
            if node.get("original_node_id") is None or node.get("relative_id") is None:
                raise ValueError(f"{bug_id}: minimum subtree node lacks original_node_id or relative_id")
            original_id = int(node["original_node_id"])
            relative_id = int(node["relative_id"])
            if original_id not in by_original_id:
                raise ValueError(f"{bug_id}: original_node_id {original_id} is absent from historical AST")
            original_to_relative[original_id] = relative_id

        restored: list[dict[str, Any]] = []
        root_original_id: int | None = None
        for node in minimum_nodes:
            original_id = int(node["original_node_id"])
            relative_id = int(node["relative_id"])
            raw = by_original_id[original_id]
            raw_parent = raw.get("parent_id")
            expected_parent = None if raw_parent is None else original_to_relative.get(int(raw_parent))
            recorded_parent = node.get("parent_relative_id")
            if recorded_parent is not None:
                recorded_parent = int(recorded_parent)
            if recorded_parent != expected_parent:
                raise ValueError(
                    f"{bug_id}: relative node {relative_id} parent mismatch "
                    f"({recorded_parent} != {expected_parent})"
                )
            raw_child_index = int(raw.get("child_index", -1))
            if int(node.get("child_index", -1)) != raw_child_index:
                raise ValueError(f"{bug_id}: relative node {relative_id} child_index mismatch")
            if recorded_parent is None:
                if root_original_id is not None:
                    raise ValueError(f"{bug_id}: minimum subtree has multiple roots")
                root_original_id = original_id
            restored.append(
                {
                    "relative_id": relative_id,
                    "original_node_id": original_id,
                    "kind": str(raw.get("kind") or ""),
                    "parent_relative_id": recorded_parent,
                    "child_index": raw_child_index,
                    "source_span_utf16": raw.get("source_span_utf16"),
                }
            )
        if root_original_id is None:
            raise ValueError(f"{bug_id}: minimum subtree has no root")
        recorded_root = subtree.get("root_original_node_id")
        if recorded_root is not None and int(recorded_root) != root_original_id:
            raise ValueError(f"{bug_id}: root_original_node_id does not identify the restored root")

        restored.sort(key=lambda node: int(node["relative_id"]))
        signature = _native_signature(restored)
        ancestor_path = _native_ancestor_path(by_original_id, root_original_id, bug_id)
        return tuple(restored), signature, ancestor_path

    def idf(self, kind: str) -> float:
        return math.log((len(self.patterns) + 1.0) / (self.kind_document_frequency[kind] + 1.0)) + 1.0

    def candidates(self, query_kinds: Iterable[str]) -> set[str]:
        result: set[str] = set()
        for kind in query_kinds:
            result.update(self.inverted_index.get(kind, ()))
        return result

    def reference_subtree(self, pattern: HistoricalPattern) -> str:
        return str(self.reference_material(pattern)["historical_subtree"])

    def reference_material(self, pattern: HistoricalPattern) -> dict[str, str]:
        """Load the report fields needed only after a pattern has matched."""
        if pattern.source_span_utf16 is None:
            raise ValueError(f"{pattern.bug_id} has no source span")
        source_path = self.source_dir / pattern.source_file
        if not source_path.exists():
            source_path = self.source_dir / f"{pattern.bug_id}.json"
        record = json.loads(source_path.read_text(encoding="utf-8", errors="replace"))
        source = str(record.get("extracted_code") or "")
        start, end = pattern.source_span_utf16
        raw = source.encode("utf-16-le", errors="surrogatepass")
        snippet = raw[start * 2 : end * 2].decode("utf-16-le", errors="replace")
        if not snippet.strip():
            raise ValueError(f"{pattern.bug_id} historical subtree source is empty")
        return {
            "historical_subtree": snippet,
            "Summary": str(record.get("Summary") or record.get("summary") or ""),
            "Description": str(record.get("Description") or record.get("description") or ""),
            "extracted_code": source,
        }

    def stats(self) -> dict[str, Any]:
        statuses = Counter(pattern.status for pattern in self.patterns.values())
        collisions = Counter(pattern.collision_count for pattern in self.patterns.values())
        return {
            "library_dir": str(self.library_dir),
            "source_dir": str(self.source_dir),
            "ast_backend": "javac",
            "ast_kind_domain": "com.sun.source.tree.Tree.Kind",
            "restoration": "astwithcontext.ast.nodes joined by original_node_id",
            "patterns": len(self.patterns),
            "statuses": dict(statuses),
            "indexed_kinds": len(self.inverted_index),
            "records_in_collision_groups": sum(count for size, count in collisions.items() if size > 1),
        }


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _signature_key(signature: dict[str, Any]) -> str:
    return json.dumps(signature, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _native_signature(nodes: list[dict[str, Any]]) -> dict[str, Any]:
    kinds = {int(node["relative_id"]): str(node["kind"]) for node in nodes}
    parents = {
        int(node["relative_id"]): (
            None if node.get("parent_relative_id") is None else int(node["parent_relative_id"])
        )
        for node in nodes
    }
    roots = [node_id for node_id, parent_id in parents.items() if parent_id is None]
    if len(roots) != 1:
        raise ValueError(f"restored minimum subtree must have one root, found {len(roots)}")
    root_id = roots[0]
    node_multiset = Counter(kinds.values())
    edge_multiset: Counter[str] = Counter()
    paths: list[str] = []
    for node_id, kind in kinds.items():
        parent_id = parents[node_id]
        if parent_id is not None:
            if parent_id not in kinds:
                raise ValueError(f"restored node {node_id} refers to absent parent {parent_id}")
            edge_multiset[f"{kinds[parent_id]}>{kind}"] += 1
        path: list[str] = []
        current: int | None = node_id
        seen: set[int] = set()
        while current is not None:
            if current in seen:
                raise ValueError("cycle in restored minimum subtree")
            seen.add(current)
            path.append(kinds[current])
            current = parents[current]
        paths.append(">".join(reversed(path)))
    required = list(dict.fromkeys(kinds[node_id] for node_id in sorted(kinds)))
    signature = {
        "version": "javac-native-subtree-signature-v1",
        "kind_domain": "com.sun.source.tree.Tree.Kind",
        "restoration": "astwithcontext.ast.nodes joined by original_node_id",
        "root_kind": kinds[root_id],
        "required_kinds": required,
        "node_kind_multiset": dict(sorted(node_multiset.items())),
        "edge_kind_multiset": dict(sorted(edge_multiset.items())),
        "root_to_node_paths": sorted(set(paths)),
        "node_count": len(nodes),
    }
    digest_payload = _signature_key(signature).encode("utf-8")
    signature["sha256"] = hashlib.sha256(digest_payload).hexdigest()
    return signature


def _native_ancestor_path(
    by_original_id: dict[int, dict[str, Any]],
    root_original_id: int,
    bug_id: str,
) -> tuple[str, ...]:
    path: list[str] = []
    current: int | None = root_original_id
    seen: set[int] = set()
    while current is not None:
        if current in seen:
            raise ValueError(f"{bug_id}: cycle in historical AST ancestor chain")
        seen.add(current)
        node = by_original_id.get(current)
        if node is None:
            raise ValueError(f"{bug_id}: ancestor node {current} is absent from historical AST")
        path.append(str(node.get("kind") or ""))
        parent = node.get("parent_id")
        current = None if parent is None else int(parent)
    return tuple(reversed(path))
