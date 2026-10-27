"""Data contracts for matching, mutation, and auditable output."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class SyntaxNode:
    node_id: int
    raw_kind: str
    parent_id: int | None
    child_ids: list[int]
    child_index: int
    start_byte: int
    end_byte: int
    start_line: int
    end_line: int

    @property
    def primary_kind(self) -> str:
        return self.raw_kind


@dataclass(slots=True)
class HistoricalPattern:
    bug_id: str
    source_file: str
    status: str
    reason: str
    language_feature: str
    required_kinds: tuple[str, ...]
    abstract_raw_kinds: tuple[str, ...]
    abstract_canonical_kinds: tuple[str, ...]
    phases: tuple[str, ...]
    diagnostic: str
    confidence: float
    candidate_confidence: float
    coverage: float
    corpus_rarity: float
    ancestor_path: tuple[str, ...]
    source_span_utf16: tuple[int, int] | None
    signature: dict[str, Any]
    subtree_nodes: tuple[dict[str, Any], ...]
    collision_count: int = 1

    @property
    def node_count(self) -> int:
        return int(self.signature.get("node_count") or len(self.subtree_nodes))


@dataclass(slots=True)
class LocalSubtree:
    root_id: int
    selected_ids: tuple[int, ...]
    covered_kinds: tuple[str, ...]
    missing_kinds: tuple[str, ...]
    signature: dict[str, Any]
    ancestor_path: tuple[str, ...]
    start_byte: int
    end_byte: int

    @property
    def coverage(self) -> float:
        total = len(self.covered_kinds) + len(self.missing_kinds)
        return len(self.covered_kinds) / total if total else 0.0


@dataclass(slots=True)
class MatchResult:
    pattern: HistoricalPattern
    local_subtree: LocalSubtree
    prefilter_score: float
    structural_score: float
    quality_score: float
    specificity_score: float
    final_score: float
    components: dict[str, float] = field(default_factory=dict)
    feature_evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, source_path: Path | None = None) -> dict[str, Any]:
        root = {
            "bug_id": self.pattern.bug_id,
            "score": round(self.final_score, 6),
            "prefilter_score": round(self.prefilter_score, 6),
            "structural_score": round(self.structural_score, 6),
            "quality_score": round(self.quality_score, 6),
            "specificity_score": round(self.specificity_score, 6),
            "score_components": {key: round(value, 6) for key, value in self.components.items()},
            "feature_compatibility": self.feature_evidence,
            "historical": {
                "status": self.pattern.status,
                "native_required_kinds": list(self.pattern.required_kinds),
                "abstract_raw_kinds": list(self.pattern.abstract_raw_kinds),
                "abstract_canonical_kinds": list(self.pattern.abstract_canonical_kinds),
                "ast_kind_domain": "com.sun.source.tree.Tree.Kind",
                "coverage": self.pattern.coverage,
                "confidence": self.pattern.confidence,
                "language_feature": self.pattern.language_feature,
                "phases": list(self.pattern.phases),
                "diagnostic": self.pattern.diagnostic,
                "node_count": self.pattern.node_count,
                "signature_collision_count": self.pattern.collision_count,
                "native_signature": self.pattern.signature,
                "native_subtree_nodes": list(self.pattern.subtree_nodes),
            },
            "input_subtree": {
                "root_node_id": self.local_subtree.root_id,
                "selected_node_ids": list(self.local_subtree.selected_ids),
                "covered_kinds": list(self.local_subtree.covered_kinds),
                "missing_kinds": list(self.local_subtree.missing_kinds),
                "coverage": round(self.local_subtree.coverage, 6),
                "start_byte": self.local_subtree.start_byte,
                "end_byte": self.local_subtree.end_byte,
                "ancestor_path": list(self.local_subtree.ancestor_path),
                "signature": self.local_subtree.signature,
            },
        }
        if source_path is not None:
            root["input_file"] = str(source_path)
        return root


@dataclass(slots=True)
class MutationRequest:
    bug_id: str
    host_subtree: str
    historical_subtree: str
    language_feature: str
    required_kinds: tuple[str, ...]
    phases: tuple[str, ...]
    diagnostic: str
    match_score: float
    historical_summary: str = ""
    historical_description: str = ""
    historical_source: str = ""
    abstract_raw_kinds: tuple[str, ...] = ()
    abstract_canonical_kinds: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "bug_id": self.bug_id,
            "host_subtree": self.host_subtree,
            "historical_subtree": self.historical_subtree,
            "pattern": {
                "L": self.language_feature,
                "A_raw": list(self.abstract_raw_kinds),
                "A_canonical": list(self.abstract_canonical_kinds),
                "A_native": list(self.required_kinds),
                "P": list(self.phases),
                "D": self.diagnostic,
            },
            "historical_report": {
                "Summary": self.historical_summary,
                "Description": self.historical_description,
                "extracted_code": self.historical_source,
            },
            "match_score": round(self.match_score, 6),
            "instruction": (
                "Return only a mutated replacement for host_subtree. Preserve its syntactic role and "
                "introduce the historical anomaly structure. Do not return a complete Java file."
            ),
        }


@dataclass(slots=True)
class AnchorEdge:
    edge_type: str
    source_id: int
    target_id: int
    strength: float
    evidence: dict[str, Any]
    phase: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_type": self.edge_type,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "strength": self.strength,
            "evidence": self.evidence,
            "phase": self.phase,
        }


@dataclass(slots=True)
class FusionPlan:
    fusion_kind: str
    operation: str
    match_1: MatchResult
    match_2: MatchResult
    edge: AnchorEdge | None
    score: float
    components: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fusion_kind": self.fusion_kind,
            "operation": self.operation,
            "score": self.score,
            "score_components": self.components,
            "trace_to_bug_1": self.match_1.pattern.bug_id,
            "trace_to_bug_2": self.match_2.pattern.bug_id,
            "anchor_1": self.match_1.to_dict(),
            "anchor_2": self.match_2.to_dict(),
            "interaction_edge": self.edge.to_dict() if self.edge else None,
            "edit_operation_1": "replace_subtree",
            "edit_operation_2": "fuse_into_anchor_1" if self.fusion_kind == "same_node" else "replace_subtree",
            "binding_constraints": ["preserve_host_syntactic_roles", "retain_both_pattern_backbones"],
        }


@dataclass(slots=True)
class FusionMutationRequest:
    base: MutationRequest
    host_subtree_2: str
    pattern_2: dict[str, Any]
    historical_subtree_2: str
    historical_report_2: dict[str, str]
    fusion_role: str
    bridging_context: str
    instruction: str
    bug_id_2: str
    match_score_2: float
    fusion_kind: str = "same_node"
    edge_evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = self.base.to_dict()
        payload.update({
            "protocol_version": 2,
            "fusion_kind": self.fusion_kind,
            "host_subtree_1": self.base.host_subtree,
            "host_subtree_2": self.host_subtree_2,
            "pattern_1": payload["pattern"],
            "pattern_2": self.pattern_2,
            "historical_subtree_1": self.base.historical_subtree,
            "historical_subtree_2": self.historical_subtree_2,
            "historical_report_1": payload["historical_report"],
            "historical_report_2": self.historical_report_2,
            "bug_id_1": self.base.bug_id,
            "bug_id_2": self.bug_id_2,
            "match_score_1": self.base.match_score,
            "match_score_2": self.match_score_2,
            "fusion_role": self.fusion_role,
            "bridging_context": self.bridging_context,
            "edge_evidence": self.edge_evidence,
            "instruction": self.instruction,
        })
        return payload

