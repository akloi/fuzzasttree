"""Data contracts for the Shannon anomaly-pattern matching and mutation protocol."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROLE_WEIGHTS = {"CORE_TRIGGER": 1.0, "CONTEXT": 0.6, "DEPENDENT": 0.3}

CORE_TRIGGER = "CORE_TRIGGER"
CONTEXT = "CONTEXT"
DEPENDENT = "DEPENDENT"


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
class PatternNode:
    node_id: str
    kind: str
    source_code: str
    role: str

    @property
    def weight(self) -> float:
        return ROLE_WEIGHTS.get(self.role, 0.3)


@dataclass(slots=True)
class PatternEdge:
    source: str
    target: str
    type: str


class ShannonPattern:
    """One multi-dimensional anomaly pattern from the ast_shannon library.

    The pattern is a role-labeled heterogeneous graph: nodes carry javac kinds
    plus extraction roles (sink / closure / dependent), edges carry one of the
    18 semantic relation types, and features carry information-theoretic weights.
    All derived signatures are precomputed once at load time.
    """

    __slots__ = (
        "pattern_id", "pattern_type", "summary", "description", "original_code",
        "nodes", "edges", "type_constraints", "structural_context", "features",
        "collision_count", "corpus_rarity",
        "_core_kinds", "_context_kinds", "_dependent_kinds", "_required_kinds",
        "_kind_multiset", "_edge_type_multiset", "_primary_core",
        "_enclosing_decl_kinds", "_signature",
    )

    def __init__(
        self,
        pattern_id: str,
        pattern_type: str,
        summary: str,
        description: str,
        original_code: str,
        nodes: list[PatternNode],
        edges: list[PatternEdge],
        type_constraints: dict[str, str],
        structural_context: str,
        features: list[tuple[str, float]],
    ) -> None:
        self.pattern_id = pattern_id
        self.pattern_type = pattern_type
        self.summary = summary
        self.description = description
        self.original_code = original_code
        self.nodes = tuple(nodes)
        self.edges = tuple(edges)
        self.type_constraints = dict(type_constraints or {})
        self.structural_context = str(structural_context or "")
        self.features = tuple((str(name), float(weight)) for name, weight in features or ())
        self.collision_count = 1
        self.corpus_rarity = 0.0

        by_role: dict[str, list[str]] = {CORE_TRIGGER: [], CONTEXT: [], DEPENDENT: []}
        multiset: Counter[str] = Counter()
        primary: PatternNode | None = None
        enclosing: list[str] = []
        for node in self.nodes:
            by_role.setdefault(node.role, []).append(node.kind)
            multiset[node.kind] += node.weight
            if node.role == CORE_TRIGGER and primary is None:
                primary = node
            if node.role == CONTEXT and node.kind in ENCLOSING_DECL_KINDS:
                enclosing.append(node.kind)
        if primary is None:
            primary = self.nodes[0] if self.nodes else None
        self._core_kinds = tuple(dict.fromkeys(by_role[CORE_TRIGGER]))
        self._context_kinds = tuple(dict.fromkeys(by_role[CONTEXT]))
        self._dependent_kinds = tuple(dict.fromkeys(by_role[DEPENDENT]))
        self._required_kinds = tuple(dict.fromkeys(self._core_kinds + self._context_kinds))
        self._kind_multiset = dict(sorted(multiset.items()))
        self._edge_type_multiset = dict(sorted(Counter(edge.type for edge in self.edges).items()))
        self._primary_core = primary
        self._enclosing_decl_kinds = tuple(dict.fromkeys(enclosing))
        self._signature = {
            "version": "astshannon-pattern-signature-v1",
            "pattern_type": self.pattern_type,
            "node_count": len(self.nodes),
            "core_kinds": list(self._core_kinds),
            "required_kinds": list(self._required_kinds),
            "kind_role_multiset": self._kind_multiset,
            "edge_type_multiset": self._edge_type_multiset,
        }

    # -- role-aware kind views ------------------------------------------------

    @property
    def core_kinds(self) -> tuple[str, ...]:
        return self._core_kinds

    @property
    def context_kinds(self) -> tuple[str, ...]:
        return self._context_kinds

    @property
    def dependent_kinds(self) -> tuple[str, ...]:
        return self._dependent_kinds

    @property
    def required_kinds(self) -> tuple[str, ...]:
        """Kinds the host must plausibly exhibit for the pattern to apply."""
        return self._required_kinds

    @property
    def kind_role_multiset(self) -> dict[str, float]:
        """Kind multiset with CORE/CONTEXT/DEPENDENT weights (extraction roles as match weights)."""
        return self._kind_multiset

    @property
    def edge_type_multiset(self) -> dict[str, int]:
        return self._edge_type_multiset

    @property
    def primary_core(self) -> PatternNode | None:
        """The sink node: first CORE_TRIGGER node recorded by the extractor."""
        return self._primary_core

    @property
    def primary_core_kind(self) -> str:
        return self._primary_core.kind if self._primary_core else ""

    @property
    def primary_core_source(self) -> str:
        return (self._primary_core.source_code if self._primary_core else "").strip()

    @property
    def enclosing_decl_kinds(self) -> tuple[str, ...]:
        """Declaration kinds the pattern's CORE must stay inside (DIE feasibility)."""
        return self._enclosing_decl_kinds

    @property
    def feature_weights(self) -> dict[str, float]:
        return {name: weight for name, weight in self.features}

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def signature(self) -> dict[str, Any]:
        return dict(self._signature)

    @property
    def edge_type_count(self) -> int:
        return len(self._edge_type_multiset)

    def to_dict(self) -> dict[str, Any]:
        """Serializable pattern view embedded into the mutation protocol."""
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type,
            "nodes": [
                {"node_id": node.node_id, "kind": node.kind, "source_code": node.source_code, "role": node.role}
                for node in self.nodes
            ],
            "edges": [{"source": edge.source, "target": edge.target, "type": edge.type} for edge in self.edges],
            "die_guardrails": {
                "type_constraints": self.type_constraints,
                "structural_context": self.structural_context,
            },
            "semantic_features": [
                {"feature": name, "weight": round(weight, 4)} for name, weight in self.features
            ],
        }


ENCLOSING_DECL_KINDS = {"CLASS", "INTERFACE", "ENUM", "RECORD", "METHOD"}


@dataclass(slots=True)
class HostRegion:
    """The closure-region aligned to one pattern anchor on the host side.

    Built by the same closure operator that built the pattern (sink -> crossing
    semantic edges -> 1-2 hops -> enclosing declaration + control wrapper), so
    pattern and region live in one coordinate system: kinds, roles, and the
    18 typed relations.
    """

    anchor_id: int
    region_ids: tuple[int, ...]
    roles: dict[int, str]
    edges: tuple[dict[str, Any], ...]
    anchor_kind: str
    start_byte: int
    end_byte: int
    kind_role_multiset: dict[str, float]
    edge_type_multiset: dict[str, int]
    features: tuple[str, ...]
    covered_core_kinds: tuple[str, ...]
    missing_core_kinds: tuple[str, ...]

    @property
    def region_size(self) -> int:
        return len(self.region_ids)


@dataclass(slots=True)
class MatchResult:
    pattern: ShannonPattern
    region: HostRegion
    prefilter_score: float
    structural_score: float
    feature_score: float
    quality_score: float
    specificity_score: float
    final_score: float
    feasible: bool = True
    feasibility_reasons: tuple[str, ...] = ()
    components: dict[str, float] = field(default_factory=dict)
    feature_evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, source_path: Path | None = None) -> dict[str, Any]:
        root = {
            "bug_id": self.pattern.pattern_id,
            "score": round(self.final_score, 6),
            "prefilter_score": round(self.prefilter_score, 6),
            "structural_score": round(self.structural_score, 6),
            "feature_score": round(self.feature_score, 6),
            "quality_score": round(self.quality_score, 6),
            "specificity_score": round(self.specificity_score, 6),
            "score_components": {key: round(value, 6) for key, value in self.components.items()},
            "feature_evidence": self.feature_evidence,
            "historical": {
                "pattern_type": self.pattern.pattern_type,
                "core_kinds": list(self.pattern.core_kinds),
                "required_kinds": list(self.pattern.required_kinds),
                "edge_type_multiset": self.pattern.edge_type_multiset,
                "node_count": self.pattern.node_count,
                "signature_collision_count": self.pattern.collision_count,
                "summary": self.pattern.summary,
                "die_guardrails": {
                    "type_constraints": self.pattern.type_constraints,
                    "structural_context": self.pattern.structural_context,
                },
                "semantic_features": [
                    {"feature": name, "weight": round(weight, 4)} for name, weight in self.pattern.features
                ],
                "pattern": self.pattern.to_dict(),
            },
            "host_region": {
                "anchor_node_id": self.region.anchor_id,
                "anchor_kind": self.region.anchor_kind,
                "region_node_ids": list(self.region.region_ids),
                "region_roles": {str(key): value for key, value in self.region.roles.items()},
                "region_edge_types": dict(self.region.edge_type_multiset),
                "covered_core_kinds": list(self.region.covered_core_kinds),
                "missing_core_kinds": list(self.region.missing_core_kinds),
                "start_byte": self.region.start_byte,
                "end_byte": self.region.end_byte,
            },
        }
        if source_path is not None:
            root["input_file"] = str(source_path)
        return root


@dataclass(slots=True)
class MutationRequest:
    """Snippet protocol: everything the mutator needs, never the whole host file."""

    bug_id: str
    host_subtree: str
    historical_subtree: str
    pattern_payload: dict[str, Any]
    match_score: float
    mutation_mode: str = "single_pattern"
    host_region_edges: tuple[dict[str, Any], ...] = ()
    enclosing_context: str = ""
    score_components: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": "astshannon-snippet-v1",
            "bug_id": self.bug_id,
            "mutation_mode": self.mutation_mode,
            "host_subtree": self.host_subtree,
            "enclosing_context": self.enclosing_context,
            "pattern": self.pattern_payload,
            "historical_subtree": self.historical_subtree,
            "historical_report": {
                "Summary": self.pattern_payload.get("_summary", ""),
                "Description": self.pattern_payload.get("_description", ""),
                "extracted_code": self.pattern_payload.get("_original_code", ""),
            },
            "match_score": round(self.match_score, 6),
            "score_components": {key: round(value, 6) for key, value in self.score_components.items()},
            "host_region_evidence": [
                {"source": edge["source"], "target": edge["target"], "type": edge["type"]}
                for edge in self.host_region_edges
            ],
            "instruction": (
                "Return one JSON object with exactly the field mutated_subtree: the replacement Java "
                "code for host_subtree. Transfer the historical anomaly pattern into it: reproduce the "
                "CORE_TRIGGER node structure, keep the multi-node synergy described by pattern.edges, "
                "obey pattern.die_guardrails (type_constraints and structural_context), and adapt "
                "identifiers and types to enclosing_context. Supporting declarations the pattern needs "
                "(e.g. local sealed interfaces, records, helper variables) may be embedded inside the "
                "replacement as long as it stays syntactically valid at the anchor position. Preserve "
                "the syntactic role of host_subtree. Do not return a complete Java file, imports, "
                "Markdown, or explanations."
            ),
        }


@dataclass(slots=True)
class AnchorEdge:
    """Interaction evidence between two host anchors (structural or host-semantic)."""

    edge_type: str
    source_id: int
    target_id: int
    strength: float
    evidence: dict[str, Any]
    host_relation: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "edge_type": self.edge_type,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "strength": self.strength,
            "evidence": self.evidence,
        }
        if self.host_relation:
            payload["host_relation"] = self.host_relation
        return payload


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
            "score": round(self.score, 6),
            "score_components": {key: round(value, 6) for key, value in self.components.items()},
            "trace_to_bug_1": self.match_1.pattern.pattern_id,
            "trace_to_bug_2": self.match_2.pattern.pattern_id,
            "anchor_1": self.match_1.to_dict(),
            "anchor_2": self.match_2.to_dict(),
            "interaction_edge": self.edge.to_dict() if self.edge else None,
            "edit_operation_1": "replace_subtree",
            "edit_operation_2": "fuse_into_anchor_1" if self.fusion_kind == "same_node" else "replace_subtree",
            "binding_constraints": ["preserve_host_syntactic_roles", "retain_both_pattern_backbones", "respect_both_die_guardrails"],
        }


@dataclass(slots=True)
class FusionMutationRequest:
    base: MutationRequest
    host_subtree_2: str
    pattern_payload_2: dict[str, Any]
    historical_subtree_2: str
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
            "pattern_1": {key: value for key, value in payload["pattern"].items() if not key.startswith("_")},
            "pattern_2": {key: value for key, value in self.pattern_payload_2.items() if not key.startswith("_")},
            "historical_subtree_1": self.base.historical_subtree,
            "historical_subtree_2": self.historical_subtree_2,
            "historical_report_1": payload["historical_report"],
            "historical_report_2": {
                "Summary": self.pattern_payload_2.get("_summary", ""),
                "Description": self.pattern_payload_2.get("_description", ""),
                "extracted_code": self.pattern_payload_2.get("_original_code", ""),
            },
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
