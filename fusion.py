"""Constrained same-anchor pairing and phase-one syntax interaction graphs."""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations, product
import math
import re
from typing import Iterable

from .matcher import FEATURE_RULES
from .models import AnchorEdge, FusionPlan, MatchResult
from .parser import JavaSyntaxTree


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
    "wrapper": "Use pattern 2 as an outer AST wrapper around pattern 1 while retaining both backbones.",
    "role_extension": "Extend the role of the host expression to an operand or conditional position and preserve both mechanisms.",
    "constraint_conjunction": "Make both historical type/lambda constraints act on the same target expression; retain both constraint sources.",
    "branch_splitting": "Place the two mechanisms into distinct parts of one conditional expression under the existing host context.",
    "nested_expression": "Nest the pattern-1 lambda or method reference into an argument slot of the pattern-2 invocation.",
    "sibling": "Coordinate both sibling replacements using their shared operator, conditional, or argument context.",
    "control_flow": "Preserve the control-flow relationship and place the two mechanisms in their respective branch or loop slots.",
    "ancestor_descendant": "Preserve the enclosing/contained relationship of the two anomaly anchors.",
}


def syntactic_role(kind: str) -> str:
    if kind in EXPRESSION_KINDS:
        return "expression"
    if kind in STATEMENT_KINDS:
        return "statement"
    return kind


def group_matches(matches: Iterable[MatchResult]) -> dict[int, list[MatchResult]]:
    groups: dict[int, list[MatchResult]] = defaultdict(list)
    for match in matches:
        groups[match.local_subtree.root_id].append(match)
    return {root: sorted(group, key=lambda m: (-m.final_score, m.pattern.bug_id)) for root, group in groups.items()}


def _distance(left: dict | Iterable, right: dict | Iterable) -> float:
    a, b = Counter(left), Counter(right)
    union = sum((a | b).values())
    return 1.0 - sum((a & b).values()) / union if union else 0.0


def mechanism_orthogonality(first: MatchResult, second: MatchResult) -> float:
    a, b = first.pattern, second.pattern
    phases_a = {p.casefold() for p in a.phases} - {"unknown"}
    phases_b = {p.casefold() for p in b.phases} - {"unknown"}
    phase = _distance(phases_a, phases_b) if phases_a and phases_b else 0.0
    signature = _distance(a.signature.get("edge_kind_multiset", {}), b.signature.get("edge_kind_multiset", {}))
    diagnostic_a = set(re.findall(r"[a-z0-9]+", a.diagnostic.casefold())) - {"unknown"}
    diagnostic_b = set(re.findall(r"[a-z0-9]+", b.diagnostic.casefold())) - {"unknown"}
    diagnostic = _distance(diagnostic_a, diagnostic_b) if diagnostic_a and diagnostic_b else 0.0
    return 0.5 * phase + 0.3 * signature + 0.2 * diagnostic


def _features(match: MatchResult) -> set[str]:
    text = match.pattern.language_feature.lower().replace("_", " ")
    kinds = set(match.pattern.required_kinds)
    return {name for name, text_rule, native, _ in FEATURE_RULES if text_rule.search(text) or native & kinds}


def _operations(tree: JavaSyntaxTree, first: MatchResult, second: MatchResult) -> list[tuple[str, float]]:
    host = tree.nodes[first.local_subtree.root_id]
    one = first.pattern.signature.get("root_kind", "")
    two = second.pattern.signature.get("root_kind", "")
    role = syntactic_role(host.raw_kind)
    if syntactic_role(one) != role or syntactic_role(two) != role:
        return []
    operations = []
    if role == "statement":
        if two in STATEMENT_WRAPPERS:
            operations.append(("wrapper", 0.9))
        return operations
    if role != "expression":
        return []
    if two in WRAPPER_KINDS:
        operations.append(("wrapper", 0.9))
    if one in {"MEMBER_REFERENCE", "LAMBDA_EXPRESSION"} and two == "METHOD_INVOCATION":
        operations.append(("nested_expression", 0.9))
    if one in REFERENCE_OPERANDS:
        operations.append(("role_extension", 0.8))
    constraints = {"lambda", "generics", "checked_exception"}
    if _features(first) & constraints and _features(second) & constraints:
        operations.append(("constraint_conjunction", 0.7))
    if any(tree.nodes[n].raw_kind == "CONDITIONAL_EXPRESSION" for n in tree.ancestors(host.node_id)):
        operations.append(("branch_splitting", 0.7))
    return operations


def same_node_plans(tree: JavaSyntaxTree, groups: dict[int, list[MatchResult]], pairs_per_anchor: int = 4) -> list[FusionPlan]:
    plans = []
    for anchor_id, group in groups.items():
        if tree.nodes[anchor_id].raw_kind in EXCLUDED_ANCHORS:
            continue
        candidates = []
        for left, right in combinations(group[:4], 2):
            if left.pattern.bug_id == right.pattern.bug_id:
                continue
            structure_distance = _distance(left.pattern.signature.get("node_kind_multiset", {}), right.pattern.signature.get("node_kind_multiset", {}))
            type_feasibility = math.sqrt(max(0.0, left.components.get("feature_compatibility", 1.0)) * max(0.0, right.components.get("feature_compatibility", 1.0)))
            compatibility = structure_distance * type_feasibility
            orthogonality = mechanism_orthogonality(left, right)
            base_score = math.sqrt(max(0.0, left.final_score * right.final_score))
            for first, second in ((left, right), (right, left)):
                for operation, prior in _operations(tree, first, second):
                    score = base_score * compatibility * orthogonality * prior
                    if score <= 0:
                        continue
                    candidates.append(FusionPlan("same_node", operation, first, second, None, score, {
                        "match_geometric_mean": base_score,
                        "compatibility": compatibility,
                        "transform_non_conflict": structure_distance,
                        "type_constraint_feasible": type_feasibility,
                        "mechanism_orthogonality": orthogonality,
                        "transform_composability": prior,
                    }))
        candidates.sort(key=plan_sort_key)
        plans.extend(candidates[:pairs_per_anchor])
    return sorted(plans, key=plan_sort_key)


def lowest_common_ancestor(tree: JavaSyntaxTree, first: int, second: int) -> int:
    result = tree.root_id
    for a, b in zip(tree.ancestors(first), tree.ancestors(second)):
        if a != b:
            break
        result = a
    return result


def disjoint_spans(first: tuple[int, int], second: tuple[int, int]) -> bool:
    return first[1] <= second[0] or second[1] <= first[0]


def _span(match: MatchResult) -> tuple[int, int]:
    return match.local_subtree.start_byte, match.local_subtree.end_byte


def build_anchor_interaction_graph(tree: JavaSyntaxTree, groups: dict[int, list[MatchResult]], anchor_limit: int = 6) -> list[AnchorEdge]:
    roots = sorted((root for root in groups if tree.nodes[root].raw_kind not in EXCLUDED_ANCHORS), key=lambda root: (-groups[root][0].final_score, root))[:anchor_limit]
    edges = []
    for a, b in combinations(roots, 2):
        pa, pb = tree.ancestors(a), tree.ancestors(b)
        lca = lowest_common_ancestor(tree, a, b)
        na, nb = tree.nodes[a], tree.nodes[b]
        nonoverlap = disjoint_spans((na.start_byte, na.end_byte), (nb.start_byte, nb.end_byte))
        evidence = {"lca_id": lca, "lca_kind": tree.nodes[lca].raw_kind, "span_non_conflict": nonoverlap}
        if a in pb or b in pa:
            outer, inner, path = (a, b, pb) if a in pb else (b, a, pa)
            distance = len(path) - 1 - path.index(outer)
            strength = min(1.0, 1.0 / (1 + distance) + (0.1 if any(tree.nodes[n].raw_kind in {"METHOD", "BLOCK"} for n in tree.ancestors(outer)) else 0.0))
            edges.append(AnchorEdge("ancestor_descendant", outer, inner, strength, {**evidence, "ancestor_distance": distance, "execution_exclusion": "overlapping_anchor_spans"}))
            continue
        if na.parent_id == nb.parent_id and na.parent_id is not None:
            parent_kind = tree.nodes[na.parent_id].raw_kind
            slot_distance = abs(na.child_index - nb.child_index)
            semantic_slots = parent_kind in EXPRESSION_KINDS
            if slot_distance == 1 or semantic_slots:
                strength = 0.95 if parent_kind == "CONDITIONAL_EXPRESSION" else (0.8 if semantic_slots else 0.6)
                edges.append(AnchorEdge("sibling", a, b, strength, {**evidence, "parent_id": na.parent_id, "parent_kind": parent_kind, "child_indices": [na.child_index, nb.child_index]}))
        if tree.nodes[lca].raw_kind in CONTROL_KINDS:
            branch_a = pa[pa.index(lca) + 1]
            branch_b = pb[pb.index(lca) + 1]
            if branch_a != branch_b:
                edges.append(AnchorEdge("control_flow", a, b, 0.65, {**evidence, "branch_roots": [branch_a, branch_b]}))
    return edges


def edge_compatibility(first: MatchResult, second: MatchResult, edge: AnchorEdge) -> float:
    one, two = _features(first), _features(second)
    if edge.edge_type == "control_flow":
        return 1.0 if (one | two) & {"conditional", "switch", "loop", "try_catch", "checked_exception"} else 0.5
    if edge.edge_type in {"sibling", "ancestor_descendant"}:
        interacting = {"lambda", "method_reference", "generics", "cast", "conditional", "assignment"}
        return 1.0 if one & interacting and two & interacting else 0.5
    return 0.0


def distance_prior(tree: JavaSyntaxTree, first: MatchResult, second: MatchResult) -> float:
    path = tree.ancestors(lowest_common_ancestor(tree, first.local_subtree.root_id, second.local_subtree.root_id))
    kinds = {tree.nodes[n].raw_kind for n in path}
    scope = 1.0 if "METHOD" in kinds else (0.7 if kinds & {"CLASS", "INTERFACE", "ENUM", "RECORD"} else 0.4)
    a, b = _span(first), _span(second)
    gap = max(0, max(a[0], b[0]) - min(a[1], b[1]))
    return scope * math.exp(-gap / 2000.0)


def cross_node_plans(tree: JavaSyntaxTree, groups: dict[int, list[MatchResult]], edges: list[AnchorEdge], pairs_per_edge: int = 4, max_bridge_bytes: int = 8000) -> list[FusionPlan]:
    plans = []
    for edge in edges:
        if not edge.evidence["span_non_conflict"]:
            continue
        lca = tree.nodes[edge.evidence["lca_id"]]
        if lca.raw_kind == "COMPILATION_UNIT" or lca.end_byte - lca.start_byte > max_bridge_bytes:
            continue
        if tree.snippet(lca.start_byte, lca.end_byte).strip() == tree.source.strip():
            continue
        candidates = []
        for first, second in product(groups[edge.source_id][:4], groups[edge.target_id][:4]):
            if first.pattern.bug_id == second.pattern.bug_id:
                continue
            compatible = edge_compatibility(first, second, edge)
            orthogonality = mechanism_orthogonality(first, second)
            distance = distance_prior(tree, first, second)
            base_score = math.sqrt(max(0.0, first.final_score * second.final_score))
            score = base_score * edge.strength * compatible * orthogonality * distance
            if score > 0:
                candidates.append(FusionPlan("cross_node", edge.edge_type, first, second, edge, score, {
                    "match_geometric_mean": base_score,
                    "interaction_strength": edge.strength,
                    "edge_compatibility": compatible,
                    "mechanism_orthogonality": orthogonality,
                    "distance_prior": distance,
                }))
        candidates.sort(key=plan_sort_key)
        plans.extend(candidates[:pairs_per_edge])
    return sorted(plans, key=plan_sort_key)


def plan_sort_key(plan: FusionPlan) -> tuple:
    return (-plan.score, plan.fusion_kind, plan.operation, plan.match_1.pattern.bug_id, plan.match_2.pattern.bug_id, plan.match_1.local_subtree.root_id, plan.match_2.local_subtree.root_id)
