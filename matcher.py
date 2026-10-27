"""Two-stage javac-native node retrieval and local subtree reranking."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
import re
from typing import Any, Iterable

from .library import HistoricalLibrary
from .models import HistoricalPattern, LocalSubtree, MatchResult
from .parser import JavaSyntaxTree


STRUCTURAL_WEIGHTS = {
    "required_kind_f1": 0.25,
    "node_multiset": 0.10,
    "edge_multiset": 0.20,
    "path_set": 0.13,
    "root_kind": 0.08,
    "ancestor_path": 0.05,
    "ordered_children": 0.05,
    "shape": 0.09,
    "size": 0.05,
}
TOKEN_RE = re.compile(r"[A-Za-z_$][A-Za-z0-9_$-]{2,}")
STOP_WORDS = {
    "and", "are", "compiler", "error", "from", "java", "javac", "not", "the", "this", "with",
}
FEATURE_RULES = (
    ("lambda", re.compile(r"\blambda(?:expression)?s?\b"), {"LAMBDA_EXPRESSION"}, None),
    (
        "method_reference",
        re.compile(r"\bmethod\s+references?\b|\bmethodreferences?\b"),
        {"MEMBER_REFERENCE"},
        re.compile(r"::"),
    ),
    (
        "checked_exception",
        re.compile(r"\bchecked\s+exceptions?\b|\bgeneric\s+exceptions?\b"),
        {"THROW", "TRY", "CATCH"},
        re.compile(r"\bthrows\b|\b(?:IOException|Exception|Throwable)\b"),
    ),
    (
        "generics",
        re.compile(
            r"\bgenerics?\b|\btype\s+(?:parameter|inference|variable|bound)s?\b|"
            r"\bdiamond(?:\s+operator|\s+notation)?\b|\bwildcards?\b|\bintersection\s+types?\b"
        ),
        {
            "PARAMETERIZED_TYPE",
            "TYPE_PARAMETER",
            "UNBOUNDED_WILDCARD",
            "EXTENDS_WILDCARD",
            "SUPER_WILDCARD",
            "INTERSECTION_TYPE",
        },
        None,
    ),
    (
        "annotation",
        re.compile(r"\bannotations?\b|\btype-use\b|\bannotation\s+processing\b"),
        {"ANNOTATION", "ANNOTATED_TYPE", "ANNOTATION_TYPE"},
        re.compile(r"@\s*[A-Za-z_$]"),
    ),
    ("record", re.compile(r"\brecords?\b"), {"RECORD", "RECORD_COMPONENT", "RECORD_PATTERN"}, None),
    ("sealed_type", re.compile(r"\bsealed\b|\bpermits\b"), set(), re.compile(r"\b(?:sealed|permits)\b")),
    (
        "switch",
        re.compile(r"\bswitch(?:\s+(?:expression|statement|pattern))?s?\b"),
        {"SWITCH", "SWITCH_EXPRESSION", "CASE", "YIELD"},
        None,
    ),
    (
        "pattern_matching",
        re.compile(r"\bpattern\s+matching\b|\brecord\s+patterns?\b|\btype\s+patterns?\b"),
        {"BINDING_PATTERN", "DECONSTRUCTION_PATTERN", "ANY_PATTERN", "INSTANCE_OF"},
        None,
    ),
    ("instanceof", re.compile(r"\binstanceof\b"), {"INSTANCE_OF"}, None),
    (
        "inheritance",
        re.compile(r"\binheritance\b|\bextends\b|\bimplements?\b|\bsubclasses?\b"),
        {"CLASS", "INTERFACE", "ENUM", "RECORD"},
        re.compile(r"\b(?:extends|implements)\b"),
    ),
    (
        "nested_type",
        re.compile(r"\b(?:inner|nested|local|anonymous)\s+(?:class|classes|type|types)\b"),
        {"CLASS", "NEW_CLASS"},
        None,
    ),
    ("interface", re.compile(r"\binterfaces?\b"), {"INTERFACE"}, None),
    ("enum", re.compile(r"\benums?\b"), {"ENUM", "ENUM_CONSTANT"}, None),
    (
        "constructor",
        re.compile(r"\bconstructors?\b|\bsuperconstructor\b"),
        {"METHOD", "NEW_CLASS"},
        re.compile(r"\b(?:super|this)\s*\("),
    ),
    ("array", re.compile(r"\barrays?\b"), {"ARRAY_TYPE", "ARRAY_ACCESS", "NEW_ARRAY"}, None),
    ("cast", re.compile(r"\bcasts?\b|\bcasting\b"), {"TYPE_CAST"}, None),
    ("varargs", re.compile(r"\bvarargs\b"), {"VARIABLE"}, re.compile(r"\.\.\.")),
    (
        "loop",
        re.compile(r"\bloop\b|\bfor-loop\b|\bdo\.\.\.while\b"),
        {"FOR_LOOP", "ENHANCED_FOR_LOOP", "WHILE_LOOP", "DO_WHILE_LOOP"},
        None,
    ),
    ("try_catch", re.compile(r"\btry(?:-with-resources)?\b|\bcatch\b"), {"TRY", "CATCH"}, None),
    ("synchronized", re.compile(r"\bsynchronized\b"), {"SYNCHRONIZED"}, None),
    (
        "assignment",
        re.compile(r"\bassignments?\b|\breassignment\b"),
        {
            "ASSIGNMENT",
            "MULTIPLY_ASSIGNMENT",
            "DIVIDE_ASSIGNMENT",
            "REMAINDER_ASSIGNMENT",
            "PLUS_ASSIGNMENT",
            "MINUS_ASSIGNMENT",
            "LEFT_SHIFT_ASSIGNMENT",
            "RIGHT_SHIFT_ASSIGNMENT",
            "UNSIGNED_RIGHT_SHIFT_ASSIGNMENT",
            "AND_ASSIGNMENT",
            "XOR_ASSIGNMENT",
            "OR_ASSIGNMENT",
        },
        None,
    ),
    ("conditional", re.compile(r"\bconditional\b|\bternary\b"), {"CONDITIONAL_EXPRESSION"}, None),
    ("null", re.compile(r"\bnull\b"), set(), re.compile(r"\bnull\b")),
    ("import", re.compile(r"\bimports?\b"), {"IMPORT"}, None),
    ("package", re.compile(r"\bpackages?\b"), {"PACKAGE"}, None),
    ("module", re.compile(r"\bmodules?\b|\bmodularization\b"), {"MODULE"}, None),
)
FEATURE_CONTEXT_ROOTS = {
    "ANNOTATION_TYPE",
    "BLOCK",
    "CLASS",
    "ENUM",
    "EXPRESSION_STATEMENT",
    "INTERFACE",
    "METHOD",
    "RECORD",
    "RETURN",
    "VARIABLE",
}


class SubtreeMatcher:
    def __init__(self, library: HistoricalLibrary) -> None:
        self.library = library

    def prefilter(self, tree: JavaSyntaxTree, limit: int = 200) -> list[tuple[HistoricalPattern, float]]:
        query_kinds = tree.kinds()
        ranked: list[tuple[HistoricalPattern, float]] = []
        for bug_id in self.library.candidates(query_kinds):
            pattern = self.library.patterns[bug_id]
            denominator = sum(self.library.idf(kind) for kind in pattern.required_kinds)
            numerator = sum(self.library.idf(kind) for kind in pattern.required_kinds if kind in query_kinds)
            score = numerator / denominator if denominator else 0.0
            if score > 0.0:
                ranked.append((pattern, score))
        ranked.sort(key=lambda item: (-item[1], -item[0].corpus_rarity, item[0].bug_id))
        return ranked[:limit] if limit > 0 else ranked

    def match(
        self,
        tree: JavaSyntaxTree,
        *,
        prefilter_limit: int = 200,
        top_k: int = 20,
        min_score: float = 0.0,
        query_language_feature: str = "",
        query_phases: Iterable[str] = (),
        query_diagnostic: str = "",
        max_span_bytes: int = 0,
        max_span_fraction: float = 0.0,
    ) -> list[MatchResult]:
        results: list[MatchResult] = []
        phases = tuple(query_phases)
        for pattern, prefilter_score in self.prefilter(tree, prefilter_limit):
            local = locate_local_subtree(tree, pattern)
            if local is None or not span_allowed(tree, local, max_span_bytes, max_span_fraction):
                continue
            result = score_match(
                tree,
                pattern,
                local,
                prefilter_score,
                language_feature=query_language_feature,
                phases=phases,
                diagnostic=query_diagnostic,
            )
            if result is None or result.final_score < min_score:
                continue
            results.append(result)
        results.sort(key=lambda item: (-item.final_score, item.pattern.bug_id))
        return results[:top_k] if top_k > 0 else results

    def match_anchors(
        self,
        tree: JavaSyntaxTree,
        *,
        candidates_per_pattern: int = 3,
        prefilter_limit: int = 200,
        top_k: int = 20,
        min_score: float = 0.0,
        query_language_feature: str = "",
        query_phases: Iterable[str] = (),
        query_diagnostic: str = "",
        max_span_bytes: int = 0,
        max_span_fraction: float = 0.0,
    ) -> dict[int, list[MatchResult]]:
        """Group per-pattern candidate anchors for same-node and cross-node pairing.

        Unlike match(), every pattern may report several distinct anchors, so two
        patterns can be paired on one anchor (principle 9) or on interacting
        anchors (principle 10) even when their best anchors differ.
        """
        groups: dict[int, list[MatchResult]] = defaultdict(list)
        phases = tuple(query_phases)
        for pattern, prefilter_score in self.prefilter(tree, prefilter_limit):
            for local in locate_local_subtrees(tree, pattern, candidates_per_pattern):
                if not span_allowed(tree, local, max_span_bytes, max_span_fraction):
                    continue
                result = score_match(
                    tree,
                    pattern,
                    local,
                    prefilter_score,
                    language_feature=query_language_feature,
                    phases=phases,
                    diagnostic=query_diagnostic,
                )
                if result is None or result.final_score < min_score:
                    continue
                groups[local.root_id].append(result)
        for group in groups.values():
            group.sort(key=lambda item: (-item.final_score, item.pattern.bug_id))
        if top_k > 0:
            groups = {root: group[:top_k] for root, group in groups.items()}
        return dict(groups)


def span_allowed(
    tree: JavaSyntaxTree,
    local: LocalSubtree,
    max_span_bytes: int = 0,
    max_span_fraction: float = 0.0,
) -> bool:
    """Apply the pipeline's host-size gates before anchors consume top-k slots."""
    span = local.end_byte - local.start_byte
    if max_span_bytes and span > max_span_bytes:
        return False
    if max_span_fraction:
        total = max(len(tree.source_bytes), 1)
        if span / total > max_span_fraction:
            return False
    return True


def score_match(
    tree: JavaSyntaxTree,
    pattern: HistoricalPattern,
    local: LocalSubtree,
    prefilter_score: float,
    *,
    language_feature: str,
    phases: tuple[str, ...],
    diagnostic: str,
) -> MatchResult | None:
    components = structural_components(pattern, local)
    structural_score = sum(STRUCTURAL_WEIGHTS[key] * components[key] for key in STRUCTURAL_WEIGHTS)
    context_score = semantic_context_score(
        pattern,
        language_feature=language_feature,
        phases=phases,
        diagnostic=diagnostic,
    )
    quality_score = reference_quality(pattern)
    specificity_score = reference_specificity(pattern)
    feature_score, feature_evidence = feature_compatibility_score(tree, pattern, local)
    if language_feature.strip() or phases or diagnostic.strip():
        raw_score = (
            0.10 * prefilter_score
            + 0.60 * structural_score
            + 0.15 * feature_score
            + 0.15 * context_score
        )
    else:
        raw_score = 0.15 * prefilter_score + 0.70 * structural_score + 0.15 * feature_score
    final_score = raw_score * (0.75 + 0.25 * quality_score) * specificity_score
    final_score = max(0.0, min(1.0, final_score))
    components["semantic_context"] = context_score
    components["feature_compatibility"] = feature_score
    return MatchResult(
        pattern=pattern,
        local_subtree=local,
        prefilter_score=prefilter_score,
        structural_score=structural_score,
        quality_score=quality_score,
        specificity_score=specificity_score,
        final_score=final_score,
        components=components,
        feature_evidence=feature_evidence,
    )


def locate_local_subtrees(
    tree: JavaSyntaxTree,
    pattern: HistoricalPattern,
    count: int = 1,
) -> list[LocalSubtree]:
    """Return up to ``count`` ranked anchor subtrees for one pattern.

    Candidates keep the historical preference order (coverage, root-kind match,
    size proximity, span); taking several lets a second pattern anchor on a
    different node than the pattern's single best anchor.
    """
    required = tuple(dict.fromkeys(pattern.required_kinds))
    reference_counts = Counter(
        {
            str(kind): int(count_)
            for kind, count_ in (pattern.signature.get("node_kind_multiset") or {}).items()
            if int(count_) > 0
        }
    )
    if not reference_counts:
        reference_counts.update(required)
    occurrences = {kind: tree.occurrences(kind) for kind in reference_counts}
    present_ids = [node_id for values in occurrences.values() for node_id in values]
    if not present_ids or count < 1:
        return []

    ancestor_paths = {node_id: tree.ancestors(node_id) for node_id in set(present_ids)}
    ancestor_sets = {node_id: set(path) for node_id, path in ancestor_paths.items()}
    root_candidates = {ancestor for path in ancestor_paths.values() for ancestor in path}
    historical_root = str(pattern.signature.get("root_kind") or "")
    candidates: list[tuple[tuple[Any, ...], int, set[int], set[str]]] = []

    for root_id in root_candidates:
        node = tree.nodes[root_id]
        root_match = historical_root == node.raw_kind
        if root_match:
            selected = _align_reference_subtree(tree, root_id, pattern.subtree_nodes)
            matched_node_count = len(selected)
            covered = {tree.nodes[node_id].raw_kind for node_id in selected}
        else:
            selected = {root_id}
            covered = set()
            matched_node_count = 0
            for kind, target_count in reference_counts.items():
                choices = [node_id for node_id in occurrences[kind] if root_id in ancestor_sets[node_id]]
                if not choices:
                    continue
                chosen_nodes = sorted(
                    choices,
                    key=lambda node_id: (
                        len(_path_from_ancestor(tree, root_id, node_id)),
                        tree.nodes[node_id].start_byte,
                    ),
                )[:target_count]
                for chosen in chosen_nodes:
                    selected.update(_path_from_ancestor(tree, root_id, chosen))
                matched_node_count += len(chosen_nodes)
                covered.add(kind)
        if not covered:
            continue
        key = (
            -matched_node_count,
            0 if root_match else 1,
            abs(len(selected) - pattern.node_count),
            len(selected),
            node.end_byte - node.start_byte,
            node.start_byte,
            root_id,
        )
        candidates.append((key, root_id, selected, covered))

    candidates.sort(key=lambda item: item[0])
    subtrees: list[LocalSubtree] = []
    seen_roots: set[int] = set()
    for _key, root_id, selected, covered in candidates:
        if len(subtrees) >= count:
            break
        if root_id in seen_roots:
            continue
        seen_roots.add(root_id)
        signature = make_query_signature(tree, root_id, selected, required, historical_root)
        ancestor_path = tuple(tree.nodes[node_id].raw_kind for node_id in tree.ancestors(root_id))
        root = tree.nodes[root_id]
        subtrees.append(
            LocalSubtree(
                root_id=root_id,
                selected_ids=tuple(sorted(selected, key=lambda item: (tree.nodes[item].start_byte, item))),
                covered_kinds=tuple(kind for kind in required if kind in covered),
                missing_kinds=tuple(kind for kind in required if kind not in covered),
                signature=signature,
                ancestor_path=ancestor_path,
                start_byte=root.start_byte,
                end_byte=root.end_byte,
            )
        )
    return subtrees


def locate_local_subtree(tree: JavaSyntaxTree, pattern: HistoricalPattern) -> LocalSubtree | None:
    """Backward-compatible single-best-anchor view of locate_local_subtrees."""
    subtrees = locate_local_subtrees(tree, pattern, count=1)
    return subtrees[0] if subtrees else None


def _align_reference_subtree(
    tree: JavaSyntaxTree,
    candidate_root_id: int,
    reference_nodes: tuple[dict[str, Any], ...],
) -> set[int]:
    """Greedily embed the restored ordered reference tree under one native root."""
    reference = {int(node["relative_id"]): node for node in reference_nodes}
    children: dict[int, list[int]] = defaultdict(list)
    roots: list[int] = []
    for node_id, node in reference.items():
        parent = node.get("parent_relative_id")
        if parent is None:
            roots.append(node_id)
        else:
            children[int(parent)].append(node_id)
    if len(roots) != 1:
        return set()
    for child_ids in children.values():
        child_ids.sort(key=lambda node_id: (int(reference[node_id].get("child_index", -1)), node_id))

    selected: set[int] = set()

    def align(reference_id: int, candidate_id: int) -> None:
        if str(reference[reference_id].get("kind")) != tree.nodes[candidate_id].raw_kind:
            return
        selected.add(candidate_id)
        available = list(tree.nodes[candidate_id].child_ids)
        used: set[int] = set()
        for reference_child_id in children.get(reference_id, ()):
            reference_child = reference[reference_child_id]
            expected_kind = str(reference_child.get("kind"))
            expected_index = int(reference_child.get("child_index", -1))
            choices = [
                node_id
                for node_id in available
                if node_id not in used and tree.nodes[node_id].raw_kind == expected_kind
            ]
            if not choices:
                continue
            candidate_child_id = min(
                choices,
                key=lambda node_id: (
                    tree.nodes[node_id].child_index != expected_index,
                    abs(tree.nodes[node_id].child_index - expected_index),
                    tree.nodes[node_id].start_byte,
                ),
            )
            used.add(candidate_child_id)
            align(reference_child_id, candidate_child_id)

    align(roots[0], candidate_root_id)
    return selected


def _path_from_ancestor(tree: JavaSyntaxTree, root_id: int, node_id: int) -> list[int]:
    path: list[int] = []
    current: int | None = node_id
    while current is not None:
        path.append(current)
        if current == root_id:
            return list(reversed(path))
        current = tree.nodes[current].parent_id
    return []


def make_query_signature(
    tree: JavaSyntaxTree,
    root_id: int,
    selected: set[int],
    required: tuple[str, ...],
    historical_root: str,
) -> dict[str, Any]:
    kinds: dict[int, str] = {}
    for node_id in selected:
        node = tree.nodes[node_id]
        preferred = historical_root if node_id == root_id else ""
        if preferred and preferred == node.raw_kind:
            kinds[node_id] = preferred
            continue
        kinds[node_id] = node.raw_kind

    node_multiset = Counter(kinds.values())
    edge_multiset: Counter[str] = Counter()
    children: dict[int, list[int]] = defaultdict(list)
    for node_id in selected:
        parent_id = tree.nodes[node_id].parent_id
        if parent_id in selected:
            edge_multiset[f"{kinds[parent_id]}>{kinds[node_id]}"] += 1
            children[parent_id].append(node_id)

    paths: list[str] = []
    for node_id in selected:
        path = _path_from_ancestor(tree, root_id, node_id)
        paths.append(">".join(kinds[item] for item in path if item in selected))
    ordered = _ordered_tokens(
        kinds,
        children,
        child_index=lambda node_id: tree.nodes[node_id].child_index,
    )
    max_depth, branch_count, max_degree = _shape_values(root_id, children)
    return {
        "version": "javac-native-subtree-signature-v1",
        "kind_domain": "com.sun.source.tree.Tree.Kind",
        "root_kind": kinds[root_id],
        "required_kinds": list(required),
        "node_kind_multiset": dict(sorted(node_multiset.items())),
        "edge_kind_multiset": dict(sorted(edge_multiset.items())),
        "root_to_node_paths": sorted(set(paths)),
        "ordered_children": ordered,
        "node_count": len(selected),
        "max_depth": max_depth,
        "branch_count": branch_count,
        "max_degree": max_degree,
    }


def structural_components(pattern: HistoricalPattern, local: LocalSubtree) -> dict[str, float]:
    reference = enriched_reference_signature(pattern)
    candidate = local.signature
    return {
        "required_kind_f1": _counter_f1(reference.get("required_kinds") or (), candidate.get("node_kind_multiset") or {}),
        "node_multiset": _multiset_jaccard(reference.get("node_kind_multiset") or {}, candidate.get("node_kind_multiset") or {}),
        "edge_multiset": _multiset_jaccard(reference.get("edge_kind_multiset") or {}, candidate.get("edge_kind_multiset") or {}),
        "path_set": _set_jaccard(reference.get("root_to_node_paths") or (), candidate.get("root_to_node_paths") or ()),
        "root_kind": float(reference.get("root_kind") == candidate.get("root_kind")),
        "ancestor_path": _suffix_similarity(pattern.ancestor_path, local.ancestor_path),
        "ordered_children": _set_jaccard(reference.get("ordered_children") or (), candidate.get("ordered_children") or ()),
        "shape": _shape_similarity(reference, candidate),
        "size": _size_similarity(int(reference.get("node_count") or 0), int(candidate.get("node_count") or 0)),
    }


def enriched_reference_signature(pattern: HistoricalPattern) -> dict[str, Any]:
    signature = dict(pattern.signature)
    kinds: dict[int, str] = {}
    children: dict[int, list[int]] = defaultdict(list)
    child_indexes: dict[int, int] = {}
    root_id = 0
    for raw in pattern.subtree_nodes:
        node_id = int(raw.get("relative_id", 0))
        kinds[node_id] = str(raw.get("kind") or "UNKNOWN")
        child_indexes[node_id] = int(raw.get("child_index", -1))
        parent = raw.get("parent_relative_id")
        if parent is None:
            root_id = node_id
        else:
            children[int(parent)].append(node_id)
    signature["ordered_children"] = _ordered_tokens(kinds, children, child_index=lambda node_id: child_indexes[node_id])
    max_depth, branch_count, max_degree = _shape_values(root_id, children)
    signature["max_depth"] = max_depth
    signature["branch_count"] = branch_count
    signature["max_degree"] = max_degree
    return signature


def _ordered_tokens(
    kinds: dict[int, str],
    children: dict[int, list[int]],
    *,
    child_index: Any,
) -> list[str]:
    tokens: list[str] = []
    for parent_id, child_ids in children.items():
        ordered = sorted(child_ids, key=lambda node_id: (child_index(node_id), node_id))
        if ordered:
            tokens.append(f"{kinds[parent_id]}>[{','.join(kinds[node_id] for node_id in ordered)}]")
    return sorted(tokens)


def _shape_values(root_id: int, children: dict[int, list[int]]) -> tuple[int, int, int]:
    max_depth = 0
    branch_count = 0
    max_degree = 0
    stack = [(root_id, 0)]
    seen: set[int] = set()
    while stack:
        node_id, depth = stack.pop()
        if node_id in seen:
            continue
        seen.add(node_id)
        degree = len(children.get(node_id, ()))
        max_depth = max(max_depth, depth)
        max_degree = max(max_degree, degree)
        branch_count += int(degree > 1)
        stack.extend((child_id, depth + 1) for child_id in children.get(node_id, ()))
    return max_depth, branch_count, max_degree


def _counter_f1(required: Iterable[str], actual: dict[str, int]) -> float:
    required_counts = Counter(required)
    actual_presence = {kind: 1 for kind, count in actual.items() if int(count) > 0}
    true_positive = sum(min(count, actual_presence.get(kind, 0)) for kind, count in required_counts.items())
    if not required_counts or not actual:
        return 0.0
    precision = true_positive / len(actual_presence)
    recall = true_positive / sum(required_counts.values())
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _multiset_jaccard(left: dict[str, int], right: dict[str, int]) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 0.0
    numerator = sum(min(int(left.get(key, 0)), int(right.get(key, 0))) for key in keys)
    denominator = sum(max(int(left.get(key, 0)), int(right.get(key, 0))) for key in keys)
    return numerator / denominator if denominator else 0.0


def _set_jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def _suffix_similarity(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    if not left or not right:
        return 0.0
    common = 0
    for left_item, right_item in zip(reversed(left), reversed(right)):
        if left_item != right_item:
            break
        common += 1
    return 2 * common / (len(left) + len(right))


def _shape_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    parts = []
    for key in ("max_depth", "branch_count", "max_degree"):
        first = int(left.get(key) or 0)
        second = int(right.get(key) or 0)
        parts.append(1.0 - abs(first - second) / max(first, second, 1))
    return sum(parts) / len(parts)


def _size_similarity(left: int, right: int) -> float:
    return 1.0 - abs(left - right) / max(left, right, 1)


def reference_quality(pattern: HistoricalPattern) -> float:
    status_factor = 1.0 if pattern.status == "candidate" else 0.85
    return max(0.0, min(1.0, pattern.confidence * pattern.coverage * status_factor))


def reference_specificity(pattern: HistoricalPattern) -> float:
    node_factor = min(pattern.node_count / 5.0, 1.0)
    rarity_factor = 1.0 - math.exp(-max(pattern.corpus_rarity, 0.0) / 4.0)
    collision_factor = 1.0 / math.sqrt(max(pattern.collision_count, 1))
    score = 0.50 + 0.25 * node_factor + 0.15 * rarity_factor + 0.10 * collision_factor
    if pattern.node_count == 1:
        score = min(score, 0.68)
    return max(0.45, min(1.0, score))


def feature_compatibility_score(
    tree: JavaSyntaxTree,
    pattern: HistoricalPattern,
    local: LocalSubtree,
) -> tuple[float, dict[str, Any]]:
    """Compare language features named in historical L with the host context."""
    feature_text = pattern.language_feature.lower().replace("_", " ")
    recognized = [rule for rule in FEATURE_RULES if rule[1].search(feature_text)]
    context_root_id = _feature_context_root(tree, local.root_id)
    context_ids = tree.descendants(context_root_id)
    context_kinds = {tree.nodes[node_id].raw_kind for node_id in context_ids}
    context_node = tree.nodes[context_root_id]
    context_source = tree.snippet(context_node.start_byte, context_node.end_byte)

    matched: list[str] = []
    missing: list[str] = []
    for name, _feature_pattern, expected_kinds, source_pattern in recognized:
        compatible = bool(expected_kinds & context_kinds)
        if source_pattern is not None and source_pattern.search(context_source):
            compatible = True
        (matched if compatible else missing).append(name)

    score = len(matched) / len(recognized) if recognized else 0.5
    evidence = {
        "score": round(score, 6),
        "scope_root_node_id": context_root_id,
        "scope_root_kind": context_node.raw_kind,
        "recognized": [rule[0] for rule in recognized],
        "matched": matched,
        "missing": missing,
        "neutral": not recognized,
    }
    return score, evidence


def _feature_context_root(tree: JavaSyntaxTree, local_root_id: int) -> int:
    for node_id in reversed(tree.ancestors(local_root_id)):
        if node_id == local_root_id:
            continue
        node = tree.nodes[node_id]
        if node.raw_kind in FEATURE_CONTEXT_ROOTS:
            return node_id
    return local_root_id


def semantic_context_score(
    pattern: HistoricalPattern,
    *,
    language_feature: str,
    phases: tuple[str, ...],
    diagnostic: str,
) -> float:
    scores: list[tuple[float, float]] = []
    if language_feature.strip():
        scores.append((0.35, _token_jaccard(language_feature, pattern.language_feature)))
    if phases:
        scores.append((0.25, _set_jaccard((item.lower() for item in phases), (item.lower() for item in pattern.phases))))
    if diagnostic.strip():
        scores.append((0.40, _token_jaccard(diagnostic, pattern.diagnostic)))
    total_weight = sum(weight for weight, _score in scores)
    return sum(weight * score for weight, score in scores) / total_weight if total_weight else 0.0


def _token_jaccard(left: str, right: str) -> float:
    def tokens(value: str) -> set[str]:
        return {token.lower() for token in TOKEN_RE.findall(value) if token.lower() not in STOP_WORDS}

    return _set_jaccard(tokens(left), tokens(right))
