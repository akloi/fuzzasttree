"""End-to-end matching, snippet mutation, validation, and output writing."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import re
from typing import Iterable

from .library import HistoricalLibrary
from .matcher import SubtreeMatcher
from .models import FusionMutationRequest, FusionPlan, MatchResult, MutationRequest
from .mutator import MutationOutput, SnippetMutationError, SnippetMutator, compose_subtree, compose_two_spans, parse_fusion_output
from .parser import JavaSyntaxTree, parse_java


@dataclass(slots=True)
class PipelineOptions:
    prefilter_limit: int = 200
    top_k: int = 20
    max_outputs: int = 5
    min_score: float = 0.0
    max_host_bytes: int = 4000
    max_host_fraction: float = 0.75
    allow_new_syntax_errors: bool = False
    query_language_feature: str = ""
    query_phases: tuple[str, ...] = ()
    query_diagnostic: str = ""
    fusion: str = "off"
    fusion_pairs: int = 4
    anchor_limit: int = 6
    fusion_budget: float = 0.4
    anchor_candidates: int = 3
    fusion_selector: str = "greedy"
    pattern_cap: int = 0

    def __post_init__(self) -> None:
        if self.fusion not in {"off", "same", "cross", "both"}:
            raise ValueError("fusion must be off, same, cross, or both")
        if self.fusion_pairs < 1 or self.anchor_limit < 1 or self.anchor_candidates < 1:
            raise ValueError("fusion_pairs, anchor_limit, and anchor_candidates must be positive")
        if not 0 <= self.fusion_budget <= 1:
            raise ValueError("fusion_budget must lie between 0 and 1")
        if self.fusion_selector not in {"greedy", "matching"}:
            raise ValueError("fusion_selector must be greedy or matching")
        if self.pattern_cap < 0:
            raise ValueError("pattern_cap must not be negative")


class MutationPipeline:
    def __init__(self, library: HistoricalLibrary) -> None:
        self.library = library
        self.matcher = SubtreeMatcher(library)
        # Run-level emission counts per bug id, shared across inputs so the
        # matching selector can decay repeatedly emitted generic pairs.
        self._fusion_usage: dict[str, int] = {}

    def match_source(self, source: str, source_name: str, options: PipelineOptions) -> tuple[JavaSyntaxTree, list[MatchResult]]:
        tree = parse_java(source, source_name)
        matches = self.matcher.match(
            tree,
            prefilter_limit=options.prefilter_limit,
            top_k=options.top_k,
            min_score=options.min_score,
            query_language_feature=options.query_language_feature,
            query_phases=options.query_phases,
            query_diagnostic=options.query_diagnostic,
            max_span_bytes=options.max_host_bytes,
            max_span_fraction=options.max_host_fraction,
        )
        return tree, matches

    def mutate_file(
        self,
        source_path: Path,
        output_dir: Path,
        mutator: SnippetMutator,
        options: PipelineOptions,
    ) -> dict[str, object]:
        if options.fusion != "off":
            return FusionPipeline(self.library, usage=self._fusion_usage).mutate_file(source_path, output_dir, mutator, options)
        source = source_path.read_text(encoding="utf-8", errors="replace")
        tree, matches = self.match_source(source, source_path.name, options)
        original_error_count = syntax_error_count(tree)
        written: list[dict[str, object]] = []
        rejected: list[dict[str, str]] = []
        codes_dir = output_dir / "codes"
        mutations_dir = output_dir / "mutations"
        summaries_dir = output_dir / "summaries"

        for rank, match in enumerate(matches, 1):
            if len(written) >= options.max_outputs:
                break
            span_size = match.local_subtree.end_byte - match.local_subtree.start_byte
            root = tree.nodes[match.local_subtree.root_id]
            if root.raw_kind == "COMPILATION_UNIT":
                rejected.append({"bug_id": match.pattern.bug_id, "reason": "input match covers the whole program"})
                continue
            if span_size > options.max_host_bytes:
                rejected.append(
                    {"bug_id": match.pattern.bug_id, "reason": f"host subtree has {span_size} bytes"}
                )
                continue
            source_byte_count = max(len(source.encode("utf-8")), 1)
            if span_size / source_byte_count > options.max_host_fraction:
                rejected.append(
                    {
                        "bug_id": match.pattern.bug_id,
                        "reason": (
                            f"host subtree covers {span_size / source_byte_count:.1%} of the input; "
                            f"limit is {options.max_host_fraction:.1%}"
                        ),
                    }
                )
                continue
            try:
                historical_material = self.library.reference_material(match.pattern)
                historical = historical_material["historical_subtree"]
                host = tree.snippet(match.local_subtree.start_byte, match.local_subtree.end_byte)
                request = MutationRequest(
                    bug_id=match.pattern.bug_id,
                    host_subtree=host,
                    historical_subtree=historical,
                    language_feature=match.pattern.language_feature,
                    required_kinds=match.pattern.required_kinds,
                    phases=match.pattern.phases,
                    diagnostic=match.pattern.diagnostic,
                    match_score=match.final_score,
                    historical_summary=historical_material["Summary"],
                    historical_description=historical_material["Description"],
                    historical_source=historical_material["extracted_code"],
                    abstract_raw_kinds=match.pattern.abstract_raw_kinds,
                    abstract_canonical_kinds=match.pattern.abstract_canonical_kinds,
                    enclosing_context=enclosing_skeleton(tree, match.local_subtree.root_id),
                )
                if hasattr(mutator, "mutate_with_trace"):
                    mutation_output = mutator.mutate_with_trace(request)
                else:
                    replacement = mutator.mutate(request)
                    mutation_output = MutationOutput(
                        replacement=replacement,
                        prompt=json.dumps(request.to_dict(), ensure_ascii=False, indent=2),
                        raw_output=replacement,
                    )
                composed, normalized_replacement = compose_subtree(
                    source,
                    match.local_subtree.start_byte,
                    match.local_subtree.end_byte,
                    mutation_output.replacement,
                )
                mutated_tree = parse_java(composed, source_path.name)
                new_error_count = syntax_error_count(mutated_tree)
                if not options.allow_new_syntax_errors and new_error_count > original_error_count:
                    raise SnippetMutationError(
                        f"composition adds syntax errors ({original_error_count} -> {new_error_count})"
                    )
                replacement_start = match.local_subtree.start_byte
                replacement_region = (replacement_start, replacement_start + len(normalized_replacement.encode("utf-8")))
                mechanism = mechanism_retention(mutated_tree, replacement_region, match.pattern.required_kinds)
                if not mechanism["preserved"]:
                    raise SnippetMutationError(
                        "mutation loses the historical mechanism backbone "
                        f"({mechanism['backbone_check']}, retention={mechanism['retention']:.2f})"
                    )
                mutation_id = _result_name(source_path.stem, match.pattern.bug_id, rank)
                result_dir = mutations_dir / mutation_id
                result_dir.mkdir(parents=True, exist_ok=True)
                codes_dir.mkdir(parents=True, exist_ok=True)
                fragment_path = result_dir / "mutated_subtree.javafrag"
                original_path = result_dir / "Original.java"
                mutated_path = result_dir / "Mutated.java"
                candidate_path = codes_dir / f"{mutation_id}.java"
                prompt_path = result_dir / "prompt.txt"
                raw_output_path = result_dir / "raw_llm_output.txt"
                request_path = result_dir / "request.json"
                metadata_path = result_dir / "match.json"
                validation_path = result_dir / "parse_validation.json"
                provenance_path = result_dir / "provenance.json"
                fragment_path.write_text(normalized_replacement + "\n", encoding="utf-8")
                original_path.write_text(source, encoding="utf-8")
                mutated_path.write_text(composed, encoding="utf-8")
                candidate_path.write_text(composed, encoding="utf-8")
                prompt_path.write_text(mutation_output.prompt, encoding="utf-8")
                raw_output_path.write_text(mutation_output.raw_output, encoding="utf-8")
                request_path.write_text(
                    json.dumps(request.to_dict(), ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                metadata = match.to_dict(source_path)
                metadata["composition"] = {
                    "strategy": type(mutator).__name__,
                    "host_root_kind": root.raw_kind,
                    "host_start_byte": match.local_subtree.start_byte,
                    "host_end_byte": match.local_subtree.end_byte,
                    "host_byte_count": span_size,
                    "historical_subtree_byte_count": len(historical.encode("utf-8")),
                    "mutated_subtree_byte_count": len(normalized_replacement.encode("utf-8")),
                    "syntax_errors_before": original_error_count,
                    "syntax_errors_after": new_error_count,
                    "fragment_file": fragment_path.name,
                    "original_file": original_path.name,
                    "mutated_file": mutated_path.name,
                    "code_file": str(candidate_path.relative_to(output_dir)),
                    "prompt_file": prompt_path.name,
                    "raw_llm_output_file": raw_output_path.name,
                    "request_file": request_path.name,
                }
                metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                validation = {
                    "parse_ok": new_error_count == 0,
                    "syntax_errors_before": original_error_count,
                    "syntax_errors_after": new_error_count,
                    "new_syntax_errors": max(new_error_count - original_error_count, 0),
                    "mechanism_retention": mechanism,
                    "validation_scope": "parse_and_required_kind_backbone",
                    "symbol_resolution": "not_performed",
                }
                validation_path.write_text(
                    json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                provenance = {
                    "mutation_id": mutation_id,
                    "seed_path": str(source_path),
                    "historical_bug_id": match.pattern.bug_id,
                    "strategy": type(mutator).__name__,
                    "code_file": str(candidate_path),
                    "files": {
                        "original": original_path.name,
                        "mutated": mutated_path.name,
                        "fragment": fragment_path.name,
                        "prompt": prompt_path.name,
                        "raw_llm_output": raw_output_path.name,
                        "request": request_path.name,
                        "match": metadata_path.name,
                        "parse_validation": validation_path.name,
                    },
                }
                provenance_path.write_text(
                    json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                written.append(
                    {
                        "mutation_id": mutation_id,
                        "bug_id": match.pattern.bug_id,
                        "score": round(match.final_score, 6),
                        "mutation_dir": str(result_dir),
                        "code_file": str(candidate_path),
                    }
                )
            except (OSError, ValueError, RuntimeError, SnippetMutationError) as exc:
                rejected.append({"bug_id": match.pattern.bug_id, "reason": str(exc)})

        summaries_dir.mkdir(parents=True, exist_ok=True)
        summary: dict[str, object] = {
            "input_file": str(source_path),
            "matches_considered": len(matches),
            "written": written,
            "rejected": rejected,
        }
        (summaries_dir / f"{source_path.stem}.summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return summary


class FusionPipeline(MutationPipeline):
    """Add budgeted fusion outputs while reusing the original single-pattern path."""

    def __init__(self, library: HistoricalLibrary, usage: dict[str, int] | None = None) -> None:
        super().__init__(library)
        # Share the parent pipeline's run-level usage unless a fresh one is given.
        self.usage = usage if usage is not None else self._fusion_usage

    def mutate_file(self, source_path: Path, output_dir: Path, mutator: SnippetMutator, options: PipelineOptions) -> dict[str, object]:
        from .fusion import (
            EXCLUDED_ANCHORS, build_anchor_interaction_graph, cross_node_plans,
            plan_sort_key, same_node_plans, select_fusion_plans,
        )

        if options.fusion == "off":
            return super().mutate_file(source_path, output_dir, mutator, options)
        source = source_path.read_text(encoding="utf-8", errors="replace")
        tree = parse_java(source, source_path.name)
        # Every pattern may report several candidate anchors, so two patterns can
        # pair on one node (principle 9) or on interacting nodes (principle 10).
        groups = self.matcher.match_anchors(
            tree,
            candidates_per_pattern=options.anchor_candidates,
            prefilter_limit=options.prefilter_limit,
            top_k=options.top_k,
            min_score=options.min_score,
            query_language_feature=options.query_language_feature,
            query_phases=options.query_phases,
            query_diagnostic=options.query_diagnostic,
            max_span_bytes=options.max_host_bytes,
            max_span_fraction=options.max_host_fraction,
        )
        groups = {
            root: [m for m in group if (
                tree.nodes[m.local_subtree.root_id].raw_kind not in EXCLUDED_ANCHORS
                and m.local_subtree.end_byte > m.local_subtree.start_byte
                and m.final_score >= options.min_score
            )]
            for root, group in groups.items()
        }
        groups = {root: group for root, group in groups.items() if group}
        edges = build_anchor_interaction_graph(tree, groups, options.anchor_limit) if options.fusion in {"cross", "both"} else []
        plans = same_node_plans(tree, groups, options.fusion_pairs) if options.fusion in {"same", "both"} else []
        if options.fusion in {"cross", "both"}:
            plans.extend(cross_node_plans(tree, groups, edges, options.fusion_pairs, 2 * options.max_host_bytes))
        plans.sort(key=plan_sort_key)
        if options.fusion_selector == "matching":
            plans = select_fusion_plans(plans, usage=self.usage, per_file_cap=options.pattern_cap)
        quota = min(max(0, options.max_outputs), math.ceil(max(0, options.max_outputs) * options.fusion_budget))
        stats = {
            "planned": len(plans), "quota": quota, "selector": options.fusion_selector, "attempted": 0, "generated": 0,
            "mechanism_checked": 0, "both_mechanisms_preserved": 0, "accepted": 0,
            "rejected": 0, "by_operation": {}, "by_kind": {},
        }
        written, rejected = [], []
        for rank, plan in enumerate(plans, 1):
            if len(written) >= quota:
                break
            stats["attempted"] += 1
            bucket = stats["by_operation"].setdefault(plan.operation, {"attempted": 0, "accepted": 0, "rejected": 0})
            kind_bucket = stats["by_kind"].setdefault(plan.fusion_kind, {"attempted": 0, "accepted": 0, "rejected": 0})
            bucket["attempted"] += 1
            kind_bucket["attempted"] += 1
            validation = None
            try:
                request = self._fusion_request(tree, plan)
                if hasattr(mutator, "mutate_with_trace"):
                    output = mutator.mutate_with_trace(request)
                else:
                    raw = mutator.mutate(request)
                    one, two = parse_fusion_output(raw, plan.fusion_kind)
                    output = MutationOutput(one, json.dumps(request.to_dict(), ensure_ascii=False), raw, two)
                stats["generated"] += 1
                first = plan.match_1.local_subtree
                second = plan.match_2.local_subtree
                spans = [(first.start_byte, first.end_byte)]
                if plan.fusion_kind == "cross_node":
                    if not output.replacement_2:
                        raise SnippetMutationError("cross-node fusion requires both replacements")
                    spans.append((second.start_byte, second.end_byte))
                    composed, one, two = compose_two_spans(source, spans[0], spans[1], output.replacement, output.replacement_2)
                    fragments = [one, two]
                else:
                    composed, one = compose_subtree(source, *spans[0], output.replacement)
                    fragments = [one]
                if any(tree.snippet(*span).strip() == fragment.strip() for span, fragment in zip(spans, fragments)):
                    raise SnippetMutationError("fusion must change every selected anchor")
                regions = replacement_regions(spans, fragments)
                mutated_tree = parse_java(composed, source_path.name)
                validation = validate_fusion(tree, mutated_tree, plan, regions)
                stats["mechanism_checked"] += 1
                stats["both_mechanisms_preserved"] += int(validation["both_mechanisms_preserved"])
                if not validation["both_mechanisms_preserved"]:
                    raise SnippetMutationError("fusion loses a historical mechanism backbone")
                if not validation["roles_preserved"]:
                    raise SnippetMutationError("fusion changes a host subtree's syntactic role")
                if not validation["replacement_syntax_valid"]:
                    raise SnippetMutationError("fusion replacement contains an erroneous AST node")
                before, after = syntax_error_count(tree), syntax_error_count(mutated_tree)
                if not options.allow_new_syntax_errors and after > before:
                    raise SnippetMutationError(f"composition adds syntax errors ({before} -> {after})")
                record = self._write_fusion(source_path, output_dir, source, composed, plan, rank, request, output, fragments, regions, validation, type(mutator).__name__)
                written.append(record)
                stats["accepted"] += 1
                bucket["accepted"] += 1
                kind_bucket["accepted"] += 1
                self.usage[plan.match_1.pattern.bug_id] = self.usage.get(plan.match_1.pattern.bug_id, 0) + 1
                self.usage[plan.match_2.pattern.bug_id] = self.usage.get(plan.match_2.pattern.bug_id, 0) + 1
            except (OSError, ValueError, RuntimeError) as exc:
                stats["rejected"] += 1
                bucket["rejected"] += 1
                kind_bucket["rejected"] += 1
                rejection = {"bug_ids": [plan.match_1.pattern.bug_id, plan.match_2.pattern.bug_id], "operation": plan.operation, "reason": str(exc)}
                if validation is not None:
                    rejection["validation"] = validation
                rejected.append(rejection)
        # Failed fusion plans release their output slots back to single-pattern generation.
        single = super().mutate_file(source_path, output_dir, mutator, replace(options, fusion="off", max_outputs=max(0, options.max_outputs - len(written))))
        stats["mechanism_retention_rate"] = stats["both_mechanisms_preserved"] / stats["mechanism_checked"] if stats["mechanism_checked"] else None
        summary = {
            **single,
            "written": written + single["written"],
            "rejected": rejected + single["rejected"],
            "fusion_stats": stats,
            "anchor_interaction_graph": {"anchor_ids": sorted(groups, key=lambda n: (-groups[n][0].final_score, n))[:options.anchor_limit] if options.fusion in {"cross", "both"} else [], "edges": [e.to_dict() for e in edges]},
        }
        summary_path = output_dir / "summaries" / f"{source_path.stem}.summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return summary

    def _request_for_match(self, tree: JavaSyntaxTree, match: MatchResult) -> MutationRequest:
        material = self.library.reference_material(match.pattern)
        return MutationRequest(
            bug_id=match.pattern.bug_id,
            host_subtree=tree.snippet(match.local_subtree.start_byte, match.local_subtree.end_byte),
            historical_subtree=material["historical_subtree"],
            language_feature=match.pattern.language_feature,
            required_kinds=match.pattern.required_kinds,
            phases=match.pattern.phases,
            diagnostic=match.pattern.diagnostic,
            match_score=match.final_score,
            historical_summary=material["Summary"],
            historical_description=material["Description"],
            historical_source=material["extracted_code"],
            abstract_raw_kinds=match.pattern.abstract_raw_kinds,
            abstract_canonical_kinds=match.pattern.abstract_canonical_kinds,
            enclosing_context=enclosing_skeleton(tree, match.local_subtree.root_id),
        )

    def _fusion_request(self, tree: JavaSyntaxTree, plan: FusionPlan) -> FusionMutationRequest:
        from .fusion import OPERATION_INSTRUCTIONS

        first = self._request_for_match(tree, plan.match_1)
        second = self._request_for_match(tree, plan.match_2)
        payload_2 = second.to_dict()
        pattern_2 = {**payload_2["pattern"], "root_kind": plan.match_2.pattern.signature.get("root_kind", "")}
        bridge = ""
        if plan.edge:
            lca = tree.nodes[plan.edge.evidence["lca_id"]]
            bridge = tree.snippet(lca.start_byte, lca.end_byte)
        return FusionMutationRequest(
            base=first,
            host_subtree_2=second.host_subtree if plan.fusion_kind == "cross_node" else "",
            pattern_2=pattern_2,
            historical_subtree_2=second.historical_subtree,
            historical_report_2=payload_2["historical_report"],
            fusion_role=plan.operation,
            bridging_context=bridge,
            instruction=OPERATION_INSTRUCTIONS[plan.operation],
            bug_id_2=second.bug_id,
            match_score_2=second.match_score,
            fusion_kind=plan.fusion_kind,
            edge_evidence=plan.edge.to_dict() if plan.edge else {},
        )

    @staticmethod
    def _write_fusion(source_path, output_dir, source, composed, plan, rank, request, output, fragments, regions, validation, strategy):
        bugs = [plan.match_1.pattern.bug_id, plan.match_2.pattern.bug_id]
        mutation_id = _result_name(source_path.stem, "+".join(bugs) + "__" + plan.operation, rank)
        result_dir = output_dir / "mutations" / mutation_id
        result_dir.mkdir(parents=True, exist_ok=True)
        codes_dir = output_dir / "codes"
        codes_dir.mkdir(parents=True, exist_ok=True)
        candidate_path = codes_dir / f"{mutation_id}.java"
        candidate_path.write_text(composed, encoding="utf-8")
        files = {"original": "Original.java", "mutated": "Mutated.java", "prompt": "prompt.txt", "raw_llm_output": "raw_llm_output.txt", "request": "request.json", "match": "match.json", "parse_validation": "parse_validation.json", "fusion_plan": "fusion_plan.json"}
        for name, text in (("Original.java", source), ("Mutated.java", composed), ("prompt.txt", output.prompt), ("raw_llm_output.txt", output.raw_output)):
            (result_dir / name).write_text(text, encoding="utf-8")
        for index, fragment in enumerate(fragments):
            name = "mutated_subtree.javafrag" if len(fragments) == 1 else f"mutated_subtree_{index + 1}.javafrag"
            (result_dir / name).write_text(fragment + "\n", encoding="utf-8")
            files[f"fragment_{index + 1}"] = name
        evidence = plan.to_dict()
        evidence["replacement_spans"] = regions
        evidence["composed_template"] = fragments[0] if len(fragments) == 1 else {"mutated_subtree_1": fragments[0], "mutated_subtree_2": fragments[1]}
        provenance = {"mutation_id": mutation_id, "seed_path": str(source_path), "historical_bug_ids": bugs, "trace_to_bug_1": bugs[0], "trace_to_bug_2": bugs[1], "fusion_kind": plan.fusion_kind, "operation": plan.operation, "strategy": strategy, "code_file": str(candidate_path), "files": files}
        payloads = {
            "request.json": request.to_dict(),
            "fusion_plan.json": evidence,
            "parse_validation.json": validation,
            "match.json": {"pattern_1": plan.match_1.to_dict(source_path), "pattern_2": plan.match_2.to_dict(source_path), "fusion_score": plan.score},
            "provenance.json": provenance,
        }
        for name, payload in payloads.items():
            (result_dir / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {"mutation_id": mutation_id, "bug_ids": bugs, "score": plan.score, "fusion_kind": plan.fusion_kind, "operation": plan.operation, "mutation_dir": str(result_dir), "code_file": str(candidate_path)}


def replacement_regions(spans: list[tuple[int, int]], fragments: list[str]) -> list[tuple[int, int]]:
    regions = [(0, 0)] * len(spans)
    shift = 0
    for index in sorted(range(len(spans)), key=lambda i: spans[i][0]):
        start, end = spans[index]
        length = len(fragments[index].encode("utf-8"))
        regions[index] = (start + shift, start + shift + length)
        shift += length - (end - start)
    return regions


# Kinds that occur in almost every Java fragment; they never carry a historical
# anomaly mechanism by themselves, so retention checks must look past them.
TRIVIAL_KINDS = {
    "IDENTIFIER", "MODIFIERS", "BLOCK", "EXPRESSION_STATEMENT", "PARENTHESIZED",
    "METHOD", "CLASS", "METHOD_INVOCATION", "RETURN", "EMPTY_STATEMENT",
    "COMPILATION_UNIT", "IMPORT", "PACKAGE",
}


def mechanism_retention(
    mutated: JavaSyntaxTree,
    region: tuple[int, int],
    required_kinds: Iterable[str],
    other_required: Iterable[str] = (),
) -> dict:
    """Kind-level mechanism check inside one replacement region of the composed file.

    A preserved mechanism means the fragment keeps >=60% of the pattern's required
    kinds AND at least one backbone kind that is distinctive of this pattern
    (required, non-trivial, and not shared with a co-fused pattern).
    """
    start, end = region
    fragment = mutated.snippet(start, end)
    start += len(fragment[:len(fragment) - len(fragment.lstrip())].encode("utf-8"))
    end -= len(fragment[len(fragment.rstrip()):].encode("utf-8"))
    kinds = {n.raw_kind for n in mutated.nodes.values() if n.start_byte >= start and n.end_byte <= end}
    required = set(required_kinds)
    backbone = required - TRIVIAL_KINDS
    exclusive = backbone - set(other_required)
    fraction = len(required & kinds) / len(required) if required else 0.0
    if exclusive:
        backbone_check, present = "exclusive_backbone", bool(exclusive & kinds)
    elif backbone:
        backbone_check, present = "shared_backbone", bool(backbone & kinds)
    else:
        backbone_check, present = "total_retention", fraction >= 0.9
    return {
        "region": [start, end],
        "required_kinds": sorted(required),
        "missing_kinds": sorted(required - kinds),
        "backbone_kinds": sorted(backbone),
        "exclusive_kinds": sorted(exclusive),
        "backbone_check": backbone_check,
        "backbone_present": present,
        "retention": round(fraction, 6),
        "preserved": fraction >= 0.6 and present,
    }


def validate_fusion(original: JavaSyntaxTree, mutated: JavaSyntaxTree, plan: FusionPlan, regions: list[tuple[int, int]]) -> dict:
    from .fusion import syntactic_role

    retention, roles = [], []
    no_errors = True
    for index, match in enumerate((plan.match_1, plan.match_2)):
        region = regions[index if plan.fusion_kind == "cross_node" else 0]
        other = plan.match_2 if index == 0 else plan.match_1
        item = mechanism_retention(mutated, region, match.pattern.required_kinds, other.pattern.required_kinds)
        item["bug_id"] = match.pattern.bug_id
        retention.append(item)
        start, end = item["region"]
        nodes = [n for n in mutated.nodes.values() if n.start_byte >= start and n.end_byte <= end]
        kinds = {n.raw_kind for n in nodes}
        roots = [n for n in nodes if n.start_byte == start and n.end_byte == end]
        expected_role = syntactic_role(original.nodes[match.local_subtree.root_id].raw_kind)
        roles.append(any(syntactic_role(n.raw_kind) == expected_role for n in roots))
        no_errors = no_errors and not bool(kinds & {"ERROR", "ERRONEOUS"})
    before, after = syntax_error_count(original), syntax_error_count(mutated)
    return {
        "parse_ok": after == 0,
        "syntax_errors_before": before,
        "syntax_errors_after": after,
        "new_syntax_errors": max(0, after - before),
        "mechanism_retention": retention,
        "both_mechanisms_preserved": all(item["preserved"] for item in retention),
        "roles_preserved": all(roles),
        "replacement_syntax_valid": no_errors,
        "validation_scope": "parse_and_required_kind_backbones",
        "symbol_resolution": "not_performed",
    }


def enclosing_skeleton(tree: JavaSyntaxTree, root_id: int, max_bytes: int = 1000) -> str:
    """Signature-only view of the anchor's enclosing type and method (no bodies).

    Gives the mutator model the declared methods and fields around the anchor so
    adapted identifiers can actually resolve, without leaking the whole file.
    """
    lines: list[str] = []
    ancestors = tree.ancestors(root_id)
    type_id = next(
        (n for n in reversed(ancestors) if tree.nodes[n].raw_kind in {"CLASS", "INTERFACE", "ENUM", "RECORD"}),
        None,
    )
    method_id = next((n for n in reversed(ancestors) if tree.nodes[n].raw_kind == "METHOD"), None)
    if type_id is not None:
        type_node = tree.nodes[type_id]
        type_text = tree.snippet(type_node.start_byte, type_node.end_byte)
        lines.append(type_text.split("{", 1)[0].strip() + " { ... }")
        for child_id in type_node.child_ids:
            child = tree.nodes[child_id]
            if child.raw_kind == "METHOD":
                child_text = tree.snippet(child.start_byte, child.end_byte)
                signature = child_text.split("{", 1)[0].strip()
                lines.append("    " + signature if signature.endswith(";") else "    " + signature + " { ... }")
            elif child.raw_kind == "VARIABLE":
                child_text = tree.snippet(child.start_byte, child.end_byte)
                lines.append("    " + child_text.split("=", 1)[0].strip() + ";")
    if method_id is not None:
        method_text = tree.snippet(tree.nodes[method_id].start_byte, tree.nodes[method_id].end_byte)
        signature = method_text.split("{", 1)[0].strip()
        lines.append(signature if signature.endswith((";", "}")) else signature + " { ... }")
    text = "\n".join(lines)
    if len(text.encode("utf-8")) > max_bytes:
        text = text[:max_bytes] + " ..."
    return text


def aggregate_fusion_stats(summaries: Iterable[dict]) -> dict:
    totals = {key: 0 for key in ("planned", "quota", "attempted", "generated", "mechanism_checked", "both_mechanisms_preserved", "accepted", "rejected")}
    totals.update({"by_operation": {}, "by_kind": {}})
    for summary in summaries:
        stats = summary.get("fusion_stats", {})
        for key in ("planned", "quota", "attempted", "generated", "mechanism_checked", "both_mechanisms_preserved", "accepted", "rejected"):
            totals[key] += stats.get(key, 0)
        for grouping in ("by_operation", "by_kind"):
            for name, counts in stats.get(grouping, {}).items():
                bucket = totals[grouping].setdefault(name, {"attempted": 0, "accepted": 0, "rejected": 0})
                for key in bucket:
                    bucket[key] += counts[key]
    totals["mechanism_retention_rate"] = totals["both_mechanisms_preserved"] / totals["mechanism_checked"] if totals["mechanism_checked"] else None
    return totals


def syntax_error_count(tree: JavaSyntaxTree) -> int:
    erroneous_nodes = sum(node.raw_kind in {"ERROR", "ERRONEOUS"} for node in tree.nodes.values())
    return max(tree.syntax_error_count, erroneous_nodes)


def java_inputs(path: Path, glob: str = "*.java") -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    yield from sorted(candidate for candidate in path.rglob(glob) if candidate.is_file())


def _result_name(source_stem: str, bug_id: str, rank: int) -> str:
    safe_source = re.sub(r"[^A-Za-z0-9_.-]+", "_", source_stem)
    safe_bug = re.sub(r"[^A-Za-z0-9_.-]+", "_", bug_id)
    return f"{safe_source}__{safe_bug}__r{rank:03d}"


def candidate_filename(source_path: Path, source: str) -> str:
    match = re.search(
        r"\bpublic\s+(?:(?:sealed|non-sealed|abstract|final|strictfp)\s+)*(?:class|interface|enum|record)\s+([A-Za-z_$][A-Za-z0-9_$]*)",
        source,
    )
    return f"{match.group(1)}.java" if match else source_path.name

