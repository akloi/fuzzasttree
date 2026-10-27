"""End-to-end matching, snippet mutation, validation, and output writing."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Iterable

from .library import HistoricalLibrary
from .matcher import SubtreeMatcher
from .models import MatchResult, MutationRequest
from .mutator import MutationOutput, SnippetMutationError, SnippetMutator, compose_subtree
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


class MutationPipeline:
    def __init__(self, library: HistoricalLibrary) -> None:
        self.library = library
        self.matcher = SubtreeMatcher(library)

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
        )
        return tree, matches

    def mutate_file(
        self,
        source_path: Path,
        output_dir: Path,
        mutator: SnippetMutator,
        options: PipelineOptions,
    ) -> dict[str, object]:
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

