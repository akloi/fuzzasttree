"""Command-line interface for historical subtree matching and mutation."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Sequence

from .evaluation import evaluate_manifest
from .library import DEFAULT_HISTORICAL_SOURCE_DIR, DEFAULT_LIBRARY_DIR, HistoricalLibrary
from .local_model_adapter import LocalModelSnippetMutator
from .mutator import CommandSnippetMutator, HistoricalSnippetMutator, OpenAICompatibleSnippetMutator
from .pipeline import MutationPipeline, PipelineOptions, aggregate_fusion_stats, java_inputs


DEFAULT_RUNS_DIR = Path(__file__).resolve().parent / "runs"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Match Java inputs against historical anomaly AST subtrees and mutate only local snippets."
    )
    parser.add_argument("--library-dir", type=Path, default=DEFAULT_LIBRARY_DIR)
    parser.add_argument("--historical-source-dir", type=Path, default=DEFAULT_HISTORICAL_SOURCE_DIR)
    parser.add_argument("--full-only", action="store_true", help="Exclude candidate_partial references.")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("stats", help="Show indexed historical library statistics.")

    match = commands.add_parser("match", help="Rank historical patterns for one Java source file.")
    match.add_argument("input", type=Path)
    match.add_argument("--output", type=Path)
    _add_match_options(match)

    evaluate = commands.add_parser("evaluate", help="Evaluate retrieval from a JSON/JSONL ground-truth manifest.")
    evaluate.add_argument("manifest", type=Path)
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument("--prefilter-limit", type=int, default=200)
    evaluate.add_argument("--top-k", type=int, default=10)

    mutate = commands.add_parser("mutate", help="Match, mutate local subtrees, and compose candidate Java files.")
    mutate.add_argument("input", type=Path, help="One .java file or a directory tree.")
    mutate.add_argument("--glob", default="*.java")
    mutate.add_argument("--output-dir", type=Path)
    mutate.add_argument("--strategy", choices=("historical", "command", "openai", "local"), default="historical")
    mutate.add_argument("--mutator-command", default="")
    mutate.add_argument("--mutator-timeout", type=float, default=120.0)
    mutate.add_argument("--api-base", default="https://api.deepseek.com/chat/completions")
    mutate.add_argument("--model", default="deepseek-chat")
    mutate.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    mutate.add_argument("--no-api-key", action="store_true", help="Do not send Authorization to a local endpoint.")
    mutate.add_argument("--temperature", type=float)
    mutate.add_argument("--adapter-path")
    mutate.add_argument("--max-new-tokens", type=int, default=512)
    mutate.add_argument("--gpu", type=int, nargs="+", default=[0])
    mutate.add_argument("--max-outputs", type=int, default=5)
    mutate.add_argument("--max-host-bytes", type=int, default=4000)
    mutate.add_argument("--max-host-fraction", type=float, default=0.75)
    mutate.add_argument("--allow-new-syntax-errors", action="store_true")
    mutate.add_argument("--fusion", choices=("off", "same", "cross", "both"), default="off")
    mutate.add_argument("--fusion-pairs", type=_positive_int, default=4, help="Keep this many plans per anchor or interaction edge.")
    mutate.add_argument("--anchor-limit", type=_positive_int, default=6, help="Maximum number of cross-node graph vertices.")
    mutate.add_argument("--fusion-budget", type=_fraction, default=0.4, help="Fusion share of max-outputs, rounded up; failed slots return to single-pattern generation.")
    mutate.add_argument("--anchor-candidates", type=_positive_int, default=3, help="Candidate anchors reported per pattern for fusion grouping.")
    _add_match_options(mutate)
    return parser


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _fraction(value: str) -> float:
    number = float(value)
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return number


def _add_match_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prefilter-limit", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--language-feature", default="")
    parser.add_argument("--phase", action="append", default=[])
    parser.add_argument("--diagnostic", default="")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    statuses = ("candidate",) if args.full_only else ("candidate", "candidate_partial")
    library = HistoricalLibrary(args.library_dir, args.historical_source_dir, statuses=statuses)
    if args.command == "stats":
        print(json.dumps(library.stats(), ensure_ascii=False, indent=2))
        return 0
    if args.command == "evaluate":
        report = evaluate_manifest(
            library,
            args.manifest,
            prefilter_limit=args.prefilter_limit,
            top_k=args.top_k,
        )
        rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        else:
            sys.stdout.write(rendered)
        return 0

    options = PipelineOptions(
        prefilter_limit=args.prefilter_limit,
        top_k=args.top_k,
        min_score=args.min_score,
        query_language_feature=args.language_feature,
        query_phases=tuple(args.phase),
        query_diagnostic=args.diagnostic,
        max_outputs=getattr(args, "max_outputs", 5),
        max_host_bytes=getattr(args, "max_host_bytes", 4000),
        max_host_fraction=getattr(args, "max_host_fraction", 0.75),
        allow_new_syntax_errors=getattr(args, "allow_new_syntax_errors", False),
        fusion=getattr(args, "fusion", "off"),
        fusion_pairs=getattr(args, "fusion_pairs", 4),
        anchor_limit=getattr(args, "anchor_limit", 6),
        fusion_budget=getattr(args, "fusion_budget", 0.4),
        anchor_candidates=getattr(args, "anchor_candidates", 3),
    )
    pipeline = MutationPipeline(library)
    if args.command == "match":
        if not args.input.is_file():
            raise SystemExit(f"input file does not exist: {args.input}")
        source = args.input.read_text(encoding="utf-8", errors="replace")
        _tree, matches = pipeline.match_source(source, args.input.name, options)
        payload = {
            "input_file": str(args.input),
            "library_patterns": len(library.patterns),
            "matches": [match.to_dict(args.input) for match in matches],
        }
        rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        else:
            sys.stdout.write(rendered)
        return 0


    if args.strategy == "command":
        if not args.mutator_command:
            raise SystemExit("--mutator-command is required for --strategy command")
        mutator = CommandSnippetMutator(args.mutator_command, args.mutator_timeout)
    elif args.strategy == "openai":
        mutator = OpenAICompatibleSnippetMutator(
            api_base=args.api_base,
            model=args.model,
            api_key_env=None if args.no_api_key else args.api_key_env,
            timeout_seconds=args.mutator_timeout,
            temperature=args.temperature if args.temperature is not None else 0.0,
        )
    elif args.strategy == "local":
        mutator = LocalModelSnippetMutator(
            model=args.model,
            adapter_path=args.adapter_path,
            temperature=args.temperature if args.temperature is not None else 0.2,
            max_new_tokens=args.max_new_tokens,
            gpu_devices=args.gpu,
        )
    else:
        mutator = HistoricalSnippetMutator()
    output_dir = args.output_dir or DEFAULT_RUNS_DIR / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    inputs = list(java_inputs(args.input, args.glob))
    if not inputs:
        raise SystemExit(f"no Java inputs found under {args.input}")
    summaries = []
    for index, path in enumerate(inputs, 1):
        print(f"[astsubtree] Input {index}/{len(inputs)}: {path}", file=sys.stderr, flush=True)
        summary = pipeline.mutate_file(path, output_dir, mutator, options)
        summaries.append(summary)
        print(
            f"[astsubtree] Completed {index}/{len(inputs)}: "
            f"written={len(summary['written'])}, rejected={len(summary['rejected'])}",
            file=sys.stderr,
            flush=True,
        )
    run_summary = {
        "library": library.stats(),
        "input_count": len(inputs),
        "output_dir": str(output_dir),
        "written_count": sum(len(item["written"]) for item in summaries),
        "inputs": summaries,
    }
    if options.fusion != "off":
        run_summary["fusion_stats"] = aggregate_fusion_stats(summaries)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = {
        "output_dir": str(output_dir),
        "mutation_count": run_summary["written_count"],
        "mutations": [record for summary in summaries for record in summary["written"]],
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

