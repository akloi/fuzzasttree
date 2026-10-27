#!/usr/bin/env python3
"""Adapt the repository's local Transformers model to the subtree JSON protocol."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import html
import json
import re
import sys
from typing import Any, Callable, TextIO

from .local_generator import LocalGenerator
from .mutator import MutationOutput


def build_subtree_prompt(request: dict[str, Any]) -> str:
    """Build a prompt from the protocol fields without accepting a full host file."""
    pattern = request.get("pattern") if isinstance(request.get("pattern"), dict) else {}
    report = request.get("historical_report") if isinstance(request.get("historical_report"), dict) else {}
    payload = {
        "bug_id": str(request.get("bug_id") or ""),
        "host_subtree": str(request.get("host_subtree") or ""),
        "historical_subtree": str(request.get("historical_subtree") or ""),
        "pattern": {
            "L": str(pattern.get("L") or ""),
            "A_raw": [str(item) for item in pattern.get("A_raw") or ()],
            "A_canonical": [str(item) for item in pattern.get("A_canonical") or ()],
            "A_native": [str(item) for item in pattern.get("A_native") or ()],
            "P": [str(item) for item in pattern.get("P") or ()],
            "D": str(pattern.get("D") or ""),
        },
        "historical_report": {
            "Summary": str(report.get("Summary") or ""),
            "Description": str(report.get("Description") or ""),
            "extracted_code": str(report.get("extracted_code") or ""),
        },
        "match_score": request.get("match_score"),
    }
    if not payload["host_subtree"].strip():
        raise ValueError("request has no host_subtree")
    if not payload["historical_subtree"].strip():
        raise ValueError("request has no historical_subtree")

    return (
        "You mutate exactly one Java AST subtree for javac compiler fuzzing.\n"
        "Use the historical bug report, complete historical reproducer, anomaly subtree, and L/A/P/D "
        "pattern to transfer the anomaly structure into host_subtree. Adapt identifiers, types, and "
        "syntax to the host context. Preserve the syntactic role of host_subtree.\n"
        "Return exactly one JSON object with one key, mutated_subtree. Its value must contain only the "
        "replacement Java subtree. Do not return imports, a surrounding class, a complete Java file, "
        "Markdown, or an explanation.\n\n"
        "Mutation request:\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def parse_model_output(output: str | None) -> str:
    """Extract a subtree from JSON, code tags, a Markdown fence, or plain code."""
    text = (output or "").strip()
    if not text:
        raise ValueError("model returned empty output")

    tagged = re.search(r"<code>(.*?)</code>", text, flags=re.DOTALL | re.IGNORECASE)
    if tagged:
        text = html.unescape(tagged.group(1).strip())

    fenced = re.fullmatch(r"```(?:json|java)?\s*\n?(.*?)\n?```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()

    payload = _find_json_object(text)
    if payload is not None:
        snippet = payload.get("mutated_subtree")
        if not isinstance(snippet, str) or not snippet.strip():
            raise ValueError("model JSON has no non-empty mutated_subtree")
        return snippet.strip()
    return text


def _find_json_object(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return parsed

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            candidate, _end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and "mutated_subtree" in candidate:
            return candidate
    return None


def _load_generator(args: argparse.Namespace):
    return load_local_generator(
        model=args.model,
        adapter_path=args.adapter_path,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        gpu_devices=args.gpu,
    )


def load_local_generator(
    *,
    model: str,
    adapter_path: str | None,
    temperature: float,
    max_new_tokens: int,
    gpu_devices: list[int],
):
    return LocalGenerator(
        model_name=model,
        adapter_path=adapter_path,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        batch_size=1,
        gpu_devices=gpu_devices,
    )


def _mutate_with_trace(generator: Any, request: dict[str, Any]) -> MutationOutput:
    prompt = build_subtree_prompt(request)
    tokenizer = getattr(generator, "tokenizer", None)
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    outputs = generator.generate_batch([prompt])
    if not outputs or outputs[0] is None:
        raise RuntimeError("local model did not produce an output")
    raw_output = str(outputs[0])
    return MutationOutput(
        replacement=parse_model_output(raw_output),
        prompt=prompt,
        raw_output=raw_output,
    )


def _mutate(generator: Any, request: dict[str, Any]) -> str:
    return _mutate_with_trace(generator, request).replacement


class LocalModelSnippetMutator:
    """Reuse one local Transformers model across a complete mutation run."""

    def __init__(
        self,
        *,
        model: str = "",
        adapter_path: str | None = None,
        temperature: float = 0.2,
        max_new_tokens: int = 512,
        gpu_devices: list[int] | None = None,
        generator: Any | None = None,
        log_stream: TextIO | None = None,
    ) -> None:
        self.log_stream = log_stream or sys.stderr
        if generator is not None:
            self.generator = generator
            return
        if not model:
            raise ValueError("local model path must not be empty")
        with redirect_stdout(self.log_stream):
            self.generator = load_local_generator(
                model=model,
                adapter_path=adapter_path,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                gpu_devices=gpu_devices or [0],
            )
        print("[astsubtree] Local model is ready.", file=self.log_stream, flush=True)

    def mutate(self, request: Any) -> str:
        return self.mutate_with_trace(request).replacement

    def mutate_with_trace(self, request: Any) -> MutationOutput:
        payload = request.to_dict() if hasattr(request, "to_dict") else request
        if not isinstance(payload, dict):
            raise ValueError("mutation request must be an object")
        with redirect_stdout(self.log_stream):
            return _mutate_with_trace(self.generator, payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local-model adapter for astsubtree mutation requests.")
    parser.add_argument("--model", required=True, help="Local Hugging Face model path or model identifier.")
    parser.add_argument("--adapter-path", help="Optional PEFT/LoRA adapter directory.")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--gpu", type=int, nargs="+", default=[0], help="CUDA device indices visible to torch.")
    return parser


def main(
    argv: list[str] | None = None,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    generator_factory: Callable[[argparse.Namespace], Any] = _load_generator,
) -> int:
    args = build_parser().parse_args(argv)
    input_stream = stdin or sys.stdin
    output_stream = stdout or sys.stdout
    error_stream = stderr or sys.stderr
    try:
        request = json.load(input_stream)
        if not isinstance(request, dict):
            raise ValueError("stdin JSON must be an object")

        # Model implementations may log with print; stdout is the JSON protocol channel.
        with redirect_stdout(error_stream):
            generator = generator_factory(args)
            mutated_subtree = _mutate(generator, request)
        json.dump({"mutated_subtree": mutated_subtree}, output_stream, ensure_ascii=False)
        output_stream.write("\n")
        output_stream.flush()
        return 0
    except Exception as exc:
        error_stream.write(f"[astsubtree local model adapter] {type(exc).__name__}: {exc}\n")
        error_stream.flush()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
