#!/usr/bin/env python3
"""Adapt a local Transformers model to the astshannon snippet JSON protocol."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import html
import json
import re
import sys
from typing import Any, Callable, TextIO

from .local_generator import LocalGenerator
from .mutator import MutationOutput, parse_fusion_output


def build_subtree_prompt(request: dict[str, Any]) -> str:
    """Build a prompt from the protocol fields without accepting a full host file."""
    pattern = request.get("pattern") if isinstance(request.get("pattern"), dict) else {}
    report = request.get("historical_report") if isinstance(request.get("historical_report"), dict) else {}
    payload = {
        "bug_id": str(request.get("bug_id") or ""),
        "host_subtree": str(request.get("host_subtree") or ""),
        "historical_subtree": str(request.get("historical_subtree") or ""),
        "pattern": {
            "pattern_type": str(pattern.get("pattern_type") or ""),
            "nodes": [dict(item) for item in pattern.get("nodes") or ()],
            "edges": [dict(item) for item in pattern.get("edges") or ()],
            "die_guardrails": dict(pattern.get("die_guardrails") or {}),
            "semantic_features": [dict(item) for item in pattern.get("semantic_features") or ()],
        },
        "historical_report": {
            "Summary": str(report.get("Summary") or ""),
            "Description": str(report.get("Description") or ""),
            "extracted_code": str(report.get("extracted_code") or ""),
        },
        "match_score": request.get("match_score"),
        "host_region_evidence": [dict(item) for item in request.get("host_region_evidence") or ()],
        "enclosing_context": str(request.get("enclosing_context") or ""),
    }
    if not payload["host_subtree"].strip():
        raise ValueError("request has no host_subtree")
    if not payload["historical_subtree"].strip():
        raise ValueError("request has no historical_subtree")

    return (
        "You mutate exactly one Java AST subtree for javac compiler fuzzing.\n"
        "The pattern is a role-labeled graph distilled from a historical javac bug: nodes carry javac kinds and "
        "roles (CORE_TRIGGER = the anomaly sink, CONTEXT = required enclosing structure, DEPENDENT = supporting "
        "nodes), edges are typed semantic relations, and semantic_features are information-weighted signals. "
        "Transfer the anomaly into host_subtree: rebuild the CORE_TRIGGER node structure, keep the typed-edge "
        "synergy among the pattern's nodes, and obey die_guardrails (type_constraints, structural_context). "
        "Adapt identifiers and types to enclosing_context (nearby declared signatures); host_region_evidence "
        "shows which typed relations the matched host region already exhibits. Small supporting declarations "
        "the pattern needs (local sealed interfaces, records, helper variables) may be embedded inside the "
        "replacement if it stays syntactically valid at the anchor position. Preserve the syntactic role of "
        "host_subtree.\n"
        "Hard requirements: (1) the replacement MUST differ from host_subtree -- returning it unchanged is a "
        "failure; (2) reconstruct the CORE_TRIGGER as real code structure -- never quote or mention the "
        "historical snippet inside a string literal, comment, or identifier name; (3) renaming or reordering "
        "tokens of host_subtree alone is not a mutation -- the anomaly-bearing structure itself must appear.\n"
        "Return exactly one JSON object with one key, mutated_subtree. Its value must contain only the "
        "replacement Java subtree. Do not return imports, a surrounding class, a complete Java file, "
        "Markdown, or an explanation.\n\n"
        "Mutation request:\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def build_fusion_prompt(request: dict[str, Any]) -> str:
    """The fusion payload contains two historical patterns and only local host context."""
    fields = (
        "protocol_version", "fusion_kind", "fusion_role", "instruction", "bug_id_1", "bug_id_2",
        "host_subtree_1", "host_subtree_2", "pattern_1", "pattern_2",
        "historical_subtree_1", "historical_subtree_2", "historical_report_1", "historical_report_2",
        "match_score_1", "match_score_2", "bridging_context", "edge_evidence", "enclosing_context",
    )
    payload = {key: request[key] for key in fields if key in request}
    if not payload.get("host_subtree_1") or not payload.get("historical_subtree_1") or not payload.get("historical_subtree_2"):
        raise ValueError("fusion request requires the host and both historical snippets")
    cross = request.get("fusion_kind") == "cross_node"
    if cross and not payload.get("host_subtree_2"):
        raise ValueError("cross-node fusion requires host_subtree_2")
    output = '{"mutated_subtree_1": "...", "mutated_subtree_2": "..."}' if cross else '{"mutated_subtree": "..."}'
    return (
        "You fuse TWO historical javac anomaly patterns into one host context.\n"
        "Each pattern is a role-labeled graph (CORE_TRIGGER nodes, typed edges, die_guardrails). Keep BOTH "
        "anomaly mechanisms recognizable. Preserve each host subtree's syntactic role. Adapt identifiers and "
        "types to the supplied hosts, the enclosing_context signatures, and the bridging context. Follow "
        "fusion_role and its directed instruction; preserve the interaction edge — if edge_evidence carries a "
        "host_relation, keep that relation between the two anchors intact.\n"
        "Hard requirements: every replacement MUST differ from its host_subtree, and each CORE_TRIGGER must "
        "be reconstructed as real code structure -- never quoted inside a string literal, comment, or "
        "identifier name. Both mechanisms must remain structurally present, not merely mentioned.\n"
        f"Return exactly one JSON object shaped as {output}. "
        "Each value contains only replacement Java subtree code. Return no imports, complete file, "
        "Markdown, or explanation.\n\nFusion request:\n"
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
        max_input_tokens=args.max_input_tokens,
        gpu_devices=args.gpu,
    )


def load_local_generator(
    *,
    model: str,
    adapter_path: str | None,
    temperature: float,
    max_new_tokens: int,
    max_input_tokens: int = 16384,
    gpu_devices: list[int],
):
    return LocalGenerator(
        model_name=model,
        adapter_path=adapter_path,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        max_input_tokens=max_input_tokens,
        batch_size=1,
        gpu_devices=gpu_devices,
    )


def _mutate_with_trace(generator: Any, request: dict[str, Any]) -> MutationOutput:
    fusion_kind = request.get("fusion_kind")
    prompt = build_fusion_prompt(request) if fusion_kind else build_subtree_prompt(request)
    tokenizer = getattr(generator, "tokenizer", None)
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        # Qwen3 系默认开思考模式，512 token 预算会被推理文本吃光，变异负载永远轮不到。
        # enable_thinking=False 让模型直接作答；Qwen2.5 等旧模板不引用该变量，jinja 静默忽略。
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    outputs = generator.generate_batch([prompt])
    if not outputs or outputs[0] is None:
        raise RuntimeError("local model did not produce an output")
    raw_output = str(outputs[0])
    replacement, replacement_2 = parse_fusion_output(raw_output, fusion_kind) if fusion_kind else (parse_model_output(raw_output), None)
    return MutationOutput(
        replacement=replacement,
        prompt=prompt,
        raw_output=raw_output,
        replacement_2=replacement_2,
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
        max_input_tokens: int = 16384,
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
                max_input_tokens=max_input_tokens,
                gpu_devices=gpu_devices or [0],
            )
        print("[astshannon] Local model is ready.", file=self.log_stream, flush=True)

    def mutate(self, request: Any) -> str:
        return self.mutate_with_trace(request).replacement

    def mutate_with_trace(self, request: Any) -> MutationOutput:
        payload = request.to_dict() if hasattr(request, "to_dict") else request
        if not isinstance(payload, dict):
            raise ValueError("mutation request must be an object")
        with redirect_stdout(self.log_stream):
            return _mutate_with_trace(self.generator, payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local-model adapter for astshannon mutation requests.")
    parser.add_argument("--model", required=True, help="Local Hugging Face model path or model identifier.")
    parser.add_argument("--adapter-path", help="Optional PEFT/LoRA adapter directory.")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-input-tokens", type=int, default=16384, help="Hard cap on prompt tokens; longer prompts are truncated with a warning.")
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
            result = _mutate_with_trace(generator, request)
        payload = {"mutated_subtree": result.replacement}
        if result.replacement_2 is not None:
            payload = {"mutated_subtree_1": result.replacement, "mutated_subtree_2": result.replacement_2}
        json.dump(payload, output_stream, ensure_ascii=False)
        output_stream.write("\n")
        output_stream.flush()
        return 0
    except Exception as exc:
        error_stream.write(f"[astshannon local model adapter] {type(exc).__name__}: {exc}\n")
        error_stream.flush()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
