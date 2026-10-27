"""Subtree mutation providers and byte-accurate source composition."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import html
import os
import re
import shlex
import subprocess
import textwrap
import urllib.error
import urllib.request

from .models import FusionMutationRequest, MutationRequest


class SnippetMutationError(RuntimeError):
    pass


@dataclass(slots=True)
class MutationOutput:
    """Parsed replacement plus the exact model input and unparsed response."""

    replacement: str
    prompt: str
    raw_output: str
    replacement_2: str | None = None


class SnippetMutator(ABC):
    @abstractmethod
    def mutate(self, request: MutationRequest) -> str:
        """Return only the replacement subtree source."""

    def mutate_with_trace(self, request: MutationRequest) -> MutationOutput:
        replacement = self.mutate(request)
        prompt = json.dumps(request.to_dict(), ensure_ascii=False, indent=2)
        return MutationOutput(replacement=replacement, prompt=prompt, raw_output=replacement)


class HistoricalSnippetMutator(SnippetMutator):
    """A deterministic transplant baseline using the historical anomaly snippet."""

    def mutate(self, request: MutationRequest) -> str:
        if isinstance(request, FusionMutationRequest):
            return self.mutate_with_trace(request).replacement
        snippet = request.historical_subtree.strip()
        if not snippet:
            raise SnippetMutationError("historical subtree is empty")
        return snippet

    def mutate_with_trace(self, request: MutationRequest | FusionMutationRequest) -> MutationOutput:
        if not isinstance(request, FusionMutationRequest):
            return super().mutate_with_trace(request)
        first, second = historical_fusion(request)
        payload = {"mutated_subtree": first} if second is None else {"mutated_subtree_1": first, "mutated_subtree_2": second}
        return MutationOutput(first, json.dumps(request.to_dict(), ensure_ascii=False, indent=2), json.dumps(payload, ensure_ascii=False), second)


class CommandSnippetMutator(SnippetMutator):
    """Run an external model adapter over a JSON stdin/stdout protocol."""

    def __init__(self, command: str, timeout_seconds: float = 120.0) -> None:
        self.argv = shlex.split(command)
        if not self.argv:
            raise ValueError("mutator command must not be empty")
        self.timeout_seconds = timeout_seconds

    def mutate(self, request: MutationRequest) -> str:
        return self.mutate_with_trace(request).replacement

    def mutate_with_trace(self, request: MutationRequest) -> MutationOutput:
        prompt = json.dumps(request.to_dict(), ensure_ascii=False)
        try:
            result = subprocess.run(
                self.argv,
                input=prompt,
                text=True,
                encoding="utf-8",
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise SnippetMutationError(f"mutator command timed out after {self.timeout_seconds}s") from exc
        if result.returncode:
            raise SnippetMutationError(
                f"mutator command exited with {result.returncode}: {result.stderr.strip()[:1000]}"
            )
        replacement, replacement_2 = parse_fusion_output(result.stdout, request.fusion_kind) if isinstance(request, FusionMutationRequest) else (parse_mutator_output(result.stdout), None)
        return MutationOutput(
            replacement=replacement,
            prompt=prompt,
            raw_output=result.stdout,
            replacement_2=replacement_2,
        )


class OpenAICompatibleSnippetMutator(SnippetMutator):
    """Call an OpenAI-compatible endpoint without sending the complete host file."""

    def __init__(
        self,
        *,
        api_base: str,
        model: str,
        api_key_env: str | None,
        timeout_seconds: float = 120.0,
        temperature: float = 0.0,
    ) -> None:
        self.api_base = api_base
        self.model = model
        self.api_key_env = api_key_env
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature

    def mutate(self, request: MutationRequest) -> str:
        return self.mutate_with_trace(request).replacement

    def mutate_with_trace(self, request: MutationRequest) -> MutationOutput:
        api_key = os.environ.get(self.api_key_env) if self.api_key_env else None
        if self.api_key_env and not api_key:
            raise SnippetMutationError(f"missing API key environment variable: {self.api_key_env}")
        body = {
            "model": self.model,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You mutate one Java AST subtree for compiler fuzzing. Return one JSON object with "
                        "exactly the field mutated_subtree. Preserve the host subtree's syntactic role. "
                        "Return no complete file, imports, Markdown, or explanation."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(request.to_dict(), ensure_ascii=False),
                },
            ],
        }
        if isinstance(request, FusionMutationRequest):
            output_keys = "mutated_subtree_1 and mutated_subtree_2" if request.fusion_kind == "cross_node" else "mutated_subtree"
            body["messages"][0]["content"] = (
                "You fuse TWO historical javac anomaly patterns. Keep BOTH anomaly mechanisms recognizable. "
                "Preserve each host subtree's syntactic role and use the supplied interaction evidence. "
                f"Return exactly one JSON object with only the fields {output_keys}, containing replacement Java snippets. "
                "Return no complete file, imports, Markdown, or explanation. " + request.instruction
            )
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        http_request = urllib.request.Request(
            self.api_base,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8", errors="replace"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise SnippetMutationError(f"snippet mutation API failed: {exc}") from exc
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise SnippetMutationError("snippet mutation API returned an invalid response") from exc
        raw_output = str(content)
        replacement, replacement_2 = parse_fusion_output(raw_output, request.fusion_kind) if isinstance(request, FusionMutationRequest) else (parse_mutator_output(raw_output), None)
        return MutationOutput(
            replacement=replacement,
            prompt=json.dumps(body, ensure_ascii=False, indent=2),
            raw_output=raw_output,
            replacement_2=replacement_2,
        )


def parse_mutator_output(output: str) -> str:
    output = output.strip()
    if not output:
        raise SnippetMutationError("mutator returned empty output")
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        snippet = output
    else:
        if not isinstance(payload, dict):
            raise SnippetMutationError("mutator JSON output must be an object")
        snippet = str(payload.get("mutated_subtree") or "").strip()
    snippet = _strip_code_fence(snippet)
    if not snippet:
        raise SnippetMutationError("mutator output has no mutated_subtree")
    return snippet


def parse_fusion_output(output: str, fusion_kind: str) -> tuple[str, str | None]:
    """Require a complete one- or two-snippet fusion result, allowing common wrappers."""
    text = output.strip()
    tagged = re.search(r"<code>(.*?)</code>", text, flags=re.DOTALL | re.IGNORECASE)
    if tagged:
        text = html.unescape(tagged.group(1).strip())
    text = _strip_code_fence(text)
    decoder = json.JSONDecoder()
    payload = None
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            payload = candidate
            break
    keys = ("mutated_subtree_1", "mutated_subtree_2") if fusion_kind == "cross_node" else ("mutated_subtree",)
    if payload is None or set(payload) != set(keys):
        raise SnippetMutationError("fusion output must contain exactly " + ", ".join(keys))
    if any(not isinstance(payload[key], str) or not payload[key].strip() for key in keys):
        raise SnippetMutationError("fusion output contains an empty or non-string replacement")
    fragments = [_strip_code_fence(payload[key]).strip() for key in keys]
    if not all(fragments):
        raise SnippetMutationError("fusion output contains an empty replacement")
    return fragments[0], fragments[1] if len(fragments) == 2 else None


def compose_subtree(source: str, start_byte: int, end_byte: int, replacement: str) -> tuple[str, str]:
    source_bytes = source.encode("utf-8")
    if start_byte < 0 or end_byte < start_byte or end_byte > len(source_bytes):
        raise ValueError(f"invalid replacement span: {start_byte}:{end_byte}")
    line_start = source_bytes.rfind(b"\n", 0, start_byte) + 1
    prefix_on_line = source_bytes[line_start:start_byte].decode("utf-8", errors="replace")
    indentation = prefix_on_line if prefix_on_line.strip() == "" else ""
    normalized = indent_fragment(replacement, indentation)
    composed = source_bytes[:start_byte] + normalized.encode("utf-8") + source_bytes[end_byte:]
    return composed.decode("utf-8", errors="replace"), normalized


def compose_two_spans(source: str, span1: tuple[int, int], span2: tuple[int, int], replacement1: str, replacement2: str) -> tuple[str, str, str]:
    """Replace disjoint UTF-8 ranges from right to left, preserving caller order."""
    size = len(source.encode("utf-8"))
    spans = [span1, span2]
    for start, end in spans:
        if not 0 <= start < end <= size:
            raise ValueError(f"invalid replacement span: {start}:{end}")
    if not (span1[1] <= span2[0] or span2[1] <= span1[0]):
        raise ValueError("fusion replacement spans overlap")
    normalized = ["", ""]
    composed = source
    replacements = [replacement1, replacement2]
    for index in sorted(range(2), key=lambda i: spans[i][0], reverse=True):
        composed, normalized[index] = compose_subtree(composed, *spans[index], replacements[index])
    return composed, normalized[0], normalized[1]


def historical_fusion(request: FusionMutationRequest) -> tuple[str, str | None]:
    """AST-slot templates for the zero-model fusion baseline."""
    from .fusion import EXPRESSION_KINDS, STATEMENT_KINDS
    from .parser import parse_java

    first = request.base.historical_subtree.strip()
    second = request.historical_subtree_2.strip()
    if not first or not second:
        raise SnippetMutationError("fusion requires two historical subtrees")
    if request.fusion_kind == "cross_node":
        return first, second
    operation = request.fusion_role
    if operation in {"role_extension", "branch_splitting", "constraint_conjunction"}:
        # The conditional's common target type couples both historical expressions.
        condition = f"({request.base.host_subtree}) != null" if operation == "role_extension" else "true"
        return f"({condition} ? ({first}) : ({second}))", None

    root_kind = request.pattern_2.get("root_kind", "")
    expression = root_kind in EXPRESSION_KINDS
    prefix = "class __Fusion { Object value = " if expression else "class __Fusion { void method() { "
    suffix = "; }" if expression else " } }"
    source = prefix + second + suffix
    tree = parse_java(source, "__Fusion.java")
    if tree.syntax_error_count:
        raise SnippetMutationError("historical wrapper cannot be parsed in its syntactic role")
    start = len(prefix.encode("utf-8"))
    end = start + len(second.encode("utf-8"))
    roots = [n for n in tree.nodes.values() if n.start_byte == start and n.end_byte == end]
    if not roots:
        raise SnippetMutationError("historical wrapper has no complete AST root")
    root = min(roots, key=lambda n: len(tree.ancestors(n.node_id)))
    children = [tree.nodes[n] for n in root.child_ids]
    if root.raw_kind == "BLOCK":
        # Insert a nested block before the outer wrapper's closing brace.
        return second[:-1] + "\n{\n" + first + "\n}\n}", None
    if expression:
        slots = [n for n in children if n.raw_kind in EXPRESSION_KINDS]
        if root.raw_kind == "METHOD_INVOCATION":
            # Explicit type arguments precede the method select in TreeScanner order.
            # Find the select whose source is followed by the invocation's '(' .
            raw_source = source.encode("utf-8")
            selects = []
            for child in children:
                if child.raw_kind not in {"IDENTIFIER", "MEMBER_SELECT"}:
                    continue
                tail = raw_source[child.end_byte:end].decode("utf-8")
                tail = re.sub(r"^(?:\s|/\*.*?\*/|//[^\n]*(?:\n|$))*", "", tail, flags=re.DOTALL)
                if tail.startswith("("):
                    selects.append(child)
            if not selects:
                raise SnippetMutationError("historical invocation has no identifiable method select")
            select_end = selects[0].end_byte
            slots = [n for n in slots if n.start_byte > select_end]
        elif root.raw_kind in {"TYPE_CAST", "PARENTHESIZED"}:
            slots = slots[-1:]
        else:
            slots = []
        if not slots:
            raise SnippetMutationError("historical template has no expression argument/operand slot")
        slot = slots[0]
        # Keep casts/parentheses belonging to the outer mechanism around the inserted expression.
        if operation == "wrapper":
            while slot.raw_kind in {"TYPE_CAST", "PARENTHESIZED"}:
                operands = [tree.nodes[n] for n in slot.child_ids if tree.nodes[n].raw_kind in EXPRESSION_KINDS]
                if not operands:
                    break
                slot = operands[-1]
        replacement = "(" + first + ")"
    else:
        slots = [n for n in children if n.raw_kind in STATEMENT_KINDS]
        if not slots:
            raise SnippetMutationError("historical wrapper has no statement body slot")
        slot = slots[-1]
        body = tree.snippet(slot.start_byte, slot.end_byte)
        replacement = "{\n" + body + "\n{\n" + first + "\n}\n}"
    raw = second.encode("utf-8")
    fused = raw[:slot.start_byte - start] + replacement.encode("utf-8") + raw[slot.end_byte - start:]
    return fused.decode("utf-8"), None


def indent_fragment(fragment: str, indentation: str) -> str:
    cleaned = textwrap.dedent(fragment).strip("\n")
    lines = cleaned.splitlines()
    if not lines:
        return ""
    if len(lines) > 1 and not lines[0].startswith((" ", "\t")):
        closing_indent = len(lines[-1]) - len(lines[-1].lstrip(" "))
        if closing_indent and lines[-1].lstrip().startswith(("}", ")", "]")):
            lines[1:] = [
                line[closing_indent:] if line.startswith(" " * closing_indent) else line
                for line in lines[1:]
            ]
    return lines[0] + "".join(f"\n{indentation}{line}" for line in lines[1:])


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()

