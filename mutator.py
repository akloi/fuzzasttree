"""Subtree mutation providers and byte-accurate source composition."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import os
import shlex
import subprocess
import textwrap
import urllib.error
import urllib.request

from .models import MutationRequest


class SnippetMutationError(RuntimeError):
    pass


@dataclass(slots=True)
class MutationOutput:
    """Parsed replacement plus the exact model input and unparsed response."""

    replacement: str
    prompt: str
    raw_output: str


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
        snippet = request.historical_subtree.strip()
        if not snippet:
            raise SnippetMutationError("historical subtree is empty")
        return snippet


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
        return MutationOutput(
            replacement=parse_mutator_output(result.stdout),
            prompt=prompt,
            raw_output=result.stdout,
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
        return MutationOutput(
            replacement=parse_mutator_output(raw_output),
            prompt=json.dumps(body, ensure_ascii=False, indent=2),
            raw_output=raw_output,
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

