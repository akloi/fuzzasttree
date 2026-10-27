"""Parse Java into a byte-addressable javac Tree API AST."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import tempfile
from typing import Iterable

from .models import SyntaxNode


@dataclass(slots=True)
class JavaSyntaxTree:
    source: str
    source_name: str
    nodes: dict[int, SyntaxNode]
    root_id: int
    parse_status: str = "parsed"
    syntax_error_count: int = 0

    @property
    def source_bytes(self) -> bytes:
        return self.source.encode("utf-8")

    def snippet(self, start_byte: int, end_byte: int) -> str:
        return self.source_bytes[start_byte:end_byte].decode("utf-8", errors="replace")

    def ancestors(self, node_id: int, include_self: bool = True) -> list[int]:
        result: list[int] = []
        current: int | None = node_id if include_self else self.nodes[node_id].parent_id
        while current is not None:
            result.append(current)
            current = self.nodes[current].parent_id
        result.reverse()
        return result

    def descendants(self, root_id: int) -> set[int]:
        result: set[int] = set()
        stack = [root_id]
        while stack:
            node_id = stack.pop()
            if node_id in result:
                continue
            result.add(node_id)
            stack.extend(reversed(self.nodes[node_id].child_ids))
        return result

    def kinds(self) -> set[str]:
        return {node.raw_kind for node in self.nodes.values()}

    def occurrences(self, kind: str) -> list[int]:
        return [node.node_id for node in self.nodes.values() if node.raw_kind == kind]


def parse_java(source: str, source_name: str = "Test.java", backend: str = "javac") -> JavaSyntaxTree:
    """Parse with the bundled javac worker so both sides share one native kind domain."""
    if backend not in {"auto", "javac"}:
        raise ValueError("astsubtree supports only the native javac parser backend")
    return _parse_with_javac(source, source_name)


def _parse_with_javac(source: str, source_name: str) -> JavaSyntaxTree:
    class_dir = _ensure_javac_worker()
    safe_name = Path(source_name).name
    if not safe_name.endswith(".java"):
        safe_name += ".java"
    with tempfile.TemporaryDirectory(prefix="astsubtree-source-") as temporary:
        source_path = Path(temporary) / safe_name
        source_path.write_text(source, encoding="utf-8")
        result = subprocess.run(
            ["java", "-cp", str(class_dir), "astsubtree.JavacAstWorker", str(source_path)],
            text=True,
            encoding="utf-8",
            capture_output=True,
            timeout=30,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(f"javac AST worker failed ({result.returncode}): {result.stderr.strip()[:2000]}")

    utf16_to_byte = _utf16_to_utf8_boundaries(source)
    nodes: dict[int, SyntaxNode] = {}
    parse_status = "parser_failure"
    syntax_errors = 0
    for line in result.stdout.splitlines():
        fields = line.split("\t")
        if not fields:
            continue
        if fields[0] == "S" and len(fields) >= 3:
            parse_status = fields[1]
            syntax_errors = int(fields[2])
            continue
        if fields[0] != "N" or len(fields) != 9:
            continue
        node_id = int(fields[1])
        raw_kind = fields[2]
        parent_value = int(fields[3])
        start_utf16 = int(fields[5])
        end_utf16 = int(fields[6])
        start_byte = _boundary(utf16_to_byte, start_utf16, 0)
        end_byte = _boundary(utf16_to_byte, end_utf16, len(source.encode("utf-8")))
        nodes[node_id] = SyntaxNode(
            node_id=node_id,
            raw_kind=raw_kind,
            parent_id=None if parent_value < 0 else parent_value,
            child_ids=[],
            child_index=int(fields[4]),
            start_byte=start_byte,
            end_byte=max(start_byte, end_byte),
            start_line=max(1, int(fields[7])),
            end_line=max(1, int(fields[8])),
        )
    for node in nodes.values():
        if node.parent_id in nodes:
            nodes[node.parent_id].child_ids.append(node.node_id)
    for node in nodes.values():
        node.child_ids.sort(key=lambda node_id: (nodes[node_id].child_index, node_id))
    roots = sorted(node.node_id for node in nodes.values() if node.parent_id is None)
    if not roots:
        raise RuntimeError("javac AST worker returned no root node")
    return JavaSyntaxTree(
        source=source,
        source_name=source_name,
        nodes=nodes,
        root_id=roots[0],
        parse_status=parse_status,
        syntax_error_count=syntax_errors,
    )


def _ensure_javac_worker() -> Path:
    java_source = Path(__file__).resolve().parent / "java" / "astsubtree" / "JavacAstWorker.java"
    class_dir = Path(tempfile.gettempdir()) / "astsubtree-javac-worker-v1"
    class_file = class_dir / "astsubtree" / "JavacAstWorker.class"
    if class_file.exists() and class_file.stat().st_mtime >= java_source.stat().st_mtime:
        return class_dir
    class_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["javac", "-encoding", "UTF-8", "-d", str(class_dir), str(java_source)],
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=30,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"failed to compile javac AST worker: {result.stderr.strip()[:2000]}")
    return class_dir


def _utf16_to_utf8_boundaries(source: str) -> list[int]:
    boundaries = [0]
    utf8_offset = 0
    for character in source:
        width16 = 2 if ord(character) > 0xFFFF else 1
        if width16 == 2:
            boundaries.append(utf8_offset)
        utf8_offset += len(character.encode("utf-8", errors="replace"))
        boundaries.append(utf8_offset)
    return boundaries


def _boundary(boundaries: list[int], offset: int, fallback: int) -> int:
    if offset < 0:
        return fallback
    if offset >= len(boundaries):
        return boundaries[-1]
    return boundaries[offset]


def parse_java_file(path: Path) -> JavaSyntaxTree:
    return parse_java(path.read_text(encoding="utf-8", errors="replace"), path.name)


def has_syntax_errors(tree: JavaSyntaxTree) -> bool:
    return tree.syntax_error_count > 0 or any(node.raw_kind in {"ERROR", "ERRONEOUS"} for node in tree.nodes.values())


def native_path(tree: JavaSyntaxTree, node_ids: Iterable[int]) -> tuple[str, ...]:
    return tuple(tree.nodes[node_id].primary_kind for node_id in node_ids)

