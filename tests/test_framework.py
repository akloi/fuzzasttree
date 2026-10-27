from __future__ import annotations

from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from astsubtree.library import HistoricalLibrary
from astsubtree.local_model_adapter import (
    LocalModelSnippetMutator,
    build_subtree_prompt,
    main as local_adapter_main,
    parse_model_output,
)
from astsubtree.matcher import SubtreeMatcher
from astsubtree.mutator import (
    CommandSnippetMutator,
    MutationOutput,
    OpenAICompatibleSnippetMutator,
    compose_subtree,
    indent_fragment,
    parse_mutator_output,
)
from astsubtree.parser import parse_java
from astsubtree.pipeline import MutationPipeline, PipelineOptions, candidate_filename


SOURCE = """class T {
  void test() {
    for (int i = 0; i < 1; i++);
    int value = 1;
    value = 2;
  }
}
"""


class FrameworkTests(unittest.TestCase):
    def test_parser_preserves_byte_spans_with_unicode(self) -> None:
        source = 'class T { String s = "中"; void m() { s = "x"; } }'
        tree = parse_java(source, "T.java")
        assignment = next(node for node in tree.nodes.values() if node.raw_kind == "ASSIGNMENT")
        self.assertEqual(tree.snippet(assignment.start_byte, assignment.end_byte), 's = "x"')

    def test_compose_replaces_only_local_byte_span(self) -> None:
        source = 'class T { String s = "中"; void m() { s = "x"; } }'
        tree = parse_java(source, "T.java")
        assignment = next(node for node in tree.nodes.values() if node.raw_kind == "ASSIGNMENT")
        composed, fragment = compose_subtree(source, assignment.start_byte, assignment.end_byte, 's = "y"')
        self.assertEqual(fragment, 's = "y"')
        self.assertIn('String s = "中"', composed)
        self.assertIn('s = "y"', composed)
        self.assertNotIn('s = "x"', composed)

    def test_two_stage_match_finds_structural_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            library_dir = root / "library"
            source_dir = root / "sources"
            library_dir.mkdir()
            source_dir.mkdir()
            (library_dir / "JDK-test.json").write_text(
                json.dumps(_historical_record(), indent=2), encoding="utf-8"
            )
            (source_dir / "JDK-test.json").write_text(
                json.dumps(
                    _historical_source_record(
                        summary="Compiler crashes on a loop and assignment",
                        description="A historical report description.",
                    )
                ),
                encoding="utf-8",
            )
            library = HistoricalLibrary(library_dir, source_dir)
            pattern = library.patterns["JDK-test"]
            self.assertIn("FOR_LOOP", pattern.required_kinds)
            self.assertNotIn("FOR", pattern.required_kinds)
            self.assertEqual(pattern.signature["version"], "javac-native-subtree-signature-v1")
            self.assertEqual(pattern.ancestor_path, ("COMPILATION_UNIT", "CLASS", "METHOD", "BLOCK"))
            material = library.reference_material(library.patterns["JDK-test"])
            self.assertEqual(material["Summary"], "Compiler crashes on a loop and assignment")
            self.assertEqual(material["Description"], "A historical report description.")
            self.assertEqual(material["extracted_code"], SOURCE)
            tree = parse_java(SOURCE, "T.java")
            matches = SubtreeMatcher(library).match(tree, prefilter_limit=10, top_k=3)
            self.assertEqual(len(matches), 1)
            self.assertEqual(matches[0].pattern.bug_id, "JDK-test")
            self.assertEqual(
                set(matches[0].local_subtree.covered_kinds),
                {"BLOCK", "FOR_LOOP", "VARIABLE", "EXPRESSION_STATEMENT", "ASSIGNMENT"},
            )
            self.assertGreater(matches[0].final_score, 0.3)

    def test_feature_compatibility_reranks_identical_structures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            library_dir = root / "library"
            source_dir = root / "sources"
            library_dir.mkdir()
            source_dir.mkdir()
            records = []
            for bug_id, feature in (
                ("JDK-z-compatible", "for loop, variable assignment"),
                ("JDK-a-incompatible", "lambda expressions, record patterns"),
            ):
                record = deepcopy(_historical_record())
                record["bug_id"] = bug_id
                record["source_file"] = f"{bug_id}.json"
                record["abstraction"]["L"] = feature
                records.append(record)
                (library_dir / f"{bug_id}.json").write_text(json.dumps(record), encoding="utf-8")
                (source_dir / f"{bug_id}.json").write_text(
                    json.dumps(_historical_source_record()), encoding="utf-8"
                )

            matches = SubtreeMatcher(HistoricalLibrary(library_dir, source_dir)).match(
                parse_java(SOURCE, "T.java"), prefilter_limit=10, top_k=2
            )
            self.assertEqual(matches[0].pattern.bug_id, "JDK-z-compatible")
            self.assertEqual(matches[0].components["feature_compatibility"], 1.0)
            self.assertEqual(matches[1].components["feature_compatibility"], 0.0)
            self.assertEqual(matches[0].feature_evidence["matched"], ["loop", "assignment"])

    def test_native_restoration_rejects_missing_original_node(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            library_dir = root / "library"
            source_dir = root / "sources"
            library_dir.mkdir()
            source_dir.mkdir()
            record = _historical_record()
            record["minimum_anomaly_subtree"]["nodes"][0]["original_node_id"] = 999
            (library_dir / "JDK-test.json").write_text(json.dumps(record), encoding="utf-8")
            (source_dir / "JDK-test.json").write_text(
                json.dumps(_historical_source_record()), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "original_node_id 999"):
                HistoricalLibrary(library_dir, source_dir)

    def test_command_mutator_json_protocol(self) -> None:
        script = Path(__file__).resolve().parents[1] / "examples" / "passthrough_mutator.py"
        from astsubtree.models import MutationRequest

        request = MutationRequest(
            bug_id="JDK-test",
            host_subtree="x = 1",
            historical_subtree="x = 2",
            language_feature="assignment",
            required_kinds=("ASSIGNMENT",),
            phases=("Attr",),
            diagnostic="error",
            match_score=0.5,
        )
        result = CommandSnippetMutator(f"python3 {script}").mutate(request)
        self.assertEqual(result, "x = 2")

    def test_fragment_indentation_removes_historical_line_indent(self) -> None:
        fragment = "for (;;) {\n                work();\n            }"
        self.assertEqual(
            indent_fragment(fragment, "    "),
            "for (;;) {\n        work();\n    }",
        )

    def test_plain_mutator_output_is_supported(self) -> None:
        self.assertEqual(parse_mutator_output("```java\nx = 2;\n```"), "x = 2;")

    def test_local_openai_endpoint_can_omit_api_key(self) -> None:
        from astsubtree.models import MutationRequest

        request = MutationRequest(
            bug_id="JDK-test",
            host_subtree="x = 1",
            historical_subtree="x = 2",
            language_feature="assignment",
            required_kinds=("ASSIGNMENT",),
            phases=("Attr",),
            diagnostic="error",
            match_score=0.5,
        )

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(
                    {"choices": [{"message": {"content": '{"mutated_subtree":"x = 3"}'}}]}
                ).encode()

        mutator = OpenAICompatibleSnippetMutator(
            api_base="http://127.0.0.1:8000/v1/chat/completions",
            model="local-model",
            api_key_env=None,
        )
        with patch("urllib.request.urlopen", return_value=Response()) as mocked:
            self.assertEqual(mutator.mutate(request), "x = 3")
        sent_request = mocked.call_args.args[0]
        self.assertIsNone(sent_request.get_header("Authorization"))

    def test_candidate_filename_uses_public_type(self) -> None:
        self.assertEqual(candidate_filename(Path("seed.java"), "public class Actual {}"), "Actual.java")

    def test_local_model_prompt_contains_historical_report_only(self) -> None:
        request = _mutation_request().to_dict()
        request["complete_host_file"] = "class MustNotBeSent {}"
        prompt = build_subtree_prompt(request)
        self.assertIn('"Summary": "Historical summary"', prompt)
        self.assertIn('"Description": "Historical description"', prompt)
        self.assertIn('"extracted_code": "class Historical {}"', prompt)
        self.assertIn('"L": "assignment"', prompt)
        self.assertIn('"A_canonical": [', prompt)
        self.assertIn('"A_native": [', prompt)
        self.assertIn('"P": [', prompt)
        self.assertIn('"D": "NullPointerException"', prompt)
        self.assertNotIn("MustNotBeSent", prompt)

    def test_local_model_mutator_reuses_generator_and_cleans_json(self) -> None:
        class FakeGenerator:
            def __init__(self):
                self.prompts = []

            def generate_batch(self, prompts):
                self.prompts.extend(prompts)
                return ['```json\n{"mutated_subtree":"x = 7;"}\n```']

        generator = FakeGenerator()
        mutator = LocalModelSnippetMutator(generator=generator)
        self.assertEqual(mutator.mutate(_mutation_request()), "x = 7;")
        self.assertEqual(mutator.mutate(_mutation_request()), "x = 7;")
        self.assertEqual(len(generator.prompts), 2)

    def test_pipeline_writes_codes_and_complete_mutation_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            library_dir = root / "library"
            source_dir = root / "sources"
            library_dir.mkdir()
            source_dir.mkdir()
            (library_dir / "JDK-test.json").write_text(
                json.dumps(_historical_record()), encoding="utf-8"
            )
            (source_dir / "JDK-test.json").write_text(
                json.dumps(_historical_source_record()), encoding="utf-8"
            )
            seed_path = root / "Seed.java"
            seed_path.write_text(SOURCE, encoding="utf-8")

            class RecordingMutator:
                def mutate_with_trace(self, request):
                    return MutationOutput(
                        replacement=request.host_subtree,
                        prompt="EXACT MODEL PROMPT",
                        raw_output='{"mutated_subtree":"MODEL RAW OUTPUT"}',
                    )

            output_dir = root / "run"
            summary = MutationPipeline(HistoricalLibrary(library_dir, source_dir)).mutate_file(
                seed_path,
                output_dir,
                RecordingMutator(),
                PipelineOptions(top_k=1, max_outputs=1, max_host_fraction=1.0),
            )
            self.assertEqual(len(summary["written"]), 1)
            mutation_id = summary["written"][0]["mutation_id"]
            mutation_dir = output_dir / "mutations" / mutation_id
            self.assertEqual((mutation_dir / "Original.java").read_text(), SOURCE)
            mutated_source = (mutation_dir / "Mutated.java").read_text()
            self.assertEqual(
                (output_dir / "codes" / f"{mutation_id}.java").read_text(),
                mutated_source,
            )
            self.assertEqual((mutation_dir / "prompt.txt").read_text(), "EXACT MODEL PROMPT")
            self.assertIn("MODEL RAW OUTPUT", (mutation_dir / "raw_llm_output.txt").read_text())
            self.assertTrue((mutation_dir / "request.json").is_file())
            self.assertTrue((mutation_dir / "match.json").is_file())
            self.assertTrue((mutation_dir / "parse_validation.json").is_file())
            self.assertTrue((mutation_dir / "provenance.json").is_file())
            self.assertTrue((output_dir / "summaries" / "Seed.summary.json").is_file())

    def test_model_output_parser_accepts_wrapped_json(self) -> None:
        output = 'result:\n{"mutated_subtree":"for (;;) { work(); }"}\n'
        self.assertEqual(parse_model_output(output), "for (;;) { work(); }")

    def test_standalone_adapter_keeps_logs_out_of_stdout(self) -> None:
        class NoisyGenerator:
            def generate_batch(self, _prompts):
                print("generation log")
                return ['{"mutated_subtree":"x = 9;"}']

        def factory(_args):
            print("loading log")
            return NoisyGenerator()

        stdout = io.StringIO()
        stderr = io.StringIO()
        status = local_adapter_main(
            ["--model", "fake"],
            stdin=io.StringIO(json.dumps(_mutation_request().to_dict())),
            stdout=stdout,
            stderr=stderr,
            generator_factory=factory,
        )
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(stdout.getvalue()), {"mutated_subtree": "x = 9;"})
        self.assertIn("loading log", stderr.getvalue())
        self.assertIn("generation log", stderr.getvalue())


def _historical_record() -> dict:
    source_utf16_length = len(SOURCE.encode("utf-16-le")) // 2
    return {
        "bug_id": "JDK-test",
        "source_file": "JDK-test.json",
        "abstraction": {
            "L": "for loop and reassignment",
            "A_canonical": ["FOR", "VARIABLE", "ASSIGNMENT"],
            "P": ["Generate"],
            "D": "compiler error",
            "confidence": 0.9,
        },
        "ast_relevance": {
            "status": "candidate",
            "reason": "test",
            "candidate_confidence": 0.9,
        },
        "minimum_anomaly_subtree": {
            "coverage": 1.0,
            "corpus_rarity": 2.0,
            "ancestor_path": ["COMPILATION_UNIT", "TYPE_DECLARATION", "METHOD", "BLOCK"],
            "source_span_utf16": {"start": 0, "end": source_utf16_length},
            "nodes": [
                {"relative_id": 0, "original_node_id": 3, "canonical_kind": "BLOCK", "parent_relative_id": None, "child_index": 0},
                {"relative_id": 1, "original_node_id": 4, "canonical_kind": "FOR", "parent_relative_id": 0, "child_index": 0},
                {"relative_id": 2, "original_node_id": 5, "canonical_kind": "VARIABLE", "parent_relative_id": 0, "child_index": 1},
                {"relative_id": 3, "original_node_id": 6, "canonical_kind": "EXPRESSION_STATEMENT", "parent_relative_id": 0, "child_index": 2},
                {"relative_id": 4, "original_node_id": 7, "canonical_kind": "ASSIGNMENT", "parent_relative_id": 3, "child_index": 0},
            ],
            "root_original_node_id": 3,
            "matching_signature": {
                "root_kind": "BLOCK",
                "required_kinds": ["FOR", "VARIABLE", "ASSIGNMENT"],
                "node_kind_multiset": {
                    "BLOCK": 1,
                    "FOR": 1,
                    "VARIABLE": 1,
                    "EXPRESSION_STATEMENT": 1,
                    "ASSIGNMENT": 1,
                },
                "edge_kind_multiset": {
                    "BLOCK>FOR": 1,
                    "BLOCK>VARIABLE": 1,
                    "BLOCK>EXPRESSION_STATEMENT": 1,
                    "EXPRESSION_STATEMENT>ASSIGNMENT": 1,
                },
                "root_to_node_paths": [
                    "BLOCK",
                    "BLOCK>FOR",
                    "BLOCK>VARIABLE",
                    "BLOCK>EXPRESSION_STATEMENT",
                    "BLOCK>EXPRESSION_STATEMENT>ASSIGNMENT",
                ],
                "node_count": 5,
                "sha256": "test-signature",
            },
        },
    }


def _mutation_request():
    from astsubtree.models import MutationRequest

    return MutationRequest(
        bug_id="JDK-test",
        host_subtree="x = 1;",
        historical_subtree="x = 2;",
        language_feature="assignment",
        required_kinds=("ASSIGNMENT",),
        phases=("Attr",),
        diagnostic="NullPointerException",
        match_score=0.8,
        historical_summary="Historical summary",
        historical_description="Historical description",
        historical_source="class Historical {}",
        abstract_raw_kinds=("Assignment",),
        abstract_canonical_kinds=("ASSIGNMENT",),
    )


def _historical_source_record(summary: str = "", description: str = "") -> dict:
    return {
        "Summary": summary,
        "Description": description,
        "extracted_code": SOURCE,
        "ast": {
            "parse_status": "parsed",
            "nodes": [
                {"node_id": 0, "kind": "COMPILATION_UNIT", "parent_id": None, "child_index": -1},
                {"node_id": 1, "kind": "CLASS", "parent_id": 0, "child_index": 0},
                {"node_id": 2, "kind": "METHOD", "parent_id": 1, "child_index": 0},
                {"node_id": 3, "kind": "BLOCK", "parent_id": 2, "child_index": 0},
                {"node_id": 4, "kind": "FOR_LOOP", "parent_id": 3, "child_index": 0},
                {"node_id": 5, "kind": "VARIABLE", "parent_id": 3, "child_index": 1},
                {"node_id": 6, "kind": "EXPRESSION_STATEMENT", "parent_id": 3, "child_index": 2},
                {"node_id": 7, "kind": "ASSIGNMENT", "parent_id": 6, "child_index": 0},
            ],
        },
    }


if __name__ == "__main__":
    unittest.main()
