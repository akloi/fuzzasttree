from __future__ import annotations

from contextlib import redirect_stdout, redirect_stderr
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from astsubtree.cli import main as cli_main
from astsubtree.fusion import (
    build_anchor_interaction_graph, cross_node_plans, group_matches,
    mechanism_orthogonality, same_node_plans,
)
from astsubtree.library import HistoricalLibrary
from astsubtree.local_model_adapter import LocalModelSnippetMutator, build_fusion_prompt, main as adapter_main
from astsubtree.matcher import SubtreeMatcher
from astsubtree.models import FusionMutationRequest, MutationRequest
from astsubtree.mutator import (
    CommandSnippetMutator, HistoricalSnippetMutator, MutationOutput,
    OpenAICompatibleSnippetMutator, SnippetMutationError, compose_two_spans, historical_fusion, parse_fusion_output,
)
from astsubtree.parser import parse_java
from astsubtree.pipeline import FusionPipeline, MutationPipeline, PipelineOptions, replacement_regions


SAME_SEED = "class Seed { Object m(Object x) { return use((Object)x, x::toString); } }"
CROSS_SEED = 'class Seed { String label = "中"; void m(Object y) { use((Object)y, y::toString); } }'
CONTROL_SEED = 'class Seed { void m(Object y, boolean flag) { if (flag) use((Object)y); else use(y::toString); } }'


def write_reference(root: Path, bug: str, expression: str, kind: str, feature: str, phase: str) -> None:
    """Build real historical records from the same javac parser used in production."""
    library, sources = root / "library", root / "sources"
    library.mkdir(exist_ok=True)
    sources.mkdir(exist_ok=True)
    source = "class Historical { Object m(Object x) { return " + expression + "; } }"
    tree = parse_java(source, "Historical.java")
    anchor = next(n for n in tree.nodes.values() if n.raw_kind == kind and tree.snippet(n.start_byte, n.end_byte) == expression)
    selected = sorted(tree.descendants(anchor.node_id))
    relative = {node_id: i for i, node_id in enumerate(selected)}
    start = len(tree.source_bytes[:anchor.start_byte].decode().encode("utf-16-le")) // 2
    end = len(tree.source_bytes[:anchor.end_byte].decode().encode("utf-16-le")) // 2
    record = {
        "bug_id": bug, "source_file": bug + ".json",
        "abstraction": {"L": feature, "P": [phase], "D": phase + " historical failure", "confidence": 0.9},
        "ast_relevance": {"status": "candidate", "candidate_confidence": 0.9},
        "minimum_anomaly_subtree": {
            "coverage": 1.0, "root_original_node_id": anchor.node_id,
            "source_span_utf16": {"start": start, "end": end},
            "nodes": [{"relative_id": relative[n], "original_node_id": n, "parent_relative_id": relative.get(tree.nodes[n].parent_id), "child_index": tree.nodes[n].child_index} for n in selected],
        },
    }
    historical = {
        "Summary": "Summary for " + bug, "Description": "Description for " + bug,
        "extracted_code": source,
        "ast": {"nodes": [{"node_id": n.node_id, "parent_id": n.parent_id, "kind": n.raw_kind, "child_index": n.child_index} for n in tree.nodes.values()]},
    }
    (library / (bug + ".json")).write_text(json.dumps(record))
    (sources / (bug + ".json")).write_text(json.dumps(historical))


def make_library(root: Path, same: bool = False) -> HistoricalLibrary:
    if same:
        write_reference(root, "JDK-reference", "use(x::toString)", "METHOD_INVOCATION", "method reference", "Attr")
        write_reference(root, "JDK-cast", "use((String)x)", "METHOD_INVOCATION", "cast", "Flow")
    else:
        write_reference(root, "JDK-cast", "(String)x", "TYPE_CAST", "cast", "Attr")
        write_reference(root, "JDK-reference", "x::hashCode", "MEMBER_REFERENCE", "method reference", "Flow")
    return HistoricalLibrary(root / "library", root / "sources")


def cross_request(root: Path):
    library = make_library(root)
    pipeline = FusionPipeline(library)
    tree, matches = pipeline.match_source(CROSS_SEED, "Seed.java", PipelineOptions())
    groups = group_matches(matches)
    edges = build_anchor_interaction_graph(tree, groups)
    plan = cross_node_plans(tree, groups, edges)[0]
    return pipeline._fusion_request(tree, plan)


class FusionTests(unittest.TestCase):
    def test_all_five_same_node_templates_preserve_their_ast_constructs(self):
        cases = (
            ("wrapper", "x::toString", "(Object)x", "TYPE_CAST", {"TYPE_CAST", "MEMBER_REFERENCE"}),
            ("nested_expression", "x::toString", "bind(x, () -> x)", "METHOD_INVOCATION", {"METHOD_INVOCATION", "MEMBER_REFERENCE", "LAMBDA_EXPRESSION"}),
            ("nested_expression", "x::toString", "Factory.<String>bind /* slot */ (x, () -> x)", "METHOD_INVOCATION", {"METHOD_INVOCATION", "MEMBER_REFERENCE", "LAMBDA_EXPRESSION"}),
            ("role_extension", "(Object)x", "x::toString", "MEMBER_REFERENCE", {"CONDITIONAL_EXPRESSION", "TYPE_CAST", "MEMBER_REFERENCE"}),
            ("constraint_conjunction", "() -> x", "generic((Object)x)", "METHOD_INVOCATION", {"CONDITIONAL_EXPRESSION", "LAMBDA_EXPRESSION", "TYPE_CAST"}),
            ("branch_splitting", "(Object)x", "x::toString", "MEMBER_REFERENCE", {"CONDITIONAL_EXPRESSION", "TYPE_CAST", "MEMBER_REFERENCE"}),
        )
        for operation, one, two, root_kind, expected_kinds in cases:
            with self.subTest(operation=operation):
                base = MutationRequest("JDK-one", "x", one, "", (), ("Attr",), "", 0.8)
                request = FusionMutationRequest(base, "", {"root_kind": root_kind}, two, {}, operation, "", "", "JDK-two", 0.8)
                snippet, second = historical_fusion(request)
                self.assertIsNone(second)
                tree = parse_java("class T { Object x; Object v = " + snippet + "; }")
                self.assertEqual(tree.syntax_error_count, 0, snippet)
                self.assertTrue(expected_kinds <= tree.kinds(), snippet)

    def test_statement_wrapper_keeps_both_control_constructs(self):
        base = MutationRequest("JDK-one", "work();", "while (flag) { work(); }", "", (), ("Attr",), "", 0.8)
        request = FusionMutationRequest(base, "", {"root_kind": "IF"}, "if (flag) { work(); }", {}, "wrapper", "", "", "JDK-two", 0.8)
        snippet, _ = historical_fusion(request)
        tree = parse_java("class T { void m() { " + snippet + " } }")
        self.assertEqual(tree.syntax_error_count, 0, snippet)
        self.assertTrue({"IF", "WHILE_LOOP"} <= tree.kinds())

    def test_unicode_two_span_composition_preserves_order_and_outside_bytes(self):
        source = 'class T { String s = "中"; int x = 1; int y = 2; }'
        raw = source.encode()
        one = raw.index(b"1")
        two = raw.index(b"2")
        composed, first, second = compose_two_spans(source, (two, two + 1), (one, one + 1), "2000", "10")
        self.assertEqual(composed, source.replace("= 1", "= 10").replace("= 2", "= 2000"))
        regions = replacement_regions([(two, two + 1), (one, one + 1)], [first, second])
        self.assertEqual([composed.encode()[a:b].decode() for a, b in regions], ["2000", "10"])
        adjacent, _, _ = compose_two_spans("abcd", (0, 2), (2, 4), "X", "Y")
        self.assertEqual(adjacent, "XY")
        with self.assertRaisesRegex(ValueError, "overlap"):
            compose_two_spans(source, (one, two), (one, one + 1), "x", "y")
        with self.assertRaisesRegex(ValueError, "invalid"):
            compose_two_spans(source, (-1, 1), (two, two + 1), "x", "y")

    def test_same_anchor_retrieval_and_historical_fusion_end_to_end(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            library = make_library(root, same=True)
            tree = parse_java(SAME_SEED)
            groups = SubtreeMatcher(library).match_anchors(tree)
            self.assertTrue(any(len(group) == 2 for group in groups.values()))
            plans = same_node_plans(tree, groups)
            self.assertTrue(plans)
            self.assertGreater(plans[0].components["mechanism_orthogonality"], 0)
            self.assertEqual(mechanism_orthogonality(plans[0].match_1, plans[0].match_1), 0)
            unknown = replace(plans[0].match_1, pattern=replace(plans[0].match_1.pattern, phases=(), diagnostic="Unknown"))
            self.assertEqual(mechanism_orthogonality(unknown, plans[0].match_1), 0)
            seed = root / "Seed.java"
            seed.write_text(SAME_SEED)
            summary = MutationPipeline(library).mutate_file(seed, root / "run", HistoricalSnippetMutator(), PipelineOptions(fusion="same", fusion_budget=1, max_outputs=1))
            self.assertEqual(summary["fusion_stats"]["accepted"], 1, summary)
            record = summary["written"][0]
            self.assertEqual(record["fusion_kind"], "same_node")
            directory = Path(record["mutation_dir"])
            plan = json.loads((directory / "fusion_plan.json").read_text())
            self.assertEqual({plan["trace_to_bug_1"], plan["trace_to_bug_2"]}, {"JDK-cast", "JDK-reference"})
            self.assertIn("composed_template", plan)
            self.assertEqual(parse_java(Path(record["code_file"]).read_text()).syntax_error_count, 0)

    def test_cross_node_sibling_and_control_flow_execute_real_templates(self):
        for source, edge_type in ((CROSS_SEED, "sibling"), (CONTROL_SEED, "control_flow")):
            with self.subTest(edge=edge_type), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                library = make_library(root)
                seed = root / "Seed.java"
                seed.write_text(source)
                summary = MutationPipeline(library).mutate_file(seed, root / "run", HistoricalSnippetMutator(), PipelineOptions(fusion="cross", fusion_budget=1, max_outputs=1))
                self.assertEqual(summary["fusion_stats"]["accepted"], 1, summary)
                record = summary["written"][0]
                self.assertEqual(record["operation"], edge_type)
                directory = Path(record["mutation_dir"])
                request = json.loads((directory / "request.json").read_text())
                self.assertTrue(request["bridging_context"])
                self.assertNotEqual(request["bridging_context"].strip(), source.strip())
                self.assertEqual(request["edge_evidence"]["phase"], 1)
                self.assertTrue((directory / "mutated_subtree_1.javafrag").is_file())
                self.assertTrue((directory / "mutated_subtree_2.javafrag").is_file())
                validation = json.loads((directory / "parse_validation.json").read_text())
                self.assertTrue(validation["both_mechanisms_preserved"])
                self.assertTrue(validation["roles_preserved"])
                self.assertEqual(validation["symbol_resolution"], "not_performed")
                mutated = Path(record["code_file"]).read_text()
                self.assertIn("(String)x", mutated)
                self.assertIn("x::hashCode", mutated)
                self.assertEqual(parse_java(mutated).syntax_error_count, 0)

    def test_graph_records_ancestor_edge_but_never_schedules_overlapping_edits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            library = make_library(root)
            tree = parse_java(CROSS_SEED)
            matches = SubtreeMatcher(library).match(tree)
            cast = next(m for m in matches if m.pattern.bug_id == "JDK-cast")
            invocation = tree.nodes[tree.nodes[cast.local_subtree.root_id].parent_id]
            outer = replace(matches[1], local_subtree=replace(cast.local_subtree, root_id=invocation.node_id, start_byte=invocation.start_byte, end_byte=invocation.end_byte))
            groups = group_matches([cast, outer])
            edges = build_anchor_interaction_graph(tree, groups)
            self.assertEqual([e.edge_type for e in edges], ["ancestor_descendant"])
            self.assertFalse(edges[0].evidence["span_non_conflict"])
            self.assertEqual(cross_node_plans(tree, groups, edges), [])

    def test_lost_mechanism_is_rejected_and_budget_returns_to_single_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            library = make_library(root)
            seed = root / "Seed.java"
            seed.write_text(CROSS_SEED)

            class ErasingMutator:
                def mutate_with_trace(self, request):
                    if hasattr(request, "fusion_kind"):
                        return MutationOutput("null", "fusion prompt", "raw", "null")
                    return MutationOutput(request.host_subtree, "single prompt", "raw")

            summary = MutationPipeline(library).mutate_file(seed, root / "run", ErasingMutator(), PipelineOptions(fusion="cross", max_outputs=2, fusion_budget=0.5))
            self.assertEqual(summary["fusion_stats"]["accepted"], 0)
            self.assertEqual(summary["fusion_stats"]["mechanism_retention_rate"], 0)
            self.assertEqual(len(summary["written"]), 2)
            self.assertTrue(any("loses a historical mechanism" in r["reason"] for r in summary["rejected"]))
            self.assertFalse(any((Path(r["mutation_dir"]) / "fusion_plan.json").exists() for r in summary["written"]))

    def test_generic_shared_nodes_cannot_hide_loss_of_a_distinctive_mechanism(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            library = make_library(root, same=True)
            seed = root / "Seed.java"
            seed.write_text(SAME_SEED)

            class SingleMechanismMutator:
                def mutate_with_trace(self, request):
                    return MutationOutput("use(x::hashCode)", "prompt", "raw")

            summary = MutationPipeline(library).mutate_file(seed, root / "run", SingleMechanismMutator(), PipelineOptions(fusion="same", max_outputs=1, fusion_budget=1))
            self.assertEqual(summary["fusion_stats"]["accepted"], 0)
            validations = [r["validation"] for r in summary["rejected"] if "validation" in r]
            self.assertTrue(validations)
            self.assertTrue(any(not v["both_mechanisms_preserved"] for v in validations))

    def test_zero_fusion_budget_and_off_mode_preserve_single_pattern_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            library = make_library(root)
            seed = root / "Seed.java"
            seed.write_text(CROSS_SEED)
            for mode in ("off", "both"):
                with self.subTest(mode=mode):
                    summary = MutationPipeline(library).mutate_file(seed, root / mode, HistoricalSnippetMutator(), PipelineOptions(fusion=mode, max_outputs=1, fusion_budget=0))
                    self.assertEqual(len(summary["written"]), 1)
                    self.assertNotIn("fusion_kind", summary["written"][0])
                    if mode == "both":
                        self.assertEqual(summary["fusion_stats"]["attempted"], 0)
                    else:
                        self.assertNotIn("fusion_stats", summary)
            for options in ({"fusion_budget": 1.1}, {"fusion_pairs": 0}, {"anchor_limit": 0}):
                with self.assertRaises(ValueError):
                    PipelineOptions(**options)

    def test_cross_protocol_local_command_and_openai_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            request = cross_request(Path(temporary))
            raw = json.dumps({"mutated_subtree_1": "(String)y", "mutated_subtree_2": "y::hashCode"})

            class Generator:
                def generate_batch(self, prompts):
                    self.prompt = prompts[0]
                    return ["```json\n" + raw + "\n```"]

            generator = Generator()
            result = LocalModelSnippetMutator(generator=generator).mutate_with_trace(request)
            self.assertEqual((result.replacement, result.replacement_2), ("(String)y", "y::hashCode"))
            payload = request.to_dict()
            payload["complete_host_file"] = "DO_NOT_SEND_FULL_SEED"
            self.assertNotIn("DO_NOT_SEND_FULL_SEED", build_fusion_prompt(payload))
            self.assertIn("JDK-cast", generator.prompt)
            self.assertIn("JDK-reference", generator.prompt)
            script = Path(temporary) / "adapter.py"
            script.write_text("import sys,json\nr=json.load(sys.stdin)\nassert r['protocol_version']==2\nprint(" + repr(raw) + ")\n")
            command = CommandSnippetMutator(f"{sys.executable} {script}").mutate_with_trace(request)
            self.assertEqual(command.replacement_2, "y::hashCode")

            class Response:
                def __enter__(self): return self
                def __exit__(self, *_): return False
                def read(self): return json.dumps({"choices": [{"message": {"content": raw}}]}).encode()

            api = OpenAICompatibleSnippetMutator(api_base="http://127.0.0.1:8000/v1/chat/completions", model="test", api_key_env=None)
            with patch("urllib.request.urlopen", return_value=Response()) as mocked:
                result = api.mutate_with_trace(request)
            sent = json.loads(mocked.call_args.args[0].data)
            self.assertIn("mutated_subtree_2", sent["messages"][0]["content"])
            self.assertEqual(json.loads(sent["messages"][1]["content"])["bug_id_2"], request.bug_id_2)
            self.assertEqual(result.replacement_2, "y::hashCode")
            stdout = io.StringIO()
            status = adapter_main(["--model", "test"], stdin=io.StringIO(json.dumps(payload)), stdout=stdout, stderr=io.StringIO(), generator_factory=lambda _: generator)
            self.assertEqual(status, 0)
            self.assertEqual(json.loads(stdout.getvalue()), json.loads(raw))

    def test_fusion_output_requires_both_nonempty_snippets(self):
        for raw in ('{"mutated_subtree_1":"x"}', '{"mutated_subtree_1":"x","mutated_subtree_2":""}', '{"mutated_subtree_1":3,"mutated_subtree_2":"y"}'):
            with self.subTest(raw=raw), self.assertRaises(SnippetMutationError):
                parse_fusion_output(raw, "cross_node")
        self.assertEqual(parse_fusion_output('{"mutated_subtree":"(Object)x"}', "same_node"), ("(Object)x", None))

    def test_cli_run_aggregates_fusion_evidence_across_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_library(root)
            seeds = root / "seeds"
            seeds.mkdir()
            (seeds / "One.java").write_text(CROSS_SEED)
            (seeds / "Two.java").write_text(CONTROL_SEED)
            output = root / "run"
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                code = cli_main(["--library-dir", str(root / "library"), "--historical-source-dir", str(root / "sources"), "mutate", str(seeds), "--fusion", "both", "--fusion-budget", "1", "--max-outputs", "1", "--output-dir", str(output)])
            self.assertEqual(code, 0)
            summary = json.loads((output / "run_summary.json").read_text())
            self.assertEqual(summary["fusion_stats"]["accepted"], 2)
            self.assertEqual(summary["written_count"], 2)
            self.assertEqual(set(summary["fusion_stats"]["by_operation"]), {"sibling", "control_flow"})
            self.assertEqual(len(json.loads((output / "report.json").read_text())["mutations"]), 2)


if __name__ == "__main__":
    unittest.main()
