# Historical AST Subtree Matcher

This package uses `astpatterns_minimal` as a historical anomaly library. It performs:

1. native javac `Tree.Kind` retrieval for high recall;
2. minimum local-subtree alignment in the input program;
3. structural reranking against historical anomaly subtrees;
4. snippet-only mutation;
5. byte-accurate composition back into the local source file.

The external mutator never receives the complete input program. It receives only the matched host subtree, the historical anomaly subtree, and the `L/A/P/D` metadata. The composer is the only component that reads the complete local source.

## Data flow

```text
astpatterns_minimal + astwithcontext
       | original_node_id join
       v
javac-native subtree signatures and kind index
       |
input.java -> bundled javac worker -> native-kind prefilter -> ordered subtree alignment
                                                   |
                                                   v
                                       structural score and Top-k
                                                   |
                         host subtree + historical subtree + L/A/P/D
                                                   |
                                                   v
                                         snippet mutation provider
                                                   |
                                                   v
                              byte-span composition into input.java
```

## Commands

Run commands from the `FuzzLLMs-jdk` repository root:

```bash
PYTHONPATH=ast_mutation python -m astsubtree stats
```

Rank historical patterns without producing source code:

```bash
PYTHONPATH=ast_mutation python -m astsubtree match path/to/Test.java \
  --top-k 20 \
  --output /tmp/Test.matches.json
```

Evaluate a labeled query set:

```bash
PYTHONPATH=ast_mutation python -m astsubtree evaluate queries.jsonl \
  --top-k 10 \
  --output ast_mutation/astsubtree/runs/evaluation.json
```

Each JSONL row uses this schema:

```json
{"query_file":"reproducers/Test.java","expected_bug_ids":["JDK-1234567"],"L":"lambda inference","P":["Attr"],"D":"NullPointerException"}
```

Paths may be absolute or relative to the manifest. `L/P/D` are optional. The report contains prefilter recall, Recall@1/5/10, MRR, and per-query rankings.

Use the deterministic historical-subtree transplant baseline:

```bash
PYTHONPATH=ast_mutation python -m astsubtree mutate path/to/Test.java \
  --strategy historical \
  --top-k 20 \
  --max-outputs 5 \
  --output-dir ast_mutation/astsubtree/runs/example
```

Use an external LLM or mutation adapter:

```bash
PYTHONPATH=ast_mutation python -m astsubtree mutate path/to/Test.java \
  --strategy command \
  --mutator-command "python ast_mutation/astsubtree/examples/passthrough_mutator.py" \
  --output-dir ast_mutation/astsubtree/runs/example-command
```

Call an OpenAI-compatible endpoint directly while sending only subtree context:

```bash
export DEEPSEEK_API_KEY=...
PYTHONPATH=ast_mutation python -m astsubtree mutate path/to/Test.java \
  --strategy openai \
  --api-base https://api.deepseek.com/chat/completions \
  --model deepseek-chat \
  --output-dir ast_mutation/astsubtree/runs/example-openai
```

Use `--api-key-env` to select another environment variable. The API request contains no complete input file.

### Local model services

For a local vLLM OpenAI-compatible server:

```bash
vllm serve /path/to/model --host 127.0.0.1 --port 8000

PYTHONPATH=ast_mutation python -m astsubtree mutate path/to/Test.java \
  --strategy openai \
  --api-base http://127.0.0.1:8000/v1/chat/completions \
  --model /path/to/model \
  --no-api-key \
  --output-dir ast_mutation/astsubtree/runs/vllm-local
```

For Ollama's OpenAI-compatible endpoint:

```bash
ollama serve
ollama pull qwen2.5-coder:7b

PYTHONPATH=ast_mutation python -m astsubtree mutate path/to/Test.java \
  --strategy openai \
  --api-base http://127.0.0.1:11434/v1/chat/completions \
  --model qwen2.5-coder:7b \
  --no-api-key \
  --output-dir ast_mutation/astsubtree/runs/ollama-local
```

For a local Transformers, llama.cpp, or custom inference script, use `--strategy command`. The process reads one request JSON from stdin and writes `{"mutated_subtree":"..."}` to stdout. This avoids coupling the framework to one model runtime or GPU layout.

Use the package's self-contained Transformers loader with the in-process local strategy:

```bash
PYTHONPATH=ast_mutation python -m astsubtree mutate path/to/Test.java \
  --strategy local \
  --model /path/to/local/model \
  --adapter-path /path/to/lora \
  --gpu 0 \
  --temperature 0.2 \
  --max-new-tokens 512 \
  --output-dir ast_mutation/astsubtree/runs/local-model
```

The model is loaded once and reused for every selected subtree in that run. Omit `--adapter-path` for a base model. The same adapter can also be used with the generic command protocol, although that mode starts a new process and reloads the model for each request:

```bash
--strategy command \
--mutator-command "python3 ast_mutation/astsubtree/local_model_adapter.py --model /path/to/local/model --gpu 0"
```

A directory can be supplied in place of one Java file. `--glob` controls recursive file selection.

The default library paths are:

```text
/disk1/users/zengj/jdk-fuzzllm/javac_data/javac_data/javaccode_8635/astpatterns_minimal
/disk1/users/zengj/jdk-fuzzllm/javac_data/javac_data/javaccode_8635/astwithcontext
```

Override them with `--library-dir` and `--historical-source-dir`. By default both `candidate` and `candidate_partial` records are indexed. Pass `--full-only` to use only complete-anchor references.

The parser always compiles and runs the included public javac Tree API worker. It does not import `asttreethree`, tree-sitter, or another repository parser. A JDK with `java` and `javac` on `PATH` is required. For reproducible experiments, use the same JDK major version that produced the historical `astwithcontext` files.

At library load time, every minimum-subtree node is joined to `astwithcontext.ast.nodes` through `original_node_id`. The loader verifies the referenced node, parent relation, child index, and root, then rebuilds the node multiset, native parent-child edges, paths, ancestor path, and signature hash from `com.sun.source.tree.Tree.Kind`. Missing or inconsistent records stop loading with an explicit error; the matcher never falls back to canonical-name guessing.

## Matching

Stage 1 computes IDF-weighted native-node coverage. Common nodes contribute less than rare nodes. The highest-scoring references proceed to local alignment.

Stage 2 aligns the restored ordered native subtree under candidate roots. It preserves repeated node counts and prefers the original child positions, then compares:

- required-kind F1;
- node-kind multiset;
- parent-child edge multiset;
- root-to-node paths;
- root kind;
- ancestor context;
- ordered child groups;
- depth and branching shape;
- subtree size.

The reranker also computes language-feature compatibility from historical `L`. Recognized features such as lambdas, method references, checked exceptions, generics, annotations, records, switch constructs, pattern matching, inheritance, arrays, casts, loops, and assignments are checked against the nearest host statement or declaration. The evidence is recorded as `feature_compatibility` in `match.json`.

Without query-side `L/P/D`, the final raw score is `0.15 * prefilter + 0.70 * structure + 0.15 * feature_compatibility`. When query-side context is supplied, it is `0.10 * prefilter + 0.60 * structure + 0.15 * feature_compatibility + 0.15 * semantic_context`. Reference quality and specificity factors are then applied.

The final score also applies reference-quality and specificity factors. Single-node and collision-heavy historical signatures are downweighted. `--language-feature`, repeated `--phase`, and `--diagnostic` add optional runtime/report context to reranking; they are not required.

## Snippet mutation protocol

For `--strategy command`, the adapter reads one JSON object from standard input:

```json
{
  "bug_id": "JDK-4076729",
  "host_subtree": "{ ... }",
  "historical_subtree": "{ ... }",
  "historical_report": {
    "Summary": "Historical javac issue title",
    "Description": "Original issue report and failure details",
    "extracted_code": "complete historical reproducer source"
  },
  "pattern": {
    "L": "for loop with semicolon, variable reassignment",
    "A_raw": ["ForLoopTree", "VariableTree", "AssignmentTree"],
    "A_canonical": ["FOR", "VARIABLE", "ASSIGNMENT"],
    "A_native": ["BLOCK", "FOR_LOOP", "VARIABLE", "EXPRESSION_STATEMENT", "ASSIGNMENT"],
    "P": ["Generate"],
    "D": "Segmentation fault during bytecode verification"
  },
  "match_score": 0.71,
  "instruction": "Return only a mutated replacement for host_subtree..."
}
```

It must write either plain subtree source or:

```json
{"mutated_subtree": "{ ... }"}
```

Do not return imports, surrounding classes, Markdown explanation, or a complete Java file unless the matched host itself is that syntactic construct.

`Summary`, `Description`, and `extracted_code` are joined lazily from the matching `astwithcontext/JDK-*.json` record. They are sent as historical evidence after retrieval; the complete current host file is never sent to the model. The minimal pattern files remain compact and do not duplicate these fields.

## Output

Each run separates directly consumable Java files from complete per-mutation evidence:

```text
run_.../
  codes/
    Test__JDK-4076729__r001.java
  mutations/
    Test__JDK-4076729__r001/
      Original.java
      Mutated.java
      mutated_subtree.javafrag
      prompt.txt
      raw_llm_output.txt
      request.json
      match.json
      parse_validation.json
      provenance.json
  summaries/
    Test.summary.json
  report.json
  run_summary.json
```

`codes/` contains only the composed mutation results. `mutations/` preserves the complete evidence for each result: source before and after mutation, the exact prompt sent to the provider, its unparsed response, the structured request, normalized replacement subtree, match evidence, parser validation, and provenance. For a local chat model, `prompt.txt` includes the tokenizer's applied chat template.

By default, a mutation is rejected if it introduces additional parser errors, covers the entire compilation unit, replaces more than 4000 host bytes, or spans more than 75% of the input file. The corresponding limits can be changed with `--allow-new-syntax-errors`, `--max-host-bytes`, and `--max-host-fraction`.

## Two-pattern fusion

Fusion is opt-in. `--fusion off` keeps the original single-pattern behavior.

```bash
PYTHONPATH=ast_mutation python -m astsubtree mutate path/to/Seed.java \
  --strategy historical \
  --fusion both \
  --fusion-pairs 4 \
  --anchor-limit 6 \
  --anchor-candidates 3 \
  --fusion-budget 0.4 \
  --max-outputs 5 \
  --output-dir ast_mutation/astsubtree/runs/fusion
```

Use `same` for two patterns on one anchor, `cross` for two interacting anchors, or
`both` to rank both kinds together. The same options work with `local`, `openai`,
and `command`. Each model call handles one complete fusion plan.

Same-anchor plans support `wrapper`, `role_extension`, `constraint_conjunction`,
`branch_splitting`, and `nested_expression`. A role table filters candidates;
ranking combines geometric-mean match quality, structural non-conflict,
feature compatibility, phase/edge/diagnostic orthogonality, and operation priors.
The historical baseline uses AST wrapper/argument slots and conditional
expressions, while model providers receive the directed operation and both
historical reports. Templates that lack an appropriate AST slot are rejected.

The phase-one Anchor Interaction Graph records `ancestor_descendant`, `sibling`,
and `control_flow` edges with node IDs, LCA/slot/branch evidence and strengths.
Cross-anchor plans use disjoint replacements and combine interaction strength,
feature compatibility, mechanism orthogonality and scope/distance priors.
Ancestor/descendant subtrees overlap, so their edges are recorded with
`execution_exclusion: overlapping_anchor_spans` and are not scheduled as two-span
edits. Nested transformations use same-anchor wrapper/nesting operations.
Name-based dataflow and attributed type/symbol edges belong to later phases.

The v2 request retains the original first-pattern fields and adds
`host_subtree_1/2`, `pattern_1/2`, `historical_subtree_1/2`,
`historical_report_1/2`, both bug IDs and match scores, `fusion_role`,
`edge_evidence`, and `bridging_context`. The latter is the LCA source snippet,
bounded by twice `--max-host-bytes`; plans requiring a larger bridge or the
complete host file are excluded. Historical reports can contain complete
historical reproducers.

For same-anchor fusion, return:

```json
{"mutated_subtree": "replacement Java subtree"}
```

For cross-anchor fusion, return both snippets in one response:

```json
{"mutated_subtree_1": "first replacement", "mutated_subtree_2": "second replacement"}
```

The composer applies disjoint UTF-8 spans from right to left. Validation reparses
the complete result, checks each replacement's syntactic role, and requires at
least 60% of each pattern's required node kinds within its replacement region.
The retention check first subtracts a fixed set of trivial scaffolding kinds
(`IDENTIFIER`, `BLOCK`, plain method/class declarations, bare invocations,
returns, imports, and the like); what remains is the pattern's backbone. For a
co-fused pattern the backbone minus the other pattern's required kinds must be
represented by at least one node — shared scaffolding alone cannot establish
two-mechanism retention — and the same gate applies to single-pattern outputs
(`mutation loses the historical mechanism backbone`). Every selected anchor must
change. These are syntax/backbone checks; symbol binding, full type checking and
actual bug reproduction are measured downstream.

Anchor selection reports `--anchor-candidates` (default 3) ranked anchors per
pattern instead of only the best one, and applies the `--max-host-bytes` /
`--max-host-fraction` gates inside the matcher, so oversized or whole-file
anchors never consume top-k slots. Requests additionally carry
`enclosing_context`: the signature-only skeleton of the anchor's enclosing type
and method (member signatures, no bodies, bounded to ~1 KB) so adapted
identifiers can resolve without leaking the complete host file.

Each accepted fusion adds `fusion_plan.json`, with both bug traces, both match
signatures, operation/edge evidence, score components, replacement spans and the
composed template. Cross-anchor results save `mutated_subtree_1.javafrag` and
`mutated_subtree_2.javafrag`; prompt, raw output, request, validation, source and
provenance files follow the existing layout. Per-input summaries also preserve
the interaction graph and failed-plan validation details.

Fusion reserves `ceil(max_outputs * fusion_budget)` output slots. Failed fusion
slots return to single-pattern generation, and the total output count remains
bounded by `max_outputs`. `run_summary.json` aggregates `fusion_stats`: plans,
attempts, generated responses, backbone retention, accepted/rejected counts and
operation/kind breakdowns. A missing retention denominator is reported as
`null`, rather than a measured zero rate.

Run the parser, protocol and pipeline tests from the parent repository:

```bash
PYTHONPATH=ast_mutation python -m unittest discover -s ast_mutation/astsubtree/tests -v
```

## Experimental use

For an initial retrieval experiment:

1. use `--full-only` as the primary historical library;
2. evaluate `candidate_partial` in a separate recall experiment;
3. report Recall@1/5/10 and MRR using independently reproduced failures;
4. compare node-only retrieval, structural reranking, and optional `L/P/D` context;
5. report results separately for singleton, chain, and branching references.

Self-matching a historical record against its own extracted source is a smoke test, not a retrieval-quality evaluation.

