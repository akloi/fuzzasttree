# Shannon Anomaly-Pattern Matching and Fusion Mutation (astshannon)

Self-contained mutation engine over the `ast_shannon` anomaly-pattern library
(5028 patterns extracted from 8635 javac bug reports: 3141 Multi-Node Synergy,
1887 Single-Point; typed-relation edge signatures, extraction roles, KL/IDF
feature weights, DIE guardrails). No modules are imported from sibling
packages; everything — javac AST worker, parser, relation detectors, matcher,
fusion planner, pipelines, model adapters, CLI, tests — lives in this folder.

The engine supports all three mutation granularities:

| mode | granularity | what it does |
| --- | --- | --- |
| `single` | single-node single-anomaly | transplant one pattern's mechanism at its best host anchor |
| `same`   | single-node multi-anomaly  | fuse two patterns at the *same* anchor (wrapper / role_extension / nested_expression / constraint_conjunction / branch_splitting / sibling) |
| `cross`  | multi-node multi-anomaly  | fuse two patterns at *two interacting* anchors (sibling / control_flow / ancestor_descendant / semantic_flow over host relations) |

---

## 1. The matching algorithm: Inverse-Generative Matching (IGM)

A pattern in this library is not a code template — it is the *output of an
extraction pipeline* (P1 sink location → P2 dependency closure → P3 DIE
guardrails → P4 information weighting). The matcher is that pipeline run in
reverse against a host program, so pattern and host always live in **one
coordinate system**: javac `Tree.Kind`s, the same 18 typed relations, the same
KL/IDF weights, the same extraction roles.

| extraction phase (library) | matching stage (this package) |
| --- | --- |
| P1 sink location (+ Shannon oversize refinement) | anchor proposal from core-kind occurrences, lifted to statement anchors, Shannon-refined when oversized |
| P2 dependency closure | region alignment: the *same* closure operator run on the host around the anchor |
| P3 DIE guardrails | hard feasibility gate (enclosing-declaration containment, span limits) applied at *proposal* time |
| P4 KL/IDF information weights | IDF/KL-weighted feature recall inside the composite score |

**Why this matters (the paper claim).** Because region construction is the
extractor's own closure operator, a correct match reproduces the pattern's
signature *exactly, not approximately*. Verified on the library itself: for
JDK-8270151, running the host-side closure on the bug's own
`original_code` reproduces the extracted pattern's kind-role multiset, its
edge multiset (`DEF_USE:3, IMPLEMENTS:1, SWITCH_EXHAUSTIVENESS:1`), and its
feature set byte-for-byte. Matching is therefore a *fixed-point search* —
score components measure distance-to-fixed-point along independent axes
(syntax, typed relations, information weights), never ad-hoc code similarity.

### 1.1 Two-signal prefilter

Candidate pool = kind-inverted-index ∪ edge-inverted-index, ranked by

```
prefilter(p) = 0.5 · kind-IDF-coverage(role-weighted)      (signal 1)
             + 0.5 · edge-IDF-recall(multiplicity-aware)   (signal 2)
```

Edge-less Single-Point patterns are neutral (0.5) on signal 2, so they
survive on kinds alone, while Multi-Node patterns whose exact relation
cocktail is present outrank generic kind-matches. A single-signal prefilter
lets generic `{CLASS, VARIABLE}` patterns crowd specific ones out of the
candidate window — measured prefilter recall rose from 0.34 to 0.78 when the
second signal was added (0.83 at `--prefilter-limit 800`).

### 1.2 Composite score

For each surviving pattern, up to `anchor_candidates` feasible anchors are
probed and the best feasible region kept:

```
S(p, a) = Gate(p,a) · Q(p) · ( 0.15·Cov + 0.45·Str + 0.40·Sem )
  Cov  = prefilter coverage                       (signals 1+2 above)
  Str  = role-weighted kind-multiset Jaccard       (structural alignment)
  Sem  = 0.65 · edge-IDF multiset Jaccard          (typed-relation cocktail)
       + 0.35 · squashed-KL feature recall         (information-theoretic)
  Q(p) = quality factor: Multi-Node synergy, top KL feature weight
  Gate = DIE feasibility (hard)
```

The typed-relation cocktail carries the largest single semantic weight
because it is what distinguishes a Multi-Node Synergy pattern from a bag of
kinds: rare relations (`SWITCH_EXHAUSTIVENESS`, `RECORD_DECONSTRUCTION`, …)
carry almost all of an anomaly's identity; `DEF_USE` is common and cheap.

### 1.3 Feasibility is a proposal gate, not a post-filter

`die_feasible` (the pattern's `structural_context` containment requirements)
runs **inside** `propose_regions`: an anchor that can never host the pattern
does not consume one of the few candidate slots. This fixed concrete
retrieval failures where a top-level infeasible anchor masked the feasible
inner anchor (JDK-4039191, JDK-4087323).

---

## 2. Fusion planning (multi-anomaly mutation)

### 2.1 Same-anchor fusion (single-node multi-anomaly)

Per anchor, the top matches are paired; the AST relationship between the two
patterns' primary core kinds decides the operation (`wrapper`,
`role_extension`, `nested_expression`, `constraint_conjunction`,
`branch_splitting`, `sibling`). Both mechanisms must survive inside one
replaced subtree.

### 2.2 Cross-anchor fusion (multi-node multi-anomaly)

The host's *own* relation edges between two anchors become the interaction
graph: `sibling`, `control_flow`, `ancestor_descendant`, plus `semantic_flow`
edges derived from the 18 host relations (edge strength 0.85, the concrete
host relation recorded as `host_relation` in the plan). A plan's score:

```
S(plan) = √(S₁·S₂) · interaction_strength · edge_compatibility
        · mechanism_orthogonality · distance_prior · hosting_feasibility
```

- **mechanism_orthogonality** = 0.5·edge-multiset + 0.3·KL-feature +
  0.2·summary-token distances — fusing two copies of one mechanism is
  pointless, so near-duplicate pairs are demoted (identical ⇒ 0).
- **hosting_feasibility** (slot physics): a pattern with several distinct
  core kinds, or any declaration core kind, can only be realized verbatim at
  a *statement/declaration* slot — only there can supporting declarations be
  grafted next to the sink snippet. Expression slots get ×0.3, not a ban:
  plans are ranked, and the syntax/mechanism gates remain the final judge.
- Selection is decayed-greedy b-matching over run-level usage
  (`--fusion-selector matching`), so one pattern cannot monopolize outputs.
- Failed fusion plans **release their output slots** back to single-pattern
  generation — fusion never reduces total output volume.

---

## 3. Validation gates (what makes an output "written")

Every composed candidate passes, in order:

1. **parse gate** — the mutated file must reparse with no *new* syntax errors;
2. **no-op gate** — every selected anchor must actually change;
3. **mechanism retention** — per fused mechanism, ≥60% of the pattern's core
   kinds must appear fully inside the traced region *and* at least one
   exclusive backbone core kind (non-trivial, not shared with the co-fused
   pattern) must survive; core kinds provided by the enclosing declaration
   context (region ancestors) are credited, symmetric with the DIE gate;
4. **syntactic-role preservation at slot granularity** — the replacement's
   top-level fragments must occupy the same slot role as the host anchor
   (a statement slot may legally host a *sequence* of statements — this is
   how multi-core grafts realize full kind backbones; an expression slot
   must remain exactly one expression);
5. **replacement syntax validity** — no `ERROR`/`ERRONEOUS` nodes inside the
   replaced spans;
6. **DIE guardrail adherence** — enclosing-declaration containment holds in
   the mutated tree.

Symbol resolution is honestly recorded as `not_performed`: the gates are
syntactic + structural-backbone, which is the right floor for a fuzzing
candidate pool.

---

## 4. Zero-model baseline and model mutators

`--strategy historical` is a **deterministic transplant baseline**: every
`CORE_TRIGGER` snippet becomes Java code — statement slots receive all core
snippets as sibling statements (the full kind backbone is realized), expression
slots receive the primary sink snippet; bare declarations are terminated so
the fragment parses. It is the floor every learned mutator must beat, and it
makes the whole pipeline runnable with no model at all.

Model mutators (all snippet-protocol: **the complete host file is never
sent**) adapt those snippets to the local context:

- `--strategy command` — external adapter over a JSON stdin/stdout protocol;
- `--strategy openai` — any OpenAI-compatible endpoint (default DeepSeek);
- `--strategy local` — local Transformers / Qwen adapter (LoRA-aware, OOM
  batch bisection).

The mutation request carries the pattern payload (nodes/edges/
`die_guardrails`/`semantic_features`, hidden underscore keys stripped), the
host region evidence, role semantics, and — for fusion — the interaction edge
plus a bridging context carved from the two anchors' LCA.

---

## 5. Self-retrieval calibration (honest numbers)

Methodology (`selfcheck`): a pattern should retrieve itself when matched
against its own bug's `original_code`. Because many patterns are
*information-theoretically indistinguishable* (identical signatures — the
library has large collision groups), we report both **strict** identity and
**signature-group** metrics (a retrieved pattern with the same signature hash
counts as a hit; misses count as zero; every query stays in the denominator).
Patterns whose `extracted_code` contains duplicated reproducers (557/5028 =
11.1% dataset artifact — no single anchor closure can realize them) are
excluded with `--clean-only`. Node-snippet roundtrip into `original_code` is
otherwise 1.0.

60-sample, seed 7, clean-only, prefilter 200 / top-10:

| subset | strict R@1 | strict R@5 | strict MRR | group R@1 | group MRR |
| --- | --- | --- | --- | --- | --- |
| Multi-Node Synergy | 0.258 | 0.581 | 0.384 | 0.290 | 0.408 |
| Single-Point | ≈0 | ≈0 | ≈0 | ≈0 | ≈0 |

At `--prefilter-limit 800`: Multi-Node MRR rises to 0.460 (R@1 0.286,
R@5 0.643), prefilter recall 0.83 — retrieval is prefilter-bound, not
scoring-bound.

**Single-Point ≈ 0 is the expected result, not a failure.** Hundreds of
Single-Point patterns share one signature (e.g. every "one suspicious
cast/literal" pattern), so *identity* retrieval is impossible inside a
collision group by construction. For these patterns the matcher's job — and
its measured behavior — is to find *viable transplant sites*: anchors whose
slot role, DIE context, and kind environment can host the mechanism. The
same analysis drives the composite score: `specificity_factor` caps
edge-less patterns at 0.68 so they cannot out-rank synergy patterns on
generality alone.

---

## 6. End-to-end results (historical baseline)

Seeds: two hand-written javac-shaped hosts (lambda, cast, instanceof, switch,
generics). 28/28 unit tests pass (`python3 -m unittest tests.test_framework`).

| run | mode | planned | attempted | accepted | mech-retention |
| --- | --- | --- | --- | --- | --- |
| SeedOne | single | – | 10 | 5 | gate-enforced |
| SeedTwo | single | – | 12 | 4-5 | gate-enforced |
| SeedTwo | same | 8 | 8 | 4 | 0.5 |
| SeedOne | cross | 9 | 9 | 6 | **1.0** |
| SeedTwo | cross | 6 | 6 | 3 | 0.5 |
| SeedOne | both | – | – | 4 (4 cross + 4 single) | **1.0** |
| SeedTwo | both | – | – | 4 (2 cross + 2 same + 4 single) | 0.5 |

Accepted cross-node plans all carry `hosting_feasibility = 1.0` (statement
slots, host relation `DEF_USE`); rejects are honest mechanism-backbone losses
(e.g. a `TYPE_PARAMETER` core that verbatim grafting could not realize) —
exactly the cases a model mutator is expected to recover by adapting
snippets instead of transplanting them verbatim.

---

## 7. Commands

Run from the `ast_mutation` directory (package imports as `astshannon`):

```bash
# library statistics
python3 -m astshannon.cli stats

# rank patterns for one host file (no mutation)
python3 -m astshannon.cli match path/to/Seed.java --top-k 20

# self-retrieval calibration
python3 -m astshannon.cli selfcheck --sample 60 --seed 7 --clean-only

# single-node single-anomaly mutation (zero-model baseline)
python3 -m astshannon.cli mutate path/to/seeds/ --strategy historical --output-dir runs/single

# single-node multi-anomaly (same-anchor fusion)
python3 -m astshannon.cli mutate path/to/Seed.java --strategy historical \
  --fusion same --fusion-budget 0.5 --output-dir runs/same

# multi-node multi-anomaly (cross-anchor fusion)
python3 -m astshannon.cli mutate path/to/Seed.java --strategy historical \
  --fusion cross --fusion-budget 1.0 --max-outputs 6 \
  --top-k 12 --anchor-candidates 4 --output-dir runs/cross

# model-backed mutation (snippet protocol; host file never sent whole)
python3 -m astshannon.cli mutate path/to/Seed.java --strategy openai \
  --api-base https://api.deepseek.com/chat/completions --model deepseek-chat
python3 -m astshannon.cli mutate path/to/Seed.java --strategy local \
  --model Qwen/Qwen3-4B-Instruct --adapter-path /path/to/lora
```

Each run writes `run_summary.json` (per-input written/rejected records,
fusion stats, anchor interaction graph), `report.json`, and per mutation:
`codes/` (composed candidate), `mutations/<id>/` (fusion_plan.json /
match.json / provenance.json / prompt + raw model output / parse_validation.json),
and `summaries/`.

## 8. Layout

```
astshannon/
  java/astshannon/JavacAstWorker.java  javac -Xjcov AST dump worker
  parser.py        JavaSyntaxTree: spans, snippets, ancestors, kinds, occurrences
  relations.py     host-side 18 typed-relation detectors + region closure + entropy
  library.py       merged-library index: kinds, edges, IDF/KL weights, signatures
  models.py        ShannonPattern, HostRegion, MatchResult, MutationRequest, FusionPlan
  matcher.py       IGM: two-signal prefilter, anchor proposal, composite scoring
  fusion.py        anchor interaction graph, same/cross plans, orthogonality, hosting
  mutator.py       historical baseline, command/openai adapters, byte-accurate composition
  pipeline.py      three-mode orchestration + six validation gates
  local_generator.py / local_model_adapter.py   local model inference + prompts
  evaluation.py    self-retrieval calibration (strict + signature-group metrics)
  cli.py           stats / match / evaluate / selfcheck / mutate
../tests/test_framework.py             28 tests over all layers
```
