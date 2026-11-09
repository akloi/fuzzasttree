# Run evidence (qwen35_9b, seed-compilable inputs)

Companion evidence for the run stored in `runs/qwen35_9b_seedcompilable/`
(not committed: 2.5G of generated cases and per-case artifacts). Kept here are
the two verification bundles that back the crash claims, plus the aggregate
numbers below.

## Run summary

- Model: `qwen35_9b` served in-process via Transformers; 24h generation on
  4347 seed-compilable inputs (the same seed set as the astsubtree 2026-11-02
  comparison run).
- Pattern library: 5028 patterns (3141 Multi-Node Synergy / 1887 Single-Point).
- Written cases: 16301 (13224 `single_pattern` / 1758 `same_node` / 1319
  `cross_node`); fusion: 27351 attempted, 3077 accepted (mechanism-retention
  0.714 over gate-checked outputs).
- Detection (`--batch-mode mtime`, jacoco LINE coverage on jdk.compiler):
  29932 covered lines over 26h of detection; 5 true crashes (exit 3/4 with an
  exception stack) — 1 `single_pattern`, 4 `same_node`.
- Crash rate per compiled case 0.0823% vs 0.0103% for the astsubtree
  comparison run on the same seeds (8x); accepted `same_node` fusion alone:
  4/538 = 0.74%.

## `origin_mutation_verification/`

For each of the 5 crashes: `original/` recompiles the seed file (exit 0, no
exception — the "zero" state), `mutated/` reproduces the crash (exit 3/4 with
stack). `results.json` records exit codes, exceptions and
`original_matches_current_seed` per case. All 5 are zero-to-one samples: the
seed compiles clean, the mutation crashes javac itself.

## `origin_mutation_verification/historical_comparison/`

Reproduction of the referenced historical bug patterns, plus the ablations
that localize the one drifting crash:

- `history_8333766` → StackOverflowError, stack identical to the fusion crash;
- `history_8267220` → AssertionError in `Types.isCastable`, identical stack;
- `history_8066541` → does not crash under this build (honest negative control);
- `npe_*` / `minimal_*` ablations: flipping the lambda-target type
  (boolean vs Double) flips NPE ↔ AssertionError, localizing the JDK-4202833
  NPE crash's drift from its referenced pattern to that single change.

`results.json` carries the per-case verdicts.
