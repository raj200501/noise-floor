# Gate contracts

`tools/ship_check.py` implements G1 to G14 of docs/BUILD_BRIEF.md section 6. This file records
the artifact contracts it reads and the interpretations it applies (each also in DECISIONS.md).
A gate is never passed by weakening these checks; a contract may only become stricter.

## Where checks run

- Original checkout: working tree clean, nothing under `private/` tracked, and every entry of
  `artifacts/private_manifest.json` (`{"files": {"private/...": "<sha256>"}}`) matches on disk.
- Everything else runs in a fresh clone of HEAD in a temporary directory. Sub-commands get an
  environment with tokens and API keys removed. Nothing is written inside the working tree.

## G1 Build and tests
`make setup`, then `make test` with pytest's junit and JSON coverage reports forced through
`PYTEST_ADDOPTS`. Needs: tests ran and none failed; `[tool.coverage.run] source` includes both
packages; every tracked module in both packages appears in the coverage report unless listed in
`omit`; each omitted module `x.py` has `tests/contract/test_x.py` using recorded fixtures; line
coverage (covered lines / statements) is at least 85%. No test is skipped for want of R, lme4, or
rpy2: the GLMM checks the preregistrations rely on are among those tests (D-039).

## G2 Static quality
`ruff check .`, `ruff format --check .`, `mypy --strict src/evalstats src/noisefloor`. Ruff may
not exclude `src`; no mypy override may target the packages; no `# mypy: ignore-errors`; no
file-level `# type: ignore` in `src`; every `type: ignore` in src, tests, and tools is followed by
a `#` reason; no inline `mypy:` setting comment anywhere in them (D-039).

## G3 CI parity
`.github/workflows/ci.yml` runs `make ci` with a matrix naming 3.11 and 3.12; the Makefile `ci`
target covers lint, typecheck, test, and demo; `make ci` passes in the clone (fast mode infers it
from the component results).

## G4 Offline demo
`make -s demo` under a network namespace (`unshare -rn`) when the OS allows it, otherwise with
dead proxies (evidence says `env-only`), plus `UV_OFFLINE=1`, `NOISEFLOOR_OFFLINE=1`,
`NOISEFLOOR_NO_MODELS=1`. Under 180 s, the usage ledger unchanged, stdout byte-identical to
`tests/golden/demo.txt`.

## G5 Library validity (E0)
- Every preregistration in `prereg/CHAIN.jsonl` with a ` ```json e0-grid ` block is checked: one
  base grid, and amendments of it (front matter `amends: NNNN`). The base grid's cells are banked
  in `artifacts/e0/coverage.json`, each amendment's in its own `artifacts/e0/coverage-NNNN.json`
  (D-026): `{"prereg": "prereg/NNNN-....md", "cells": [{"cell", "method", "interval_type",
  "nominal", "replicates", "covered"}]}`. Each artifact cites its own grid, lists exactly that
  grid's cells, and its first commit comes strictly after the grid's (D-017).
- `artifacts/e0/raw/<cell>.jsonl` holds one line per replicate (`i`, `truth`, `lo`, `hi`,
  `estimate`, `hit`); G5 recounts each cell's coverage from it and requires the banked count to
  match. A replicate with a missing bound counts as not covered. The line counted for an index
  is the last one written for it; for a grouped cell it carries `e0_glmm_seed(group, i)`, and for
  an artifact banked with stamps it names that artifact's environment and analysis code from a
  clean `src/` (D-039). Limitation: the three simultaneous rank cells (`R-*-sim`) record the
  simulator's own joint verdict per replicate (`hit`, with no truth or interval), so G5 counts
  those verdicts rather than recounting them from ranks (D-040).
- Replicates at least 2,000 per cell, or 500 when `interval_type` is a bootstrap (`bootstrap`,
  `bca`, `bootstrap_max_t`, `parametric_bootstrap`) or GLMM (`glmm`, `glmm_profile`,
  `profile_glmm`, `glmm_task_jackknife`, `glmm_plugin_share`; D-033) type.
- Rule: coverage >= nominal - 0.015 - 2 * sqrt(c(1 - c)/R), with c the observed coverage.
- Headline cells come from the frozen preregistrations: the base grid's "Headline cells (...)"
  list and every amendment's ` ```json headline ` block, less the cells an amendment frozen
  before the first confirmatory E1 run lists in a ` ```json retired-headline ` block, which it
  may do only beside new headline cells of the same kinds (its ` ```json headline-kinds `) that
  test another method (another `method` or `interval_type`; for a GLMM kind, another rule in the
  amendment's own ` ```json recovery-config ` settings: `boot_interval`; the fixed rule, the task
  jackknife or the bootstrap box with its `box_interval`, which has no effect under the
  jackknife; `share_interval`; `mde_interval`), and only when a frozen amendment of the E1
  preregistration (0004) from before that run has a ` ```json analysis-code ` block (D-034
  items 13 and 16). All
  must pass, and all must appear in `artifacts/headline_cells.json` (`{"cells": [...]}`, written
  by `python -m noisefloor.e0_validity`). Every failing cell must be in
  `evalstats.validity.UNSUPPORTED_CELLS`, and `evalstats.validity.check` with the cell's own
  function (`noisefloor.e0_validity.FUNCTIONS`), interval type, and task count must name it (the
  library warns on those).
- `artifacts/e0/crosscheck.json`: `{"checks": [{"name", "value", "reference", "tolerance",
  "reference_impl"}]}`, at least one against statsmodels, each within its stated tolerance.
- Property-based suite: `pytest -m property` runs at least one test and all pass.

## G6 Orchestrator validity
`artifacts/g6/recovery.json`: `{"prereg", "nominal", "config", "environment", "analysis_code", "datasets":
[{"seed", "components": {name: {"planted", "estimate", "lo", "hi"}}, "error"}], "bias": {name:
value}}`. At least 200 datasets; seeds equal the prereg's ` ```json g6-seeds ` list; `config`
equals its ` ```json recovery-config ` block and `nominal` that block's level; the components are
exactly its ` ```json g6-components ` list, and each one's coverage meets the G5 rule (a dataset
without an interval, such as a failed fit, counts as a miss); `environment.env` names the one
environment every row ran in (`noisefloor.provenance`), and `analysis_code` the one 16-hex
analysis-code hash, which prereg 0004's analysis requires and which must equal the
`analysis_code` of `artifacts/e0/coverage-0005.json` once E0 part 2 is banked (for a rerun of G6
under an amendment of its preregistration, a replaced GLMM method, the hash that a frozen
amendment of 0004 allows in its ` ```json analysis-code ` block); reported bias
equals the recomputed mean of estimate minus planted. Each banked dataset equals its row in
`artifacts/g6/raw.jsonl`; the raw rows are exactly the preregistered seeds, each stamped with the
artifact's environment and analysis code from a clean `src/`; and every dataset's planted value
for a component the recovery-config fixes (the task, model-by-task, and model variances, each
two-level factor's contribution, effect squared over four) is that value (D-039).

## G7 Factorial study (E1)
- Prereg ` ```json e1-design ` block: `{"factors": {name: [levels]}, "models": [{"id", "family",
  "tier", "backend"}], ...}` with at least 4 non-model factors.
- `artifacts/e1/manifest.json`: `{"prereg", "files": {path: sha256}}`, hashes must match.
- `artifacts/e1/runs.jsonl`: one row per task run with `run_id`, `prereg`, `prereg_sha256`,
  `started_at`, `model_id`, `day`, `factors`, `outcome` (binary), `status`, `confirmatory`. Every
  confirmatory row cites the E1 prereg with the SHA-256 of its chain entry and started no earlier
  than that entry's `frozen_at`. Counted task runs are the confirmatory runs that ran and that
  the primary analyses use (status other than `infra_error`, no `excluded` reason such as a
  Claude run served by another model ID): at least 1,000, at least 150 with a `backend:
  claude` model, at least 3 models over at least 2 families, local-arm runs on at least 3
  days, all factor levels inside the design. A run id that appears more than once fails the gate.
- `artifacts/e1/variance_components.json` (`components`: `estimate`, `lo`, `hi`, `share`,
  `share_lo`, `share_hi`) and `artifacts/e1/mde.json` (`rows`: `mde`, `lo`, `hi`), all finite
  with lo <= estimate <= hi.
- `artifacts/e1/post_check.json`, the post-E1 check of prereg 0004 item 10 (which alone applies
  item 10's labels and item 9's batch rule): it cites the E1 preregistration with its chain
  SHA-256, which predates it, has `synthetic` true, every entry of `components` on
  `datasets` 200, a `not_validated` list, and an `analysis_code` equal to
  `variance_components.json`'s `analysis_code.hash`.
- Full mode: `python -m noisefloor.analysis.e1 --verify` recomputes the tables (four GLMM fits
  with bootstrap and jackknife, so it has its own 6-hour timeout); it refuses analysis code other
  than the code G6 and E0 part 2 ran (their `analysis_code`), unless the manifest names a frozen
  amendment of the E1 preregistration whose ` ```json analysis-code ` block allows it.

## G8 Leaderboards and bounds (E2, E3)
At least 2 `artifacts/e2/<board>/manifest.json` with `source_url`, `retrieved`, `license`,
`snapshot`, `sha256` (snapshot hash checked unless the snapshot is under `private/`, where the
private manifest covers it); `docs/leaderboards/<board>.md` for each; `artifacts/e3/bounds.json`
`entries` with `source` covering every board, each entry equal to its board's `results.json`
q3 for the entry's analysis set plus `source` and `analysis_set` (so the full mode's recompute of
`results.json` covers every E3 number; D-040); and E3 on E1, `artifacts/e1/e3_bounds.json`
(`source` `e1`, non-empty `entries`, a non-empty `rankings` object, and a non-empty `adjacent`
list), written by the E1 analysis
under the E1 preregistration, which must predate it. Full mode: `python -m noisefloor.reanalysis --verify` (E1's file is
recomputed by G7's verify).

## G9 Library shipped
`uv build --wheel` gives one `evalstats-*.whl`; it installs (with its documented `glmm` extra,
which example 04 needs for a GLMM backend) into a clean venv, imports, and
`evalstats --help` works; at least 5 `docs/examples/*.py` run with the wheel's Python and no
network; `docs/api/` is generated (says so) and names everything in `evalstats.__all__`.

## G10 Claim integrity
README.md and docs/REPORT.md exist; the `claimcheck` and `freshness` make targets name both;
both targets pass. They read `artifacts/claims.json` and `docs/claims_exempt.toml` (docs/INTEGRITY.md).
The committed rebuild check of each derived artifact that claims cite passes:
`tools/noise_floor.py --check`, `python -m noisefloor.method_pilots --check`, and
`tools/validity_summary.py --check` (D-039).

## G11 Preregistration integrity
At least one `prereg/NNNN-*.md`; each touched by exactly one commit; every artifact JSON with a
`prereg` field has that prereg committed strictly before the artifact's first commit;
`make prereg-verify` passes (hash chain in `prereg/CHAIN.jsonl`; docs/INTEGRITY.md).

## G12 Independent review
`artifacts/review/latest.json`: verdict SHIP, no blocking issues, reviewed commit equal to HEAD or
an ancestor differing only in `artifacts/review/`, `STATUS.md`, `docs/LOG.md`; non-blocking ids
in docs/REVIEW_LOG.md, each as a whole id; the stdout of `tools/review.py --validate` ends with
`REVIEW: SHIP` (D-039).

## G13 Documentation
README.md is not the brief and follows section 11 (name first, headline SVG, table, interval and
quickstart on the first screen; the required sections); README.md and docs/REPORT.md have no em
dashes or hype words; REPORT.md has 4,000 to 8,000 words outside code blocks and the required
sections, and its figures exist; every external URL in either file (other than this repository)
is in `artifacts/citecheck.json` with `verified: true`; `make citecheck` passes; full mode runs
`make report` and requires the headline SVG to be reproduced byte for byte.

## G14 Hygiene
Original checkout clean and `private/` untracked (see above); LICENSE, CITATION.cff, SECURITY.md,
CONTRIBUTING.md exist; `budget.toml` has `cash_usd = 0` and no cap above the brief's defaults
(`disk_floor_gb` may only rise); `make secrets-scan` clean, run with the sensitive variables
that sub-commands otherwise lose, so it also searches for their literal values (it prints
pattern names only; D-040); `make budget` passes and prints
`cash_usd = 0`. Its turn check reads the `agent_usage` rows on which the client logs a Claude
Code run's turns, and any run above the cap fails it, including one the stream check stopped at
the first message of a turn over the cap (D-037 item 2, made strict by D-039).
