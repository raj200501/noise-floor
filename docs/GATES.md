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
coverage (covered lines / statements) is at least 85%.

## G2 Static quality
`ruff check .`, `ruff format --check .`, `mypy --strict src/evalstats src/noisefloor`. Ruff may
not exclude `src`; no mypy override may target the packages; no `# mypy: ignore-errors`; no
file-level `# type: ignore` in `src`; every `type: ignore` in src, tests, and tools is followed by
a `#` reason.

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
- `artifacts/e0/coverage.json`: `{"prereg": "prereg/NNNN-....md", "cells": [{"cell", "method",
  "interval_type", "nominal", "replicates", "covered"}]}`.
- The prereg holds a ` ```json e0-grid ` block listing the same cells (`cell`, `method`,
  `interval_type`, `nominal`, `replicates`), committed strictly before the artifact.
- Replicates at least 2,000 per cell, or 500 when `interval_type` is a resampling or GLMM type.
- Rule: coverage >= nominal - 0.015 - 2 * sqrt(c(1 - c)/R), with c the observed coverage.
- `artifacts/headline_cells.json`: `{"cells": [...]}` naming every cell used by an E1 or E2
  headline interval; all must pass. Every failing cell must be in
  `evalstats.validity.UNSUPPORTED_CELLS` (the library warns on those).
- `artifacts/e0/crosscheck.json`: `{"checks": [{"name", "value", "reference", "tolerance",
  "reference_impl"}]}`, at least one against statsmodels, each within its stated tolerance.
- Property-based suite: `pytest -m property` runs at least one test and all pass.

## G6 Orchestrator validity
`artifacts/g6/recovery.json`: `{"prereg", "nominal", "datasets": [{"seed", "components":
{name: {"planted", "estimate", "lo", "hi"}}}], "bias": {name: value}}`. At least 200 datasets;
seeds equal the prereg's ` ```json g6-seeds ` list; each component's coverage meets the G5 rule;
reported bias equals the recomputed mean of estimate minus planted.

## G7 Factorial study (E1)
- Prereg ` ```json e1-design ` block: `{"factors": {name: [levels]}, "models": [{"id", "family",
  "tier", "backend"}], ...}` with at least 4 non-model factors.
- `artifacts/e1/manifest.json`: `{"prereg", "files": {path: sha256}}`, hashes must match.
- `artifacts/e1/runs.jsonl`: one row per task run with `run_id`, `prereg`, `model_id`, `day`,
  `factors`, `outcome` (binary), `confirmatory`. At least 1,000 confirmatory runs, at least 150
  with a `backend: claude` model, at least 3 models over at least 2 families, at least 3 days, all
  factor levels inside the design.
- `artifacts/e1/variance_components.json` (`components`: `estimate`, `lo`, `hi`) and
  `artifacts/e1/mde.json` (`rows`: `mde`, `lo`, `hi`), all finite with lo <= estimate <= hi.
- Full mode: `python -m noisefloor.analysis.e1 --verify` recomputes the tables.

## G8 Leaderboards and bounds (E2, E3)
At least 2 `artifacts/e2/<board>/manifest.json` with `source_url`, `retrieved`, `license`,
`snapshot`, `sha256` (snapshot hash checked unless the snapshot is under `private/`, where the
private manifest covers it); `docs/leaderboards/<board>.md` for each; `artifacts/e3/bounds.json`
`entries` with `source` covering `e1` and every board. Full mode:
`python -m noisefloor.reanalysis --verify`.

## G9 Library shipped
`uv build --wheel` gives one `evalstats-*.whl`; it installs into a clean venv, imports, and
`evalstats --help` works; at least 5 `docs/examples/*.py` run with the wheel's Python and no
network; `docs/api/` is generated (says so) and names everything in `evalstats.__all__`.

## G10 Claim integrity
README.md and docs/REPORT.md exist; the `claimcheck` and `freshness` make targets name both;
both targets pass. They read `artifacts/claims.json` and `docs/claims_exempt.toml` (docs/INTEGRITY.md).

## G11 Preregistration integrity
At least one `prereg/NNNN-*.md`; each touched by exactly one commit; every artifact JSON with a
`prereg` field has that prereg committed strictly before the artifact's first commit;
`make prereg-verify` passes (hash chain in `prereg/CHAIN.jsonl`; docs/INTEGRITY.md).

## G12 Independent review
`artifacts/review/latest.json`: verdict SHIP, no blocking issues, reviewed commit equal to HEAD or
an ancestor differing only in `artifacts/review/`, `STATUS.md`, `docs/LOG.md`; non-blocking ids
in docs/REVIEW_LOG.md; `tools/review.py --validate` ends with `REVIEW: SHIP`.

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
(`disk_floor_gb` may only rise); `make secrets-scan` clean; `make budget` passes and prints
`cash_usd = 0`.
