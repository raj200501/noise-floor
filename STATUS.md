goal_turn: 1
STATE: ACTIVE

# Status

Updated 2026-10-23, goal_turn 1 (Loop Protocol).

## Gates

Fast ship-check (does not count) at f0c00dd, 3/14:

| Gate | Status | Evidence |
|------|--------|----------|
| G1 Build and tests | PASS | setup ok; 18 tests pass; line coverage 95.8% |
| G2 Static quality | PASS | ruff check, ruff format, mypy --strict clean |
| G3 CI parity | FAIL | ci.yml in place; `make ci` fails at `demo` |
| G4 Offline demo | FAIL | no demo or golden output yet |
| G5 Library validity | FAIL | E0 not built |
| G6 Orchestrator validity | FAIL | not built |
| G7 Factorial study | FAIL | not started |
| G8 Leaderboards and bounds | FAIL | not started |
| G9 Library shipped | FAIL | wheel builds and installs; no examples or API reference |
| G10 Claim integrity | FAIL | no REPORT.md; claimcheck and freshness not built |
| G11 Preregistration integrity | FAIL | no preregistrations; prereg-verify not built |
| G12 Independent review | FAIL | no review yet |
| G13 Documentation | FAIL | README.md is still the brief |
| G14 Hygiene | PASS | secrets-scan clean; cash_usd = 0; private/ untracked |

## Now

Bootstrap done (docs/BUILD_BRIEF.md section 10.1). Next iteration: M1, evalstats core (critical path for G1 to G4).

## Next three actions

1. M1: evalstats long-format schema, generic CSV/JSONL reader, paired bootstrap, cluster-robust
   SEs, analytic power and MDE, CLI `compare` and `power`, tests to 85% (G1, G2, G3).
2. M2: offline demo on labeled synthetic fixtures with golden output (G4, tag v0.1).
3. M3: prereg chain, claimcheck, freshness (port from raise-v1 with attribution) (G10, G11).

## Environment (make doctor, 2026-10-23)

- Linux x86_64 container (Claude Code on the web), 4 CPUs, 15.7 GiB RAM, about 30 GiB free disk
  (floor 8 GiB), no GPU. Python 3.13 system; 3.11 and 3.12 available; uv 0.11.
- Docker binary present, daemon not running: Docker suites unusable, Docker-free fallback.
- No R or lme4: GLMMs need a named Python backend (or an R install, to try).
- No Ollama or llama.cpp: local model backend unusable until a runtime and models are installed.
- OS sandbox: `unshare -rn` works (network namespace); write confinement still to build (D-006).
- `claude` CLI 2.1.309 present, but the billing guard is tripped by `ANTHROPIC_BASE_URL` and
  `CLAUDE_CODE_USE_CCR_V2` (set by the hosting environment): Claude backend unusable here (D-004).
- Free tiers: not confirmed (`FREE_TIER_KEYS_CONFIRMED` unset), ignored.
- Backends usable now: none for model runs. Fallbacks: subagent reviews, local models after install.

## Usage to date

cash_usd = 0. cap_share = 0.2 (D-007). Claude subject runs 0, single calls 0, review runs 0, local model runs 0.

## Needs Raj

1. Claude subject arm (G7 needs at least 150 Claude-tier runs): the billing guard blocks Claude
   children in this hosted session. Action: when E1 is preregistered, run the Claude arm on a
   machine where `make doctor` shows `billing guard clean` and the `claude` CLI is logged in to
   the Max plan (for example your laptop), following the run instructions that will be in
   docs/PLAN.md. Not blocking yet.
2. Optional: add `apt-get install -y r-base-core r-base-dev r-cran-lme4` to this environment's
   setup script so R and lme4 survive container restarts (D-008). Not blocking.

## Blocked

None.

## Backlog (suggestions for Raj, not ladder rungs)

1. A `evalstats compare --from-inspect <log dir>` one-liner so an evals engineer gets a paired
   interval from existing Inspect logs without writing any code.
