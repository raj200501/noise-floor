goal_turn: 1
STATE: ACTIVE

# Status

Updated 2026-10-23, goal_turn 1 (Loop Protocol).

## Gates

Fast ship-check (does not count) at 2b82ae3: 5/14 (G1, G2, G3, G4, G14 pass).

| Gate | Status | Evidence / next step |
|------|--------|----------------------|
| G1 Build and tests | PASS | 124 tests; line coverage 99% |
| G2 Static quality | PASS | ruff, ruff format, mypy --strict clean |
| G3 CI parity | PASS | `make ci` passes locally (lint, typecheck, test, demo) |
| G4 Offline demo | PASS | 3 s in a network namespace, matches golden (tag v0.1 local) |
| G5 Library validity | FAIL | library methods built; E0 grid prereg next |
| G6 Orchestrator validity | FAIL | planted-effects simulator built; recovery study after E0 |
| G7 Factorial study | FAIL | local runtime building; suite, wrapper, pilot next |
| G8 Leaderboards and bounds | FAIL | bounds built; leaderboard sources being scouted |
| G9 Library shipped | FAIL | wheel installs; examples and API reference later |
| G10 Claim integrity | FAIL | claimcheck and freshness being built (subagent) |
| G11 Preregistration integrity | FAIL | prereg chain being built (subagent) |
| G12 Independent review | FAIL | first review at the E0 milestone |
| G13 Documentation | FAIL | 7 references verified; README and REPORT at the end |
| G14 Hygiene | PASS | secrets-scan clean; cash_usd = 0 |

## Now

goal_turn 1: evalstats core, GLMM decomposition (lme4 via rpy2), rank-confidence sets, policy
bounds, offline demo, verified references. In flight: integrity tools (subagent), adapters
(subagent), leaderboard scouting (subagent), llama.cpp CPU build.

## Next three actions

1. Subjects for E1: Docker-free tool-use suite, minimal-agent scaffold, client wrapper with caps
   and ledger, sandbox, local GGUF models; pilot on disjoint tasks (0.2 to 0.8 success).
2. Freeze the E0 simulation grid in prereg/ (after the prereg chain lands), then run E0 (G5) and
   the planted-component recovery study (G6) in the background.
3. Wire claimcheck, freshness, prereg-verify and the adapters into make targets and exports.

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

1. raise-v1-public: its web page returns a proxy 403 here ("GitHub access to this repository is
   not enabled for this session"), so it is dropped from the verified references; a git clone
   worked, so its integrity tools were ported with Apache-2.0 attribution (THIRD_PARTY_NOTICES.md).
   Action (optional): check the reference by hand, or enable web access to it for this environment.
2. Claude subject arm (G7 needs at least 150 Claude-tier runs): the billing guard blocks Claude
   children in this hosted session. Action: when E1 is preregistered, run the Claude arm on a
   machine where `make doctor` shows `billing guard clean` and the `claude` CLI is logged in to
   the Max plan (for example your laptop), following the run instructions that will be in
   docs/PLAN.md. Not blocking yet.
3. Optional: add `apt-get install -y r-base-core r-base-dev r-cran-lme4` to this environment's
   setup script so R and lme4 survive container restarts (D-008). Not blocking.
4. FYI: one batch of citation fetches sent a User-Agent containing the local part of your email
   address (not the full address) to arxiv.org, anthropic.com, and metr.org. Later fetches did
   not. Fetch tooling now uses a generic User-Agent.

## Blocked

None.

## Backlog (suggestions for Raj, not ladder rungs)

1. A `evalstats compare --from-inspect <log dir>` one-liner so an evals engineer gets a paired
   interval from existing Inspect logs without writing any code.
