goal_turn: 3
STATE: ACTIVE

# Status

Updated 2026-11-04, goal_turn 3 (Loop Protocol, continuing after Raj's "continue"). History is in
docs/LOG.md.

## Gates

Last fast ship-check (does not count) at dea4149: 8/14.

| Gate | Status | Evidence / next step |
|------|--------|----------------------|
| G1 Build and tests | PASS | fresh clone: setup ok, tests pass, line coverage above 85% |
| G2 Static quality | PASS | ruff, ruff format, mypy --strict clean; every `type: ignore` has a reason |
| G3 CI parity | PASS | ci.yml runs `make ci` on 3.11 and 3.12; `make prepush` passes before every push |
| G4 Offline demo | PASS | runs in a network namespace in seconds, matches golden |
| G5 Library validity | PASS (part 1) | E0 part 2 (0005) cells must run and pass once 0005 is frozen |
| G6 Orchestrator validity | FAIL | 0006 to freeze, then 200 datasets (`runs/e0g6/driver.sh`) |
| G7 Factorial study | FAIL | 0004 in final review (80 local tasks, 4 days); local arm after G6 and E0 part 2; Claude slice needs Raj |
| G8 Leaderboards and bounds | FAIL | E2 and both reports done; E3 on E1 comes with the E1 analysis |
| G9 Library shipped | PASS | wheel installs in a clean venv; 5 examples run; API reference generated |
| G10 Claim integrity | FAIL | README and docs/REPORT.md not written yet |
| G11 Preregistration integrity | PASS | 3 preregistrations frozen, artifacts predated by them |
| G12 Independent review | FAIL | `make review` after E1 |
| G13 Documentation | FAIL | README and REPORT at the end; 24 references verified (citecheck passes) |
| G14 Hygiene | PASS | secrets-scan clean; cash_usd = 0, caps respected; tree clean |

## Now

goal_turn 3: the second review of the 0004 to 0006 drafts (five lenses, adversarially
verified) is answered in code and text (D-033). The power analysis now sets the local sample:
80 tasks (16 per family, the smallest count at which every local harness contrast and model pair
has power 0.8 under the Holm reading), over 4 days; the Claude slice keeps 30 tasks. New: code
pins from G6 and E0 part 2 to the E1 analysis, run-code checks before every batch, server
settings in the pins, a runner lock, per-pair Claude checks (failure streak, contamination,
smoke, per-tier end), cell labels that make every failing cell's function warn, replicates in
the simulator with a test that the G6 and E1 analysis code agree, validated ranges per kind,
and a sampling simulator. A final review of the third drafts is running before the freeze.

## Next three actions

1. Read the final review, fix what it confirms, freeze 0004 to 0006 in one commit,
   `make prepush`, push.
2. Run G6 and E0 part 2 (1,200 GLMM datasets at about 8 core-minutes each, about 40 hours on 4
   workers), commit raw rows at check-ins, bank both; then launch the E1 local arm at 00:00 UTC
   (at least 4 UTC days), with check-ins.
3. Ask Raj to start the Claude slice on his machine once G6 and E0 part 2 are banked
   (docs/PLAN.md); the slice takes about 30 days at 6 runs a day.

## Usage to date

cash_usd = 0. cap_share = 0.2 (D-007). Claude subject runs 0, single calls 0, review runs 0.
Local model runs: pilot p1 24, screening s2 11 (exploratory, suite v1), s3 84 (suite v2; stopped early, D-025).

## Needs Raj

1. Claude subject arm (G7 needs at least 150 Claude-tier runs): the billing guard blocks Claude
   children in this hosted session by design. When E1 is preregistered, run the arm on a machine
   where `make doctor` shows the guard clean, following docs/PLAN.md ("Claude arm on Raj's
   machine"), starting with `python -m noisefloor.claude_arm smoke`. The daily cap is budget.toml's
   cap times the smaller of its `cap_share` (0.2) and `CAP_SHARE`: 6 runs a day, about 30 days
   for the 180 planned runs; only Raj can raise `cap_share` in budget.toml.
2. raise-v1-public: its web page returns a proxy 403 here, so it is dropped from the verified
   references; its integrity tools were ported from a clone with Apache-2.0 attribution.
   Optional: check the reference by hand.
3. Optional: add `apt-get install -y r-base-core r-base-dev r-cran-lme4` to this environment's
   setup script so R and lme4 survive container restarts (D-008).
4. FYI (from goal_turn 1): one batch of citation fetches sent a User-Agent containing the local
   part of your email address to arxiv.org, anthropic.com, and metr.org; fetch tooling now uses
   a generic User-Agent.

## Blocked

None.

## Backlog (suggestions for Raj, not ladder rungs)

1. A `evalstats compare --from-inspect <log dir>` one-liner so an evals engineer gets a paired
   interval from existing Inspect logs without writing any code.
2. `evalstats decompose --bootstrap` on the command line, so a team can get variance shares with
   intervals from its own run logs in one call.
