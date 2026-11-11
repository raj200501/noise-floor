goal_turn: 4
STATE: ACTIVE

# Status

Updated 2026-11-04, goal_turn 3 (Loop Protocol, continuing after Raj's "continue"). History is in
docs/LOG.md.

## Gates

Last fast ship-check (does not count) at c39565e: 7/14 (G2 failed on two test-only `type: ignore` lines, fixed in 22f7de8).

| Gate | Status | Evidence / next step |
|------|--------|----------------------|
| G1 Build and tests | PASS | fresh clone: setup ok, tests pass, line coverage above 85% |
| G2 Static quality | PASS | ruff, ruff format, mypy --strict clean; every `type: ignore` has a reason |
| G3 CI parity | PASS | ci.yml runs `make ci` on 3.11 and 3.12; `make prepush` passes before every push |
| G4 Offline demo | PASS | runs in a network namespace in seconds, matches golden |
| G5 Library validity | PASS (part 1) | E0 part 2 (0005, frozen) cells must run and pass |
| G6 Orchestrator validity | FAIL | 0006 frozen; 200 datasets to run (`runs/e0g6/driver.sh`) |
| G7 Factorial study | FAIL | 0004 frozen (85 local tasks, 4 days); local arm after G6 and E0 part 2; Claude slice needs Raj |
| G8 Leaderboards and bounds | FAIL | E2 and both reports done; E3 on E1 comes with the E1 analysis |
| G9 Library shipped | PASS | wheel installs in a clean venv; 5 examples run; API reference generated |
| G10 Claim integrity | FAIL | README passes claimcheck and freshness; docs/REPORT.md in progress |
| G11 Preregistration integrity | PASS | 6 preregistrations frozen, artifacts predated by them |
| G12 Independent review | FAIL | `make review` after E1 |
| G13 Documentation | FAIL | README written (section 11); REPORT in progress; 24 references verified |
| G14 Hygiene | PASS | secrets-scan clean; cash_usd = 0, caps respected; tree clean |

## Now

goal_turn 4: preregistrations 0004 (E1), 0005 (E0 part 2), and 0006 (G6) are frozen (4907a03),
after three review rounds and a verification of the final fixes (D-032 to D-034). README.md now
presents the project (every number claim-checked against its artifact), `make reproduce`
recomputes E2 and E3 from the pinned snapshots, and docs/REPORT.md is being written. No
confirmatory E0 part 2, G6, or E1 data exist yet.

## Next three actions

1. Run G6 and E0 part 2 (`runs/e0g6/driver.sh`, a little over 40 hours on 4 workers), commit and
   push raw rows at check-ins, bank both, run `python -m noisefloor.e0_validity` and commit it.
2. Launch the E1 local arm at 00:00 UTC (`python -m noisefloor.e1 local
   prereg/0004-e1-factorial.md`, at least 4 UTC days, with check-ins); ask Raj to start the
   Claude slice on his machine (docs/PLAN.md).
3. After both arms end: the E1 analysis, the post-E1 check, E3 on E1, the report's E1 sections,
   `make review`, and the final sequence (brief section 10.6).

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
