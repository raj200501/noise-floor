goal_turn: 3
STATE: ACTIVE

# Status

Updated 2026-11-04, goal_turn 3 (Loop Protocol, continuing after Raj's "continue"). History is in
docs/LOG.md.

## Gates

Fast ship-check (does not count) at 90b4028: 4/14 (G2, G4, G11, G14 pass).

| Gate | Status | Evidence / next step |
|------|--------|----------------------|
| G1 Build and tests | FAIL | full suite passes in this checkout; `make test` fails in the fresh clone (prepush run diagnosing) |
| G2 Static quality | PASS | ruff, ruff format, mypy --strict clean |
| G3 CI parity | FAIL | follows G1 |
| G4 Offline demo | PASS | 6 s in a network namespace, matches golden |
| G5 Library validity | FAIL | checker parsed a uv warning as JSON (fixed); E0 part 2 (GLMM cells like E1) to preregister and run |
| G6 Orchestrator validity | FAIL | recovery simulator and bootstrap intervals built; exploratory method pilot running; prereg next |
| G7 Factorial study | FAIL | E1 design module built; screening s3 on suite v2 running; E1 prereg after s3; Claude arm needs Raj |
| G8 Leaderboards and bounds | FAIL | E2 and both confidence reports done; E3 still needs E1 |
| G9 Library shipped | FAIL | API reference lacked `__version__` (fixed in the working tree) |
| G10 Claim integrity | FAIL | README and docs/REPORT.md not written yet |
| G11 Preregistration integrity | PASS | 3 preregistrations frozen |
| G12 Independent review | FAIL | first review after E1 is preregistered |
| G13 Documentation | FAIL | README and REPORT at the end |
| G14 Hygiene | PASS | secrets-scan clean; cash_usd = 0 |

## Now

goal_turn 3: Claude arm built and tested with a fake CLI (D-019); the adversarial suite review
fixed into nf-tools suite v2 (D-021, harness parity tests); SWE-bench adapter contract tests
(D-020); parametric-bootstrap intervals in `evalstats.decompose` (profile intervals took 8 minutes
on an E1-sized table and failed); G6 recovery simulator; E1 design module. Running: screening s3
(100 runs, suite v2) and an exploratory interval-method pilot for the E1 decomposition.

## Next three actions

1. Finish the method pilot (with and without a batch term), record the analysis choice, then
   preregister E1 (design, power, analysis), E0 part 2 (GLMM cells), and G6 (seeds) together.
2. Analyze s3: pick at least 3 local models over at least 2 families with success 0.2 to 0.8 and
   the difficulty mix; run E0 part 2 and G6 before E1 so E1 runs never share the CPU.
3. Run the E1 local arm over at least 3 days; write the E1 analysis module and E3 bounds on E1.

## Usage to date

cash_usd = 0. cap_share = 0.2 (D-007). Claude subject runs 0, single calls 0, review runs 0.
Local model runs: pilot p1 24, screening s2 11 (exploratory, suite v1), s3 in progress.

## Needs Raj

1. Claude subject arm (G7 needs at least 150 Claude-tier runs): the billing guard blocks Claude
   children in this hosted session by design. When E1 is preregistered, run the arm on a machine
   where `make doctor` shows the guard clean, following docs/PLAN.md ("Claude arm on Raj's
   machine"), starting with `python -m noisefloor.claude_arm smoke`. At `CAP_SHARE = 0.2` the
   daily cap is 6 runs, so 150 runs take about 25 days; at `CAP_SHARE = 1`, about 5 days.
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
