goal_turn: 3
STATE: ACTIVE

# Status

Updated 2026-11-04, goal_turn 3 (Loop Protocol, continuing after Raj's "continue"). History is in
docs/LOG.md.

## Gates

Last fast ship-check (does not count) at 90b4028: 4/14. Fixes since then (G1 temp roots, G5
stdout parsing, G9 `__version__`) wait for the next fast check, which runs when no screening run
shares the CPU.

| Gate | Status | Evidence / next step |
|------|--------|----------------------|
| G1 Build and tests | FAIL | fresh-clone `make test` failure traced to concurrent pytest temp roots; fixed, to re-check |
| G2 Static quality | PASS | ruff, ruff format, mypy --strict clean |
| G3 CI parity | FAIL | follows G1 |
| G4 Offline demo | PASS | 6 s in a network namespace, matches golden |
| G5 Library validity | FAIL | E0 part 2 (E1-shaped GLMM, share, MDE, score, and contrast cells) to preregister and run |
| G6 Orchestrator validity | FAIL | recovery runner ready; Laplace method pilot 6 running; prereg 0006 next |
| G7 Factorial study | FAIL | E1 models chosen from s3 (D-025: qwen3-1.7b, qwen3-4b, granite-4.0-micro); prereg in review; Claude arm needs Raj |
| G8 Leaderboards and bounds | FAIL | E2 and both confidence reports done; E3 still needs E1 |
| G9 Library shipped | FAIL | API reference lacked `__version__` (fixed); to re-check |
| G10 Claim integrity | FAIL | README and docs/REPORT.md not written yet |
| G11 Preregistration integrity | PASS | 3 preregistrations frozen |
| G12 Independent review | FAIL | first review after E1 is preregistered |
| G13 Documentation | FAIL | README and REPORT at the end; 24 references verified (citecheck passes) |
| G14 Hygiene | PASS | secrets-scan clean; cash_usd = 0 |

## Now

goal_turn 3: the response to the E1 preregistration review is nearly done. Pilot 7 (banked,
D-029) supports task-jackknife intervals for fixed contributions. Code now covers batch-level
failures and arm stops with a resume step and lock (D-029), the Claude-slice rules (pair order,
turn-cap refusal, CLI and model changes, own ledger, end point; D-030), replicate runs with a
sampling table, harness-effect differences and tier pairs, provenance stamps on simulation rows,
E3 on E1 as its own artifact, the post-E1 check, and per-group planted overrides (D-031). The E1
dry run on pilot seeds is running; `tools/prereg_blocks.py` generates the prereg blocks, and the
second drafts of 0004 to 0006 are assembled and pass the consistency check.

## Next three actions

1. Read the dry run's infrastructure summary, record D-032, run the full tests, commit; then an
   independent review of the assembled 0004 to 0006, fix what it confirms, freeze in one commit,
   `make prepush`, push.
2. Run G6 and E0 part 2 (about 1,200 Laplace-bootstrap datasets with jackknife, roughly 18 hours
   on 4 CPUs), bank them; then launch the E1 local arm (at least 3 UTC days), with check-ins.
3. Ask Raj to start the Claude slice on his machine once 0004 is frozen (docs/PLAN.md); the
   slice takes about 30 days at 6 runs a day.

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
