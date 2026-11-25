goal_turn: 7
STATE: BLOCKED ON RAJ

# Status

Updated 2026-11-25 06:20 UTC, goal_turn 7 (Raj paused the build on 2026-11-13; a one-hour
finalization on 2026-11-25). History is in docs/LOG.md.

## Gates

Last full ship-check (counts) at d2d0bca: SHIP-CHECK: FAIL 9/14. G1 to G4, G9 to G11, G13, and
G14 pass (1,424 tests, none skipped, 97.0% line coverage); G5 and G6 wait on their running
studies, G7 and G8 on E1, and G12 on a SHIP review (the release review of d9d04b2 returned BLOCK
on B3, fixed in D-041; one review a UTC day). Released: main fast-forwarded to 4f54a6e (docs-only
changes since d2d0bca; prepush, claimcheck, and freshness pass) on 2026-11-12 at 01:14 UTC.

| Gate | Status | Evidence / next step |
|------|--------|----------------------|
| G1 Build and tests | PASS | fresh clone: setup ok, tests pass, line coverage above 85% |
| G2 Static quality | PASS | ruff, ruff format, mypy --strict clean; every `type: ignore` has a reason |
| G3 CI parity | PASS | ci.yml runs `make ci` on 3.11 and 3.12; `make prepush` passes before every push |
| G4 Offline demo | PASS | runs in a network namespace in seconds, matches golden |
| G5 Library validity | FAIL | part 1 passes; E0 part 2 (0005) paused part way since 2026-11-13 (group e1 complete, e1-het 143 of 500, the rest not started); resuming needs its recorded environment (D-045) |
| G6 Orchestrator validity | FAIL | 0006 frozen; all 200 rows written in one environment from a clean `src/`; banked with E0 part 2 once that finishes |
| G7 Factorial study | FAIL | 0004 frozen; local arm after G5 and G6 pass (needs the pinned llama-server build or a 0004 amendment, D-045); the Claude slice can no longer reach 150 runs by its end without Raj (D-045 item 5) |
| G8 Leaderboards and bounds | FAIL | E2 and both reports done; E3 on E1 comes with the E1 analysis, after the Claude slice ends |
| G9 Library shipped | PASS | wheel installs in a clean venv; 5 examples run; API reference generated |
| G10 Claim integrity | PASS | README and REPORT pass claimcheck and freshness (261 claims); derived artifacts reproduce |
| G11 Preregistration integrity | PASS | 6 preregistrations frozen, artifacts predated by them |
| G12 Independent review | FAIL | release review of d9d04b2: BLOCK on B3, fixed (D-041); next review a later UTC day (one a day) |
| G13 Documentation | PASS | README per section 11; REPORT within the word range with all sections; 24 references verified |
| G14 Hygiene | PASS | secrets-scan clean; cash_usd = 0, caps respected; tree clean |

## Now

Raj paused the build on 2026-11-13 to save usage: the G6 and E0 part 2 driver was stopped at about
00:23 UTC with every raw row whole and committed (40daf13), and every scheduled check-in was
deleted. On 2026-11-25 the session resumed in a new container (no virtualenv, R, model weights,
llama-server build, or scratchpad) for a one-hour finalization at Raj's request: main's two CI
fixes are merged into the branch (c68090c); README and REPORT say the validation runs are paused,
not running; the README carries this repository's CI badge, and its first screen (from a
two-lens first-impression review) now leads with what evalstats does, an install-and-use
quickstart, and the Inspect, Harbor, and SWE-bench readers, renders its headline correctly (no
line opens with a claim marker), and no longer lists `evalstats report`, which is not
implemented; `make ci` passes here (1,431 tests,
92% line coverage; the 18 tests that need R skip without it); and main is updated by a
fast-forward. Nothing is banked, and no result changed. D-045 records how to resume: E0 part 2
only in its recorded environment (`src/` tree 036a0cb, R 4.3.3 with lme4 1.1-35.1, and the
rest of `artifacts/e0/raw/environments.json`), the E1 arm only with the pinned llama-server build
or a frozen 0004 amendment, and the banking-day drafts redrafted from docs/PLAN.md step 5.
Earlier notes are in docs/LOG.md.

## Next three actions

1. On Raj's go-ahead to spend the usage: restore the recorded environment (D-045 item 3), run
   `python3 tools/runcheck.py e0g6-preflight` until it prints PASS, relaunch with `WORKERS=4
   setsid nohup bash tools/e0g6_driver.sh >> runs/e0g6/nohup.out 2>&1 < /dev/null &`, and keep
   `tools/wait_e0g6.sh` running; about a day of CPU on 4 workers remains.
2. When the driver ends, follow docs/PLAN.md "Banking day and launch night" (D-036 item 8, D-042,
   D-043), redrafting the outcome texts first; RELEASE-READY in either outcome.
3. Before the E1 local arm: download and check the three weight files, rebuild llama-server, and
   if its digest differs from the frozen pin freeze a 0004 amendment with a llama-server block
   (D-045 item 4); then `e1-preflight` and the launch-night list of docs/PLAN.md.

## Usage to date

cash_usd = 0. cap_share = 0.2 (D-007). Claude subject runs 0, single calls 0, review runs 2 (the
milestone review of 2026-11-11, D-037 item 9, and the release review of 2026-11-12, D-041).
Local model runs: pilot p1 24, screening s2 11 (exploratory, suite v1), s3 84 (suite v2; stopped early, D-025).

## Needs Raj

1. Decide whether and when to resume the paused runs (D-045): E0 part 2 needs about a day of CPU in
   its recorded environment, then banking, then the E1 local arm (4 UTC days).
2. Claude subject arm (G7 needs at least 150 Claude-tier runs). The billing guard blocks Claude
   children in this hosted session by design, so the slice runs on Raj's machine, following
   docs/PLAN.md ("Claude arm on Raj's machine"), starting with `python -m noisefloor.claude_arm
   smoke`, and only once this file says G5 PASS and G6 PASS at the banking commit and the local
   arm has started (D-036 item 1). At `cap_share` 0.2 the slice gets 6 runs a UTC day and starts
   no run from 2026-12-26T04:21:42Z on, so the 150 runs G7 needs are no longer reachable without
   a decision only Raj can make: raise `cap_share` in budget.toml (0.5 gives 15 runs a day, so
   150 runs take 10 days) or amend the end date, a deviation (D-045 item 5). As preregistered,
   the E1 analysis waits until both arms have ended, so E3 on E1 (G8), the post-E1 check (G7),
   and the final review (G12) wait for the slice's end point.
3. raise-v1-public: its web page returns a proxy 403 here, so it is dropped from the verified
   references; its integrity tools were ported from a clone with Apache-2.0 attribution.
   Optional: check the reference by hand.
4. Needed before any resume: a personal environment for this repository whose setup script runs
   `apt-get install -y r-base-core r-base-dev r-cran-lme4` (and checks the versions D-045 item 3
   lists), so R survives a new container; this container has none (D-008, D-045).
5. FYI (from goal_turn 1): one batch of citation fetches sent a User-Agent containing the local
   part of your email address to arxiv.org, anthropic.com, and metr.org; fetch tooling now uses
   a generic User-Agent.

## Blocked

None.

## Backlog (suggestions for Raj, not ladder rungs)

1. A `evalstats compare --from-inspect <log dir>` one-liner so an evals engineer gets a paired
   interval from existing Inspect logs without writing any code.
2. `evalstats decompose --bootstrap` on the command line, so a team can get variance shares with
   intervals from its own run logs in one call.
