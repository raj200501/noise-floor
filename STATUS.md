goal_turn: 5
STATE: ACTIVE

# Status

Updated 2026-11-11, goal_turn 5 (Raj's "crunch time" order: G5 and G6, then the E1 local arm,
E3 on E1, and the review). History is in docs/LOG.md.

## Gates

Last fast ship-check (does not count) at 3ad6ea2: 9/14 (G1 to G4, G9 to G11, G13, G14 pass; G5 to G8 wait on E0 part 2, G6, and E1; G12 on `make review`).

| Gate | Status | Evidence / next step |
|------|--------|----------------------|
| G1 Build and tests | PASS | fresh clone: setup ok, tests pass, line coverage above 85% |
| G2 Static quality | PASS | ruff, ruff format, mypy --strict clean; every `type: ignore` has a reason |
| G3 CI parity | PASS | ci.yml runs `make ci` on 3.11 and 3.12; `make prepush` passes before every push |
| G4 Offline demo | PASS | runs in a network namespace in seconds, matches golden |
| G5 Library validity | FAIL | part 1 passes; E0 part 2 (0005, frozen) queued after G6 in `runs/e0g6/driver.sh` |
| G6 Orchestrator validity | FAIL | 0006 frozen; running (`runs/e0g6/driver.sh`, resumable per dataset) |
| G7 Factorial study | FAIL | 0004 frozen (85 local tasks, 4 days); local arm after G6 and E0 part 2; Claude slice needs Raj |
| G8 Leaderboards and bounds | FAIL | E2 and both reports done; E3 on E1 comes with the E1 analysis |
| G9 Library shipped | PASS | wheel installs in a clean venv; 5 examples run; API reference generated |
| G10 Claim integrity | PASS | README and REPORT pass claimcheck and freshness (260 claim markers) |
| G11 Preregistration integrity | PASS | 6 preregistrations frozen, artifacts predated by them |
| G12 Independent review | FAIL | `make review` after E1 |
| G13 Documentation | PASS | README per section 11; REPORT within the word range with all sections; 24 references verified |
| G14 Hygiene | PASS | secrets-scan clean; cash_usd = 0, caps respected; tree clean |

## Now

Crunch time (Raj, 2026-11-11). Raj merged the release-ready commit into main (bbe8e62, a merge
commit with the same tree as 3a451b0) and made the repository public; this branch fast-forwarded
to it. Raj authorizes updating main at each RELEASE-READY point (`git merge --ff-only
origin/main`, fresh-clone `make ci`, claimcheck and freshness, then push HEAD and HEAD:main,
never force). The Needs Raj list had no "move main" item, so nothing was dropped.

The container restarted at 21:04 UTC and killed the G6 run after its first rows; the driver was
relaunched at 21:07 UTC with 4 workers at bbe8e62 (`src/` unchanged, same environment hash) and
resumes per dataset. README.md now says plainly that the Claude slice of E1 is pending. No `src/`
change until G6 and E0 part 2 are banked: their rows refuse a dirty or changed `src/`.

## Next three actions

1. Keep G6 and E0 part 2 running (`runs/e0g6/driver.sh`, a little over 40 hours on 4 workers;
   relaunch after any container restart), commit and push raw rows at check-ins.
2. When both have written every row: bank G6 (`python -m noisefloor.recovery g6-bank
   prereg/0006-g6-recovery.md`) and E0 part 2 (`python -m noisefloor.e0_run bank
   prereg/0003-e0-coverage-grid-part1.md prereg/0005-e0-glmm-cells.md --only
   prereg/0005-e0-glmm-cells.md`), run `python -m noisefloor.e0_validity`, commit, fast
   ship-check; on G5 and G6 passing, update main (RELEASE-READY).
3. Launch the E1 local arm at the next 00:00 UTC (`python -m noisefloor.e1 local
   prereg/0004-e1-factorial.md`, at least 4 UTC days, with check-ins); after it: the E1
   analysis, the post-E1 check, E3 on E1, the report's E1 sections, and `make review`.

## Usage to date

cash_usd = 0. cap_share = 0.2 (D-007). Claude subject runs 0, single calls 0, review runs 0.
Local model runs: pilot p1 24, screening s2 11 (exploratory, suite v1), s3 84 (suite v2; stopped early, D-025).

## Needs Raj

1. Claude subject arm (G7 needs at least 150 Claude-tier runs): the billing guard blocks Claude
   children in this hosted session by design. When E1 is preregistered, run the arm on a machine
   where `make doctor` shows the guard clean, following docs/PLAN.md ("Claude arm on Raj's
   machine"), starting with `python -m noisefloor.claude_arm smoke`. The daily cap is budget.toml's
   cap times the smaller of its `cap_share` (0.2) and `CAP_SHARE`: 6 runs a day, about 30 days
   for the 180 planned runs; only Raj can raise `cap_share` in budget.toml. Timing: the runner
   needs a clone with the G6 and E0 part 2 artifacts banked (expected around 2026-11-13), and
   the slice starts no run 45 days after the freeze of 0004 (from 2026-12-26). At 6 runs a day
   the 150 Claude-tier runs that G7 needs take 25 days, so start by 2026-12-01 (by 2026-11-26
   for all 180).
2. raise-v1-public: its web page returns a proxy 403 here, so it is dropped from the verified
   references; its integrity tools were ported from a clone with Apache-2.0 attribution.
   Optional: check the reference by hand.
3. Optional: this session runs in the Default environment, which has no setup script. A
   personal environment for this repository whose setup script runs `apt-get install -y
   r-base-core r-base-dev r-cran-lme4` would keep R and lme4 in fresh containers (D-008). Not
   proposed from here, since the approval prompt would stall the unattended run.
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
