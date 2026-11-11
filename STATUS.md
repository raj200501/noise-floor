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
| G7 Factorial study | FAIL | 0004 frozen (85 local tasks, 4 days); local arm after G5 and G6 pass; Claude slice needs Raj; analysis after both arms |
| G8 Leaderboards and bounds | FAIL | E2 and both reports done; E3 on E1 comes with the E1 analysis, after the Claude slice ends |
| G9 Library shipped | PASS | wheel installs in a clean venv; 5 examples run; API reference generated |
| G10 Claim integrity | PASS | README and REPORT pass claimcheck and freshness (260 claim markers) |
| G11 Preregistration integrity | PASS | 6 preregistrations frozen, artifacts predated by them |
| G12 Independent review | FAIL | `make review` after the E1 analysis (after the Claude slice ends) |
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
resumes per dataset; a harness-tracked waiter has kept the container up since (no gap in its
minute log). README.md says plainly that the Claude slice of E1 is pending. No `src/` change
until both arms end, apart from the `e0_validity` commit: rows from a changed `src/` are refused
at once, rows from a dirty one only at banking.

A read-only pre-mortem (a five-lens workflow, each finding checked by an adversarial verifier)
of the validation runs, their banking, the E1 arm, resources, and the Claude runbook found
procedure gaps and no defect that changes a result. D-036 records the operating rules;
`tools/runcheck.py` (relaunch preflight, E1 gap test, E1 relaunch check), `tools/e0g6_driver.sh`,
and the waiters `tools/wait_e0g6.sh` and `tools/wait_e1.sh` implement them, and the Claude runbook
in docs/PLAN.md is corrected. Two consequences: the local arm launches only if G5 and G6 pass at
the banking commit, and, as prereg 0004 says, the E1 analysis waits for both arms, so E3 on E1
(G8), the post-E1 check (G7), and the final review (G12) wait for the end of the Claude slice.

## Next three actions

1. Keep G6 and E0 part 2 running (a little over 40 hours on 4 workers in all): `bash
   tools/wait_e0g6.sh` as a background task, restarted first on every wake-up; relaunch only with
   `WORKERS=4 setsid nohup bash tools/e0g6_driver.sh >> runs/e0g6/nohup.out 2>&1 < /dev/null &`;
   commit and push raw rows at check-ins.
2. When the driver has ended: check completeness from the raw rows (D-036 item 8), bank G6
   (`python -m noisefloor.recovery g6-bank prereg/0006-g6-recovery.md`) and E0 part 2 (`python
   -m noisefloor.e0_run bank prereg/0003-e0-coverage-grid-part1.md prereg/0005-e0-glmm-cells.md
   --only prereg/0005-e0-glmm-cells.md`), run `python -m noisefloor.e0_validity`, update the
   REPORT claim that counts headline cells, commit, fast ship-check. If G5 and G6 pass, update
   main (RELEASE-READY); if either fails, hold E1 and freeze amendments first (D-036 item 1).
3. If G5 and G6 pass, launch the E1 local arm between 21:00 and 23:45 UTC as docs/PLAN.md
   ("Operating rules") says, with `tools/wait_e1.sh` and an hourly check-in; during batches,
   check-ins only (`tools/runcheck.py e1-gap`). When the arm ends, the analysis waits for the
   Claude slice: STATE BLOCKED ON RAJ unless other work remains.

## Usage to date

cash_usd = 0. cap_share = 0.2 (D-007). Claude subject runs 0, single calls 0, review runs 0.
Local model runs: pilot p1 24, screening s2 11 (exploratory, suite v1), s3 84 (suite v2; stopped early, D-025).

## Needs Raj

1. Claude subject arm (G7 needs at least 150 Claude-tier runs). The billing guard blocks Claude
   children in this hosted session by design, so the slice runs on Raj's machine, following
   docs/PLAN.md ("Claude arm on Raj's machine"), starting with `python -m noisefloor.claude_arm
   smoke`. As preregistered, the E1 analysis waits until both arms have ended, so E3 on E1 (G8),
   the post-E1 check (G7), and the final review (G12) all wait for the slice's end point. Start
   on the first day the banked G6 and E0 part 2 artifacts are pushed (expected around 2026-11-13
   or 14), from the branch `claude/gifted-darwin-k7q99b` at or after the `e0_validity` commit.
   The daily cap is budget.toml's cap times the smaller of `cap_share` (0.2) and `CAP_SHARE`: 6
   runs a UTC day, and the smoke run takes one of its day's runs. All 180 planned runs then need
   31 UTC days and the 150 that G7 needs 26, so start by 2026-11-25 for all 180 and by
   2026-11-30 for 150; the slice starts no run from 2026-12-26T04:21:42Z on. Only Raj can raise
   `cap_share` in budget.toml (0.5 gives 15 runs a day); a raised value ships with the ledger and
   stays at the highest value used. Push mid-slice backups to a separate branch, and merge the
   session branch into the slice clone only after the end point.
2. raise-v1-public: its web page returns a proxy 403 here, so it is dropped from the verified
   references; its integrity tools were ported from a clone with Apache-2.0 attribution.
   Optional: check the reference by hand.
3. Optional but now more useful: this session runs in the Default environment, which has no
   setup script. A personal environment for this repository whose setup script runs `apt-get
   install -y r-base-core r-base-dev r-cran-lme4` would keep R and lme4 in fresh containers
   (D-008). The E1 analysis runs weeks from now, after the Claude slice, and its analysis-code
   pin needs the exact package versions D-037 item 4 lists, so a container replaced before then
   without them needs an amendment of 0004. Not proposed from here, since the approval prompt
   would stall the unattended run.
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
