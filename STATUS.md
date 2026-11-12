goal_turn: 6
STATE: ACTIVE

# Status

Updated 2026-11-12 14:20 UTC, goal_turn 6 (after Raj's final-hour sprint and the release to main;
E0 part 2 running). History is in docs/LOG.md.

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
| G5 Library validity | FAIL | part 1 passes; E0 part 2 (0005, frozen) running since 2026-11-12 03:36 UTC, same environment as G6 |
| G6 Orchestrator validity | FAIL | 0006 frozen; all 200 rows written (2026-11-12 03:36 UTC, one environment, clean `src/`); banked with E0 part 2 once that finishes |
| G7 Factorial study | FAIL | 0004 frozen (85 local tasks, 4 days); local arm after G5 and G6 pass; Claude slice needs Raj; analysis after both arms |
| G8 Leaderboards and bounds | FAIL | E2 and both reports done; E3 on E1 comes with the E1 analysis, after the Claude slice ends |
| G9 Library shipped | PASS | wheel installs in a clean venv; 5 examples run; API reference generated |
| G10 Claim integrity | PASS | README and REPORT pass claimcheck and freshness (261 claims); derived artifacts reproduce |
| G11 Preregistration integrity | PASS | 6 preregistrations frozen, artifacts predated by them |
| G12 Independent review | FAIL | release review of d9d04b2: BLOCK on B3, fixed (D-041); next review a later UTC day (one a day) |
| G13 Documentation | PASS | README per section 11; REPORT within the word range with all sections; 24 references verified |
| G14 Hygiene | PASS | secrets-scan clean; cash_usd = 0, caps respected; tree clean |

## Now

E0 part 2 (0005) has run since 2026-11-12 03:36 UTC in G6's environment, right after G6 wrote all
200 of its preregistered rows; at 4 workers its first group should end around 19:40 UTC and the
whole run during 2026-11-13. Raw rows are committed and pushed at each check-in (about every 2
hours); nothing is banked or read as a result until the driver ends. A harness-tracked waiter
keeps the container up (no gap in its minute log since the 21:04 UTC restart on 2026-11-11).
README and REPORT now say G6's rows are written and E0 part 2 is running (they had said G6 was
running and E0 part 2 queued). For banking day (D-042): `python3 tools/runcheck.py e0g6-complete`
reads completeness from the raw rows (indices, seeds, and stamps only). The bank commands of G6
and of E0 part 2 (on its partial rows, outputs unread) were rehearsed in throwaway clones; all
ran without error except `e0_validity` on cells not yet started, which complete cells cannot
hit (D-042 item 4). Nothing was banked or committed from the rehearsals.

Final-hour sprint (Raj, 2026-11-12 00:50 UTC). The README gained a block on why its numbers hold
up. The G6 run was paused with SIGSTOP (no work lost) so the tree stays clean for a release review,
the day's one review run (logged), and the full ship-check before main is updated; it resumed at
01:15 UTC with nothing lost. The release review returned BLOCK on B3: G12's drift check missed files moved into
artifacts/review/. B3 is fixed, with N20 (E0 part 2 is queued after G6, not running), N21 (the
safeguard bullet now names the one interval without a coverage cell), and N23 (raw rows must
postdate their plan, in G5 and G6) (D-041). G12 waits for a later day's review; G5 to G8 wait
for their studies, and nothing from them is presented as a result. G5 now has the counterpart
of G6's planted-truth check (N22, D-041 item 7), in place before E0 part 2 is banked.

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

A milestone review by the `reviewer` subagent (efbff39; the day's one review run, logged)
returned BLOCK on one item, a literal em dash in a test, now fixed with a test that keeps em
dashes out of every authored file. Its 14 non-blocking issues are in docs/REVIEW_LOG.md with what
was done about each: gate checks tightened in `tools/` (G1, G2, G5, G6, G10, G12, G14; D-039),
report and README wording, the model weights in docs/DATA.md, and one rank-interval defect in a
pinned file (N2) that waits until after the E1 analysis. The workflow's completeness critic found
one more blocking gap, fixed: nothing tied `artifacts/e3/bounds.json` to a recomputation (now G8
and a test do; D-040), plus four smaller ones, also in docs/REVIEW_LOG.md.

## Next three actions

1. Keep E0 part 2 running (G6's rows are complete; E0 part 2 should end during 2026-11-13): `bash
   tools/wait_e0g6.sh` as a background task, restarted first on every wake-up; relaunch only with
   `WORKERS=4 setsid nohup bash tools/e0g6_driver.sh >> runs/e0g6/nohup.out 2>&1 < /dev/null &`;
   commit and push raw rows at check-ins.
2. When the driver has ended: `python3 tools/runcheck.py e0g6-complete` must print `COMPLETE:
   YES` (completeness from the raw rows, D-036 item 8; D-042), then bank G6
   (`python -m noisefloor.recovery g6-bank prereg/0006-g6-recovery.md`) and E0 part 2 (`python
   -m noisefloor.e0_run bank prereg/0003-e0-coverage-grid-part1.md prereg/0005-e0-glmm-cells.md
   --only prereg/0005-e0-glmm-cells.md`), run `python -m noisefloor.e0_validity` and `python
   tools/validity_summary.py` (G10 checks its outputs), update the REPORT claim that counts
   headline cells, commit, fast ship-check. If G5 and G6 pass, update
   main (RELEASE-READY); if either fails, hold E1 and freeze amendments first (D-036 item 1).
3. If G5 and G6 pass, launch the E1 local arm between 21:00 and 23:45 UTC as docs/PLAN.md
   ("Operating rules") says, with `tools/wait_e1.sh` and an hourly check-in; during batches,
   check-ins only (`tools/runcheck.py e1-gap`). When the arm ends, the analysis waits for the
   Claude slice: STATE BLOCKED ON RAJ unless other work remains.

## Usage to date

cash_usd = 0. cap_share = 0.2 (D-007). Claude subject runs 0, single calls 0, review runs 1 (the
milestone review of 2026-11-11, D-037 item 9).
Local model runs: pilot p1 24, screening s2 11 (exploratory, suite v1), s3 84 (suite v2; stopped early, D-025).

## Needs Raj

1. Claude subject arm (G7 needs at least 150 Claude-tier runs). The billing guard blocks Claude
   children in this hosted session by design, so the slice runs on Raj's machine, following
   docs/PLAN.md ("Claude arm on Raj's machine"), starting with `python -m noisefloor.claude_arm
   smoke`. As preregistered, the E1 analysis waits until both arms have ended, so E3 on E1 (G8),
   the post-E1 check (G7), and the final review (G12) all wait for the slice's end point. Start
   on the first day this file says G5 PASS and G6 PASS at the banking commit and the local arm
   has started (expected around 2026-11-14), from the branch `claude/gifted-darwin-k7q99b` at or
   after that commit; a start before a failing validation is resolved would close the
   preregistered route for replacing the failing method (D-036 item 1).
   The daily cap is budget.toml's cap times the smaller of `cap_share` (0.2) and `CAP_SHARE`: 6
   runs a UTC day, and the smoke run takes one of its day's runs. All 180 planned runs then need
   31 UTC days and the 150 that G7 needs 26, so start by 2026-11-25 for all 180 and by
   2026-11-30 for 150 (with no run slot lost to an infra_error, an excluded run, or a refusal;
   earlier is safer); the slice starts no run from 2026-12-26T04:21:42Z on. Only Raj can raise
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
