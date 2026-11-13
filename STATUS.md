goal_turn: 6
STATE: ACTIVE

# Status

Updated 2026-11-13 00:20 UTC, goal_turn 6 (after Raj's final-hour sprint and the release to main;
E0 part 2 running, banking day prepared). History is in docs/LOG.md.

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
200 of its preregistered rows. Group e1 wrote all 500 of its datasets by 19:58 UTC; group e1-het
started at 20:05 UTC and, at e1's rate (from row stamp times only), should end around 12:15 UTC on
2026-11-13, followed by the cheaper local-arm and ungrouped cells. The container restarted at
about 00:13 UTC on 2026-11-13 (a new boot and kernel; the disk, R, and the environment hash
c1569b433f0f5a99 unchanged); the driver was relaunched at 00:15 UTC through
`tools/e0g6_driver.sh` (preflight PASS, no cut line; every e1-het file held 139 whole lines),
and e1-het resumed where it stopped. Raw rows are committed and pushed at each check-in (about
every 2 hours); nothing is banked or read as a result until the driver ends. A harness-tracked
waiter keeps the container up (its minute log's one gap since 2026-11-11 is the restart,
00:12 to 00:15 UTC on 2026-11-13).

Banking day is prepared (D-042, D-043). `python3 tools/runcheck.py e0g6-complete` reads
completeness from the raw rows, and the bank commands of both studies were rehearsed in
throwaway clones (E0 part 2 on its partial rows, outputs unread). A read-only readiness review
of the banking commit and the launch (four lenses, each checked by an adversarial verifier, and
a completeness critic) found one launch defect, now fixed: `tools/wait_e1.sh` watched the PID of
the launch line's bash subshell instead of the arm, so on launch night it would have ended at
once and left nothing keeping the container up until the first batch; the arm is now found by
its argv (`e1-alive`). The review also led to `e1-preflight` (the arm's start-up checks called
one by one, never starting a batch), `tools/wait_until.sh` (keep-alive from the driver's end to
the launch window), tests over the banking outputs, an `oom_score_adj` of 0 for the arm, the
RELEASE-READY procedure with a fetch, and README and REPORT wording that stays true across the
launch. docs/PLAN.md ("Banking day and launch night") has the ordered checklist. The banking
commit's README and REPORT text and its claim definitions are drafted outside the repository in
three outcome variants (all pass; only reported extras fail; a headline cell or a G6 component
fails), each tested end to end with invented fixtures in a throwaway clone (claimcheck,
freshness, G13); the scratchpad's drafts/notes.md gives the order and the placeholders.
D-042 item 5 records that the G6 gate was also run on the rehearsal output, its verdict held
back until both studies are banked. Earlier notes
(the release to main, Raj's crunch-time instruction, the restart, the pre-mortem, and the
milestone review) are in docs/LOG.md; Raj's authorization to update main at RELEASE-READY points
stands (D-043 item 6).

## Next three actions

1. Keep E0 part 2 running: `bash tools/wait_e0g6.sh` as a background task, restarted first on
   every wake-up; relaunch only with `WORKERS=4 setsid nohup bash tools/e0g6_driver.sh >>
   runs/e0g6/nohup.out 2>&1 < /dev/null &`. At each check-in, stage raw rows with `python3
   tools/stage_rows.py --changed`, commit as Raj, `make prepush`, push, and confirm that `git status --porcelain -- src uv.lock pyproject.toml` is empty and that `git
   rev-parse HEAD:src HEAD:uv.lock` prints the `src_tree` and `uv_lock` recorded for
   c1569b433f0f5a99 in `artifacts/e0/raw/environments.json` (D-043 item 12).
2. When the driver has ended, follow docs/PLAN.md "Banking day and launch night" (D-036 item 8,
   D-042, D-043): `COMPLETE: YES` from `python3 tools/runcheck.py e0g6-complete`; the bank
   commands, `e0_validity`, and `tools/validity_summary.py`; the claims and the README and REPORT
   text for the outcome; one banking commit staged by path; the fast ship-check saved;
   `e1-preflight`; a docs-only commit with the verdict; RELEASE-READY in either outcome.
3. If G5 and G6 pass and RELEASE-READY has ended by about 23:30 UTC, launch the E1 local arm
   between 21:00 and 23:45 UTC with the D-043 launch command and its checks, `tools/wait_e1.sh`,
   and an hourly E1 check-in; otherwise the next night. If either fails, hold E1, ask Raj not to
   start the slice, and freeze amendments first (D-036 item 1). When the arm ends, the analysis
   waits for the Claude slice: STATE BLOCKED ON RAJ unless other work remains.

## Usage to date

cash_usd = 0. cap_share = 0.2 (D-007). Claude subject runs 0, single calls 0, review runs 2 (the
milestone review of 2026-11-11, D-037 item 9, and the release review of 2026-11-12, D-041).
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
