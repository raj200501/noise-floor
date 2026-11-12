# Log

History moved out of STATUS.md, newest first.

## 2026-11-11 to 2026-11-12

Moved from STATUS.md on 2026-11-12 (as written then, apart from the correction it notes).

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
at a relaunch preflight, on resuming a started cell or group, and at banking (corrected by
D-043 item 12, which had said at once).

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

## 2026-10-23 to 2026-11-04 goal_turns 1 and 2

goal_turn 1 (2026-10-23): evalstats core (schema, cluster-t and bootstrap intervals, paired
comparisons, power and MDE, rank-confidence sets, policy bounds), logit GLMM decomposition with
lme4 through rpy2 and a statsmodels fallback, offline demo (tag v0.1 local), verified references,
integrity tools ported from raise-v1-public, adapters for Inspect, Harbor, and SWE-bench, and a
llama.cpp CPU build.

goal_turn 2 (2026-11-04): E2 reanalysis of Terminal-Bench 2.0 and Epoch AI SWE-bench Verified
with confidence reports (prereg 0001, amendment 0002); E0 part 1 (prereg 0003, 22 cells) with
library warnings for unsupported cells; cross-checks against statsmodels, scipy, and R; nf-tools
suite, minimal-agent and Inspect harnesses, orchestrator, namespace sandbox; pilot p1 (stopped,
D-018) and screening s2 (stopped, D-020); five examples and the generated API reference; CI
limited to main and pull requests (D-015); commits authored as Raj (D-014). Raj paused the work
to save usage, then asked to continue.

## 2026-10-23 bootstrap

Bootstrap per docs/BUILD_BRIEF.md section 10.1, started by the heartbeat loop in a Claude Code on
the web session. Brief copied and hashed; doctor, scaffold, make targets, ship-check, review
tooling, subagents, plan, and state files created.
