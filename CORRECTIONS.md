# Corrections

Dated entries for every changed or withdrawn claim, newest first.

## 2026-11-11

- Changed (D-035): the Imbens and Manski interval of a model's success rate under the policy
  bounds (E3, `evalstats.policy_bounds`) is now cut to [0, 1], the range of a success rate. On
  Terminal-Bench 2.0 the leading entry's upper end changed from 1.001 to 1.000
  (`artifacts/e2/tb2/results.json`, `artifacts/e3/bounds.json`,
  `docs/leaderboards/tb2.md`); no other banked number and no reading changed. Cutting an interval
  to the parameter space never lowers its coverage. Found by the pre-release review.

## 2026-11-04

- Withdrawn (D-033 item 1): D-032 item 2 said the local E1 design was the largest that fits the
  7-day window at the dry run's wall times. That was wrong: at those rates the runs needed about
  a day. The preregistered power analysis now sets the local sample.
- Changed (D-034 item 1): D-033 item 1 said the sizing rule gives 16 tasks per family. That
  holds only for normal test statistics; with the t statistics of the tests E1 runs, the rule
  gives 17 tasks per family (85 tasks), which the design now uses.
- Corrected (D-034 item 10): commit f4a1046 and D-033 item 13 said the sandboxed runner dies with
  the process that launched it and that a relaunch never runs beside a live runner of the same
  arm. Neither held: `unshare --fork` blocked the parent-death SIGTERM, and the relaunch check
  looked only at the planned batch being started. The sandbox now uses SIGKILL, and no batch
  starts beside any live runner.
- Corrected (D-034 item 9): D-033 item 11 said the frozen llama-server pin is its binary's
  SHA-256. The code pins and checks the digest of the binary and the shared libraries beside it.
