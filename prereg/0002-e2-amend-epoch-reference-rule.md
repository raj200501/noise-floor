---
amends: 0001
date: 2026-11-04
---

# 0002 Amendment to 0001: the L2 reference score follows Epoch's own rule exactly

- Date: 2026-11-04
- Amends: `prereg/0001-e2-leaderboard-reanalysis.md` (frozen earlier the same day).
- Reason: 0001 names the L2 (Epoch AI, SWE-bench Verified) reference policy as `exclude`, but
  the `exclude` policy of `evalstats.policy_bounds` drops every ambiguous run (statuses
  `infra_error`, `timeout`, and `flagged`), while Epoch's published rule leaves only errored
  samples out of the denominator; a sample that ended at a limit keeps its recorded score. The
  intent stated in 0001 is to use each board's own rule, so the reference score must drop
  errored samples only.
- Timing: committed before any interval, ordering, rank set, or bound of 0001 is computed. No
  E2 or E3 result exists yet.

## Change

1. L2 reference score (0001, Methods 1): the mean over tasks of the per-task outcome after
   dropping samples whose status is `infra_error` (errored samples, as Epoch does). Samples with
   status `timeout` (a limit ended them) or `flagged` keep their recorded outcome. The order
   (Methods 2), Q1 (Methods 3), and Q2 (Methods 4) use this reference score.
2. Q3 (Methods 5) is unchanged: `policy_bounds` with ambiguous statuses `infra_error`,
   `timeout`, and `flagged`, and policies `as_scored`, `fail`, `success`, and `exclude`. Its
   reference ranking for counting ranking changes is the reference ranking of item 1, and its
   pair bounds use the item 1 order.
3. L1 (Terminal-Bench 2.0) is unchanged: the reference policy is `as_scored`, which equals the
   board's rule (a trial with no reward counts as 0).

This amendment is reported as a deviation from 0001 in docs/REPORT.md.

```json e2-amendment
{
  "amends": "prereg/0001-e2-leaderboard-reanalysis.md",
  "board": "epoch-swebench",
  "reference_rule": {"drop_statuses": ["infra_error"], "keep_statuses": ["ok", "timeout", "flagged"]}
}
```
