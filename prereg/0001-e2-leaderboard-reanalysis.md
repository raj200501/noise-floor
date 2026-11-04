# 0001 E2 and E3 (public data): leaderboard reanalysis and policy-sensitivity bounds

- Date: 2026-11-04
- Experiments: E2 (public reanalysis, RQ3) and the public-data part of E3 (policy-sensitivity
  bounds, RQ4). E3 on the factorial study (E1) is preregistered with E1.
- Status: confirmatory analysis plan, frozen by `tools/prereg.py freeze` before any interval,
  ordering, rank set, or bound in this plan is computed on the data below.

## Disclosure of what was seen before freezing

A scouting pass (2026-10-23) fetched job-level files of the Terminal-Bench 2.0 submissions and
headers of the Epoch AI logs to learn their formats, counted submissions, tasks, trials, and
exception types, and checked that recomputed aggregate scores matched the published ones. It
computed no confidence interval, paired comparison, rank interval, or bound. The exclusion
rules below come from the maintainers' published actions and from file formats, not from any
result.

## Questions (no direction presumed)

- Q1 (RQ3): on each leaderboard, what share of adjacent pairs in the board's own order has a
  95% paired interval for the score difference that excludes zero?
- Q2 (RQ3): how wide are simultaneous 95% rank intervals, and how many entries could be first
  (rank interval includes 1)?
- Q3 (RQ4): how many adjacent orderings stay the same under every rule for scoring ambiguous
  runs (infrastructure errors, timeouts, flagged runs), and how many change?

Descriptive threshold (frozen): a leaderboard is called "mostly resolved" if the share in Q1 is
at least 0.5 and "mostly unresolved" otherwise. Both labels are reported with the same
prominence.

## Data

### L1: Terminal-Bench 2.0 submissions

- Source: Hugging Face dataset `harborframework/terminal-bench-2-leaderboard`, revision
  `572b2614be2c0cb2527e14f5b1e4026f1072e6c1`, license Apache-2.0. Only job-level
  `result.json`, `config.json`, and `metadata.yaml` files are fetched; every fetched file's
  SHA-256 is recorded in `artifacts/e2/tb2/manifest.json`, and the derived trial table (with the
  Terminal-Bench canary GUID) is committed as the snapshot.
- Unit: one trial. Outcome: the trial's reward (0 or 1); a trial with no reward counts as 0, the
  board's own rule. Task names are mapped through the alias table in
  `noisefloor.reanalysis.tb2.TASK_ALIASES`.
- Status: `timeout` for AgentTimeoutError, VerifierTimeoutError, and TimeoutException;
  `infra_error` for every other exception type and for trials with neither reward nor exception;
  `ok` otherwise (`noisefloor.reanalysis.tb2`).
- Exclusions (entry level), each reported with its reason:
  1. jobs whose configuration names a dataset other than Terminal-Bench 2.0 (a 2.1 run);
  2. entries the maintainers removed from the board or rescored, per their published
     "Leaderboard Integrity Update" (folders starting `OB-1`, `pilot`, and `Forge`), because the
     snapshot does not carry the maintainers' corrections;
  3. entries with no job-level `result.json` in the documented format (they cannot be parsed
     without trial-level files).
- Analysis sets: L1-primary is every remaining entry with at least one trial on each of the 89
  tasks; L1-secondary is every remaining entry, on the tasks attempted by all of them.

### L2: Epoch AI Benchmarking Hub, SWE-bench Verified

- Source: `https://epoch.ai/data/benchmark_data.zip` (license CC BY 4.0), file
  `swe_bench_verified.csv`, retrieved on the fetch date recorded in the manifest with the
  archive's SHA-256; each row's Inspect log linked in its `Logs` column, read with HTTP range
  requests for `header.json` and `summaries.json` only. Rows whose log link does not open are
  excluded and counted. Every fetched member's SHA-256, the object's ETag, and its version id
  are recorded in `artifacts/e2/epoch-swebench/manifest.json`; the derived sample table is
  committed with attribution.
- Unit: one sample (one epoch per instance). Outcome: the `swe_bench_scorer` value (C = 1,
  I = 0). Status from the Inspect adapter's mapping (`evalstats.adapters.inspect`): errored
  samples `infra_error`, limited samples `timeout`, invalidated samples `flagged`.
- Board rule: errored samples are left out of the denominator, so the reference policy for L2 is
  `exclude`.
- Analysis set: every row with a readable log, as a separate entry (a model that appears twice
  is two entries, labeled by model and date).

## Methods

Implemented in `noisefloor.reanalysis` on top of evalstats; fixed seeds below.

1. Score of an entry: mean over tasks of the per-task mean outcome under the board's reference
   policy (L1 `as_scored`, L2 `exclude`). The board's own pooled score is also reported.
2. Order: descending reference score, ties broken by entry name.
3. Q1: for each adjacent pair, the paired difference on the tasks both entries attempted, with a
   95% task-clustered t interval (`evalstats.paired_difference`, method `cluster_t`); the pair is
   supported if the interval excludes zero. Report the count and share per analysis set. A
   bootstrap version (method `bootstrap`, 4000 task resamples, seed 20261104) is secondary.
4. Q2: simultaneous 95% rank intervals from `evalstats.rank_confidence` (task bootstrap of the
   maximum studentized pairwise deviation, 4000 resamples, seed 20261104) on the tasks attempted by
   every entry in the set; marginal intervals are secondary. Report each entry's interval, the
   median interval width, and the number of entries whose interval includes rank 1.
5. Q3: `evalstats.policy_bounds` with ambiguous statuses `infra_error`, `timeout`, `flagged`;
   policies `as_scored`, `fail`, `success`, `exclude`; worst-case bounds with Imbens and Manski
   95% intervals. For each adjacent pair in the reference order, report whether the lower bound
   of the difference is positive (identified) and whether its interval excludes zero (robust),
   and count the policies whose ranking differs from the reference ranking.
6. Validity dependency: every interval method used for a headline number must pass its E0 cell
   (gate G5). If a needed cell fails, the result is reported with the method marked unsupported,
   and the secondary method is reported if its cell passes.

## Confidence reports

One report per leaderboard in `docs/leaderboards/<board>.md`, generated from the artifacts by a
committed script: entries, scores with intervals, adjacent-pair support, rank intervals, and
policy bounds, with neutral wording (rule 15).

## Deviations

Any change is a dated amendment that cites this file, committed before the affected analysis.

## Machine-readable plan

```json e2-plan
{
  "boards": {
    "tb2": {
      "dataset": "harborframework/terminal-bench-2-leaderboard",
      "revision": "572b2614be2c0cb2527e14f5b1e4026f1072e6c1",
      "license": "Apache-2.0",
      "reference_policy": "as_scored",
      "exclude_prefixes": ["OB-1", "pilot", "Forge"],
      "exclude_non_tb20_dataset": true,
      "n_canonical_tasks": 89
    },
    "epoch-swebench": {
      "archive": "https://epoch.ai/data/benchmark_data.zip",
      "csv": "swe_bench_verified.csv",
      "license": "CC-BY-4.0",
      "scorer": "swe_bench_scorer",
      "reference_policy": "exclude"
    }
  },
  "level": 0.95,
  "adjacent_method": "cluster_t",
  "secondary_method": "bootstrap",
  "rank": {"simultaneous": true, "n_boot": 4000, "seed": 20261104},
  "bootstrap_seed": 20261104,
  "ambiguous_statuses": ["infra_error", "timeout", "flagged"],
  "policies": ["as_scored", "fail", "success", "exclude"],
  "resolved_share_threshold": 0.5
}
```
