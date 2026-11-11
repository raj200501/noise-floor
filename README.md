# noise-floor

**How much of a leaderboard gap is the model, and how much is noise?**

On Terminal-Bench 2.0, the top two submissions differ by <!-- claim:e2.tb2.top.diff -->+0.018 in task
success (95% paired CI <!-- claim:e2.tb2.top.lo -->-0.041 to <!-- claim:e2.tb2.top.hi -->+0.077), only
<!-- claim:e2.tb2.adjacent.supported -->2 of <!-- claim:e2.tb2.adjacent.pairs -->65 adjacent pairs on
the board are statistically separated, and <!-- claim:e2.tb2.first -->15 of its
<!-- claim:e2.tb2.entries -->66 entries could plausibly rank first. The board's own design detects
a gap of <!-- claim:nf.tb2.mde -->0.097 with 80% power (median over adjacent pairs), while the median
adjacent gap is <!-- claim:nf.tb2.gap -->0.007: most of the ordering sits below the noise floor.

![Terminal-Bench entries: success rate of each submission with its task-clustered interval](docs/figures/e2-tb2-scores.svg)

| Leaderboard (confidence report) | Entries | Tasks | Adjacent pairs separated | Median rank-interval width | Could rank first | Detectable gap* | Median adjacent gap* |
|---|---:|---:|---:|---:|---:|---:|---:|
| [Terminal-Bench 2.0](docs/leaderboards/tb2.md) | <!-- claim:e2.tb2.entries -->66 | <!-- claim:e2.tb2.tasks -->89 | <!-- claim:e2.tb2.adjacent.supported -->2 of <!-- claim:e2.tb2.adjacent.pairs -->65 | <!-- claim:e2.tb2.rankwidth -->35 ranks | <!-- claim:e2.tb2.first -->15 | <!-- claim:nf.tb2.mde -->0.097 | <!-- claim:nf.tb2.gap -->0.007 |
| [SWE-bench Verified (Epoch AI)](docs/leaderboards/epoch-swebench.md) | <!-- claim:e2.epoch.entries -->32 | <!-- claim:e2.epoch.tasks -->454 | <!-- claim:e2.epoch.adjacent.supported -->3 of <!-- claim:e2.epoch.adjacent.pairs -->31 | <!-- claim:e2.epoch.rankwidth -->21 ranks | <!-- claim:e2.epoch.first -->4 | <!-- claim:nf.epoch.mde -->0.048 | <!-- claim:nf.epoch.gap -->0.004 |

Pairs are separated when their paired, task-clustered 95% interval excludes zero; rank intervals
are simultaneous 95% intervals from a task bootstrap. *Exploratory, not preregistered
([`tools/noise_floor.py`](tools/noise_floor.py), [artifact](artifacts/e2/noise_floor.json)): the
detectable gap is the difference in success rate that the board's own tasks and runs detect with
80% power at two-sided alpha 0.05, from each adjacent pair's banked interval; both columns are
medians over adjacent pairs, and only <!-- claim:nf.tb2.atleast -->1 and <!-- claim:nf.epoch.atleast -->2
pairs, respectively, have a gap at least that large. Both analyses follow plans frozen before any
result existed: [`prereg/0001`](prereg/0001-e2-leaderboard-reanalysis.md), amended by
[`prereg/0002`](prereg/0002-e2-amend-epoch-reference-rule.md); `make prereg-verify` checks that
each plan predates the artifacts that cite it.

## Quickstart (about a minute; after setup, nothing touches the network)

```bash
git clone https://github.com/raj200501/noise-floor && cd noise-floor
make setup    # uv sync --locked (Python 3.11 or later)
make demo     # offline demo on recorded fixtures, checked against its golden output
uv run python docs/examples/01_compare_two_models.py
```

On your own results (one row per run: model, task, run, outcome):

```python
from evalstats import paired_difference, read_csv

results = read_csv("my_runs.csv")
cmp = paired_difference(results, "agent-a", "agent-b")  # paired by task, clustered by task
print(cmp.difference.estimate, cmp.difference.lo, cmp.difference.hi, cmp.supported)
```

```bash
uv run evalstats compare my_runs.csv --a agent-a --b agent-b   # the same from the shell
uv run evalstats rank --help                                    # also: power, decompose, report
```

## What this is

- **evalstats**, a typed Python library that makes the right analysis of eval results the
  default: paired and task-clustered comparisons, Wilson and cluster-robust intervals,
  bootstrap and mixed-model (GLMM) variance components, power and minimum detectable effects,
  simultaneous rank-confidence sets, and policy-sensitivity bounds for runs whose outcome is
  ambiguous (timeouts, infrastructure errors). Every interval method carries its own validity
  record from simulation: using a method on a design where it failed its coverage check raises
  `UnsupportedDesignWarning`. API reference in [docs/api](docs/api), runnable examples in
  [docs/examples](docs/examples).
- **Confidence reports for public leaderboards** (experiments E2 and E3): each submission's score
  with an interval, every adjacent ordering tested on shared tasks, simultaneous rank intervals,
  and how the ranking moves when ambiguous runs are scored differently.
- **A preregistered factorial study of agent success** (E1), frozen with its validation plan
  before any data: how the variance in task success splits among model, harness, wall-clock
  timeout, step budget, sampling temperature, concurrency, day, and time of day, on
  procedurally generated mock tasks, with local open-weight models and a Claude slice.
- **Integrity tooling that checks this repository's own claims**: a hash chain of frozen
  preregistrations, a claim check that ties every number in this README and the report to the
  artifact it came from, and a ship-check of fourteen gates run on a fresh clone.

What it is not: a new benchmark, a ranking of anyone's agent, or a statement about why any
submission scored as it did. It measures how much of an observed gap a given eval design can
resolve.

## Key results

| Result | Finding | Artifact | Plan |
|---|---|---|---|
| Terminal-Bench 2.0 reanalysis | <!-- claim:e2.tb2.adjacent.supported -->2 of <!-- claim:e2.tb2.adjacent.pairs -->65 adjacent pairs separated; median rank interval <!-- claim:e2.tb2.rankwidth -->35 ranks wide | [results.json](artifacts/e2/tb2/results.json) | [`prereg/0001`](prereg/0001-e2-leaderboard-reanalysis.md) |
| SWE-bench Verified reanalysis | <!-- claim:e2.epoch.adjacent.supported -->3 of <!-- claim:e2.epoch.adjacent.pairs -->31 adjacent pairs separated; the leader's margin over second is <!-- claim:e2.epoch.top.diff -->+0.041 (CI <!-- claim:e2.epoch.top.lo -->+0.010 to <!-- claim:e2.epoch.top.hi -->+0.073) | [results.json](artifacts/e2/epoch-swebench/results.json) | [`prereg/0001`](prereg/0001-e2-leaderboard-reanalysis.md), [`prereg/0002`](prereg/0002-e2-amend-epoch-reference-rule.md) |
| Policy sensitivity (E3) | Adjacent orderings that hold under every scoring of ambiguous runs: <!-- claim:e3.tb2.identified -->1 on Terminal-Bench, <!-- claim:e3.epoch.identified -->19 on SWE-bench Verified (<!-- claim:e3.epoch.robust -->3 also with sampling error) | [bounds.json](artifacts/e3/bounds.json) | [`prereg/0001`](prereg/0001-e2-leaderboard-reanalysis.md) |
| Library validity (E0, synthetic) | <!-- claim:e0.cells -->22 coverage cells simulated; the cells that fail the coverage rule (a small-sample bootstrap design) are flagged as unsupported and make the library warn; <!-- claim:e0.crosschecks -->22 cross-checks against statsmodels and R agree within tolerance | [coverage.json](artifacts/e0/coverage.json) | [`prereg/0003`](prereg/0003-e0-coverage-grid-part1.md) |
| Factorial study (E1) | Preregistered and frozen; not yet run, so no results | n/a | [`prereg/0004`](prereg/0004-e1-factorial.md), [`prereg/0005`](prereg/0005-e0-glmm-cells.md), [`prereg/0006`](prereg/0006-g6-recovery.md) |

The study plan is in [docs/PLAN.md](docs/PLAN.md) and every design choice, with its reason, in
[DECISIONS.md](DECISIONS.md). Changed or withdrawn claims are dated in
[CORRECTIONS.md](CORRECTIONS.md).

## How it works

```mermaid
flowchart LR
  logs["Per-task run logs<br/>(a leaderboard snapshot, or your own runs)"] --> lib["evalstats<br/>paired, task-clustered intervals<br/>rank sets, variance components, bounds"]
  sim["E0: coverage simulations<br/>frozen grid, many replicates per cell"] -->|"failing designs warn"| lib
  lib --> rep["Confidence reports<br/>docs/leaderboards"]
  pre["Frozen preregistrations<br/>hash chain in prereg/"] --> rep
  rep --> gate["claimcheck and freshness:<br/>every number traced to an artifact"]
```

The unit of inference is the task. Two agents scored on the same tasks are compared task by
task, and the interval is clustered on tasks, because tasks, not runs, are the scarce sample:
repeated runs of one task are correlated, and treating them as independent overstates
precision. A gap counts as resolved only when that paired interval excludes zero. Rank intervals
come from a bootstrap over tasks of the maximum standardized gap, so they hold for every entry
at once.

## E1: the preregistered factorial study

E1 asks how much of the variance in agent success comes from the model and how much from
everything else a leaderboard holds fixed or ignores. Its plan (frozen in
[`prereg/0004`](prereg/0004-e1-factorial.md) with its validation in
[`prereg/0005`](prereg/0005-e0-glmm-cells.md) and [`prereg/0006`](prereg/0006-g6-recovery.md)) fixes, before any
data exist:

- the sample, set by its power analysis: three local open-weight models on 85 mock tasks, each
  task run 18 times per model (4,590 local runs over four UTC days and two time-of-day blocks),
  plus a Claude slice of three tiers on 30 tasks;
- a half-fraction factorial over harness, timeout, step budget, temperature, and concurrency,
  crossed with model, day, and time-of-day block, with replicate runs to measure run-to-run
  sampling directly;
- a logit GLMM decomposition with parametric-bootstrap and task-jackknife intervals, whose
  coverage is checked first on synthetic data shaped like E1 (E0 part 2 and the planted-component
  recovery study G6);
- a power analysis that sets the sample size, Holm-adjusted readings, and the labels a reader
  needs (underpowered families, intervals outside their validated range);
- operational rules: pinned model files and server build, run-code checks before every batch,
  stop rules for infrastructure failures, and sandboxed agent runs with no network.

## Reproduce

```bash
make reproduce     # recomputes E2 and E3 from the pinned snapshots and regenerates the figures
make ship-check    # all fourteen gates on a fresh clone of HEAD
```

`make claimcheck` and `make freshness` check that every number in this README and in
[docs/REPORT.md](docs/REPORT.md) equals what its artifact says now; `make prereg-verify` checks
that every preregistration is frozen, unedited, and older than the data that cite it. The
leaderboard snapshots are pinned by URL, retrieval date, and hash in [docs/DATA.md](docs/DATA.md).

## Limitations and threats to validity

- The reanalysis takes each board's published per-task results as given. Entries differ in runs
  per task, and the boards score ambiguous runs in their own ways; E3 shows how far that moves
  the ranking.
- Intervals describe uncertainty about a population of tasks like the benchmark's, not about the
  fixed tasks themselves. Adjacent-pair intervals are per comparison and unadjusted, so a
  multiplicity-adjusted reading would separate even fewer pairs.
- Coverage is validated by simulation at designs like these, not guaranteed elsewhere; the one
  failing cell (a small-sample bootstrap) makes the library warn.
- E1 has not run. Its tasks are procedurally generated mock tasks, so its results, once banked,
  will be results on those tasks, not on real-world work.

## Responsible reporting

This project reports statistical resolution only. It does not assess any submission's
correctness, never speculates about why an entry scored as it did, and states for every ordering
whether it is or is not statistically supported. The data are used under their licenses
(Apache-2.0 and CC-BY-4.0; see [docs/DATA.md](docs/DATA.md)).

## Related work

Every reference below was opened and checked (title, authors, year, URL) and is recorded in
[artifacts/citecheck.json](artifacts/citecheck.json).

- [Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations](https://arxiv.org/abs/2411.00640) (Miller)
- [Signal and Noise: A Framework for Reducing Uncertainty in Language Model Evaluation](https://arxiv.org/abs/2508.13144) (Heineman and others)
- [Quantifying infrastructure noise in agentic coding evals](https://www.anthropic.com/research/infrastructure-noise) (Anthropic)
- [Clarifying limitations of time horizon](https://metr.org/notes/2026-01-22-time-horizon-limitations/) (METR)
- [Measuring what Matters: Construct Validity in Large Language Model Benchmarks](https://arxiv.org/abs/2511.04703) (Bean and others)
- [tinyBenchmarks: evaluating LLMs with fewer examples](https://arxiv.org/abs/2402.14992) (Polo and others)
- [Inference for Ranks with Applications to Mobility across Neighbourhoods and Academic Achievement](https://doi.org/10.1093/restud/rdad006) (Mogstad and others)
- [Confidence Intervals for Partially Identified Parameters](https://doi.org/10.1111/j.1468-0262.2004.00555.x) (Imbens and Manski)
- [Terminal-Bench: Benchmarking Agents on Hard, Realistic Tasks in Command Line Interfaces](https://arxiv.org/abs/2601.11868) and [SWE-bench](https://arxiv.org/abs/2310.06770), the benchmarks reanalyzed here

## How this was built

Raj Kashikar built this with Claude Code working as an autonomous research engineer from a
written brief ([docs/BUILD_BRIEF.md](docs/BUILD_BRIEF.md)), under rules the repository enforces
on itself: no hand-typed result numbers, preregistration before confirmatory data, failures
published beside successes, and no paid compute. Progress and open items are in
[STATUS.md](STATUS.md).

## Citation and license

Cite with [CITATION.cff](CITATION.cff). Code under the [MIT License](LICENSE); leaderboard data
under their own licenses ([docs/DATA.md](docs/DATA.md)).
