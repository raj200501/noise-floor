# noise-floor: technical report

**How much of a leaderboard gap is the model? Rank confidence on public agent leaderboards, a statistics library with simulation-checked intervals, and a preregistered factorial design.**

*Report state as of 2026-11-11.* E0 part 1, E2, and E3 on the public boards are complete; E1 and its validation studies (E0 part 2 and G6) are frozen and have not run, so there are no E1 results and no E3 bounds on E1's runs. Every result number carries a hidden marker that ties it to a banked artifact.

## Abstract

Agentic leaderboards order submissions by small score gaps, while each score also depends on the tasks sampled, the runs per task, and how runs that ended in infrastructure errors or timeouts were scored. evalstats, a typed Python library, makes task-paired and task-clustered comparisons, simultaneous rank-confidence sets, worst-case bounds for ambiguous runs, logit mixed-model (GLMM) variance decomposition, and minimum-detectable-effect (MDE) calculations the default, and it warns when an interval method is used where its preregistered coverage simulation failed. Under a preregistered plan we reanalyzed public leaderboards with per-task results. On Terminal-Bench 2.0 (TB2; <!-- claim:report.e2.tb2.entries -->66 entries, <!-- claim:report.e2.tb2.tasks -->89 tasks), <!-- claim:report.e2.tb2.supported -->2 of <!-- claim:report.e2.tb2.pairs -->65 adjacent orderings have a paired 95% interval that excludes zero, the median simultaneous rank interval spans <!-- claim:report.e2.tb2.width -->35 ranks, and <!-- claim:report.e2.tb2.first -->15 entries could rank first. On SWE-bench Verified from the Epoch AI Benchmarking Hub (<!-- claim:report.e2.ep.entries -->32 entries, at most <!-- claim:report.e2.ep.tasks_max -->484 tasks each), <!-- claim:report.e2.ep.supported -->3 of <!-- claim:report.e2.ep.pairs -->31 adjacent orderings are supported on each pair's shared tasks, the median rank interval on the <!-- claim:report.e2.ep.rank_tasks -->454 tasks every entry attempted spans <!-- claim:report.e2.ep.width -->21 ranks, and <!-- claim:report.e2.ep.first -->4 entries could rank first. Under every way of scoring ambiguous runs (infrastructure errors, timeouts, and flagged runs), <!-- claim:report.e3.tb2.identified -->1 adjacent ordering on TB2 and <!-- claim:report.e3.ep.identified -->19 on SWE-bench Verified keep their order, many of the latter simply because neither entry has an ambiguous sample; <!-- claim:report.e3.tb2.robust -->0 and <!-- claim:report.e3.ep.robust -->3 of them also have a 95% interval that excludes zero. In the synthetic coverage study (<!-- claim:report.e0.cells -->22 cells), every cell behind these results passes; the small-sample paired bootstrap cell fails and now triggers a library warning, and the cells for the GLMM and Wilson intervals have not run. Each entry is one configuration of model, harness, and settings, so these results concern entries as run, not how much of a gap is the model. E1, a factorial study of model, harness, timeout, step budget, temperature, concurrency, day, and time-of-day effects on mock tasks, is designed to measure that split on its own tasks; it is frozen with its validation plan and has not run, and we give its design and no results.

## Introduction

A leaderboard turns runs into an ordering. Agentic runs are expensive, so boards rest on tens to hundreds of tasks with few runs each, and their scores move with more than the model: the harness, resource limits, timeouts, when a run happened, sampling, and the rule for runs that ended abnormally. [Segato (Anthropic)](https://www.anthropic.com/research/infrastructure-noise) reports that container resource settings moved TB2 scores by several percentage points for Claude models, and [Bean et al.](https://arxiv.org/abs/2511.04703) find that few LLM benchmarks report uncertainty estimates or statistical tests. A reader needs to know which gaps the data resolve, not only which entry is on top.

We ask how much of a leaderboard gap is the model. Existing data answer a narrower, prior question: which gaps are too large for task and run sampling alone to explain, and which survive other rules for scoring ambiguous runs. For public leaderboards with per-task results, a plan frozen before any interval was computed reports which adjacent orderings are supported, how wide simultaneous rank intervals are, and how far rankings move under those rules. Each entry is one configuration of model, harness, and settings, so even a supported gap is a difference between entries as run and does not by itself say how much of it is the model. E1, a preregistered factorial study that varies model, harness, and run settings on mock tasks, is designed to measure how success splits among them; it has not run. Both rest on evalstats, which makes the task the unit of inference, because runs of the same task are correlated and treating them as independent overstates precision ([Miller](https://arxiv.org/abs/2411.00640)).

| Part | State | Artifact | Plan |
|---|---|---|---|
| evalstats library | in the repository, typed and tested | `src/evalstats/` | none |
| E0 part 1: interval coverage (synthetic) | complete | `artifacts/e0/coverage.json` | [`prereg/0003`](../prereg/0003-e0-coverage-grid-part1.md) |
| E2: supported orderings on public boards | complete | `artifacts/e2/*/results.json` | [`prereg/0001`](../prereg/0001-e2-leaderboard-reanalysis.md), [`prereg/0002`](../prereg/0002-e2-amend-epoch-reference-rule.md) |
| E3: policy bounds on public boards | complete | `artifacts/e3/bounds.json` | [`prereg/0001`](../prereg/0001-e2-leaderboard-reanalysis.md) |
| GLMM interval-method pilots (synthetic) | exploratory | `artifacts/method_pilots/summary.json` | not preregistered |
| E0 part 2 and G6: E1 interval coverage (synthetic) | frozen, not run | none yet | [`prereg/0005`](../prereg/0005-e0-glmm-cells.md), [`prereg/0006`](../prereg/0006-g6-recovery.md) |
| E1: factorial study on mock tasks | frozen, not run | none yet | [`prereg/0004`](../prereg/0004-e1-factorial.md) |
| E4: efficient task subsets | not attempted | none | none |

The confidence reports ([TB2](leaderboards/tb2.md), [SWE-bench Verified](leaderboards/epoch-swebench.md)) list every entry, adjacent pair, rank interval, and bound, and regenerate from pinned snapshots.

## Related work

**Statistics for eval scores.** [Miller](https://arxiv.org/abs/2411.00640) treats eval questions as draws from a super-population and gives clustered standard errors, paired differences, and power analysis; evalstats makes this view the default and extends it to rank sets, partial identification, and variance partitions. [Heineman et al.](https://arxiv.org/abs/2508.13144) define benchmark signal and noise, where noise is variability between training checkpoints rather than between runs. [Bean et al.](https://arxiv.org/abs/2511.04703) recommend power-justified sample sizes and uncertainty estimates for primary scores.

**Noise in agentic evals.** [Segato (Anthropic)](https://www.anthropic.com/research/infrastructure-noise) ran TB2 with Claude models under several container resource configurations; time-of-day effects were noted anecdotally, and other model families were not rigorously tested. [Kwa (METR)](https://metr.org/notes/2026-01-22-time-horizon-limitations/) describes wide intervals for time-horizon estimates and their sensitivity to modeling and human-baseline choices. The reanalyzed boards build on [Terminal-Bench](https://arxiv.org/abs/2601.11868) (Merrill et al.) and [SWE-bench](https://arxiv.org/abs/2310.06770) (Jimenez et al.).

**Methods.** Rank intervals follow the simultaneous confidence sets for ranks of [Mogstad, Romano, Shaikh, and Wilhelm](https://doi.org/10.1093/restud/rdad006), computed here from a task-bootstrap critical value; bounded scores use the intervals of [Imbens and Manski](https://doi.org/10.1111/j.1468-0262.2004.00555.x) for partially identified parameters; GLMMs are fitted with [lme4](https://doi.org/10.18637/jss.v067.i01) (Bates et al.), with the latent-scale variance partition of [Nakagawa and Schielzeth](https://doi.org/10.1111/j.2041-210x.2012.00261.x); and bootstrap intervals follow [Davison and Hinkley](https://doi.org/10.1017/CBO9780511802843) and, for BCa, [Efron](https://doi.org/10.1080/01621459.1987.10478410). [tinyBenchmarks](https://arxiv.org/abs/2402.14992) (Polo et al.) picks small evaluation sets for static benchmarks by item response theory; the agentic analogue (E4) was not attempted.

**Tools and data.** evalstats reads [Inspect](https://inspect.aisi.org.uk/) logs, and Inspect's basic agent is a harness level in E1. E2 uses the [TB2 leaderboard dataset](https://huggingface.co/datasets/harborframework/terminal-bench-2-leaderboard) and the [Epoch AI Benchmarking Hub](https://epoch.ai/benchmarks).

## Method

### Estimands and intervals

Results are long-format rows, one per run: an entry, a task, a run index, an outcome in the unit interval, and an optional status (`ok`, `timeout`, `infra_error`, `flagged`) and covariates. An entry's score is the mean over tasks of its per-task mean outcome. Intervals treat tasks as a sample from a population like the benchmark's: they describe how a score could move under a fresh draw of tasks and runs, not under infrastructure or harness changes. The default, `cluster_t`, averages runs within each task and builds a Student t interval on `T - 1` degrees of freedom from the per-task values, for a difference on the tasks both entries attempted. Task bootstraps (percentile, BCa) resample whole tasks. Near the floor or the ceiling, `cluster_wilson` takes a Wilson interval at the design-effect sample size, and `mover_wilson` combines the rates' Wilson intervals for a paired difference; `cluster_robust_mean` gives CR1 errors, and `contrast_difference` a task-paired interval for a difference of contrasts.

### Rank-confidence sets

For M entries on T shared tasks, each pairwise difference `d_jk` has a task-paired standard error `se_jk`. A task bootstrap of the largest studentized deviation over all pairs gives a critical value c at which every interval `d_jk +/- c * se_jk` holds at once with about the nominal probability. Entry j is surely better than k when `d_jk - c * se_jk > 0`, and j's simultaneous rank interval is `[1 + #(surely better than j), M - #(surely worse than j)]`; marginal intervals are secondary. An adjacent ordering can be supported by its own paired interval while the rank intervals overlap, since the paired test controls the error of a single comparison and the rank intervals the error over all of them.

### Policy-sensitivity bounds

Runs with status `infra_error`, `timeout`, or `flagged` are ambiguous. `policy_bounds` scores each entry under `as_scored`, `fail`, `success`, and `exclude`. With no assumption about ambiguous runs the score lies between its `fail` and `success` values, and the Imbens and Manski interval covers the true score rather than the whole set. An adjacent ordering is *identified* when the lower bound of the per-task difference is positive, and *robust* when its interval also excludes zero. A success rate's interval is cut to the unit interval, which never lowers coverage (a change made before release and recorded in [CORRECTIONS.md](../CORRECTIONS.md)).

### Variance decomposition

`decompose` fits a logit GLMM with crossed random intercepts by lme4's `glmer` through rpy2 (Laplace, `nAGQ = 1`) and splits the latent variance into random-effect variances, the variance over runs of each fixed factor's contribution to the linear predictor, and the logistic residual `pi^2/3`, with shares of their sum. The residual absorbs run-to-run sampling and omitted interactions, so E1 measures sampling separately, from replicate runs. No interval uses the fit's model-based (Wald) standard errors; the task-jackknife intervals below are Student t intervals with jackknife standard errors. In E1's frozen configuration, random variances get a parametric bootstrap with the `basic_log` rule; fixed contributions get their range over a Bonferroni box of task-jackknife coefficient intervals, which keeps task-to-task variation in factor effects that a bootstrap from the main-effects model omits; shares get the `plugin` rule, which maps each component's interval through `v / (v + rest)` with the rest at its estimate; and MDE rows get the bias-corrected percentile interval of their bootstrap replicates. Without R, a labeled statsmodels variational-Bayes fit is used, and it backs no headline result.

### Power and MDE

For a paired comparison on T tasks with k runs per entry and task, the per-task difference has variance `between + within / k`. `mde`, `power_at`, and `tasks_needed` invert this by the normal approximation, and `design_from_logit` and `design_from_results` supply the two variances from a fitted logit model or from observed runs.

### Validity records and integrity controls

Interval methods are checked by cells of a preregistered simulation grid (E0). Only E0 part 1 has run; the cells for `cluster_wilson`, `mover_wilson`, `contrast_difference`, and the GLMM intervals are in E0 part 2, which has not. The interval for a bounded difference between two entries, behind E3's robust counts, has no cell of its own; the bound cells that the plan maps to E3 test one entry's bounded score. Failing cells are listed in `src/evalstats/data/validity.json`, and the score, comparison, contrast, rank, bound, cluster-robust, and decomposition functions emit `UnsupportedDesignWarning`, naming an alternative, on a design at least as small as a failing cell's. A method or design that no banked cell covers raises no warning, so the absence of a warning is not evidence of coverage. A typical call sequence:

```python
from evalstats import (
    design_from_results,
    mde,
    paired_difference,
    policy_bounds,
    rank_confidence,
    read_csv,
    tasks_needed,
)

results = read_csv("runs.csv")  # one row per run: model, task, run, outcome, status
cmp = paired_difference(results, "agent-a", "agent-b")  # paired by task, cluster_t
print(cmp.difference.estimate, cmp.difference.lo, cmp.difference.hi, cmp.supported)

ranks = rank_confidence(results, simultaneous=True, n_boot=4000, seed=0)
bounds = policy_bounds(results)  # infra_error, timeout, flagged are ambiguous
design = design_from_results(results, "agent-a", "agent-b")
print(mde(design, n_tasks=89, runs_per_task=5), tasks_needed(0.05, design, runs_per_task=5))
```

The repository checks itself the same way. Plans under `prereg/` are hashed into `prereg/CHAIN.jsonl` and never edited, changes are dated amendments, and `make prereg-verify` checks that every artifact citing a plan was committed after it. Each result number here follows a marker naming a claim in `artifacts/claims.json` (artifact, JSON pointer, derivation, format); `make claimcheck` rejects any other number not covered by a narrow, categorized exemption, and `make freshness` re-derives every claim from its artifact. Public snapshots are pinned by URL, revision or archive hash, and retrieval date, and must reproduce the banked results byte for byte.

## Experiments

### E0 part 1: interval coverage (synthetic data)

[`prereg/0003`](../prereg/0003-e0-coverage-grid-part1.md) froze a grid of cells before any replicate was banked. Each cell fixes an interval method, a data-generating process (binary outcomes from a logit model with normal task and model-by-task effects, with truths computed by quadrature over the task population), the nominal level, and a replicate count. Cells mimic the boards (TB2-shaped: <!-- claim:report.e0.tb2.tasks -->89 tasks with <!-- claim:report.e0.tb2.runs -->5 runs per model and task; Epoch-shaped: <!-- claim:report.e0.ep.tasks -->484 tasks with <!-- claim:report.e0.ep.runs -->1 run) and small designs (<!-- claim:report.e0.small.tasks_min -->20 to <!-- claim:report.e0.small.tasks_max -->40 tasks), all at a nominal level of 0.95. The pass rule is the brief's: "at least 2,000 replicates per cell (500 for bootstrap or GLMM intervals); coverage no lower than nominal minus 1.5 points minus twice its Monte Carlo standard error". Over-coverage is reported, not failed. The <!-- claim:report.e0.headline -->8 headline cells, behind the E2 headline intervals and the E3 bounds on each entry's score, must pass, and the ship-check recounts every cell from its raw replicate lines. The interval on a pair's bounded difference, which E3's robust counts use, applies the same construction to per-task differences and has no cell of its own. Cross-checks compare evalstats with statsmodels, SciPy, and R at stated tolerances.

### E2: reanalysis of public leaderboards

[`prereg/0001`](../prereg/0001-e2-leaderboard-reanalysis.md), frozen before any interval, ordering, rank set, or bound was computed, asks, with no direction presumed: Q1, what share of adjacent pairs has a paired interval that excludes zero, with entries ordered by their task-level score under the board's scoring rule (this can differ from the order of the board's pooled scores); Q2, how wide simultaneous rank intervals are and how many entries could rank first; Q3, how many adjacent orderings survive every rule for ambiguous runs. A board is "mostly resolved" if its Q1 share is at least 0.5 and "mostly unresolved" otherwise. A scouting pass before freezing, disclosed in the plan, read file formats and counts and checked aggregate scores against the published ones, computing no interval or bound.

**TB2.** The source is the TB2 leaderboard dataset (`Apache-2.0`) at revision `572b2614be2c0cb2527e14f5b1e4026f1072e6c1`, retrieved 2026-11-04: job-level result and config files only, each hashed in the manifest. The unit is a trial and the outcome its reward, a missing reward counting as a failure as on the board. Of <!-- claim:report.e2.tb2.submissions -->75 submissions, <!-- claim:report.e2.tb2.excluded -->9 were excluded by preregistered rules, each with its reason: entries the maintainers removed or rescored in a published integrity update, an entry whose jobs ran another dataset, and entries without job-level results in the documented format. The analysis covers <!-- claim:report.e2.tb2.entries -->66 entries and <!-- claim:report.e2.tb2.runs -->28,775 trials on <!-- claim:report.e2.tb2.tasks -->89 tasks. Every entry has trials on every task, so the primary and secondary analysis sets coincide, and mean runs per task range from <!-- claim:report.e2.tb2.rpt_min -->1 to <!-- claim:report.e2.tb2.rpt_max -->10. Another <!-- claim:report.e2.tb2.unlisted -->315 trials are counted in job totals but not listed by name, so they enter the board's pooled score but not the task-level analysis. The reference rule is the board's (`as_scored`).

**SWE-bench Verified.** The source is the archive of the Epoch AI Benchmarking Hub (`CC-BY-4.0`), pinned by its sha256 and retrieved 2026-11-04, with each row's Inspect log read by HTTP range requests for its header and sample summaries only. The unit is a sample, with a single run per task; errored, limited, and invalidated samples map to `infra_error`, `timeout`, and `flagged`. Of the board's rows, <!-- claim:report.e2.ep.fetch_failures -->3 were excluded because their logs did not open (<!-- claim:report.e2.ep.nolink -->1 had no link). The analysis covers <!-- claim:report.e2.ep.entries -->32 entries and <!-- claim:report.e2.ep.runs -->15,458 samples, at most <!-- claim:report.e2.ep.tasks_max -->484 tasks per entry; rank intervals use the <!-- claim:report.e2.ep.rank_tasks -->454 tasks every entry attempted (<!-- claim:report.e2.ep.dropped -->30 dropped). The reference rule is Epoch's: errored samples leave the denominator and limited ones keep their recorded score ([`prereg/0002`](../prereg/0002-e2-amend-epoch-reference-rule.md)).

Q1 uses `paired_difference` (`cluster_t`, with the task bootstrap as secondary), Q2 `rank_confidence` (simultaneous, with marginal intervals as secondary), and Q3 `policy_bounds`; seeds are fixed in the plan.

### E3: policy-sensitivity bounds on the public boards

E3 applies Q3 to both boards (`artifacts/e3/bounds.json`). E3 on E1's runs is preregistered with E1.

### Exploratory: the noise floor of each board's own design

Not preregistered (`D-035`, [`tools/noise_floor.py`](../tools/noise_floor.py)): for each adjacent pair, the standard error implied by its banked paired interval gives the smallest true gap that the pair's own design, its shared tasks and runs, detects at two-sided level 0.05 with power 0.8, by the normal approximation of `mde`. Each board's median over adjacent pairs is banked beside the median absolute adjacent gap in `artifacts/e2/noise_floor.json`.

### E1: the preregistered factorial study (not run)

E1 is frozen in [`prereg/0004`](../prereg/0004-e1-factorial.md) with its validation studies [`prereg/0005`](../prereg/0005-e0-glmm-cells.md) (E0 part 2) and [`prereg/0006`](../prereg/0006-g6-recovery.md) (G6), before any of their data exist and after repeated adversarial reviews by reviewer agents (decisions `D-022` to `D-034` in [DECISIONS.md](../DECISIONS.md)). Its tasks are procedurally generated mock tasks, so its results will hold for those tasks only.

**Design.** The local models [`qwen3-1.7b`](https://huggingface.co/Qwen/Qwen3-1.7B-GGUF), [`qwen3-4b`](https://huggingface.co/Qwen/Qwen3-4B-GGUF) (Qwen3), and [`granite-4.0-micro`](https://huggingface.co/ibm-granite/granite-4.0-micro-GGUF) (Granite) were chosen in an exploratory screening because their screening success rates fell in the brief's target range (it asks that mean success for each model configuration is between 0.2 and 0.8) and their runs fit the shorter timeout; each is pinned by revision, file sha256, server arguments, and server build digest. The Claude slice runs haiku, sonnet, and opus through headless Claude Code. Tasks come from nf-tools v2, this project's Docker-free suite of tool-use tasks on a mock file system with deterministic graders. For each local model and each of the 85 tasks, the local arm runs the 16 runs of a half fraction of the two-level factors harness (minimal-agent, this project's scaffold, against Inspect's basic agent), timeout, step budget, temperature, and concurrency, plus reruns of the `replicates` configurations to measure run-to-run sampling: 4,590 local runs over 4 UTC days with 2 time-of-day blocks each, with day and block as fixed blocking factors. Each Claude tier runs its tasks under minimal-agent and under Claude Code, for 180 Claude-slice runs; Claude-versus-local comparisons are not crossed with the scaffold and are labeled scaffold-confounded. The frozen values:

```json
{
  "e1-plan": {
    "models": ["qwen3-1.7b", "qwen3-4b", "granite-4.0-micro"],
    "claude_tiers": ["claude-haiku", "claude-sonnet", "claude-opus"],
    "families": ["config", "csv_sum", "pointer", "derive", "violation"],
    "difficulties": [1, 2, 3],
    "tasks_per_family": 17,
    "claude_tasks_per_family": 6,
    "levels": {
      "harness": ["minimal", "native"],
      "timeout": [150.0, 300.0],
      "steps": [8, 15],
      "temperature": [0.0, 0.7],
      "concurrency": [1, 2]
    },
    "runs_per_cell": 16,
    "replicates": [[0, 1, 1, 0, 0], [0, 1, 1, 1, 1]],
    "claude_runs_per_task": 2,
    "days": 4,
    "blocks_per_day": 2,
    "batch_size": 10,
    "max_tokens": 384
  },
  "e1-design": {
    "tasks": 85,
    "claude_tasks": 30,
    "local_runs": 4590,
    "replicate_runs": 510,
    "claude_runs": 180
  },
  "e1-analysis": {
    "random": ["task", "model:task"],
    "fixed": ["model", "harness", "timeout", "steps", "temperature", "concurrency", "day", "block"],
    "n_boot": 200,
    "nagq": 1,
    "boot_interval": "basic_log",
    "fixed_interval": "jackknife",
    "share_interval": "plugin",
    "mde": {
      "designs": [[20, 1], [20, 3], [20, 5], [40, 1], [40, 3], [40, 5], [89, 1], [89, 3], [89, 5], [200, 1], [200, 3], [200, 5], [500, 1], [500, 3], [500, 5]],
      "reference": {
        "harness": "minimal",
        "timeout": "300.0",
        "steps": "15",
        "temperature": "0.0",
        "concurrency": "1"
      },
      "gaps": [0.02, 0.05, 0.1],
      "alpha": 0.05,
      "power": 0.8,
      "interval": "bc"
    }
  }
}
```

Each `replicates` entry indexes the levels in `levels` order: the first is the MDE reference configuration (minimal-agent, the longer timeout, the larger step budget, and the lower temperature and concurrency), and the second is the same configuration at the higher temperature and concurrency.

**Questions and sizing.** RQ1 splits the latent variance among task, model-by-task, and the fixed factors, reading a factor as non-zero when its jackknife interval's lower end is above zero, with no multiplicity adjustment over the eight factors. RQ2 gives the MDE at the leaderboard-like designs in `mde.designs`, at two-sided level 0.05, and the tasks needed for each gap in `mde.gaps`. RQ3 compares harness effects and models, and a contrast counts as supported on a Holm-adjusted p-value below 0.05 within its family. The local sample is the smallest number of tasks per family at which every local harness contrast and model pair reaches power 0.8 under the Holm reading, at effects stated in the plan, which gives 17 tasks per family. The Claude slice is bounded by the usage caps instead (at most 6 runs a day), so its size is the G7 floor with a margin; G7 asks for at least 1,000 task runs in total, at least 150 of them on Claude tiers. The harness-difference family and every Claude family fall below that power and will be labeled underpowered.

**Validation.** E0 part 2 (175 cells, synthetic) analyzes E1-shaped datasets by exactly the E1 analysis, including a group with planted task-by-factor variation. A failing cell marks every interval of its kind as resting on an unsupported method and fails G5, unless a dated amendment frozen before E1 data replaces the method with new cells that pass. G6 analyzes 200 datasets with planted components under the same rule and reports each component's bias and the power of the fixed-factor reading, and a post-E1 check reanalyzes 200 datasets simulated at the E1 estimates.

**Safeguards.** The runners refuse to start on any mismatch with the frozen hashes, on uncommitted changes, before G6 and E0 part 2 are banked, or under other analysis code; while an arm runs, its log shows only banked and error counts; and the analysis waits for both arms to end. A batch fails on a model hash mismatch, a server failure, 3 consecutive infrastructure errors, or an early runner exit; an arm stops when infrastructure errors reach more than 5 percent and at least 5 of its banked runs; no local batch starts after day 7 of the arm, and the Claude slice ends 45 days after the freeze at the latest. Local runs execute in namespaces with loopback networking only and credentials removed, and their agent tools are in-process mocks; Claude Code children get only their file tools, confined to a disposable workspace; and a single client enforces the usage caps and refuses any billed login.

### Deviations from preregistration

- [`prereg/0002`](../prereg/0002-e2-amend-epoch-reference-rule.md) amends [`prereg/0001`](../prereg/0001-e2-leaderboard-reanalysis.md): the policy named as Epoch's reference rule, `exclude`, drops every ambiguous run, while Epoch drops only errored samples. The amendment, frozen before any E2 result existed, follows Epoch's rule.
- [`prereg/0005`](../prereg/0005-e0-glmm-cells.md) amends [`prereg/0003`](../prereg/0003-e0-coverage-grid-part1.md): beyond the GLMM cells that the original announced, it adds local-arm, Claude-slice, sampling, and bound cells and the map from E1 headline intervals to cells.
- E1's plan states two departures from the brief: Claude success is expected above the target range, because the tiers could not be piloted here, and CPU and memory are held fixed. A third, no Terminal-Bench arm through Harbor because no Docker daemon was available, is recorded before any data in `D-010` ([DECISIONS.md](../DECISIONS.md)), not in the plan.

No other amendment of `prereg/0001` or `prereg/0003` exists.

## Results

### E0 part 1: coverage (synthetic data)

| Cell | Interval it backs | Covered / replicates | Coverage | MCSE |
|---|---|---:|---:|---:|
| `S-tb2-t` | TB2 entry scores (`cluster_t`) | <!-- claim:report.e0.S-tb2-t.covered -->1,900 / <!-- claim:report.e0.S-tb2-t.reps -->2,000 | <!-- claim:report.e0.S-tb2-t.cov -->0.950 | <!-- claim:report.e0.S-tb2-t.mcse -->0.005 |
| `S-ep-t` | Epoch entry scores (`cluster_t`) | <!-- claim:report.e0.S-ep-t.covered -->1,912 / <!-- claim:report.e0.S-ep-t.reps -->2,000 | <!-- claim:report.e0.S-ep-t.cov -->0.956 | <!-- claim:report.e0.S-ep-t.mcse -->0.005 |
| `P-tb2-t` | TB2 adjacent pairs (`cluster_t`) | <!-- claim:report.e0.P-tb2-t.covered -->1,926 / <!-- claim:report.e0.P-tb2-t.reps -->2,000 | <!-- claim:report.e0.P-tb2-t.cov -->0.963 | <!-- claim:report.e0.P-tb2-t.mcse -->0.004 |
| `P-ep-t` | Epoch adjacent pairs (`cluster_t`) | <!-- claim:report.e0.P-ep-t.covered -->1,909 / <!-- claim:report.e0.P-ep-t.reps -->2,000 | <!-- claim:report.e0.P-ep-t.cov -->0.955 | <!-- claim:report.e0.P-ep-t.mcse -->0.005 |
| `R-tb2-sim` | TB2 simultaneous rank intervals | <!-- claim:report.e0.R-tb2-sim.covered -->500 / <!-- claim:report.e0.R-tb2-sim.reps -->500 | <!-- claim:report.e0.R-tb2-sim.cov -->1.000 | <!-- claim:report.e0.R-tb2-sim.mcse -->0.000 |
| `R-ep-sim` | Epoch simultaneous rank intervals | <!-- claim:report.e0.R-ep-sim.covered -->500 / <!-- claim:report.e0.R-ep-sim.reps -->500 | <!-- claim:report.e0.R-ep-sim.cov -->1.000 | <!-- claim:report.e0.R-ep-sim.mcse -->0.000 |
| `B-tb2` | TB2 policy bounds (Imbens and Manski) | <!-- claim:report.e0.B-tb2.covered -->2,000 / <!-- claim:report.e0.B-tb2.reps -->2,000 | <!-- claim:report.e0.B-tb2.cov -->1.000 | <!-- claim:report.e0.B-tb2.mcse -->0.000 |
| `B-ep` | Epoch policy bounds (Imbens and Manski) | <!-- claim:report.e0.B-ep.covered -->1,956 / <!-- claim:report.e0.B-ep.reps -->2,000 | <!-- claim:report.e0.B-ep.cov -->0.978 | <!-- claim:report.e0.B-ep.mcse -->0.003 |

Every headline cell passes, and so does every other cell except `P-small-boot`, the paired task bootstrap at <!-- claim:report.e0.fail.tasks -->20 tasks with <!-- claim:report.e0.fail.runs -->1 run per task, which covered in <!-- claim:report.e0.fail.covered -->454 of <!-- claim:report.e0.fail.reps -->500 replicates (coverage <!-- claim:report.e0.fail.cov -->0.908, MCSE <!-- claim:report.e0.fail.mcse -->0.013), below the pass point. It is listed as unsupported, so `paired_difference` with the bootstrap on <!-- claim:report.e0.fail.tasks -->20 or fewer tasks warns and points to `cluster_t`, whose cell at that design (`P-small-t`) passes with coverage <!-- claim:report.e0.psmallt.cov -->0.944. Every rank cell covered in all of its replicates, and `B-tb2` in all <!-- claim:report.e0.B-tb2.covered -->2,000: at these designs the rank and bound intervals are conservative, so the rank widths below are likely wider than an exact-coverage method would give. All <!-- claim:report.e0.cc.checks -->22 cross-checks agree within tolerance; the largest difference, <!-- claim:report.e0.cc.maxdiff -->0.022 at a tolerance of <!-- claim:report.e0.cc.maxdiff_tol -->0.25, is between lme4's Laplace fit and statsmodels' variational-Bayes fit, which are different estimators. Coverage of `cluster_wilson`, `mover_wilson`, and the GLMM intervals at the E1 design awaits E0 part 2.

### E2: which orderings are supported

| | TB2 | SWE-bench Verified (Epoch AI) |
|---|---:|---:|
| Entries analyzed | <!-- claim:report.e2.tb2.entries -->66 | <!-- claim:report.e2.ep.entries -->32 |
| Tasks used for rank intervals | <!-- claim:report.e2.tb2.tasks -->89 | <!-- claim:report.e2.ep.rank_tasks -->454 |
| Runs analyzed | <!-- claim:report.e2.tb2.runs -->28,775 | <!-- claim:report.e2.ep.runs -->15,458 |
| Adjacent orderings supported (`cluster_t`) | <!-- claim:report.e2.tb2.supported -->2 of <!-- claim:report.e2.tb2.pairs -->65 | <!-- claim:report.e2.ep.supported -->3 of <!-- claim:report.e2.ep.pairs -->31 |
| Share supported | <!-- claim:report.e2.tb2.share -->0.031 | <!-- claim:report.e2.ep.share -->0.097 |
| Supported by the task bootstrap (secondary) | <!-- claim:report.e2.tb2.supported_boot -->2 | <!-- claim:report.e2.ep.supported_boot -->3 |
| Preregistered label | mostly unresolved | mostly unresolved |
| Median simultaneous rank-interval width (ranks) | <!-- claim:report.e2.tb2.width -->35 | <!-- claim:report.e2.ep.width -->21 |
| Entries whose rank interval includes first place | <!-- claim:report.e2.tb2.first -->15 | <!-- claim:report.e2.ep.first -->4 |
| Simultaneous critical value | <!-- claim:report.e2.tb2.crit -->4.05 | <!-- claim:report.e2.ep.crit -->3.75 |

![TB2: each adjacent gap with its paired, task-clustered interval](figures/e2-tb2-gaps.svg)

*TB2: every adjacent gap with its 95% interval, drawn by `make report` from `artifacts/e2/tb2/results.json`. Also drawn: [`e2-epoch-swebench-gaps.svg`](figures/e2-epoch-swebench-gaps.svg) and the per-entry score figures [`e2-tb2-scores.svg`](figures/e2-tb2-scores.svg) and [`e2-epoch-swebench-scores.svg`](figures/e2-epoch-swebench-scores.svg).*

**TB2.** <!-- claim:report.e2.tb2.supported -->2 of <!-- claim:report.e2.tb2.pairs -->65 adjacent orderings (<!-- claim:report.e2.tb2.share -->0.031) are supported, and the task bootstrap also supports <!-- claim:report.e2.tb2.supported_boot -->2; the board is mostly unresolved. The supported orderings, both at the bottom of the board, are `spoox-o-m__GPT-5-Nano` over `terminus-2__AfterQuery-GPT-OSS-20B` (<!-- claim:report.e2.tb2.pair62.diff -->+0.049, interval <!-- claim:report.e2.tb2.pair62.lo -->+0.006 to <!-- claim:report.e2.tb2.pair62.hi -->+0.093) and `BashAgent__TermiGen-32B` over `little-coder__qwen3.5-9b` (<!-- claim:report.e2.tb2.pair64.diff -->+0.070, <!-- claim:report.e2.tb2.pair64.lo -->+0.028 to <!-- claim:report.e2.tb2.pair64.hi -->+0.111). The first and second entries differ by <!-- claim:report.e2.tb2.pair0.diff -->+0.018 (<!-- claim:report.e2.tb2.pair0.lo -->-0.041 to <!-- claim:report.e2.tb2.pair0.hi -->+0.077), and the largest unsupported adjacent gap is <!-- claim:report.e2.tb2.pair44.diff -->+0.063 (<!-- claim:report.e2.tb2.pair44.lo -->-0.032 to <!-- claim:report.e2.tb2.pair44.hi -->+0.158), between `Mux__Claude-Opus-4.5` and `Simplai-Agent__Claude-Sonnet-4.6`. The top entry, `vix__claude-opus-4-7`, scores <!-- claim:report.e2.tb2.top.score -->0.899 (<!-- claim:report.e2.tb2.top.lo -->0.852 to <!-- claim:report.e2.tb2.top.hi -->0.946), with a simultaneous rank interval from <!-- claim:report.e2.tb2.top.best -->1 to <!-- claim:report.e2.tb2.top.worst -->15.

**SWE-bench Verified.** <!-- claim:report.e2.ep.supported -->3 of <!-- claim:report.e2.ep.pairs -->31 adjacent orderings (<!-- claim:report.e2.ep.share -->0.097) are supported, and the task bootstrap also supports <!-- claim:report.e2.ep.supported_boot -->3: `claude-opus-4-7_max` over `gemini-3.5-flash_high` (<!-- claim:report.e2.ep.pair0.diff -->+0.041, <!-- claim:report.e2.ep.pair0.lo -->+0.010 to <!-- claim:report.e2.ep.pair0.hi -->+0.073), `gemini-2.5-pro` over `gpt-4.1-2025-04-14` (<!-- claim:report.e2.ep.pair29.diff -->+0.086, <!-- claim:report.e2.ep.pair29.lo -->+0.043 to <!-- claim:report.e2.ep.pair29.hi -->+0.128), and `gpt-4.1-2025-04-14` over `gpt-4o-2024-11-20` (<!-- claim:report.e2.ep.pair30.diff -->+0.177, <!-- claim:report.e2.ep.pair30.lo -->+0.134 to <!-- claim:report.e2.ep.pair30.hi -->+0.220). The largest unsupported gap is <!-- claim:report.e2.ep.pair27.diff -->+0.031 (<!-- claim:report.e2.ep.pair27.lo -->-0.010 to <!-- claim:report.e2.ep.pair27.hi -->+0.072), between `claude-3-7-sonnet-20250219` and `qwen3.6-plus`, and the adjacent `gpt-5.1-2025-11-13_high` entries, listed separately with their own logs, differ by <!-- claim:report.e2.ep.pair23.diff -->+0.021 (<!-- claim:report.e2.ep.pair23.lo -->-0.019 to <!-- claim:report.e2.ep.pair23.hi -->+0.060). The top entry scores <!-- claim:report.e2.ep.top.score -->0.835 (<!-- claim:report.e2.ep.top.lo -->0.802 to <!-- claim:report.e2.ep.top.hi -->0.868); its simultaneous rank interval is <!-- claim:report.e2.ep.top.best -->1 to <!-- claim:report.e2.ep.top.worst -->4 and the second entry's <!-- claim:report.e2.ep.second.best -->1 to <!-- claim:report.e2.ep.second.worst -->19, so the simultaneous intervals do not separate a pair that its own paired interval supports.

Neither board resolves most of its adjacent orderings. The supported orderings are among each board's largest adjacent gaps, but support depends on a pair's standard error as well as its gap: on TB2 the unsupported gap of <!-- claim:report.e2.tb2.pair44.diff -->+0.063 exceeds the supported gap of <!-- claim:report.e2.tb2.pair62.diff -->+0.049. The simultaneous critical values (<!-- claim:report.e2.tb2.crit -->4.05 and <!-- claim:report.e2.ep.crit -->3.75) are well above a single comparison's quantile, while the per-pair readings are unadjusted, and an adjustment could only lower their counts. These intervals describe sampling over tasks and runs, not why any entry scored as it did.

### E3: policy sensitivity

| | TB2 | SWE-bench Verified (Epoch AI) |
|---|---:|---:|
| Highest share of ambiguous runs in any entry | <!-- claim:report.e3.tb2.amb_max -->0.528 | <!-- claim:report.e3.ep.amb_max -->0.599 |
| Adjacent orderings identified (hold under every rescoring) | <!-- claim:report.e3.tb2.identified -->1 of <!-- claim:report.e3.tb2.pairs -->65 | <!-- claim:report.e3.ep.identified -->19 of <!-- claim:report.e3.ep.pairs -->31 |
| Adjacent orderings robust (identified, interval excludes zero) | <!-- claim:report.e3.tb2.robust -->0 | <!-- claim:report.e3.ep.robust -->3 |
| Named policies whose ranking differs from the reference | <!-- claim:report.e3.tb2.changes -->3 | <!-- claim:report.e3.ep.changes -->4 |

On TB2 the ambiguous runs include every timeout, agent timeouts (`AgentTimeoutError`) among them, and the board scores these by their recorded reward. The share of ambiguous runs per entry ranges from <!-- claim:report.e3.tb2.amb_min -->0.004 to <!-- claim:report.e3.tb2.amb_max -->0.528, and the bounds let every ambiguous run score anywhere from failure to success, so they are wide next to the adjacent gaps: <!-- claim:report.e3.tb2.identified -->1 of <!-- claim:report.e3.tb2.pairs -->65 adjacent orderings is identified and <!-- claim:report.e3.tb2.robust -->0 are robust. The identified one, `BashAgent__TermiGen-32B` over `little-coder__qwen3.5-9b`, has bounds from <!-- claim:report.e3.tb2.idpair.bound_lo -->0.038 to <!-- claim:report.e3.tb2.idpair.bound_hi -->0.431 on its difference and an interval whose lower end is <!-- claim:report.e3.tb2.idpair.ci_lo -->-0.00002, just below zero. Of `as_scored`, `fail`, `success`, and `exclude`, <!-- claim:report.e3.tb2.changes -->3 give a ranking other than the reference, which is `as_scored` itself. The top entry, ambiguous on a share of <!-- claim:report.e3.tb2.top.amb -->0.128 of its runs, has bounds from <!-- claim:report.e3.tb2.top.lower -->0.870 to <!-- claim:report.e3.tb2.top.upper -->0.998.

On SWE-bench Verified most entries have few ambiguous samples, and <!-- claim:report.e3.ep.amb_zero -->19 have none. The bounds are therefore narrow: <!-- claim:report.e3.ep.identified -->19 of <!-- claim:report.e3.ep.pairs -->31 adjacent orderings are identified and <!-- claim:report.e3.ep.robust -->3 are robust, the same orderings that the paired test supports. The exception is `qwen3.6-plus`, ambiguous on a share of <!-- claim:report.e3.ep.amb_max -->0.599 of its samples, with bounds from <!-- claim:report.e3.ep.q.lower -->0.364 to <!-- claim:report.e3.ep.q.upper -->0.963. Identification here reflects how little is ambiguous, not how large the gaps are: when neither entry of a pair has an ambiguous sample, the bounds collapse to the observed gap, so any positive gap, down to one task, is identified, and only the robust count also accounts for sampling error. All <!-- claim:report.e3.ep.changes -->4 named policies change the ranking, because the board's rule (drop errored samples only) is none of them; on both boards this count records any change in the full order, however small. On TB2 both sampling and the worst-case reading of ambiguous runs leave neighbors unresolved; on SWE-bench Verified mainly sampling does.

### Exploratory: the noise floor of each board's own design

On TB2 the median detectable gap over adjacent pairs is <!-- claim:report.nf.tb2.mde -->0.097, against a median adjacent gap of <!-- claim:report.nf.tb2.gap -->0.007, and <!-- claim:report.nf.tb2.atleast -->1 of <!-- claim:report.e2.tb2.pairs -->65 adjacent gaps reaches its own pair's detectable gap. On SWE-bench Verified the medians are <!-- claim:report.nf.ep.mde -->0.048 and <!-- claim:report.nf.ep.gap -->0.004, and <!-- claim:report.nf.ep.atleast -->2 of <!-- claim:report.e2.ep.pairs -->31 gaps reach their detectable gap. On both boards the typical observed gap between neighbors is far below what the board's own design can detect. Each detectable gap is a fixed multiple of the standard error behind the pair's Q1 interval, larger than the multiple at which that interval excludes zero, so a gap can be supported without reaching it. The comparison restates the unresolved orderings above on the scale of the gap rather than adding evidence, and a gap below its detectable gap does not show that the true gap is small. This describes the boards' designs, not the entries.

### Exploratory: GLMM interval rules (synthetic)

Before E1 was frozen, method pilots on synthetic datasets with planted components chose the rules for random variances, shares, and MDE rows by a selection rule written before the deciding pilot was read (`D-023`, `D-028`); the fixed-factor rule, the task jackknife, was set by argument and then checked in `pilot7a` and `pilot7b` (`D-028`, `D-029`). They are exploratory, smaller than E1, and noisy, with tens to about a hundred datasets each; E0 part 2 and G6, not these pilots, test E1's intervals.

- **Random variances.** In `pilot6` (<!-- claim:report.mp.p6.datasets -->100 datasets, Laplace fits), `basic_log` had the highest minimum coverage, <!-- claim:report.mp.p6.random.min -->0.96, against <!-- claim:report.mp.p6.random.pct -->0.82 for the percentile rule.
- **Shares.** Percentile, bias-corrected, and log-scale intervals of replicate shares cannot cover a planted share of zero: the percentile rule's coverage of the concurrency share, planted at zero, was <!-- claim:report.mp.p6.share.null_pct -->0.00, while `plugin` reached a minimum of <!-- claim:report.mp.p6.share.min -->0.92.
- **MDE rows.** The bias-corrected percentile rule reached a minimum of <!-- claim:report.mp.p6.mde.min -->0.93.
- **Fixed contributions.** Under `nAGQ = 0` (`pilot4` and `pilot5`, <!-- claim:report.mp.p45.datasets -->120 datasets) the harness box covered <!-- claim:report.mp.p45.harness -->0.917, against <!-- claim:report.mp.p6.harness -->0.96 with Laplace fits in `pilot6` (minimum over factors <!-- claim:report.mp.p6.fixed.min -->0.94). `pilot7a` and `pilot7b` (<!-- claim:report.mp.p7a.datasets -->60 and <!-- claim:report.mp.p7b.datasets -->60 datasets, without and with planted task-by-factor variation) checked E1's task jackknife: minimum coverage <!-- claim:report.mp.p7a.min -->0.950 and <!-- claim:report.mp.p7b.min -->0.917, pooled <!-- claim:report.mp.p7a.pooled -->0.977 and <!-- claim:report.mp.p7b.pooled -->0.971. The move to the jackknife rests on an argument, not on a measured failure of the bootstrap box.

Profile-likelihood intervals were not usable at an E1-sized fit (`D-022`), which is why E1 uses the bootstrap and the jackknife.

### E1: no results

No E1 results exist: no confirmatory run has started and neither E0 part 2 nor G6 has been banked, so G7 cannot pass yet. The local arm waits for those simulations, planned to take more than a day of CPU, and the Claude slice must run on a machine where the billing guard is clean, at most 6 runs a day.

## Limitations and threats to validity

- **Scope.** The E2 intervals cover sampling over tasks and runs for the submissions as run, not infrastructure, harness, or benchmark changes. As preregistered, they treat each board's tasks as a draw from a population like the benchmark's. Holding these exact tasks fixed would leave only run-to-run variation. That would narrow the intervals where entries have several runs per task, and with a single run per task, as on SWE-bench Verified, it cannot be estimated without a model. This analysis is not reported, so the supported and unsupported labels concern the task population, not these tasks alone.
- **Heterogeneous submissions.** Entries differ in harness, runs per task (<!-- claim:report.e2.tb2.rpt_min -->1 to <!-- claim:report.e2.tb2.rpt_max -->10 on TB2), and ambiguous share, and task-level and pooled scores can differ; both are in the confidence reports.
- **Unusable data.** <!-- claim:report.e2.tb2.unlisted -->315 TB2 trials cannot be attributed to a task, <!-- claim:report.e2.tb2.excluded -->9 TB2 submissions were excluded by rule, and <!-- claim:report.e2.ep.fetch_failures -->3 Epoch rows had no readable log.
- **Multiplicity.** Adjacent-pair readings are unadjusted; only the rank intervals control error jointly.
- **Simulation-based validity.** E0 assumes a logit model with normal effects, while real boards have unequal runs per task and missing tasks that need not be missing at random, and the Epoch-shaped cells have <!-- claim:report.e0.ep.tasks -->484 tasks against the <!-- claim:report.e2.ep.rank_tasks -->454 of the rank set. The rank and bound intervals over-covered at the simulated designs, which need not carry over to the boards. For the bounds this follows from where the simulated truth sits: the cells mark runs ambiguous at random, so the true score lies inside the bounds, and a true score at an end of the bounds, as when every ambiguous run would have failed, gives coverage nearer the nominal level.
- **Pessimistic bounds.** The bounds use no information about ambiguous runs and treat timeouts as ambiguous even where a board counts them as failures; they are a sensitivity analysis, not a verdict on any board's rule.
- **E1.** It has not run, and its scope is narrow: mock tasks, small local models on a single CPU-only machine, fixed CPU and memory, local time-of-day effects that need not match a hosted API's, expected Claude success near the ceiling, scaffold-confounded Claude-versus-local comparisons, 200 bootstrap replicates, and underpowered families.
- **Library gaps.** Bradley-Terry and IRT models, which the brief lists, are not implemented; E4 was not attempted; and the `evalstats report` subcommand is not implemented, so the confidence reports come from `python -m noisefloor.reanalysis report`.
- **Review.** The design and code were reviewed by adversarial reviewer agents, not yet by independent human statisticians.

## Ethics and disclosure

All agent runs work on mock tasks, with mock tools or, for Claude Code, its own file tools confined to a disposable workspace; no run touches real credentials, real systems, or harmful payloads, and nothing here helps an agent evade graders or monitors. Entries are named as the boards publish them; every adjacent ordering is reported as supported or not, and nothing is said about why any entry scored as it did. TB2 data are used under `Apache-2.0` with the canary string and a notice of changes, and Epoch AI data under `CC-BY-4.0` with credit ([docs/DATA.md](DATA.md)). Nothing was sent to the boards' maintainers or anyone else. The scouting pass before the E2 freeze is disclosed, and the method pilots are labeled exploratory. Cash spend is zero. The library, the studies, and this report were built by Claude Code (Anthropic) working autonomously from a written brief, and corrections to earlier claims are dated in [CORRECTIONS.md](../CORRECTIONS.md).

## Reproducibility

After `make setup` (uv with locked dependencies; R and lme4 are needed only for the GLMM path), the following run offline on a fresh clone.

- `uv run python -m noisefloor.reanalysis verify` checks both snapshot hashes, recomputes every E2 and E3 result, and requires byte-identical JSON; it printed `E2-VERIFY: PASS` on the artifacts this report cites. `make reproduce` runs that check, redraws the figures, checks the noise-floor file, and fails if a committed figure changes; `uv run python -m noisefloor.reanalysis report` rebuilds the confidence reports.
- E0's raw replicate lines are in `artifacts/e0/raw/`, and `uv run python -m noisefloor.e0_crosscheck` reruns the cross-checks; `uv run python -m noisefloor.method_pilots` recomputes the pilot summary from its raw rows.
- `make claimcheck`, `make freshness`, `make prereg-verify`, and `make citecheck` check this report's numbers, the plan chain, and the banked record of every reference.
- E1's commands are in [`prereg/0004`](../prereg/0004-e1-factorial.md) and its schedule in [docs/PLAN.md](PLAN.md); each study's seeds are fixed in its plan.

## References

Each reference was checked once (title, authors, year, URL) and is banked in `artifacts/citecheck.json` with its retrieval date, final URL, and page hash; blocked publisher pages were checked through Crossref.

- Bates, D., et al. [Fitting Linear Mixed-Effects Models Using lme4](https://doi.org/10.18637/jss.v067.i01). Journal of Statistical Software.
- Bean, A. M., et al. [Measuring what Matters: Construct Validity in Large Language Model Benchmarks](https://arxiv.org/abs/2511.04703). arXiv:2511.04703.
- Davison, A. C., and Hinkley, D. V. [Bootstrap Methods and their Application](https://doi.org/10.1017/CBO9780511802843). Cambridge University Press.
- Efron, B. [Better Bootstrap Confidence Intervals](https://doi.org/10.1080/01621459.1987.10478410). Journal of the American Statistical Association.
- Epoch AI. [AI benchmarks and capabilities](https://epoch.ai/benchmarks), the Epoch AI Benchmarking Hub.
- Harbor. [`harborframework/terminal-bench-2-leaderboard`](https://huggingface.co/datasets/harborframework/terminal-bench-2-leaderboard), Hugging Face dataset.
- Heineman, D., et al. [Signal and Noise: A Framework for Reducing Uncertainty in Language Model Evaluation](https://arxiv.org/abs/2508.13144). arXiv:2508.13144.
- Imbens, G. W., and Manski, C. F. [Confidence Intervals for Partially Identified Parameters](https://doi.org/10.1111/j.1468-0262.2004.00555.x). Econometrica.
- Jimenez, C. E., et al. [SWE-bench: Can Language Models Resolve Real-World GitHub Issues?](https://arxiv.org/abs/2310.06770). arXiv:2310.06770.
- Kwa, T. (METR). [Clarifying limitations of time horizon](https://metr.org/notes/2026-01-22-time-horizon-limitations/). METR research note.
- Merrill, M. A., et al. [Terminal-Bench: Benchmarking Agents on Hard, Realistic Tasks in Command Line Interfaces](https://arxiv.org/abs/2601.11868). arXiv:2601.11868.
- Miller, E. [Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations](https://arxiv.org/abs/2411.00640). arXiv:2411.00640.
- Mogstad, M., Romano, J. P., Shaikh, A. M., and Wilhelm, D. [Inference for Ranks with Applications to Mobility across Neighbourhoods and Academic Achievement across Countries](https://doi.org/10.1093/restud/rdad006). Review of Economic Studies.
- Nakagawa, S., and Schielzeth, H. [A general and simple method for obtaining R2 from generalized linear mixed-effects models](https://doi.org/10.1111/j.2041-210x.2012.00261.x). Methods in Ecology and Evolution.
- Polo, F. M., et al. [tinyBenchmarks: evaluating LLMs with fewer examples](https://arxiv.org/abs/2402.14992). arXiv:2402.14992.
- Segato, G. (Anthropic). [Quantifying infrastructure noise in agentic coding evals](https://www.anthropic.com/research/infrastructure-noise). Anthropic.
- UK AI Security Institute. [Inspect](https://inspect.aisi.org.uk/), an open-source framework for large language model evaluations.
