# 0003 E0 part 1: simulation coverage grid for the evalstats intervals (gate G5)

- Date: 2026-11-04
- Experiment: E0 (library validity), part 1. Part 2 (the logit GLMM variance-component cells,
  shaped like the E1 design) will be a dated amendment of this file, frozen before E0 part 2 and
  before any E1 confirmatory data.
- Status: frozen by `tools/prereg.py freeze` before any replicate of these cells is banked.

## What is tested

Each cell fixes an evalstats interval method, a data-generating process (DGP), the nominal
level, and the number of replicates. Each replicate draws synthetic data (seeded from the
cell id and the replicate index, `noisefloor.e0.Cell.seed_for`), computes the interval, and
records whether it covers the true value. All data here is synthetic.

DGP (`noisefloor.e0`): binary outcomes from
`logit p = intercept + model effect + task effect + model-by-task effect`, with task effects
`N(0, task_sd^2)` and model-by-task effects `N(0, model_task_sd^2)`, `n_tasks` tasks drawn from
the population and `runs` runs per (model, task). Truths are super-population quantities
(Gauss-Hermite quadrature): a model's true score is `E[sigmoid(...)]` over the task
population; a difference is the difference of true scores; true ranks follow the model effects
(all distinct). For `bounded_score`, each run is ambiguous with probability `ambiguous_rate`,
recorded as status `infra_error` with outcome 0, and the truth uses the unmasked outcomes.

Methods and their truths:

- `score_interval`: evalstats `score_interval` (cluster_t, bootstrap, or bca) for one model.
- `paired_difference`: evalstats `paired_difference` for two models on shared tasks.
- `cluster_robust_mean`: evalstats `cluster_robust_mean` (CR1) of all runs of one model.
- `rank_simultaneous`: evalstats `rank_confidence` (simultaneous); a replicate covers when
  every model's rank interval contains its true rank.
- `rank_marginal`: marginal rank interval of the model at true position `target` (0 based).
- `bounded_score`: the Imbens and Manski interval of evalstats `policy_bounds`.

Cells shaped like the real boards: `tb2` cells use 89 tasks and 5 runs (66 models for rank
cells), the shape of the Terminal-Bench 2.0 analysis set; `ep` cells use 484 tasks and 1 run
(32 models for rank cells), the shape of the Epoch AI SWE-bench Verified set; `small` cells
probe 20 to 40 tasks.

## Rule (gate G5, docs/BUILD_BRIEF.md section 6)

A cell passes if its coverage is at least `nominal - 0.015 - 2 * MCSE`, with MCSE
`sqrt(c(1 - c) / R)` for observed coverage `c` and `R` replicates; over-coverage is reported,
not failed. Replicates: 2000 per cell, 500 for bootstrap-based intervals (`bootstrap`, `bca`,
`bootstrap_max_t`). A replicate whose interval cannot be computed counts as not covered.

Headline cells (every interval behind an E2 headline number): `S-tb2-t`, `S-ep-t`,
`P-tb2-t`, `P-ep-t`, `R-tb2-sim`, `R-ep-sim`, `B-tb2`, `B-ep`. Every headline cell must
pass; a failing headline cell makes the corresponding E2 result reported as resting on an
unsupported method (prereg/0001, Methods 6). Every other failing cell is listed as unsupported
in `evalstats.validity` and triggers a library warning when the method is used in a matching
design.

## Grid

```json e0-grid
[
 {
  "cell": "S-tb2-t",
  "method": "score_interval",
  "interval_type": "cluster_t",
  "nominal": 0.95,
  "replicates": 2000,
  "params": {
   "n_tasks": 89,
   "runs": 5,
   "intercept": 0.5,
   "task_sd": 2.0,
   "model_task_sd": 1.0
  }
 },
 {
  "cell": "S-ep-t",
  "method": "score_interval",
  "interval_type": "cluster_t",
  "nominal": 0.95,
  "replicates": 2000,
  "params": {
   "n_tasks": 484,
   "runs": 1,
   "intercept": 1.0,
   "task_sd": 2.0,
   "model_task_sd": 1.0
  }
 },
 {
  "cell": "S-small-t",
  "method": "score_interval",
  "interval_type": "cluster_t",
  "nominal": 0.95,
  "replicates": 2000,
  "params": {
   "n_tasks": 20,
   "runs": 1,
   "intercept": 0.0,
   "task_sd": 1.0,
   "model_task_sd": 0.5
  }
 },
 {
  "cell": "S-tb2-boot",
  "method": "score_interval",
  "interval_type": "bootstrap",
  "nominal": 0.95,
  "replicates": 500,
  "params": {
   "n_tasks": 89,
   "runs": 5,
   "intercept": 0.5,
   "task_sd": 2.0,
   "model_task_sd": 1.0,
   "n_boot": 2000
  }
 },
 {
  "cell": "S-tb2-bca",
  "method": "score_interval",
  "interval_type": "bca",
  "nominal": 0.95,
  "replicates": 500,
  "params": {
   "n_tasks": 89,
   "runs": 5,
   "intercept": 0.5,
   "task_sd": 2.0,
   "model_task_sd": 1.0,
   "n_boot": 2000
  }
 },
 {
  "cell": "S-small-boot",
  "method": "score_interval",
  "interval_type": "bootstrap",
  "nominal": 0.95,
  "replicates": 500,
  "params": {
   "n_tasks": 20,
   "runs": 1,
   "intercept": 0.0,
   "task_sd": 1.0,
   "model_task_sd": 0.5,
   "n_boot": 2000
  }
 },
 {
  "cell": "S-small-bca",
  "method": "score_interval",
  "interval_type": "bca",
  "nominal": 0.95,
  "replicates": 500,
  "params": {
   "n_tasks": 20,
   "runs": 1,
   "intercept": 0.0,
   "task_sd": 1.0,
   "model_task_sd": 0.5,
   "n_boot": 2000
  }
 },
 {
  "cell": "P-tb2-t",
  "method": "paired_difference",
  "interval_type": "cluster_t",
  "nominal": 0.95,
  "replicates": 2000,
  "params": {
   "n_tasks": 89,
   "runs": 5,
   "intercept": 0.5,
   "task_sd": 2.0,
   "model_task_sd": 1.0,
   "effect_a": 0.2,
   "effect_b": 0.0
  }
 },
 {
  "cell": "P-tb2-null-t",
  "method": "paired_difference",
  "interval_type": "cluster_t",
  "nominal": 0.95,
  "replicates": 2000,
  "params": {
   "n_tasks": 89,
   "runs": 5,
   "intercept": 0.5,
   "task_sd": 2.0,
   "model_task_sd": 1.0,
   "effect_a": 0.0,
   "effect_b": 0.0
  }
 },
 {
  "cell": "P-ep-t",
  "method": "paired_difference",
  "interval_type": "cluster_t",
  "nominal": 0.95,
  "replicates": 2000,
  "params": {
   "n_tasks": 484,
   "runs": 1,
   "intercept": 1.0,
   "task_sd": 2.0,
   "model_task_sd": 1.0,
   "effect_a": 0.2,
   "effect_b": 0.0
  }
 },
 {
  "cell": "P-small-t",
  "method": "paired_difference",
  "interval_type": "cluster_t",
  "nominal": 0.95,
  "replicates": 2000,
  "params": {
   "n_tasks": 20,
   "runs": 1,
   "intercept": 0.0,
   "task_sd": 1.0,
   "model_task_sd": 0.5,
   "effect_a": 0.3,
   "effect_b": 0.0
  }
 },
 {
  "cell": "P-tb2-boot",
  "method": "paired_difference",
  "interval_type": "bootstrap",
  "nominal": 0.95,
  "replicates": 500,
  "params": {
   "n_tasks": 89,
   "runs": 5,
   "intercept": 0.5,
   "task_sd": 2.0,
   "model_task_sd": 1.0,
   "effect_a": 0.2,
   "effect_b": 0.0,
   "n_boot": 2000
  }
 },
 {
  "cell": "P-ep-boot",
  "method": "paired_difference",
  "interval_type": "bootstrap",
  "nominal": 0.95,
  "replicates": 500,
  "params": {
   "n_tasks": 484,
   "runs": 1,
   "intercept": 1.0,
   "task_sd": 2.0,
   "model_task_sd": 1.0,
   "effect_a": 0.2,
   "effect_b": 0.0,
   "n_boot": 2000
  }
 },
 {
  "cell": "P-small-boot",
  "method": "paired_difference",
  "interval_type": "bootstrap",
  "nominal": 0.95,
  "replicates": 500,
  "params": {
   "n_tasks": 20,
   "runs": 1,
   "intercept": 0.0,
   "task_sd": 1.0,
   "model_task_sd": 0.5,
   "effect_a": 0.3,
   "effect_b": 0.0,
   "n_boot": 2000
  }
 },
 {
  "cell": "C-tb2",
  "method": "cluster_robust_mean",
  "interval_type": "cluster_robust",
  "nominal": 0.95,
  "replicates": 2000,
  "params": {
   "n_tasks": 89,
   "runs": 5,
   "intercept": 0.5,
   "task_sd": 2.0
  }
 },
 {
  "cell": "C-small",
  "method": "cluster_robust_mean",
  "interval_type": "cluster_robust",
  "nominal": 0.95,
  "replicates": 2000,
  "params": {
   "n_tasks": 20,
   "runs": 3,
   "intercept": 0.0,
   "task_sd": 1.0
  }
 },
 {
  "cell": "R-tb2-sim",
  "method": "rank_simultaneous",
  "interval_type": "bootstrap_max_t",
  "nominal": 0.95,
  "replicates": 500,
  "params": {
   "n_tasks": 89,
   "runs": 5,
   "intercept": 0.5,
   "task_sd": 2.0,
   "model_task_sd": 1.0,
   "effects": [
    -1.5,
    -1.4538,
    -1.4077,
    -1.3615,
    -1.3154,
    -1.2692,
    -1.2231,
    -1.1769,
    -1.1308,
    -1.0846,
    -1.0385,
    -0.9923,
    -0.9462,
    -0.9,
    -0.8538,
    -0.8077,
    -0.7615,
    -0.7154,
    -0.6692,
    -0.6231,
    -0.5769,
    -0.5308,
    -0.4846,
    -0.4385,
    -0.3923,
    -0.3462,
    -0.3,
    -0.2538,
    -0.2077,
    -0.1615,
    -0.1154,
    -0.0692,
    -0.0231,
    0.0231,
    0.0692,
    0.1154,
    0.1615,
    0.2077,
    0.2538,
    0.3,
    0.3462,
    0.3923,
    0.4385,
    0.4846,
    0.5308,
    0.5769,
    0.6231,
    0.6692,
    0.7154,
    0.7615,
    0.8077,
    0.8538,
    0.9,
    0.9462,
    0.9923,
    1.0385,
    1.0846,
    1.1308,
    1.1769,
    1.2231,
    1.2692,
    1.3154,
    1.3615,
    1.4077,
    1.4538,
    1.5
   ],
   "n_boot": 2000
  }
 },
 {
  "cell": "R-ep-sim",
  "method": "rank_simultaneous",
  "interval_type": "bootstrap_max_t",
  "nominal": 0.95,
  "replicates": 500,
  "params": {
   "n_tasks": 484,
   "runs": 1,
   "intercept": 1.0,
   "task_sd": 2.0,
   "model_task_sd": 1.0,
   "effects": [
    -1.2,
    -1.1226,
    -1.0452,
    -0.9677,
    -0.8903,
    -0.8129,
    -0.7355,
    -0.6581,
    -0.5806,
    -0.5032,
    -0.4258,
    -0.3484,
    -0.271,
    -0.1935,
    -0.1161,
    -0.0387,
    0.0387,
    0.1161,
    0.1935,
    0.271,
    0.3484,
    0.4258,
    0.5032,
    0.5806,
    0.6581,
    0.7355,
    0.8129,
    0.8903,
    0.9677,
    1.0452,
    1.1226,
    1.2
   ],
   "n_boot": 2000
  }
 },
 {
  "cell": "R-small-sim",
  "method": "rank_simultaneous",
  "interval_type": "bootstrap_max_t",
  "nominal": 0.95,
  "replicates": 500,
  "params": {
   "n_tasks": 40,
   "runs": 1,
   "intercept": 0.0,
   "task_sd": 1.0,
   "model_task_sd": 0.5,
   "effects": [
    0.6,
    0.3,
    0.0,
    -0.3,
    -0.6
   ],
   "n_boot": 2000
  }
 },
 {
  "cell": "R-tb2-marg",
  "method": "rank_marginal",
  "interval_type": "bootstrap_max_t",
  "nominal": 0.95,
  "replicates": 500,
  "params": {
   "n_tasks": 89,
   "runs": 5,
   "intercept": 0.5,
   "task_sd": 2.0,
   "model_task_sd": 1.0,
   "effects": [
    -1.5,
    -1.4538,
    -1.4077,
    -1.3615,
    -1.3154,
    -1.2692,
    -1.2231,
    -1.1769,
    -1.1308,
    -1.0846,
    -1.0385,
    -0.9923,
    -0.9462,
    -0.9,
    -0.8538,
    -0.8077,
    -0.7615,
    -0.7154,
    -0.6692,
    -0.6231,
    -0.5769,
    -0.5308,
    -0.4846,
    -0.4385,
    -0.3923,
    -0.3462,
    -0.3,
    -0.2538,
    -0.2077,
    -0.1615,
    -0.1154,
    -0.0692,
    -0.0231,
    0.0231,
    0.0692,
    0.1154,
    0.1615,
    0.2077,
    0.2538,
    0.3,
    0.3462,
    0.3923,
    0.4385,
    0.4846,
    0.5308,
    0.5769,
    0.6231,
    0.6692,
    0.7154,
    0.7615,
    0.8077,
    0.8538,
    0.9,
    0.9462,
    0.9923,
    1.0385,
    1.0846,
    1.1308,
    1.1769,
    1.2231,
    1.2692,
    1.3154,
    1.3615,
    1.4077,
    1.4538,
    1.5
   ],
   "n_boot": 2000,
   "target": 33
  }
 },
 {
  "cell": "B-tb2",
  "method": "bounded_score",
  "interval_type": "imbens_manski",
  "nominal": 0.95,
  "replicates": 2000,
  "params": {
   "n_tasks": 89,
   "runs": 5,
   "intercept": 0.5,
   "task_sd": 2.0,
   "ambiguous_rate": 0.12
  }
 },
 {
  "cell": "B-ep",
  "method": "bounded_score",
  "interval_type": "imbens_manski",
  "nominal": 0.95,
  "replicates": 2000,
  "params": {
   "n_tasks": 484,
   "runs": 1,
   "intercept": 1.0,
   "task_sd": 2.0,
   "ambiguous_rate": 0.02
  }
 }
]
```
