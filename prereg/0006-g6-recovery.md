# 0006 G6: recovery of planted variance components by the E1 analysis

- Date: 2026-11-11
- Gate: G6 (docs/BUILD_BRIEF.md section 6): on at least 200 synthetic datasets with planted
  variance components, each component's interval covers its planted value at a rate meeting the
  G5 rule, and each component's bias is reported.
- Frozen in the same commit as `prereg/0004-e1-factorial.md` and `prereg/0005-e0-glmm-cells.md`,
  before any of these datasets is simulated. All data here is synthetic.

## What is tested

The E1 analysis of prereg 0004 (the ```json e1-analysis``` settings; `noisefloor.recovery`
calls the same `evalstats.decompose` with the same settings, which the ```json recovery-config```
block repeats and `noisefloor.prereg_consistency` checks against 0004): a logit GLMM with task and
model:task random intercepts, fixed factors model, harness, timeout, steps, temperature,
concurrency, day, and block, Laplace fits, a parametric bootstrap with B = 200 for the random
variances and the MDE rows, task-jackknife boxes for the fixed contributions, shares mapped from
those intervals (a fixed share maps its jackknife box), and the local-arm score and contrast
intervals. The test `tests/test_recovery.py::test_g6_analysis_equals_the_e1_analysis`, which
the test suite (gate G1) runs, feeds one simulated dataset through `noisefloor.analysis.e1`
(`runs_table`, `analysis_frame`, `decomposition`, `mde_tables`, `scores_table`,
`contrasts_table`) as banked run records and through `noisefloor.recovery`, and requires the
same estimates and intervals; the G6 rows and the E1 analysis also run one pinned analysis code
(prereg 0004). The analysis is applied to 200 datasets simulated in the shape of the E1 local
arm (`noisefloor.recovery.simulate`: 3 models x 85 tasks x the 16 runs of the half fraction,
with each cell's 2 replicate runs dealt into the batches of one model and one concurrency level
over 4 days x 2 time-of-day blocks by the function that `noisefloor.e1.local_plan` uses, and
then left out; every batch on its planned slot), with the planted values of the ```json
recovery-config``` block (byte-identical to the one in prereg 0005, which describes the planted
values and where they come from). Seeds: the ```json g6-seeds``` list (970000 to 970199;
disjoint from the exploratory pilot seeds 990000 to 990999, from the E0 part 2 seeds, which are
derived from cell and group names, and from the post-E1 check seeds 980000 to 980199).

## Components

G6 checks exactly the components in the ```json g6-components``` block, which are the ones every
dataset row reports:

- the task and model:task variances (planted: their variance parameters);
- the contribution of each fixed factor (planted: the variance over the dataset's runs of the
  factor's planted contribution to the linear predictor);
- the share of the total of each of these and of the logistic residual (planted: the planted
  value over the sum of the planted values of the analyzed components plus pi^2/3);
- the MDE of each design in the E1 MDE table (planted: the MDE the planted model implies at the
  E1 reference configuration, `noisefloor.recovery.planted_mde`);
- the local-arm intervals of prereg 0004 items 3 to 5 (`noisefloor.recovery.local_intervals`):
  each model's success rate over all its runs (`score:<m>`), its harness contrast, level 1 minus
  level 0 paired by task (`harness:<m>`), the difference of two models' harness contrasts paired
  by task (`hdiff:<a>-<b>`), and each pair of models on the harness level 0 runs (`pair:<a>-<b>`),
  all with `cluster_t` intervals; planted values are the means, over the same runs, of each run's
  success probability with its task and model:task effects integrated out
  (`noisefloor.recovery.planted_success`). Models are identified by their position in the planted
  model effects.

## Rule and outputs

- Each component's coverage (planted value inside its interval) must meet the G5 rule: coverage
  at least `0.95 - 0.015 - 2 * sqrt(c (1 - c) / 200)`. A dataset whose interval for a component
  is missing (a failed fit; more than 10 percent of a random variance's or an MDE row's bootstrap
  replicates failed, or of the totals behind the residual share; or more than 10 percent of a
  fixed factor's jackknife refits failed) counts as not covered for that component.
- Bias: the mean of estimate minus planted value over the datasets with an estimate, per
  component, reported with that count beside coverage whether the gate passes or not.
- Above zero: for each component, the share of all 200 datasets whose interval's lower end is
  strictly above 0, a missing interval counting as not above 0. For the fixed contributions this
  is the power of prereg 0004's fixed-factor reading (H0 rejected when the factor's task-jackknife
  interval has its lower end above 0), which 0004 reads as underpowered below 0.8; for
  concurrency, planted at 0, it is that reading's false-positive rate.
- Failed fits: the number of datasets whose fit failed, recorded as datasets without estimates or
  intervals; no dataset is rerun or dropped.
- Run with `python -m noisefloor.recovery g6-run prereg/0006-g6-recovery.md --workers N`
  (resumable; raw rows in `artifacts/g6/raw.jsonl`), and bank with `python -m noisefloor.recovery
  g6-bank prereg/0006-g6-recovery.md` into `artifacts/g6/recovery.json`, whose `config` must
  equal the ```json recovery-config``` block and whose components must equal the list
  in the ```json g6-components``` block (gate G6).
- Every row carries a provenance stamp (environment hash, analysis-code hash, commit, a dirty
  flag for `src/`, time; `noisefloor.provenance`, D-031 item 6, D-033 item 10). No commit changes
  `src/` or `uv.lock` while G6 runs (E0 part 2's list of failing cells is committed only after G6
  has written all its rows, prereg 0005): `g6-run` refuses to add rows beside rows from another
  environment, and banking refuses rows that do not share one environment and one analysis code
  from a clean `src/`. If rows from two environments exist anyway, the rows are set aside unread
  and G6 runs again in one environment, reported as a deviation. The artifact names the
  environment and the analysis code, which gate G6 requires to be the one E0 part 2 ran (for
  a rerun of G6 under an amendment of this file, which a replaced GLMM method needs, the one a
  frozen amendment of prereg 0004 allows); prereg 0004's analysis runs only under that analysis
  code.

## Blocks

```json recovery-config
{
 "design": {
  "n_models": 3,
  "n_tasks": 85,
  "runs_per_cell": 16,
  "batch_size": 10,
  "days": 4,
  "blocks_per_day": 2,
  "replicates": [
   [
    0,
    1,
    1,
    0,
    0
   ],
   [
    0,
    1,
    1,
    1,
    1
   ]
  ]
 },
 "planted": {
  "intercept": 0.0,
  "model": [
   0.6,
   0.0,
   -0.6
  ],
  "factors": {
   "harness": 0.5,
   "timeout": 0.4,
   "steps": 0.4,
   "temperature": 0.1,
   "concurrency": 0.0
  },
  "day": [
   0.0,
   0.1,
   -0.1,
   0.05
  ],
  "block": [
   0.0,
   0.05
  ],
  "task_var": 4.0,
  "model_task_var": 0.5,
  "batch_var": 0.0,
  "interactions": {},
  "model_factors": {}
 },
 "settings": {
  "n_boot": 200,
  "nagq": 1,
  "boot_interval": "basic_log",
  "box_interval": "basic",
  "share_interval": "plugin",
  "fixed_interval": "jackknife",
  "level": 0.95,
  "random": [
   "task",
   "model:task"
  ],
  "mde_designs": [
   [
    20,
    1
   ],
   [
    20,
    3
   ],
   [
    20,
    5
   ],
   [
    40,
    1
   ],
   [
    40,
    3
   ],
   [
    40,
    5
   ],
   [
    89,
    1
   ],
   [
    89,
    3
   ],
   [
    89,
    5
   ],
   [
    200,
    1
   ],
   [
    200,
    3
   ],
   [
    200,
    5
   ],
   [
    500,
    1
   ],
   [
    500,
    3
   ],
   [
    500,
    5
   ]
  ],
  "mde_interval": "bc",
  "score_interval": "cluster_t",
  "contrast_interval": "cluster_t"
 }
}
```

```json g6-seeds
[
 970000,
 970001,
 970002,
 970003,
 970004,
 970005,
 970006,
 970007,
 970008,
 970009,
 970010,
 970011,
 970012,
 970013,
 970014,
 970015,
 970016,
 970017,
 970018,
 970019,
 970020,
 970021,
 970022,
 970023,
 970024,
 970025,
 970026,
 970027,
 970028,
 970029,
 970030,
 970031,
 970032,
 970033,
 970034,
 970035,
 970036,
 970037,
 970038,
 970039,
 970040,
 970041,
 970042,
 970043,
 970044,
 970045,
 970046,
 970047,
 970048,
 970049,
 970050,
 970051,
 970052,
 970053,
 970054,
 970055,
 970056,
 970057,
 970058,
 970059,
 970060,
 970061,
 970062,
 970063,
 970064,
 970065,
 970066,
 970067,
 970068,
 970069,
 970070,
 970071,
 970072,
 970073,
 970074,
 970075,
 970076,
 970077,
 970078,
 970079,
 970080,
 970081,
 970082,
 970083,
 970084,
 970085,
 970086,
 970087,
 970088,
 970089,
 970090,
 970091,
 970092,
 970093,
 970094,
 970095,
 970096,
 970097,
 970098,
 970099,
 970100,
 970101,
 970102,
 970103,
 970104,
 970105,
 970106,
 970107,
 970108,
 970109,
 970110,
 970111,
 970112,
 970113,
 970114,
 970115,
 970116,
 970117,
 970118,
 970119,
 970120,
 970121,
 970122,
 970123,
 970124,
 970125,
 970126,
 970127,
 970128,
 970129,
 970130,
 970131,
 970132,
 970133,
 970134,
 970135,
 970136,
 970137,
 970138,
 970139,
 970140,
 970141,
 970142,
 970143,
 970144,
 970145,
 970146,
 970147,
 970148,
 970149,
 970150,
 970151,
 970152,
 970153,
 970154,
 970155,
 970156,
 970157,
 970158,
 970159,
 970160,
 970161,
 970162,
 970163,
 970164,
 970165,
 970166,
 970167,
 970168,
 970169,
 970170,
 970171,
 970172,
 970173,
 970174,
 970175,
 970176,
 970177,
 970178,
 970179,
 970180,
 970181,
 970182,
 970183,
 970184,
 970185,
 970186,
 970187,
 970188,
 970189,
 970190,
 970191,
 970192,
 970193,
 970194,
 970195,
 970196,
 970197,
 970198,
 970199
]
```

```json g6-components
[
 "block",
 "concurrency",
 "day",
 "harness",
 "harness:m0",
 "harness:m1",
 "harness:m2",
 "hdiff:m0-m1",
 "hdiff:m0-m2",
 "hdiff:m1-m2",
 "mde:T200:k1",
 "mde:T200:k3",
 "mde:T200:k5",
 "mde:T20:k1",
 "mde:T20:k3",
 "mde:T20:k5",
 "mde:T40:k1",
 "mde:T40:k3",
 "mde:T40:k5",
 "mde:T500:k1",
 "mde:T500:k3",
 "mde:T500:k5",
 "mde:T89:k1",
 "mde:T89:k3",
 "mde:T89:k5",
 "model",
 "model:task",
 "pair:m0-m1",
 "pair:m0-m2",
 "pair:m1-m2",
 "score:m0",
 "score:m1",
 "score:m2",
 "share:block",
 "share:concurrency",
 "share:day",
 "share:harness",
 "share:model",
 "share:model:task",
 "share:residual",
 "share:steps",
 "share:task",
 "share:temperature",
 "share:timeout",
 "steps",
 "task",
 "temperature",
 "timeout"
]
```
