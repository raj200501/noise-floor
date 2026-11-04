"""E0 cross-checks: evalstats against reference implementations on shared datasets.

``python -m noisefloor.e0_crosscheck`` writes ``artifacts/e0/crosscheck.json``: one
entry per check with evalstats' value, the reference value, the stated tolerance, and
the reference implementation. Datasets: the committed demo fixture
(``fixtures/demo/synthetic_results.csv``, synthetic) and synthetic planted data drawn
here with fixed seeds. Gate G5 requires every check to agree within its tolerance.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import statsmodels
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.weightstats import DescrStatsW

from evalstats import (
    EvalResults,
    PairedDesign,
    bca_interval,
    cluster_robust_mean,
    paired_difference,
    percentile_interval,
    power_at,
    read_csv,
    t_interval,
)
from evalstats.glmm import decompose, lme4_available
from noisefloor.synthetic import PlantedSpec, simulate

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "fixtures" / "demo" / "synthetic_results.csv"
OUT = REPO / "artifacts" / "e0" / "crosscheck.json"


def check(
    name: str, value: float, reference: float, tolerance: float, impl: str, *, note: str = ""
) -> dict[str, Any]:
    """One cross-check record."""
    return {
        "name": name,
        "value": float(value),
        "reference": float(reference),
        "tolerance": tolerance,
        "reference_impl": impl,
        "abs_diff": abs(float(value) - float(reference)),
        "note": note,
    }


def library_checks() -> list[dict[str, Any]]:
    """Checks of the analytic and bootstrap intervals on the demo fixture."""
    res = read_csv(FIXTURE)
    a, b = res.models[0], res.models[1]
    out = []
    rows = res.frame.loc[res.frame["model"] == a]
    y = rows["outcome"].to_numpy(dtype=float)
    _, se = cluster_robust_mean(y, rows["task"].to_numpy())
    fit = sm.OLS(y, np.ones_like(y)).fit(cov_type="cluster", cov_kwds={"groups": rows["task"]})
    out.append(
        check(
            "CR1 standard error of a mean (demo fixture)",
            se,
            float(fit.bse[0]),
            1e-10,
            f"statsmodels {statsmodels.__version__} OLS cluster",
        )
    )
    scores = res.task_scores(a).to_numpy()
    lo, hi = DescrStatsW(scores).tconfint_mean(alpha=0.05)
    iv = t_interval(scores, 0.95)
    out.append(
        check(
            "t interval lower bound for a score",
            iv.lo,
            lo,
            1e-10,
            f"statsmodels {statsmodels.__version__} DescrStatsW.tconfint_mean",
        )
    )
    out.append(
        check(
            "t interval upper bound for a score",
            iv.hi,
            hi,
            1e-10,
            f"statsmodels {statsmodels.__version__} DescrStatsW.tconfint_mean",
        )
    )
    cmp = paired_difference(res, a, b)
    diffs = res.paired(a, b).differences
    _, p_sm, _ = DescrStatsW(diffs).ttest_mean(0.0)
    out.append(
        check(
            "paired t-test p-value",
            cmp.p_value,
            float(p_sm),
            1e-10,
            f"statsmodels {statsmodels.__version__} DescrStatsW.ttest_mean",
        )
    )
    ref = stats.ttest_rel(res.task_scores(a), res.task_scores(b))
    out.append(
        check(
            "paired t statistic",
            cmp.t_stat,
            float(ref.statistic),
            1e-10,
            f"scipy {scipy.__version__} stats.ttest_rel",
        )
    )
    design = PairedDesign(between_task_var=0.02, within_task_var=0.3)
    for n_tasks, effect in ((100, 0.05), (400, 0.03)):
        var = design.between_task_var + design.within_task_var
        d = effect / math.sqrt(var)
        ref_power = NormalIndPower().power(effect_size=d, nobs1=n_tasks, alpha=0.05, ratio=0)
        out.append(
            check(
                f"two-sided power, {n_tasks} tasks",
                power_at(effect, design, n_tasks),
                float(ref_power),
                1e-8,
                f"statsmodels {statsmodels.__version__} NormalIndPower (one sample)",
            )
        )
    for method, fn in (("percentile", percentile_interval), ("BCa", bca_interval)):
        mine = fn(scores, 0.95, n_boot=20000, seed=11)
        theirs = stats.bootstrap(
            (scores,),
            np.mean,
            confidence_level=0.95,
            n_resamples=20000,
            method="percentile" if method == "percentile" else "BCa",
            random_state=np.random.default_rng(12),
        )
        ci = theirs.confidence_interval
        note = "independent resampling streams; tolerance covers Monte Carlo error"
        out.append(
            check(
                f"{method} bootstrap lower bound",
                mine.lo,
                ci.low,
                0.01,
                f"scipy {scipy.__version__} stats.bootstrap",
                note=note,
            )
        )
        out.append(
            check(
                f"{method} bootstrap upper bound",
                mine.hi,
                ci.high,
                0.01,
                f"scipy {scipy.__version__} stats.bootstrap",
                note=note,
            )
        )
    return out


R_SCRIPT = r"""
args <- commandArgs(trailingOnly = TRUE)
df <- read.csv(args[1], stringsAsFactors = TRUE)
df$harness <- factor(df$harness)
fit <- lme4::glmer(outcome ~ 1 + harness + (1 | task) + (1 | model), data = df,
                   family = binomial, control = lme4::glmerControl(optimizer = "bobyqa"))
vc <- as.data.frame(lme4::VarCorr(fit))
cat(sprintf("%.10f %.10f %.10f\n", lme4::fixef(fit)[["harnessh1"]],
            vc$vcov[vc$grp == "task"], vc$vcov[vc$grp == "model"]))
"""


def glmm_checks() -> list[dict[str, Any]]:
    """lme4 through evalstats against lme4 called directly in R, and against statsmodels."""
    spec = PlantedSpec(
        models={"m1": 0.5, "m2": 0.0, "m3": -0.5, "m4": 0.2},
        n_tasks=120,
        runs=2,
        task_sd=1.2,
        model_task_sd=0.0,
        factors={"harness": {"h0": 0.0, "h1": 0.5}},
    )
    frame = simulate(spec, seed=20261104)
    res = EvalResults.from_frame(frame)
    out = []
    sm_fit = decompose(
        res, random=("task", "model"), fixed=("harness",), intervals=False, backend="statsmodels"
    )
    if lme4_available() and shutil.which("Rscript"):
        ours = decompose(
            res, random=("task", "model"), fixed=("harness",), intervals=False, backend="lme4"
        )
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "d.csv"
            frame.to_csv(data, index=False)
            script = Path(tmp) / "fit.R"
            script.write_text(R_SCRIPT)
            proc = subprocess.run(
                ["Rscript", str(script), str(data)], capture_output=True, text=True, check=True
            )
        beta, task_var, model_var = (float(x) for x in proc.stdout.split())
        impl = "R lme4 glmer called directly with Rscript"
        out.append(
            check(
                "GLMM harness coefficient, rpy2 path versus direct R",
                ours.fixed_effects["harnessh1"],
                beta,
                1e-6,
                impl,
            )
        )
        out.append(
            check(
                "GLMM task variance, rpy2 path versus direct R",
                ours.component("task").variance,
                task_var,
                1e-6,
                impl,
            )
        )
        out.append(
            check(
                "GLMM model variance, rpy2 path versus direct R",
                ours.component("model").variance,
                model_var,
                1e-6,
                impl,
            )
        )
        out.append(
            check(
                "GLMM harness coefficient, lme4 (Laplace) versus statsmodels (variational Bayes)",
                ours.fixed_effects["harnessh1"],
                sm_fit.fixed_effects["harnessh1"],
                0.05,
                f"statsmodels {statsmodels.__version__} BinomialBayesMixedGLM",
                note="different estimators; tolerance on the logit scale",
            )
        )
        out.append(
            check(
                "GLMM task variance, lme4 versus statsmodels",
                ours.component("task").variance,
                sm_fit.component("task").variance,
                0.25,
                f"statsmodels {statsmodels.__version__} BinomialBayesMixedGLM",
                note="different estimators; tolerance on the variance scale",
            )
        )
    return out


def main() -> int:
    """Run every check and write the artifact; exit 1 if any check is outside tolerance."""
    checks = library_checks() + glmm_checks()
    artifact = {
        "generator": "python -m noisefloor.e0_crosscheck",
        "python": sys.version.split()[0],
        "checks": checks,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(artifact, indent=1, sort_keys=True) + "\n")
    bad = [c for c in checks if c["abs_diff"] > c["tolerance"]]
    for c in checks:
        flag = "ok" if c["abs_diff"] <= c["tolerance"] else "OUTSIDE TOLERANCE"
        print(f"{flag:18s} {c['name']}: diff {c['abs_diff']:.3g} (tol {c['tolerance']:g})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
