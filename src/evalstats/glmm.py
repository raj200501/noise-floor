"""Logit GLMMs with crossed random effects, and variance partitioning.

``decompose`` fits ``outcome ~ fixed factors + (1 | g1) + (1 | g2) + ...`` and splits
the latent (logit-scale) variance into random-effect variances, the variance of
each fixed factor's contribution to the linear predictor, and the logistic
residual variance ``pi^2 / 3``.

Backends (named in every result):

* ``lme4``: ``glmer`` (Laplace, bobyqa) through rpy2, with profile-likelihood
  intervals. Used when R, lme4, and rpy2 are available.
* ``statsmodels-vb``: ``BinomialBayesMixedGLM`` fitted by variational Bayes, with
  approximate posterior intervals (labeled ``vb_posterior``). The fallback.

No variance-component interval is a Wald interval. A random term ``a:b`` means
one random intercept per observed combination of ``a`` and ``b``.
"""

from __future__ import annotations

import math
import re
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from evalstats.schema import OUTCOME, EvalResults

LOGISTIC_RESIDUAL_VAR = math.pi**2 / 3
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class VarianceComponent:
    """One latent-scale variance component with its interval (None if not available)."""

    name: str
    kind: str
    variance: float
    share: float
    lo: float | None
    hi: float | None
    interval_method: str


@dataclass(frozen=True)
class Decomposition:
    """Variance decomposition of eval outcomes on the logit scale."""

    backend: str
    formula: str
    n_obs: int
    level: float
    components: tuple[VarianceComponent, ...]
    fixed_effects: dict[str, float]
    converged: bool
    messages: tuple[str, ...] = field(default=())

    def component(self, name: str) -> VarianceComponent:
        """The component with this name."""
        for comp in self.components:
            if comp.name == name:
                return comp
        raise KeyError(name)

    @property
    def total_variance(self) -> float:
        """Sum of all latent-scale components, including the residual."""
        return sum(c.variance for c in self.components)


@dataclass
class _RawFit:
    backend: str
    random_var: dict[str, float]
    random_ci: dict[str, tuple[float, float] | None]
    fixed: dict[str, float]
    fixed_ci: dict[str, tuple[float, float] | None]
    interval_method: str
    converged: bool
    messages: list[str]


def _check_names(names: Sequence[str], frame: pd.DataFrame) -> None:
    for term in names:
        for part in term.split(":"):
            if not _NAME.match(part):
                raise ValueError(f"invalid column name in term {term!r}")
            if part not in frame.columns:
                raise ValueError(f"unknown column {part!r} in term {term!r}")


def _prepare(frame: pd.DataFrame, random: Sequence[str], fixed: Sequence[str]) -> pd.DataFrame:
    cols = sorted({p for t in [*random, *fixed] for p in t.split(":")})
    data = frame[[OUTCOME, *cols]].copy()
    for col in cols:
        data[col] = data[col].astype(str)
    for term in random:
        if ":" in term:
            data[term.replace(":", "__")] = data[term.split(":")].agg("|".join, axis=1)
    return data


def formula_for(random: Sequence[str], fixed: Sequence[str]) -> str:
    """The lme4 formula for these terms."""
    rhs = " + ".join([*(f"{f}" for f in fixed), *(f"(1 | {r})" for r in random)])
    return f"{OUTCOME} ~ 1" + (f" + {rhs}" if rhs else "")


def lme4_available() -> bool:
    """True if rpy2 imports and R has lme4."""
    try:
        from rpy2.robjects.packages import isinstalled
    except Exception:  # rpy2 missing or R not found
        return False
    return bool(isinstalled("lme4"))


_R_FIT = """
function(df, formula, factors, level, do_profile) {
  for (col in factors) df[[col]] <- factor(df[[col]])
  msgs <- character(0)
  keep <- function(w) {
    msgs <<- c(msgs, conditionMessage(w))
    invokeRestart("muffleWarning")
  }
  ctrl <- lme4::glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))
  fit <- withCallingHandlers(
    lme4::glmer(stats::as.formula(formula), data = df, family = stats::binomial, control = ctrl),
    warning = keep
  )
  profile_ci <- function(f) {
    # Profile the random-effect SDs and the factor coefficients, not the intercept.
    n_theta <- length(lme4::getME(f, "theta"))
    fe_names <- names(lme4::fixef(f))
    parm <- c(seq_len(n_theta), n_theta + which(fe_names != "(Intercept)"))
    tryCatch(
      withCallingHandlers(
        # devtol: deviance gains below 1e-4 found while profiling are numerical noise.
        stats::confint(
          f, parm = parm, method = "profile", level = level, oldNames = FALSE, devtol = 1e-4
        ),
        warning = keep
      ),
      error = function(e) conditionMessage(e)
    )
  }
  ci <- NULL
  if (do_profile) {
    ci <- profile_ci(fit)
    if (is.character(ci) && grepl("lower deviance", ci)) {
      # The profile found a better optimum: polish the fit with tight tolerances and retry.
      msgs <- c(msgs, paste("refit before profiling:", ci))
      tight <- lme4::glmerControl(
        optimizer = "nloptwrap",
        optCtrl = list(xtol_abs = 1e-10, ftol_abs = 1e-12, maxeval = 1e5)
      )
      start <- list(theta = lme4::getME(fit, "theta"), fixef = lme4::fixef(fit))
      fit <- withCallingHandlers(
        stats::update(fit, start = start, control = tight),
        warning = keep
      )
      ci <- profile_ci(fit)
    }
    if (is.character(ci)) {
      msgs <- c(msgs, paste("profile failed:", ci))
      ci <- NULL
    }
  }
  vc <- as.data.frame(lme4::VarCorr(fit))
  fe <- lme4::fixef(fit)
  conv <- fit@optinfo$conv$lme4$messages
  list(
    grp = as.character(vc$grp), vcov = as.numeric(vc$vcov),
    fe_names = names(fe), fe = as.numeric(fe),
    ci_names = if (is.null(ci)) character(0) else rownames(ci),
    ci_lo = if (is.null(ci)) numeric(0) else as.numeric(ci[, 1]),
    ci_hi = if (is.null(ci)) numeric(0) else as.numeric(ci[, 2]),
    msgs = unique(c(as.character(msgs), as.character(conv))),
    version = as.character(utils::packageVersion("lme4"))
  )
}
"""


def _fit_lme4(
    data: pd.DataFrame, random: Sequence[str], fixed: Sequence[str], level: float, profile: bool
) -> _RawFit:
    import rpy2.robjects as ro
    from rpy2.rinterface_lib import callbacks
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    rand = [r.replace(":", "__") for r in random]
    formula = formula_for(rand, fixed)
    factors = ro.StrVector([*fixed, *rand])
    console: list[str] = []
    saved = callbacks.consolewrite_warnerror
    callbacks.consolewrite_warnerror = console.append
    try:
        with localconverter(ro.default_converter + pandas2ri.converter):
            rdf = ro.conversion.get_conversion().py2rpy(data)
        out = ro.r(_R_FIT)(rdf, formula, factors, level, profile)
    finally:
        callbacks.consolewrite_warnerror = saved
    get = dict(zip(out.names, list(out), strict=True))
    grp = list(get["grp"])
    vcov = [float(v) for v in get["vcov"]]
    random_var = {
        g.replace("__", ":"): v for g, v in zip(grp, vcov, strict=True) if g != "Residual"
    }
    fe = dict(zip(list(get["fe_names"]), [float(v) for v in get["fe"]], strict=True))
    ci = {
        str(n): (float(lo), float(hi))
        for n, lo, hi in zip(get["ci_names"], get["ci_lo"], get["ci_hi"], strict=True)
    }
    random_ci: dict[str, tuple[float, float] | None] = {}
    for g in random_var:
        key = f"sd_(Intercept)|{g.replace(':', '__')}"
        bounds = ci.get(key)
        random_ci[g] = None if bounds is None else (bounds[0] ** 2, bounds[1] ** 2)
    fixed_ci: dict[str, tuple[float, float] | None] = {k: ci.get(k) for k in fe}
    msgs = [str(m) for m in get["msgs"]]
    converged = not any("converge" in m.lower() and "refit" not in m for m in msgs)
    return _RawFit(
        backend=f"lme4 {next(iter(get['version']))} via rpy2",
        random_var=random_var,
        random_ci=random_ci,
        fixed=fe,
        fixed_ci=fixed_ci,
        interval_method="profile" if profile and ci else "none",
        converged=converged,
        messages=msgs,
    )


def _fit_statsmodels(
    data: pd.DataFrame, random: Sequence[str], fixed: Sequence[str], level: float, seed: int
) -> _RawFit:
    import statsmodels
    from scipy import stats
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    vc = {r: f"0 + C({r.replace(':', '__')})" for r in random}
    rhs = " + ".join(f"C({f})" for f in fixed) or "1"
    np.random.seed(seed)  # fit_vb starts from a random point in some statsmodels versions
    model = BinomialBayesMixedGLM.from_formula(f"{OUTCOME} ~ {rhs}", vc, data)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = model.fit_vb()
    z = float(stats.norm.ppf(0.5 + level / 2))
    random_var: dict[str, float] = {}
    random_ci: dict[str, tuple[float, float] | None] = {}
    for i, name in enumerate(random):
        mean, sd = float(res.vcp_mean[i]), float(res.vcp_sd[i])
        random_var[name] = math.exp(2 * mean)
        random_ci[name] = (math.exp(2 * (mean - z * sd)), math.exp(2 * (mean + z * sd)))
    fixed_est: dict[str, float] = {}
    fixed_ci: dict[str, tuple[float, float] | None] = {}
    for i, name in enumerate(model.exog_names):
        key = "(Intercept)" if name == "Intercept" else _sm_to_r_name(name)
        mean, sd = float(res.fe_mean[i]), float(res.fe_sd[i])
        fixed_est[key] = mean
        fixed_ci[key] = (mean - z * sd, mean + z * sd)
    return _RawFit(
        backend=f"statsmodels {statsmodels.__version__} BinomialBayesMixedGLM (variational Bayes)",
        random_var=random_var,
        random_ci=random_ci,
        fixed=fixed_est,
        fixed_ci=fixed_ci,
        interval_method="vb_posterior",
        converged=True,
        messages=[str(w.message) for w in caught],
    )


def _sm_to_r_name(name: str) -> str:
    # statsmodels/patsy writes C(f)[T.level]; lme4 writes flevel.
    m = re.match(r"^C\((\w+)\)\[T\.(.*)\]$", name)
    return f"{m.group(1)}{m.group(2)}" if m else name


def _fixed_components(
    data: pd.DataFrame, fixed: Sequence[str], raw: _RawFit
) -> list[tuple[str, float, tuple[float, float] | None]]:
    """Variance of each factor's contribution to the linear predictor, with an interval."""
    out = []
    for factor in fixed:
        levels = sorted(data[factor].unique())
        coefs = {lvl: raw.fixed.get(f"{factor}{lvl}", 0.0) for lvl in levels}
        contribution = data[factor].map(coefs).to_numpy(dtype=np.float64)
        variance = float(np.var(contribution))
        interval: tuple[float, float] | None = None
        if len(levels) == 2:
            p = float(np.mean(data[factor] == levels[1]))
            bounds = raw.fixed_ci.get(f"{factor}{levels[1]}")
            if bounds is not None:
                lo_b, hi_b = bounds
                sq_lo = 0.0 if lo_b <= 0 <= hi_b else min(lo_b**2, hi_b**2)
                interval = (p * (1 - p) * sq_lo, p * (1 - p) * max(lo_b**2, hi_b**2))
        out.append((factor, variance, interval))
    return out


def decompose(
    results: EvalResults,
    *,
    random: Sequence[str] = ("task", "model", "model:task"),
    fixed: Sequence[str] = (),
    level: float = 0.95,
    intervals: bool = True,
    backend: str = "auto",
    seed: int = 0,
) -> Decomposition:
    """Fit a logit GLMM and partition the latent variance.

    ``random`` lists grouping columns (``a:b`` for crossed combinations) given
    random intercepts; ``fixed`` lists categorical covariates entered as fixed
    effects. Outcomes must be binary.
    """
    if not results.is_binary:
        raise ValueError("decompose needs binary outcomes")
    if not 0 < level < 1:
        raise ValueError("level must be in (0, 1)")
    frame = results.frame
    _check_names([*random, *fixed], frame)
    data = _prepare(frame, random, fixed)
    if backend == "auto":
        backend = "lme4" if lme4_available() else "statsmodels"
    if backend == "lme4":
        raw = _fit_lme4(data, random, fixed, level, intervals)
    elif backend == "statsmodels":
        raw = _fit_statsmodels(data, random, fixed, level, seed)
    else:
        raise ValueError(f"unknown backend {backend!r}")
    method = raw.interval_method if intervals else "none"
    parts: list[tuple[str, str, float, tuple[float, float] | None]] = []
    for name in random:
        ci = raw.random_ci.get(name) if intervals else None
        parts.append((name, "random", raw.random_var[name], ci))
    for name, variance, ci in _fixed_components(data, fixed, raw):
        parts.append((name, "fixed", variance, ci if intervals else None))
    parts.append(("residual", "residual", LOGISTIC_RESIDUAL_VAR, None))
    total = sum(p[2] for p in parts)
    components = tuple(
        VarianceComponent(
            name=name,
            kind=kind,
            variance=variance,
            share=variance / total,
            lo=None if ci is None else ci[0],
            hi=None if ci is None else ci[1],
            interval_method="fixed_logistic" if kind == "residual" else method,
        )
        for name, kind, variance, ci in parts
    )
    return Decomposition(
        backend=raw.backend,
        formula=formula_for(random, fixed),
        n_obs=len(frame),
        level=level,
        components=components,
        fixed_effects=dict(raw.fixed),
        converged=raw.converged,
        messages=tuple(raw.messages),
    )


def describe(decomp: Decomposition) -> list[dict[str, Any]]:
    """Plain rows for printing or JSON output."""
    return [
        {
            "component": c.name,
            "kind": c.kind,
            "variance": c.variance,
            "share": c.share,
            "lo": c.lo,
            "hi": c.hi,
            "interval": c.interval_method,
        }
        for c in decomp.components
    ]
