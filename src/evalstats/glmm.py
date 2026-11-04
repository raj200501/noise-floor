"""Logit GLMMs with crossed random effects, and variance partitioning.

``decompose`` fits ``outcome ~ fixed factors + (1 | g1) + (1 | g2) + ...`` and splits
the latent (logit-scale) variance into random-effect variances, the variance of
each fixed factor's contribution to the linear predictor, and the logistic
residual variance ``pi^2 / 3``.

Backends (named in every result):

* ``lme4``: ``glmer`` (Laplace, bobyqa) through rpy2, with profile-likelihood
  intervals (``method="profile"``) or a parametric bootstrap
  (``method="parametric_bootstrap"``: each replicate draws new random effects and new
  outcomes from the fitted model and refits it, optionally with ``nagq=0``, which is
  about ten times faster). Used when R, lme4, and rpy2 are available.
* ``statsmodels-vb``: ``BinomialBayesMixedGLM`` fitted by variational Bayes, with
  approximate posterior intervals (labeled ``vb_posterior``). The fallback.

No variance-component interval is a Wald interval. A random term ``a:b`` means
one random intercept per observed combination of ``a`` and ``b``.

Bootstrap intervals: a random-effect variance gets the ``percentile``, ``basic``
(``2 * estimate - quantiles``, cut at 0), or ``bc`` (bias-corrected percentile)
interval of its replicates (``boot_interval``). A fixed factor's contribution gets the
range of the contribution over a box of percentile intervals for the factor's
coefficients, each at level ``1 - (1 - level) / (k - 1)`` for ``k`` levels (Bonferroni),
so it covers at least as often as the box. Shares get percentile intervals.
"""

from __future__ import annotations

import itertools
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
    """One latent-scale variance component with its interval (None if not available).

    ``share_lo`` and ``share_hi`` bound the component's share of the total latent
    variance (bootstrap intervals only).
    """

    name: str
    kind: str
    variance: float
    share: float
    lo: float | None
    hi: float | None
    interval_method: str
    share_lo: float | None = None
    share_hi: float | None = None


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
    replicates: dict[str, tuple[float, ...]] = field(default_factory=dict)
    """Bootstrap replicate values of each non-residual component (bootstrap only)."""

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
    boot_random: dict[str, np.ndarray] = field(default_factory=dict)  # replicate variances
    boot_fixed: dict[str, np.ndarray] = field(default_factory=dict)  # replicate coefficients


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
  profile_one <- function(f, i) {
    tryCatch(
      withCallingHandlers(
        # devtol: deviance gains below 1e-4 found while profiling are numerical noise.
        stats::confint(
          f, parm = i, method = "profile", level = level, oldNames = FALSE, devtol = 1e-4
        ),
        warning = keep
      ),
      error = function(e) conditionMessage(e)
    )
  }
  profile_ci <- function(f) {
    # Profile each random-effect SD and factor coefficient on its own (not the intercept),
    # so a failure on one parameter (for example an SD on the boundary) loses only that one.
    n_theta <- length(lme4::getME(f, "theta"))
    fe_names <- names(lme4::fixef(f))
    parm <- c(seq_len(n_theta), n_theta + which(fe_names != "(Intercept)"))
    rows <- list()
    for (i in parm) {
      ci <- profile_one(f, i)
      if (is.character(ci)) {
        if (grepl("lower deviance", ci)) return(ci)
        msgs <<- c(msgs, paste("profile failed for parameter", i, ":", ci))
      } else {
        rows[[length(rows) + 1]] <- ci
      }
    }
    if (length(rows) == 0) return(NULL)
    do.call(rbind, rows)
  }
  ci <- NULL
  if (do_profile) {
    ci <- profile_ci(fit)
    if (is.character(ci)) {
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
      if (is.character(ci)) {
        msgs <- c(msgs, paste("profile failed:", ci))
        ci <- NULL
      }
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


_R_BOOT = """
function(df, formula, factors, nagq, n_boot, seed) {
  for (col in factors) df[[col]] <- factor(df[[col]])
  msgs <- character(0)
  keep <- function(w) {
    msgs <<- c(msgs, conditionMessage(w))
    invokeRestart("muffleWarning")
  }
  ctrl <- lme4::glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))
  fit <- withCallingHandlers(
    lme4::glmer(
      stats::as.formula(formula), data = df, family = stats::binomial, nAGQ = nagq,
      control = ctrl
    ),
    warning = keep
  )
  vc <- as.data.frame(lme4::VarCorr(fit))
  grp <- as.character(vc$grp)
  fe_names <- names(lme4::fixef(fit))
  set.seed(seed)
  sims <- stats::simulate(fit, nsim = n_boot)  # use.u = FALSE: new random effects each time
  boot_vc <- matrix(NA_real_, n_boot, length(grp))
  boot_fe <- matrix(NA_real_, n_boot, length(fe_names))
  failed <- 0L
  warned <- 0L
  for (b in seq_len(n_boot)) {
    hit <- FALSE
    f <- tryCatch(
      withCallingHandlers(
        lme4::refit(fit, sims[[b]]),
        warning = function(w) { hit <<- TRUE; invokeRestart("muffleWarning") }
      ),
      error = function(e) NULL
    )
    if (is.null(f)) { failed <- failed + 1L; next }
    if (hit) warned <- warned + 1L
    v <- as.data.frame(lme4::VarCorr(f))
    boot_vc[b, ] <- as.numeric(v$vcov[match(grp, as.character(v$grp))])
    boot_fe[b, ] <- as.numeric(lme4::fixef(f)[fe_names])
  }
  conv <- fit@optinfo$conv$lme4$messages
  list(
    grp = grp, vcov = as.numeric(vc$vcov), fe_names = fe_names,
    fe = as.numeric(lme4::fixef(fit)), boot_vc = boot_vc, boot_fe = boot_fe,
    failed = failed, warned = warned,
    msgs = unique(c(as.character(msgs), as.character(conv))),
    version = as.character(utils::packageVersion("lme4"))
  )
}
"""


def _fit_lme4_bootstrap(
    data: pd.DataFrame,
    random: Sequence[str],
    fixed: Sequence[str],
    *,
    nagq: int,
    n_boot: int,
    seed: int,
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
        out = ro.r(_R_BOOT)(rdf, formula, factors, nagq, n_boot, seed)
    finally:
        callbacks.consolewrite_warnerror = saved
    get = dict(zip(out.names, list(out), strict=True))
    grp = [str(g) for g in get["grp"]]
    vcov = [float(v) for v in get["vcov"]]
    fe_names = [str(n) for n in get["fe_names"]]
    boot_vc = np.asarray(get["boot_vc"], dtype=np.float64).reshape(n_boot, len(grp), order="F")
    boot_fe = np.asarray(get["boot_fe"], dtype=np.float64).reshape(n_boot, len(fe_names), order="F")
    random_var = {
        g.replace("__", ":"): v for g, v in zip(grp, vcov, strict=True) if g != "Residual"
    }
    boot_random = {
        g.replace("__", ":"): boot_vc[:, i] for i, g in enumerate(grp) if g != "Residual"
    }
    msgs = [str(m) for m in get["msgs"]]
    failed, warned = int(next(iter(get["failed"]))), int(next(iter(get["warned"])))
    msgs.append(f"bootstrap: {n_boot} replicates, {failed} failed, {warned} with warnings")
    return _RawFit(
        backend=f"lme4 {next(iter(get['version']))} via rpy2 (nAGQ={nagq})",
        random_var=random_var,
        random_ci={},
        fixed=dict(zip(fe_names, [float(v) for v in get["fe"]], strict=True)),
        fixed_ci={},
        interval_method="parametric_bootstrap",
        converged=not any("converge" in m.lower() for m in msgs[:-1]),
        messages=msgs,
        boot_random=boot_random,
        boot_fixed={n: boot_fe[:, i] for i, n in enumerate(fe_names)},
    )


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
        interval_method="profile" if profile else "none",
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


BOOT_INTERVALS = ("percentile", "basic", "bc")
METHODS = ("profile", "parametric_bootstrap")
_MAX_BOX_COEFS = 12  # a factor with more levels gets no box interval (2**12 vertices)


def boot_variance_interval(
    estimate: float, draws: np.ndarray, level: float, kind: str = "percentile"
) -> tuple[float, float] | None:
    """Bootstrap interval for a variance from its replicates (None if fewer than 2).

    ``percentile``: the replicate quantiles. ``basic``: ``2 * estimate`` minus the
    quantiles, cut at 0. ``bc``: the bias-corrected percentile interval (quantiles at
    ``Phi(2 z0 -+ z)``, with ``z0`` from the share of replicates below the estimate).
    """
    from scipy import stats

    d = draws[np.isfinite(draws)]
    if d.size < 2:
        return None
    a = (1 - level) / 2
    if kind == "percentile":
        lo, hi = np.quantile(d, [a, 1 - a])
    elif kind == "basic":
        q_lo, q_hi = np.quantile(d, [a, 1 - a])
        lo, hi = max(0.0, 2 * estimate - q_hi), max(0.0, 2 * estimate - q_lo)
    elif kind == "bc":
        below = float(np.mean(d < estimate) + 0.5 * np.mean(d == estimate))
        below = min(max(below, 1 / (d.size + 1)), d.size / (d.size + 1))
        z0, z = float(stats.norm.ppf(below)), float(stats.norm.ppf(1 - a))
        lo, hi = np.quantile(d, [stats.norm.cdf(2 * z0 - z), stats.norm.cdf(2 * z0 + z)])
    else:
        raise ValueError(f"unknown bootstrap interval {kind!r}; expected one of {BOOT_INTERVALS}")
    return float(lo), float(hi)


def _level_weights(data: pd.DataFrame, factor: str) -> tuple[list[str], np.ndarray]:
    levels = sorted(data[factor].unique())
    weights = np.array([float(np.mean(data[factor] == lvl)) for lvl in levels])
    return levels, weights


def _contribution_variance(weights: np.ndarray, coefs: np.ndarray) -> float:
    """Variance over rows of a factor's contribution; ``coefs`` omits the reference level."""
    c = np.concatenate([[0.0], coefs])
    return float(weights @ (c - weights @ c) ** 2)


def _box_interval(
    weights: np.ndarray, estimate: np.ndarray, draws: np.ndarray, level: float
) -> tuple[float, float] | None:
    """Range of the contribution variance over a Bonferroni box of percentile intervals."""
    from scipy import optimize

    k1 = draws.shape[1]
    if k1 == 0 or k1 > _MAX_BOX_COEFS:
        return None
    a = (1 - level) / k1
    bounds = []
    for j in range(k1):
        d = draws[:, j][np.isfinite(draws[:, j])]
        if d.size < 2:
            return None
        lo, hi = np.quantile(d, [a / 2, 1 - a / 2])
        bounds.append((float(lo), float(hi)))
    # A convex quadratic: its maximum over the box is at a vertex, its minimum is found by
    # bounded minimization from the clipped estimate.
    hi_v = max(_contribution_variance(weights, np.array(v)) for v in itertools.product(*bounds))
    if all(lo <= 0.0 <= hi for lo, hi in bounds):
        return 0.0, hi_v  # the box holds the origin, where the contribution is exactly 0
    start = np.clip(estimate, [b[0] for b in bounds], [b[1] for b in bounds])

    def objective(x: np.ndarray) -> float:
        return _contribution_variance(weights, x)

    res = optimize.minimize(objective, start, method="L-BFGS-B", bounds=bounds)
    return max(0.0, float(res.fun)), hi_v


def _boot_parts(
    data: pd.DataFrame,
    random: Sequence[str],
    fixed: Sequence[str],
    *,
    raw: _RawFit,
    level: float,
    boot_interval: str,
) -> list[tuple[str, str, float, tuple[float, float] | None, np.ndarray]]:
    """(name, kind, estimate, interval, replicate values) for every non-residual component."""
    parts = []
    for name in random:
        draws = raw.boot_random[name]
        est = raw.random_var[name]
        ci = boot_variance_interval(est, draws, level, boot_interval)
        parts.append((name, "random", est, ci, draws))
    for factor in fixed:
        levels, weights = _level_weights(data, factor)
        names = [f"{factor}{lvl}" for lvl in levels[1:]]
        estimate = np.array([raw.fixed.get(n, 0.0) for n in names])
        cols = [raw.boot_fixed.get(n) for n in names]
        n_boot = len(next(iter(raw.boot_fixed.values()))) if raw.boot_fixed else 0
        draws = (
            np.column_stack([c if c is not None else np.full(n_boot, np.nan) for c in cols])
            if names
            else np.empty((n_boot, 0))
        )
        variance = _contribution_variance(weights, estimate)
        ci = _box_interval(weights, estimate, draws, level)
        reps = np.array(
            [
                _contribution_variance(weights, row) if np.all(np.isfinite(row)) else np.nan
                for row in draws
            ]
        )
        parts.append((factor, "fixed", variance, ci, reps))
    return parts


def _check_options(level: float, method: str, boot_interval: str, nagq: int, n_boot: int) -> None:
    if not 0 < level < 1:
        raise ValueError("level must be in (0, 1)")
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    if boot_interval not in BOOT_INTERVALS:
        raise ValueError(f"unknown bootstrap interval {boot_interval!r}")
    if nagq not in (0, 1) or n_boot < 2:
        raise ValueError("nagq must be 0 or 1 and n_boot at least 2")


def decompose(
    results: EvalResults,
    *,
    random: Sequence[str] = ("task", "model", "model:task"),
    fixed: Sequence[str] = (),
    level: float = 0.95,
    intervals: bool = True,
    backend: str = "auto",
    seed: int = 0,
    method: str = "profile",
    n_boot: int = 200,
    nagq: int = 1,
    boot_interval: str = "percentile",
) -> Decomposition:
    """Fit a logit GLMM and partition the latent variance.

    ``random`` lists grouping columns (``a:b`` for crossed combinations) given
    random intercepts; ``fixed`` lists categorical covariates entered as fixed
    effects. Outcomes must be binary. ``method`` picks the lme4 intervals:
    ``profile`` (profile likelihood) or ``parametric_bootstrap`` (``n_boot``
    replicates, refits with ``nagq`` = 1 for Laplace or 0 for the faster
    approximation, random variances with ``boot_interval``; see the module docstring).
    """
    if not results.is_binary:
        raise ValueError("decompose needs binary outcomes")
    _check_options(level, method, boot_interval, nagq, n_boot)
    frame = results.frame
    _check_names([*random, *fixed], frame)
    data = _prepare(frame, random, fixed)
    if backend == "auto":
        backend = "lme4" if lme4_available() else "statsmodels"
    if method == "parametric_bootstrap":
        if backend != "lme4":
            raise ValueError("the parametric bootstrap needs the lme4 backend")
        raw = _fit_lme4_bootstrap(data, random, fixed, nagq=nagq, n_boot=n_boot, seed=seed)
        return _bootstrap_decomposition(
            frame, data, random, fixed, raw=raw, level=level, boot_interval=boot_interval
        )
    if backend == "lme4":
        raw = _fit_lme4(data, random, fixed, level, intervals)
    elif backend == "statsmodels":
        raw = _fit_statsmodels(data, random, fixed, level, seed)
    else:
        raise ValueError(f"unknown backend {backend!r}")
    method_name = raw.interval_method if intervals else "none"
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
            interval_method="fixed_logistic" if kind == "residual" else method_name,
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


def _bootstrap_decomposition(
    frame: pd.DataFrame,
    data: pd.DataFrame,
    random: Sequence[str],
    fixed: Sequence[str],
    *,
    raw: _RawFit,
    level: float,
    boot_interval: str,
) -> Decomposition:
    parts = _boot_parts(data, random, fixed, raw=raw, level=level, boot_interval=boot_interval)
    total = sum(p[2] for p in parts) + LOGISTIC_RESIDUAL_VAR
    reps = np.column_stack([p[4] for p in parts]) if parts else np.empty((0, 0))
    totals = reps.sum(axis=1) + LOGISTIC_RESIDUAL_VAR if parts else np.empty(0)
    a = (1 - level) / 2

    def share_ci(values: np.ndarray) -> tuple[float | None, float | None]:
        ok = np.isfinite(values) & np.isfinite(totals)
        if ok.sum() < 2:
            return None, None
        lo, hi = np.quantile(values[ok] / totals[ok], [a, 1 - a])
        return float(lo), float(hi)

    components = []
    for name, kind, variance, ci, values in parts:
        s_lo, s_hi = share_ci(values)
        components.append(
            VarianceComponent(
                name=name,
                kind=kind,
                variance=variance,
                share=variance / total,
                lo=None if ci is None else ci[0],
                hi=None if ci is None else ci[1],
                interval_method="parametric_bootstrap_"
                + (boot_interval if kind == "random" else "box"),
                share_lo=s_lo,
                share_hi=s_hi,
            )
        )
    s_lo, s_hi = share_ci(np.full(len(totals), LOGISTIC_RESIDUAL_VAR))
    components.append(
        VarianceComponent(
            name="residual",
            kind="residual",
            variance=LOGISTIC_RESIDUAL_VAR,
            share=LOGISTIC_RESIDUAL_VAR / total,
            lo=None,
            hi=None,
            interval_method="fixed_logistic",
            share_lo=s_lo,
            share_hi=s_hi,
        )
    )
    return Decomposition(
        backend=raw.backend,
        formula=formula_for(random, fixed),
        n_obs=len(frame),
        level=level,
        components=tuple(components),
        fixed_effects=dict(raw.fixed),
        converged=raw.converged,
        messages=tuple(raw.messages),
        replicates={name: tuple(float(v) for v in values) for name, _, _, _, values in parts},
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
