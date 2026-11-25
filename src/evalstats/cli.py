"""Command-line interface: ``evalstats compare|power|rank|decompose|report``."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from dataclasses import asdict
from typing import Any, TextIO

from evalstats import __version__
from evalstats.compare import paired_difference
from evalstats.glmm import decompose, describe
from evalstats.intervals import INTERVAL_METHODS
from evalstats.power import PairedDesign, design_from_results, mde, tasks_needed
from evalstats.rank import rank_confidence
from evalstats.readers import read_results
from evalstats.report import is_synthetic, render_report, report_inputs
from evalstats.schema import EvalResults, SchemaError

SUBCOMMANDS: dict[str, str] = {
    "compare": "paired comparison of two models on shared tasks",
    "power": "power and minimum detectable effect for an eval design",
    "rank": "rank-confidence sets for a leaderboard",
    "decompose": "variance components of eval outcomes",
    "report": "confidence report for a leaderboard",
}
# Input formats: a results table, or a harness's own output read by evalstats.adapters.
SOURCES = ("table", "inspect", "harbor")


def _add_input(p: argparse.ArgumentParser, required: bool) -> None:
    p.add_argument(
        "input",
        nargs=None if required else "?",
        help="a CSV, TSV, or JSONL results table; with --from, an Inspect AI log file or "
        "directory, or a Harbor job directory",
    )
    p.add_argument(
        "--from",
        dest="source",
        default="table",
        choices=SOURCES,
        help="input format (default: a results table)",
    )
    p.add_argument("--model-col", default="model", help="tables only")
    p.add_argument("--task-col", default="task", help="tables only")
    p.add_argument("--run-col", default="run", help="tables only; use '' if there is no run column")
    p.add_argument("--outcome-col", default="outcome", help="tables only")
    p.add_argument("--scorer", help="with --from inspect: the scorer whose value is the outcome")
    p.add_argument("--value-key", help="with --from inspect: the entry of dict-valued scores")
    p.add_argument(
        "--run-key",
        default="epoch",
        choices=["epoch", "eval_epoch"],
        help="with --from inspect: eval_epoch keeps repeated logs of one model and task apart",
    )
    p.add_argument("--reward-key", help="with --from harbor: the reward used as the outcome")
    p.add_argument(
        "--model-label",
        default="model",
        choices=["model", "agent_model"],
        help="with --from harbor: agent_model names entries <agent>__<model>, so two agents "
        "with one model stay apart",
    )


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, not {value}")
    return value


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(prog="evalstats", description=__doc__)
    parser.add_argument("--version", action="version", version=f"evalstats {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in SUBCOMMANDS.items():
        p = sub.add_parser(name, help=help_text, description=help_text)
        if name == "compare":
            _add_input(p, required=True)
            p.add_argument("--a", required=True, help="first model")
            p.add_argument("--b", required=True, help="second model")
            p.add_argument("--method", default="cluster_t", choices=sorted(INTERVAL_METHODS))
            p.add_argument("--level", type=float, default=0.95)
            p.add_argument("--n-boot", type=_positive_int, default=10_000)
            p.add_argument("--seed", type=int, default=None)
            p.add_argument("--json", action="store_true", help="print JSON")
        elif name == "power":
            _add_input(p, required=False)
            p.add_argument("--a", help="first model (estimate variances from input)")
            p.add_argument("--b", help="second model (estimate variances from input)")
            p.add_argument(
                "--between-var", type=float, help="between-task variance of the difference"
            )
            p.add_argument("--within-var", type=float, help="run-level variance of A plus B")
            p.add_argument("--tasks", type=int, nargs="+", default=[50, 100, 200, 500])
            p.add_argument("--runs", type=int, nargs="+", default=[1, 3, 5])
            p.add_argument("--effect", type=float, help="also report tasks needed for this effect")
            p.add_argument("--alpha", type=float, default=0.05)
            p.add_argument("--power", type=float, default=0.8)
            p.add_argument("--json", action="store_true", help="print JSON")
        elif name == "rank":
            _add_input(p, required=True)
            p.add_argument("--level", type=float, default=0.95)
            p.add_argument(
                "--marginal",
                action="store_true",
                help="marginal rank intervals instead of simultaneous ones",
            )
            p.add_argument("--n-boot", type=_positive_int, default=2000)
            p.add_argument("--seed", type=int, default=None)
            p.add_argument("--json", action="store_true", help="print JSON")
        elif name == "decompose":
            _add_input(p, required=True)
            p.add_argument(
                "--random",
                nargs="+",
                default=["task", "model", "model:task"],
                help="grouping terms with random intercepts (a:b for combinations)",
            )
            p.add_argument("--fixed", nargs="*", default=[], help="categorical fixed factors")
            p.add_argument("--backend", default="auto", choices=["auto", "lme4", "statsmodels"])
            p.add_argument("--level", type=float, default=0.95)
            p.add_argument("--no-intervals", action="store_true", help="skip profile intervals")
            p.add_argument("--json", action="store_true", help="print JSON")
        elif name == "report":
            _add_input(p, required=True)
            p.add_argument("--title", default="leaderboard", help="report title")
            p.add_argument("--level", type=float, default=0.95, help="confidence level")
            p.add_argument(
                "--n-boot", type=_positive_int, default=2000, help="task-bootstrap replicates"
            )
            p.add_argument(
                "--seed", type=int, default=0, help="bootstrap seed (fixed, so reruns match)"
            )
            p.add_argument("--json", action="store_true", help="print JSON")
    return parser


def _load(args: argparse.Namespace) -> EvalResults:
    if args.source == "inspect":
        from evalstats.adapters import read_inspect  # noqa: PLC0415 - only when asked

        return read_inspect(
            args.input, scorer=args.scorer, value_key=args.value_key, run_key=args.run_key
        )
    if args.source == "harbor":
        from evalstats.adapters import read_harbor  # noqa: PLC0415 - only when asked

        return read_harbor(args.input, reward_key=args.reward_key, model_label=args.model_label)
    return read_results(
        args.input,
        model=args.model_col,
        task=args.task_col,
        run=args.run_col or None,
        outcome=args.outcome_col,
    )


def _finite(value: Any) -> Any:
    """JSON-safe values: NaN and infinities become None, recursively."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_finite(v) for v in value]
    return value


def _emit(payload: dict[str, Any], text: str, as_json: bool) -> None:
    if as_json:
        text = json.dumps(_finite(payload), indent=2, sort_keys=True, allow_nan=False)
    print(text)


def cmd_compare(args: argparse.Namespace) -> int:
    """Run ``evalstats compare``."""
    res = paired_difference(
        _load(args),
        args.a,
        args.b,
        method=args.method,
        level=args.level,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    d = res.difference
    text = "\n".join(
        [
            f"{res.model_a} vs {res.model_b} on {res.n_tasks} shared tasks "
            f"(unpaired: {res.unpaired_tasks_a} only in A, {res.unpaired_tasks_b} only in B)",
            f"score A {res.score_a:.4f}, score B {res.score_b:.4f}",
            f"difference A - B {d.estimate:+.4f}, {100 * d.level:g}% CI "
            f"[{d.lo:+.4f}, {d.hi:+.4f}] ({d.method})",
            f"paired t {res.t_stat:.3f}, p = {res.p_value:.4g}; ordering "
            + ("supported" if res.supported else "not supported")
            + f" at {100 * d.level:g}%",
        ]
    )
    payload = asdict(res) | {"supported": res.supported}
    _emit(payload, text, args.json)
    return 0


def cmd_power(args: argparse.Namespace) -> int:
    """Run ``evalstats power``."""
    if args.input:
        if not (args.a and args.b):
            raise SchemaError("--a and --b are required with an input file")
        design = design_from_results(_load(args), args.a, args.b)
        source = f"estimated from {args.input} ({args.a} vs {args.b})"
    elif args.source != "table":
        raise SchemaError("--from needs an input path")
    elif args.between_var is not None and args.within_var is not None:
        design = PairedDesign(args.between_var, args.within_var)
        source = "given"
    else:
        raise SchemaError("give an input file with --a/--b, or --between-var and --within-var")
    rows = [
        {"tasks": t, "runs": k, "mde": mde(design, t, k, alpha=args.alpha, power=args.power)}
        for t in args.tasks
        for k in args.runs
    ]
    lines = [
        f"variances {source}: between {design.between_task_var:.5f}, "
        f"within {design.within_task_var:.5f}",
        f"MDE at alpha {args.alpha:g}, power {args.power:g} (normal approximation)",
        f"{'tasks':>6} {'runs':>5} {'mde':>8}",
        *[f"{r['tasks']:>6} {r['runs']:>5} {r['mde']:>8.4f}" for r in rows],
    ]
    payload: dict[str, Any] = {"design": asdict(design), "source": source, "rows": rows}
    if args.effect is not None:
        needed = {
            str(k): tasks_needed(args.effect, design, k, alpha=args.alpha, power=args.power)
            for k in args.runs
        }
        payload["tasks_needed"] = {"effect": args.effect, "by_runs": needed}
        lines.append(
            f"tasks needed for effect {args.effect:g}: "
            + ", ".join(f"{n} at {k} runs" for k, n in needed.items())
        )
    _emit(payload, "\n".join(lines), args.json)
    return 0


def cmd_decompose(args: argparse.Namespace) -> int:
    """Run ``evalstats decompose``."""
    d = decompose(
        _load(args),
        random=args.random,
        fixed=args.fixed,
        level=args.level,
        intervals=not args.no_intervals,
        backend=args.backend,
    )
    rows = describe(d)

    def fmt(x: float | None) -> str:
        return "-" if x is None else f"{x:.4f}"

    lines = [
        f"{d.formula}  [{d.backend}; {d.n_obs} runs; converged: {d.converged}]",
        f"latent (logit) scale; {100 * d.level:g}% intervals",
        f"{'component':<16} {'kind':<9} {'variance':>9} {'share':>6}  interval",
        *[
            f"{r['component']:<16} {r['kind']:<9} {r['variance']:9.4f} {r['share']:6.3f}  "
            f"[{fmt(r['lo'])}, {fmt(r['hi'])}] {r['interval']}"
            for r in rows
        ],
        *[f"note: {m}" for m in d.messages],
    ]
    payload = {
        "backend": d.backend,
        "formula": d.formula,
        "n_obs": d.n_obs,
        "converged": d.converged,
        "components": rows,
        "messages": list(d.messages),
    }
    _emit(payload, "\n".join(lines), args.json)
    return 0


def cmd_rank(args: argparse.Namespace) -> int:
    """Run ``evalstats rank``."""
    rc = rank_confidence(
        _load(args),
        level=args.level,
        simultaneous=not args.marginal,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    kind = "simultaneous" if rc.simultaneous else "marginal"
    width = max(len(r.model) for r in rc.intervals)
    lines = [
        f"{len(rc.intervals)} models on {rc.n_tasks} shared tasks "
        f"({rc.dropped_tasks} tasks dropped: not attempted by every model)",
        f"{100 * rc.level:g}% {kind} rank intervals ({rc.n_boot} task resamples)",
        f"{'rank':>4}  {'model':<{width}} {'score':>6}  interval",
        *[
            f"{r.rank:>4}  {r.model:<{width}} {r.score:6.3f}  [{r.best_rank}, {r.worst_rank}]"
            for r in rc.intervals
        ],
        f"adjacent pairs (paired difference, {100 * rc.level:g}% marginal interval):",
        *[
            f"  {p.higher} > {p.lower}: {p.difference:+.3f} [{p.lo:+.3f}, {p.hi:+.3f}] "
            + ("supported" if p.supported else "not supported")
            for p in rc.adjacent
        ],
        f"share of adjacent pairs supported: {rc.share_adjacent_supported:.3f}",
    ]
    payload = asdict(rc) | {
        "share_adjacent_supported": rc.share_adjacent_supported,
        "top_set": list(rc.top_set),
    }
    _emit(payload, "\n".join(lines), args.json)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """``evalstats report``: a Markdown confidence report for a leaderboard."""
    results = _load(args)
    ranks, bounds = report_inputs(results, level=args.level, n_boot=args.n_boot, seed=args.seed)
    synthetic = is_synthetic(results)
    payload = {
        "ranks": asdict(ranks),
        "policy_bounds": None if bounds is None else asdict(bounds),
        "synthetic": synthetic,
    }
    text = render_report(ranks, bounds, title=args.title, synthetic=synthetic).rstrip("\n")
    _emit(payload, text, args.json)
    return 0


COMMANDS = {
    "compare": cmd_compare,
    "power": cmd_power,
    "decompose": cmd_decompose,
    "rank": cmd_rank,
    "report": cmd_report,
}


# Library messages name keyword arguments ("pass run_key='eval_epoch'"); here they are flags.
_KEYWORD = re.compile(r"\bpass (run_key|reward_key|value_key|model_label)(?:='([^']*)')?")


def _flags(message: str) -> str:
    def flag(m: re.Match[str]) -> str:
        value = f" {m.group(2)}" if m.group(2) else ""
        return f"pass --{m.group(1).replace('_', '-')}{value}"

    return _KEYWORD.sub(flag, message)


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    args = build_parser().parse_args(argv)
    prefix = f"evalstats {args.command}"

    def show(  # noqa: PLR0917 - the signature of warnings.showwarning
        message: Warning | str,
        category: type[Warning],
        filename: str,
        lineno: int,
        file: TextIO | None = None,
        line: str | None = None,
    ) -> None:
        print(f"{prefix}: warning: {_flags(str(message))}", file=sys.stderr)

    with warnings.catch_warnings():
        warnings.showwarning = show
        try:
            return COMMANDS[args.command](args)
        except (SchemaError, ValueError, OSError) as exc:
            print(f"{prefix}: error: {_flags(str(exc))}", file=sys.stderr)
            return 1


if __name__ == "__main__":
    sys.exit(main())
