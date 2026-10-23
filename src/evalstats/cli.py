"""Command-line interface: ``evalstats compare|power|rank|decompose|report``."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from typing import Any

from evalstats import __version__
from evalstats.compare import paired_difference
from evalstats.intervals import INTERVAL_METHODS
from evalstats.power import PairedDesign, design_from_results, mde, tasks_needed
from evalstats.readers import read_results
from evalstats.schema import EvalResults, SchemaError

SUBCOMMANDS: dict[str, str] = {
    "compare": "paired comparison of two models on shared tasks",
    "power": "power and minimum detectable effect for an eval design",
    "rank": "rank-confidence sets for a leaderboard",
    "decompose": "variance components of eval outcomes",
    "report": "confidence report for a leaderboard",
}
IMPLEMENTED = ("compare", "power")


def _add_input(p: argparse.ArgumentParser, required: bool) -> None:
    p.add_argument("input", nargs=None if required else "?", help="CSV, TSV, or JSONL results")
    p.add_argument("--model-col", default="model")
    p.add_argument("--task-col", default="task")
    p.add_argument("--run-col", default="run", help="use '' if there is no run column")
    p.add_argument("--outcome-col", default="outcome")


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
            p.add_argument("--n-boot", type=int, default=10_000)
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
    return parser


def _load(args: argparse.Namespace) -> EvalResults:
    return read_results(
        args.input,
        model=args.model_col,
        task=args.task_col,
        run=args.run_col or None,
        outcome=args.outcome_col,
    )


def _emit(payload: dict[str, Any], text: str, as_json: bool) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True) if as_json else text)


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


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    args = build_parser().parse_args(argv)
    if args.command not in IMPLEMENTED:
        print(f"evalstats {args.command}: not implemented yet", file=sys.stderr)
        return 2
    try:
        return cmd_compare(args) if args.command == "compare" else cmd_power(args)
    except (SchemaError, ValueError, OSError) as exc:
        print(f"evalstats {args.command}: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
