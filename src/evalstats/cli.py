"""Command-line interface: ``evalstats compare|power|rank|decompose|report``."""

from __future__ import annotations

import argparse
import sys

from evalstats import __version__

SUBCOMMANDS: dict[str, str] = {
    "compare": "paired comparison of two models on shared tasks",
    "power": "power and minimum detectable effect for an eval design",
    "rank": "rank-confidence sets for a leaderboard",
    "decompose": "variance components of eval outcomes",
    "report": "confidence report for a leaderboard",
}


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(prog="evalstats", description=__doc__)
    parser.add_argument("--version", action="version", version=f"evalstats {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in SUBCOMMANDS.items():
        sub.add_parser(name, help=help_text, description=help_text)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    args = build_parser().parse_args(argv)
    print(f"evalstats {args.command}: not implemented yet", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
