"""``python -m noisefloor.reanalysis build|analyze|verify|report`` (E2 and public E3)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from noisefloor.reanalysis import e2, report


def main(argv: list[str] | None = None) -> int:
    """Run one stage of the reanalysis."""
    parser = argparse.ArgumentParser(prog="python -m noisefloor.reanalysis")
    parser.add_argument("--verify", action="store_true", help="same as the verify command")
    sub = parser.add_subparsers(dest="command")
    b = sub.add_parser("build", help="snapshot and manifest from a raw fetch")
    b.add_argument("board", choices=e2.BOARDS)
    b.add_argument("raw", type=Path)
    a = sub.add_parser("analyze", help="compute and bank the results")
    a.add_argument("boards", nargs="*", default=list(e2.BOARDS))
    v = sub.add_parser("verify", help="recompute and compare with the banked results")
    v.add_argument("boards", nargs="*", default=list(e2.BOARDS))
    r = sub.add_parser("report", help="write docs/leaderboards/<board>.md")
    r.add_argument("boards", nargs="*", default=list(e2.BOARDS))
    args = parser.parse_args(argv)
    if args.verify or args.command == "verify":
        boards = tuple(getattr(args, "boards", None) or e2.BOARDS)
        problems = e2.verify(boards)
        for problem in problems:
            print(f"problem: {problem}")
        print("E2-VERIFY: " + ("PASS" if not problems else f"FAIL {len(problems)} problems"))
        return 0 if not problems else 1
    if args.command == "build":
        builder = e2.build_tb2 if args.board == "tb2" else e2.build_epoch
        manifest = builder(args.raw)
        print(f"built {manifest['snapshot']} ({manifest['sha256'][:12]})")
        return 0
    if args.command == "analyze":
        results = e2.analyze(tuple(args.boards))
        for board, res in results.items():
            q1 = res["sets"]["primary"]["q1"]
            print(f"{board}: {q1['n_supported']} of {q1['n_pairs']} adjacent pairs supported")
        return 0
    if args.command == "report":
        for path in report.write_reports(tuple(args.boards)):
            print(f"wrote {path.relative_to(report.REPO)}")
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
