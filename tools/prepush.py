#!/usr/bin/env python3
"""Pre-push check (DECISIONS.md D-015): `make setup` and `make ci` on a fresh clone of HEAD.

Run before every push so nothing that reaches GitHub fails CI. Standard library only.
Prints one line per step and a final `PREPUSH: PASS <sha>` or `PREPUSH: FAIL <step>`.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main() -> int:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()
    tmp = Path(tempfile.mkdtemp(prefix="prepush-"))
    try:
        clone = tmp / "clone"
        subprocess.run(
            ["git", "clone", "--quiet", "--no-hardlinks", str(REPO), str(clone)], check=True
        )
        subprocess.run(
            ["git", "-C", str(clone), "checkout", "--quiet", "--detach", head], check=True
        )
        for step in ("setup", "ci"):
            proc = subprocess.run(
                ["make", "-s", "--no-print-directory", step],
                cwd=clone,
                capture_output=True,
                text=True,
                check=False,
            )
            (tmp / f"{step}.log").write_text(proc.stdout + proc.stderr)
            print(f"make {step}: exit {proc.returncode}")
            if proc.returncode != 0:
                tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-15:])
                print(tail)
                print(f"PREPUSH: FAIL make {step}")
                return 1
        print(f"PREPUSH: PASS {head}")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
