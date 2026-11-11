#!/usr/bin/env python3
"""Budget report (`make budget`).

Reads budget.toml and the usage ledgers written by the client wrapper
(artifacts/usage/ledger.jsonl for this machine, artifacts/usage/ledger-claude.jsonl for the
Claude slice on Raj's machine; daily caps count both) and checks that cash spent is exactly
0 and that every daily cap was respected. `total_cost_usd` from Claude stream-json is an API-price
estimate and is reported as plan_equivalent_usd, never as cash.

Ledger entries are JSON objects with at least ``ts`` (ISO 8601) and ``kind``
(agent_run, single_call, review_run, local, free_tier); optional ``cash_usd``,
``plan_equivalent_usd`` and ``turns``. Exit 1 if any check fails.
"""

from __future__ import annotations

import json
import math
import os
import sys
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
LEDGERS = (
    REPO / "artifacts" / "usage" / "ledger.jsonl",
    REPO / "artifacts" / "usage" / "ledger-claude.jsonl",
)
DAILY_CAPS = {
    "agent_run": "claude_agent_runs_per_day",
    "single_call": "claude_single_calls_per_day",
    "review_run": "claude_review_runs_per_day",
}


def cap_share(caps: dict[str, Any], env: str | None) -> float:
    """Effective share: the smaller of budget.toml cap_share and the CAP_SHARE env value."""
    share = float(caps.get("cap_share", 1.0))
    if env:
        share = min(share, float(env))
    return max(0.0, min(1.0, share))


def effective_caps(caps: dict[str, Any], share: float) -> dict[str, Any]:
    """Every claude_* cap multiplied by the share and rounded down; other keys unchanged."""
    return {k: math.floor(v * share) if k.startswith("claude_") else v for k, v in caps.items()}


def load_caps(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        return tomllib.load(fh)


def load_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def check(caps: dict[str, Any], entries: list[dict[str, Any]]) -> tuple[list[str], bool]:
    """Report lines and overall ok flag."""
    ok = True
    lines = []
    cash = float(caps.get("cash_usd", -1)) + sum(float(e.get("cash_usd", 0)) for e in entries)
    if caps.get("cash_usd") != 0 or cash != 0:
        ok = False
    lines.append("cash_usd = 0" if ok else f"cash_usd = {cash:g}")
    plan = sum(float(e.get("plan_equivalent_usd", 0)) for e in entries)
    lines.append(f"plan_equivalent_usd = {plan:.2f} (API-price estimate, not cash)")
    per_day = Counter((str(e["ts"])[:10], str(e["kind"])) for e in entries)
    for kind, cap_name in DAILY_CAPS.items():
        cap = int(caps[cap_name])
        days = {d: n for (d, k), n in per_day.items() if k == kind}
        peak_day, peak = max(days.items(), key=lambda kv: kv[1], default=("-", 0))
        status = "OK" if peak <= cap else "EXCEEDED"
        ok = ok and peak <= cap
        lines.append(f"{cap_name}: peak {peak}/{cap} on {peak_day} {status}")
    max_turns = int(caps["claude_agent_max_turns"])
    worst = max(
        (int(e.get("turns", 0)) for e in entries if e.get("kind") == "agent_run"), default=0
    )
    ok = ok and worst <= max_turns
    status = "OK" if worst <= max_turns else "EXCEEDED"
    lines.append(f"claude_agent_max_turns: worst {worst}/{max_turns} {status}")
    lines.append(f"ledger entries: {len(entries)}")
    lines.append("BUDGET: " + ("OK" if ok else "VIOLATED"))
    return lines, ok


def main() -> int:
    caps = load_caps(REPO / "budget.toml")
    share = cap_share(caps, os.environ.get("CAP_SHARE"))
    entries = [e for path in LEDGERS for e in load_ledger(path)]
    lines, ok = check(effective_caps(caps, share), entries)
    print(f"cap_share = {share:g}")
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
