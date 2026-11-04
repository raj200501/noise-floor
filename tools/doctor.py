#!/usr/bin/env python3
"""Environment doctor for noise-floor (`make doctor`).

Detects what this machine can run and which model backends are usable, then
prints a report. It never prints secret values: environment variables are
reported by name only. Standard library only, so it runs before `make setup`.

Exit status is 0 whenever the report was produced; usability is in the output.
Use `--json` for machine-readable output (consumed by the client wrapper).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent

# Section 8 billing guard: any of these makes the Claude backend unusable.
BILLING_GUARD_EXACT = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL")
BILLING_GUARD_PREFIX = "CLAUDE_CODE_USE_"
FREE_TIER_KEYS = ("GEMINI_API_KEY", "OPENROUTER_API_KEY", "GITHUB_MODELS_TOKEN")
CLAUDE_SETTINGS_FILES = (
    Path.home() / ".claude" / "settings.json",
    Path.home() / ".claude" / "settings.local.json",
    REPO / ".claude" / "settings.json",
    REPO / ".claude" / "settings.local.json",
    Path("/etc/claude-code/managed-settings.json"),
    Path("/Library/Application Support/ClaudeCode/managed-settings.json"),
)


def _run(cmd: list[str], timeout: float = 10.0) -> tuple[int, str]:
    """Run a command and return (exit code, combined output); never raises."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, str(exc)
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def billing_guard_findings() -> list[str]:
    """Names of variables or settings that trip the section 8 billing guard."""
    found = [name for name in BILLING_GUARD_EXACT if name in os.environ]
    found += sorted(n for n in os.environ if n.startswith(BILLING_GUARD_PREFIX))
    for path in CLAUDE_SETTINGS_FILES:
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and "apiKeyHelper" in data:
            found.append(f"apiKeyHelper in {path}")
    return found


def ram_gb() -> float | None:
    """Total physical memory in GiB, if it can be read."""
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        size = os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return None
    return round(pages * size / 2**30, 1)


def free_disk_gb(path: Path) -> float:
    """Free disk space in GiB at `path`."""
    return round(shutil.disk_usage(path).free / 2**30, 1)


def load_budget() -> dict[str, Any]:
    """Read budget.toml if present."""
    path = REPO / "budget.toml"
    if not path.exists():
        return {}
    with path.open("rb") as fh:
        return tomllib.load(fh)


def probe() -> dict[str, Any]:
    """Collect the environment report."""
    report: dict[str, Any] = {
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
        "python": sys.version.split()[0],
        "cpus": os.cpu_count(),
        "ram_gb": ram_gb(),
        "free_disk_gb": free_disk_gb(REPO),
    }
    tools: dict[str, Any] = {}
    for name in (
        "uv",
        "git",
        "make",
        "docker",
        "ollama",
        "Rscript",
        "bwrap",
        "unshare",
        "sandbox-exec",
        "nvidia-smi",
        "claude",
        "llama-server",
    ):
        tools[name] = shutil.which(name)
    report["tools"] = tools

    docker_ok = False
    if tools["docker"]:
        code, _ = _run(["docker", "info"], timeout=15)
        docker_ok = code == 0
    report["docker_daemon"] = docker_ok

    gpu = None
    if tools["nvidia-smi"]:
        code, out = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
        gpu = out if code == 0 else None
    report["gpu"] = gpu

    r_lme4 = False
    if tools["Rscript"]:
        code, _ = _run(["Rscript", "-e", "library(lme4)"], timeout=30)
        r_lme4 = code == 0
    report["r_lme4"] = r_lme4

    ollama_models: list[str] = []
    if tools["ollama"]:
        code, out = _run(["ollama", "list"], timeout=15)
        if code == 0:
            ollama_models = [ln.split()[0] for ln in out.splitlines()[1:] if ln.strip()]
    report["ollama_models"] = ollama_models

    sandbox = None
    if tools["bwrap"]:
        code, _ = _run(["bwrap", "--ro-bind", "/", "/", "--unshare-net", "true"])
        sandbox = "bwrap" if code == 0 else None
    if sandbox is None and tools["unshare"]:
        code, _ = _run(["unshare", "-rn", "true"])
        sandbox = "unshare-netns" if code == 0 else None
    if sandbox is None and tools["sandbox-exec"]:
        sandbox = "sandbox-exec"
    report["os_sandbox"] = sandbox

    claude_version = None
    if tools["claude"]:
        code, out = _run(["claude", "--version"])
        claude_version = out.splitlines()[0] if code == 0 and out else None
    report["claude_cli_version"] = claude_version
    guard = billing_guard_findings()
    report["billing_guard"] = guard
    # A hosted Claude Code session (CLAUDE_CODE_REMOTE) has no Max-plan login of its own.
    hosted = os.environ.get("CLAUDE_CODE_REMOTE", "").lower() in ("1", "true")
    report["claude_hosted_session"] = hosted
    report["claude_backend_usable"] = bool(claude_version) and not guard

    confirmed = os.environ.get("FREE_TIER_KEYS_CONFIRMED") == "1"
    report["free_tier_confirmed"] = confirmed
    report["free_tier_keys"] = [k for k in FREE_TIER_KEYS if k in os.environ] if confirmed else []

    budget = load_budget()
    cap = budget.get("local_model_max_params_b")
    report["local_model_max_params_b"] = cap
    ram = report["ram_gb"]
    # Rough rule: a 4-bit quantized model needs about 0.75 GiB per billion parameters
    # plus 2 GiB of headroom for the runtime and the OS.
    if ram is not None:
        report["ram_fits_params_b"] = max(0, int((ram - 2) / 0.75))
    floor = budget.get("disk_floor_gb")
    report["disk_floor_gb"] = floor
    report["disk_ok_for_downloads"] = floor is None or report["free_disk_gb"] > floor

    report["local_backend_usable"] = bool(ollama_models) or bool(tools["llama-server"])
    report["git_push_allowed_env"] = os.environ.get("ALLOW_PUSH") == "1"
    report["cap_share"] = os.environ.get("CAP_SHARE")
    return report


def _yes(flag: object) -> str:
    return "yes" if flag else "no"


def render(report: dict[str, Any]) -> str:
    """Human-readable report."""
    tools = report["tools"]
    if report["ollama_models"]:
        ollama = "models: " + ", ".join(report["ollama_models"])
    else:
        ollama = "installed, no models" if tools["ollama"] else "not installed"
    claude = report["claude_cli_version"] or "not found"
    if report["claude_hosted_session"]:
        claude += " (hosted session)"
    guard = report["billing_guard"]
    if report["free_tier_keys"]:
        free = ", ".join(report["free_tier_keys"])
    else:
        free = "confirmed, no keys" if report["free_tier_confirmed"] else "not confirmed (ignored)"
    rows = [
        ("platform", f"{report['platform']}, python {report['python']}"),
        ("cpus / ram", f"{report['cpus']} cpus, {report['ram_gb']} GiB RAM"),
        ("free disk", f"{report['free_disk_gb']} GiB (floor {report['disk_floor_gb']} GiB)"),
        ("docker", f"binary={_yes(tools['docker'])}, daemon={_yes(report['docker_daemon'])}"),
        ("gpu", report["gpu"] or "none"),
        ("R + lme4", _yes(report["r_lme4"])),
        ("os sandbox", report["os_sandbox"] or "none"),
        ("ollama", ollama),
        ("claude cli", claude),
        ("billing guard", "TRIPPED by " + ", ".join(guard) if guard else "clean"),
        ("free tiers", free),
        (
            "ram fits",
            f"about {report.get('ram_fits_params_b', '?')}B params at 4-bit "
            f"(cap {report['local_model_max_params_b']}B)",
        ),
    ]
    backends = [
        ("claude (plan)", "USABLE" if report["claude_backend_usable"] else "UNUSABLE"),
        (
            "local models",
            "USABLE" if report["local_backend_usable"] else "UNUSABLE (no runtime or models yet)",
        ),
        ("free tiers", "USABLE" if report["free_tier_keys"] else "UNUSABLE"),
        (
            "docker suites",
            "USABLE" if report["docker_daemon"] else "UNUSABLE (Docker-free fallback)",
        ),
        ("glmm backend", "lme4 via rpy2" if report["r_lme4"] else "python fallback"),
    ]
    lines = ["noise-floor doctor", ""]
    lines += [f"{label:<15} {value}" for label, value in rows]
    lines += ["", "backends"]
    lines += [f"  {label:<14} {value}" for label, value in backends]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print JSON instead of text")
    args = parser.parse_args(argv)
    report = probe()
    print(json.dumps(report, indent=2, sort_keys=True) if args.json else render(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
