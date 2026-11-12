#!/usr/bin/env python3
"""ship-check: the 14 Definition-of-Done gates of docs/BUILD_BRIEF.md section 6.

Order of work:

1. In the original checkout, check the G14 parts about the working tree and
   private/ (clean tree, nothing under private/ tracked, private hash manifest).
2. Clone HEAD into a temporary directory and run every other check there.

It verifies banked artifacts (presence, provenance, hashes, preregistration links,
thresholds) and recomputes statistics from them. It never runs an experiment or
calls a model, and it writes nothing inside the working tree: the clone, the
logs, and every build product live in the temporary directory.

Stdout is exactly::

    HEAD <sha> fresh-clone
    G1 PASS|FAIL <short evidence>
    ...
    G14 PASS|FAIL <short evidence>
    SHIP-CHECK: PASS 14/14            (or SHIP-CHECK: FAIL <passed>/14)

With ``--fast`` some slow steps are skipped (named in the evidence) and the last
line is ``SHIP-CHECK-FAST (does not count): <passed>/14``. Only a full run counts.

Artifact contracts that the gates read are documented in docs/GATES.md.
Standard library only (Python 3.11+).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import tokenize
import tomllib
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
PACKAGES = ("evalstats", "noisefloor")
COVERAGE_MIN = 85.0
DEMO_LIMIT_S = 180.0
MIN_REPS = 2000
MIN_REPS_RESAMPLING = 500  # bootstrap or GLMM intervals
RESAMPLING_TYPES = {
    "bootstrap",
    "bca",
    "bootstrap_max_t",
    "parametric_bootstrap",
    "glmm",
    "glmm_profile",
    "profile_glmm",
    "glmm_task_jackknife",  # GLMM fixed contributions (D-033), as D-017 read bca
    "glmm_plugin_share",  # GLMM shares mapped from the component intervals (D-033)
}
# E0 cell method -> the function whose validity check matches it (noisefloor.e0_validity).
FUNCTION_OF = {
    "score_interval": "score_interval",
    "paired_difference": "paired_difference",
    "contrast_difference": "contrast_difference",
    "cluster_robust_mean": "cluster_robust_mean",
    "rank_simultaneous": "rank_confidence",
    "rank_marginal": "rank_confidence",
    "bounded_score": "policy_bounds",
    "glmm_component": "decompose",
    "glmm_mde": "replicate_mdes",
    "sampling_discordance": "score_interval",
}
G6_MIN_DATASETS = 200
E1_MIN_RUNS = 1000
E1_MIN_CLAUDE_RUNS = 150
POST_E1_DATASETS = 200  # prereg 0004 item 10: the post-E1 check's seeds 980000 to 980199
E1_VERIFY_TIMEOUT = 6 * 3600  # the recompute fits four GLMMs with bootstrap and jackknife
E0_PART2_ARTIFACT = "artifacts/e0/coverage-0005.json"  # prereg 0005, banked beside G6
E2_MIN_BOARDS = 2
REQUIRED_FILES = ("LICENSE", "CITATION.cff", "SECURITY.md", "CONTRIBUTING.md")
SECRET_ENV = (
    "GH_TOKEN",
    "GITHUB_TOKEN",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "GEMINI_API_KEY",
    "OPENROUTER_API_KEY",
    "GITHUB_MODELS_TOKEN",
    "OPENAI_API_KEY",
    "SSH_AUTH_SOCK",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "HF_TOKEN",
)
HYPE_WORDS = ("revolutionary", "groundbreaking", "state of the art", "state-of-the-art")
README_SECTIONS = (
    ("what it is", r"what (it|this) is"),
    ("key results", r"key results|results"),
    ("how it works", r"how it works"),
    ("own benchmark", r"your own|own benchmark|use it on"),
    ("reproduce", r"reproduc"),
    ("limitations", r"limitations|threats"),
    ("related work", r"related work"),
    ("citation", r"citation|cite"),
    ("license", r"license"),
)
REPORT_SECTIONS = (
    "abstract",
    "introduction",
    "related work",
    "method",
    "experiments",
    "results",
    "limitations",
    "ethics",
    "reproducibility",
)
EM_DASH = chr(0x2014)
FENCE_RE = re.compile(r"```json[ \t]+([\w-]+)[ \t]*\n(.*?)\n```", re.S)


# --------------------------------------------------------------------------- helpers


class GateError(Exception):
    """A gate cannot pass; the message is the short evidence."""


@dataclass
class Result:
    code: int
    out: str  # stdout then stderr, for logs and messages
    seconds: float
    stdout: str = ""  # stdout alone, for parsing (tools may warn on stderr)


@dataclass
class Ctx:
    repo: Path
    clone: Path
    tmp: Path
    fast: bool
    env: dict[str, str]
    netns: list[str]
    setup_ok: bool | None = None
    results: dict[str, Result] = field(default_factory=dict)


def sha256_file(path: Path) -> str:
    h = sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_env() -> dict[str, str]:
    """Environment for sub-commands: secrets removed, ship-check marker set."""
    env = {k: v for k, v in os.environ.items() if k not in SECRET_ENV}
    env["NOISEFLOOR_SHIP_CHECK"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("VIRTUAL_ENV", None)
    return env


def run(
    ctx: Ctx,
    name: str,
    cmd: list[str],
    *,
    cwd: Path | None = None,
    timeout: float = 1800,
    env: dict[str, str] | None = None,
) -> Result:
    """Run a command, log its output under the temp dir, never raise."""
    start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd or ctx.clone,
            env=env or ctx.env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        res = Result(
            proc.returncode, proc.stdout + proc.stderr, time.monotonic() - start, proc.stdout
        )
        stdout = proc.stdout
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or b"").decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        res = Result(124, out + f"\nTIMEOUT after {timeout}s", time.monotonic() - start)
        stdout = out
    except OSError as exc:
        res = Result(127, str(exc), time.monotonic() - start)
        stdout = ""
    log = ctx.tmp / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(f"$ {' '.join(cmd)}\n[exit {res.code}, {res.seconds:.1f}s]\n{res.out}")
    (ctx.tmp / "logs" / f"{name}.stdout").write_text(stdout)
    ctx.results[name] = res
    return res


def make(
    ctx: Ctx,
    target: str,
    *,
    timeout: float = 1800,
    env: dict[str, str] | None = None,
    prefix: list[str] | None = None,
) -> Result:
    cmd = [*(prefix or []), "make", "-s", "--no-print-directory", target]
    return run(ctx, f"make-{target}", cmd, timeout=timeout, env=env)


def tail(text: str, n: int = 1) -> str:
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return " | ".join(lines[-n:])[:120] if lines else "no output"


def id_listed(issue_id: str, text: str) -> bool:
    """True if ``issue_id`` appears in ``text`` as a whole id (N1 is not listed by N10)."""
    return (
        bool(issue_id) and re.search(rf"(?<![\w-]){re.escape(issue_id)}(?![\w-])", text) is not None
    )


# Derived artifacts that claims cite, with the command that rebuilds and compares each (G10).
DERIVED_CHECKS = (
    ("artifacts/e2/noise_floor.json", ["python", "tools/noise_floor.py", "--check"]),
    (
        "artifacts/method_pilots/summary.json",
        ["python", "-m", "noisefloor.method_pilots", "--check"],
    ),
    ("validation summaries", ["python", "tools/validity_summary.py", "--check"]),
)


def need(cond: bool, evidence: str) -> None:
    if not cond:
        raise GateError(evidence)


def need_setup(ctx: Ctx) -> None:
    need(bool(ctx.setup_ok), "make setup failed (see G1)")


def _shown(path: Path) -> str:
    parts = path.parts
    return "/".join(parts[parts.index("clone") + 1 :]) if "clone" in parts else str(path)


def load_json(path: Path) -> Any:
    need(path.exists(), f"missing {_shown(path)}")
    try:
        return json.loads(path.read_text())
    except ValueError as exc:
        raise GateError(f"invalid JSON in {path.name}: {exc}") from exc


def rel(ctx: Ctx, path: str) -> Path:
    return ctx.clone / path


def git(cwd: Path, *args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise GateError(f"git {' '.join(args[:2])} failed: {tail(proc.stderr)}")
    return proc.stdout.strip()


def first_commit(cwd: Path, path: str) -> str:
    """The first commit that touched `path` (empty if never committed)."""
    out = git(cwd, "log", "--format=%H", "--reverse", "--", path)
    return out.splitlines()[0] if out else ""


def commits_touching(cwd: Path, path: str) -> list[str]:
    out = git(cwd, "log", "--format=%H", "--", path)
    return out.splitlines() if out else []


def strictly_before(cwd: Path, earlier: str, later: str) -> bool:
    if not earlier or not later or earlier == later:
        return False
    proc = subprocess.run(
        ["git", "merge-base", "--is-ancestor", earlier, later], cwd=cwd, check=False
    )
    return proc.returncode == 0


def prereg_block(ctx: Ctx, prereg: str, tag: str) -> Any:
    """Parse the fenced ```json <tag>``` block from a preregistration file."""
    path = rel(ctx, prereg)
    need(path.exists() and path.parent.name == "prereg", f"prereg file {prereg} missing")
    blocks = {m.group(1): m.group(2) for m in FENCE_RE.finditer(path.read_text())}
    need(tag in blocks, f"{prereg} has no ```json {tag}``` block")
    try:
        return json.loads(blocks[tag])
    except ValueError as exc:
        raise GateError(f"{prereg} {tag} block is not valid JSON") from exc


def check_prereg_link(ctx: Ctx, prereg: str, artifact: str) -> None:
    """The preregistration's commit must strictly predate the artifact's first commit."""
    p, a = first_commit(ctx.clone, prereg), first_commit(ctx.clone, artifact)
    need(bool(p), f"{prereg} not committed")
    need(bool(a), f"{artifact} not committed")
    need(strictly_before(ctx.clone, p, a), f"{prereg} does not predate {artifact}")


def utc_time(stamp: str) -> Any:
    """An ISO 8601 timestamp as an aware UTC datetime (``Z`` and offsets accepted)."""
    from datetime import UTC, datetime

    value = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    return value if value.tzinfo else value.replace(tzinfo=UTC)


def coverage_rule(covered: int, reps: int, nominal: float) -> tuple[bool, float, float]:
    """G5 rule: coverage >= nominal - 0.015 - 2 * MCSE. Returns (ok, coverage, threshold)."""
    if reps <= 0:
        return False, float("nan"), float("nan")
    c = covered / reps
    mcse = math.sqrt(c * (1 - c) / reps)
    threshold = nominal - 0.015 - 2 * mcse
    return c >= threshold - 1e-12, c, threshold


def finite_interval(
    row: dict[str, Any], keys: tuple[str, str, str] = ("estimate", "lo", "hi")
) -> bool:
    try:
        est, lo, hi = (float(row[k]) for k in keys)
    except (KeyError, TypeError, ValueError):
        return False
    return all(math.isfinite(x) for x in (est, lo, hi)) and lo <= est <= hi


def check_manifest_hashes(ctx: Ctx, manifest: dict[str, Any], label: str) -> None:
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        need(False, f"{label} manifest lists no files")
        return
    for path, digest in files.items():
        p = rel(ctx, path)
        need(p.exists(), f"{label}: {path} missing")
        need(sha256_file(p) == digest, f"{label}: hash mismatch for {path}")


def makefile_rule(ctx: Ctx, target: str) -> str:
    """Prerequisites and recipe text of a Makefile target."""
    text = rel(ctx, "Makefile").read_text()
    m = re.search(rf"^{re.escape(target)}:(.*?)(?=^\S|\Z)", text, re.M | re.S)
    need(m is not None, f"Makefile has no {target} target")
    assert m is not None
    return m.group(1)


def junit_counts(path: Path) -> tuple[int, int]:
    """(tests, failures + errors) from a pytest junit XML file."""
    need(path.exists(), "pytest produced no junit report (make test did not run pytest?)")
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    tests = sum(int(s.get("tests", 0)) for s in suites)
    bad = sum(int(s.get("failures", 0)) + int(s.get("errors", 0)) for s in suites)
    return tests, bad


R_SKIP_RE = re.compile(r"\bR\b|lme4|rpy2")


def junit_skips(path: Path) -> list[tuple[str, str]]:
    """(test id, skip message) of every skipped test in a pytest junit XML file."""
    root = ET.parse(path).getroot()
    out = []
    for case in root.iter("testcase"):
        for skip in case.findall("skipped"):
            name = f"{case.get('classname', '')}::{case.get('name', '')}"
            out.append((name, str(skip.get("message", ""))))
    return out


def tracked_py(ctx: Ctx, *roots: str) -> list[Path]:
    out = git(ctx.clone, "ls-files", "--", *roots)
    return [ctx.clone / p for p in out.splitlines() if p.endswith(".py")]


def brief_caps(ctx: Ctx) -> dict[str, Any]:
    """Default caps from the toml block in section 8 of the brief."""
    text = rel(ctx, "docs/BUILD_BRIEF.md").read_text()
    m = re.search(r"```toml\n(.*?)\n```", text, re.S)
    need(m is not None, "no toml block in docs/BUILD_BRIEF.md")
    assert m is not None
    return tomllib.loads(m.group(1))


# --------------------------------------------------------------------------- gates


def g1(ctx: Ctx) -> str:
    res = make(ctx, "setup", timeout=1800)
    ctx.setup_ok = res.code == 0
    need(ctx.setup_ok, f"make setup exit {res.code}: {tail(res.out)}")
    logs = ctx.tmp / "logs"
    env = dict(ctx.env)
    env["PYTEST_ADDOPTS"] = (
        f"--junitxml={logs / 'junit.xml'} --cov-report=json:{logs / 'coverage.json'} "
        f"--basetemp={ctx.tmp / 'pytest-g1'}"  # its own temp root: no other pytest prunes it
    )
    res = make(ctx, "test", timeout=3600, env=env)
    need(res.code == 0, f"make test exit {res.code}: {tail(res.out)}")
    tests, bad = junit_counts(logs / "junit.xml")
    need(tests > 0 and bad == 0, f"junit: {tests} tests, {bad} failed")
    # The R-backed tests carry the GLMM validity checks that preregistrations 0004 to 0006 rely
    # on (for example test_recovery.py::test_g6_analysis_equals_the_e1_analysis), so a run that
    # skipped them for want of R, lme4, or rpy2 does not pass (D-039).
    skipped = junit_skips(logs / "junit.xml")
    r_skips = [name for name, why in skipped if R_SKIP_RE.search(why)]
    need(not r_skips, f"{len(r_skips)} R-backed tests skipped, e.g. {', '.join(r_skips[:2])}")

    with (ctx.clone / "pyproject.toml").open("rb") as fh:
        cov_cfg = tomllib.load(fh).get("tool", {}).get("coverage", {}).get("run", {})
    sources = {s.rstrip("/") for s in cov_cfg.get("source", [])}
    need(
        {f"src/{p}" for p in PACKAGES} <= sources or set(PACKAGES) <= sources,
        f"coverage source {sorted(sources)} does not include both packages",
    )
    omit = list(cov_cfg.get("omit", []))
    for pattern in omit:
        stem = Path(pattern).stem
        contract = ctx.clone / "tests" / "contract" / f"test_{stem}.py"
        need(contract.exists(), f"omitted {pattern} has no tests/contract/test_{stem}.py")
        need("fixtures" in contract.read_text(), f"{contract.name} uses no recorded fixtures")

    cov = load_json(logs / "coverage.json")
    measured = {str(Path(f)) for f in cov.get("files", {})}
    for path in tracked_py(ctx, *(f"src/{p}" for p in PACKAGES)):
        relp = str(path.relative_to(ctx.clone))
        if any(Path(relp).match(o) or relp == o for o in omit):
            continue
        need(
            relp in measured or str(path) in measured,
            f"{relp} missing from coverage report",
        )
    totals = cov["totals"]
    statements = int(totals["num_statements"])
    covered = int(totals["covered_lines"])
    need(statements > 0, "coverage measured no statements")
    pct = 100.0 * covered / statements
    need(pct >= COVERAGE_MIN, f"line coverage {pct:.1f}% < {COVERAGE_MIN:.0f}%")
    return (
        f"setup ok; {tests} tests pass ({len(skipped)} skipped, none for want of R); line "
        f"coverage {pct:.1f}% (omitted {len(omit)})"
    )


def g2(ctx: Ctx) -> str:
    need_setup(ctx)
    with (ctx.clone / "pyproject.toml").open("rb") as fh:
        tool = tomllib.load(fh).get("tool", {})
    ruff = tool.get("ruff", {})
    for key in ("exclude", "extend-exclude"):
        for pat in ruff.get(key, []):
            need(not str(pat).lstrip("./").startswith("src"), f"ruff {key} hides {pat}")
    for override in tool.get("mypy", {}).get("overrides", []):
        mods = override.get("module", [])
        mods = [mods] if isinstance(mods, str) else mods
        for mod in mods:
            need(
                not any(str(mod).split(".")[0] == p for p in PACKAGES),
                f"mypy override weakens {mod}",
            )
    res = run(ctx, "ruff-check", ["uv", "run", "--no-sync", "ruff", "check", "."])
    need(res.code == 0, f"ruff check: {tail(res.out)}")
    res = run(ctx, "ruff-format", ["uv", "run", "--no-sync", "ruff", "format", "--check", "."])
    need(res.code == 0, f"ruff format --check: {tail(res.out)}")
    res = run(
        ctx,
        "mypy",
        ["uv", "run", "--no-sync", "mypy", "--strict", *(f"src/{p}" for p in PACKAGES)],
    )
    need(res.code == 0, f"mypy --strict: {tail(res.out)}")
    ignores = 0
    reason = re.compile(r"type:\s*ignore(\[[^\]]*\])?\s*#\s*\S.{2,}")
    for path in tracked_py(ctx, "src", "tests", "tools"):
        relp = path.relative_to(ctx.clone)
        for lineno, comment, own_line in comments(path):
            where = f"{relp}:{lineno}"
            need(
                re.search(r"\bmypy:\s*\S", comment) is None,
                f"{where} sets mypy options inline (it would relax --strict for the file)",
            )
            if re.search(r"type:\s*ignore", comment) is None:
                continue
            ignores += 1
            need(not (own_line and relp.parts[0] == "src"), f"{where} file-level type: ignore")
            need(reason.search(comment) is not None, f"{where} type: ignore without a reason")
    return f"ruff check, ruff format, mypy --strict clean; {ignores} type: ignore, all with reasons"


def comments(path: Path) -> list[tuple[int, str, bool]]:
    """(line number, comment text, comment is alone on its line) for each comment token."""
    with tokenize.open(path) as fh:
        tokens = list(tokenize.generate_tokens(fh.readline))
    return [
        (tok.start[0], tok.string, tok.line.strip().startswith("#"))
        for tok in tokens
        if tok.type == tokenize.COMMENT
    ]


def g3(ctx: Ctx) -> str:
    ci = ctx.clone / ".github" / "workflows" / "ci.yml"
    need(ci.exists(), "missing .github/workflows/ci.yml")
    text = ci.read_text()
    need(re.search(r"run:\s*make\s+ci\b", text) is not None, "ci.yml does not run `make ci`")
    for ver in ("3.11", "3.12"):
        need(re.search(rf"['\"]?{re.escape(ver)}['\"]?", text) is not None, f"ci.yml lacks {ver}")
    need("matrix" in text, "ci.yml has no Python version matrix")
    rule = makefile_rule(ctx, "ci")
    for target in ("lint", "typecheck", "test", "demo"):
        need(re.search(rf"\b{target}\b", rule) is not None, f"make ci does not run {target}")
    if ctx.fast:
        keys = ("make-test", "ruff-check", "ruff-format", "mypy", "make-demo")
        parts = [ctx.results.get(k) for k in keys]
        need(all(r is not None and r.code == 0 for r in parts), "fast: a ci component failed")
        return "ci.yml runs make ci on 3.11, 3.12; make ci not run (fast: components pass)"
    need_setup(ctx)
    env = {**ctx.env, "PYTEST_ADDOPTS": f"--basetemp={ctx.tmp / 'pytest-g3'}"}
    res = make(ctx, "ci", timeout=3600, env=env)
    need(res.code == 0, f"make ci exit {res.code}: {tail(res.out)}")
    return f"ci.yml runs make ci on 3.11, 3.12; make ci passes locally ({res.seconds:.0f}s)"


def g4(ctx: Ctx) -> str:
    need_setup(ctx)
    golden = ctx.clone / "tests" / "golden" / "demo.txt"
    need(golden.exists(), "missing tests/golden/demo.txt")
    ledger = ctx.clone / "artifacts" / "usage" / "ledger.jsonl"
    before = ledger.read_bytes() if ledger.exists() else b""
    env = dict(ctx.env)
    dead = "http://127.0.0.1:9"
    env.update(
        {
            "HTTP_PROXY": dead,
            "HTTPS_PROXY": dead,
            "ALL_PROXY": dead,
            "http_proxy": dead,
            "https_proxy": dead,
            "NO_PROXY": "",
            "UV_OFFLINE": "1",
            "UV_NO_SYNC": "1",
            "NOISEFLOOR_OFFLINE": "1",
            "NOISEFLOOR_NO_MODELS": "1",
        }
    )
    res = make(ctx, "demo", timeout=DEMO_LIMIT_S + 60, env=env, prefix=ctx.netns)
    isolation = "netns" if ctx.netns else "env-only"
    need(res.code == 0, f"make demo exit {res.code} ({isolation}): {tail(res.out)}")
    need(res.seconds < DEMO_LIMIT_S, f"make demo took {res.seconds:.0f}s >= 180s")
    after = ledger.read_bytes() if ledger.exists() else b""
    need(before == after, "make demo wrote to the usage ledger (model call)")
    stdout = (ctx.tmp / "logs" / "make-demo.stdout").read_text()
    need(stdout == golden.read_text(), "demo output differs from tests/golden/demo.txt")
    return f"demo {res.seconds:.0f}s, no network ({isolation}), no model calls, matches golden"


def e0_glmm_seed(group: str, replicate: int) -> int:
    """``noisefloor.recovery.e0_glmm_seed`` restated with the standard library (a test pins it)."""
    return int(sha256(f"e0-glmm:{group}:{replicate}".encode()).hexdigest()[:12], 16)


def raw_covered(
    ctx: Ctx,
    cell: str,
    reps: int,
    *,
    group: str | None = None,
    env: str | None = None,
    code: str | None = None,
    frozen: Any = None,
    truth: float | None = None,
) -> int:
    """Recount a cell's covered replicates from artifacts/e0/raw/<cell>.jsonl (indices < reps).

    The line counted for each index is the last one written for it (prereg 0005). For a grouped
    cell each counted line must carry its dataset's seed, ``e0_glmm_seed(group, i)``; with
    ``env`` and ``code`` (an artifact banked with stamps) each must name that environment and
    analysis code from a clean ``src/`` (D-039); with ``truth``, the planted value the cell's
    recovery-config fixes (:func:`grouped_truths`), each must carry it as its ``truth`` (N22).
    """
    path = rel(ctx, f"artifacts/e0/raw/{cell}.jsonl")
    need(path.exists(), f"cell {cell}: raw replicates missing")
    rows: dict[int, dict[str, Any]] = {}
    lines: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[int(row["i"])] = row
            lines.append(row)
    need(all(i in rows for i in range(reps)), f"cell {cell}: raw replicates incomplete")
    # Every line of a counted index is checked, as the bank checks every line it reads.
    for row in lines:
        i = int(row["i"])
        if i >= reps:
            continue
        if group is not None:
            need(
                row.get("seed") is not None and int(row["seed"]) == e0_glmm_seed(group, i),
                f"cell {cell}: replicate {i} does not carry the seed of its group's dataset",
            )
        if truth is not None:
            value = row.get("truth")
            need(
                value is not None and math.isclose(float(value), truth, rel_tol=0.0, abs_tol=1e-12),
                f"cell {cell}: replicate {i} has truth {value}, not the planted value "
                f"{truth:.12g} its recovery-config fixes",
            )
        if env is not None or code is not None:
            stamp = row.get("stamp") or {}
            need(
                stamp.get("env") == env
                and stamp.get("analysis") == code
                and stamp.get("src_dirty") is False,
                f"cell {cell}: replicate {i} ran in another environment or analysis code, or "
                "from a dirty src/",
            )
            if frozen is not None:
                need(
                    utc_time(str(stamp.get("at", ""))) >= frozen,
                    f"cell {cell}: replicate {i} was run before its preregistration was frozen",
                )

    def covered(row: dict[str, Any]) -> bool:
        if row.get("hit") is not None:
            return bool(row["hit"])
        lo, hi, planted = row.get("lo"), row.get("hi"), row.get("truth")
        if lo is None or hi is None or planted is None:
            return False
        return float(lo) <= float(planted) <= float(hi)

    return sum(covered(rows[i]) for i in range(reps))


PART1_HEADLINE_RE = re.compile(r"Headline cells \([^)]*\):(.*?)\.\s+Every headline cell", re.S)


def chain_files(ctx: Ctx) -> list[str]:
    """Preregistration files of prereg/CHAIN.jsonl, in freeze order."""
    path = rel(ctx, "prereg/CHAIN.jsonl")
    need(path.exists(), "missing prereg/CHAIN.jsonl")
    return [str(json.loads(ln)["file"]) for ln in path.read_text().splitlines() if ln.strip()]


def has_block(ctx: Ctx, prereg: str, tag: str) -> bool:
    """True if the preregistration has a fenced ```json <tag>``` block."""
    return any(m.group(1) == tag for m in FENCE_RE.finditer(rel(ctx, prereg).read_text()))


def amends(ctx: Ctx, prereg: str) -> str | None:
    """The NNNN of an `amends: NNNN` front-matter line, if any."""
    text = rel(ctx, prereg).read_text()
    if not text.startswith("---"):
        return None
    front = text.split("---", 2)[1]
    found = re.search(r"^amends:\s*(\d{4})\s*$", front, re.M)
    return found.group(1) if found else None


def e0_grids(ctx: Ctx) -> list[tuple[str, str]]:
    """(prereg, coverage artifact) for every frozen E0 grid: the base grid first, then each
    amendment of it, each banked in its own artifact (D-026)."""
    preregs = [f for f in chain_files(ctx) if has_block(ctx, f, "e0-grid")]
    bases = [f for f in preregs if amends(ctx, f) is None]
    need(len(bases) == 1, f"expected one base E0 grid in the chain, found {len(bases)}")
    base = bases[0]
    number = Path(base).name[:4]
    out = [(base, "artifacts/e0/coverage.json")]
    for f in preregs:
        if f != base:
            need(amends(ctx, f) == number, f"{f} has an e0-grid but does not amend {base}")
            out.append((f, f"artifacts/e0/coverage-{Path(f).name[:4]}.json"))
    return out


def chain_frozen_at(ctx: Ctx, prereg: str) -> Any:
    """When ``prereg`` was frozen (its prereg/CHAIN.jsonl entry)."""
    for ln in rel(ctx, "prereg/CHAIN.jsonl").read_text().splitlines():
        if ln.strip() and json.loads(ln).get("file") == prereg:
            return utc_time(str(json.loads(ln)["frozen_at"]))
    raise GateError(f"{prereg} is not in prereg/CHAIN.jsonl")


def first_e1_run(ctx: Ctx) -> Any:
    """Start of the earliest confirmatory E1 run, or None before any E1 data exists."""
    path = rel(ctx, "artifacts/e1/runs.jsonl")
    if not path.exists():
        return None
    rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    starts = [utc_time(str(r["started_at"])) for r in rows if r.get("confirmatory")]
    return min(starts) if starts else None


E1_NUMBER = "0004"  # the E1 preregistration, whose analysis names each kind's method
GLMM_KINDS = ("random", "fixed", "share", "mde")


def glmm_rule(kind: str, settings: dict[str, Any]) -> tuple[Any, ...]:
    """The recovery-config setting that is a GLMM kind's method.

    Its cells' interval_type label does not change with it, so a replacement of the kind
    must change this. Under the task jackknife the bootstrap box rule (``box_interval``)
    has no effect, so a fixed contribution's rule is the jackknife, or the box with its
    ``box_interval``.
    """
    if kind == "fixed":
        rule = settings.get("fixed_interval", "box")
        return ("jackknife",) if rule == "jackknife" else ("box", settings.get("box_interval"))
    key = {"random": "boot_interval", "share": "share_interval", "mde": "mde_interval"}[kind]
    return (settings.get(key),)


def _settings(ctx: Ctx, prereg: str) -> dict[str, Any]:
    if not has_block(ctx, prereg, "recovery-config"):
        return {}
    return dict(prereg_block(ctx, prereg, "recovery-config").get("settings") or {})


def method_replaced(
    ctx: Ctx, prereg: str, kind: str, gone: set[str], new: set[str], grid: dict[str, Any]
) -> bool:
    """True when ``prereg``'s new cells of ``kind`` test another method than ``gone`` did.

    ``grid`` maps every frozen cell to (its preregistration, its cell entry). A GLMM kind
    must change its rule (:func:`glmm_rule`) in ``prereg``'s own recovery-config block;
    another kind's new cells must name another (method, interval_type) than every retired
    cell.
    """
    if kind in GLMM_KINDS:
        mine = _settings(ctx, prereg)
        if not mine or not new:
            return False
        sources = {grid[c][0] for c in gone if c in grid}
        return all(glmm_rule(kind, mine) != glmm_rule(kind, _settings(ctx, src)) for src in sources)

    def label(cid: str) -> tuple[str, str]:
        entry = grid[cid][1]
        return str(entry.get("method")), str(entry.get("interval_type"))

    old = {label(c) for c in gone if c in grid}
    return bool(new) and not ({label(c) for c in new if c in grid} & old)


def frozen_headline_cells(ctx: Ctx, grids: list[tuple[str, str]]) -> set[str]:
    """Headline cells named by the frozen preregistrations: the base grid's list and every
    amendment's ```json headline``` block, less the cells of a replaced method.

    An amendment that replaces a kind's method lists the replaced cells in a ```json
    retired-headline``` block (prereg 0005); they then back no headline interval, so G5
    needs only their listing and warning. The block counts only from an amendment frozen
    before the first confirmatory E1 run, beside new headline cells of the kind that test
    another method (:func:`method_replaced`) and a frozen amendment of the E1
    preregistration from before that run whose ```json analysis-code``` block allows the
    new analysis code (an amendment of 0004 for another purpose, such as a rebuilt
    llama-server, does not count).
    """
    base = grids[0][0]
    found = PART1_HEADLINE_RE.search(rel(ctx, base).read_text())
    need(found is not None, f"{base} names no headline cells")
    cells = set(re.findall(r"`([A-Za-z0-9_-]+)`", found.group(1)))
    retired: set[str] = set()
    kinds: dict[str, str] = {}
    first = first_e1_run(ctx)
    grid: dict[str, Any] = {}
    for prereg, _ in grids:
        for entry in prereg_block(ctx, prereg, "e0-grid"):
            grid[str(entry["cell"])] = (prereg, entry)
    for prereg, _ in grids[1:]:
        own_kinds: dict[str, str] = {}
        if has_block(ctx, prereg, "headline-kinds"):
            own_kinds = {
                str(k): str(v) for k, v in prereg_block(ctx, prereg, "headline-kinds").items()
            }
        own_cells: set[str] = set()
        if has_block(ctx, prereg, "headline"):
            block = prereg_block(ctx, prereg, "headline")
            need(isinstance(block, dict), f"{prereg} headline block is not an object")
            own_cells = {str(v) for v in block.values()}
            cells |= own_cells
        if has_block(ctx, prereg, "retired-headline"):
            gone = prereg_block(ctx, prereg, "retired-headline")
            need(isinstance(gone, list), f"{prereg} retired-headline block is not a list")
            frozen = chain_frozen_at(ctx, prereg)
            need(
                first is None or frozen < first,
                f"{prereg} retires headline cells after E1 data exist",
            )
            # A replaced method's cells retire only beside new headline cells of the same
            # kind, which G5 then requires to pass (prereg 0005).
            missing = sorted({kinds.get(str(c), "?") for c in gone} - set(own_kinds.values()))
            need(not missing, f"{prereg} retires {', '.join(missing)} cells with no replacement")
            for kind in sorted({kinds[str(c)] for c in gone}):
                old = {str(c) for c in gone if kinds.get(str(c)) == kind}
                new = {c for c in own_cells if own_kinds.get(c) == kind}
                need(
                    method_replaced(ctx, prereg, kind, old, new, grid),
                    f"{prereg} retires {kind} cells but its new cells test the same method",
                )
            e1_amendments = [
                f
                for f in chain_files(ctx)
                if amends(ctx, f) == E1_NUMBER
                and has_block(ctx, f, "analysis-code")
                and (first is None or chain_frozen_at(ctx, f) < first)
            ]
            need(
                bool(e1_amendments),
                f"{prereg} retires headline cells, but no amendment of prereg {E1_NUMBER} "
                "frozen before E1 data allows the new method",
            )
            retired |= {str(c) for c in gone}
        kinds.update(own_kinds)
    return cells - retired


def grouped_truths(ctx: Ctx, prereg: str, cells: Iterable[dict[str, Any]]) -> dict[str, float]:
    """The grouped cells of ``prereg`` whose recovery-config fixes their planted value, with it.

    A group's datasets use the recovery-config's ``planted`` object updated with the cell's
    ``params.planted`` overrides, whole top-level keys replaced, as
    ``noisefloor.recovery.run_e0_glmm`` builds it. A cell gets the value that
    :func:`g6_planted_truths` gives its component on that object; a component whose value depends
    on a dataset's own draws (day, block, most shares, MDE and local-arm rows) gets none (N22).
    """
    grouped = [c for c in cells if (c.get("params") or {}).get("group") is not None]
    if not grouped:
        return {}
    config = prereg_block(ctx, prereg, "recovery-config")
    need(isinstance(config, dict), f"{prereg} recovery-config block is not an object")
    planted = config.get("planted") or {}
    out: dict[str, float] = {}
    for c in grouped:
        params = c["params"]
        truths = g6_planted_truths({"planted": {**planted, **(params.get("planted") or {})}})
        component = str(params.get("component"))
        if component in truths:
            out[str(c["cell"])] = truths[component]
    return out


def g5(ctx: Ctx) -> str:
    grids = e0_grids(ctx)
    planned: dict[str, Any] = {}
    cells: dict[str, Any] = {}
    stamped: dict[str, tuple[Any, Any, Any]] = {}  # cell -> (env, analysis code, plan frozen at)
    truths: dict[str, float] = {}  # grouped cell -> the planted value its config fixes (N22)
    for prereg, art in grids:
        data = load_json(rel(ctx, art))
        need(data.get("prereg") == prereg, f"{art} cites {data.get('prereg')}, not {prereg}")
        check_prereg_link(ctx, prereg, art)
        block = prereg_block(ctx, prereg, "e0-grid")
        need(isinstance(block, list) and bool(block), f"e0-grid in {prereg} is empty")
        mine = {str(c["cell"]): c for c in block}
        need(len(mine) == len(block), f"cell ids repeat in {prereg}")
        need(not set(mine) & set(planned), "cell ids repeat across the frozen grids")
        got = {str(c["cell"]): c for c in data.get("cells", [])}
        need(set(got) == set(mine), f"{art} cells differ from the frozen grid of {prereg}")
        planned.update(mine)
        cells.update(got)
        truths.update(grouped_truths(ctx, prereg, block))
        banked_env = (data.get("environment") or {}).get("env")
        if amends(ctx, prereg) is not None:
            # An amendment's cells ran with stamps on every line (prereg 0005), so its artifact
            # names one environment and one analysis code, which every counted line must match.
            for value, what in (
                (banked_env, "environment.env"),
                (data.get("analysis_code"), "analysis_code"),
            ):
                need(
                    isinstance(value, str) and re.fullmatch(r"[0-9a-f]{16}", value) is not None,
                    f"{art} has no 16-hex {what}",
                )
        if banked_env is not None or data.get("analysis_code") is not None:
            when = chain_frozen_at(ctx, prereg)  # stamped lines postdate their plan (D-041)
            stamped.update(dict.fromkeys(got, (banked_env, data.get("analysis_code"), when)))
    failing: list[str] = []
    for cid, cell in cells.items():
        plan = planned[cid]
        for key in ("method", "interval_type", "nominal"):
            need(cell.get(key) == plan.get(key), f"cell {cid} {key} differs from prereg")
        reps = int(cell["replicates"])
        floor = MIN_REPS_RESAMPLING if cell["interval_type"] in RESAMPLING_TYPES else MIN_REPS
        need(reps >= max(floor, int(plan.get("replicates", 0))), f"cell {cid}: {reps} replicates")
        env, code, frozen = stamped.get(cid, (None, None, None))
        group = (plan.get("params") or {}).get("group")
        covered = raw_covered(
            ctx,
            cid,
            reps,
            group=None if group is None else str(group),
            env=env,
            code=code,
            frozen=frozen,
            truth=truths.get(cid),
        )
        need(
            covered == int(cell["covered"]),
            f"cell {cid}: banked covered count does not match its raw replicates",
        )
        ok, _, _ = coverage_rule(covered, reps, float(cell["nominal"]))
        if not ok:
            failing.append(cid)
    headline = frozen_headline_cells(ctx, grids)
    head = load_json(rel(ctx, "artifacts/headline_cells.json"))
    declared = {str(c) for c in head.get("cells", [])}
    need(bool(headline), "no headline cells declared")
    need(headline <= declared, "artifacts/headline_cells.json lacks a frozen headline cell")
    need(headline <= set(cells), "a headline cell is not in the E0 grids")
    bad_head = sorted(headline & set(failing))
    need(not bad_head, f"headline cells fail coverage: {', '.join(bad_head[:3])}")
    need_setup(ctx)
    res = run(
        ctx,
        "unsupported",
        [
            "uv",
            "run",
            "--no-sync",
            "python",
            "-c",
            "import json, evalstats.validity as v; print(json.dumps(sorted(v.UNSUPPORTED_CELLS)))",
        ],
    )
    need(res.code == 0, f"evalstats.validity.UNSUPPORTED_CELLS: {tail(res.out)}")
    unsupported = set(json.loads(res.stdout.strip().splitlines()[-1]))
    missing = sorted(set(failing) - unsupported)
    need(not missing, f"failing cells not marked unsupported: {', '.join(missing[:3])}")
    if failing:  # each failing cell must also make its own function warn (the library rule)
        probes = [
            [
                FUNCTION_OF.get(str(cells[c]["method"]), "?"),
                cells[c]["interval_type"],
                c,
                int(cells[c]["params"]["n_tasks"]),
            ]
            for c in failing
        ]
        code = (
            "import json, sys, warnings, evalstats.validity as v\n"
            "warnings.simplefilter('ignore')\n"
            "bad = [c for f, m, c, n in json.loads(sys.argv[1]) if c not in v.check(f, m, n)]\n"
            "print(json.dumps(bad))"
        )
        res = run(
            ctx,
            "warnings",
            ["uv", "run", "--no-sync", "python", "-c", code, json.dumps(probes)],
        )
        need(res.code == 0, f"validity warnings: {tail(res.out)}")
        silent = json.loads(res.stdout.strip().splitlines()[-1])
        need(not silent, f"failing cells that trigger no warning: {', '.join(silent[:3])}")
    cross = load_json(rel(ctx, "artifacts/e0/crosscheck.json")).get("checks", [])
    need(bool(cross), "no cross-checks banked")
    need(
        any("statsmodels" in str(c.get("reference_impl", "")) for c in cross),
        "no statsmodels cross-check",
    )
    for c in cross:
        tol = float(c["tolerance"])
        need(tol > 0, f"cross-check {c.get('name')} states no tolerance")
        diff = abs(float(c["value"]) - float(c["reference"]))
        need(diff <= tol, f"cross-check {c.get('name')} off by {diff:.3g} > {tol:.3g}")
    logs = ctx.tmp / "logs"
    res = run(
        ctx,
        "property",
        [
            "uv",
            "run",
            "--no-sync",
            "pytest",
            "-m",
            "property",
            "-q",
            "-p",
            "no:cacheprovider",
            f"--junitxml={logs / 'property.xml'}",
            f"--basetemp={ctx.tmp / 'pytest-property'}",
        ],
    )
    tests, bad = junit_counts(logs / "property.xml")
    need(res.code == 0 and tests > 0 and bad == 0, f"property suite: {tests} tests, {bad} failed")
    return (
        f"{len(cells)} cells, {len(failing)} failing (all unsupported), headline "
        f"{len(headline)} pass; {len(cross)} cross-checks; {tests} property tests"
    )


def g6_planted_truths(config: dict[str, Any]) -> dict[str, float]:
    """Planted values a recovery-config fixes for every dataset (prereg 0006; G5 grouped cells).

    The task variance plus half of each planted task-by-factor variance; the model-by-task
    variance as given; the model variance as the population variance of the planted model
    effects; each two-level fixed factor's contribution as effect^2 / 4 (a balanced design), with
    a share of 0 for a factor planted at 0. Components whose planted value depends on a dataset's
    own draws (day, block, per-model and MDE rows) are not listed.
    """
    planted = config.get("planted") or {}
    out: dict[str, float] = {}
    if "task_var" in planted:
        # The task truth of noisefloor.recovery.planted_values (src/noisefloor/recovery.py line
        # 388): the analysis model's task term carries half of each task-by-factor variance.
        inter = planted.get("interactions") or {}
        out["task"] = float(planted["task_var"]) + sum(float(v) for v in inter.values()) / 2
    if "model_task_var" in planted:
        out["model:task"] = float(planted["model_task_var"])
    model = [float(x) for x in planted.get("model", [])]
    if model:
        mean = sum(model) / len(model)
        out["model"] = sum((x - mean) ** 2 for x in model) / len(model)
    for factor, effect in (planted.get("factors") or {}).items():
        out[str(factor)] = float(effect) ** 2 / 4
        if float(effect) == 0:
            out[f"share:{factor}"] = 0.0
    return out


def g6_hit(row: dict[str, Any]) -> bool:
    """True if a G6 component's interval covers its planted value (a missing bound never does)."""
    if row.get("lo") is None or row.get("hi") is None:
        return False
    return float(row["lo"]) <= float(row["planted"]) <= float(row["hi"])


def g6(ctx: Ctx) -> str:
    art = "artifacts/g6/recovery.json"
    data = load_json(rel(ctx, art))
    prereg = str(data.get("prereg", ""))
    seeds = prereg_block(ctx, prereg, "g6-seeds")
    config = prereg_block(ctx, prereg, "recovery-config")
    listed = prereg_block(ctx, prereg, "g6-components")
    check_prereg_link(ctx, prereg, art)
    datasets = data.get("datasets", [])
    need(len(datasets) >= G6_MIN_DATASETS, f"{len(datasets)} datasets < {G6_MIN_DATASETS}")
    need(
        sorted(int(d["seed"]) for d in datasets) == sorted(int(s) for s in seeds),
        "dataset seeds differ from prereg",
    )
    need(data.get("config") == config, "artifact config differs from the prereg's")
    nominal = float(data["nominal"])
    need(nominal == float(config["settings"]["level"]), "nominal level differs from the prereg's")
    need(bool((data.get("environment") or {}).get("env")), "rows carry no single environment")
    # The one analysis-code hash the E1 analysis requires (prereg 0004): the one E0 part 2
    # ran once it is banked, or, for a rerun of G6 under an amendment of its preregistration
    # (a replaced GLMM method), one that a frozen amendment of 0004 allows.
    code = data.get("analysis_code")
    need(
        isinstance(code, str) and re.fullmatch(r"[0-9a-f]{16}", code) is not None,
        "no 16-hex analysis_code",
    )
    if amends(ctx, prereg) is None:
        e0 = rel(ctx, E0_PART2_ARTIFACT)
        if e0.exists():
            other = load_json(e0).get("analysis_code")
            need(other == code, f"G6 ran analysis code {code}, E0 part 2 {other}")
    else:
        allowed = {
            str(prereg_block(ctx, f, "analysis-code").get("allows"))
            for f in chain_files(ctx)
            if amends(ctx, f) == E1_NUMBER and has_block(ctx, f, "analysis-code")
        }
        need(code in allowed, f"no frozen amendment of {E1_NUMBER} allows G6's code {code}")
    names = sorted({n for d in datasets for n in d["components"]})
    need(bool(names), "no variance components")
    # Every preregistered component, and only those, decides the gate (prereg 0006).
    need(names == sorted(listed), "components differ from the prereg's g6-components list")
    # The artifact is what the committed raw rows say, dataset by dataset, and those rows ran in
    # the artifact's one environment and analysis code from a clean src/ (D-039).
    raw = rel(ctx, "artifacts/g6/raw.jsonl")
    need(raw.exists(), "missing artifacts/g6/raw.jsonl")
    frozen = chain_frozen_at(ctx, prereg)  # every raw row must postdate its plan (D-041)
    by_seed: dict[int, dict[str, Any]] = {}
    for line in raw.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            need(int(row["seed"]) not in by_seed, f"seed {row['seed']} has two raw rows")
            by_seed[int(row["seed"])] = row
    need(set(by_seed) == {int(s) for s in seeds}, "raw rows are not exactly the prereg's seeds")
    env = (data.get("environment") or {}).get("env")
    for d in datasets:
        row = by_seed[int(d["seed"])]
        same = row.get("components") == d["components"] and all(
            row.get(k) == d.get(k) for k in ("converged", "error")
        )
        need(same, f"dataset {d['seed']} differs from its raw row")
        stamp = row.get("stamp") or {}
        need(
            stamp.get("env") == env
            and stamp.get("analysis") == code
            and stamp.get("src_dirty") is False,
            f"raw row {d['seed']} ran in another environment or analysis code, or a dirty src/",
        )
        need(
            utc_time(str(stamp.get("at", ""))) >= frozen,
            f"raw row {d['seed']} was run before {prereg} was frozen",
        )
    # Planted values that the recovery-config fixes are the ones every dataset was scored on.
    for name, value in g6_planted_truths(config).items():
        if name in names:
            need(
                all(
                    math.isclose(
                        float(d["components"][name]["planted"]), value, rel_tol=0.0, abs_tol=1e-12
                    )
                    for d in datasets
                ),
                f"{name} planted value differs from the recovery-config's {value:g}",
            )
    reported_bias = data.get("bias", {})
    parts = []
    for name in names:
        rows = [d["components"].get(name) for d in datasets]
        need(all(r is not None for r in rows), f"component {name} missing in some datasets")
        covered = sum(1 for r in rows if g6_hit(r))
        ok, cov, thr = coverage_rule(covered, len(rows), nominal)
        need(ok, f"{name} coverage {cov:.3f} < {thr:.3f}")
        errors = [
            float(r["estimate"]) - float(r["planted"]) for r in rows if r["estimate"] is not None
        ]
        need(bool(errors), f"no estimates for {name}")
        bias = sum(errors) / len(errors)
        need(name in reported_bias, f"bias for {name} not reported")
        need(
            math.isclose(float(reported_bias[name]), bias, rel_tol=1e-9, abs_tol=1e-12),
            f"reported bias for {name} does not match recomputation",
        )
        parts.append(f"{name} {cov:.3f}")
    failed = sum(1 for d in datasets if d.get("error"))
    head = f"{len(datasets)} datasets ({failed} failed fits); coverage "
    return head + ", ".join(parts)[:90]


def g7(ctx: Ctx) -> str:
    manifest = load_json(rel(ctx, "artifacts/e1/manifest.json"))
    prereg = str(manifest.get("prereg", ""))
    design = prereg_block(ctx, prereg, "e1-design")
    check_prereg_link(ctx, prereg, "artifacts/e1/manifest.json")
    check_manifest_hashes(ctx, manifest, "E1")
    runs_path = rel(ctx, "artifacts/e1/runs.jsonl")
    need(runs_path.exists(), "missing artifacts/e1/runs.jsonl")
    check_prereg_link(ctx, prereg, "artifacts/e1/runs.jsonl")
    factors: dict[str, list[Any]] = design["factors"]
    models = {str(m["id"]): m for m in design["models"]}
    need(len(factors) >= 4, f"design has {len(factors)} non-model factors < 4")
    rows = [json.loads(ln) for ln in runs_path.read_text().splitlines() if ln.strip()]
    seen: dict[str, int] = {}
    for r in rows:
        seen[str(r.get("run_id"))] = seen.get(str(r.get("run_id")), 0) + 1
    dups = sorted(k for k, n in seen.items() if n > 1)
    need(not dups, f"{len(dups)} run ids banked more than once (first {dups[:1]})")
    conf = [r for r in rows if r.get("confirmatory") is True]
    entry = next(
        (
            json.loads(ln)
            for ln in rel(ctx, "prereg/CHAIN.jsonl").read_text().splitlines()
            if ln.strip() and json.loads(ln).get("file") == prereg
        ),
        None,
    )
    need(entry is not None, f"{prereg} is not in prereg/CHAIN.jsonl")
    frozen_at = utc_time(str(entry["frozen_at"]))
    for r in conf:
        rid = r.get("run_id")
        need(r.get("prereg") == prereg, f"run {rid} cites another prereg")
        need(r.get("prereg_sha256") == entry["sha256"], f"run {rid} ran under another prereg hash")
        need(bool(r.get("started_at")), f"run {rid} has no start time")
        need(utc_time(str(r["started_at"])) >= frozen_at, f"run {rid} started before the freeze")
        need(str(r.get("model_id")) in models, f"run {rid} model not in design")
        for name, level in r.get("factors", {}).items():
            need(
                name in factors and level in factors[name],
                f"run {rid} factor {name}={level} not in design",
            )
        need(r.get("outcome") in (0, 1, True, False), f"run {rid} outcome not binary")
    # A task run counts once it ran and the primary analyses use it: runs with status
    # infra_error, and Claude runs left out (another model ID or step budget), do not count
    # toward the floors (prereg 0004); a planned run the runner never started is never banked.
    ran = [r for r in conf if r.get("status") != "infra_error" and not r.get("excluded")]
    need(len(ran) >= E1_MIN_RUNS, f"{len(ran)} confirmatory task runs < {E1_MIN_RUNS}")
    claude = sum(1 for r in ran if models[str(r["model_id"])].get("backend") == "claude")
    need(claude >= E1_MIN_CLAUDE_RUNS, f"{claude} Claude-tier runs < {E1_MIN_CLAUDE_RUNS}")
    used = {str(r["model_id"]) for r in ran}
    families = {models[m].get("family") for m in used}
    need(len(used) >= 3 and len(families) >= 2, f"{len(used)} models, {len(families)} families")
    local = [r for r in ran if models[str(r["model_id"])].get("backend") == "local"]
    days = {r.get("day") for r in local if r.get("day")}
    need(len(days) >= 3, f"local arm on {len(days)} days < 3")
    vc = load_json(rel(ctx, "artifacts/e1/variance_components.json")).get("components", [])
    need(bool(vc) and all(finite_interval(c) for c in vc), "variance table lacks intervals")
    need(
        all(finite_interval(c, ("share", "share_lo", "share_hi")) for c in vc),
        "variance table lacks share intervals",
    )
    mde = load_json(rel(ctx, "artifacts/e1/mde.json")).get("rows", [])
    need(
        bool(mde) and all(finite_interval(r, ("mde", "lo", "hi")) for r in mde),
        "MDE table lacks intervals",
    )
    vc_code = (
        load_json(rel(ctx, "artifacts/e1/variance_components.json")).get("analysis_code") or {}
    ).get("hash")
    post_check_ran(ctx, prereg, str(entry["sha256"]), vc_code)
    note = ""
    if ctx.fast:
        note = "; recompute skipped (fast)"
    else:
        need_setup(ctx)
        res = run(
            ctx,
            "e1-verify",
            ["uv", "run", "--no-sync", "python", "-m", "noisefloor.analysis.e1", "--verify"],
            timeout=E1_VERIFY_TIMEOUT,
        )
        need(res.code == 0, f"E1 recompute: {tail(res.out)}")
        note = "; tables recomputed"
    return f"{len(ran)} runs ({claude} Claude), {len(used)} models, {len(days)} days{note}"


def post_check_ran(ctx: Ctx, prereg: str, sha256: str, code: Any) -> None:
    """The post-E1 check of prereg 0004 item 10 ran, under the E1 tables' analysis code.

    It alone applies item 10's labels and item 9's batch rule, so G7 needs its artifact:
    synthetic, citing the E1 preregistration and its chain hash, every component on all
    :data:`POST_E1_DATASETS` datasets, and its list of intervals not validated.
    """
    art = "artifacts/e1/post_check.json"
    post = load_json(rel(ctx, art))
    need(post.get("prereg") == prereg, f"{art} cites {post.get('prereg')}, not {prereg}")
    need(post.get("prereg_sha256") == sha256, f"{art} ran under another prereg hash")
    check_prereg_link(ctx, prereg, art)
    need(post.get("synthetic") is True, f"{art} is not labeled synthetic")
    comps = post.get("components") or {}
    need(
        bool(comps) and all(int(c.get("datasets", 0)) == POST_E1_DATASETS for c in comps.values()),
        f"{art} lacks {POST_E1_DATASETS} datasets per component",
    )
    need(isinstance(post.get("not_validated"), list), f"{art} has no not_validated list")
    need(bool(code) and post.get("analysis_code") == code, f"{art} ran other analysis code")


def g8(ctx: Ctx) -> str:
    manifests = sorted(rel(ctx, "artifacts/e2").glob("*/manifest.json"))
    need(len(manifests) >= E2_MIN_BOARDS, f"{len(manifests)} leaderboards < {E2_MIN_BOARDS}")
    boards = []
    for mpath in manifests:
        m = load_json(mpath)
        board = str(m.get("board", mpath.parent.name))
        for key in ("source_url", "retrieved", "license", "snapshot", "sha256"):
            need(bool(m.get(key)), f"{board} manifest lacks {key}")
        snap = str(m["snapshot"])
        if not snap.startswith("private/"):
            p = rel(ctx, snap)
            need(p.exists(), f"{board} snapshot {snap} missing")
            need(sha256_file(p) == m["sha256"], f"{board} snapshot hash mismatch")
        if m.get("prereg"):
            check_prereg_link(ctx, str(m["prereg"]), str(mpath.relative_to(ctx.clone)))
        need(rel(ctx, f"docs/leaderboards/{board}.md").exists(), f"no docs/leaderboards/{board}.md")
        boards.append(board)
    bounds = load_json(rel(ctx, "artifacts/e3/bounds.json"))
    sources = {str(e.get("source")) for e in bounds.get("entries", [])}
    missing = [b for b in boards if b not in sources]
    need(not missing, f"E3 bounds missing for {', '.join(missing)}")
    # Each public board's E3 entry is the policy question (q3) of that board's results.json,
    # which the full mode recomputes from the snapshot, so no E3 number stands on its own (D-040).
    for entry in bounds.get("entries", []):
        board = str(entry.get("source"))
        results = load_json(rel(ctx, f"artifacts/e2/{board}/results.json"))
        aset = str(entry.get("analysis_set"))
        q3 = ((results.get("sets") or {}).get(aset) or {}).get("q3")
        need(isinstance(q3, dict), f"{board} results.json has no q3 for analysis set {aset}")
        need(
            entry == {"source": board, "analysis_set": aset, **q3},
            f"artifacts/e3/bounds.json entry for {board} differs from its results.json q3",
        )
    # E3 on E1 comes from the E1 analysis under the E1 preregistration (prereg 0004).
    on_e1 = load_json(rel(ctx, "artifacts/e1/e3_bounds.json"))
    need(on_e1.get("source") == "e1" and bool(on_e1.get("entries")), "E3 bounds do not cover E1")
    need(
        isinstance(on_e1.get("rankings"), dict) and bool(on_e1["rankings"]),
        "E3 on E1 has no rankings",
    )
    need(
        isinstance(on_e1.get("adjacent"), list) and bool(on_e1["adjacent"]),
        "E3 on E1 has no adjacent-pair bounds",
    )
    check_prereg_link(ctx, str(on_e1.get("prereg", "")), "artifacts/e1/e3_bounds.json")
    note = ""
    if ctx.fast:
        note = "; recompute skipped (fast)"
    else:
        need_setup(ctx)
        res = run(
            ctx,
            "e2-verify",
            ["uv", "run", "--no-sync", "python", "-m", "noisefloor.reanalysis", "--verify"],
        )
        need(res.code == 0, f"E2/E3 recompute: {tail(res.out)}")
        note = "; recomputed from snapshots"
    return f"{len(boards)} leaderboards ({', '.join(boards)[:50]}), E3 bounds{note}"


def g9(ctx: Ctx) -> str:
    need_setup(ctx)
    dist = ctx.tmp / "dist"
    res = run(ctx, "build-wheel", ["uv", "build", "--wheel", "--out-dir", str(dist)])
    need(res.code == 0, f"uv build: {tail(res.out)}")
    wheels = sorted(dist.glob("evalstats-*.whl"))
    need(len(wheels) == 1, f"{len(wheels)} evalstats wheels built")
    venv = ctx.tmp / "wheel-venv"
    res = run(ctx, "wheel-venv", ["uv", "venv", "--quiet", str(venv)], cwd=ctx.tmp)
    need(res.code == 0, f"uv venv: {tail(res.out)}")
    py = venv / "bin" / "python"
    res = run(
        ctx,
        "wheel-install",
        ["uv", "pip", "install", "--quiet", "--python", str(py), f"{wheels[0]}[glmm]"],
        cwd=ctx.tmp,
    )
    need(res.code == 0, f"wheel install: {tail(res.out)}")
    res = run(
        ctx,
        "wheel-import",
        [str(py), "-c", "import json, evalstats; print(json.dumps(evalstats.__all__))"],
        cwd=ctx.tmp,
    )
    need(res.code == 0, f"import from wheel: {tail(res.out)}")
    public = json.loads(res.stdout.strip().splitlines()[-1])
    res = run(ctx, "wheel-cli", [str(venv / "bin" / "evalstats"), "--help"], cwd=ctx.tmp)
    need(res.code == 0, f"evalstats --help from wheel: {tail(res.out)}")
    examples = sorted(rel(ctx, "docs/examples").glob("*.py"))
    need(len(examples) >= 5, f"{len(examples)} runnable examples < 5")
    for ex in examples:
        res = run(
            ctx, f"example-{ex.stem}", [*ctx.netns, str(py), str(ex)], cwd=ex.parent, timeout=600
        )
        need(res.code == 0, f"example {ex.name} failed: {tail(res.out)}")
    api = rel(ctx, "docs/api")
    api_files = [p for p in api.rglob("*") if p.is_file()] if api.exists() else []
    need(bool(api_files), "no generated API reference in docs/api/")
    api_text = "\n".join(p.read_text(errors="replace") for p in api_files)
    need("generated" in api_text.lower(), "docs/api does not say it is generated")
    absent = [n for n in public if n not in api_text]
    need(not absent, f"API reference lacks {', '.join(absent[:3])}")
    return (
        f"wheel {wheels[0].name} installs in a clean venv; {len(examples)} examples run; "
        f"API reference covers {len(public)} names"
    )


def g10(ctx: Ctx) -> str:
    need_setup(ctx)
    for doc in ("README.md", "docs/REPORT.md"):
        need(rel(ctx, doc).exists(), f"missing {doc}")
    for target in ("claimcheck", "freshness"):
        rule = makefile_rule(ctx, target)
        need(
            "README.md" in rule and "docs/REPORT.md" in rule,
            f"make {target} does not cover README.md and docs/REPORT.md",
        )
    res = make(ctx, "claimcheck", timeout=900)
    need(res.code == 0, f"make claimcheck: {tail(res.out)}")
    res2 = make(ctx, "freshness", timeout=3600)
    need(res2.code == 0, f"make freshness: {tail(res2.out)}")
    # Derived artifacts that claims cite must be what their committed scripts make of the
    # banked sources (rule 1; D-039), not only internally consistent with the documents.
    for name, cmd in DERIVED_CHECKS:
        res3 = run(ctx, f"derived-{name}", ["uv", "run", "--no-sync", *cmd], timeout=1800)
        need(res3.code == 0, f"{name} does not reproduce: {tail(res3.stdout or res3.out)}")
    return (
        f"claimcheck: {tail(res.stdout)[:50]}; freshness: {tail(res2.stdout)[:50]}; "
        f"{len(DERIVED_CHECKS)} derived artifacts reproduce"
    )


def g11(ctx: Ctx) -> str:
    files = sorted(rel(ctx, "prereg").glob("[0-9][0-9][0-9][0-9]-*.md"))
    need(bool(files), "no preregistrations in prereg/")
    for f in files:
        relp = str(f.relative_to(ctx.clone))
        n = len(commits_touching(ctx.clone, relp))
        need(n == 1, f"{relp} changed after it was frozen ({n} commits)")
    linked = 0
    for art in sorted(rel(ctx, "artifacts").rglob("*.json")):
        try:
            data = json.loads(art.read_text())
        except ValueError:
            continue
        if isinstance(data, dict) and data.get("prereg"):
            check_prereg_link(ctx, str(data["prereg"]), str(art.relative_to(ctx.clone)))
            linked += 1
    need_setup(ctx)
    res = make(ctx, "prereg-verify", timeout=600)
    need(res.code == 0, f"make prereg-verify: {tail(res.out)}")
    return f"{len(files)} preregistrations frozen; {linked} artifacts predated by their prereg"


def review_drift(clone: Path, reviewed: str, head: str) -> list[str]:
    """Paths changed between the reviewed commit and HEAD, a moved file under both of its paths.

    Without ``--no-renames`` git lists only a moved file's new path, so a file moved into
    artifacts/review/ would hide that it left the reviewed code (D-041).
    """
    return git(clone, "diff", "--no-renames", "--name-only", reviewed, head).splitlines()


def g12(ctx: Ctx) -> str:
    data = load_json(rel(ctx, "artifacts/review/latest.json"))
    need(data.get("verdict") == "SHIP", f"latest review verdict {data.get('verdict')}")
    need(data.get("blocking") == [], "latest review lists blocking issues")
    reviewed = str(data.get("reviewed_commit", ""))
    head = git(ctx.clone, "rev-parse", "HEAD")
    if reviewed != head:
        need(strictly_before(ctx.clone, reviewed, head), "reviewed commit is not an ancestor")
        changed = review_drift(ctx.clone, reviewed, head)
        allowed = ("artifacts/review/", "STATUS.md", "docs/LOG.md")
        stray = [c for c in changed if not (c.startswith(allowed[0]) or c in allowed[1:])]
        need(not stray, f"code changed since review: {', '.join(stray[:3])}")
    log = rel(ctx, "docs/REVIEW_LOG.md")
    need(log.exists(), "missing docs/REVIEW_LOG.md")
    log_text = log.read_text()
    for issue in data.get("non_blocking", []):
        need(
            id_listed(str(issue.get("id")), log_text), f"non-blocking {issue.get('id')} not in log"
        )
    need_setup(ctx)
    res = run(
        ctx,
        "review-validate",
        ["uv", "run", "--no-sync", "python", "tools/review.py", "--validate"],
    )
    # stdout alone: uv may print a warning on stderr after the verdict line (D-039).
    need(res.code == 0 and tail(res.stdout) == "REVIEW: SHIP", f"make review: {tail(res.out)}")
    return f"REVIEW: SHIP for {reviewed[:12]}; {len(data.get('non_blocking', []))} non-blocking"


def _headings(text: str) -> list[str]:
    return [ln.lstrip("#").strip().lower() for ln in text.splitlines() if ln.startswith("#")]


def _strip_code(text: str) -> str:
    return re.sub(r"```.*?```", "", text, flags=re.S)


def first_screen_interval(text: str) -> bool:
    """True if ``text`` shows a bracketed interval and names it a CI, as rendered.

    Claim markers (``<!-- claim:id -->``, docs/INTEGRITY.md) do not render, so they are
    removed first; a signed bound (``+0.077``) is an interval end like any other.
    """
    visible = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    interval = r"\[\s*[-+]?\d[\d.]*%?\s*,\s*[-+]?\d[\d.]*%?\s*\]"
    named = re.search(r"\bCI\b|confidence interval", visible) is not None
    return re.search(interval, visible) is not None and named


def g13(ctx: Ctx) -> str:
    readme_path = rel(ctx, "README.md")
    need(readme_path.exists(), "missing README.md")
    readme = readme_path.read_text()
    need("This README is a build brief" not in readme, "README.md is still the build brief")
    for name, text in (
        ("README.md", readme),
        ("docs/REPORT.md", _read(rel(ctx, "docs/REPORT.md"))),
    ):
        need(EM_DASH not in text, f"{name} contains an em dash")
        low = text.lower()
        hype = [w for w in HYPE_WORDS if w in low]
        need(not hype, f"{name} uses hype words: {', '.join(hype)}")
    lines = readme.splitlines()
    need(
        bool(lines) and lines[0].strip().lower().startswith("# noise-floor"),
        "README does not open with the project name",
    )
    second = next((i for i, ln in enumerate(lines) if ln.startswith("## ")), len(lines))
    first_screen = "\n".join(lines[:second])
    svgs = re.findall(r"\(([^)\s]+\.svg)\)", first_screen)
    need(bool(svgs), "no headline SVG on the first screen")
    need(all(rel(ctx, s).exists() for s in svgs), "headline SVG file missing")
    need(
        re.search(r"^\|.*\|\s*\n\|[\s:|-]+\|", first_screen, re.M) is not None,
        "no headline table on the first screen",
    )
    need(first_screen_interval(first_screen), "no confidence interval in the first-screen finding")
    need(
        "quickstart" in first_screen.lower() and "```" in first_screen,
        "no quickstart on the first screen",
    )
    heads = " | ".join(_headings(readme))
    for label, pattern in README_SECTIONS:
        need(re.search(pattern, heads) is not None, f"README lacks a '{label}' section")
    report = _read(rel(ctx, "docs/REPORT.md"))
    need(bool(report), "missing docs/REPORT.md")
    words = len(_strip_code(report).split())
    need(4000 <= words <= 8000, f"REPORT.md has {words} words, not 4000 to 8000")
    rheads = " | ".join(_headings(report))
    for sec in REPORT_SECTIONS:
        need(sec in rheads, f"REPORT.md lacks '{sec}'")
    report_dir = rel(ctx, "docs")
    for fig in re.findall(r"!\[[^\]]*\]\(([^)\s]+)\)", report):
        need((report_dir / fig).exists() or rel(ctx, fig).exists(), f"REPORT figure {fig} missing")
    cites = load_json(rel(ctx, "artifacts/citecheck.json"))
    entries = cites.get("references", [])
    need(bool(entries), "artifacts/citecheck.json lists no references")
    for e in entries:
        for key in ("url", "final_url", "title", "retrieved", "content_sha256"):
            need(bool(e.get(key)), f"citecheck entry lacks {key}")
        need(e.get("verified") is True, f"reference not verified: {str(e.get('url'))[:60]}")
    known = {str(e["url"]) for e in entries} | {str(e["final_url"]) for e in entries}
    own = "github.com/raj200501/noise-floor"
    cited = {u.rstrip(".,") for u in re.findall(r"https?://[^\s)>\]\"'`]+", readme + report)}
    unverified = sorted(u for u in cited if u not in known and own not in u)
    need(not unverified, f"cited but not verified: {unverified[0][:60] if unverified else ''}")
    need_setup(ctx)
    res = make(ctx, "citecheck", timeout=600)
    need(res.code == 0, f"make citecheck: {tail(res.out)}")
    note = ""
    if ctx.fast:
        note = "; figure regeneration skipped (fast)"
    else:
        res = make(ctx, "report", timeout=3600)
        need(res.code == 0, f"make report: {tail(res.out)}")
        dirty = git(ctx.clone, "status", "--porcelain", "--", *svgs)
        need(not dirty, "make report changes the headline figure")
        note = "; headline figure reproduces"
    return f"README per section 11; REPORT {words} words; {len(entries)} references verified{note}"


def _read(path: Path) -> str:
    return path.read_text() if path.exists() else ""


def g14(ctx: Ctx, pre: list[str]) -> str:
    need(not pre, "; ".join(pre))
    for name in REQUIRED_FILES:
        need(rel(ctx, name).exists(), f"missing {name}")
    budget_path = rel(ctx, "budget.toml")
    need(budget_path.exists(), "missing budget.toml")
    with budget_path.open("rb") as fh:
        caps = tomllib.load(fh)
    need(caps.get("cash_usd") == 0, "budget.toml cash_usd is not 0")
    defaults = brief_caps(ctx)
    for key, default in defaults.items():
        need(key in caps, f"budget.toml lacks {key}")
        if key == "disk_floor_gb":
            need(caps[key] >= default, "disk_floor_gb lowered below the brief default")
        else:
            need(caps[key] <= default, f"cap {key} raised above the brief default")
    need_setup(ctx)
    # The scanner also searches for the literal values of sensitive variables, which clean_env
    # removes; it only reads files and prints pattern names, never values (D-040).
    scan_env = {**ctx.env, **{k: v for k, v in os.environ.items() if k in SECRET_ENV}}
    res = make(ctx, "secrets-scan", timeout=900, env=scan_env)
    need(res.code == 0, f"make secrets-scan: {tail(res.out)}")
    res = make(ctx, "budget", timeout=300)
    need(res.code == 0, f"make budget: {tail(res.out)}")
    need(re.search(r"^cash_usd = 0$", res.out, re.M) is not None, "make budget: cash_usd != 0")
    return "secrets-scan clean; cash_usd = 0, caps respected; private/ untracked; tree clean"


# --------------------------------------------------------------------------- driver


def precheck(repo: Path) -> list[str]:
    """G14 parts that must be checked in the original checkout."""
    problems = []
    status = git(repo, "status", "--porcelain")
    if status:
        problems.append(f"working tree not clean ({len(status.splitlines())} entries)")
    tracked = git(repo, "ls-files", "--", "private")
    if tracked:
        problems.append(f"{len(tracked.splitlines())} files under private/ are tracked")
    manifest = repo / "artifacts" / "private_manifest.json"
    if manifest.exists():
        try:
            entries = json.loads(manifest.read_text()).get("files", {})
        except ValueError:
            entries = None
            problems.append("artifacts/private_manifest.json is not valid JSON")
        for path, digest in (entries or {}).items():
            p = repo / path
            if not str(path).startswith("private/"):
                problems.append(f"private manifest entry outside private/: {path}")
            elif not p.exists():
                problems.append(f"private file missing: {path}")
            elif sha256_file(p) != digest:
                problems.append(f"private file hash mismatch: {path}")
    return problems


def netns_prefix() -> list[str]:
    """A no-network prefix if this OS supports one without privileges."""
    if shutil.which("unshare"):
        proc = subprocess.run(["unshare", "-rn", "true"], capture_output=True, check=False)
        if proc.returncode == 0:
            return ["unshare", "-rn"]
    return []


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the 14 ship gates on a fresh clone.")
    parser.add_argument("--fast", action="store_true", help="skip slow steps (does not count)")
    parser.add_argument("--keep", action="store_true", help="keep the temp dir (path on stderr)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    repo = REPO
    head = git(repo, "rev-parse", "HEAD")
    pre = precheck(repo)
    tmp = Path(tempfile.mkdtemp(prefix="ship-check-"))
    clone = tmp / "clone"
    subprocess.run(
        ["git", "clone", "--quiet", "--no-hardlinks", str(repo), str(clone)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(clone), "checkout", "--quiet", "--detach", head],
        check=True,
        capture_output=True,
    )
    ctx = Ctx(
        repo=repo, clone=clone, tmp=tmp, fast=args.fast, env=clean_env(), netns=netns_prefix()
    )

    print(f"HEAD {head} fresh-clone", flush=True)
    gates: list[tuple[str, Callable[[], str]]] = [
        ("G1", lambda: g1(ctx)),
        ("G2", lambda: g2(ctx)),
        ("G4", lambda: g4(ctx)),
        ("G3", lambda: g3(ctx)),
        ("G5", lambda: g5(ctx)),
        ("G6", lambda: g6(ctx)),
        ("G7", lambda: g7(ctx)),
        ("G8", lambda: g8(ctx)),
        ("G9", lambda: g9(ctx)),
        ("G10", lambda: g10(ctx)),
        ("G11", lambda: g11(ctx)),
        ("G12", lambda: g12(ctx)),
        ("G13", lambda: g13(ctx)),
        ("G14", lambda: g14(ctx, pre)),
    ]
    lines: dict[str, str] = {}
    passed = 0
    for name, fn in gates:
        try:
            evidence = fn()
            ok = True
        except GateError as exc:
            evidence, ok = str(exc), False
        except Exception as exc:  # a crash in one gate must not hide the others
            evidence, ok = f"ship-check error: {type(exc).__name__}: {exc}", False
        passed += ok
        lines[name] = f"{name} {'PASS' if ok else 'FAIL'} {' '.join(evidence.split())[:200]}"
    # G3 runs after G4 so fast mode can reuse the demo result; print in gate order.
    for n in range(1, 15):
        print(lines[f"G{n}"], flush=True)
    if args.fast:
        print(f"SHIP-CHECK-FAST (does not count): {passed}/14")
    elif passed == 14:
        print("SHIP-CHECK: PASS 14/14")
    else:
        print(f"SHIP-CHECK: FAIL {passed}/14")
    if args.keep:
        print(f"ship-check temp dir: {tmp}", file=sys.stderr)
    else:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0 if passed == 14 else 1


if __name__ == "__main__":
    sys.exit(main())
