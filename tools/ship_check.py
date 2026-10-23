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
RESAMPLING_TYPES = {"bootstrap", "glmm", "parametric_bootstrap", "profile_glmm"}
G6_MIN_DATASETS = 200
E1_MIN_RUNS = 1000
E1_MIN_CLAUDE_RUNS = 150
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
    out: str
    seconds: float


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
        res = Result(proc.returncode, proc.stdout + proc.stderr, time.monotonic() - start)
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


def need(cond: bool, evidence: str) -> None:
    if not cond:
        raise GateError(evidence)


def need_setup(ctx: Ctx) -> None:
    need(bool(ctx.setup_ok), "make setup failed (see G1)")


def load_json(path: Path) -> Any:
    need(path.exists(), f"missing {path.name if path.is_absolute() else path}")
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
    need(isinstance(files, dict) and bool(files), f"{label} manifest lists no files")
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
        f"--junitxml={logs / 'junit.xml'} --cov-report=json:{logs / 'coverage.json'}"
    )
    res = make(ctx, "test", timeout=3600, env=env)
    need(res.code == 0, f"make test exit {res.code}: {tail(res.out)}")
    tests, bad = junit_counts(logs / "junit.xml")
    need(tests > 0 and bad == 0, f"junit: {tests} tests, {bad} failed")

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
    return f"setup ok; {tests} tests pass; line coverage {pct:.1f}% (omitted {len(omit)})"


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
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            where = f"{path.relative_to(ctx.clone)}:{lineno}"
            if re.search(r"#\s*mypy:\s*ignore-errors", line):
                raise GateError(f"{where} disables mypy for the file")
            if "type: ignore" not in line and "type:ignore" not in line:
                continue
            ignores += 1
            need(
                not (line.strip().startswith("#") and str(path).find("/src/") >= 0),
                f"{where} file-level type: ignore",
            )
            need(reason.search(line) is not None, f"{where} type: ignore without a reason")
    return f"ruff check, ruff format, mypy --strict clean; {ignores} type: ignore, all with reasons"


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
    res = make(ctx, "ci", timeout=3600)
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


def g5(ctx: Ctx) -> str:
    art = "artifacts/e0/coverage.json"
    data = load_json(rel(ctx, art))
    prereg = str(data.get("prereg", ""))
    grid = prereg_block(ctx, prereg, "e0-grid")
    check_prereg_link(ctx, prereg, art)
    need(isinstance(grid, list) and bool(grid), "e0-grid is empty")
    planned = {str(c["cell"]): c for c in grid}
    cells = {str(c["cell"]): c for c in data.get("cells", [])}
    need(set(cells) == set(planned), "E0 cells differ from the frozen grid")
    failing: list[str] = []
    for cid, cell in cells.items():
        plan = planned[cid]
        for key in ("method", "interval_type", "nominal"):
            need(cell.get(key) == plan.get(key), f"cell {cid} {key} differs from prereg")
        reps = int(cell["replicates"])
        floor = MIN_REPS_RESAMPLING if cell["interval_type"] in RESAMPLING_TYPES else MIN_REPS
        need(reps >= max(floor, int(plan.get("replicates", 0))), f"cell {cid}: {reps} replicates")
        ok, _, _ = coverage_rule(int(cell["covered"]), reps, float(cell["nominal"]))
        if not ok:
            failing.append(cid)
    head = load_json(rel(ctx, "artifacts/headline_cells.json"))
    headline = {str(c) for c in head.get("cells", [])}
    need(bool(headline), "no headline cells declared")
    need(headline <= set(cells), "a headline cell is not in the E0 grid")
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
    unsupported = set(json.loads(res.out.strip().splitlines()[-1]))
    missing = sorted(set(failing) - unsupported)
    need(not missing, f"failing cells not marked unsupported: {', '.join(missing[:3])}")
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
        ],
    )
    tests, bad = junit_counts(logs / "property.xml")
    need(res.code == 0 and tests > 0 and bad == 0, f"property suite: {tests} tests, {bad} failed")
    return (
        f"{len(cells)} cells, {len(failing)} failing (all unsupported), headline "
        f"{len(headline)} pass; {len(cross)} cross-checks; {tests} property tests"
    )


def g6(ctx: Ctx) -> str:
    art = "artifacts/g6/recovery.json"
    data = load_json(rel(ctx, art))
    prereg = str(data.get("prereg", ""))
    seeds = prereg_block(ctx, prereg, "g6-seeds")
    check_prereg_link(ctx, prereg, art)
    datasets = data.get("datasets", [])
    need(len(datasets) >= G6_MIN_DATASETS, f"{len(datasets)} datasets < {G6_MIN_DATASETS}")
    need(
        sorted(int(d["seed"]) for d in datasets) == sorted(int(s) for s in seeds),
        "dataset seeds differ from prereg",
    )
    nominal = float(data["nominal"])
    names = sorted({n for d in datasets for n in d["components"]})
    need(bool(names), "no variance components")
    reported_bias = data.get("bias", {})
    parts = []
    for name in names:
        rows = [d["components"][name] for d in datasets]
        need(len(rows) == len(datasets), f"component {name} missing in some datasets")
        covered = sum(1 for r in rows if float(r["lo"]) <= float(r["planted"]) <= float(r["hi"]))
        ok, cov, thr = coverage_rule(covered, len(rows), nominal)
        need(ok, f"{name} coverage {cov:.3f} < {thr:.3f}")
        bias = sum(float(r["estimate"]) - float(r["planted"]) for r in rows) / len(rows)
        need(name in reported_bias, f"bias for {name} not reported")
        need(
            math.isclose(float(reported_bias[name]), bias, rel_tol=1e-9, abs_tol=1e-12),
            f"reported bias for {name} does not match recomputation",
        )
        parts.append(f"{name} {cov:.3f}")
    return f"{len(datasets)} datasets; coverage " + ", ".join(parts)[:90]


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
    conf = [r for r in rows if r.get("confirmatory") is True]
    for r in conf:
        need(r.get("prereg") == prereg, f"run {r.get('run_id')} cites another prereg")
        need(str(r.get("model_id")) in models, f"run {r.get('run_id')} model not in design")
        for name, level in r.get("factors", {}).items():
            need(
                name in factors and level in factors[name],
                f"run {r.get('run_id')} factor {name}={level} not in design",
            )
        need(r.get("outcome") in (0, 1, True, False), f"run {r.get('run_id')} outcome not binary")
    need(len(conf) >= E1_MIN_RUNS, f"{len(conf)} confirmatory runs < {E1_MIN_RUNS}")
    claude = sum(1 for r in conf if models[str(r["model_id"])].get("backend") == "claude")
    need(claude >= E1_MIN_CLAUDE_RUNS, f"{claude} Claude-tier runs < {E1_MIN_CLAUDE_RUNS}")
    used = {str(r["model_id"]) for r in conf}
    families = {models[m].get("family") for m in used}
    need(len(used) >= 3 and len(families) >= 2, f"{len(used)} models, {len(families)} families")
    days = {r.get("day") for r in conf}
    need(len(days) >= 3, f"{len(days)} days < 3")
    vc = load_json(rel(ctx, "artifacts/e1/variance_components.json")).get("components", [])
    need(bool(vc) and all(finite_interval(c) for c in vc), "variance table lacks intervals")
    mde = load_json(rel(ctx, "artifacts/e1/mde.json")).get("rows", [])
    need(
        bool(mde) and all(finite_interval(r, ("mde", "lo", "hi")) for r in mde),
        "MDE table lacks intervals",
    )
    note = ""
    if ctx.fast:
        note = "; recompute skipped (fast)"
    else:
        need_setup(ctx)
        res = run(
            ctx,
            "e1-verify",
            ["uv", "run", "--no-sync", "python", "-m", "noisefloor.analysis.e1", "--verify"],
        )
        need(res.code == 0, f"E1 recompute: {tail(res.out)}")
        note = "; tables recomputed"
    return f"{len(conf)} runs ({claude} Claude), {len(used)} models, {len(days)} days{note}"


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
    need("e1" in sources, "E3 bounds do not cover E1")
    missing = [b for b in boards if b not in sources]
    need(not missing, f"E3 bounds missing for {', '.join(missing)}")
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
        ["uv", "pip", "install", "--quiet", "--python", str(py), str(wheels[0])],
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
    public = json.loads(res.out.strip().splitlines()[-1])
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
    return f"claimcheck: {tail(res.out)[:50]}; freshness: {tail(res2.out)[:50]}"


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


def g12(ctx: Ctx) -> str:
    data = load_json(rel(ctx, "artifacts/review/latest.json"))
    need(data.get("verdict") == "SHIP", f"latest review verdict {data.get('verdict')}")
    need(data.get("blocking") == [], "latest review lists blocking issues")
    reviewed = str(data.get("reviewed_commit", ""))
    head = git(ctx.clone, "rev-parse", "HEAD")
    if reviewed != head:
        need(strictly_before(ctx.clone, reviewed, head), "reviewed commit is not an ancestor")
        changed = git(ctx.clone, "diff", "--name-only", reviewed, head).splitlines()
        allowed = ("artifacts/review/", "STATUS.md", "docs/LOG.md")
        stray = [c for c in changed if not (c.startswith(allowed[0]) or c in allowed[1:])]
        need(not stray, f"code changed since review: {', '.join(stray[:3])}")
    log = rel(ctx, "docs/REVIEW_LOG.md")
    need(log.exists(), "missing docs/REVIEW_LOG.md")
    log_text = log.read_text()
    for issue in data.get("non_blocking", []):
        need(str(issue.get("id")) in log_text, f"non-blocking {issue.get('id')} not in log")
    need_setup(ctx)
    res = run(
        ctx,
        "review-validate",
        ["uv", "run", "--no-sync", "python", "tools/review.py", "--validate"],
    )
    need(res.code == 0 and tail(res.out) == "REVIEW: SHIP", f"make review: {tail(res.out)}")
    return f"REVIEW: SHIP for {reviewed[:12]}; {len(data.get('non_blocking', []))} non-blocking"


def _headings(text: str) -> list[str]:
    return [ln.lstrip("#").strip().lower() for ln in text.splitlines() if ln.startswith("#")]


def _strip_code(text: str) -> str:
    return re.sub(r"```.*?```", "", text, flags=re.S)


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
    interval = r"\[\s*-?\d[\d.]*%?\s*,\s*-?\d[\d.]*%?\s*\]"
    need(
        re.search(interval, first_screen) is not None
        and re.search(r"\bCI\b|confidence interval", first_screen) is not None,
        "no confidence interval in the first-screen finding",
    )
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
    res = make(ctx, "secrets-scan", timeout=900)
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
