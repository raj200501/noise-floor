"""Unit tests for the stdlib-only tools that guard the ship gates."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools"


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"tools_{name}", TOOLS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ship_check = load("ship_check")
secrets_scan = load("secrets_scan")
budget = load("budget")
doctor = load("doctor")


def test_coverage_rule_exact_nominal_passes() -> None:
    ok, cov, _ = ship_check.coverage_rule(1900, 2000, 0.95)
    assert ok and cov == pytest.approx(0.95)


def test_coverage_rule_tolerance_band() -> None:
    # 0.93 observed with R = 2000: MCSE about 0.0057, threshold about 0.9236.
    assert ship_check.coverage_rule(1860, 2000, 0.95)[0]
    assert not ship_check.coverage_rule(1840, 2000, 0.95)[0]


def test_coverage_rule_over_coverage_passes_and_zero_reps_fails() -> None:
    assert ship_check.coverage_rule(2000, 2000, 0.95)[0]
    assert not ship_check.coverage_rule(0, 0, 0.95)[0]


def test_finite_interval() -> None:
    assert ship_check.finite_interval({"estimate": 1, "lo": 0, "hi": 2})
    assert not ship_check.finite_interval({"estimate": 3, "lo": 0, "hi": 2})
    assert not ship_check.finite_interval({"estimate": float("nan"), "lo": 0, "hi": 2})
    assert not ship_check.finite_interval({"lo": 0, "hi": 2})


def test_secrets_patterns_detect_fake_tokens() -> None:
    fake_github = ("gh" + "p_" + "a" * 36).encode()
    fake_anthropic = ("sk-" + "ant-" + "b" * 30).encode()
    fake_assignment = b'api_key = "' + b"c" * 30 + b'"'
    assert "github-token" in secrets_scan.scan_bytes(fake_github, [])
    assert "anthropic-key" in secrets_scan.scan_bytes(fake_anthropic, [])
    assert "generic-assignment" in secrets_scan.scan_bytes(fake_assignment, [])


def test_secrets_scan_env_value_and_clean_text() -> None:
    assert secrets_scan.scan_bytes(b"xx s3cr3t-value-0123456789 yy", [b"s3cr3t-value-0123456789"])
    assert secrets_scan.scan_bytes(b'"content_sha256": "' + b"f" * 64 + b'"', []) == []


CAPS = {
    "cash_usd": 0,
    "claude_agent_runs_per_day": 2,
    "claude_agent_max_turns": 40,
    "claude_single_calls_per_day": 300,
    "claude_review_runs_per_day": 6,
}


def test_budget_ok_and_plan_equivalent_not_cash() -> None:
    entries = [{"ts": "2026-10-23T01:00:00", "kind": "agent_run", "plan_equivalent_usd": 1.5}]
    lines, ok = budget.check(CAPS, entries)
    assert ok and "cash_usd = 0" in lines


def test_budget_daily_cap_and_cash_violations() -> None:
    runs = [{"ts": f"2026-10-23T0{i}:00:00", "kind": "agent_run"} for i in range(3)]
    assert not budget.check(CAPS, runs)[1]
    lines, ok = budget.check(CAPS, [{"ts": "2026-10-23", "kind": "local", "cash_usd": 0.01}])
    assert not ok and "cash_usd = 0" not in lines


def test_budget_turn_cap() -> None:
    entries = [{"ts": "2026-10-23", "kind": "agent_run", "turns": 41}]
    assert not budget.check(CAPS, entries)[1]


def test_budget_reads_turns_where_the_client_logs_them() -> None:
    # The client logs turns on agent_usage rows (its agent_run rows carry none).
    cap = int(CAPS["claude_agent_max_turns"])
    run = {"ts": "2026-10-23", "kind": "agent_run"}

    def usage(turns: int) -> dict[str, object]:
        return {"ts": "2026-10-23", "kind": "agent_usage", "turns": turns}

    lines, ok = budget.check(CAPS, [run, usage(cap)])
    assert ok and f"worst {cap}/{cap} OK" in "\n".join(lines)
    lines, ok = budget.check(CAPS, [run, usage(cap + 1)])  # stopped by the stream check
    assert ok and "1 stopped at the cap" in "\n".join(lines)
    assert not budget.check(CAPS, [run, usage(cap + 2)])[1]


def test_billing_guard_detects_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(doctor.os.environ):
        if name.startswith("ANTHROPIC_") or name.startswith("CLAUDE_CODE_USE_"):
            monkeypatch.delenv(name)
    monkeypatch.setattr(doctor, "CLAUDE_SETTINGS_FILES", ())
    assert doctor.billing_guard_findings() == []
    monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "1")
    assert doctor.billing_guard_findings() == ["CLAUDE_CODE_USE_BEDROCK"]


def test_routing_findings_names_overrides_and_settings_env_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    for name in list(doctor.os.environ):
        if name.startswith("ANTHROPIC_") or name.startswith("CLAUDE_CODE_USE_"):
            monkeypatch.delenv(name)
    settings = tmp_path / "settings.json"
    monkeypatch.setattr(doctor, "CLAUDE_SETTINGS_FILES", (settings,))
    assert doctor.routing_findings() == []
    monkeypatch.setenv("ANTHROPIC_DEFAULT_HAIKU_MODEL", "x")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")  # the billing guard's, not repeated here
    settings.write_text(
        json.dumps({"env": {"ANTHROPIC_BASE_URL": "x", "DISABLE_AUTOUPDATER": "1"}})
    )
    assert doctor.routing_findings() == [
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        f"ANTHROPIC_BASE_URL in {settings}",
    ]


def test_cap_share_is_the_smaller_and_floors_claude_caps() -> None:
    caps = {
        **CAPS,
        "claude_agent_runs_per_day": 30,
        "claude_review_runs_per_day": 6,
        "cap_share": 0.2,
        "disk_floor_gb": 8,
    }
    assert budget.cap_share(caps, None) == pytest.approx(0.2)
    assert budget.cap_share(caps, "0.1") == pytest.approx(0.1)
    assert budget.cap_share(caps, "0.9") == pytest.approx(0.2)
    eff = budget.effective_caps(caps, 0.2)
    assert eff["claude_agent_runs_per_day"] == 6
    assert eff["claude_review_runs_per_day"] == 1
    assert eff["claude_agent_max_turns"] == 8
    assert eff["disk_floor_gb"] == 8


def _ctx(root: Path):
    return ship_check.Ctx(repo=root, clone=root, tmp=root, fast=True, env={}, netns=[])


def test_e0_grids_and_frozen_headline_cells(tmp_path: Path) -> None:
    import json

    pre = tmp_path / "prereg"
    pre.mkdir()
    (pre / "0001-other.md").write_text("# no grid\n")
    (pre / "0003-grid.md").write_text(
        "# 0003\n\nHeadline cells (every interval behind an E2 headline number): `S-a`,\n"
        "`P-b`. Every headline cell must\npass; see `evalstats.validity`.\n\n"
        '```json e0-grid\n[{"cell": "S-a"}]\n```\n'
    )
    (pre / "0005-more.md").write_text(
        "---\namends: 0003\ndate: 2026-11-04\n---\n# 0005\n\n"
        '```json e0-grid\n[{"cell": "G-c"}]\n```\n\n'
        '```json headline\n{"e1 thing": "G-c"}\n```\n'
    )
    (pre / "0006-bad.md").write_text("---\namends: 0001\n---\n")
    files = ["prereg/0001-other.md", "prereg/0003-grid.md", "prereg/0005-more.md"]
    (pre / "CHAIN.jsonl").write_text("".join(json.dumps({"file": f}) + "\n" for f in files))
    ctx = _ctx(tmp_path)
    assert ship_check.amends(ctx, "prereg/0005-more.md") == "0003"
    assert ship_check.amends(ctx, "prereg/0003-grid.md") is None
    grids = ship_check.e0_grids(ctx)
    assert grids == [
        ("prereg/0003-grid.md", "artifacts/e0/coverage.json"),
        ("prereg/0005-more.md", "artifacts/e0/coverage-0005.json"),
    ]
    assert ship_check.frozen_headline_cells(ctx, grids) == {"S-a", "P-b", "G-c"}
    (pre / "0006-bad.md").write_text(
        '---\namends: 0001\n---\n```json e0-grid\n[{"cell": "x"}]\n```\n'
    )
    (pre / "CHAIN.jsonl").write_text(
        "".join(json.dumps({"file": f}) + "\n" for f in [*files, "prereg/0006-bad.md"])
    )
    with pytest.raises(ship_check.GateError, match="does not amend"):
        ship_check.e0_grids(ctx)


def test_utc_time() -> None:
    a = ship_check.utc_time("2026-11-05T00:00:00Z")
    b = ship_check.utc_time("2026-11-05T01:00:00+01:00")
    c = ship_check.utc_time("2026-11-05T00:00:01")
    assert a == b and c > a


def _g7_repo(
    root: Path, rows: list[dict[str, Any]], post_datasets: int = 200, post_code: str = ""
) -> None:
    """A git repo with a frozen E1 preregistration, then E1 artifacts built from ``rows``."""
    import hashlib
    import json
    import subprocess

    def git(*args: str) -> None:
        env = {
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@example.invalid",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@example.invalid",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "PATH": "/usr/bin:/bin",
        }
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, env=env)

    design = {
        "factors": {f: ["0", "1"] for f in ("harness", "timeout", "steps", "temperature")},
        "models": [
            {"id": "m1", "family": "a", "backend": "local"},
            {"id": "m2", "family": "b", "backend": "local"},
            {"id": "c1", "family": "claude", "backend": "claude"},
        ],
    }
    (root / "prereg").mkdir(parents=True)
    prereg = root / "prereg" / "0004-e1.md"
    prereg.write_text("# E1\n\n```json e1-design\n" + json.dumps(design) + "\n```\n")
    sha = hashlib.sha256(prereg.read_bytes()).hexdigest()
    entry = {"file": "prereg/0004-e1.md", "sha256": sha, "frozen_at": "2026-11-05T00:00:00Z"}
    (root / "prereg" / "CHAIN.jsonl").write_text(json.dumps(entry) + "\n")
    git("init", "-q")
    git("add", "-A")
    git("commit", "-q", "-m", "freeze")
    e1 = root / "artifacts" / "e1"
    e1.mkdir(parents=True)
    for row in rows:
        row.update(prereg="prereg/0004-e1.md", prereg_sha256=sha)
    (e1 / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    interval = {"estimate": 0.1, "lo": 0.05, "hi": 0.2, "share": 0.1, "share_lo": 0.0}
    code = {"hash": "c" * 16, "validated_by": {}, "amendment": None}
    vc = {"components": [{**interval, "name": "task", "share_hi": 0.3}], "analysis_code": code}
    (e1 / "variance_components.json").write_text(json.dumps(vc))
    post = {
        "prereg": "prereg/0004-e1.md",
        "prereg_sha256": sha,
        "synthetic": True,
        "analysis_code": post_code or code["hash"],
        "components": {"task": {"datasets": 200}, "harness": {"datasets": post_datasets}},
        "not_validated": [],
    }
    (e1 / "post_check.json").write_text(json.dumps(post))
    (e1 / "mde.json").write_text(json.dumps({"rows": [{"mde": 0.1, "lo": 0.05, "hi": 0.2}]}))
    files = {
        f"artifacts/e1/{n}": hashlib.sha256((e1 / n).read_bytes()).hexdigest()
        for n in ("runs.jsonl", "variance_components.json", "mde.json")
    }
    manifest = {"prereg": "prereg/0004-e1.md", "files": files}
    (e1 / "manifest.json").write_text(json.dumps(manifest))
    git("add", "-A")
    git("commit", "-q", "-m", "bank")


def _g7_rows(local: int, claude: int, status: str = "ok") -> list[dict[str, Any]]:
    rows = []
    for i in range(local + claude):
        model = ("m1", "m2")[i % 2] if i < local else "c1"
        rows.append(
            {
                "run_id": f"r{i}",
                "confirmatory": True,
                "started_at": "2026-11-06T01:00:00+00:00",
                "model_id": model,
                "factors": {"harness": "0", "timeout": "1"},
                "outcome": i % 2,
                "status": status,
                "day": f"2026-11-0{6 + i % 3}",
            }
        )
    return rows


def test_g7_counts_runs_that_ran_and_rejects_duplicates(tmp_path: Path) -> None:
    ok = tmp_path / "ok"
    _g7_repo(ok, _g7_rows(900, 160))
    assert ship_check.g7(_ctx(ok)).startswith("1060 runs (160 Claude), 3 models, 3 days")
    # Padding with infra_error rows (runs that did not run the task) cannot pass the floors.
    padded = tmp_path / "padded"
    rows = _g7_rows(800, 160) + [
        {**r, "run_id": f"x{i}"} for i, r in enumerate(_g7_rows(200, 0, "infra_error"))
    ]
    _g7_repo(padded, rows)
    with pytest.raises(ship_check.GateError, match="960 confirmatory task runs < 1000"):
        ship_check.g7(_ctx(padded))
    claude_short = tmp_path / "claude"
    rows = _g7_rows(1000, 140) + [
        {**r, "run_id": f"y{i}"} for i, r in enumerate(_g7_rows(0, 20, "infra_error"))
    ]
    _g7_repo(claude_short, rows)
    with pytest.raises(ship_check.GateError, match="140 Claude-tier runs < 150"):
        ship_check.g7(_ctx(claude_short))
    dup = tmp_path / "dup"
    rows = _g7_rows(900, 160)
    _g7_repo(dup, [*rows, dict(rows[0])])
    with pytest.raises(ship_check.GateError, match="banked more than once"):
        ship_check.g7(_ctx(dup))
    # The post-E1 check must have run on all its datasets, under the tables' analysis code.
    short = tmp_path / "post-short"
    _g7_repo(short, _g7_rows(900, 160), post_datasets=150)
    with pytest.raises(ship_check.GateError, match="lacks 200 datasets"):
        ship_check.g7(_ctx(short))
    other = tmp_path / "post-code"
    _g7_repo(other, _g7_rows(900, 160), post_code="d" * 16)
    with pytest.raises(ship_check.GateError, match="ran other analysis code"):
        ship_check.g7(_ctx(other))
    missing = tmp_path / "post-missing"
    _g7_repo(missing, _g7_rows(900, 160))
    (missing / "artifacts" / "e1" / "post_check.json").unlink()
    with pytest.raises(ship_check.GateError):
        ship_check.g7(_ctx(missing))


def test_budget_counts_both_ledgers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    runs = [{"ts": "2026-11-05T01:00:00+00:00", "kind": "agent_run", "turns": 8}] * 4
    a, b = tmp_path / "ledger.jsonl", tmp_path / "ledger-claude.jsonl"
    a.write_text("".join(json.dumps(r) + "\n" for r in runs))
    monkeypatch.setattr(budget, "LEDGERS", (a, b))
    monkeypatch.delenv("CAP_SHARE", raising=False)
    assert budget.main() == 0  # 4 agent runs, cap 6 at cap share 0.2
    b.write_text("".join(json.dumps(r) + "\n" for r in runs))
    assert budget.main() == 1  # 8 on one day across the two ledgers


def _g6_repo(
    root: Path, components: list[str], *, rows_ok: bool = True, code: str | None = "a" * 16
) -> Path:
    import hashlib
    import json
    import subprocess

    def git(*args: str) -> None:
        env = {
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@example.invalid",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@example.invalid",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "PATH": "/usr/bin:/bin",
        }
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, env=env)

    config = {"design": {"n_tasks": 30}, "settings": {"level": 0.95}}
    seeds = list(range(200))
    (root / "prereg").mkdir(parents=True)
    text = "# G6\n\n"
    for name, value in (
        ("recovery-config", config),
        ("g6-seeds", seeds),
        ("g6-components", ["task", "harness"]),
    ):
        text += f"```json {name}\n{json.dumps(value)}\n```\n\n"
    (root / "prereg" / "0006-g6.md").write_text(text)
    git("init", "-q")
    git("add", "-A")
    git("commit", "-q", "-m", "freeze")

    def comp(i: int) -> dict[str, float]:
        hit = rows_ok or i % 4  # a quarter of misses when not ok
        return {"planted": 1.0, "estimate": 1.0, "lo": 0.5, "hi": 1.5 if hit else 0.9}

    datasets = [
        {"seed": s, "components": {c: comp(s) for c in components}, "error": None} for s in seeds
    ]
    bias = {c: 0.0 for c in components}
    artifact = {
        "prereg": "prereg/0006-g6.md",
        "nominal": 0.95,
        "config": config,
        "environment": {"env": hashlib.sha256(b"e").hexdigest()[:16]},
        "analysis_code": code,
        "datasets": datasets,
        "bias": bias,
    }
    (root / "artifacts" / "g6").mkdir(parents=True)
    (root / "artifacts" / "g6" / "recovery.json").write_text(json.dumps(artifact))
    git("add", "-A")
    git("commit", "-q", "-m", "bank")
    return root


def test_g6_checks_components_config_and_coverage(tmp_path: Path) -> None:
    good = _g6_repo(tmp_path / "good", ["task", "harness"])
    assert ship_check.g6(_ctx(good)).startswith("200 datasets (0 failed fits)")
    extra = _g6_repo(tmp_path / "extra", ["task", "harness", "score:m0"])
    with pytest.raises(ship_check.GateError, match="g6-components"):
        ship_check.g6(_ctx(extra))
    short = _g6_repo(tmp_path / "short", ["task"])
    with pytest.raises(ship_check.GateError, match="g6-components"):
        ship_check.g6(_ctx(short))
    low = _g6_repo(tmp_path / "low", ["task", "harness"], rows_ok=False)
    with pytest.raises(ship_check.GateError, match="coverage"):
        ship_check.g6(_ctx(low))
    blank = _g6_repo(tmp_path / "blank", ["task", "harness"], code=None)
    with pytest.raises(ship_check.GateError, match="analysis_code"):
        ship_check.g6(_ctx(blank))
    # E0 part 2, once banked, must have run the same analysis code.
    e0 = good / "artifacts" / "e0"
    e0.mkdir(parents=True)
    (e0 / "coverage-0005.json").write_text('{"analysis_code": "b' + "b" * 15 + '"}')
    with pytest.raises(ship_check.GateError, match="E0 part 2"):
        ship_check.g6(_ctx(good))
    (e0 / "coverage-0005.json").write_text('{"analysis_code": "' + "a" * 16 + '"}')
    assert ship_check.g6(_ctx(good)).startswith("200 datasets")


def test_retired_headline_cells_count_only_before_e1_data(tmp_path: Path) -> None:
    import json

    pre = tmp_path / "prereg"
    pre.mkdir()
    (pre / "0003-grid.md").write_text(
        "# 0003\n\nHeadline cells (E2): `S-a`. Every headline cell must\npass.\n\n"
        '```json e0-grid\n[{"cell": "S-a"}]\n```\n'
    )
    config = '```json recovery-config\n{"settings": {"share_interval": "%s"}}\n```\n\n'
    (pre / "0005-more.md").write_text(
        "---\namends: 0003\n---\n# 0005\n\n"
        '```json e0-grid\n[{"cell": "G-c"}, {"cell": "G-d"}, {"cell": "L-a", "method": "m",'
        ' "interval_type": "cluster_t"}]\n```\n\n'
        '```json headline\n{"e1 c": "G-c", "e1 d": "G-d", "e1 l": "L-a"}\n```\n\n'
        '```json headline-kinds\n{"G-c": "share", "G-d": "mde", "L-a": "local-score"}\n```\n\n'
        + config
        % "plugin"
    )
    swap = (
        "---\namends: 0003\n---\n# 0007\n\n"
        '```json e0-grid\n[{"cell": "G-e"}]\n```\n\n'
        '```json headline\n{"e1 c": "G-e"}\n```\n\n'
        '```json headline-kinds\n{"G-e": "%s"}\n```\n\n'
        '```json retired-headline\n["G-c"]\n```\n\n'
    )
    (pre / "0007-swap.md").write_text(swap % "share" + config % "percentile")
    allows = '```json analysis-code\n{"allows": "' + "e" * 16 + '"}\n```\n'
    (pre / "0008-e1.md").write_text("---\namends: 0004\n---\n# allows the new code\n\n" + allows)
    chain = [
        {"file": "prereg/0003-grid.md", "frozen_at": "2026-11-01T00:00:00Z"},
        {"file": "prereg/0005-more.md", "frozen_at": "2026-11-02T00:00:00Z"},
        {"file": "prereg/0007-swap.md", "frozen_at": "2026-11-03T00:00:00Z"},
        {"file": "prereg/0008-e1.md", "frozen_at": "2026-11-06T00:00:00Z"},
    ]
    (pre / "CHAIN.jsonl").write_text("".join(json.dumps(e) + "\n" for e in chain))
    ctx = _ctx(tmp_path)
    grids = ship_check.e0_grids(ctx)
    assert ship_check.frozen_headline_cells(ctx, grids) == {"S-a", "G-d", "G-e", "L-a"}
    (pre / "0007-swap.md").write_text(swap % "mde" + config % "percentile")  # no share cell
    with pytest.raises(ship_check.GateError, match="retires share cells with no replacement"):
        ship_check.frozen_headline_cells(ctx, grids)
    # The same method on new seeds is no replacement (review final-stats-4).
    (pre / "0007-swap.md").write_text(swap % "share" + config % "plugin")
    with pytest.raises(ship_check.GateError, match="test the same method"):
        ship_check.frozen_headline_cells(ctx, grids)
    (pre / "0007-swap.md").write_text(swap % "share")  # no recovery-config of its own
    with pytest.raises(ship_check.GateError, match="test the same method"):
        ship_check.frozen_headline_cells(ctx, grids)
    # A replaced non-GLMM kind needs new cells with another method or interval type.
    local = (
        "---\namends: 0003\n---\n# 0007\n\n"
        '```json e0-grid\n[{"cell": "L-b", "method": "m", "interval_type": "%s"}]\n```\n\n'
        '```json headline\n{"e1 l": "L-b"}\n```\n\n'
        '```json headline-kinds\n{"L-b": "local-score"}\n```\n\n'
        '```json retired-headline\n["L-a"]\n```\n'
    )
    (pre / "0007-swap.md").write_text(local % "cluster_t")
    with pytest.raises(ship_check.GateError, match="test the same method"):
        ship_check.frozen_headline_cells(ctx, grids)
    (pre / "0007-swap.md").write_text(local % "cluster_bootstrap")
    assert "L-b" in ship_check.frozen_headline_cells(ctx, grids)
    # Without an amendment of the E1 preregistration, nothing allows the new method.
    (pre / "CHAIN.jsonl").write_text("".join(json.dumps(e) + "\n" for e in chain[:3]))
    with pytest.raises(ship_check.GateError, match="no amendment of prereg 0004"):
        ship_check.frozen_headline_cells(ctx, grids)
    # An amendment of 0004 for another purpose (a rebuilt server) allows no new method.
    server = '```json llama-server\n{"sha256": "' + "f" * 64 + '"}\n```\n'
    (pre / "0008-e1.md").write_text("---\namends: 0004\n---\n# rebuilt server\n\n" + server)
    (pre / "CHAIN.jsonl").write_text("".join(json.dumps(e) + "\n" for e in chain))
    with pytest.raises(ship_check.GateError, match="no amendment of prereg 0004"):
        ship_check.frozen_headline_cells(ctx, grids)
    (pre / "0008-e1.md").write_text("---\namends: 0004\n---\n# allows the new code\n\n" + allows)
    # Under the task jackknife the box rule has no effect: changing it alone is no new method.
    jack = {"fixed_interval": "jackknife"}
    assert ship_check.glmm_rule("fixed", {**jack, "box_interval": "basic"}) == (
        ship_check.glmm_rule("fixed", {**jack, "box_interval": "percentile"})
    )
    box = {"fixed_interval": "box", "box_interval": "basic"}
    assert ship_check.glmm_rule("fixed", box) != ship_check.glmm_rule("fixed", jack)
    assert ship_check.glmm_rule("fixed", box) != (
        ship_check.glmm_rule("fixed", {**box, "box_interval": "student"})
    )
    (pre / "0007-swap.md").write_text(swap % "share" + config % "percentile")
    e1 = tmp_path / "artifacts" / "e1"
    e1.mkdir(parents=True)
    run = {"confirmatory": True, "started_at": "2026-11-05T00:01:00+00:00"}
    (e1 / "runs.jsonl").write_text(json.dumps(run) + "\n")
    with pytest.raises(ship_check.GateError, match="no amendment of prereg 0004"):
        ship_check.frozen_headline_cells(ctx, grids)  # 0008 was frozen after E1 data
    late = [*chain[:2], {**chain[2], "frozen_at": "2026-11-06T00:00:00Z"}, chain[3]]
    (pre / "CHAIN.jsonl").write_text("".join(json.dumps(e) + "\n" for e in late))
    with pytest.raises(ship_check.GateError, match="after E1 data exist"):
        ship_check.frozen_headline_cells(ctx, grids)


def test_ship_check_and_generator_map_cells_to_the_same_functions() -> None:
    from noisefloor import e0_validity

    assert ship_check.FUNCTION_OF == e0_validity.FUNCTIONS
    assert {"glmm_task_jackknife", "glmm_plugin_share"} <= ship_check.RESAMPLING_TYPES


def test_noise_floor_artifact_is_current_and_follows_its_formula() -> None:
    import math

    from scipy import stats

    noise_floor = load("noise_floor")
    assert noise_floor.main(["--check"]) == 0  # the banked artifact equals a fresh build
    pair = {"higher": "a", "lower": "b", "n_tasks": 51, "difference": 0.02}
    t = float(stats.t.ppf(0.975, 50))
    pair.update(lo=0.02 - 0.1 * t, hi=0.02 + 0.1 * t)  # standard error 0.1
    floor = noise_floor.board_floor({"sets": {"primary": {"level": 0.95, "q1": {"pairs": [pair]}}}})
    z = float(stats.norm.ppf(0.975) + stats.norm.ppf(0.8))
    assert math.isclose(floor["median_mde"], z * 0.1, rel_tol=1e-9)
    assert floor["n_gap_at_least_mde"] == 0 and floor["median_abs_gap"] == 0.02


def test_first_screen_interval_reads_the_rendered_text() -> None:
    marked = "differ by (95% CI [<!-- claim:a -->-0.041, <!-- claim:b -->+0.077])"
    assert ship_check.first_screen_interval(marked)
    assert ship_check.first_screen_interval("gap 0.02 (CI [0.01, 0.03])")
    assert not ship_check.first_screen_interval("gap 0.02, CI from 0.01 to 0.03")
    assert not ship_check.first_screen_interval("range [0.01, 0.03] with no interval named")
