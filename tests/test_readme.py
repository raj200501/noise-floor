"""The README's examples run as written: every Python block and every `evalstats` command line.

Inputs are synthetic rows (tests/synth.py) and the mock Inspect and Harbor fixtures under
tests/fixtures/adapters, written under the file names the README uses, so these runs are
synthetic and mock data, never model results.
"""

import json
import re
import shlex
import shutil
from pathlib import Path

import pytest

from evalstats import cli
from synth import make_results

ROOT = Path(__file__).resolve().parent.parent
ADAPTERS = Path(__file__).resolve().parent / "fixtures" / "adapters"
BLOCKS = re.findall(r"```(\w*)\n(.*?)```", (ROOT / "README.md").read_text(), flags=re.S)
PYTHON = [body for lang, body in BLOCKS if lang == "python"]
COMMANDS = [
    shlex.split(line.split("#", 1)[0].split(">", 1)[0])[1:]
    for lang, body in BLOCKS
    if lang == "bash"
    for line in body.splitlines()
    if line.startswith("evalstats ")
]


@pytest.fixture
def readme_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The files the README's examples read, in a fresh working directory."""
    monkeypatch.chdir(tmp_path)
    models, p = ("agent-a", "agent-b", "agent-c"), (0.6, 0.5, 0.4)
    make_results(n_tasks=40, runs=3, models=models, p=p).frame.to_csv("my_runs.csv", index=False)
    (tmp_path / "logs").mkdir()
    source = next((ADAPTERS / "inspect" / "json").glob("*.json"))
    for i, model in enumerate(("model-a", "model-b")):
        log = json.loads(source.read_text())
        log["eval"]["model"] = model
        log["eval"]["eval_id"] = f"{log['eval']['eval_id']}-{i}"
        (tmp_path / "logs" / f"log-{i}.json").write_text(json.dumps(log))
    shutil.copytree(ADAPTERS / "harbor" / "jobs" / "2026-10-23__02-00-00", "jobs/my-job")
    return tmp_path


def test_the_readme_has_examples_of_both_kinds() -> None:
    assert len(PYTHON) >= 2
    assert {argv[0] for argv in COMMANDS} == set(cli.COMMANDS)  # every subcommand is shown


@pytest.mark.parametrize("block", PYTHON, ids=[f"python-{i}" for i in range(len(PYTHON))])
def test_python_example_runs(
    block: str, readme_inputs: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exec(compile(block, "README.md", "exec"), {"__name__": "readme_example"})
    assert capsys.readouterr().out.strip()


@pytest.mark.parametrize("argv", COMMANDS, ids=[" ".join(a) for a in COMMANDS])
def test_command_example_runs(
    argv: list[str], readme_inputs: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    assert out.strip()
    if argv[0] == "report":
        assert out.startswith("# Confidence report:") and "## Adjacent orderings" in out
        assert ("## Ambiguous runs" in out) == ("--from" in argv)  # the readers add a status
