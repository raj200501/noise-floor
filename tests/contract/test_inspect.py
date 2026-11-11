"""Contract tests for the Inspect AI log reader (evalstats.adapters.inspect).

The recorded logs under tests/fixtures/adapters/inspect/ were written by Inspect itself for
its mock model, so every outcome in them is mock data, not a model result (see the README
there). Expected values come from the raw log members, read here with zipfile and json, and
from the events that make_logs.py designs; none is copied from the adapter's output.
Malformed and edge-case logs are built in tmp_path.
"""

from __future__ import annotations

import importlib
import json
import math
import re
import shutil
import struct
import sys
import zipfile
import zlib
from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd
import pytest

from evalstats import EvalResults, SchemaError
from evalstats.adapters import STATUSES, read_inspect
from evalstats.adapters import inspect as adapter

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "adapters" / "inspect"
FORMATS = ("eval-deflate", "eval-zstd", "json", "legacy")
# Python 3.14 reads zstd members itself. (zipfile.ZIP_ZSTANDARD is no test: importing
# inspect_ai imports zipfile_zstd, which patches zstd support into zipfile for the process.)
NATIVE_ZSTD = sys.version_info >= (3, 14)

# The tiny_qa task of make_logs.py (README): four samples, three epochs, four scorers.
SAMPLE_IDS = ("sum", "product", "diff", "loop")
EPOCHS = (1, 2, 3)
SCORERS = ("match", "letter_grade", "closeness", "has_answer")
COVARIATES = (  # the module docstring's covariate list, in order
    "status",
    "status_reason",
    "task_name",
    "sample_id",
    "scorer",
    "score_value",
    "total_time",
    "working_time",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "retries",
    "eval_id",
    "log",
)
TOKENS = ("input_tokens", "output_tokens", "total_tokens")
LETTERS = {"C": 1.0, "P": 0.5, "I": 0.0, "N": 0.0}  # Inspect's value_to_float defaults
NAN = float("nan")

# Raw score values that make_logs.py's mock answers imply. The mock answers "5" to sum
# (target 5), "24 or 25" to product (target 24), "4", "5", and "I don't know." to diff in
# epochs 1 to 3 (target 5), and "1" to loop (target "never"). Sample sum raises in epoch 2
# before it is scored (None). NAN is Score.unscored(reason="scoring_failed"), which closeness
# returns when the answer or the target has no number; Inspect 0.3.205 has no
# Score.unscored, so the legacy log holds 0.0 there instead.
RAW: dict[str, dict[str, tuple[Any, Any, Any]]] = {
    "match": {
        "sum": ("C", None, "C"),
        "product": ("C", "C", "C"),
        "diff": ("I", "C", "I"),
        "loop": ("I", "I", "I"),
    },
    "letter_grade": {
        "sum": ("C", None, "C"),
        "product": ("P", "P", "P"),
        "diff": ("I", "C", "N"),
        "loop": ("I", "I", "I"),
    },
    "closeness": {
        "sum": (1.0, None, 1.0),
        "product": (1.0, 1.0, 1.0),
        "diff": (0.5, 1.0, NAN),
        "loop": (NAN, NAN, NAN),
    },
    "has_answer": {
        "sum": (True, None, True),
        "product": (True, True, True),
        "diff": (True, True, False),
        "loop": (True, True, True),
    },
}


# --------------------------------------------------------------------------- helpers


def only_log(directory: str) -> Path:
    """The one log file in a fixture directory."""
    [path] = [p for p in (FIXTURES / directory).iterdir() if p.suffix in (".eval", ".json")]
    return path


def raw_log(directory: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Header and samples of a fixture log, read with zipfile and json only.

    eval-zstd/ is read through eval-deflate/, which the README says holds the same members
    byte for byte (test_zstd_fixture_holds_the_deflate_members checks their CRCs).
    """
    path = only_log("eval-deflate" if directory == "eval-zstd" else directory)
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        return data, data["samples"]
    with zipfile.ZipFile(path) as zf:
        header = json.loads(zf.read("header.json"))
        samples = [json.loads(zf.read(n)) for n in zf.namelist() if n.startswith("samples/")]
    return header, samples


def is_nan(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)


def raw_value(directory: str, scorer: str, sample_id: str, epoch: int) -> Any:
    value = RAW[scorer][sample_id][epoch - 1]
    return 0.0 if directory == "legacy" and is_nan(value) else value


def designed_outcome(value: Any) -> float:
    """The documented outcome of a raw value: no score and NaN give 0."""
    if value is None or is_nan(value):
        return 0.0
    return LETTERS[value] if isinstance(value, str) else float(value)


def designed_status(
    directory: str, scorer: str, sample_id: str, epoch: int
) -> tuple[str, str | None]:
    """Status and reason that the README's designed events imply (first match wins)."""
    if (sample_id, epoch) == ("diff", 1):
        return "flagged", "invalidated"
    if (sample_id, epoch) == ("sum", 2):
        return "infra_error", "error"
    if sample_id == "loop":
        return "timeout", "limit:message"
    if is_nan(raw_value(directory, scorer, sample_id, epoch)):
        return "infra_error", "unscored:scoring_failed"
    return "ok", None


def records(res: EvalResults) -> dict[tuple[str, int], dict[str, Any]]:
    """Rows keyed by (sample id, epoch), with missing values as None."""
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for row in res.frame.to_dict("records"):
        clean = {
            str(k): None if not isinstance(v, str) and pd.isna(v) else v for k, v in row.items()
        }
        epoch = int(str(clean["run"]).rsplit("/", 1)[-1])
        rows[(str(clean["sample_id"]), epoch)] = clean
    return rows


def write_log(path: Path, samples: list[dict[str, Any]], results: Any = None, **spec: Any) -> Path:
    """Write a minimal JSON-format Inspect log (mock data) and return its path."""
    log: dict[str, Any] = {
        "version": 2,
        "status": "success",
        "eval": {"task": "t", "model": "m", "eval_id": "E", **spec},
        "samples": samples,
    }
    if results is not None:
        log["results"] = results
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(log))
    return path


def sample(sample_id: Any = "a", epoch: Any = 1, value: Any = "C", **fields: Any) -> dict[str, Any]:
    """A sample that scorer ``s`` scored ``value``; ``fields`` add or replace keys."""
    return {"id": sample_id, "epoch": epoch, "scores": {"s": {"value": value}}, **fields}


def read_samples(tmp_path: Path, *samples: dict[str, Any], **options: Any) -> EvalResults:
    return read_inspect(write_log(tmp_path / "log.json", list(samples)), **options)


class Member(NamedTuple):
    """One zip member, written field by field so that a test can set any header value."""

    name: bytes
    data: bytes  # the stored (compressed) bytes
    method: int = 0  # 0 stored, 8 deflate, 12 bzip2, 14 LZMA, 93 zstd, others unknown
    crc: int | None = None  # CRC-32 of the uncompressed bytes (default: of data)
    size: int | None = None  # uncompressed size (default: len(data))
    flags: int = 0  # general purpose flags, in both headers
    local_name: bytes | None = None  # the name in the local header (default: name)


def zip_bytes(members: list[Member], *, version: int = 20, cd_offset_shift: int = 0) -> bytes:
    """A zip archive: local headers and data, then the central directory and its end record.

    ``version`` is the version needed to extract; ``cd_offset_shift`` is added to the central
    directory offset in the end record.
    """
    body = central = b""
    for m in members:
        crc = zlib.crc32(m.data) if m.crc is None else m.crc
        size = len(m.data) if m.size is None else m.size
        local_name = m.name if m.local_name is None else m.local_name
        offset = len(body)
        body += struct.pack(
            "<4s5H3I2H", b"PK\x03\x04", 20, m.flags, m.method, 0, 0x21, crc, len(m.data), size,
            len(local_name), 0,
        )  # fmt: skip
        body += local_name + m.data
        central += struct.pack(
            "<4s6H3I5H2I", b"PK\x01\x02", 20, version, m.flags, m.method, 0, 0x21, crc,
            len(m.data), size, len(m.name), 0, 0, 0, 0, 0, offset,
        )  # fmt: skip
        central += m.name
    n = len(members)
    end = struct.pack(
        "<4s4H2IH", b"PK\x05\x06", 0, 0, n, n, len(central), len(body) + cd_offset_shift, 0
    )
    return body + central + end


HEADER = json.dumps({"version": 2, "eval": {"task": "t", "model": "m", "eval_id": "E"}}).encode()
SAMPLE = json.dumps(sample()).encode()


def stored(name: str, data: bytes) -> Member:
    return Member(name.encode(), data)


def zstd_member(name: str, data: bytes) -> Member:
    """A zstd member (compression method 93) as Inspect 0.3.206 and later write them."""
    zstandard = pytest.importorskip("zstandard")
    packed = zstandard.ZstdCompressor().compress(data)
    return Member(name.encode(), packed, method=93, crc=zlib.crc32(data), size=len(data))


# --------------------------------------------------------------------------- recorded logs


@pytest.mark.parametrize("directory", ["eval-deflate", "json", "legacy"])
def test_fixture_logs_hold_the_designed_events(directory: str) -> None:
    header, samples = raw_log(directory)
    spec = header["eval"]
    assert (spec["task"], spec["model"]) == ("tiny_qa", "mockllm/model")
    assert spec["dataset"]["sample_ids"] == list(SAMPLE_IDS)
    assert spec["config"]["epochs"] == len(EPOCHS)
    assert [s["name"] for s in spec["scorers"]] == list(SCORERS)
    assert sorted((s["id"], s["epoch"]) for s in samples) == sorted(
        (i, e) for i in SAMPLE_IDS for e in EPOCHS
    )
    for s in samples:
        key = (s["id"], s["epoch"])
        for scorer in SCORERS:
            want = raw_value(directory, scorer, *key)
            score = (s.get("scores") or {}).get(scorer)
            if want is None:
                assert score is None, (key, scorer)
            elif is_nan(want):
                assert is_nan(score["value"]) and score["reason"] == "scoring_failed"
            else:
                assert score["value"] == want and type(score["value"]) is type(want)
        assert bool(s.get("error")) == (key == ("sum", 2))
        assert bool(s.get("invalidation")) == (key == ("diff", 1))
        limit = s.get("limit")
        assert (limit is not None) == (s["id"] == "loop")
        if limit is not None:
            assert (limit["type"], limit["limit"]) == ("message", 6)


def test_zstd_fixture_holds_the_deflate_members() -> None:
    with (
        zipfile.ZipFile(only_log("eval-zstd")) as zz,
        zipfile.ZipFile(only_log("eval-deflate")) as zd,
    ):
        assert {i.compress_type for i in zz.infolist()} == {93}
        assert {i.compress_type for i in zd.infolist()} == {zipfile.ZIP_DEFLATED}
        members = [(i.filename, i.CRC, i.file_size) for i in zz.infolist()]
        assert members == [(i.filename, i.CRC, i.file_size) for i in zd.infolist()]
    with zipfile.ZipFile(only_log("legacy")) as zl:
        assert {i.compress_type for i in zl.infolist()} == {zipfile.ZIP_DEFLATED}


@pytest.mark.parametrize("scorer", [None, *SCORERS])
@pytest.mark.parametrize("directory", FORMATS)
def test_outcomes_and_statuses_follow_the_designed_events(
    directory: str, scorer: str | None
) -> None:
    res = read_inspect(FIXTURES / directory, scorer=scorer)
    name = scorer or SCORERS[0]  # the default is the first scorer
    rows = records(res)
    assert len(res) == len(rows) == len(SAMPLE_IDS) * len(EPOCHS)
    assert set(rows) == {(i, e) for i in SAMPLE_IDS for e in EPOCHS}
    for (sample_id, epoch), row in rows.items():
        value = raw_value(directory, name, sample_id, epoch)
        status, reason = designed_status(directory, name, sample_id, epoch)
        assert row["outcome"] == designed_outcome(value), (sample_id, epoch)
        assert (row["status"], row["status_reason"]) == (status, reason), (sample_id, epoch)
        assert row["scorer"] == name
        assert row["score_value"] == (None if value is None else str(value))
        assert row["task"] == f"tiny_qa/{sample_id}" and row["run"] == str(epoch)


@pytest.mark.parametrize("directory", FORMATS)
def test_every_status_occurs_in_the_fixtures(directory: str) -> None:
    res = read_inspect(FIXTURES / directory, scorer="closeness")
    assert set(res.frame["status"]) == set(STATUSES) == {"ok", "infra_error", "timeout", "flagged"}
    reasons = {"invalidated", "error", "limit:message"}
    if directory != "legacy":  # Inspect 0.3.205 had no Score.unscored
        reasons.add("unscored:scoring_failed")
    assert set(res.frame["status_reason"].dropna()) == reasons
    assert int(res.frame["status_reason"].isna().sum()) == (res.frame["status"] == "ok").sum()


@pytest.mark.parametrize("directory", FORMATS)
def test_covariates_come_from_the_raw_samples(directory: str) -> None:
    header, samples = raw_log(directory)
    spec = header["eval"]
    res = read_inspect(FIXTURES / directory, scorer="closeness")
    assert list(res.frame.columns) == ["model", "task", "run", "outcome", *COVARIATES]
    assert res.covariates == list(COVARIATES)
    numeric = ["total_time", "working_time", *TOKENS, "retries"]
    assert all(res.frame[c].dtype == "float64" for c in numeric)
    rows = records(res)
    for s in samples:
        row = rows[(s["id"], s["epoch"])]
        assert row["model"] == spec["model"] and row["task_name"] == spec["task"]
        assert row["sample_id"] == s["id"] and row["eval_id"] == spec["eval_id"]
        assert row["log"] == only_log(directory).name
        assert row["total_time"] == s["total_time"]
        assert row["working_time"] == s["working_time"]
        assert row["retries"] == len(s["error_retries"])
        for key in TOKENS:
            counts = [usage[key] for usage in s["model_usage"].values() if key in usage]
            assert row[key] == (sum(counts) if counts else None), (s["id"], s["epoch"], key)
        score = (s.get("scores") or {}).get("closeness")
        assert row["score_value"] == (str(score["value"]) if score else None)


def test_legacy_mock_reports_no_tokens_and_current_mock_does() -> None:
    legacy = read_inspect(FIXTURES / "legacy").frame
    assert legacy[list(TOKENS)].isna().all().all()
    current = records(read_inspect(FIXTURES / "json"))
    _, samples = raw_log("json")
    usage = {(s["id"], s["epoch"]): s["model_usage"] for s in samples}
    assert usage[("sum", 2)] == {}  # the errored sample made no model call
    assert current[("sum", 2)]["input_tokens"] is None
    assert (
        current[("diff", 3)]["total_tokens"] == usage[("diff", 3)]["mockllm/model"]["total_tokens"]
    )


@pytest.mark.parametrize("directory", FORMATS)
def test_models_tasks_and_runs(directory: str) -> None:
    header, _ = raw_log(directory)
    spec = header["eval"]
    ids, epochs = spec["dataset"]["sample_ids"], spec["config"]["epochs"]
    res = read_inspect(FIXTURES / directory)
    assert res.models == [spec["model"]]
    assert sorted(res.tasks) == sorted(f"{spec['task']}/{i}" for i in ids)
    assert res.run_counts(spec["model"]).tolist() == [epochs] * len(ids)
    assert sorted(set(res.frame["run"])) == [str(e) for e in range(1, epochs + 1)]
    by_eval = read_inspect(FIXTURES / directory, run_key="eval_epoch")
    runs = sorted(set(by_eval.frame["run"]))
    assert runs == [f"{spec['eval_id']}/{e}" for e in range(1, epochs + 1)]


@pytest.mark.parametrize(
    ("scorer", "binary"),
    [("match", True), ("letter_grade", False), ("closeness", False), ("has_answer", True)],
)
def test_is_binary_follows_the_scorer(scorer: str, binary: bool) -> None:
    # letter_grade gives product P (0.5); closeness gives diff epoch 1 a 0.5
    for directory in FORMATS:
        assert read_inspect(FIXTURES / directory, scorer=scorer).is_binary is binary


@pytest.mark.parametrize("directory", FORMATS)
def test_a_single_log_file_reads_like_its_directory(directory: str) -> None:
    path = only_log(directory)
    rows = records(read_inspect(str(path)))  # a str path works as well as a Path
    assert set(rows) == {(i, e) for i in SAMPLE_IDS for e in EPOCHS}
    for key, row in rows.items():
        assert (row["status"], row["status_reason"]) == designed_status(directory, "match", *key)
        assert row["outcome"] == designed_outcome(raw_value(directory, "match", *key))
        assert row["log"] == path.name


def test_unknown_scorer_lists_the_recorded_ones() -> None:
    name = re.escape(only_log("json").name)
    with pytest.raises(SchemaError, match=rf"{name}: scorer 'nope' not found; available: "):
        read_inspect(FIXTURES / "json", scorer="nope")
    with pytest.raises(SchemaError, match="available: " + ", ".join(SCORERS) + "$"):
        read_inspect(FIXTURES / "legacy", scorer="nope")


def test_the_fixture_tree_holds_one_run_twice() -> None:
    # all four logs share model, task, and epoch; eval-deflate and eval-zstd share an eval id
    first, second = (re.escape(f"{d}/{only_log(d).name}") for d in ("eval-deflate", "eval-zstd"))
    clash = rf"appears in both {first} and {second}; read one log, or pass run_key='eval_epoch'"
    with pytest.raises(SchemaError, match=clash):
        read_inspect(FIXTURES)
    with pytest.raises(SchemaError, match=rf"appears in both {first} and {second}"):
        read_inspect(FIXTURES, run_key="eval_epoch")


def test_directories_are_searched_recursively(tmp_path: Path) -> None:
    runs = ("eval-deflate", "json", "legacy")  # three separate runs of tiny_qa
    for directory in runs:
        (tmp_path / "runs" / directory).mkdir(parents=True)
        shutil.copy(only_log(directory), tmp_path / "runs" / directory)
    listing = {only_log("json").name: {"version": 2, "status": "success"}}  # as logs.json
    (tmp_path / "runs" / "logs.json").write_text(json.dumps(listing))
    (tmp_path / "runs" / "notes.txt").write_text("not a log")
    eval_ids = {raw_log(d)[0]["eval"]["eval_id"] for d in runs}
    assert len(eval_ids) == len(runs)
    res = read_inspect(tmp_path, run_key="eval_epoch")
    assert len(res) == len(runs) * len(SAMPLE_IDS) * len(EPOCHS)
    assert set(res.frame["log"]) == {f"runs/{d}/{only_log(d).name}" for d in runs}
    assert set(res.frame["eval_id"]) == eval_ids
    assert res.run_counts("mockllm/model").tolist() == [len(runs) * len(EPOCHS)] * 4


def test_an_in_progress_eval_log_reads_from_its_journal(tmp_path: Path) -> None:
    kept = {("sum", 1), ("diff", 2), ("loop", 3)}
    path = tmp_path / "running.eval"
    with (
        zipfile.ZipFile(only_log("eval-deflate")) as zin,
        zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zout,
    ):
        zout.writestr("_journal/start.json", zin.read("_journal/start.json"))
        for sample_id, epoch in sorted(kept):
            name = f"samples/{sample_id}_epoch_{epoch}.json"
            zout.writestr(name, zin.read(name))
    rows = records(read_inspect(path))
    assert set(rows) == {(i, e) for i in SAMPLE_IDS for e in EPOCHS}
    for key, row in rows.items():
        if key in kept:
            status = designed_status("eval-deflate", "match", *key)
            outcome = designed_outcome(raw_value("eval-deflate", "match", *key))
        else:
            status, outcome = ("infra_error", "missing"), 0.0
        assert (row["status"], row["status_reason"], row["outcome"]) == (*status, outcome), key
    assert len(read_inspect(path, include_missing=False)) == len(kept)


def test_sample_summaries_give_the_same_error_and_limit_statuses(tmp_path: Path) -> None:
    # noisefloor.reanalysis.epoch reads header.json plus summaries.json as one JSON log
    with zipfile.ZipFile(only_log("eval-deflate")) as zf:
        header = json.loads(zf.read("header.json"))
        summaries = json.loads(zf.read("summaries.json"))
    errored = next(s for s in summaries if (s["id"], s["epoch"]) == ("sum", 2))
    assert isinstance(errored["error"], str)  # summaries hold the error and limit as text
    assert {s["limit"] for s in summaries if "limit" in s} == {"message"}
    assert not any("invalidation" in s for s in summaries)  # and no invalidation record
    path = tmp_path / "assembled.json"
    path.write_text(json.dumps({**header, "samples": summaries}))
    rows = records(read_inspect(path))
    assert set(rows) == {(i, e) for i in SAMPLE_IDS for e in EPOCHS}
    for key, row in rows.items():
        if key == ("diff", 1):
            continue  # invalidated, but a summary cannot say so
        status = designed_status("eval-deflate", "match", *key)
        assert (row["status"], row["status_reason"]) == status, key
        assert row["outcome"] == designed_outcome(raw_value("eval-deflate", "match", *key))


@pytest.fixture
def stdlib_zipfile(monkeypatch: pytest.MonkeyPatch) -> None:
    """zipfile as Python 3.11 to 3.13 ship it: zstd members raise NotImplementedError.

    Undoes zipfile_zstd's process-wide patch (imported with inspect_ai, for example by
    tests/test_inspect_harness.py), so these tests reach the adapter's own zstd path.
    """
    current = zipfile._get_decompressor  # type: ignore[attr-defined]  # private stdlib helper

    def without_zstd(compress_type: int) -> Any:
        if compress_type == adapter._ZIP_ZSTANDARD:
            raise NotImplementedError("That compression method is not supported")
        return current(compress_type)

    monkeypatch.setattr(zipfile, "_get_decompressor", without_zstd)


@pytest.mark.skipif(NATIVE_ZSTD, reason="Python 3.14 reads zstd members without zstandard")
@pytest.mark.usefixtures("stdlib_zipfile")
def test_zstd_logs_need_the_zstandard_package(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = importlib.import_module

    def without_zstandard(name: str, package: str | None = None) -> Any:
        if name == "zstandard":
            raise ModuleNotFoundError("No module named 'zstandard'")
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", without_zstandard)
    message = (
        r"\.eval:header\.json: this \.eval log uses zstd compression \(Inspect 0\.3\.206 and "
        r"later\), which needs Python 3\.14 or the zstandard package \(pip install zstandard\); "
        r"or convert it first with `inspect log convert --to json`"
    )
    with pytest.raises(SchemaError, match=message):
        read_inspect(FIXTURES / "eval-zstd")
    deflate = read_inspect(FIXTURES / "eval-deflate")  # deflate members need no package
    assert len(deflate) == len(SAMPLE_IDS) * len(EPOCHS)


# --------------------------------------------------------------------------- score values


@pytest.mark.parametrize(
    ("value", "outcome"),
    [
        ("C", 1.0),
        ("P", 0.5),
        ("I", 0.0),
        ("N", 0.0),
        (True, 1.0),
        (False, 0.0),
        (1, 1.0),
        (0, 0.0),
        (0.25, 0.25),
        ("yes", 1.0),
        (" True ", 1.0),
        ("NO", 0.0),
        ("false", 0.0),
        ("0.75", 0.75),
        (" 1 ", 1.0),
    ],
)
def test_score_values_map_to_outcomes(tmp_path: Path, value: Any, outcome: float) -> None:
    [row] = records(read_samples(tmp_path, sample(value=value))).values()
    assert (row["outcome"], row["status"], row["status_reason"]) == (outcome, "ok", None)
    assert row["score_value"] == str(value) and row["scorer"] == "s"


def test_value_map_replaces_the_letter_table(tmp_path: Path) -> None:
    strict = {"C": 1.0, "P": 0.0, "I": 0.0, "N": 0.0}  # partial credit counts as failure
    res = read_samples(tmp_path, sample("a", value="P"), sample("b", value="C"), value_map=strict)
    assert res.frame.set_index("sample_id")["outcome"].to_dict() == {"a": 0.0, "b": 1.0}
    custom = read_samples(tmp_path, sample(value="PASS"), value_map={"PASS": 1.0})
    assert custom.frame["outcome"].tolist() == [1.0]
    with pytest.raises(SchemaError, match=r"cannot convert score value 'C'; pass value_map"):
        read_samples(tmp_path, sample(value="C"), value_map={"PASS": 1.0})


def test_default_letter_table_and_grader_failures() -> None:
    assert dict(adapter.INSPECT_VALUE_MAP) == LETTERS
    table: Any = adapter.INSPECT_VALUE_MAP
    with pytest.raises(TypeError):
        table["C"] = 0.0  # read-only, so no caller can change the defaults
    assert set(adapter.GRADER_FAILURES) == {"grader_failed", "scoring_failed"}


def test_dict_scores_need_value_key(tmp_path: Path) -> None:
    value = {"pass": "C", "seconds": 3}
    res = read_samples(tmp_path, sample(value=value), value_key="pass")
    assert res.frame["outcome"].tolist() == [1.0]
    where = r"log\.json: sample a epoch 1: "
    with pytest.raises(SchemaError, match=where + r"score value is a dict \(pass, seconds\)"):
        read_samples(tmp_path, sample(value=value))
    with pytest.raises(SchemaError, match=where + "score value has no key 'grade'"):
        read_samples(tmp_path, sample(value=value), value_key="grade")
    with pytest.raises(SchemaError, match=where + "unsupported score value"):
        read_samples(tmp_path, sample(value={"pass": {"nested": 1}}), value_key="pass")


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (["C", "I"], r"unsupported score value \['C', 'I'\]"),
        ("maybe", r"cannot convert score value 'maybe'; pass value_map"),
        (1.5, r"outcome 1\.5 is outside \[0, 1\]"),
        (-0.5, r"outcome -0\.5 is outside \[0, 1\]"),
        ("2", r"outcome 2\.0 is outside \[0, 1\]"),
        ("inf", r"outcome inf is outside \[0, 1\]"),
        (math.inf, r"outcome inf is outside \[0, 1\]"),
    ],
)
def test_unmappable_values_raise(tmp_path: Path, value: Any, message: str) -> None:
    with pytest.raises(SchemaError, match=r"log\.json: sample a epoch 1: " + message):
        read_samples(tmp_path, sample(value=value))


@pytest.mark.parametrize(
    ("score", "status", "reason"),
    [
        ({"value": NAN, "reason": "grader_failed"}, "infra_error", "unscored:grader_failed"),
        ({"value": NAN, "reason": "scoring_failed"}, "infra_error", "unscored:scoring_failed"),
        ({"value": NAN, "reason": "refusal"}, "flagged", "unscored:refusal"),
        ({"value": NAN, "reason": "custom text"}, "flagged", "unscored:custom text"),
        ({"value": NAN}, "flagged", "unscored"),
        ({"value": None}, "flagged", "unscored"),
        ({"value": "NaN"}, "flagged", "unscored"),
        (
            {"value": NAN, "metadata": {"unscored_reason": "grader_failed"}},
            "infra_error",
            "unscored:grader_failed",
        ),
        (
            {"value": NAN, "metadata": {"unscored_reason": "no_response"}},
            "flagged",
            "unscored:no_response",
        ),
        (
            {"value": {"pass": NAN}, "reason": "grader_failed"},
            "infra_error",
            "unscored:grader_failed",
        ),
    ],
)
def test_unscored_samples(tmp_path: Path, score: dict[str, Any], status: str, reason: str) -> None:
    res = read_samples(tmp_path, sample(scores={"s": score}), value_key="pass")
    [row] = records(res).values()
    assert (row["status"], row["status_reason"], row["outcome"]) == (status, reason, 0.0)


# --------------------------------------------------------------------------- status mapping


@pytest.mark.parametrize(
    ("fields", "value", "status", "reason", "outcome"),
    [
        (
            {"invalidation": {"author": "x"}, "error": {"message": "boom"}, "limit": "time"},
            "C",
            "flagged",
            "invalidated",
            1.0,
        ),
        (
            {"error": {"message": "boom"}, "limit": {"type": "time"}},
            "C",
            "infra_error",
            "error",
            1.0,
        ),
        ({"error": "boom"}, "I", "infra_error", "error", 0.0),  # a summary's error is text
        ({"limit": {"type": "working", "limit": 60}}, "C", "timeout", "limit:working", 1.0),
        ({"limit": "token"}, "P", "timeout", "limit:token", 0.5),  # a summary's limit is text
        ({"limit": {"type": "time"}}, NAN, "timeout", "limit:time", 0.0),
        ({"error": None, "limit": None, "invalidation": None}, "C", "ok", None, 1.0),
    ],
)
def test_status_precedence_and_kept_scores(
    tmp_path: Path,
    fields: dict[str, Any],
    value: Any,
    status: str,
    reason: str | None,
    outcome: float,
) -> None:
    [row] = records(read_samples(tmp_path, sample(value=value, **fields))).values()
    assert (row["status"], row["status_reason"], row["outcome"]) == (status, reason, outcome)


def test_samples_without_the_chosen_score(tmp_path: Path) -> None:
    samples = [
        sample("a", scores={"s": {"value": "C"}, "t": {"value": "I"}}),
        sample("b", scores={"t": {"value": "C"}}),
        sample("c", scores={}),
        sample("d", scores=None),
        sample("e", scores={"s": None}),
    ]
    by_s = records(read_samples(tmp_path, *samples))  # s is the first scorer of sample a
    assert by_s[("a", 1)]["outcome"] == 1.0 and by_s[("a", 1)]["status"] == "ok"
    for sample_id in "bcde":
        row = by_s[(sample_id, 1)]
        assert (row["status"], row["status_reason"], row["outcome"]) == (
            "infra_error",
            "no_score",
            0.0,
        )
        assert row["score_value"] is None and row["scorer"] == "s"
    by_t = records(read_samples(tmp_path, *samples, scorer="t"))
    assert [by_t[(i, 1)]["outcome"] for i in "ab"] == [0.0, 1.0]
    assert by_t[("c", 1)]["status_reason"] == "no_score"


def test_default_scorer_is_the_first_of_the_first_scored_sample(tmp_path: Path) -> None:
    first = sample("a", scores={}, error={"message": "boom"})
    second = sample("b", scores={"z": {"value": "C"}, "y": {"value": "I"}})
    res = read_samples(tmp_path, first, second)
    assert set(res.frame["scorer"]) == {"z"}
    assert res.frame.set_index("sample_id")["outcome"].to_dict() == {"a": 0.0, "b": 1.0}


def test_default_scorer_falls_back_to_the_eval_spec(tmp_path: Path) -> None:
    errored = sample(scores={}, error={"message": "boom"})
    scorers = [{"name": "inspect_evals/swe_bench_scorer"}, {"name": "other"}]
    path = write_log(tmp_path / "spec.json", [errored], scorers=scorers)
    [row] = records(read_inspect(path)).values()
    assert (row["scorer"], row["status"], row["status_reason"]) == (
        "swe_bench_scorer",
        "infra_error",
        "error",
    )
    path = write_log(tmp_path / "none.json", [sample(scores=None)])  # no scorer anywhere
    [row] = records(read_inspect(path)).values()
    assert (row["scorer"], row["status"], row["status_reason"]) == (None, "infra_error", "no_score")
    # a named scorer cannot be checked when no sample has scores
    [row] = records(read_inspect(path, scorer="s")).values()
    assert (row["scorer"], row["status_reason"]) == ("s", "no_score")


def test_missing_samples_become_infra_errors(tmp_path: Path) -> None:
    dataset, config = {"sample_ids": ["a", "b"]}, {"epochs": 2}
    path = write_log(
        tmp_path / "log.json", [sample("a", 1), sample("b", 2)], dataset=dataset, config=config
    )
    rows = records(read_inspect(path))
    assert set(rows) == {("a", 1), ("b", 2), ("b", 1), ("a", 2)}
    for key in (("b", 1), ("a", 2)):
        row = rows[key]
        assert (row["status"], row["status_reason"], row["outcome"]) == (
            "infra_error",
            "missing",
            0.0,
        )
        assert (row["scorer"], row["task"], row["task_name"]) == ("s", f"t/{key[0]}", "t")
        assert row["score_value"] is None and row["total_time"] is None
    assert set(records(read_inspect(path, include_missing=False))) == {("a", 1), ("b", 2)}
    by_eval = read_inspect(path, run_key="eval_epoch")
    assert sorted(by_eval.frame["run"]) == ["E/1", "E/1", "E/2", "E/2"]


def test_early_stopped_samples_are_not_missing(tmp_path: Path) -> None:
    stops = [{"id": "a", "epoch": 2, "reason": "converged"}, {"id": 7, "epoch": 1}]
    results = {"early_stopping": {"manager": "m", "early_stops": stops, "metadata": {}}}
    path = write_log(
        tmp_path / "log.json",
        [sample("a", 1)],
        results=results,
        dataset={"sample_ids": ["a", "b", 7]},
        config={"epochs": 2},
    )
    rows = records(read_inspect(path))
    missing = sorted(k for k, row in rows.items() if row["status_reason"] == "missing")
    assert missing == [("7", 2), ("b", 1), ("b", 2)]


@pytest.mark.parametrize(
    "results",
    [
        {"early_stopping": None},
        {"early_stopping": {"early_stops": None}},
        {"early_stopping": {"early_stops": "a/2"}},
        ["not", "an", "object"],
    ],
)
def test_logs_without_early_stops(tmp_path: Path, results: Any) -> None:
    path = write_log(
        tmp_path / "log.json",
        [sample("a", 1)],
        results=results,
        dataset={"sample_ids": ["a"]},
        config={"epochs": 2},
    )
    assert set(records(read_inspect(path))) == {("a", 1), ("a", 2)}


@pytest.mark.parametrize(
    "stop", [{"id": "a"}, {"id": "a", "epoch": None}, {"id": "a", "epoch": "two"}, "a/2"]
)
def test_malformed_early_stops_raise(tmp_path: Path, stop: Any) -> None:
    results = {"early_stopping": {"manager": "m", "early_stops": [stop], "metadata": {}}}
    path = write_log(
        tmp_path / "log.json",
        [sample("a", 1)],
        results=results,
        dataset={"sample_ids": ["a"]},
        config={"epochs": 2},
    )
    with pytest.raises(SchemaError, match=r"log\.json: results\.early_stopping .*integer epoch"):
        read_inspect(path)
    assert len(read_inspect(path, include_missing=False)) == 1


@pytest.mark.parametrize(
    ("config", "epochs"), [(None, 1), ({}, 1), ({"epochs": 0}, 1), ({"epochs": 3}, 3)]
)
def test_integer_ids_and_default_epochs(tmp_path: Path, config: Any, epochs: int) -> None:
    path = write_log(
        tmp_path / "log.json", [sample(1, 1)], dataset={"sample_ids": [1, 2]}, config=config
    )
    rows = records(read_inspect(path))
    assert set(rows) == {(i, e) for i in ("1", "2") for e in range(1, epochs + 1)}
    assert rows[("1", 1)]["task"] == "t/1" and rows[("1", 1)]["status"] == "ok"
    assert {r["status_reason"] for k, r in rows.items() if k != ("1", 1)} <= {"missing"}


def test_no_missing_rows_without_recorded_sample_ids(tmp_path: Path) -> None:
    for dataset in (None, {"samples": 4}, {"sample_ids": "sum"}):
        path = write_log(tmp_path / "log.json", [sample("a", 1)], dataset=dataset)
        assert len(read_inspect(path)) == 1


def test_token_usage_is_summed_over_models(tmp_path: Path) -> None:
    usage = {
        "solver/model": {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
        "grader/model": {"input_tokens": 5, "output_tokens": 1, "total_tokens": None},
        "broken": "not an object",
    }
    retries = [{"error": "rate limit"}, {"error": "rate limit"}]
    timed = sample(model_usage=usage, error_retries=retries, total_time=2.5, working_time=1.5)
    [row] = records(read_samples(tmp_path, timed)).values()
    assert [row[k] for k in TOKENS] == [15.0, 5.0, 14.0]
    assert (row["retries"], row["total_time"], row["working_time"]) == (2.0, 2.5, 1.5)
    [bare] = records(read_samples(tmp_path, sample(model_usage="none"))).values()
    assert [bare[k] for k in (*TOKENS, "retries", "total_time")] == [None] * 5


def test_eval_epoch_runs_fall_back_to_the_log_name(tmp_path: Path) -> None:
    path = write_log(tmp_path / "log.json", [sample("a", 2)], eval_id=None)
    [row] = records(read_inspect(path, run_key="eval_epoch")).values()
    assert (row["run"], row["eval_id"]) == ("log.json/2", "log.json")


def test_run_key_must_be_epoch_or_eval_epoch(tmp_path: Path) -> None:
    with pytest.raises(
        ValueError, match="run_key must be 'epoch' or 'eval_epoch', not 'sample'"
    ) as info:
        read_inspect(tmp_path, run_key="sample")
    assert type(info.value) is ValueError


# --------------------------------------------------------------------------- malformed input


def test_missing_and_empty_inputs(tmp_path: Path) -> None:
    with pytest.raises(SchemaError, match=r"absent\.eval: no such file or directory"):
        read_inspect(tmp_path / "absent.eval")
    with pytest.raises(SchemaError, match="no Inspect logs found"):
        read_inspect(tmp_path)
    (tmp_path / "logs.json").write_text(json.dumps({"x.eval": {"version": 2}}))
    (tmp_path / "notes.txt").write_text("not a log")
    with pytest.raises(SchemaError, match="no Inspect logs found"):
        read_inspect(tmp_path)
    empty = write_log(tmp_path / "empty" / "log.json", [])
    with pytest.raises(SchemaError, match=r"log\.json: no runs found"):
        read_inspect(empty)
    nulls = write_log(tmp_path / "nulls" / "log.json", [])
    nulls.write_text(nulls.read_text().replace('"samples": []', '"samples": null'))
    with pytest.raises(SchemaError, match="no runs found"):
        read_inspect(nulls)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        pytest.param(b"{", "invalid JSON", id="truncated"),
        pytest.param(b'{"version": 2, "eval": "\xff"}', "invalid JSON", id="not-utf8"),
        pytest.param(
            b'{"version": 2}',
            r"not an Inspect eval log \(no 'eval' and 'version' keys\)",
            id="no-eval",
        ),
        pytest.param(
            b'{"eval": {"task": "t", "model": "m"}}', "not an Inspect eval log", id="no-version"
        ),
        pytest.param(b'{"version": 2, "eval": "t"}', "not an Inspect eval log", id="eval-text"),
        pytest.param(b"[1, 2]", "not an Inspect eval log", id="list"),
        pytest.param(
            b'{"version": 2, "eval": {}, "samples": {"a": 1}}',
            "'samples' is not a list",
            id="samples-object",
        ),
        pytest.param(
            b'{"version": 2, "eval": {}, "samples": ["a"]}',
            "sample: expected a JSON object, got str",
            id="sample-text",
        ),
        pytest.param(
            b'{"version": 2, "eval": {"task": "t", "model": 3}}',
            "eval.model and eval.task must be strings",
            id="model-number",
        ),
        pytest.param(
            b'{"version": 2, "eval": {"model": "m"}}',
            "eval.model and eval.task must be strings",
            id="no-task",
        ),
    ],
)
def test_malformed_json_logs(tmp_path: Path, text: bytes, message: str) -> None:
    path = tmp_path / "log.json"
    path.write_bytes(text)
    with pytest.raises(SchemaError, match=message):
        read_inspect(path)


def test_a_broken_json_file_in_a_directory_is_not_skipped(tmp_path: Path) -> None:
    write_log(tmp_path / "a.json", [sample()])
    (tmp_path / "b.json").write_text('{"version": 2, "eval": ')
    with pytest.raises(SchemaError, match=r"b\.json: invalid JSON"):
        read_inspect(tmp_path)


@pytest.mark.parametrize(
    "bad",
    [
        {"epoch": 1},
        {"id": None, "epoch": 1},
        {"id": "a"},
        {"id": "a", "epoch": "1"},
        {"id": "a", "epoch": 1.0},
    ],
)
def test_samples_need_an_id_and_an_integer_epoch(tmp_path: Path, bad: dict[str, Any]) -> None:
    with pytest.raises(SchemaError, match=r"log\.json: sample without an id or integer epoch"):
        read_samples(tmp_path, bad)


def test_a_score_that_is_not_an_object_raises(tmp_path: Path) -> None:
    with pytest.raises(
        SchemaError, match=r"log\.json: sample a epoch 1: score 's': expected a JSON object"
    ):
        read_samples(tmp_path, sample(scores={"s": "C"}))
    # another scorer's malformed entry does not matter when it is not the chosen one
    res = read_samples(tmp_path, sample(scores={"s": {"value": "C"}, "t": "C"}))
    assert res.frame["outcome"].tolist() == [1.0]


def test_the_same_sample_twice_in_one_log_raises(tmp_path: Path) -> None:
    with pytest.raises(SchemaError, match=r"log\.json: sample a epoch 1 appears twice"):
        read_samples(tmp_path, sample("a", 1), sample("a", 1, value="I"))
    with pytest.raises(SchemaError, match=r"log\.json: sample 1 epoch 2 appears twice"):
        read_samples(tmp_path, sample(1, 2), sample("1", 2))


def test_the_same_run_in_two_logs_raises(tmp_path: Path) -> None:
    write_log(tmp_path / "x.json", [sample("a", 1)], eval_id="E1")
    write_log(tmp_path / "y.json", [sample("a", 1)], eval_id="E2")
    clash = r"t/a run 1 of m appears in both x\.json and y\.json; read one log, or pass run_key"
    with pytest.raises(SchemaError, match=clash):
        read_inspect(tmp_path)
    assert sorted(read_inspect(tmp_path, run_key="eval_epoch").frame["run"]) == ["E1/1", "E2/1"]


def test_a_stored_zip_log_reads(tmp_path: Path) -> None:
    path = tmp_path / "log.eval"
    path.write_bytes(zip_bytes([stored("header.json", HEADER), stored("samples/a_1.json", SAMPLE)]))
    [row] = records(read_inspect(path)).values()
    assert (row["outcome"], row["status"], row["log"], row["eval_id"]) == (
        1.0,
        "ok",
        "log.eval",
        "E",
    )


def test_repeated_zip_members_last_copy_wins(tmp_path: Path) -> None:
    first = json.dumps(sample(value="I")).encode()
    other = json.dumps({"version": 2, "eval": {"task": "x", "model": "y"}}).encode()
    members = [
        stored("header.json", other),
        stored("samples/a_epoch_1.json", first),
        stored("samples/a_epoch_1.json", SAMPLE),
        stored("header.json", HEADER),
    ]
    path = tmp_path / "log.eval"
    path.write_bytes(zip_bytes(members))
    [row] = records(read_inspect(path)).values()
    assert (row["model"], row["task"], row["outcome"]) == ("m", "t/a", 1.0)


@pytest.mark.parametrize(
    ("members", "message"),
    [
        pytest.param(
            [stored("header.json", HEADER), stored("samples/a_epoch_1/sample.json", SAMPLE)],
            r"chunked sample layout \(samples/a_epoch_1/sample\.json\) is not supported",
            id="chunked",
        ),
        pytest.param(
            [stored("samples/a_epoch_1.json", SAMPLE)],
            r"no header\.json or _journal/start\.json; not an Inspect log",
            id="no-header",
        ),
        pytest.param(
            [stored("header.json", b'{"version": 2}')],
            r"log\.eval:header\.json: not an Inspect log header",
            id="header-not-log",
        ),
        pytest.param(
            [stored("_journal/start.json", b"[]")],
            r"log\.eval:_journal/start\.json: not an Inspect log header",
            id="journal-not-log",
        ),
        pytest.param(
            [stored("header.json", b"{")], r"log\.eval:header\.json: invalid JSON", id="header-json"
        ),
        pytest.param(
            [stored("header.json", HEADER), stored("samples/a_epoch_1.json", b"[]")],
            r"samples/a_epoch_1\.json: expected a JSON object, got list",
            id="sample-list",
        ),
        pytest.param(
            [stored("header.json", HEADER), stored("samples/a_epoch_1.json", b"{")],
            r"samples/a_epoch_1\.json: invalid JSON",
            id="sample-json",
        ),
    ],
)
def test_eval_layout_errors(tmp_path: Path, members: list[Member], message: str) -> None:
    path = tmp_path / "log.eval"
    path.write_bytes(zip_bytes(members))
    with pytest.raises(SchemaError, match=message):
        read_inspect(path)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        pytest.param(b"", "File is not a zip file", id="empty"),
        pytest.param(b"PK\x03\x04 and then nothing", "File is not a zip file", id="truncated"),
        pytest.param(
            zip_bytes([stored("header.json", HEADER)], version=80),
            r"zip file version 8\.0",
            id="zip-version",
        ),
        pytest.param(
            zip_bytes([Member(b"header\xff.json", HEADER, flags=0x800)]),
            "can't decode byte 0xff",
            id="name-not-utf8",
        ),
    ],
)
def test_unreadable_zip_archives(tmp_path: Path, data: bytes, message: str) -> None:
    path = tmp_path / "log.eval"
    path.write_bytes(data)
    with pytest.raises(SchemaError, match=r"log\.eval: not a readable \.eval zip \(.*" + message):
        read_inspect(path)


LZMA_BAD_OPTIONS = b"\x09\x14\x05\x00" + b"\xff" * 5 + b"abc"  # zipfile's LZMA header


@pytest.mark.parametrize(
    ("member", "shift", "message"),
    [
        pytest.param(
            Member(b"header.json", b"\xff\xff", method=8, size=9),
            0,
            "Error -3 while decompressing",
            id="deflate-data",
        ),
        pytest.param(Member(b"header.json", HEADER, crc=1), 0, "Bad CRC-32", id="crc"),
        pytest.param(
            Member(b"header.json", b"garbage!", method=12, size=9),
            0,
            "Invalid data stream",
            id="bzip2-data",
        ),
        pytest.param(
            Member(b"header.json", LZMA_BAD_OPTIONS, method=14, size=9),
            0,
            "unsupported options",
            id="lzma-data",
        ),
        pytest.param(
            Member(b"header.json", HEADER, flags=0x1),
            0,
            "encrypted, password required",
            id="encryption-flag",
        ),
        pytest.param(
            Member(b"header.json", HEADER, flags=0x800, local_name=b"header\xff.json"),
            0,
            "can't decode byte 0xff",
            id="local-name-not-utf8",
        ),
        pytest.param(Member(b"header.json", HEADER), 100, "Invalid argument", id="bad-offset"),
    ],
)
def test_corrupt_zip_members(tmp_path: Path, member: Member, shift: int, message: str) -> None:
    path = tmp_path / "log.eval"
    path.write_bytes(zip_bytes([member], cd_offset_shift=shift))
    where = r"log\.eval:header\.json: corrupt zip member \(.*"
    with pytest.raises(SchemaError, match=where + message):
        read_inspect(path)


def test_unknown_compression_methods_are_named(tmp_path: Path) -> None:
    path = tmp_path / "log.eval"
    path.write_bytes(zip_bytes([Member(b"header.json", HEADER, method=99)]))
    with pytest.raises(SchemaError, match=r"header\.json: unsupported zip compression method 99"):
        read_inspect(path)


@pytest.mark.skipif(NATIVE_ZSTD, reason="Python 3.14 reads zstd members without zstandard")
@pytest.mark.usefixtures("stdlib_zipfile")
def test_zstd_members_are_decompressed_and_checked(tmp_path: Path) -> None:
    good = [zstd_member("header.json", HEADER), zstd_member("samples/a_epoch_1.json", SAMPLE)]
    path = tmp_path / "log.eval"
    path.write_bytes(zip_bytes(good))
    assert read_inspect(path).frame["outcome"].tolist() == [1.0]
    for member, message in [
        (good[0]._replace(data=b"not zstd data"), "corrupt zstd data"),
        (good[0]._replace(crc=1), "zstd member failed its size or CRC check"),
        (good[0]._replace(size=len(HEADER) + 1), "zstd member failed its size or CRC check"),
    ]:
        path.write_bytes(zip_bytes([member]))
        with pytest.raises(SchemaError, match=r"log\.eval:header\.json: " + message):
            read_inspect(path)


def test_zstd_member_header_checks(tmp_path: Path) -> None:
    # zipfile checks the local header before it gives up on method 93, so these guards are
    # reached only by a direct call
    pytest.importorskip("zstandard")
    path = tmp_path / "log.eval"
    path.write_bytes(b"this is not a local file header" * 2)
    info = zipfile.ZipInfo("header.json")
    info.header_offset = 0
    with pytest.raises(SchemaError, match=r"log\.eval:header\.json: bad zip member header"):
        adapter._read_zstd_member(path, info)
    info.header_offset = path.stat().st_size - 10
    with pytest.raises(SchemaError, match=r"log\.eval:header\.json: truncated zip member header"):
        adapter._read_zstd_member(path, info)
