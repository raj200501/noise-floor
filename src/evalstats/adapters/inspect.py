"""Reader for Inspect AI eval logs in the ``.eval`` (zip) and ``.json`` formats.

The files are parsed directly with the standard library; ``inspect_ai`` is not imported.

Files read. A ``.json`` log is one object with ``version``, ``status``, ``eval`` (task, model,
config, dataset, scorers), ``results``, and ``samples``. A ``.eval`` log is a zip that holds the
same header in ``header.json`` (only ``_journal/start.json`` while a run is in progress) and one
``samples/<id>_epoch_<n>.json`` member per sample; when a member name repeats, the last copy
wins, as in every zip reader. Inspect 0.3.205 and earlier wrote deflate members, which always
work. Inspect 0.3.206 and later write zstd members: the standard library reads them from
Python 3.14, and on earlier Pythons this reader uses the ``zstandard`` package when it can be
imported (``inspect_ai`` depends on it) and otherwise raises a :class:`SchemaError` saying what
to install. A directory is searched recursively for ``*.eval`` and ``*.json`` files; JSON files
that are not eval logs (such as Inspect's ``logs.json`` listing) are skipped.

Rows. One row per sample and epoch. ``model`` is the log's ``eval.model``, ``task`` is
``<eval.task>/<sample id>``, ``run`` is the epoch (``<eval_id>/<epoch>`` with
``run_key="eval_epoch"``, which keeps repeated logs of the same task apart), and ``outcome`` is
the value of the chosen scorer (default: the first scorer, taken from the key order of the
first scored sample).

Score values. Outcomes follow Inspect's ``value_to_float`` defaults: ``C`` (correct) 1,
``P`` (partial) 0.5, ``I`` (incorrect) 0, ``N`` (no answer) 0, booleans 1 and 0, numbers
unchanged, the strings ``yes`` and ``true`` 1, ``no`` and ``false`` 0, and numeric strings as
numbers. ``value_map`` replaces the letter table (for example ``{"C": 1, "P": 0, "I": 0,
"N": 0}`` counts partial credit as failure). Dict-valued scores need ``value_key``. A value
outside [0, 1], a list, or a string that cannot be converted raises :class:`SchemaError`
instead of becoming 0 as in Inspect.

Status mapping, first match wins (``status_reason`` in brackets):

1. ``sample.invalidation`` is set (``inspect_ai.log.invalidate_samples``): ``flagged``
   [``invalidated``].
2. ``sample.error`` is set (the sample raised): ``infra_error`` [``error``].
3. ``sample.limit`` is set, of any type (time, working, message, token, turn, context, cost,
   operator, custom; Inspect records all of them as sample limits): ``timeout``
   [``limit:<type>``].
4. The chosen scorer gave no score: ``infra_error`` [``no_score``].
5. The score is NaN (``Score.unscored``) with reason ``grader_failed`` or ``scoring_failed``
   (Inspect's measurement-instrument reasons): ``infra_error`` [``unscored:<reason>``].
6. Any other NaN score, which Inspect excludes from metrics: ``flagged``
   [``unscored:<reason>``].
7. An (id, epoch) pair from ``eval.dataset.sample_ids`` and ``eval.config.epochs`` that the log
   lacks, unless ``results.early_stopping`` lists it: ``infra_error`` [``missing``]
   (``include_missing=False`` drops these rows).
8. Otherwise ``ok``.

Non-ok rows keep the sample's score when it has one (a scored sample that hit a limit, or a
sample scored under ``score_on_error``) and 0 otherwise.

Covariates: ``status``, ``status_reason``, ``task_name``, ``sample_id``, ``scorer``,
``score_value`` (the raw value as text), ``total_time`` and ``working_time`` (seconds),
``input_tokens``, ``output_tokens``, and ``total_tokens`` (summed over ``model_usage``),
``retries`` (errors retried before the recorded attempt), ``eval_id``, and ``log``.
"""

from __future__ import annotations

import importlib
import math
import re
import struct
import zipfile
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from evalstats.adapters._common import (
    FLAGGED,
    INFRA_ERROR,
    OK,
    TIMEOUT,
    as_mapping,
    build_results,
    check_unit,
    load_json,
    load_json_file,
)
from evalstats.schema import EvalResults, SchemaError

INSPECT_VALUE_MAP: Mapping[str, float] = MappingProxyType({"C": 1.0, "P": 0.5, "I": 0.0, "N": 0.0})
"""Default outcomes for Inspect's letter values (as ``inspect_ai.scorer.value_to_float``)."""

GRADER_FAILURES = frozenset({"grader_failed", "scoring_failed"})
"""Unscored reasons that Inspect attributes to the measurement instrument."""

_ZIP_ZSTANDARD = 93
_LOCAL_HEADER = struct.Struct("<4s22xHH")
_LOCAL_SIGNATURE = b"PK\x03\x04"
_SAMPLE_MEMBER = re.compile(r"samples/[^/]+\.json")
_COLUMNS = (
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
_NUMERIC = (
    "total_time",
    "working_time",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "retries",
)
_USAGE_KEYS = ("input_tokens", "output_tokens", "total_tokens")


@dataclass(frozen=True)
class _Log:
    label: str
    header: Mapping[str, Any]
    samples: Sequence[Mapping[str, Any]]


@dataclass(frozen=True)
class _Options:
    scorer: str | None
    value_map: Mapping[str, float]
    value_key: str | None
    run_key: str
    include_missing: bool
    label: str = ""


def read_inspect(
    path_or_dir: str | Path,
    *,
    scorer: str | None = None,
    value_map: Mapping[str, float] | None = None,
    value_key: str | None = None,
    run_key: Literal["epoch", "eval_epoch"] = "epoch",
    include_missing: bool = True,
) -> EvalResults:
    """Read Inspect AI eval logs (``.eval`` or ``.json``, a file or a directory of them).

    ``path_or_dir`` is a log file or a directory searched recursively for logs. ``scorer``
    names the scorer whose value is the outcome (default: the first scorer). ``value_map``
    gives the outcome of string score values in place of :data:`INSPECT_VALUE_MAP`, and
    ``value_key`` picks the entry of dict-valued scores. ``run_key="eval_epoch"`` labels runs
    ``<eval_id>/<epoch>`` so that several logs of one model and task can be read together.
    ``include_missing=False`` drops the ``infra_error`` rows added for samples a log lacks.
    The module docstring gives the row layout, the value mapping, and the status mapping.
    Raises :class:`SchemaError` if a file is not a readable Inspect log, a score cannot be
    mapped to [0, 1], a log holds one sample and epoch twice, or two logs give the same
    (model, task, run).
    """
    if run_key not in ("epoch", "eval_epoch"):
        raise ValueError(f"run_key must be 'epoch' or 'eval_epoch', not {run_key!r}")
    options = _Options(
        scorer=scorer,
        value_map=INSPECT_VALUE_MAP if value_map is None else value_map,
        value_key=value_key,
        run_key=run_key,
        include_missing=include_missing,
    )
    path = Path(path_or_dir)
    rows: list[dict[str, Any]] = []
    owner: dict[tuple[str, str, str], str] = {}
    for log in _read_logs(path):
        for row in _log_rows(log, options):
            key = (row["model"], row["task"], row["run"])
            if key in owner:
                raise SchemaError(
                    f"{key[1]} run {key[2]} of {key[0]} appears in both {owner[key]} and "
                    f"{log.label}; read one log, or pass run_key='eval_epoch' to keep "
                    "repeated logs apart"
                )
            owner[key] = log.label
            rows.append(row)
    return build_results(rows, source=str(path), columns=_COLUMNS, numeric=_NUMERIC)


def _read_logs(path: Path) -> list[_Log]:
    if path.is_dir():
        logs = []
        for file in sorted(p for p in path.rglob("*") if p.is_file()):
            label = file.relative_to(path).as_posix()
            if file.suffix == ".eval":
                logs.append(_read_eval_file(file, label))
            elif file.suffix == ".json":
                log = _read_json_file(file, label)
                if log is not None:
                    logs.append(log)
        if not logs:
            raise SchemaError(f"{path}: no Inspect logs found")
        return logs
    if not path.is_file():
        raise SchemaError(f"{path}: no such file or directory")
    if path.suffix == ".eval":
        return [_read_eval_file(path, path.name)]
    log = _read_json_file(path, path.name)
    if log is None:
        raise SchemaError(f"{path}: not an Inspect eval log (no 'eval' and 'version' keys)")
    return [log]


def _is_log_header(data: Any) -> bool:
    return isinstance(data, Mapping) and isinstance(data.get("eval"), Mapping) and "version" in data


def _read_json_file(path: Path, label: str) -> _Log | None:
    data = load_json_file(path)
    if not _is_log_header(data):
        return None
    samples = data.get("samples") or []
    if not isinstance(samples, list):
        raise SchemaError(f"{path}: 'samples' is not a list")
    return _Log(label, data, [as_mapping(s, f"{path}: sample") for s in samples])


def _read_eval_file(path: Path, label: str) -> _Log:
    try:
        archive = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, NotImplementedError, OSError, ValueError) as exc:
        # a damaged central directory can also name a zip version zipfile does not know
        # (NotImplementedError) or a file name that is not UTF-8 (UnicodeDecodeError)
        raise SchemaError(f"{path}: not a readable .eval zip ({exc})") from exc
    with archive:
        members = {info.filename: info for info in archive.infolist()}
        nested = [n for n in members if n.startswith("samples/") and n.count("/") > 1]
        if nested:
            raise SchemaError(
                f"{path}: chunked sample layout ({nested[0]}) is not supported; Inspect's own "
                "read_eval_log does not read it either (0.3.283)"
            )
        name = "header.json" if "header.json" in members else "_journal/start.json"
        if name not in members:
            raise SchemaError(f"{path}: no header.json or _journal/start.json; not an Inspect log")
        header = load_json(_member(archive, members[name], path), f"{path}:{name}")
        if not _is_log_header(header):
            raise SchemaError(f"{path}:{name}: not an Inspect log header")
        samples = [
            as_mapping(load_json(_member(archive, info, path), f"{path}:{n}"), f"{path}:{n}")
            for n, info in members.items()
            if _SAMPLE_MEMBER.fullmatch(n)
        ]
    return _Log(label, header, samples)


def _member(archive: zipfile.ZipFile, info: zipfile.ZipInfo, path: Path) -> bytes:
    where = f"{path}:{info.filename}"
    try:
        return archive.read(info)
    except NotImplementedError as exc:
        if info.compress_type != _ZIP_ZSTANDARD:
            raise SchemaError(
                f"{where}: unsupported zip compression method {info.compress_type}"
            ) from exc
        return _read_zstd_member(path, info)
    except Exception as exc:
        # Damaged members raise many types: zipfile's BadZipFile, RuntimeError (encryption
        # flag), OSError (bad offset), EOFError, and UnicodeDecodeError (name), plus each
        # decompressor's own error (zlib.error, lzma.LZMAError, OSError from bz2, and
        # compression.zstd.ZstdError on Python 3.14).
        raise SchemaError(f"{where}: corrupt zip member ({exc})") from exc


def _read_zstd_member(path: Path, info: zipfile.ZipInfo) -> bytes:
    """Decompress one zstd member with the ``zstandard`` package (Python before 3.14)."""
    where = f"{path}:{info.filename}"
    try:
        zstandard = importlib.import_module("zstandard")
    except ImportError as exc:
        raise SchemaError(
            f"{where}: this .eval log uses zstd compression (Inspect 0.3.206 and later), which "
            "needs Python 3.14 or the zstandard package (pip install zstandard); or convert it "
            "first with `inspect log convert --to json`"
        ) from exc
    with path.open("rb") as fh:
        fh.seek(info.header_offset)
        head = fh.read(_LOCAL_HEADER.size)
        if len(head) < _LOCAL_HEADER.size:
            raise SchemaError(f"{where}: truncated zip member header")
        signature, name_len, extra_len = _LOCAL_HEADER.unpack(head)
        if signature != _LOCAL_SIGNATURE:
            raise SchemaError(f"{where}: bad zip member header")
        fh.seek(name_len + extra_len, 1)
        raw = fh.read(info.compress_size)
    try:
        reader = zstandard.ZstdDecompressor().stream_reader(raw, read_across_frames=True)
        data: bytes = reader.read()
    except zstandard.ZstdError as exc:
        raise SchemaError(f"{where}: corrupt zstd data ({exc})") from exc
    if len(data) != info.file_size or zlib.crc32(data) != info.CRC:
        raise SchemaError(f"{where}: zstd member failed its size or CRC check")
    return data


def _log_rows(log: _Log, options: _Options) -> list[dict[str, Any]]:
    options = replace(options, label=log.label)
    spec = as_mapping(log.header["eval"], f"{log.label}: eval")
    model, task_name = spec.get("model"), spec.get("task")
    if not isinstance(model, str) or not isinstance(task_name, str):
        raise SchemaError(f"{log.label}: eval.model and eval.task must be strings")
    eval_id = str(spec.get("eval_id") or log.label)
    chosen = options.scorer or _first_scorer(log, spec)
    _check_scorer(log, options.scorer)
    base = {"model": model, "task_name": task_name, "eval_id": eval_id, "log": log.label}
    rows = []
    seen: set[tuple[str, int]] = set()
    for sample in log.samples:
        sample_id, epoch = sample.get("id"), sample.get("epoch")
        if sample_id is None or not isinstance(epoch, int):
            raise SchemaError(f"{log.label}: sample without an id or integer epoch")
        if (str(sample_id), epoch) in seen:
            raise SchemaError(f"{log.label}: sample {sample_id} epoch {epoch} appears twice")
        seen.add((str(sample_id), epoch))
        row = _sample_row(
            sample, sample_id=str(sample_id), epoch=epoch, scorer=chosen, options=options
        )
        rows.append({**base, **row, "run": _run(eval_id, epoch, options.run_key)})
    if options.include_missing:
        for sample_id, epoch in _missing(log, spec, seen):
            rows.append(
                {
                    **base,
                    "run": _run(eval_id, epoch, options.run_key),
                    "outcome": 0.0,
                    "status": INFRA_ERROR,
                    "status_reason": "missing",
                    "sample_id": sample_id,
                    "scorer": chosen,
                }
            )
    for row in rows:
        row["task"] = f"{task_name}/{row['sample_id']}"
    return rows


def _run(eval_id: str, epoch: int, run_key: str) -> str:
    return f"{eval_id}/{epoch}" if run_key == "eval_epoch" else str(epoch)


def _first_scorer(log: _Log, spec: Mapping[str, Any]) -> str | None:
    for sample in log.samples:
        scores = sample.get("scores")
        if isinstance(scores, Mapping) and scores:
            return str(next(iter(scores)))
    scorers = spec.get("scorers")
    if isinstance(scorers, list) and scorers and isinstance(scorers[0], Mapping):
        return str(scorers[0].get("name", "")).rsplit("/", 1)[-1] or None
    return None


def _check_scorer(log: _Log, scorer: str | None) -> None:
    available: dict[str, None] = {}
    for sample in log.samples:
        scores = sample.get("scores")
        if isinstance(scores, Mapping):
            available.update(dict.fromkeys(str(k) for k in scores))
    if scorer is not None and available and scorer not in available:
        raise SchemaError(
            f"{log.label}: scorer {scorer!r} not found; available: {', '.join(available)}"
        )


def _sample_row(
    sample: Mapping[str, Any], *, sample_id: str, epoch: int, scorer: str | None, options: _Options
) -> dict[str, Any]:
    where = f"{options.label}: sample {sample_id} epoch {epoch}"
    scores = sample.get("scores")
    score = None
    if isinstance(scores, Mapping) and scorer is not None and scores.get(scorer) is not None:
        score = as_mapping(scores[scorer], f"{where}: score {scorer!r}")
    raw = None if score is None else score.get("value")
    value = None if score is None else _to_float(raw, options.value_map, options.value_key, where)
    status, reason = _status(sample, score, value)
    retries = sample.get("error_retries")
    return {
        "outcome": 0.0 if value is None else check_unit(value, where),
        "status": status,
        "status_reason": reason,
        "sample_id": sample_id,
        "scorer": scorer,
        "score_value": None if score is None else str(raw),
        "total_time": sample.get("total_time"),
        "working_time": sample.get("working_time"),
        **_usage(sample.get("model_usage")),
        "retries": len(retries) if isinstance(retries, list) else None,
    }


def _status(sample: Mapping[str, Any], score: Any, value: float | None) -> tuple[str, str | None]:
    if sample.get("invalidation"):
        return FLAGGED, "invalidated"
    if sample.get("error"):
        return INFRA_ERROR, "error"
    limit = sample.get("limit")
    if limit:
        kind = limit.get("type") if isinstance(limit, Mapping) else limit
        return TIMEOUT, f"limit:{kind}"
    if not isinstance(score, Mapping):
        return INFRA_ERROR, "no_score"
    if value is None:
        metadata = score.get("metadata")
        reason = score.get("reason") or (
            metadata.get("unscored_reason") if isinstance(metadata, Mapping) else None
        )
        status = INFRA_ERROR if reason in GRADER_FAILURES else FLAGGED
        return status, f"unscored:{reason}" if reason else "unscored"
    return OK, None


def _to_float(
    raw: Any, value_map: Mapping[str, float], value_key: str | None, where: str
) -> float | None:
    """Map one score value to a float; None for missing or NaN (unscored) values."""
    if isinstance(raw, Mapping):
        if value_key is None:
            keys = ", ".join(str(k) for k in raw)
            raise SchemaError(f"{where}: score value is a dict ({keys}); pass value_key")
        if value_key not in raw:
            raise SchemaError(f"{where}: score value has no key {value_key!r}")
        raw = raw[value_key]
    if isinstance(raw, str):
        return _text_to_float(raw, value_map, where)
    if raw is not None and not isinstance(raw, int | float):
        raise SchemaError(f"{where}: unsupported score value {raw!r}")
    if raw is None or math.isnan(raw):
        return None
    return float(raw)


def _text_to_float(raw: str, value_map: Mapping[str, float], where: str) -> float | None:
    if raw in value_map:
        return float(value_map[raw])
    text = raw.strip().lower()
    words = {"yes": 1.0, "true": 1.0, "no": 0.0, "false": 0.0}
    if text in words:
        return words[text]
    try:
        number = float(text)
    except ValueError:
        raise SchemaError(f"{where}: cannot convert score value {raw!r}; pass value_map") from None
    return None if math.isnan(number) else number


def _usage(model_usage: Any) -> dict[str, float | None]:
    totals: dict[str, float | None] = dict.fromkeys(_USAGE_KEYS)
    if isinstance(model_usage, Mapping):
        for entry in model_usage.values():
            if not isinstance(entry, Mapping):
                continue
            for key in _USAGE_KEYS:
                count = entry.get(key)
                if isinstance(count, int | float):
                    totals[key] = (totals[key] or 0.0) + float(count)
    return totals


def _missing(
    log: _Log, spec: Mapping[str, Any], seen: set[tuple[str, int]]
) -> list[tuple[str, int]]:
    dataset, config = spec.get("dataset"), spec.get("config")
    ids = dataset.get("sample_ids") if isinstance(dataset, Mapping) else None
    if not isinstance(ids, list):
        return []
    epochs = config.get("epochs") if isinstance(config, Mapping) else None
    epochs = epochs if isinstance(epochs, int) and epochs > 0 else 1
    skipped = _early_stops(log.header.get("results"), log.label)
    return [
        (str(sample_id), epoch)
        for epoch in range(1, epochs + 1)
        for sample_id in ids
        if (str(sample_id), epoch) not in seen and (str(sample_id), epoch) not in skipped
    ]


def _early_stops(results: Any, label: str) -> set[tuple[str, int]]:
    early = results.get("early_stopping") if isinstance(results, Mapping) else None
    stops = early.get("early_stops") if isinstance(early, Mapping) else None
    if not isinstance(stops, list):
        return set()
    skipped: set[tuple[str, int]] = set()
    for stop in stops:
        epoch = stop.get("epoch") if isinstance(stop, Mapping) else None
        if not isinstance(epoch, int):
            raise SchemaError(
                f"{label}: results.early_stopping lists an entry without an integer epoch"
            )
        skipped.add((str(stop.get("id")), epoch))
    return skipped
