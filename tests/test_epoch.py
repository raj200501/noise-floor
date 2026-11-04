import gzip
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import pytest
import zstandard

from noisefloor.reanalysis import epoch

HEADER = {
    "version": 2,
    "status": "success",
    "eval": {
        "task": "swe_bench_verified",
        "model": "acme/model-x",
        "eval_id": "E1",
        "dataset": {"name": "swe", "samples": 3, "sample_ids": ["a", "b", "c"]},
        "config": {"epochs": 1},
        "scorers": [{"name": "swe_bench_scorer"}],
    },
    "results": {"scores": []},
}
SUMMARIES = [
    {"id": "a", "epoch": 1, "scores": {"swe_bench_scorer": {"value": "C"}}},
    {"id": "b", "epoch": 1, "scores": {"swe_bench_scorer": {"value": "I"}}, "limit": "token"},
    {"id": "c", "epoch": 1, "scores": {}, "error": "ReadTimeout('')"},
]


def make_eval_zip() -> bytes:
    """A small .eval-like zip: a stored member, a deflate summaries.json, a zstd header.json."""
    import struct
    import zlib

    header = json.dumps(HEADER).encode()
    summaries = json.dumps(SUMMARIES).encode()
    deflater = zlib.compressobj(9, zlib.DEFLATED, -15)
    entries = [
        ("samples/a_epoch_1.json", 0, b"{}", b"{}"),
        ("summaries.json", 8, summaries, deflater.compress(summaries) + deflater.flush()),
        ("header.json", 93, header, zstandard.ZstdCompressor().compress(header)),
    ]
    out = bytearray()
    central = bytearray()
    for name, method, data, packed in entries:
        crc = zlib.crc32(data)
        offset = len(out)
        nb = name.encode()
        out += (
            struct.pack(
                "<IHHHHHIIIHH",
                0x04034B50,
                20,
                0,
                method,
                0,
                0,
                crc,
                len(packed),
                len(data),
                len(nb),
                0,
            )
            + nb
            + packed
        )
        central += (
            struct.pack(
                "<IHHHHHHIIIHHHHHII",
                0x02014B50,
                20,
                20,
                0,
                method,
                0,
                0,
                crc,
                len(packed),
                len(data),
                len(nb),
                0,
                0,
                0,
                0,
                0,
                offset,
            )
            + nb
        )
    cd_off = len(out)
    out += central
    out += struct.pack(
        "<IHHHHIIH", 0x06054B50, 0, 0, len(entries), len(entries), len(central), cd_off, 0
    )
    return bytes(out)


def fake_request_factory(blob: bytes, archive: bytes | None = None) -> Any:
    def fake(
        url: str, byte_range: tuple[int, int] | None = None, retries: int = 5
    ) -> tuple[bytes, dict[str, str]]:
        if url == epoch.ARCHIVE_URL:
            assert archive is not None
            return archive, {"ETag": '"arch"'}
        if "missing" in url:
            raise epoch.FetchError(f"{url}: HTTP 403")
        a, b = byte_range if byte_range else (0, len(blob) - 1)
        return blob[a : b + 1], {
            "Content-Range": f"bytes {a}-{b}/{len(blob)}",
            "ETag": '"e1"',
            "x-amz-version-id": "v1",
        }

    return fake


def make_archive(rows: list[dict[str, str]]) -> bytes:
    buf = io.BytesIO()
    text = io.StringIO()
    import csv

    w = csv.DictWriter(text, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(epoch.CSV_NAME, text.getvalue())
    return buf.getvalue()


def test_fetch_members_reads_deflate_and_zstd(monkeypatch: pytest.MonkeyPatch) -> None:
    blob = make_eval_zip()
    monkeypatch.setattr(epoch, "_request", fake_request_factory(blob))
    got = epoch.fetch_members("https://logs.example/x.eval")
    assert json.loads(got["members"]["header.json"]["bytes"]) == HEADER
    assert json.loads(got["members"]["summaries.json"]["bytes"]) == SUMMARIES
    assert got["members"]["header.json"]["method"] == 93 and got["version_id"] == "v1"
    with pytest.raises(epoch.FetchError, match="no member"):
        epoch.fetch_members("https://logs.example/x.eval", ("reductions.json",))


def test_fetch_and_load_samples(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        {"Model version": "model-x", "Logs": "https://logs.example/ok.eval", "id": "L1"},
        {"Model version": "model-y", "Logs": "https://logs.example/missing.eval", "id": "L2"},
        {"Model version": "model-z", "Logs": "", "id": "L3"},
    ]
    blob = make_eval_zip()
    monkeypatch.setattr(epoch, "_request", fake_request_factory(blob, make_archive(rows)))
    manifest = epoch.fetch(tmp_path)
    assert list(manifest["logs"]) == ["L1"] and len(manifest["failures"]) == 2
    assert manifest["license"] == "CC-BY-4.0" and len(manifest["archive_sha256"]) == 64
    (tmp_path / "fetch_manifest.json").write_text(json.dumps(manifest))
    samples = epoch.load_samples(tmp_path)
    assert list(samples.columns) == epoch.COLUMNS
    by_task = samples.set_index("task")
    assert by_task.loc["a", "status"] == "ok" and by_task.loc["a", "outcome"] == 1.0
    assert by_task.loc["b", "status"] == "timeout" and by_task.loc["b", "outcome"] == 0.0
    assert by_task.loc["c", "status"] == "infra_error"
    assert (samples["entry"] == "model-x [L1]").all()
    digest = epoch.write_table(samples, tmp_path / "t.csv.gz")
    assert digest == epoch.write_table(samples, tmp_path / "u.csv.gz")
    assert "CC BY 4.0" in gzip.decompress((tmp_path / "t.csv.gz").read_bytes()).decode()


def test_empty_and_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "fetch_manifest.json").write_text(json.dumps({"logs": {}}))
    assert epoch.load_samples(tmp_path).empty
    assert epoch.main([]) == 2
    rows = [{"Model version": "m", "Logs": "https://logs.example/ok.eval", "id": "L1"}]
    monkeypatch.setattr(
        epoch, "_request", fake_request_factory(make_eval_zip(), make_archive(rows))
    )
    assert epoch.main(["fetch", str(tmp_path / "out")]) == 0
    assert "fetched 1 logs" in capsys.readouterr().out


def test_decompress_and_locate_errors() -> None:
    assert epoch.decompress(0, b"x", 1) == b"x"
    with pytest.raises(epoch.FetchError, match="unsupported"):
        epoch.decompress(12, b"x", 1)
    comp = zstandard.ZstdCompressor().compress(b"abc")
    with pytest.raises(epoch.FetchError, match="size"):
        epoch.decompress(93, comp, 5)
    with pytest.raises(epoch.FetchError, match="end-of-central"):
        epoch._locate_cd(b"nothing here")


def test_request_retries_then_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.error

    calls = {"n": 0}

    def flaky(*args: object, **kwargs: object) -> object:
        calls["n"] += 1
        raise urllib.error.URLError("reset")

    monkeypatch.setattr(epoch.urllib.request, "urlopen", flaky)
    monkeypatch.setattr(epoch.time, "sleep", lambda s: None)
    with pytest.raises(epoch.FetchError, match="reset"):
        epoch._request("https://x.example", retries=3)
    assert calls["n"] == 3

    def forbidden(*args: object, **kwargs: object) -> object:
        raise urllib.error.HTTPError("u", 403, "no", None, None)  # type: ignore[arg-type]  # minimal error

    monkeypatch.setattr(epoch.urllib.request, "urlopen", forbidden)
    with pytest.raises(epoch.FetchError, match="403"):
        epoch._request("https://x.example")
