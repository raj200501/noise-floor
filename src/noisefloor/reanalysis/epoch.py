"""Epoch AI Benchmarking Hub, SWE-bench Verified (E2): fetch, parse, pin.

Source: ``https://epoch.ai/data/benchmark_data.zip`` (CC BY 4.0), file
``swe_bench_verified.csv``; each row links an Inspect ``.eval`` log in its ``Logs``
column. Logs are read with HTTP range requests: the zip central directory, then only
the ``header.json`` and ``summaries.json`` members (a sample summary carries the
sample's id, epoch, scores, error, and limit). Every fetched member is decompressed
(deflate or zstd) and hashed, and the object's ETag and version id are recorded.

Rows are produced by the evalstats Inspect adapter
(:func:`evalstats.adapters.read_inspect`) on a JSON log assembled from the header and
the summaries, so the status mapping is the adapter's. Data credit: Epoch AI,
Benchmarking Hub, CC BY 4.0.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import http.client
import io
import json
import struct
import sys
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

ARCHIVE_URL = "https://epoch.ai/data/benchmark_data.zip"
CSV_NAME = "swe_bench_verified.csv"
LICENSE = "CC-BY-4.0"
CREDIT = "Epoch AI, Benchmarking Hub (https://epoch.ai), CC BY 4.0"
USER_AGENT = "noise-floor/0.1 (statistical reanalysis; no login)"
MEMBERS = ("header.json", "summaries.json")
ZIP_ZSTD = 93
COLUMNS = ["entry", "model", "log_id", "task", "run", "outcome", "status", "status_reason"]


class FetchError(RuntimeError):
    """A remote file could not be read."""


def _request(
    url: str, byte_range: tuple[int, int] | None = None, retries: int = 5
) -> tuple[bytes, Mapping[str, str]]:
    """GET (optionally a byte range) with retries; raises :class:`FetchError`."""
    headers = {"User-Agent": USER_AGENT}
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, headers=headers), timeout=120
            ) as resp:
                return resp.read(), dict(resp.headers.items())
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt + 1 == retries:
                raise FetchError(f"{url}: HTTP {exc.code}") from exc
        except (urllib.error.URLError, http.client.HTTPException, OSError) as exc:
            if attempt + 1 == retries:
                raise FetchError(f"{url}: {exc}") from exc
        time.sleep(5 * (attempt + 1))
    raise FetchError(f"{url}: no attempts made")


def central_directory(cd_bytes: bytes) -> list[dict[str, int | str]]:
    """Parse zip central-directory entries (name, method, sizes, local header offset)."""
    entries: list[dict[str, int | str]] = []
    pos = 0
    while pos + 46 <= len(cd_bytes) and cd_bytes[pos : pos + 4] == b"PK\x01\x02":
        method = struct.unpack("<H", cd_bytes[pos + 10 : pos + 12])[0]
        csize, usize = struct.unpack("<II", cd_bytes[pos + 20 : pos + 28])
        nlen, elen, clen = struct.unpack("<HHH", cd_bytes[pos + 28 : pos + 34])
        offset = struct.unpack("<I", cd_bytes[pos + 42 : pos + 46])[0]
        name = cd_bytes[pos + 46 : pos + 46 + nlen].decode()
        extra = cd_bytes[pos + 46 + nlen : pos + 46 + nlen + elen]
        q = 0
        while q + 4 <= len(extra):
            hid, hlen = struct.unpack("<HH", extra[q : q + 4])
            if hid == 1:  # zip64 extended information
                vals, k = extra[q + 4 : q + 4 + hlen], 0
                if usize == 0xFFFFFFFF:
                    usize = struct.unpack("<Q", vals[k : k + 8])[0]
                    k += 8
                if csize == 0xFFFFFFFF:
                    csize = struct.unpack("<Q", vals[k : k + 8])[0]
                    k += 8
                if offset == 0xFFFFFFFF:
                    offset = struct.unpack("<Q", vals[k : k + 8])[0]
            q += 4 + hlen
        entries.append(
            {"name": name, "method": method, "csize": csize, "usize": usize, "offset": offset}
        )
        pos += 46 + nlen + elen + clen
    return entries


def _locate_cd(tail: bytes) -> tuple[int, int]:
    """(offset, size) of the central directory from the end-of-archive records."""
    i = tail.rfind(b"PK\x05\x06")
    if i < 0:
        raise FetchError("no end-of-central-directory record")
    entries = struct.unpack("<H", tail[i + 10 : i + 12])[0]
    cd_size, cd_off = struct.unpack("<II", tail[i + 12 : i + 20])
    if cd_off == 0xFFFFFFFF or entries == 0xFFFF:
        j = tail.rfind(b"PK\x06\x06")
        if j < 0:
            raise FetchError("zip64 archive without a zip64 end record")
        _, cd_size, cd_off = struct.unpack("<QQQ", tail[j + 32 : j + 56])
    return int(cd_off), int(cd_size)


def decompress(method: int, data: bytes, usize: int) -> bytes:
    """Decompress one zip member (stored, deflate, or zstd)."""
    if method == 0:
        return data
    if method == zipfile.ZIP_DEFLATED:
        return zlib.decompress(data, -15)
    if method == ZIP_ZSTD:
        import zstandard  # noqa: PLC0415  (only needed for zstd members)

        reader = zstandard.ZstdDecompressor().stream_reader(data, read_across_frames=True)
        out = reader.read()
        if len(out) != usize:
            raise FetchError("zstd member size mismatch")
        return bytes(out)
    raise FetchError(f"unsupported zip method {method}")


def fetch_members(url: str, names: tuple[str, ...] = MEMBERS) -> dict[str, Any]:
    """Range-read the named members of a remote zip; returns bytes, hashes, and headers."""
    _, head = _request(url, (0, 0))
    total = int(str(head.get("Content-Range", "")).rsplit("/", 1)[-1])
    tail, _ = _request(url, (max(0, total - 65558), total - 1))
    cd_off, cd_size = _locate_cd(tail)
    cd, _ = _request(url, (cd_off, cd_off + cd_size - 1))
    entries = {str(e["name"]): e for e in central_directory(cd)}
    out: dict[str, Any] = {
        "size": total,
        "etag": head.get("ETag", ""),
        "version_id": head.get("x-amz-version-id", ""),
        "members": {},
    }
    for name in names:
        if name not in entries:
            raise FetchError(f"{url}: no member {name}")
        e = entries[name]
        local, _ = _request(url, (int(e["offset"]), int(e["offset"]) + 29))
        nlen, elen = struct.unpack("<HH", local[26:30])
        start = int(e["offset"]) + 30 + nlen + elen
        raw, _ = _request(url, (start, start + int(e["csize"]) - 1))
        data = decompress(int(e["method"]), raw, int(e["usize"]))
        out["members"][name] = {
            "bytes": data,
            "sha256": hashlib.sha256(data).hexdigest(),
            "method": int(e["method"]),
        }
    return out


def read_rows(archive: Path) -> list[dict[str, str]]:
    """Rows of ``swe_bench_verified.csv`` from the archive."""
    with zipfile.ZipFile(archive) as zf, zf.open(CSV_NAME) as fh:
        return list(csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8")))


def fetch(out_dir: Path) -> dict[str, Any]:
    """Download the archive and every row's log members into ``out_dir`` (network)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_bytes, headers = _request(ARCHIVE_URL)
    archive = out_dir / "benchmark_data.zip"
    archive.write_bytes(archive_bytes)
    manifest: dict[str, Any] = {
        "archive_url": ARCHIVE_URL,
        "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "archive_etag": headers.get("ETag", ""),
        "retrieved": time.strftime("%Y-%m-%d", time.gmtime()),
        "license": LICENSE,
        "logs": {},
        "failures": [],
    }
    for row in read_rows(archive):
        log_id, url = row.get("id", ""), row.get("Logs", "")
        if not url.startswith("https://"):
            manifest["failures"].append({"id": log_id, "error": "no log link"})
            continue
        try:
            got = fetch_members(url)
        except FetchError as exc:
            manifest["failures"].append({"id": log_id, "url": url, "error": str(exc)})
            continue
        folder = out_dir / "logs" / log_id
        folder.mkdir(parents=True, exist_ok=True)
        for name, member in got["members"].items():
            (folder / name).write_bytes(member["bytes"])
        manifest["logs"][log_id] = {
            "url": url,
            "model": row.get("Model version", ""),
            "etag": got["etag"],
            "version_id": got["version_id"],
            "size": got["size"],
            "members": {n: m["sha256"] for n, m in got["members"].items()},
        }
    return manifest


def assemble_log(header: Mapping[str, Any], summaries: list[Any]) -> dict[str, Any]:
    """A JSON-format Inspect log built from a header and sample summaries."""
    return {**header, "samples": summaries}


def load_samples(out_dir: Path, scorer: str = "swe_bench_scorer") -> pd.DataFrame:
    """One row per (entry, sample) with outcome and status, via the Inspect adapter."""
    from evalstats.adapters import read_inspect  # noqa: PLC0415  (keeps import light)

    manifest = json.loads((out_dir / "fetch_manifest.json").read_text())
    frames = []
    for log_id, info in sorted(manifest["logs"].items()):
        folder = out_dir / "logs" / log_id
        header = json.loads((folder / "header.json").read_text())
        summaries = json.loads((folder / "summaries.json").read_text())
        log_path = folder / "assembled.json"
        log_path.write_text(json.dumps(assemble_log(header, summaries)))
        res = read_inspect(log_path, scorer=scorer)
        frame = res.frame.copy()
        frame["entry"] = f"{info['model']} [{log_id}]"
        frame["log_id"] = log_id
        frame["task"] = frame["sample_id"].astype(str)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    data = pd.concat(frames, ignore_index=True)
    return data[
        ["entry", "model", "log_id", "task", "run", "outcome", "status", "status_reason"]
    ].copy()


def write_table(frame: pd.DataFrame, path: Path) -> str:
    """Deterministic gzip CSV with the credit line; returns its SHA-256."""
    buf = io.StringIO()
    buf.write(f"# SWE-bench Verified sample outcomes derived from {CREDIT}.\n")
    frame.to_csv(buf, index=False, lineterminator="\n")
    raw = gzip.compress(buf.getvalue().encode(), mtime=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def main(argv: list[str] | None = None) -> int:
    """``python -m noisefloor.reanalysis.epoch fetch OUT_DIR`` (network)."""
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2 or args[0] != "fetch":
        print("usage: python -m noisefloor.reanalysis.epoch fetch OUT_DIR", file=sys.stderr)
        return 2
    out = Path(args[1])
    manifest = fetch(out)
    (out / "fetch_manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    print(f"fetched {len(manifest['logs'])} logs, {len(manifest['failures'])} failures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
