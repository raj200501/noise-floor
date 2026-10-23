"""Rewrite an .eval zip with ZIP_DEFLATED members, keeping every member's bytes and order.

Usage: python repack.py <in.eval> <out.eval>

Then checks with inspect_ai.log.read_eval_log that both files give the same samples.
"""

from __future__ import annotations

import sys
import zipfile

import zipfile_zstd  # noqa: F401  (adds zstd reading to zipfile on Python < 3.14)
from inspect_ai.log import read_eval_log

src, dst = sys.argv[1], sys.argv[2]
with (
    zipfile.ZipFile(src) as zin,
    zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED, compresslevel=5) as zout,
):
    for info in zin.infolist():
        member = zipfile.ZipInfo(info.filename, info.date_time)
        member.external_attr = info.external_attr
        zout.writestr(member, zin.read(info), compress_type=zipfile.ZIP_DEFLATED)

a, b = read_eval_log(src), read_eval_log(dst)
assert a.model_dump_json() == b.model_dump_json(), "repacked log differs"
print("repacked", dst, "samples", len(b.samples or []))
