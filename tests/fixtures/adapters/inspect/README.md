# Inspect AI log fixtures (mock data)

These are real Inspect AI log files written by Inspect itself, but every outcome in them is
**mock data**: the model is Inspect's built-in mock provider (`mockllm/model`) answering from a
fixed table, so nothing here is a model result.

Contract tests: `tests/contract/test_inspect.py`.

## How they were produced (2026-10-23)

All logs come from `make_logs.py` in this directory, run offline outside the repository in
scratch virtual environments created with `uv venv --python 3.12` and
`uv pip install inspect_ai==<version>` from PyPI.

The task `tiny_qa` has 4 samples (`sum`, `product`, `diff`, `loop`), `epochs=3`,
`fail_on_error=False`, `message_limit=6`, and four scorers in this order: built-in
`match(location="any")` (C/I), `letter_grade` (C/I/P/N), `closeness` (a float in [0, 1], or
`Score.unscored(reason="scoring_failed")` when the answer has no number), and `has_answer`
(a boolean). Designed events in every log:

- sample `sum`, epoch 2: the solver raises `RuntimeError`, so the sample has `error` set;
- sample `loop`, every epoch: the solver keeps generating until the message limit fires, so
  the sample has `limit = {"type": "message", "limit": 6}`;
- sample `diff`, epoch 1: invalidated after the run with
  `inspect_ai.log.invalidate_samples` and rewritten with `write_eval_log` (the `--invalidate`
  flag);
- sample `diff`, epoch 3: the mock answers "I don't know.", giving `N` and an unscored
  `closeness`.

| Directory | Inspect | Command | Notes |
| --- | --- | --- | --- |
| `eval-zstd/` | 0.3.283 | `python make_logs.py out eval --invalidate` | `.eval` as written: zstd members (Inspect 0.3.206 and later) |
| `eval-deflate/` | 0.3.283 | `python repack.py <eval-zstd file> <out>` | the same members, byte for byte, rewritten with `ZIP_DEFLATED`; `repack.py` checks that `inspect_ai.log.read_eval_log` returns the same log for both files |
| `json/` | 0.3.283 | `python make_logs.py out json --invalidate` | a separate run of the same task in the JSON log format (243 KB, over the 200 KB target because the errored sample carries two copies of a long traceback) |
| `legacy/` | 0.3.205 | `python make_logs.py out eval --invalidate` | `.eval` from the last Inspect release that wrote deflate members by default; `closeness` returns 0.0 instead of unscored (no `Score.unscored` yet) and the mock reports no token usage |

`eval-deflate/` exists so that the `.eval` format can be tested on Python 3.11 to 3.13 without
the `zstandard` package. The tracebacks inside the logs contain paths of the scratch
environment where they were generated; nothing else about the machine is recorded.

Inspect 0.3.283 also has a hidden `inspect log convert-chunked` command that writes samples
as `samples/<id>_epoch_<n>/sample.json` plus chunk files. Its own `read_eval_log` failed on
that output, so the adapter rejects the layout with a clear error and no fixture is kept.
