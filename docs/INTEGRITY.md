# Integrity checks: how to use them

Three stdlib-only tools guard section 9 rules 1 and 2 of docs/BUILD_BRIEF.md (gates G10 and
G11). Each prints one line per check and ends with a PASS or FAIL line; exit code 0 means PASS.

| Command | Tool | Guards |
|---|---|---|
| `make prereg-verify` | `tools/prereg.py verify` | Preregistrations are frozen, unedited, and committed before the data that cites them |
| `make claimcheck` | `tools/claimcheck.py README.md docs/REPORT.md` | Every number in the documents is a registered claim or a narrow exemption |
| `make freshness` | `tools/freshness.py README.md docs/REPORT.md` | Every claimed number equals what its artifact says now |

All three accept `--root <repo>`; paths are relative to the root.

## 1. Freeze a preregistration

1. Write `prereg/NNNN-<slug>.md`, where NNNN is the next unused number (`python
   tools/prereg.py show` prints it) and the slug is lowercase letters, digits, and hyphens.
   Machine-readable parts go in fenced blocks such as ` ```json e0-grid `; they must parse.
2. Freeze it: `python tools/prereg.py freeze prereg/NNNN-<slug>.md`. This appends an entry to
   `prereg/CHAIN.jsonl` with the file's SHA-256, the previous entry's hash, and the UTC time.
   Freeze refuses a wrong number, a second freeze, invalid JSON blocks, or a file that was
   committed with different content.
3. Commit the file and `prereg/CHAIN.jsonl` together, before any confirmatory data exists.
4. Never edit it again. The file must have exactly one commit, and the chain may only grow.

Do not keep drafts under `prereg/` with an `NNNN-` name: every such file must be in the chain.

## 2. Cite it from artifacts

A confirmatory artifact names its preregistration with a top-level field
`"prereg": "prereg/NNNN-<slug>.md"` (a JSON file under `artifacts/`) or the same field on every
row (a `.jsonl` file). A record with `"confirmatory": true` must have it. Verify checks that the
prereg file and its chain entry were both committed strictly before the first commit of the
artifact file, so commit the preregistration first and the data in a later commit.

## 3. Amend a preregistration

Never edit the original. Write a new file with front matter, then freeze it as above:

```markdown
---
amends: 0001
date: 2026-10-23
---
# 0002 Amendment to 0001: <what changed and why>
```

The amended number must be earlier and in the chain. Report the amendment as a deviation.

## 4. Register a claim

Result numbers come from `artifacts/claims.json`. Write the definitions (id, artifact, pointer,
and optionally derive, where, field, format), then fill in values and hashes from the artifacts:

```bash
python tools/freshness.py --regenerate
```

```json
{
  "schema": "noise-floor/claims/1",
  "generator": "tools/freshness.py",
  "claims": [
    {"id": "e1.vc.task", "artifact": "artifacts/e1/variance_components.json",
     "pointer": "/components/task/estimate", "format": ".2f",
     "value": "0.31", "artifact_sha256": "<written by --regenerate>"},
    {"id": "e1.runs.confirmatory", "artifact": "artifacts/e1/runs.jsonl", "pointer": "",
     "derive": "count", "where": {"confirmatory": true}, "format": ",d",
     "value": "1,042", "artifact_sha256": "<written by --regenerate>"}
  ]
}
```

- `pointer` is a JSON pointer (RFC 6901) into the artifact; a `.jsonl` artifact is a list of rows.
- `derive` is one of `len`, `count`, `sum`, `mean`, `min`, `max`. `where` filters rows by exact
  JSON equality (keys may be pointers such as `/factors/harness`); `field` picks the row value.
- `format` is a Python format spec (`.2f`, `,d`, `.1%`). Floats need one. `value` is
  `format(x, spec)`, the exact string the document shows.
- Another committed script may write the file instead; `generator` must name it.

Statistics that need numpy (intervals, variance components) are computed by the analysis code
and banked in an artifact; the claim points at the banked number.

## 5. Mark a number in a document

Put the marker immediately before the number. It does not render:

```markdown
Task variance share was <!-- claim:e1.vc.task -->0.31 (interval
<!-- claim:e1.vc.task.lo -->0.12 to <!-- claim:e1.vc.task.hi -->0.47).
```

- The rendered text after the marker must start with the claim's value exactly. Signs, percent
  signs, and thousands separators belong in the value (`-0.04`, `31.2%`, `1,042`).
- Put the marker inside emphasis (`**<!-- claim:x -->0.31**`), not outside it.
- Separate adjacent numbers with a space or a word (`0.12 to 0.47`).
- A block of program output can be banked as a file and shown verbatim: a fence whose info
  string is `text from=artifacts/<file>` must equal that file, and its numbers are then covered.

## 6. Exempt a non-result number

Design parameters, thresholds, caps, dates, versions, gate and section numbers, and quotes from
the brief go in `docs/claims_exempt.toml`, one exact string per entry:

```toml
[[exempt]]
text = "at least 2,000 replicates per cell"
docs = ["docs/REPORT.md"]
category = "threshold"
reason = "G5 replicate floor from the brief, not a measurement"
```

The text must include surrounding words, every field is required, `brief-quote` text must occur
in docs/BUILD_BRIEF.md, and an exemption that covers nothing fails. Never exempt a result.

## 7. What the claim check skips

Code blocks tagged with a programming or config language, inline code containing a letter, HTML
comments and tags, link and image targets, reference labels and definitions, footnote markers,
bare URLs, arXiv ids, DOIs, hex digests, and ordered-list markers. Untagged and `text` code
blocks are checked. Numbers spelled as words, numbers inside images and SVG files, and numbers in
skipped code are not seen, so do not put results there; the reviewer checks those by reading.

## 8. Run the checks

```bash
make prereg-verify
make claimcheck
make freshness
uv run pytest -q tests/test_prereg.py tests/test_claimcheck.py tests/test_freshness.py
```

The tests include mutation cases (a fabricated number, an edited frozen prereg, an artifact
committed before its prereg, a tampered chain, an artifact changed after the document was
rendered); each must make its tool fail.
