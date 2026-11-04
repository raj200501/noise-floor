import json
from pathlib import Path

import pytest

from noisefloor import jsonl


def test_append_and_read(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "rows.jsonl"
    assert jsonl.read(path) == []
    assert jsonl.append(path, [{"b": 1, "a": 2}, {"c": 3}]) == 2
    assert path.read_text().splitlines()[0] == '{"a": 2, "b": 1}'
    assert jsonl.read(path) == [{"a": 2, "b": 1}, {"c": 3}]


def test_cut_tail_is_skipped_or_repaired(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2, "b"')
    with pytest.raises(ValueError, match=r"rows.jsonl:3: not a JSON record"):
        jsonl.read(path)
    assert jsonl.read(path, tail_ok=True) == [{"a": 1}]
    assert jsonl.repair_tail(path) == '{"a": 2, "b"'
    assert path.read_text() == '{"a": 1}\n\n'
    assert jsonl.repair_tail(path) is None
    jsonl.append(path, [{"a": 3}])
    assert jsonl.read(path) == [{"a": 1}, {"a": 3}]


def test_repair_keeps_a_whole_record_without_newline(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    assert jsonl.repair_tail(path) is None
    path.write_text('{"a": 1}')
    assert jsonl.repair_tail(path) is None and path.read_text() == '{"a": 1}\n'
    path.write_text("")
    assert jsonl.repair_tail(path) is None


def test_a_bad_line_in_the_middle_is_never_skipped(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\nnot json\n{"a": 2}\n')
    with pytest.raises(ValueError, match=":2:"):
        jsonl.read(path, tail_ok=True)
    path.write_text(json.dumps([1, 2]) + "\n")
    with pytest.raises(ValueError, match="not a JSON object"):
        jsonl.read(path)
