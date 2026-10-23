"""Tests for the nf-tools mock file system and tools."""

import pytest

from noisefloor.suite.fs import FsError, Snapshot, WorkingCopy, normalize_path
from noisefloor.suite.tools import (
    CALC_MAX_CHARS,
    ERROR_PREFIX,
    OBSERVATION_LIMIT,
    READ_LIMIT,
    SEARCH_LIMIT,
    SEARCH_LINE_CHARS,
    TOOL_NAMES,
    TOOLS,
    TOOLS_BY_NAME,
    WRITE_LIMIT,
    CalcError,
    ToolEnv,
    evaluate_arithmetic,
    format_number,
)

EM_DASH = chr(0x2014)  # a code point, so this file itself contains no em dash

FILES = {
    "/data/a.txt": "alpha\nbeta gamma\n",
    "/data/sub/b.csv": "x,y\n1,2\n",
    "/data/sales-old/c.txt": "alpha again\n",
    "/notes.md": "",
}


def make_env(**kwargs):
    return ToolEnv(Snapshot(FILES), **kwargs)


# ---------------------------------------------------------------------------
# file system


@pytest.mark.parametrize(
    ("raw", "want"),
    [
        ("/a/b", "/a/b"),
        ("a/b", "/a/b"),
        ("//a//b/", "/a/b"),
        ("/a/./b/../c", "/a/c"),
        ("/../..", "/"),
        ("", "/"),
    ],
)
def test_normalize_path(raw, want):
    assert normalize_path(raw) == want


def test_normalize_path_rejects_nul():
    with pytest.raises(FsError, match="NUL"):
        normalize_path("/a\x00b")


def test_snapshot_is_sorted_and_read_only():
    snap = Snapshot({"/b": "2", "/a": "1"})
    assert snap.paths() == ("/a", "/b")
    assert list(snap.files) == ["/a", "/b"]
    assert snap.to_dict() == {"/a": "1", "/b": "2"}
    with pytest.raises(TypeError):
        snap.files["/c"] = "3"
    assert len(snap) == 2
    assert repr(snap) == "Snapshot(2 files)"


def test_snapshot_equality_and_hash():
    a, b, c = Snapshot({"/a": "1"}), Snapshot({"/a": "1"}), Snapshot({"/a": "2"})
    assert a == b
    assert hash(a) == hash(b)
    assert a != c
    assert a != {"/a": "1"}
    assert Snapshot() == Snapshot(None)
    assert len(Snapshot()) == 0


@pytest.mark.parametrize(
    "files",
    [{"a": "x"}, {"/a/": "x"}, {"/": "x"}, {"/a/../b": "x"}, {"/a": "x", "/a/b": "y"}],
)
def test_snapshot_rejects_bad_paths(files):
    with pytest.raises(FsError):
        Snapshot(files)


@pytest.mark.parametrize("files", [{"/a": 1}, {1: "x"}])
def test_snapshot_rejects_non_strings(files):
    with pytest.raises(TypeError):
        Snapshot(files)


def test_read_only_views():
    snap = Snapshot(FILES)
    assert snap.is_dir("/") and snap.is_dir("/data") and snap.is_dir("data/sub/")
    assert not snap.is_dir("/data/a.txt") and not snap.is_dir("/nope")
    assert snap.is_file("/data/a.txt") and not snap.is_file("/data")
    assert snap.get("data/a.txt") == FILES["/data/a.txt"]
    assert snap.get("/missing") is None
    assert snap.read("/notes.md") == ""
    with pytest.raises(FsError, match="is a directory: /data"):
        snap.read("/data")
    with pytest.raises(FsError, match="no such file: /x"):
        snap.read("/x")
    assert snap.list_dir("/") == ["data/", "notes.md"]
    assert snap.list_dir("/data") == ["a.txt", "sales-old/", "sub/"]
    with pytest.raises(FsError, match="not a directory"):
        snap.list_dir("/notes.md")
    with pytest.raises(FsError, match="no such directory"):
        snap.list_dir("/zzz")
    assert snap.files_under("/") == sorted(FILES)
    assert snap.files_under("/data/sub") == ["/data/sub/b.csv"]
    assert Snapshot().list_dir("/") == []


def test_working_copy_is_isolated_from_snapshot():
    snap = Snapshot(FILES)
    work = snap.working_copy()
    assert isinstance(work, WorkingCopy)
    assert work.base is snap
    assert work.write("new/dir//f.txt", "hi") == "/new/dir/f.txt"
    work.write("/data/a.txt", "changed")
    work.write("/notes.md", "")
    assert work.read("/new/dir/f.txt") == "hi"
    assert snap.get("/new/dir/f.txt") is None
    assert snap.read("/data/a.txt") == FILES["/data/a.txt"]
    assert work.changed_paths() == ["/data/a.txt", "/new/dir/f.txt"]
    frozen = work.snapshot()
    assert frozen.read("/new/dir/f.txt") == "hi"
    assert frozen != snap
    assert snap.working_copy().snapshot() == snap


def test_working_copy_write_errors():
    work = Snapshot(FILES).working_copy()
    with pytest.raises(FsError, match="root"):
        work.write("/", "x")
    with pytest.raises(FsError, match="is a directory: /data"):
        work.write("/data", "x")
    with pytest.raises(FsError, match=r"not a directory: /notes\.md"):
        work.write("/notes.md/inner.txt", "x")


# ---------------------------------------------------------------------------
# tool specs


def test_tool_names_and_specs():
    assert TOOL_NAMES == ("list_dir", "read_file", "write_file", "search", "calc", "submit")
    assert set(TOOLS_BY_NAME) == set(TOOL_NAMES)
    for spec in TOOLS:
        schema = spec.parameters
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) <= set(schema["properties"])
        for prop in schema["properties"].values():
            assert prop["type"] == "string"
            assert prop["description"]
        assert "\n" not in spec.description
        assert EM_DASH not in spec.description
        function = spec.as_function()
        assert function["type"] == "function"
        assert function["function"]["name"] == spec.name
        assert function["function"]["parameters"] == schema
    assert TOOLS_BY_NAME["list_dir"].parameters["required"] == []
    assert TOOLS_BY_NAME["list_dir"].parameters["properties"]["path"]["default"] == "/"
    assert TOOLS_BY_NAME["search"].parameters["required"] == ["pattern"]
    assert TOOLS_BY_NAME["write_file"].parameters["required"] == ["path", "content"]
    assert TOOLS_BY_NAME["submit"].params[0].required


# ---------------------------------------------------------------------------
# tools


def test_list_dir_tool():
    env = make_env()
    assert env.call("list_dir", {"path": "/data"}) == "a.txt\nsales-old/\nsub/"
    assert env.call("list_dir") == "data/\nnotes.md"
    assert env.call("list_dir", {"path": "/notes.md"}) == "error: not a directory: /notes.md"
    assert env.call("list_dir", {"path": "/zzz"}) == "error: no such directory: /zzz"
    assert ToolEnv(Snapshot()).call("list_dir", {}) == "(empty directory)"


def test_read_file_tool():
    env = make_env()
    assert env.call("read_file", {"path": "/data/a.txt"}) == FILES["/data/a.txt"]
    assert env.call("read_file", {"path": "/notes.md"}) == "(empty file)"
    assert env.call("read_file", {"path": "/data"}) == "error: is a directory: /data"
    assert env.call("read_file", {"path": "/x"}) == "error: no such file: /x"


def test_read_file_truncates_long_files():
    long = "x" * (READ_LIMIT + 500)
    env = ToolEnv(Snapshot({"/big.txt": long}))
    obs = env.call("read_file", {"path": "/big.txt"})
    assert obs.startswith("x" * READ_LIMIT)
    assert obs.endswith(f"[truncated: showed the first {READ_LIMIT} of {len(long)} characters]")
    assert len(obs) <= OBSERVATION_LIMIT


def test_write_file_tool():
    env = make_env()
    assert env.call("write_file", {"path": "out/r.txt", "content": "a\nb\n"}) == (
        "wrote 4 characters to /out/r.txt"
    )
    assert env.call("read_file", {"path": "/out/r.txt"}) == "a\nb\n"
    too_long = "x" * (WRITE_LIMIT + 1)
    assert env.call("write_file", {"path": "/y", "content": too_long}).startswith(
        "error: content is longer"
    )
    assert env.call("write_file", {"path": "/data", "content": "x"}) == (
        "error: is a directory: /data"
    )
    assert env.files.get("/y") is None


def test_search_tool():
    env = make_env()
    assert env.call("search", {"pattern": "alpha"}) == (
        "/data/a.txt:1: alpha\n/data/sales-old/c.txt:1: alpha again"
    )
    assert env.call("search", {"pattern": "gamma", "path": "/data/a.txt"}) == (
        "/data/a.txt:2: beta gamma"
    )
    assert env.call("search", {"pattern": "alpha", "path": "/data/sub"}) == "no matches"
    assert env.call("search", {"pattern": "", "path": "/"}) == "error: pattern must not be empty"
    assert env.call("search", {"pattern": "a", "path": "/zz"}) == (
        "error: no such file or directory: /zz"
    )


def test_search_respects_directory_boundaries():
    env = ToolEnv(Snapshot({"/data/sales/a.txt": "hit\n", "/data/sales-old/b.txt": "hit\n"}))
    assert env.call("search", {"pattern": "hit", "path": "/data/sales"}) == (
        "/data/sales/a.txt:1: hit"
    )


def test_search_caps_matches_and_line_length():
    lines = "\n".join(f"match {i}" for i in range(SEARCH_LIMIT + 7))
    env = ToolEnv(Snapshot({"/f.txt": lines, "/long.txt": "match " + "y" * 300}))
    obs = env.call("search", {"pattern": "match", "path": "/f.txt"})
    assert obs.count("\n") == SEARCH_LIMIT
    assert obs.endswith("[7 more matches not shown]")
    long_obs = env.call("search", {"pattern": "match", "path": "/long.txt"})
    assert long_obs == "/long.txt:1: " + ("match " + "y" * 300)[:SEARCH_LINE_CHARS] + "..."


def test_search_stays_within_observation_budget():
    wide = "\n".join("needle " + "z" * 90 for _ in range(SEARCH_LIMIT))
    env = ToolEnv(Snapshot({"/" + "d" * 40 + "/wide.txt": wide}))
    obs = env.call("search", {"pattern": "needle"})
    assert len(obs) <= OBSERVATION_LIMIT
    assert obs.endswith("more matches not shown]")


# ---------------------------------------------------------------------------
# calc


@pytest.mark.parametrize(
    ("expression", "want"),
    [
        ("1 + 2", "3"),
        ("(12.5 + 7) * 3", "58.5"),
        ("7 // 2", "3"),
        ("7 % 3", "1"),
        ("2 ** 10", "1024"),
        ("-3 + +2", "-1"),
        ("1 / 4", "0.25"),
        ("  2*3  ", "6"),
        ("2 ** -1", "0.5"),
        ("(-1) ** 100000001", "-1"),
        ("10.5 - 0.5", "10.0"),
        ("(\n1 + 1\n)", "2"),
    ],
)
def test_calc_evaluates_arithmetic(expression, want):
    assert format_number(evaluate_arithmetic(expression)) == want
    assert make_env().call("calc", {"expression": expression}) == want


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        ("x", "not allowed in an expression: Name"),
        ("__import__('os')", "not allowed in an expression: Call"),
        ("abs(-1)", "not allowed in an expression: Call"),
        ("(1).real", "not allowed in an expression: Attribute"),
        ("a.b", "not allowed in an expression: Attribute"),
        ("[1, 2]", "not allowed in an expression: List"),
        ("1 if 1 else 2", "not allowed in an expression: IfExp"),
        ("1 < 2", "not allowed in an expression: Compare"),
        ("lambda: 1", "not allowed in an expression: Lambda"),
        ("~1", "not allowed in an expression: UnaryOp"),
        ("1 << 2", "operator not allowed: LShift"),
        ("6 & 3", "operator not allowed: BitAnd"),
        ("'a'", "only real number literals"),
        ("True + 1", "only real number literals"),
        ("1j", "only real number literals"),
        ("2 ** 100000", "exponent is too large"),
        ("10 ** 10 ** 10", "exponent is too large"),
        ("(2 ** 2000) * (2 ** 2000) * (2 ** 2000)", "result is too large"),
        ("2.0 ** 5000", "result is too large"),
        ("1e308 * 10", "result is too large"),
        ("1e999", "result is too large"),
        ("(-8) ** 0.5", "not a real number"),
        ("1 / 0", "division by zero"),
        ("1 % 0", "division by zero"),
        ("0 ** -1", "division by zero"),
        ("", "could not parse"),
        ("1 +", "could not parse"),
        ("1; 2", "could not parse"),
        ("import os", "could not parse"),
        ("1\x00", "could not parse"),
        ("(" * 300 + "1" + ")" * 300, "could not parse"),
        ("-" * 150 + "1", "nested too deeply"),
        ("1+" * CALC_MAX_CHARS + "1", "longer than"),
    ],
)
def test_calc_rejects_unsafe_or_invalid_input(expression, message):
    with pytest.raises(CalcError, match=message):
        evaluate_arithmetic(expression)
    assert make_env().call("calc", {"expression": expression}).startswith(ERROR_PREFIX)


def test_format_number():
    assert format_number(3) == "3"
    assert format_number(0.1 + 0.2) == "0.30000000000000004"
    assert format_number(2.0) == "2.0"


# ---------------------------------------------------------------------------
# episode mechanics


def test_env_properties():
    env = make_env(instruction="do the thing")
    assert env.instruction == "do the thing"
    assert env.tool_specs() is TOOLS
    assert env.files.read("/data/a.txt") == FILES["/data/a.txt"]
    assert env.calls == 0 and env.trace == () and env.answer is None
    assert not env.done and env.end_reason is None


@pytest.mark.parametrize(
    ("name", "args", "start"),
    [
        ("nope", {}, "error: unknown tool 'nope'; available tools: list_dir, read_file"),
        (["list"], {}, "error: unknown tool ['list']"),
        ("read_file", ["/a"], "error: arguments for read_file must be an object"),
        ("read_file", "/a", "error: arguments for read_file must be an object"),
        ("read_file", {}, "error: missing argument 'path' for read_file"),
        ("read_file", {"path": "/a", "mode": "r", "x": 1}, "error: unexpected argument(s)"),
        ("read_file", {"path": ["/a"]}, "error: argument 'path' for read_file must be a string"),
        ("read_file", {"path": None}, "error: argument 'path' for read_file must be a string"),
        ("submit", {"answer": True}, "error: argument 'answer' for submit must be a string"),
    ],
)
def test_bad_calls_become_observations(name, args, start):
    env = make_env()
    obs = env.call(name, args)
    assert obs.startswith(start)
    assert env.calls == 1
    entry = env.trace[0]
    assert entry.error
    assert entry.observation_length == len(obs)
    assert not env.done


def test_numbers_are_accepted_as_strings():
    env = make_env()
    assert env.call("calc", {"expression": 7}) == "7"
    env.call("submit", {"answer": 1234.5})
    assert env.answer == "1234.5"


def test_submit_ends_the_episode():
    env = make_env()
    assert env.call("submit", {"answer": " 42 "}) == "answer submitted; the episode has ended"
    assert env.done and env.end_reason == "submitted" and env.answer == " 42 "
    obs = env.call("read_file", {"path": "/data/a.txt"})
    assert obs == "error: the episode has ended; no more tool calls are accepted"
    assert env.calls == 2
    assert env.answer == " 42 "


def test_call_limit_ends_the_episode():
    env = make_env(max_calls=2)
    env.call("list_dir")
    assert not env.done
    env.call("read_file", {"path": "/data/a.txt"})
    assert env.done and env.end_reason == "step_limit" and env.answer is None
    assert env.call("submit", {"answer": "x"}).startswith("error: the episode has ended")
    assert env.answer is None
    last = make_env(max_calls=1)
    last.call("submit", {"answer": "x"})
    assert last.end_reason == "submitted"


def test_max_calls_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        make_env(max_calls=0)


def test_trace_records_tool_args_and_observation_length():
    env = make_env()
    first = env.call("read_file", {"path": "/data/a.txt"})
    env.call("search", ["bad"])
    env.call("list_dir", None)
    tools_used = [entry.tool for entry in env.trace]
    assert tools_used == ["read_file", "search", "list_dir"]
    assert env.trace[0].args == {"path": "/data/a.txt"}
    assert env.trace[0].observation_length == len(first)
    assert not env.trace[0].error
    assert env.trace[1].args == {"_raw": "['bad']"}
    assert env.trace[2].args == {}
    assert env.calls == 3


def test_internal_errors_become_observations(monkeypatch):
    def broken(self, path):
        raise RuntimeError("boom")

    monkeypatch.setattr(WorkingCopy, "read", broken)
    env = make_env()
    assert env.call("read_file", {"path": "/data/a.txt"}) == (
        "error: internal tool error (RuntimeError)"
    )
    assert env.trace[0].error


def test_episodes_do_not_share_state():
    snap = Snapshot(FILES)
    one, two = ToolEnv(snap), ToolEnv(snap)
    one.call("write_file", {"path": "/data/a.txt", "content": "mine"})
    assert two.call("read_file", {"path": "/data/a.txt"}) == FILES["/data/a.txt"]
    assert snap.read("/data/a.txt") == FILES["/data/a.txt"]
