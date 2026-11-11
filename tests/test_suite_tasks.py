"""Tests for nf-tools task families, the scripted oracle, suites, splits, and manifests.

The oracle solves each task through the tools only (list_dir, read_file, search,
calc, write_file, submit), reading its parameters from the instruction text. It
never looks at the snapshot or the expected answer, so a pass shows that the
instruction and the files are enough to solve the task.
"""

import itertools
import json
import posixpath
import re

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from noisefloor.suite import grading
from noisefloor.suite.suite import (
    SUITE_NAME,
    SUITE_VERSION,
    build_suite,
    canonical_json,
    check_disjoint,
    split,
    suite_manifest,
    task_sha256,
)
from noisefloor.suite.tasks import DIFFICULTIES, FAMILIES, FAMILY_NAMES, generate, task_id
from noisefloor.suite.tools import ERROR_PREFIX, READ_LIMIT, ToolEnv

SEEDS = range(8)
EM_DASH = chr(0x2014)  # a code point, so this file itself contains no em dash

# Pinned so CI on Python 3.11 and 3.12 checks that generation is byte-identical. A
# deliberate change to any generator changes this hash and needs a new SUITE_VERSION.
GOLDEN_SHA256 = "6aad59b36ac1fd12af7a0289ebcf9917c425243bf7c3a6014e87d7ae843bb059"  # v2


# ---------------------------------------------------------------------------
# scripted oracle (tools only)


def call(env: ToolEnv, tool: str, **args: str) -> str:
    observation = env.call(tool, args)
    assert not observation.startswith(ERROR_PREFIX), (tool, args, observation)
    return observation


def walk(env: ToolEnv, root: str) -> list[str]:
    found = []
    for entry in call(env, "list_dir", path=root).splitlines():
        path = posixpath.join(root, entry.rstrip("/"))
        found.extend(walk(env, path) if entry.endswith("/") else [path])
    return found


def first(pattern: str, text: str) -> str:
    match = re.search(pattern, text)
    assert match, (pattern, text)
    return match.group(1)


def parse_cfg(text: str) -> dict[tuple[str, str], str]:
    section, values = "", {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        if line.startswith("["):
            section = line.strip("[]")
            continue
        key, _, value = line.partition(" = ")
        values[(section, key)] = value
    return values


def solve_config(env: ToolEnv, text: str) -> str:
    key = first(r"value of the `([^`]+)` key", text)
    section_match = re.search(r"in the `\[([^\]]+)\]` section", text)
    section = section_match.group(1) if section_match else ""
    where = first(r"the settings for (.+?)\. Settings", text)
    path = first(r"Settings live under `([^`]+)`", text)
    for value in re.findall(r"\w+ `([^`]+)`", where):
        assert value + "/" in call(env, "list_dir", path=path).splitlines()
        path = f"{path}/{value}"
    if "`local.cfg` overrides" in text:
        local = parse_cfg(call(env, "read_file", path=f"{path}/local.cfg"))
        if (section, key) in local:
            return local[(section, key)]
        return parse_cfg(call(env, "read_file", path=f"{path}/defaults.cfg"))[(section, key)]
    return parse_cfg(call(env, "read_file", path=f"{path}/settings.cfg"))[(section, key)]


def solve_csv_sum(env: ToolEnv, text: str) -> str:
    """Search-first: every CSV under the base starts with a header naming `amount`."""
    base = first(r"under `([^`]+)` \(including", text)
    conditions = re.findall(r"`(\w+)` is `([^`]+)`", text)
    hits = call(env, "search", pattern="amount", path=base).splitlines()
    assert not any(hit.startswith("[") for hit in hits), hits  # nothing cut from the listing
    paths = sorted({hit.split(":", 1)[0] for hit in hits})
    amounts = []
    for path in paths:
        if not path.endswith(".csv"):
            continue
        header, *lines = call(env, "read_file", path=path).splitlines()
        for line in lines:
            row = dict(zip(header.split(","), line.split(","), strict=True))
            if all(row[column] == value for column, value in conditions):
                amounts.append(row["amount"])
    return call(env, "calc", expression=" + ".join(amounts))


def solve_pointer(env: ToolEnv, text: str) -> str:
    current = first(r"Start at `([^`]+)`", text)
    while True:
        lines = call(env, "read_file", path=current).splitlines()
        codes = [line.removeprefix("code:").strip() for line in lines if line.startswith("code:")]
        if codes:
            return codes[0]
        link = next(
            line.removeprefix("next:").strip() for line in lines if line.startswith("next:")
        )
        if not link.startswith("/"):
            link = posixpath.normpath(posixpath.join(posixpath.dirname(current), link))
        current = link


def solve_derive(env: ToolEnv, text: str) -> str:
    source = first(r"directly in `([^`]+)`", text)
    out = first(r"Write `([^`]+)`", text)
    category = first(r"`category` is `([^`]+)`", text)
    below = re.search(r"is below (\d+)", text)
    order = re.search(r"sorted by (?:`qty`|total) in (ascending|descending) order", text)
    totals: dict[str, int] = {}
    kinds: dict[str, str] = {}
    for entry in call(env, "list_dir", path=source).splitlines():
        if not entry.endswith(".csv"):
            continue
        for line in call(env, "read_file", path=f"{source}/{entry}").splitlines()[1:]:
            sku, _, kind, qty = line.split(",")
            totals[sku] = totals.get(sku, 0) + int(qty)
            kinds[sku] = kind
    chosen = [
        sku
        for sku in totals
        if kinds[sku] == category and (below is None or totals[sku] < int(below.group(1)))
    ]
    if order:
        sign = 1 if order.group(1) == "ascending" else -1
        chosen.sort(key=lambda sku: (sign * totals[sku], sku))
    else:
        chosen.sort()
    pairs = "written as `sku," in text
    body = "".join((f"{sku},{totals[sku]}" if pairs else sku) + "\n" for sku in chosen)
    assert call(env, "write_file", path=out, content=body).startswith("wrote")
    return "done"


def solve_violation(env: ToolEnv, text: str) -> str:
    if "`/rules/limits.csv`" in text:
        rows = call(env, "read_file", path="/rules/limits.csv").splitlines()[1:]
        table = {(c, z): int(m) for c, z, m in (row.split(",") for row in rows)}

        def limit(rec):
            return table[(rec["carrier"], rec["zone"])]
    elif "`/rules/carriers.txt`" in text:
        lines = call(env, "read_file", path="/rules/carriers.txt").splitlines()
        carriers = dict(line.split(": ") for line in lines if not line.startswith("#"))

        def limit(rec):
            return int(carriers[rec["carrier"]])
    elif "its own `declared_kg`" in text:

        def limit(rec):
            return int(rec["declared_kg"])
    else:
        cap = int(first(r"at most (\d+)", text))

        def limit(rec):
            return cap

    broken = []
    for path in walk(env, "/records"):
        if path.endswith(".txt"):
            for line in call(env, "read_file", path=path).splitlines():
                record = dict(field.split("=", 1) for field in line.split())
                if int(record["weight_kg"]) > limit(record):
                    broken.append(record["id"])
    assert len(broken) == 1, broken
    return broken[0]


SOLVERS = {
    "config": solve_config,
    "csv_sum": solve_csv_sum,
    "pointer": solve_pointer,
    "derive": solve_derive,
    "violation": solve_violation,
}


E1_BUDGET = 15  # tool calls per episode in E1 (local arms)


def run_oracle(task):
    env = task.start(max_calls=E1_BUDGET)
    answer = SOLVERS[task.family](env, task.instruction)
    call(env, "submit", answer=answer)
    assert env.done and env.end_reason == "submitted"
    return env


# ---------------------------------------------------------------------------
# reference solutions


def test_every_family_has_a_solver():
    assert set(SOLVERS) == set(FAMILY_NAMES)
    assert len(FAMILY_NAMES) >= 5


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
@pytest.mark.parametrize("family", FAMILY_NAMES)
def test_oracle_solves_every_family_and_difficulty(family, difficulty):
    for seed in SEEDS:
        task = generate(family, seed, difficulty)
        env = run_oracle(task)
        result = task.grade_env(env)
        assert result.score == 1 and result.strict in (1, None), (task.id, result)
        assert not any(entry.error for entry in env.trace)


@pytest.mark.parametrize("family", FAMILY_NAMES)
def test_steps_needed_grow_with_difficulty(family):
    def mean_calls(difficulty):
        return sum(run_oracle(generate(family, s, difficulty)).calls for s in SEEDS) / len(SEEDS)

    low, mid, high = mean_calls(1), mean_calls(3), mean_calls(5)
    assert low < mid < high


@pytest.mark.property
@settings(max_examples=60, deadline=None)
@given(
    st.sampled_from(FAMILY_NAMES),
    st.integers(min_value=0, max_value=2**48),
    st.sampled_from(DIFFICULTIES),
)
def test_property_oracle_solves_any_seed(family, seed, difficulty):
    task = generate(family, seed, difficulty)
    assert task.grade_env(run_oracle(task)).passed, task.id


# ---------------------------------------------------------------------------
# wrong answers


def other_answers(task):
    """Plausible wrong answers taken from the task's own distractors."""
    files = task.files.files
    if task.family == "config":
        key = first(r"value of the `([^`]+)` key", task.instruction)
        values = {m for text in files.values() for m in re.findall(rf"^{key} = (.+)$", text, re.M)}
    elif task.family == "pointer":
        pattern = r"^(?:previous )?code: (.+)$"
        values = {m for text in files.values() for m in re.findall(pattern, text, re.M)}
    elif task.family == "violation":
        values = {m for text in files.values() for m in re.findall(r"id=(\S+)", text)}
    else:
        values = set()
    return sorted(values - {task.expected.expected})


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
@pytest.mark.parametrize("family", ["config", "pointer", "violation"])
def test_distractor_answers_grade_zero(family, difficulty):
    for seed in SEEDS:
        task = generate(family, seed, difficulty)
        wrong = other_answers(task)
        assert wrong, task.id
        for answer in wrong:
            assert task.grade(answer, task.files).score == 0, (task.id, answer)


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
@pytest.mark.parametrize("family", ["config", "csv_sum", "pointer", "violation"])
def test_wrong_and_missing_answers_grade_zero(family, difficulty):
    task = generate(family, 3, difficulty)
    expected = task.expected.expected
    assert task.grade(None, task.files) == grading.Grade(0, "no answer submitted", strict=0)
    assert task.grade("", task.files).score == 0
    assert task.grade(expected + "9", task.files).score == 0
    assert task.grade(expected, task.files).score == 1
    if task.expected.mode == "numeric":
        assert task.grade(str(float(expected) + 0.01), task.files).score == 0


@pytest.mark.parametrize("difficulty", [2, 3, 4, 5])
def test_csv_sum_distractor_files_change_the_total(difficulty):
    for seed in SEEDS:
        task = generate("csv_sum", seed, difficulty)
        env = task.start()
        conditions = re.findall(r"`(\w+)` is `([^`]+)`", task.instruction)
        cents = 0
        for path in walk(env, "/data"):
            header, *lines = env.call("read_file", {"path": path}).splitlines()
            for line in lines:
                row = dict(zip(header.split(","), line.split(","), strict=False))
                if all(row.get(column) == value for column, value in conditions):
                    whole, _, frac = row["amount"].partition(".")
                    cents += int(whole) * 100 + int(frac or "0")
        assert task.grade(f"{cents / 100:.2f}", task.files).score == 0, task.id


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_derive_wrong_files_grade_zero(difficulty):
    task = generate("derive", 5, difficulty)
    path, expected = task.expected.path, task.expected.expected
    lines = expected.splitlines()
    assert len(lines) >= 2
    assert task.grade("done", task.files).reason == f"{path} is missing"

    def graded(content, where=path):
        env = task.start()
        env.call("write_file", {"path": where, "content": content})
        return task.grade_env(env).score

    assert graded(expected) == 1
    assert graded("\n".join(lines[:-1])) == 0
    assert graded("\n".join(lines[::-1])) == 0
    assert graded(expected + "SKU-0000\n") == 0
    assert graded(expected, where=path + ".bak") == 0


# ---------------------------------------------------------------------------
# determinism and structure


@pytest.mark.parametrize("family", FAMILY_NAMES)
def test_generation_is_deterministic(family):
    for difficulty in DIFFICULTIES:
        for seed in range(4):
            one, two = generate(family, seed, difficulty), generate(family, seed, difficulty)
            assert one == two
            assert canonical_json(one) == canonical_json(two)
            assert hash(one) == hash(two)


@pytest.mark.parametrize("family", FAMILY_NAMES)
def test_different_seeds_and_difficulties_differ(family):
    for difficulty in DIFFICULTIES:
        tasks = [generate(family, seed, difficulty) for seed in range(6)]
        assert len({task.files for task in tasks}) == len(tasks)
        assert len({task_sha256(task) for task in tasks}) == len(tasks)
    by_difficulty = [generate(family, 0, d).files for d in DIFFICULTIES]
    assert len(set(by_difficulty)) == len(DIFFICULTIES)


@pytest.mark.parametrize("family", FAMILY_NAMES)
def test_task_fields(family):
    for difficulty in DIFFICULTIES:
        for seed in SEEDS:
            task = generate(family, seed, difficulty)
            assert task.id == task_id(family, seed, difficulty) == f"{family}-d{difficulty}-s{seed}"
            assert (task.family, task.seed, task.difficulty) == (family, seed, difficulty)
            assert task.knobs == FAMILIES[family].levels[difficulty].knobs()
            for path, content in task.files.files.items():
                assert len(content) <= READ_LIMIT, path
                assert content.isascii(), path
            assert task.instruction.isascii()
            assert EM_DASH not in task.instruction
            canonical = task.canonical()
            assert canonical["files"] == task.files.to_dict()
            assert canonical["knobs"] == dict(task.knobs)
            assert canonical["expected"] == task.expected.canonical()


@pytest.mark.parametrize("family", FAMILY_NAMES)
def test_knobs_never_decrease_with_difficulty(family):
    levels = FAMILIES[family].levels
    assert sorted(levels) == list(DIFFICULTIES)
    for low, high in itertools.pairwise(DIFFICULTIES):
        a, b = dict(levels[low].knobs()), dict(levels[high].knobs())
        assert a.keys() == b.keys()
        assert all(b[name] >= a[name] for name in a), (family, low, high)
        assert sum(b.values()) > sum(a.values()), (family, low, high)
    files = [len(generate(family, 0, d).files) for d in DIFFICULTIES]
    assert files[-1] > files[0]


def test_family_metadata():
    for name, family in FAMILIES.items():
        assert family.name == name
        assert family.summary.endswith(".")
        assert EM_DASH not in family.summary


@pytest.mark.parametrize(
    ("family", "seed", "difficulty", "message"),
    [
        ("nope", 0, 1, "unknown family"),
        ("pointer", -1, 1, "non-negative integer"),
        ("pointer", True, 1, "non-negative integer"),
        ("pointer", 1.5, 1, "non-negative integer"),
        ("pointer", 0, 0, "difficulty must be one of"),
        ("pointer", 0, 6, "difficulty must be one of"),
        ("pointer", 0, True, "difficulty must be one of"),
        ("pointer", 0, "3", "difficulty must be one of"),
    ],
)
def test_generate_rejects_bad_arguments(family, seed, difficulty, message):
    with pytest.raises(ValueError, match=message):
        generate(family, seed, difficulty)


def test_start_and_grade_env():
    task = generate("pointer", 1, 1)
    env = task.start(max_calls=1)
    assert env.instruction == task.instruction
    env.call("list_dir", {})
    assert env.done and env.end_reason == "step_limit"
    assert task.grade_env(env) == grading.Grade(0, "no answer submitted", strict=0)


# ---------------------------------------------------------------------------
# suites, splits, manifests


def test_build_suite_order_and_size():
    tasks = build_suite([5, 2], families=["pointer", "config"], difficulties=[3, 1])
    assert [task.id for task in tasks] == [
        "pointer-d3-s5",
        "pointer-d3-s2",
        "pointer-d1-s5",
        "pointer-d1-s2",
        "config-d3-s5",
        "config-d3-s2",
        "config-d1-s5",
        "config-d1-s2",
    ]
    assert len(build_suite(range(2))) == 2 * len(FAMILY_NAMES) * len(DIFFICULTIES)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"seeds": []}, "non-empty"),
        ({"seeds": [1], "families": []}, "non-empty"),
        ({"seeds": [1], "difficulties": []}, "non-empty"),
        ({"seeds": [1, 1]}, "duplicates"),
        ({"seeds": [-1]}, "non-negative"),
        ({"seeds": [True]}, "non-negative"),
        ({"seeds": [1], "families": ["config", "config"]}, "must not repeat"),
        ({"seeds": [1], "difficulties": [2, 2]}, "must not repeat"),
        ({"seeds": [1], "families": ["config", "nope"]}, "unknown families: nope"),
        ({"seeds": [1], "difficulties": [9]}, "difficulty must be one of"),
    ],
)
def test_build_suite_rejects_bad_arguments(kwargs, message):
    with pytest.raises(ValueError, match=message):
        build_suite(**kwargs)


def test_split_keeps_pilot_and_confirmatory_disjoint():
    parts = split(range(0, 3), range(100, 104), families=["derive"], difficulties=[2])
    assert [task.seed for task in parts.pilot] == [0, 1, 2]
    assert [task.seed for task in parts.confirmatory] == [100, 101, 102, 103]
    pilot_ids = {task.id for task in parts.pilot}
    assert pilot_ids.isdisjoint(task.id for task in parts.confirmatory)


def test_overlapping_seeds_raise():
    with pytest.raises(ValueError, match="overlap: 5, 6"):
        split(range(0, 7), range(5, 20))
    with pytest.raises(ValueError, match=r"overlap: 3$"):
        check_disjoint([1, 3], [3, 4])
    with pytest.raises(ValueError, match=r"overlap: 0, 1, .*, 9 and 5 more"):
        check_disjoint(range(15), range(15))
    with pytest.raises(ValueError, match="duplicates"):
        check_disjoint([1, 1], [2])
    check_disjoint(range(0, 1000), range(1000, 2000))


def test_manifest_contents():
    tasks = build_suite([4, 1], families=["violation", "csv_sum"], difficulties=[2])
    manifest = suite_manifest(tasks)
    assert manifest["suite"] == SUITE_NAME
    assert manifest["version"] == SUITE_VERSION
    assert manifest["procedurally_generated"] is True
    assert manifest["n_tasks"] == 4
    assert manifest["families"] == ["csv_sum", "violation"]
    assert manifest["difficulties"] == [2]
    assert manifest["seeds"] == [1, 4]
    assert manifest["task_ids"] == sorted(task.id for task in tasks)
    assert manifest["task_sha256"] == {task.id: task_sha256(task) for task in tasks}
    body = "[" + ",".join(canonical_json(t) for t in sorted(tasks, key=lambda t: t.id)) + "]"
    assert json.loads(body)[0]["id"] == manifest["task_ids"][0]
    assert len(manifest["sha256"]) == 64
    json.dumps(manifest)


def test_canonical_json_is_canonical():
    task = generate("config", 2, 4)
    text = canonical_json(task)
    assert text.isascii()
    assert ": " not in text.split('"files"')[0]
    assert json.loads(text) == task.canonical()
    assert text == json.dumps(json.loads(text), sort_keys=True, separators=(",", ":"))


def test_manifest_rejects_empty_and_repeated_tasks():
    with pytest.raises(ValueError, match="no tasks"):
        suite_manifest([])
    task = generate("pointer", 0, 1)
    with pytest.raises(ValueError, match="repeat"):
        suite_manifest([task, task])


def test_manifest_is_stable_and_seed_sensitive():
    first_hash = suite_manifest(build_suite(range(3)))["sha256"]
    assert suite_manifest(build_suite(range(3)))["sha256"] == first_hash
    assert suite_manifest(build_suite(range(1, 4)))["sha256"] != first_hash
    assert suite_manifest(build_suite([2, 0, 1]))["sha256"] == first_hash


def test_golden_manifest_hash():
    manifest = suite_manifest(build_suite(range(20)))
    assert manifest["n_tasks"] == 500
    assert manifest["sha256"] == GOLDEN_SHA256


@pytest.mark.property
@settings(max_examples=25, deadline=None)
@given(
    st.lists(st.integers(min_value=0, max_value=10**9), min_size=1, max_size=4, unique=True),
    st.lists(st.sampled_from(FAMILY_NAMES), min_size=1, max_size=3, unique=True),
    st.lists(st.sampled_from(DIFFICULTIES), min_size=1, max_size=2, unique=True),
    st.randoms(use_true_random=False),
)
def test_property_manifest_is_stable(seeds, families, difficulties, shuffler):
    manifest = suite_manifest(build_suite(seeds, families, difficulties))
    assert suite_manifest(build_suite(seeds, families, difficulties)) == manifest
    tasks = build_suite(seeds[::-1], families[::-1], difficulties)
    shuffler.shuffle(tasks)
    assert suite_manifest(tasks) == manifest
    assert manifest["n_tasks"] == len(seeds) * len(families) * len(difficulties)
    assert manifest["seeds"] == sorted(seeds)


# ---------------------------------------------------------------------------
# suite v2 checks (D-021)


def pointer_chain(files):
    """The notes on the link chain from /start.txt, in order (the last holds the code)."""
    chain, current = [], "/start.txt"
    while True:
        chain.append(current)
        lines = files[current].splitlines()
        if any(line.startswith("code: ") for line in lines):
            return chain
        link = next(line.removeprefix("next: ") for line in lines if line.startswith("next: "))
        if not link.startswith("/"):
            link = posixpath.normpath(posixpath.join(posixpath.dirname(current), link))
        current = link


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_no_pointer_line_type_marks_the_chain(difficulty):
    """A search for a distractor line type must not single out the chain or its end."""
    for seed in SEEDS:
        task = generate("pointer", seed, difficulty)
        files = task.files.files
        chain = pointer_chain(files)
        for prefix in ("old: ", "previous code: "):
            holders = {
                path
                for path, text in files.items()
                if any(line.startswith(prefix) for line in text.splitlines())
            }
            if holders:
                assert holders - set(chain), (task.id, prefix, "only chain notes carry it")
                assert holders != {chain[-1]}, (task.id, prefix, "only the chain end carries it")


@pytest.mark.parametrize("difficulty", [3, 4, 5])
def test_derive_distractors_would_change_the_output(difficulty):
    """Reading a distractor table by mistake would add at least one line to the output."""
    for seed in SEEDS:
        task = generate("derive", seed, difficulty)
        category = first(r"`category` is `([^`]+)`", task.instruction)
        below = re.search(r"is below (\d+)", task.instruction)
        for path, text in task.files.files.items():
            if not (path.startswith("/inventory/archive/") or path.endswith(".csv.bak")):
                continue
            rows = [line.split(",") for line in text.splitlines()[1:]]
            extra = [
                sku
                for sku, _, kind, qty in rows
                if kind == category and (below is None or int(qty) < int(below.group(1)))
            ]
            assert any(sku not in task.expected.expected for sku in extra), (task.id, path)


def test_instructions_ask_for_values_and_files_plainly():
    config = generate("config", 5, 3).instruction
    assert "not the key name" in config
    assert "Do not write a header line" in generate("derive", 5, 4).instruction
