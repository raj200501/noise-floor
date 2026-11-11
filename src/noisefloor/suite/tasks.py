"""Task families of the nf-tools suite.

Every task is procedurally generated mock data: a small file system, an
instruction, and an expectation that grades the outcome. Each family is a
:class:`Family` whose ``generate(seed, difficulty)`` builds a :class:`Task`. The
only randomness is a local :class:`random.Random` seeded from a SHA-256 of
(family, seed, difficulty), and generation uses only its ``randrange``,
``randint``, ``choice``, ``sample``, ``shuffle``, and ``random`` methods. Sets are
used for membership tests only, never iterated. So a (family, seed, difficulty)
triple always yields byte-identical tasks on Python 3.11 and 3.12.

Families (difficulty 1 to 5 raises depth, file counts, distractors, and the
number of steps; each family's table of knobs is in ``Family.levels``):

* ``config``: look up a key in nested settings directories with distractors.
* ``csv_sum``: sum a numeric column over filtered rows of several CSV files.
* ``pointer``: follow ``next:`` links from note to note until a code.
* ``derive``: write a filtered, sorted (at level 5, aggregated) list to a file.
* ``violation``: find the one record that breaks a stated rule.
"""

from __future__ import annotations

import hashlib
import posixpath
import random
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, fields
from typing import Any, Generic, TypeVar

from noisefloor.suite import grading
from noisefloor.suite.fs import FileView, Snapshot
from noisefloor.suite.grading import AnswerSpec, Expectation, FileSpec, Grade
from noisefloor.suite.tools import ToolEnv

DIFFICULTIES: tuple[int, ...] = (1, 2, 3, 4, 5)

_WORDS = (
    "amber",
    "basalt",
    "cedar",
    "delta",
    "ember",
    "fjord",
    "garnet",
    "harbor",
    "indigo",
    "juniper",
    "kelp",
    "lumen",
    "maple",
    "nimbus",
    "onyx",
    "pebble",
    "quartz",
    "raven",
    "sable",
    "tundra",
    "umber",
    "vale",
    "willow",
    "yarrow",
    "zephyr",
)
_UPPER = "ABCDEFGHJKLMNPQRSTUVWXYZ"
_ALNUM = "abcdefghjkmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789"
_FILLER = (
    "This note is part of a mock file system.",
    "Nothing on this line is needed.",
    "The weekly sync moved to the afternoon.",
    "Keep this folder tidy.",
    "Ask the archive team about older copies.",
    "Draft text, not reviewed.",
    "See the index for related notes.",
    "Dates in this folder use the local calendar.",
)


@dataclass(frozen=True)
class Task:
    """One generated task: its files, the instruction, and how to grade it.

    ``knobs`` records the difficulty settings that produced the task, as sorted
    (name, value) pairs.
    """

    id: str
    family: str
    seed: int
    difficulty: int
    instruction: str
    files: Snapshot
    expected: Expectation
    knobs: tuple[tuple[str, int], ...]

    def start(self, *, max_calls: int | None = None) -> ToolEnv:
        """A fresh episode on this task, with its own working copy of the files."""
        return ToolEnv(self.files, instruction=self.instruction, max_calls=max_calls)

    def grade(self, answer: str | None, files: FileView) -> Grade:
        """Grade a submitted answer (None if none) and the final file system."""
        return grading.grade(self.expected, answer, files)

    def grade_env(self, env: ToolEnv) -> Grade:
        """Grade the outcome of an episode."""
        return self.grade(env.answer, env.files)

    def canonical(self) -> dict[str, Any]:
        """Plain-data form with sorted files, for hashing and manifests."""
        return {
            "id": self.id,
            "family": self.family,
            "seed": self.seed,
            "difficulty": self.difficulty,
            "instruction": self.instruction,
            "files": self.files.to_dict(),
            "expected": self.expected.canonical(),
            "knobs": dict(self.knobs),
        }


@dataclass(frozen=True)
class _Level:
    """Base of a family's difficulty row; every field is an integer knob."""

    def knobs(self) -> tuple[tuple[str, int], ...]:
        """The fields as sorted (name, value) pairs."""
        return tuple(sorted((f.name, int(getattr(self, f.name))) for f in fields(self)))


@dataclass(frozen=True)
class _Draft:
    instruction: str
    files: Mapping[str, str]
    expected: Expectation


LevelT = TypeVar("LevelT", bound=_Level)


@dataclass(frozen=True)
class Family(Generic[LevelT]):
    """A task family: a name, a one-line summary, a difficulty table, and a builder."""

    name: str
    summary: str
    levels: Mapping[int, LevelT]
    builder: Callable[[random.Random, LevelT], _Draft]

    def generate(self, seed: int, difficulty: int) -> Task:
        """Build the task for ``seed`` (a non-negative integer) and ``difficulty`` (1 to 5)."""
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(f"seed must be a non-negative integer, got {seed!r}")
        level = self.levels.get(difficulty)
        if level is None or isinstance(difficulty, bool):
            raise ValueError(f"difficulty must be one of {DIFFICULTIES}, got {difficulty!r}")
        draft = self.builder(_rng(self.name, seed, difficulty), level)
        return Task(
            id=task_id(self.name, seed, difficulty),
            family=self.name,
            seed=seed,
            difficulty=difficulty,
            instruction=draft.instruction,
            files=Snapshot(draft.files),
            expected=draft.expected,
            knobs=level.knobs(),
        )


def task_id(family: str, seed: int, difficulty: int) -> str:
    """The task id for a (family, seed, difficulty) triple, such as ``pointer-d3-s42``."""
    return f"{family}-d{difficulty}-s{seed}"


def _rng(family: str, seed: int, difficulty: int) -> random.Random:
    digest = hashlib.sha256(f"nf-tools:{family}:{seed}:{difficulty}".encode("ascii")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _distinct(draw: Callable[[], str], avoid: Collection[str]) -> str:
    while True:
        value = draw()
        if value not in avoid:
            return value


def _unique(draw: Callable[[], str], count: int) -> list[str]:
    """``count`` distinct values from ``draw``, in the order drawn."""
    values: list[str] = []
    while len(values) < count:
        value = draw()
        if value not in values:
            values.append(value)
    return values


def _code(rng: random.Random) -> str:
    return f"{rng.choice(_UPPER)}{rng.choice(_UPPER)}-{rng.randrange(1000, 10000)}"


# ---------------------------------------------------------------------------
# config: look up a key in nested settings directories


@dataclass(frozen=True)
class _ConfigLevel(_Level):
    depth: int
    distractor_dirs: int
    keys: int
    sections: int
    backup: int
    notes: int
    layered: int


_CONFIG_LEVELS = {
    1: _ConfigLevel(depth=1, distractor_dirs=2, keys=3, sections=1, backup=0, notes=0, layered=0),
    2: _ConfigLevel(depth=2, distractor_dirs=4, keys=4, sections=1, backup=1, notes=0, layered=0),
    3: _ConfigLevel(depth=3, distractor_dirs=6, keys=4, sections=2, backup=1, notes=1, layered=0),
    4: _ConfigLevel(depth=4, distractor_dirs=8, keys=5, sections=2, backup=1, notes=1, layered=1),
    5: _ConfigLevel(depth=5, distractor_dirs=10, keys=5, sections=3, backup=1, notes=1, layered=1),
}
_CONFIG_ROOT = "/etc/apps"
_DIMENSIONS = (
    (
        "service",
        (
            "billing",
            "search",
            "ledger",
            "gateway",
            "notify",
            "catalog",
            "auth",
            "reports",
            "media",
            "scheduler",
        ),
    ),
    ("environment", ("dev", "qa", "staging", "prod", "sandbox")),
    ("region", ("north", "south", "east", "west", "central")),
    ("cluster", ("blue", "green", "amber", "violet", "gray")),
    ("tier", ("web", "worker", "batch", "edge")),
)
_SECTIONS = ("server", "database", "cache", "queue", "storage")
_ASKED_KEYS = ("port", "timeout_ms", "max_connections", "cache_ttl", "token", "bucket")
_OTHER_KEYS = (
    "log_level",
    "owner",
    "retries",
    "pool_size",
    "enabled",
    "region_hint",
    "batch_size",
    "endpoint",
)
_INT_RANGES = {
    "port": (1024, 65535),
    "timeout_ms": (100, 30000),
    "max_connections": (8, 4096),
    "cache_ttl": (30, 86400),
    "retries": (0, 9),
    "pool_size": (2, 256),
    "batch_size": (10, 1000),
}
_CHOICES = {
    "log_level": ("debug", "info", "warn", "error"),
    "enabled": ("true", "false"),
    "region_hint": ("north", "south", "east", "west", "central"),
}

Settings = dict[str, dict[str, str]]


def _config_value(rng: random.Random, key: str) -> str:
    if key in _INT_RANGES:
        low, high = _INT_RANGES[key]
        return str(rng.randint(low, high))
    if key in _CHOICES:
        return rng.choice(_CHOICES[key])
    if key == "token":
        return "tk-" + "".join(rng.choice(_ALNUM) for _ in range(8))
    if key == "bucket":
        return f"bkt-{rng.choice(_WORDS)}-{rng.randrange(100, 1000)}"
    if key == "owner":
        return f"team-{rng.choice(_WORDS)}"
    return f"{rng.choice(_WORDS)}-{rng.randrange(10, 100)}.mock.internal"


def _render_cfg(title: str, settings: Settings, sectioned: bool) -> str:
    lines = [f"# {title}"]
    for section, pairs in settings.items():
        if sectioned:
            lines.append(f"[{section}]")
        lines.extend(f"{key} = {value}" for key, value in pairs.items())
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class _ConfigPlan:
    rng: random.Random
    level: _ConfigLevel
    key: str
    section: str
    schema: dict[str, list[str]]
    answer: str

    @property
    def sectioned(self) -> bool:
        return self.level.sections > 1

    @property
    def main(self) -> str:
        return "defaults.cfg" if self.level.layered else "settings.cfg"

    def other(self) -> str:
        """A value for the asked key that differs from the answer."""
        return _distinct(lambda: _config_value(self.rng, self.key), (self.answer,))

    def value(self, key: str) -> str:
        return self.other() if key == self.key else _config_value(self.rng, key)

    def fill(self, keys_by_section: Mapping[str, Collection[str]]) -> Settings:
        return {name: {k: self.value(k) for k in keys} for name, keys in keys_by_section.items()}


def _config_target(plan: _ConfigPlan) -> tuple[Settings, Settings]:
    """Main and local settings of the target directory."""
    main = plan.fill(plan.schema)
    rng, key, section = plan.rng, plan.key, plan.section
    if plan.level.layered and rng.random() < 0.5:
        extra = rng.choice([k for k in plan.schema[section] if k != key])
        return main, {section: {key: plan.answer, extra: _config_value(rng, extra)}}
    main[section][key] = plan.answer
    if not plan.level.layered:
        return main, {}
    decoy = rng.choice([name for name in plan.schema if name != section])
    return main, {decoy: {key: plan.other()}}


def _config_leaf(plan: _ConfigPlan, leaf: tuple[str, ...], is_target: bool) -> dict[str, str]:
    rng, key, section = plan.rng, plan.key, plan.section
    directory = posixpath.join(_CONFIG_ROOT, *leaf)
    title = " / ".join(leaf)
    if is_target:
        main, local = _config_target(plan)
    else:
        main = plan.fill(plan.schema)
        name = rng.choice(list(plan.schema))
        local = plan.fill({name: rng.sample(plan.schema[name], 2)}) if plan.level.layered else {}
    files = {f"{directory}/{plan.main}": _render_cfg(f"settings for {title}", main, plan.sectioned)}
    if plan.level.layered:
        files[f"{directory}/local.cfg"] = _render_cfg(
            f"local overrides for {title}", local, plan.sectioned
        )
    if is_target and plan.level.backup:
        stale = {name: dict(pairs) for name, pairs in main.items()}
        stale[section][key] = plan.other()
        files[f"{directory}/{plan.main}.bak"] = _render_cfg(
            f"backup of settings for {title}", stale, plan.sectioned
        )
    if is_target and plan.level.notes:
        where = f" in [{section}]" if plan.sectioned else ""
        files[f"{directory}/NOTES.txt"] = (
            f"Change log: {key}{where} was {plan.other()} before the last update.\n"
        )
    return files


def _build_config(rng: random.Random, level: _ConfigLevel) -> _Draft:
    dims = _DIMENSIONS[: level.depth]
    target = tuple(rng.choice(options) for _, options in dims)
    leaves = [target]
    while len(leaves) <= level.distractor_dirs:
        split = rng.randrange(level.depth)
        alternatives = [value for value in dims[split][1] if value != target[split]]
        deeper = tuple(rng.choice(options) for _, options in dims[split + 1 :])
        leaf = (*target[:split], rng.choice(alternatives), *deeper)
        if leaf not in leaves:
            leaves.append(leaf)
    key = rng.choice(_ASKED_KEYS)
    sections = rng.sample(_SECTIONS, level.sections)
    schema = {name: [key, *rng.sample(_OTHER_KEYS, level.keys - 1)] for name in sections}
    for keys in schema.values():
        rng.shuffle(keys)
    plan = _ConfigPlan(rng, level, key, rng.choice(sections), schema, _config_value(rng, key))
    files: dict[str, str] = {}
    for leaf in leaves:
        files.update(_config_leaf(plan, leaf, leaf == target))

    names = [name for name, _ in dims]
    where = ", ".join(f"{name} `{value}`" for name, value in zip(names, target, strict=True))
    layout = "one directory per service"
    if level.depth > 1:
        layout = "nested directories: " + ", then ".join(names)
    scope = f" in the `[{plan.section}]` section of" if plan.sectioned else " in"
    parts = [
        f"Find the value of the `{key}` key{scope} the settings for {where}.",
        f"Settings live under `{_CONFIG_ROOT}` in {layout}.",
        "Each settings directory holds `defaults.cfg` and `local.cfg`; a key in `local.cfg` "
        "overrides the same key in the same section of `defaults.cfg`."
        if level.layered
        else "Each settings directory holds `settings.cfg`.",
    ]
    if level.backup:
        parts.append("Files ending in `.bak` are stale backups.")
    parts.append("Submit only the value, exactly as written in the file, not the key name.")
    mode: grading.AnswerMode = "numeric" if key in _INT_RANGES else "text"
    return _Draft(" ".join(parts), files, AnswerSpec(plan.answer, mode))


# ---------------------------------------------------------------------------
# csv_sum: sum a column over filtered rows of several CSV files


@dataclass(frozen=True)
class _CsvLevel(_Level):
    files: int
    rows: int
    columns: int
    conditions: int
    decimals: int
    nested: int
    shuffled: int
    distractors: int


_CSV_LEVELS = {
    1: _CsvLevel(
        files=2, rows=5, columns=3, conditions=1, decimals=0, nested=0, shuffled=0, distractors=0
    ),
    2: _CsvLevel(
        files=3, rows=6, columns=4, conditions=1, decimals=0, nested=0, shuffled=0, distractors=1
    ),
    3: _CsvLevel(
        files=4, rows=8, columns=5, conditions=1, decimals=1, nested=1, shuffled=0, distractors=2
    ),
    4: _CsvLevel(
        files=5, rows=10, columns=5, conditions=2, decimals=1, nested=1, shuffled=0, distractors=3
    ),
    5: _CsvLevel(
        files=6, rows=12, columns=5, conditions=2, decimals=1, nested=1, shuffled=1, distractors=4
    ),
}
_DATASETS = ("sales", "orders", "invoices", "payments")
_CSV_POOLS = {
    "region": ("north", "south", "east", "west"),
    "category": ("tools", "garden", "kitchen", "toys", "books"),
    "status": ("paid", "pending", "refunded"),
}

Row = dict[str, str]


def _cents_text(cents: int, decimals: int) -> str:
    return f"{cents // 100}.{cents % 100:02d}" if decimals else str(cents // 100)


def _cents(amount: str) -> int:
    whole, _, frac = amount.partition(".")
    return int(whole) * 100 + int(frac or "0")


@dataclass(frozen=True)
class _CsvPlan:
    rng: random.Random
    level: _CsvLevel
    columns: list[str]
    targets: dict[str, str]
    ids: list[str]

    def matches(self, row: Row) -> bool:
        return all(row[column] == value for column, value in self.targets.items())

    def rows(self, count: int, matching: Collection[int]) -> list[Row]:
        """``count`` rows; exactly those at the ``matching`` indices meet every condition."""
        rng = self.rng
        out = []
        for index in range(count):
            row = {"id": self.ids.pop()}
            hit = index in matching
            breaking = None if hit else rng.choice(list(self.targets))
            for column in self.columns[1:-1]:
                pool = _CSV_POOLS[column]
                if hit and column in self.targets:
                    row[column] = self.targets[column]
                elif column == breaking:
                    row[column] = rng.choice([v for v in pool if v != self.targets[column]])
                else:
                    row[column] = rng.choice(pool)
            cents = rng.randint(100, 99999) if self.level.decimals else rng.randint(1, 500) * 100
            row["amount"] = _cents_text(cents, self.level.decimals)
            out.append(row)
        return out

    def render(self, rows: list[Row]) -> str:
        header = list(self.columns)
        if self.level.shuffled:
            self.rng.shuffle(header)
        lines = [",".join(header), *(",".join(row[c] for c in header) for row in rows)]
        return "\n".join(lines) + "\n"

    def side_file(self) -> str:
        """A CSV that must not be counted, with at least one matching row."""
        count = self.level.rows
        return self.render(self.rows(count, self.rng.sample(range(count), 1 + count // 4)))


def _build_csv_sum(rng: random.Random, level: _CsvLevel) -> _Draft:
    columns = ["id", *list(_CSV_POOLS)[: level.columns - 2], "amount"]
    conditions = ["region", *rng.sample(columns[2:-1], level.conditions - 1)]
    targets = {column: rng.choice(_CSV_POOLS[column]) for column in conditions}
    ids = [f"r{n}" for n in rng.sample(range(1000, 10000), (level.files + 3) * level.rows)]
    plan = _CsvPlan(rng, level, columns, targets, ids)
    dataset = rng.choice(_DATASETS)
    base = f"/data/{dataset}"
    total_rows = level.files * level.rows
    matching = rng.sample(range(total_rows), rng.randint(2, total_rows // 2))
    rows = plan.rows(total_rows, matching)
    files: dict[str, str] = {}
    for index in range(level.files):
        name = f"part-{index + 1:02d}.csv"
        if level.nested:
            year, quarter = rng.choice(("2024", "2025", "2026")), rng.choice(("q1", "q2", "q3"))
            name = f"{year}/{quarter}/{name}"
        chunk = rows[index * level.rows : (index + 1) * level.rows]
        files[f"{base}/{name}"] = plan.render(chunk)
    total = sum(_cents(row["amount"]) for row in rows if plan.matches(row))
    wrong = _cents_text(total + rng.randint(1, 90) * 100, level.decimals)
    distractors = [
        (f"/data/archive/{dataset}/part-01.csv", plan.side_file()),
        (f"{base}/NOTES.txt", f"Checked last month: the total was {wrong}. Recheck it.\n"),
        (f"{base}/part-{level.files + 1:02d}.csv.bak", plan.side_file()),
        (f"/data/{dataset}-old/part-01.csv", plan.side_file()),
    ]
    files.update(distractors[: level.distractors])

    where = " and ".join(f"`{column}` is `{value}`" for column, value in targets.items())
    order = " The column order can differ between files." if level.shuffled else ""
    instruction = (
        f"Compute the sum of the `amount` column over every row where {where}, in every file "
        f"whose name ends in `.csv` under `{base}` (including subdirectories). Each file starts "
        f"with a header row.{order} Submit the total as a plain number."
    )
    return _Draft(instruction, files, AnswerSpec(_cents_text(total, level.decimals), "numeric"))


# ---------------------------------------------------------------------------
# pointer: follow next: links from note to note


@dataclass(frozen=True)
class _PointerLevel(_Level):
    hops: int
    distractors: int
    filler: int
    decoys: int
    relative: int
    previous_code: int


_POINTER_LEVELS = {
    1: _PointerLevel(hops=1, distractors=2, filler=1, decoys=0, relative=0, previous_code=0),
    2: _PointerLevel(hops=2, distractors=4, filler=2, decoys=0, relative=0, previous_code=0),
    3: _PointerLevel(hops=3, distractors=6, filler=2, decoys=1, relative=0, previous_code=0),
    4: _PointerLevel(hops=5, distractors=9, filler=3, decoys=1, relative=1, previous_code=0),
    5: _PointerLevel(hops=7, distractors=12, filler=3, decoys=1, relative=1, previous_code=1),
}
_START = "/start.txt"
_NOTE_NAMES = ("note", "memo", "draft", "log", "page")


def _note_path(rng: random.Random) -> str:
    dirs = [rng.choice(_WORDS[:8]) for _ in range(rng.randint(1, 3))]
    return posixpath.join("/notes", *dirs, f"{rng.choice(_NOTE_NAMES)}-{rng.randrange(1, 100)}.txt")


def _render_notes(
    rng: random.Random, notes: Mapping[str, list[str]], filler: int
) -> dict[str, str]:
    """Each note's special lines placed at random positions among filler lines."""
    files = {}
    for path, special in notes.items():
        lines = rng.sample(_FILLER, filler)
        for line in special:
            lines.insert(rng.randrange(len(lines) + 1), line)
        files[path] = "\n".join(lines) + "\n"
    return files


def _build_pointer(rng: random.Random, level: _PointerLevel) -> _Draft:
    paths = _unique(lambda: _note_path(rng), level.hops + level.distractors)
    codes = _unique(lambda: _code(rng), 2 + level.distractors)
    chain, decoys = paths[: level.hops], paths[level.hops :]
    notes: dict[str, list[str]] = {path: [] for path in [_START, *paths]}
    sources = [_START, *chain[:-1]]
    forced = rng.randrange(len(sources))
    for index, (source, dest) in enumerate(zip(sources, chain, strict=True)):
        relative = level.relative and (index == forced or rng.random() < 0.5)
        link = posixpath.relpath(dest, posixpath.dirname(source)) if relative else dest
        notes[source].append(f"next: {link}")
    notes[chain[-1]].append(f"code: {codes[0]}")
    for index, path in enumerate(decoys):
        if index + 1 < len(decoys) and rng.random() < 0.5:
            notes[path].append(f"next: {rng.choice(decoys[index + 1 :])}")
        else:
            notes[path].append(f"code: {codes[2 + index]}")
    # Distractor lines go on every note that can carry them, chain or decoy, so that no
    # line type tells the chain apart (a search for it must not shortcut the links).
    if level.previous_code:
        spare = [c for c in _unique(lambda: _code(rng), len(notes) + len(codes)) if c not in codes]
        holders = [lines for lines in notes.values() if any(x.startswith("code: ") for x in lines)]
        for lines, code in zip(holders, spare, strict=False):  # spare is never shorter
            lines.append(f"previous code: {code}")
    if level.decoys:
        for path, lines in notes.items():
            lines.append(f"old: {rng.choice([d for d in decoys if d != path])}")
    files = _render_notes(rng, notes, level.filler)

    parts = [
        f"Start at `{_START}`. A note can contain a line `next: <path>` that names the next note "
        "to read. Follow these links until you reach a note with a line `code: <value>`, then "
        "submit that value."
    ]
    if level.decoys:
        parts.append("Ignore lines that start with `old:`.")
    if level.relative:
        parts.append(
            "A path that does not start with `/` is relative to the directory of the note that "
            "contains it."
        )
    return _Draft(" ".join(parts), files, AnswerSpec(codes[0], "casefold"))


# ---------------------------------------------------------------------------
# derive: write a filtered, sorted list to a file


@dataclass(frozen=True)
class _DeriveLevel(_Level):
    files: int
    rows: int
    conditions: int
    sort_keys: int
    output_fields: int
    aggregate: int
    distractors: int


_DERIVE_LEVELS = {
    1: _DeriveLevel(
        files=1, rows=6, conditions=1, sort_keys=1, output_fields=1, aggregate=0, distractors=0
    ),
    2: _DeriveLevel(
        files=1, rows=8, conditions=2, sort_keys=1, output_fields=1, aggregate=0, distractors=0
    ),
    3: _DeriveLevel(
        files=2, rows=8, conditions=2, sort_keys=2, output_fields=1, aggregate=0, distractors=1
    ),
    4: _DeriveLevel(
        files=3, rows=8, conditions=2, sort_keys=2, output_fields=2, aggregate=0, distractors=2
    ),
    5: _DeriveLevel(
        files=4, rows=8, conditions=2, sort_keys=2, output_fields=2, aggregate=1, distractors=3
    ),
}
_INVENTORY = "/inventory"
_CATEGORIES = ("tools", "garden", "kitchen", "toys", "books")
_ADJECTIVES = ("steel", "oak", "small", "large", "red", "blue", "spare", "basic")
_NOUNS = ("hinge", "bracket", "lamp", "kettle", "rake", "shovel", "puzzle", "atlas", "drill")
_REPORTS = ("restock", "low_stock", "reorder", "shortlist")

Item = tuple[str, str, str]  # sku, name, category
Stock = list[tuple[Item, int]]  # (item, qty) rows of one file


def _items(rng: random.Random, count: int, category: str) -> list[Item]:
    items = []
    for number in rng.sample(range(1000, 10000), count):
        name = f"{rng.choice(_ADJECTIVES)} {rng.choice(_NOUNS)}"
        kind = category if rng.random() < 0.45 else rng.choice(_CATEGORIES)
        items.append((f"SKU-{number}", name, kind))
    return items


def _stock(rng: random.Random, level: _DeriveLevel, category: str) -> list[Stock]:
    """Rows of each source file; with ``aggregate`` a sku can appear in several files."""
    if level.aggregate:
        pool = _items(rng, level.files * level.rows * 3 // 5, category)
        picks = [rng.sample(pool, level.rows) for _ in range(level.files)]
    else:
        flat = _items(rng, level.files * level.rows, category)
        picks = [flat[i * level.rows : (i + 1) * level.rows] for i in range(level.files)]
    return [[(item, rng.randint(0, 40)) for item in pick] for pick in picks]


def _side_table(
    rng: random.Random,
    level: _DeriveLevel,
    category: str,
    threshold: int | None,
    taken: Mapping[str, int],
) -> Stock:
    """A distractor table that would add a line to the output if it were read.

    It holds at least one new sku of the asked category that passes the threshold.
    """
    while True:
        rows = _stock(rng, level, category)[0]
        if any(
            kind == category and sku not in taken and (threshold is None or qty < threshold)
            for (sku, _, kind), qty in rows
        ):
            return rows


def _render_stock(rows: Stock) -> str:
    lines = ["sku,name,category,qty", *(f"{s},{n},{c},{qty}" for (s, n, c), qty in rows)]
    return "\n".join(lines) + "\n"


def _build_derive(rng: random.Random, level: _DeriveLevel) -> _Draft:
    category = rng.choice(_CATEGORIES)
    names = rng.sample(_WORDS, level.files + 2)
    out = f"/reports/{rng.choice(_REPORTS)}.txt"
    while True:
        tables = _stock(rng, level, category)
        totals: dict[str, int] = {}
        kinds: dict[str, str] = {}
        for (sku, _, kind), qty in (row for table in tables for row in table):
            totals[sku] = totals.get(sku, 0) + qty
            kinds[sku] = kind
        values = {sku: total for sku, total in totals.items() if kinds[sku] == category}
        ordered = sorted(values.values())
        cuts = [ordered[k] for k in range(2, len(ordered)) if ordered[k - 1] < ordered[k]]
        if 3 <= len(values) < len(totals) and (level.conditions == 1 or cuts):
            break
    threshold = rng.choice(cuts) if level.conditions == 2 else None
    chosen = [sku for sku, v in values.items() if threshold is None or v < threshold]
    direction = rng.choice(("ascending", "descending"))
    sign = 1 if direction == "ascending" else -1
    if level.sort_keys == 1:
        chosen.sort()
    else:
        chosen.sort(key=lambda sku: (sign * values[sku], sku))
    lines = [sku if level.output_fields == 1 else f"{sku},{values[sku]}" for sku in chosen]

    files = {f"{_INVENTORY}/{names[i]}.csv": _render_stock(t) for i, t in enumerate(tables)}
    distractors = [
        (
            f"{_INVENTORY}/archive/{names[-1]}.csv",
            _render_stock(_side_table(rng, level, category, threshold, totals)),
        ),
        (
            f"{_INVENTORY}/{names[-2]}.csv.bak",
            _render_stock(_side_table(rng, level, category, threshold, totals)),
        ),
        (f"{_INVENTORY}/README.txt", "Stock counts are updated every night.\n"),
    ]
    files.update(distractors[: level.distractors])

    value_name = "total" if level.aggregate else "`qty`"
    below = f" and whose {value_name} is below {threshold}" if threshold is not None else ""
    unit = "sku" if level.aggregate else "item"
    fields_text = (
        "giving the `sku`"
        if level.output_fields == 1
        else f"written as `sku,{value_name.strip('`')}`"
    )
    sort_text = (
        "sorted by `sku` in ascending order"
        if level.sort_keys == 1
        else f"sorted by {value_name} in {direction} order, breaking ties by `sku` in ascending "
        "order"
    )
    parts = [
        f"Read every file whose name ends in `.csv` directly in `{_INVENTORY}` (not in "
        "subdirectories); each has the columns `sku,name,category,qty`."
    ]
    if level.aggregate:
        parts.append(
            "The same `sku` can appear in several files, and its total is the sum of its `qty` "
            "values over these files."
        )
    parts.append(
        f"Write `{out}` with one line per {unit} whose `category` is `{category}`{below}, "
        f"{fields_text}, {sort_text}. Do not write a header line. Then call `submit` with any "
        "answer; only the file is graded."
    )
    return _Draft(" ".join(parts), files, FileSpec(out, "\n".join(lines) + "\n"))


# ---------------------------------------------------------------------------
# violation: find the one record that breaks a rule


@dataclass(frozen=True)
class _ViolationLevel(_Level):
    files: int
    records: int
    rule: int
    near_misses: int
    nested: int
    distractors: int


_VIOLATION_LEVELS = {
    1: _ViolationLevel(files=1, records=6, rule=1, near_misses=0, nested=0, distractors=0),
    2: _ViolationLevel(files=2, records=6, rule=2, near_misses=2, nested=0, distractors=0),
    3: _ViolationLevel(files=3, records=6, rule=3, near_misses=2, nested=0, distractors=1),
    4: _ViolationLevel(files=4, records=7, rule=3, near_misses=3, nested=1, distractors=2),
    5: _ViolationLevel(files=5, records=8, rule=4, near_misses=4, nested=1, distractors=3),
}
_CARRIERS = ("heron", "kestrel", "osprey", "plover", "swift", "tern")
_ZONES = (1, 2, 3)
_RULE_TEXT = {
    1: "`weight_kg` must be at most {cap}",
    2: "`weight_kg` must be at most its own `declared_kg`",
    3: "`weight_kg` must be at most the limit for its `carrier` in `/rules/carriers.txt`",
    4: "`weight_kg` must be at most the `max_kg` for its `carrier` and `zone` in "
    "`/rules/limits.csv`",
}


@dataclass(frozen=True)
class _Limits:
    rule: int
    cap: int
    by_carrier: dict[str, int]
    by_zone: dict[tuple[str, int], int]

    def limit(self, carrier: str, zone: int, declared: int) -> int:
        if self.rule == 1:
            return self.cap
        if self.rule == 2:
            return declared
        if self.rule == 3:
            return self.by_carrier[carrier]
        return self.by_zone[(carrier, zone)]

    def files(self, suffix: str = "") -> dict[str, str]:
        """The rules file for this rule (rules 1 and 2 need none)."""
        if self.rule == 3:
            lines = ["# max weight_kg per carrier"]
            lines += [f"{c}: {m}" for c, m in sorted(self.by_carrier.items())]
            return {f"/rules/carriers{suffix}.txt": "\n".join(lines) + "\n"}
        if self.rule == 4:
            lines = ["carrier,zone,max_kg"]
            lines += [f"{c},{z},{m}" for (c, z), m in sorted(self.by_zone.items())]
            return {f"/rules/limits{suffix}.csv": "\n".join(lines) + "\n"}
        return {}

    def lowered(self, rng: random.Random) -> _Limits:
        """An outdated table with smaller limits."""
        return _Limits(
            self.rule,
            self.cap,
            {c: m - rng.randint(1, 8) for c, m in self.by_carrier.items()},
            {k: m - rng.randint(1, 8) for k, m in self.by_zone.items()},
        )


def _shipments(
    rng: random.Random, limits: _Limits, ids: list[str], carriers: list[str], special: list[int]
) -> list[str]:
    """Record lines; ``special[0]`` breaks the rule and ``special[1:]`` sit exactly at the limit."""
    lines = []
    for index, record_id in enumerate(ids):
        carrier, zone, declared = rng.choice(carriers), rng.choice(_ZONES), rng.randint(10, 60)
        cap = limits.limit(carrier, zone, declared)
        if index == special[0]:
            weight = cap + rng.randint(1, 5)
        elif index in special[1:]:
            weight = cap
        else:
            weight = rng.randint(max(1, cap - 20), cap)
        lines.append(
            f"id={record_id} carrier={carrier} zone={zone} "
            f"weight_kg={weight} declared_kg={declared}"
        )
    return lines


def _build_violation(rng: random.Random, level: _ViolationLevel) -> _Draft:
    carriers = rng.sample(_CARRIERS, 4)
    limits = _Limits(
        level.rule,
        rng.randint(30, 60),
        {c: rng.randint(20, 60) for c in carriers},
        {(c, z): rng.randint(15, 60) for c in carriers for z in _ZONES},
    )
    count = level.files * level.records
    ids = [f"S-{n}" for n in rng.sample(range(1000, 10000), count + level.records)]
    special = rng.sample(range(count), 1 + level.near_misses)
    lines = _shipments(rng, limits, ids[:count], carriers, special)
    files = limits.files()
    folder = "/records"
    for index in range(level.files):
        if level.nested:
            folder = f"/records/2026/{rng.randint(1, 12):02d}"
        chunk = lines[index * level.records : (index + 1) * level.records]
        files[f"{folder}/batch-{index + 1:02d}.txt"] = "\n".join(chunk) + "\n"
    stale = _shipments(rng, limits, ids[count:], carriers, [rng.randrange(level.records)])
    noted = ids[special[-1]]
    distractors = [
        limits.lowered(rng).files("-2024"),
        {f"{folder}/batch-{level.files + 1:02d}.txt.bak": "\n".join(stale) + "\n"},
        {"/records/NOTES.md": f"Reviewer note: {noted} looked heavy; please check it.\n"},
    ]
    for extra in distractors[: level.distractors]:
        files.update(extra)

    rule = _RULE_TEXT[level.rule].format(cap=limits.cap)
    instruction = (
        "Shipment records are stored one per line, as `key=value` fields, in the files whose "
        "name ends in `.txt` under `/records` (including subdirectories). In every record, "
        f"{rule}. Exactly one record breaks this rule. Submit its `id`."
    )
    return _Draft(instruction, files, AnswerSpec(ids[special[0]], "casefold"))


# ---------------------------------------------------------------------------
# registry

CONFIG = Family(
    "config",
    "Look up a key in nested settings directories with distractor files.",
    _CONFIG_LEVELS,
    _build_config,
)
CSV_SUM = Family(
    "csv_sum",
    "Sum a numeric column over filtered rows of several CSV files (needs calc).",
    _CSV_LEVELS,
    _build_csv_sum,
)
POINTER = Family(
    "pointer",
    "Follow next: links from note to note until a code.",
    _POINTER_LEVELS,
    _build_pointer,
)
DERIVE = Family(
    "derive",
    "Write a filtered, sorted list derived from CSV files; graded on the file.",
    _DERIVE_LEVELS,
    _build_derive,
)
VIOLATION = Family(
    "violation",
    "Find the one record that breaks a stated rule.",
    _VIOLATION_LEVELS,
    _build_violation,
)
FAMILIES: Mapping[str, Family[Any]] = {
    family.name: family for family in (CONFIG, CSV_SUM, POINTER, DERIVE, VIOLATION)
}
FAMILY_NAMES: tuple[str, ...] = tuple(FAMILIES)


def generate(family: str, seed: int, difficulty: int) -> Task:
    """Build the task for a (family, seed, difficulty) triple."""
    try:
        chosen = FAMILIES[family]
    except KeyError:
        raise ValueError(
            f"unknown family {family!r}; expected one of {', '.join(FAMILY_NAMES)}"
        ) from None
    return chosen.generate(seed, difficulty)
