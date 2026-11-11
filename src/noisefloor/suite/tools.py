"""Mock tools for nf-tools episodes.

Every tool is an in-process Python function over the episode's
:class:`~noisefloor.suite.fs.WorkingCopy`. Nothing runs model-generated code:
``calc`` parses its input with :mod:`ast` and evaluates only number literals and
arithmetic operators. A tool never raises. Bad arguments, missing files, and
arithmetic errors come back as observation text that starts with ``error:``.

:class:`ToolEnv` holds the state of one episode. It counts tool calls, records a
trace of (tool, args, observation length), and ends when the agent calls
``submit`` or when an optional call limit is reached. It satisfies the
``Environment`` protocol of :mod:`noisefloor.minimal_agent`.
"""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from noisefloor.suite.fs import FsError, Snapshot, WorkingCopy, normalize_path

ERROR_PREFIX = "error: "
OBSERVATION_LIMIT = 2000  # the minimal-agent default for max_observation_chars
READ_LIMIT = 1800  # characters of a file that read_file shows before truncating
SEARCH_LIMIT = 30  # most matches that search prints
SEARCH_LINE_CHARS = 100  # longest line text that search prints per match
WRITE_LIMIT = 100_000  # largest content that write_file accepts
CALC_MAX_CHARS = 2000
CALC_MAX_DEPTH = 100
CALC_MAX_BITS = 4096  # integer results must fit in this many bits

Number = int | float
EnvEnd = Literal["submitted", "step_limit"]


@dataclass(frozen=True)
class Param:
    """One string argument of a tool. A parameter with a default is optional."""

    name: str
    description: str
    default: str | None = None

    @property
    def required(self) -> bool:
        """True if the caller must supply this argument."""
        return self.default is None


@dataclass(frozen=True)
class ToolSpec:
    """Name, one-line description, and arguments of a tool."""

    name: str
    description: str
    params: tuple[Param, ...]

    @property
    def parameters(self) -> dict[str, Any]:
        """JSON-schema object for the arguments.

        The same shape serves a JSON tool-call scaffold, OpenAI-style function
        calling, and Inspect ``ToolParams`` (properties, required, no extra keys).
        """
        properties: dict[str, Any] = {}
        for param in self.params:
            entry: dict[str, Any] = {"type": "string", "description": param.description}
            if param.default is not None:
                entry["default"] = param.default
            properties[param.name] = entry
        return {
            "type": "object",
            "properties": properties,
            "required": [p.name for p in self.params if p.required],
            "additionalProperties": False,
        }

    def as_function(self) -> dict[str, Any]:
        """The spec as an OpenAI-style ``{"type": "function", ...}`` tool definition."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec(
        "list_dir",
        "List the entries of a directory; subdirectory names end with '/'.",
        (Param("path", "Absolute directory path, for example /data.", default="/"),),
    ),
    ToolSpec(
        "read_file",
        f"Read a text file; files longer than {READ_LIMIT} characters are cut with a note.",
        (Param("path", "Absolute file path, for example /data/notes.txt."),),
    ),
    ToolSpec(
        "write_file",
        "Create or replace a text file; missing parent directories are created.",
        (
            Param("path", "Absolute file path to write."),
            Param("content", "The full new content of the file."),
        ),
    ),
    ToolSpec(
        "search",
        "Find lines that contain a plain substring (case-sensitive) in the files under a "
        "path; prints 'path:line: text'.",
        (
            Param("pattern", "Substring to look for (not a regular expression)."),
            Param("path", "File or directory to search; directories are searched in full.", "/"),
        ),
    ),
    ToolSpec(
        "calc",
        "Evaluate arithmetic on numbers with + - * / // % ** and parentheses.",
        (Param("expression", "For example (12.5 + 7) * 3."),),
    ),
    ToolSpec(
        "submit",
        "Submit the final answer and end the episode.",
        (Param("answer", "The final answer as text."),),
    ),
)
TOOLS_BY_NAME: Mapping[str, ToolSpec] = {spec.name: spec for spec in TOOLS}
TOOL_NAMES: tuple[str, ...] = tuple(spec.name for spec in TOOLS)


class CalcError(ValueError):
    """An expression is not allowed or cannot be evaluated."""


_BINARY: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}


def evaluate_arithmetic(expression: str) -> Number:
    """Evaluate an arithmetic expression without :func:`eval`.

    The expression is parsed with :mod:`ast` and only number literals, unary
    ``+``/``-``, and the binary operators ``+ - * / // % **`` are evaluated, with
    parentheses for grouping. Names, attribute access, calls, strings, and every
    other syntax raise :class:`CalcError`, as do exponents or results that would
    exceed :data:`CALC_MAX_BITS` bits, non-finite results, and division by zero.
    """
    if len(expression) > CALC_MAX_CHARS:
        raise CalcError(f"expression is longer than {CALC_MAX_CHARS} characters")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
        raise CalcError("could not parse the expression as arithmetic") from exc
    try:
        return _evaluate(tree.body, 0)
    except ZeroDivisionError as exc:
        raise CalcError("division by zero") from exc
    except OverflowError as exc:
        raise CalcError("result is too large") from exc


def _evaluate(node: ast.expr, depth: int) -> Number:
    if depth > CALC_MAX_DEPTH:
        raise CalcError("expression is nested too deeply")
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise CalcError("only real number literals are allowed")
        return _checked(value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd | ast.USub):
        operand = _evaluate(node.operand, depth + 1)
        return -operand if isinstance(node.op, ast.USub) else operand
    if isinstance(node, ast.BinOp):
        # A flat chain such as a + b + c nests on the left: walk it in a loop, so that its
        # length costs no depth and no recursion; only real nesting counts toward the limit.
        links: list[ast.BinOp] = []
        while isinstance(node, ast.BinOp):
            links.append(node)
            node = node.left
        value = _evaluate(node, depth + 1)
        for link in reversed(links):
            func = _BINARY.get(type(link.op))
            if func is None:
                raise CalcError(f"operator not allowed: {type(link.op).__name__}")
            right = _evaluate(link.right, depth + 1)
            if isinstance(link.op, ast.Pow):
                _check_power(value, right)
            value = _checked(func(value, right))
        return value
    raise CalcError(f"not allowed in an expression: {type(node).__name__}")


def _check_power(base: Number, exponent: Number) -> None:
    if (
        isinstance(base, int)
        and isinstance(exponent, int)
        and exponent > 0
        and abs(base) > 1
        and abs(base).bit_length() * exponent > CALC_MAX_BITS
    ):
        raise CalcError("exponent is too large")


def _checked(value: object) -> Number:
    if isinstance(value, int):
        if value.bit_length() > CALC_MAX_BITS:
            raise CalcError("result is too large")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CalcError("result is too large")
        return value
    raise CalcError("result is not a real number")


def format_number(value: Number) -> str:
    """Text for a calc result: integers in full, floats as their shortest repr."""
    return str(value) if isinstance(value, int) else repr(value)


@dataclass(frozen=True)
class TraceEntry:
    """One tool call: the tool name, the arguments as given, and the observation size."""

    tool: str
    args: Mapping[str, object]
    observation_length: int
    error: bool


def _error(message: str) -> str:
    return ERROR_PREFIX + message


def _args_record(args: object) -> dict[str, object]:
    if args is None:
        return {}
    if isinstance(args, Mapping):
        return {str(key): value for key, value in args.items()}
    return {"_raw": repr(args)[:200]}


_PATH_WRAPPERS = "`'\""


def _clean_path(value: str) -> str:
    """A path argument without surrounding whitespace, backticks, or quotes."""
    previous = None
    while value != previous:
        previous = value
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in _PATH_WRAPPERS:
            value = value[1:-1]
    return value


def _parse_args(spec: ToolSpec, args: object) -> dict[str, str] | str:
    """Validated string arguments, or an error observation.

    Path arguments lose surrounding whitespace, backticks, and quotes (as copied from
    the instruction), and a file path for ``read_file`` or ``write_file`` must not end
    with ``/``.
    """
    if args is None:
        args = {}
    if not isinstance(args, Mapping):
        return _error(f"arguments for {spec.name} must be an object with named fields")
    known = [param.name for param in spec.params]
    unexpected = sorted(str(key) for key in args if key not in known)
    if unexpected:
        return _error(f"unexpected argument(s) for {spec.name}: {', '.join(unexpected)}")
    parsed: dict[str, str] = {}
    for param in spec.params:
        if param.name not in args:
            if param.default is None:
                return _error(f"missing argument '{param.name}' for {spec.name}")
            parsed[param.name] = param.default
            continue
        value = args[param.name]
        if isinstance(value, bool) or not isinstance(value, str | int | float):
            return _error(f"argument '{param.name}' for {spec.name} must be a string")
        parsed[param.name] = str(value)
    if "path" in parsed:
        parsed["path"] = _clean_path(parsed["path"])
        if spec.name in ("read_file", "write_file") and parsed["path"].endswith("/"):
            return _error(
                f"{parsed['path']} names a directory; give a file path (parent directories "
                "are created automatically)"
            )
    return parsed


class ToolEnv:
    """State of one episode: a working copy of the files, the call count, and the trace."""

    def __init__(
        self, files: Snapshot, *, instruction: str = "", max_calls: int | None = None
    ) -> None:
        """Start an episode on a fresh working copy of ``files``.

        If ``max_calls`` is set, the episode ends (``end_reason == "step_limit"``)
        once that many tool calls have been made without a submit.
        """
        if max_calls is not None and max_calls < 1:
            raise ValueError("max_calls must be positive")
        self._instruction = instruction
        self._files = files.working_copy()
        self._max_calls = max_calls
        self._trace: list[TraceEntry] = []
        self._answer: str | None = None
        self._end: EnvEnd | None = None

    @property
    def instruction(self) -> str:
        """The task text shown to the agent."""
        return self._instruction

    @property
    def files(self) -> WorkingCopy:
        """The episode's working copy of the file system."""
        return self._files

    @property
    def calls(self) -> int:
        """Number of tool calls so far, including invalid ones."""
        return len(self._trace)

    @property
    def trace(self) -> tuple[TraceEntry, ...]:
        """Every tool call so far, in order."""
        return tuple(self._trace)

    @property
    def answer(self) -> str | None:
        """The submitted answer, or None before a submit."""
        return self._answer

    @property
    def done(self) -> bool:
        """True once the episode has ended (submit or call limit)."""
        return self._end is not None

    @property
    def end_reason(self) -> EnvEnd | None:
        """``submitted``, ``step_limit``, or None while the episode is running."""
        return self._end

    def tool_specs(self) -> tuple[ToolSpec, ...]:
        """Tools available in this episode."""
        return TOOLS

    def call(self, name: str, args: object = None) -> str:
        """Run one tool call and return its observation; never raises."""
        observation = self._dispatch(name, args)
        self._trace.append(
            TraceEntry(
                tool=str(name),
                args=_args_record(args),
                observation_length=len(observation),
                error=observation.startswith(ERROR_PREFIX),
            )
        )
        if self._end is None and self._max_calls is not None and self.calls >= self._max_calls:
            self._end = "step_limit"
        return observation

    def _dispatch(self, name: object, args: object) -> str:
        if self._end is not None:
            return _error("the episode has ended; no more tool calls are accepted")
        spec = TOOLS_BY_NAME.get(name) if isinstance(name, str) else None
        if spec is None:
            return _error(f"unknown tool {name!r}; available tools: {', '.join(TOOL_NAMES)}")
        try:
            parsed = _parse_args(spec, args)
            if isinstance(parsed, str):
                return parsed
            return self._run(spec.name, parsed)
        except (FsError, CalcError) as exc:
            return _error(str(exc))
        except Exception as exc:  # a tool never raises, whatever goes wrong inside it
            return _error(f"internal tool error ({type(exc).__name__})")

    def _run(self, name: str, args: dict[str, str]) -> str:
        handlers: dict[str, Callable[[dict[str, str]], str]] = {
            "list_dir": self._list_dir,
            "read_file": lambda a: _read(self._files, a["path"]),
            "write_file": self._write_file,
            "search": lambda a: _search(self._files, a["pattern"], a["path"]),
            "calc": lambda a: format_number(evaluate_arithmetic(a["expression"])),
            "submit": self._submit,
        }
        return handlers[name](args)

    def _list_dir(self, args: dict[str, str]) -> str:
        entries = self._files.list_dir(args["path"])
        return "\n".join(entries) if entries else "(empty directory)"

    def _write_file(self, args: dict[str, str]) -> str:
        content = args["content"]
        if len(content) > WRITE_LIMIT:
            return _error(f"content is longer than {WRITE_LIMIT} characters")
        path = self._files.write(args["path"], content)
        return f"wrote {len(content)} characters to {path}"

    def _submit(self, args: dict[str, str]) -> str:
        self._answer = args["answer"]
        self._end = "submitted"
        return "answer submitted; the episode has ended"


def _read(files: WorkingCopy, path: str) -> str:
    content = files.read(path)
    if not content:
        return "(empty file)"
    if len(content) > READ_LIMIT:
        return (
            content[:READ_LIMIT]
            + f"\n[truncated: showed the first {READ_LIMIT} of {len(content)} characters]"
        )
    return content


def _search(files: WorkingCopy, pattern: str, path: str) -> str:
    if not pattern:
        return _error("pattern must not be empty")
    norm = normalize_path(path)
    if files.is_file(norm):
        targets = [norm]
    elif files.is_dir(norm):
        targets = files.files_under(norm)
    else:
        return _error(f"no such file or directory: {norm}")
    hits = []
    for target in targets:
        for number, line in enumerate(files.read(target).splitlines(), start=1):
            if pattern in line:
                text = line if len(line) <= SEARCH_LINE_CHARS else line[:SEARCH_LINE_CHARS] + "..."
                hits.append(f"{target}:{number}: {text}")
    if not hits:
        return "no matches"
    shown: list[str] = []
    size = 0
    for hit in hits[:SEARCH_LIMIT]:
        if size + len(hit) + 1 > OBSERVATION_LIMIT - 60:
            break
        shown.append(hit)
        size += len(hit) + 1
    if len(shown) < len(hits):
        shown.append(f"[{len(hits) - len(shown)} more matches not shown]")
    return "\n".join(shown)
