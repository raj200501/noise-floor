"""The one client wrapper for every model call (docs/BUILD_BRIEF.md section 8).

Every call goes through :class:`Client`, which:

* refuses all calls when ``NOISEFLOOR_NO_MODELS=1`` (the offline demo and ship-check);
* refuses Claude calls while the billing guard is tripped (``ANTHROPIC_API_KEY``,
  ``ANTHROPIC_AUTH_TOKEN``, ``ANTHROPIC_BASE_URL``, any ``CLAUDE_CODE_USE_*``, or an
  ``apiKeyHelper`` setting), and never uses ``--bare``;
* enforces the daily caps in budget.toml times the cap share;
* caches responses by a content hash of the full request, so a banked call is
  never re-run;
* appends one line per call (tokens, wall time, cache hit) to the usage ledger.
  ``total_cost_usd`` from Claude stream-json is logged as ``plan_equivalent_usd``
  (an API-price estimate), never as cash; ``cash_usd`` is always 0.

Backends: ``openai-local`` (an OpenAI-compatible server on loopback, such as
llama.cpp's ``llama-server``) and ``claude`` (headless Claude Code: single calls through
:meth:`Client.chat`, subject runs through :meth:`Client.agent_run`).

Every Claude child gets a clean environment (:func:`claude_child_env`: parent-session
markers and other services' credentials removed, CLAUDE.md files and auto memory off)
and is read as it streams (:class:`ClaudeStream`). It is stopped at once if it reports a
login other than the plan's (an API key source), a refused plan limit, or paid extra
usage (:class:`UsageLimitError`: stop launching Claude runs), or if it passes its turn
cap or wall-clock limit.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import re
import signal
import subprocess
import tempfile
import threading
import time
import tomllib
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

REPO = Path(__file__).resolve().parents[2]
DEFAULT_LEDGER = REPO / "artifacts" / "usage" / "ledger.jsonl"
DEFAULT_CACHE = REPO / "runs" / "cache" / "llm"
GUARD_EXACT = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL")
GUARD_PREFIX = "CLAUDE_CODE_USE_"
PARENT_MARKERS = ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT")
MANAGED_SETTINGS = (
    Path("/etc/claude-code/managed-settings.json"),
    Path("/Library/Application Support/ClaudeCode/managed-settings.json"),
)
# Credentials for other services never reach a Claude child (section 8 sandboxing).
CHILD_SECRET_NAMES = frozenset(
    {
        "GH_TOKEN",
        "GITHUB_TOKEN",
        "GITHUB_MODELS_TOKEN",
        "GEMINI_API_KEY",
        "OPENROUTER_API_KEY",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "SSH_AUTH_SOCK",
    }
)
CHILD_SECRET_PREFIXES = ("AWS_", "AZURE_", "GOOGLE_", "GCLOUD_", "OPENAI_")
CHILD_SWITCHES = {
    "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",  # Raj's CLAUDE.md files stay out of the child
    "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",  # and so does his auto memory
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": "/dev/null",
}
LIMIT_TEXT = re.compile(r"usage limit|limit reached|extra usage|credit balance", re.IGNORECASE)
# llama-server's error when a request does not fit the context ("exceed_context_size_error").
_CONTEXT_TEXT = re.compile(r"exceed_context_size|exceeds the available context", re.IGNORECASE)
CLAUDE_MODELS = ("haiku", "sonnet", "opus")
UNCAPPED_KINDS = frozenset({"agent_step"})  # steps of a run already counted as an agent_run
KIND_CAPS = {
    "agent_run": "claude_agent_runs_per_day",
    "single_call": "claude_single_calls_per_day",
    "review_run": "claude_review_runs_per_day",
}


class ModelCallRefusedError(RuntimeError):
    """A call was refused before reaching any model (guard, cap, or kill switch)."""


class UsageLimitError(ModelCallRefusedError):
    """The plan refused a call or reported paid extra usage: stop launching Claude runs.

    ``resets_at`` is the plan's reset time (Unix seconds) when the child reported one.
    """

    def __init__(self, message: str, resets_at: int | None = None) -> None:
        """Keep the message and the reset time."""
        super().__init__(message)
        self.resets_at = resets_at


class BackendError(RuntimeError):
    """The backend failed (connection, HTTP, or malformed response)."""


class BackendTimeoutError(BackendError):
    """A model call ran out of time (the episode's wall clock, not the infrastructure)."""


class ContextLimitError(BackendError):
    """The conversation no longer fits the model's context window (caused by the episode)."""


@dataclass(frozen=True)
class ChatRequest:
    """A chat completion request. Everything here is part of the cache key.

    ``extra`` holds other OpenAI request fields (such as ``tool_choice``) as
    ``(name, JSON-encoded value)`` pairs, forwarded verbatim to the backend.
    """

    backend: str
    model: str
    messages: tuple[Mapping[str, Any], ...]
    temperature: float = 0.0
    max_tokens: int = 512
    seed: int | None = None
    stop: tuple[str, ...] = ()
    tools: tuple[Mapping[str, Any], ...] = ()
    extra: tuple[tuple[str, str], ...] = ()
    salt: str = ""

    def key(self) -> str:
        """SHA-256 of the canonical JSON of the request."""
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), default=list)
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class ChatResponse:
    """A model response with usage accounting."""

    text: str
    tool_calls: tuple[Mapping[str, Any], ...] = ()
    input_tokens: int = 0
    output_tokens: int = 0
    wall_s: float = 0.0
    finish_reason: str = ""
    cached: bool = False
    plan_equivalent_usd: float = 0.0
    model_id: str = ""


def billing_guard_findings(
    env: Mapping[str, str] | None = None, settings_files: Iterable[Path] | None = None
) -> list[str]:
    """Names of variables or settings that trip the section 8 billing guard."""
    env = os.environ if env is None else env
    found = [n for n in GUARD_EXACT if n in env]
    found += sorted(n for n in env if n.startswith(GUARD_PREFIX))
    files = settings_files
    if files is None:
        files = [
            Path.home() / ".claude" / "settings.json",
            Path.home() / ".claude" / "settings.local.json",
            REPO / ".claude" / "settings.json",
            REPO / ".claude" / "settings.local.json",
            *MANAGED_SETTINGS,
        ]
    for path in files:
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and "apiKeyHelper" in data:
            found.append(f"apiKeyHelper in {path}")
    return found


def claude_child_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment for a Claude child.

    Removes the parent-session markers and other services' credentials, keeps the
    plan login (``HOME``, ``CLAUDE_CONFIG_DIR``, ``CLAUDE_CODE_OAUTH_TOKEN``) and any
    proxy settings, and turns off CLAUDE.md loading and auto memory so Raj's personal
    instructions do not reach subject runs. Billing-guard variables are not removed:
    while any is set no Claude child is launched at all.
    """
    src = os.environ if base is None else base
    env: dict[str, str] = {}
    for name, value in src.items():
        if name in PARENT_MARKERS or name in CHILD_SECRET_NAMES:
            continue
        if name.startswith(CHILD_SECRET_PREFIXES):
            continue
        keep = name == "CLAUDE_CODE_OAUTH_TOKEN" or name.startswith(GUARD_PREFIX)
        if name.startswith("CLAUDE_CODE_") and not keep:
            continue
        secretish = any(s in name for s in ("TOKEN", "SECRET", "PASSWORD", "API_KEY"))
        if secretish and not name.startswith(("ANTHROPIC_", "CLAUDE_")):
            continue
        env[name] = value
    env.update(CHILD_SWITCHES)
    return env


def effective_caps(budget_path: Path, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Caps from budget.toml with every ``claude_*`` cap times the share, rounded down."""
    env = os.environ if env is None else env
    with budget_path.open("rb") as fh:
        caps = tomllib.load(fh)
    share = float(caps.get("cap_share", 1.0))
    if env.get("CAP_SHARE"):
        share = min(share, float(env["CAP_SHARE"]))
    share = max(0.0, min(1.0, share))
    return {k: math.floor(v * share) if k.startswith("claude_") else v for k, v in caps.items()}


@dataclass
class Ledger:
    """Append-only JSON Lines usage ledger."""

    path: Path

    def append(self, entry: Mapping[str, Any]) -> None:
        """Append one entry (adds ``ts`` and ``cash_usd = 0``); one write, so lines stay whole."""
        row = {"ts": datetime.now(UTC).isoformat(timespec="seconds"), "cash_usd": 0, **entry}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(row, sort_keys=True) + "\n"
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, line.encode())
        finally:
            os.close(fd)

    def entries(self) -> list[dict[str, Any]]:
        """All entries (empty if the ledger does not exist)."""
        if not self.path.exists():
            return []
        return [json.loads(ln) for ln in self.path.read_text().splitlines() if ln.strip()]

    def count_today(self, kind: str, day: str | None = None) -> int:
        """Entries of ``kind`` dated today (UTC), excluding cache hits."""
        day = day or datetime.now(UTC).date().isoformat()
        return sum(
            1
            for e in self.entries()
            if e.get("kind") == kind
            and str(e.get("ts", "")).startswith(day)
            and not e.get("cached")
        )


@dataclass
class Client:
    """The single entry point for model calls."""

    ledger: Ledger = field(default_factory=lambda: Ledger(DEFAULT_LEDGER))
    cache_dir: Path = DEFAULT_CACHE
    caps: Mapping[str, Any] = field(default_factory=lambda: effective_caps(REPO / "budget.toml"))
    label: str = ""
    base_urls: Mapping[str, str] = field(default_factory=dict)
    claude_bin: str = "claude"

    def _check_allowed(self, request: ChatRequest, kind: str) -> None:
        if os.environ.get("NOISEFLOOR_NO_MODELS") == "1":
            raise ModelCallRefusedError("model calls are disabled (NOISEFLOOR_NO_MODELS=1)")
        if request.backend == "claude":
            guard = billing_guard_findings()
            if guard:
                raise ModelCallRefusedError(
                    "billing guard tripped by " + ", ".join(guard) + "; add to Needs Raj"
                )
            if kind in UNCAPPED_KINDS:
                return
            cap_name = KIND_CAPS.get(kind)
            if cap_name is None:
                raise ModelCallRefusedError(f"unknown Claude call kind {kind!r}")
            if self.ledger.count_today(kind) >= int(self.caps[cap_name]):
                raise ModelCallRefusedError(f"daily cap {cap_name} = {self.caps[cap_name]} reached")

    def start_agent_run(self, model: str) -> None:
        """Check the guard and the daily run cap, then log one Claude ``agent_run``."""
        probe = ChatRequest(backend="claude", model=model, messages=())
        self._check_allowed(probe, "agent_run")
        self.ledger.append(
            {
                "kind": "agent_run",
                "backend": "claude",
                "model": model,
                "label": self.label,
                "cached": False,
            }
        )

    def log_usage(
        self,
        model: str,
        *,
        input_tokens: int,
        output_tokens: int,
        wall_s: float,
        plan_equivalent_usd: float,
        turns: int,
    ) -> None:
        """Log a Claude Code run's usage (tokens, turns, API-price estimate; never cash)."""
        self.ledger.append(
            {
                "kind": "agent_usage",
                "backend": "claude",
                "model": model,
                "label": self.label,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "wall_s": round(wall_s, 3),
                "plan_equivalent_usd": plan_equivalent_usd,
                "turns": turns,
                "cached": False,
            }
        )

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    def chat(
        self, request: ChatRequest, *, kind: str = "local", timeout: float = 600.0
    ) -> ChatResponse:
        """Run (or replay from cache) one chat request."""
        self._check_allowed(request, kind)
        key = request.key()
        path = self._cache_path(key)
        if path.exists():
            data = json.loads(path.read_text())
            response = ChatResponse(
                **{**data, "tool_calls": tuple(data["tool_calls"]), "cached": True}
            )
        else:
            if request.backend == "openai-local":
                response = self._openai_local(request, timeout)
            elif request.backend == "claude":
                response = self._claude_single(request, timeout)
            else:
                raise ValueError(f"unknown backend {request.backend!r}")
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(asdict(response), sort_keys=True))
            tmp.replace(path)
        self.ledger.append(
            {
                "kind": kind,
                "backend": request.backend,
                "model": request.model,
                "label": self.label,
                "key": key[:16],
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "wall_s": round(response.wall_s, 3),
                "cached": response.cached,
                "plan_equivalent_usd": response.plan_equivalent_usd,
            }
        )
        return response

    def _openai_local(self, request: ChatRequest, timeout: float) -> ChatResponse:
        base = self.base_urls.get(request.model) or self.base_urls.get("*")
        if not base:
            raise BackendError(f"no base URL configured for {request.model!r}")
        if not base.startswith(("http://127.0.0.1", "http://localhost")):
            raise ModelCallRefusedError("openai-local backend only talks to loopback servers")
        body: dict[str, Any] = {
            "model": request.model,
            "messages": list(request.messages),
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.seed is not None:
            body["seed"] = request.seed
        if request.stop:
            body["stop"] = list(request.stop)
        if request.tools:
            body["tools"] = list(request.tools)
        for name, value in request.extra:
            body[name] = json.loads(value)
        req = urllib.request.Request(
            f"{base.rstrip('/')}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        start = time.monotonic()
        data = _post_json(req, timeout)
        wall = time.monotonic() - start
        try:
            choice = data["choices"][0]
            message = choice["message"]
            usage = data.get("usage", {})
        except (KeyError, IndexError, TypeError) as exc:
            raise BackendError(f"malformed response: {str(data)[:200]}") from exc
        return ChatResponse(
            text=message.get("content") or "",
            tool_calls=tuple(message.get("tool_calls") or ()),
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            wall_s=wall,
            finish_reason=str(choice.get("finish_reason") or ""),
            model_id=str(data.get("model") or ""),
        )

    def agent_run(
        self,
        model: str,
        prompt: str,
        *,
        cwd: Path,
        tools: Sequence[str],
        max_turns: int,
        timeout: float,
    ) -> ClaudeRun:
        """One Claude Code subject run in ``cwd`` (counted as one ``agent_run``).

        Checks the guard and the daily run cap first, runs the child under the turn cap
        and wall-clock limit, and logs its usage. Raises :class:`ModelCallRefusedError`
        (or :class:`UsageLimitError`) when the run must not count: the child was not on
        the plan login, or the plan refused it or reported paid extra usage.
        """
        self.start_agent_run(model)
        cmd = self.claude_agent_command(model, tools, max_turns)
        run = run_claude_stream(cmd, prompt, cwd=cwd, timeout=timeout, max_turns=max_turns)
        input_tokens, output_tokens = run.stream.usage()
        self.log_usage(
            model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            wall_s=run.wall_s,
            plan_equivalent_usd=run.stream.plan_equivalent_usd(),
            turns=len(run.stream.turn_ids),
        )
        raise_for_stop(run.ended, run.stream)
        return run

    def claude_agent_command(self, model: str, tools: Sequence[str], max_turns: int) -> list[str]:
        """The exact headless command for a Claude Code subject run (recorded in provenance).

        Only the listed built-in tools exist (``--tools``) and only they are allowed
        without a prompt (``--allowedTools`` with ``dontAsk``); ``--restricted`` drops
        code-running and web tools, confines file tools to the working directory, and
        ignores settings files; ``--strict-mcp-config`` loads no MCP servers.
        """
        if model not in CLAUDE_MODELS:
            raise ValueError(f"unknown Claude model {model!r}")
        if max_turns < 1:
            raise ValueError("max_turns must be positive")
        listed = ",".join(tools)
        return [
            self.claude_bin,
            "-p",
            "--model",
            model,
            "--output-format",
            "stream-json",
            "--verbose",
            "--restricted",
            "--strict-mcp-config",
            "--tools",
            listed,
            "--allowedTools",
            listed,
            "--permission-mode",
            "dontAsk",
            "--no-session-persistence",
            "--max-turns",
            str(max_turns),
        ]

    def claude_command(self, model: str, system_prompt: str) -> list[str]:
        """The exact headless command for a single Claude call (recorded in provenance)."""
        if model not in CLAUDE_MODELS:
            raise ValueError(f"unknown Claude model {model!r}")
        return [
            self.claude_bin,
            "-p",
            "--model",
            model,
            "--output-format",
            "stream-json",
            "--verbose",
            "--restricted",
            "--strict-mcp-config",
            "--tools",
            "",
            "--system-prompt",
            system_prompt,
            "--no-session-persistence",
        ]

    def _claude_single(self, request: ChatRequest, timeout: float) -> ChatResponse:
        system, prompt = claude_prompt(request.messages)
        cmd = self.claude_command(request.model, system or "Answer briefly.")
        with tempfile.TemporaryDirectory(prefix="nf-claude-") as work:
            run = run_claude_stream(cmd, prompt, cwd=Path(work), timeout=timeout)
        raise_for_stop(run.ended, run.stream)
        if run.ended == "wall_clock":
            raise BackendTimeoutError(f"claude did not finish within {timeout:.0f} s")
        return run.stream.response(run.wall_s, run.returncode or 0, run.stderr)


def _post_json(req: urllib.request.Request, timeout: float) -> Any:
    """POST to a loopback server (no proxy) and decode the JSON reply.

    Raises :class:`ContextLimitError` for llama-server's context-size error,
    :class:`BackendTimeoutError` when the call runs out of time, and
    :class:`BackendError` for any other failure.
    """
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = _error_text(exc)
        if exc.code == 400 and _CONTEXT_TEXT.search(detail):
            raise ContextLimitError(f"context limit: {detail[:200]}") from exc
        raise BackendError(f"local backend failed: {exc} {detail[:200]}".strip()) from exc
    except TimeoutError as exc:
        raise BackendTimeoutError(f"local backend timed out after {timeout:.0f} s") from exc
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            raise BackendTimeoutError(f"local backend timed out after {timeout:.0f} s") from exc
        raise BackendError(f"local backend failed: {exc}") from exc
    except (OSError, ValueError) as exc:
        raise BackendError(f"local backend failed: {exc}") from exc


def _error_text(exc: urllib.error.HTTPError) -> str:
    """The error body's message (or raw text) of an HTTP error, for classification."""
    try:
        raw = exc.read().decode("utf-8", errors="replace")
    except (OSError, ValueError):
        return ""
    try:
        error = json.loads(raw).get("error", {})
        return f"{error.get('type', '')}: {error.get('message', '')}"
    except (ValueError, AttributeError):
        return raw[:300]


def claude_prompt(messages: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    """The system text and the one prompt for a headless call.

    A single user message is sent as it is. A longer conversation (minimal-agent steps)
    is sent as one prompt with role-tagged turns, oldest first, because the CLI cannot
    replay earlier assistant turns.
    """
    system = "\n".join(str(m["content"]) for m in messages if m["role"] == "system")
    turns = [m for m in messages if m["role"] != "system"]
    if len(turns) == 1 and turns[0]["role"] == "user":
        return system, str(turns[0]["content"])
    parts = ["The conversation so far, oldest first. Write only the assistant's next reply."]
    parts += [f"<{m['role']}>\n{m['content']}\n</{m['role']}>" for m in turns]
    return system, "\n\n".join(parts)


def _int(value: object) -> int:
    """An integer from a usage field (0 when absent or not a number)."""
    if isinstance(value, int | float):
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    return 0


@dataclass
class ClaudeStream:
    """Incremental reader of headless Claude Code stream-json output.

    :meth:`feed` returns a stop reason when the child must be stopped at once:
    ``api_key_source`` (the child authenticated with an API key, not the plan login),
    ``usage_limit`` (the plan refused requests), ``overage`` (paid extra usage in use),
    or ``turn_cap`` (more distinct assistant turns than ``max_turns``).
    """

    max_turns: int | None = None
    init: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] | None = None
    rate_limit: dict[str, Any] = field(default_factory=dict)
    text_parts: list[str] = field(default_factory=list)
    turn_ids: list[str] = field(default_factory=list)
    tool_uses: int = 0
    turn_usage: dict[str, dict[str, Any]] = field(default_factory=dict)

    def feed(self, line: str) -> str:
        """Read one output line; returns a stop reason, or "" to keep going."""
        try:
            event = json.loads(line)
        except ValueError:
            return ""
        if not isinstance(event, dict):
            return ""
        kind = event.get("type")
        if kind == "system" and event.get("subtype") == "init":
            self.init = event
            return "" if event.get("apiKeySource") in (None, "none") else "api_key_source"
        if kind == "rate_limit_event":
            return self._rate_limit(event.get("rate_limit_info"))
        if kind == "assistant":
            return self._assistant(event.get("message"))
        if kind == "result":
            self.result = event
        return ""

    def _rate_limit(self, info: object) -> str:
        self.rate_limit = dict(info) if isinstance(info, dict) else {}
        if self.rate_limit.get("isUsingOverage") or self.rate_limit.get("overageInUse"):
            return "overage"
        return "usage_limit" if self.rate_limit.get("status") == "rejected" else ""

    def _assistant(self, message: object) -> str:
        if not isinstance(message, dict):
            return ""
        turn = str(message.get("id") or f"turn-{len(self.turn_ids)}")
        if turn not in self.turn_ids:
            self.turn_ids.append(turn)
        if isinstance(message.get("usage"), dict):
            self.turn_usage[turn] = message["usage"]
        for block in message.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "text":
                self.text_parts.append(str(block.get("text", "")))
            elif isinstance(block, dict) and block.get("type") == "tool_use":
                self.tool_uses += 1
        if self.max_turns is not None and len(self.turn_ids) > self.max_turns:
            return "turn_cap"
        return ""

    def usage(self) -> tuple[int, int]:
        """Input tokens (including cache reads and writes) and output tokens."""
        totals = (self.result or {}).get("usage")
        rows = [totals] if isinstance(totals, dict) else list(self.turn_usage.values())
        names = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
        inputs = sum(_int(row.get(name)) for row in rows for name in names)
        return inputs, sum(_int(row.get("output_tokens")) for row in rows)

    def plan_equivalent_usd(self) -> float:
        """The child's ``total_cost_usd`` (an API-price estimate, never cash)."""
        try:
            return float((self.result or {}).get("total_cost_usd") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def limit_message(self) -> str:
        """The error text if the run ended on a plan limit, else ""."""
        result = self.result or {}
        text = str(result.get("result") or "".join(self.text_parts))
        return text if result.get("is_error") and LIMIT_TEXT.search(text) else ""

    def response(self, wall_s: float, returncode: int = 0, stderr: str = "") -> ChatResponse:
        """The single-call response; raises if the call did not succeed."""
        if self.result is None:
            raise BackendError(f"claude returned no result (exit {returncode}): {stderr[-300:]}")
        limit = self.limit_message()
        if limit:
            raise UsageLimitError(f"plan usage limit: {limit[:200]}", _reset_epoch(limit))
        text = str(self.result.get("result") or "".join(self.text_parts))
        if self.result.get("is_error"):
            raise BackendError(f"claude reported an error: {text[:300]}")
        input_tokens, output_tokens = self.usage()
        return ChatResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            wall_s=wall_s,
            finish_reason=str(self.result.get("subtype", "")),
            plan_equivalent_usd=self.plan_equivalent_usd(),
            model_id=str(self.init.get("model") or ""),
        )


def _reset_epoch(text: str) -> int | None:
    match = re.search(r"\|(\d{9,11})\b", text)
    return int(match.group(1)) if match else None


def raise_for_stop(reason: str, stream: ClaudeStream) -> None:
    """Raise the refusal that a stop reason (or a plan-limit result) means; else return."""
    if reason == "api_key_source":
        source = stream.init.get("apiKeySource")
        raise ModelCallRefusedError(
            f"billing guard: the child used API key source {source!r}, not the plan login; "
            "add to Needs Raj"
        )
    if reason in ("usage_limit", "overage"):
        what = "paid extra usage in use" if reason == "overage" else "plan usage limit reached"
        resets = stream.rate_limit.get("resetsAt")
        raise UsageLimitError(f"{what}; stop launching Claude runs", _int(resets) or None)
    limit = stream.limit_message()
    if limit:
        raise UsageLimitError(f"plan usage limit: {limit[:200]}", _reset_epoch(limit))


@dataclass
class ClaudeRun:
    """A Claude child that finished or was stopped.

    ``ended`` is ``exit`` when the child ended by itself, else the stop reason
    (``wall_clock`` or one of :class:`ClaudeStream`'s).
    """

    stream: ClaudeStream
    ended: str
    returncode: int | None
    stderr: str
    wall_s: float
    command: list[str]


def _write_stdin(pipe: IO[str], text: str) -> None:
    try:
        pipe.write(text)
        pipe.close()
    except (BrokenPipeError, OSError, ValueError):
        pass


def _pump(pipe: IO[str], lines: queue.Queue[str | None]) -> None:
    try:
        for line in pipe:
            lines.put(line)
    except (OSError, ValueError):
        pass
    lines.put(None)


def _stop(proc: subprocess.Popen[str]) -> None:
    """Stop the child and anything it started (it leads its own process group)."""
    for sig, wait in ((signal.SIGTERM, 10.0), (signal.SIGKILL, 10.0)):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            return
        try:
            proc.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue


def run_claude_stream(
    cmd: Sequence[str],
    prompt: str,
    *,
    cwd: Path,
    timeout: float,
    max_turns: int | None = None,
    env: Mapping[str, str] | None = None,
) -> ClaudeRun:
    """Run a Claude child in ``cwd`` with ``prompt`` on stdin, reading its stream as it goes.

    The child is stopped as soon as :class:`ClaudeStream` reports a stop reason or the
    wall-clock limit passes. Raises :class:`BackendError` if it cannot be started.
    """
    stream = ClaudeStream(max_turns=max_turns)
    start = time.monotonic()
    try:
        proc = subprocess.Popen(
            list(cmd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
            env=dict(claude_child_env() if env is None else env),
            start_new_session=True,
        )
    except OSError as exc:
        raise BackendError(f"claude failed to run: {exc}") from exc
    if proc.stdin is None or proc.stdout is None or proc.stderr is None:  # pragma: no cover
        raise BackendError("claude pipes are missing")
    lines: queue.Queue[str | None] = queue.Queue()
    errors: list[str] = []
    stderr_pipe = proc.stderr
    pumps = [
        threading.Thread(target=_pump, args=(proc.stdout, lines), daemon=True),
        threading.Thread(target=lambda: errors.append(stderr_pipe.read()), daemon=True),
    ]
    threading.Thread(target=_write_stdin, args=(proc.stdin, prompt), daemon=True).start()
    for pump in pumps:
        pump.start()
    ended = "exit"
    deadline = start + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            ended = "wall_clock"
            break
        try:
            line = lines.get(timeout=min(remaining, 0.5))
        except queue.Empty:
            continue
        if line is None:
            break
        reason = stream.feed(line)
        if reason:
            ended = reason
            break
    if ended != "exit":
        _stop(proc)
    try:
        returncode: int | None = proc.wait(timeout=30)
    except subprocess.TimeoutExpired:  # pragma: no cover - the child ignores its closed pipes
        _stop(proc)
        returncode = proc.poll()
    for pump in pumps:
        pump.join(timeout=5)
    return ClaudeRun(
        stream=stream,
        ended=ended,
        returncode=returncode,
        stderr="".join(errors)[-2000:],
        wall_s=time.monotonic() - start,
        command=list(cmd),
    )


def parse_stream_json(
    lines: Sequence[str], wall_s: float, returncode: int = 0, stderr: str = ""
) -> ChatResponse:
    """Parse complete headless Claude Code stream-json output into a single-call response."""
    stream = ClaudeStream()
    for line in lines:
        reason = stream.feed(line)
        if reason:
            raise_for_stop(reason, stream)
    return stream.response(wall_s, returncode, stderr)
