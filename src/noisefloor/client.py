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
llama.cpp's ``llama-server``) and ``claude`` (headless Claude Code single calls).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
import time
import tomllib
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
DEFAULT_LEDGER = REPO / "artifacts" / "usage" / "ledger.jsonl"
DEFAULT_CACHE = REPO / "runs" / "cache" / "llm"
GUARD_EXACT = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL")
GUARD_PREFIX = "CLAUDE_CODE_USE_"
PARENT_MARKERS = ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT")
CLAUDE_MODELS = ("haiku", "sonnet", "opus")
KIND_CAPS = {
    "agent_run": "claude_agent_runs_per_day",
    "single_call": "claude_single_calls_per_day",
    "review_run": "claude_review_runs_per_day",
}


class ModelCallRefusedError(RuntimeError):
    """A call was refused before reaching any model (guard, cap, or kill switch)."""


class BackendError(RuntimeError):
    """The backend failed (connection, HTTP, or malformed response)."""


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
        ]
    for path in files:
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and "apiKeyHelper" in data:
            found.append(f"apiKeyHelper in {path}")
    return found


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
            cap_name = KIND_CAPS.get(kind)
            if cap_name is None:
                raise ModelCallRefusedError(f"unknown Claude call kind {kind!r}")
            if self.ledger.count_today(kind) >= int(self.caps[cap_name]):
                raise ModelCallRefusedError(f"daily cap {cap_name} = {self.caps[cap_name]} reached")

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
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        start = time.monotonic()
        try:
            with opener.open(req, timeout=timeout) as resp:
                data = json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            raise BackendError(f"local backend failed: {exc}") from exc
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
        )

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
        system = "\n".join(str(m["content"]) for m in request.messages if m["role"] == "system")
        prompt = "\n\n".join(str(m["content"]) for m in request.messages if m["role"] != "system")
        env = {k: v for k, v in os.environ.items() if k not in PARENT_MARKERS}
        cmd = self.claude_command(request.model, system or "Answer briefly.")
        start = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="nf-claude-") as work:
            try:
                proc = subprocess.run(
                    cmd,
                    input=prompt,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=work,
                    env=env,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise BackendError(f"claude failed to run: {exc}") from exc
        return parse_stream_json(
            proc.stdout.splitlines(), time.monotonic() - start, proc.returncode, proc.stderr
        )


def parse_stream_json(
    lines: Sequence[str], wall_s: float, returncode: int = 0, stderr: str = ""
) -> ChatResponse:
    """Parse headless Claude Code stream-json output into a response."""
    text_parts: list[str] = []
    result: dict[str, Any] | None = None
    for line in lines:
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("type") == "assistant":
            for block in event.get("message", {}).get("content", []):
                if block.get("type") == "text":
                    text_parts.append(str(block.get("text", "")))
        elif event.get("type") == "result":
            result = event
    if result is None:
        raise BackendError(f"claude returned no result (exit {returncode}): {stderr[-300:]}")
    if result.get("is_error"):
        message = str(result.get("result", ""))
        raise BackendError(f"claude reported an error: {message[:300]}")
    usage = result.get("usage", {})
    return ChatResponse(
        text=str(result.get("result") or "".join(text_parts)),
        input_tokens=int(usage.get("input_tokens", 0)),
        output_tokens=int(usage.get("output_tokens", 0)),
        wall_s=wall_s,
        finish_reason=str(result.get("subtype", "")),
        plan_equivalent_usd=float(result.get("total_cost_usd", 0.0)),
    )
