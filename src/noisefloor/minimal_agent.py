"""minimal-agent: this project's scaffold for tool-use episodes.

One model call per step. The model answers with one JSON object
``{"tool": <name>, "args": {...}}``; the scaffold runs the tool and sends back the
observation. The episode ends when the environment reports it is done (the
``submit`` tool), when the step limit or the wall-clock limit is reached
(status ``timeout``), or when the backend fails (status ``infra_error``).

The action parser accepts JSON inside markdown code fences and surrounding prose:
scoring a correct but fenced action as a failure would measure formatting, not
capability (the Fencecheck lesson). The scaffold is held constant across models.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from noisefloor.client import BackendError, ChatRequest, Client

FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.S)


class ToolSpecLike(Protocol):
    """What the scaffold needs to know about a tool."""

    @property
    def name(self) -> str:
        """Tool name."""

    @property
    def description(self) -> str:
        """One-line description."""

    @property
    def parameters(self) -> Mapping[str, Any]:
        """JSON-schema style parameter spec."""


class Environment(Protocol):
    """A tool-use environment for one episode."""

    @property
    def instruction(self) -> str:
        """The task text shown to the agent."""

    @property
    def done(self) -> bool:
        """True once the agent has submitted."""

    def tool_specs(self) -> Sequence[ToolSpecLike]:
        """Tools available in this episode."""

    def call(self, name: str, args: Mapping[str, Any]) -> str:
        """Run a tool and return the observation text (never raises)."""


@dataclass
class AgentConfig:
    """Scaffold settings that are part of an experimental cell."""

    backend: str
    model: str
    temperature: float = 0.0
    max_steps: int = 15
    max_tokens: int = 256
    wall_limit_s: float = 300.0
    seed: int = 0
    max_observation_chars: int = 2000


@dataclass
class Episode:
    """What happened in one episode."""

    status: str
    steps: int
    parse_errors: int
    input_tokens: int
    output_tokens: int
    wall_s: float
    model_wall_s: float
    trace: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""


def system_prompt(specs: Sequence[ToolSpecLike]) -> str:
    """The minimal-agent system prompt for these tools."""
    lines = [
        "You are an agent that completes tasks by calling tools.",
        "Each turn, reply with exactly one JSON object and nothing else:",
        '{"tool": "<tool name>", "args": {<arguments>}}',
        "You will then see the tool's output. When you know the answer, call the submit tool.",
        "Tools:",
    ]
    for spec in specs:
        params = json.dumps(dict(spec.parameters.get("properties", {})), sort_keys=True)
        lines.append(f"- {spec.name}: {spec.description} Arguments: {params}")
    return "\n".join(lines)


def parse_action(text: str) -> tuple[str, dict[str, Any]] | None:
    """Extract ``(tool, args)`` from a reply, or None if no valid action is present."""
    candidates = [m.group(1) for m in FENCE.finditer(text)] + [text]
    for chunk in candidates:
        for obj in _json_objects(chunk):
            tool = obj.get("tool") or obj.get("name")
            args = obj.get("args", obj.get("arguments", {}))
            if isinstance(tool, str) and isinstance(args, dict):
                return tool, args
    return None


def _json_objects(text: str) -> list[dict[str, Any]]:
    """Top-level JSON objects found anywhere in ``text``, in order."""
    found = []
    decoder = json.JSONDecoder()
    idx = text.find("{")
    while idx != -1:
        try:
            obj, end = decoder.raw_decode(text, idx)
        except ValueError:
            idx = text.find("{", idx + 1)
            continue
        if isinstance(obj, dict):
            found.append(obj)
        idx = text.find("{", end)
    return found


def run_episode(env: Environment, client: Client, config: AgentConfig) -> Episode:
    """Run one episode with minimal-agent."""
    specs = env.tool_specs()
    names = {s.name for s in specs}
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt(specs)},
        {"role": "user", "content": env.instruction},
    ]
    start = time.monotonic()
    episode = Episode("timeout", 0, 0, 0, 0, 0.0, 0.0)
    while episode.steps < config.max_steps:
        if time.monotonic() - start > config.wall_limit_s:
            break
        request = ChatRequest(
            backend=config.backend,
            model=config.model,
            messages=tuple(messages),
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            seed=config.seed + episode.steps,
        )
        try:
            remaining = max(1.0, config.wall_limit_s - (time.monotonic() - start))
            reply = client.chat(request, timeout=remaining)
        except BackendError as exc:
            episode.status, episode.error = "infra_error", str(exc)[:300]
            break
        episode.steps += 1
        episode.input_tokens += reply.input_tokens
        episode.output_tokens += reply.output_tokens
        episode.model_wall_s += reply.wall_s
        messages.append({"role": "assistant", "content": reply.text})
        action = parse_action(reply.text)
        if action is None:
            episode.parse_errors += 1
            observation = 'Error: reply with one JSON object like {"tool": ..., "args": {...}}.'
            episode.trace.append({"tool": None, "obs_chars": len(observation)})
        elif action[0] not in names:
            observation = f"Error: unknown tool {action[0]!r}. Tools: {', '.join(sorted(names))}."
            episode.trace.append({"tool": action[0], "obs_chars": len(observation)})
        else:
            observation = env.call(action[0], action[1])
            episode.trace.append({"tool": action[0], "obs_chars": len(observation)})
        if env.done:
            episode.status = "ok"
            break
        messages.append(
            {
                "role": "user",
                "content": "Observation:\n" + observation[: config.max_observation_chars],
            }
        )
    episode.wall_s = time.monotonic() - start
    return episode
