from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from noisefloor.client import BackendError, ChatRequest, ChatResponse, Client, Ledger
from noisefloor.minimal_agent import AgentConfig, parse_action, run_episode, system_prompt


@dataclass(frozen=True)
class Spec:
    name: str
    description: str
    parameters: Mapping[str, Any]


@dataclass
class FakeEnv:
    instruction: str = "find the code"
    done: bool = False
    answer: str | None = None
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def tool_specs(self) -> Sequence[Spec]:
        return [
            Spec("read_file", "Read a file.", {"properties": {"path": {"type": "string"}}}),
            Spec("submit", "Submit the answer.", {"properties": {"answer": {}}}),
        ]

    def call(self, name: str, args: Mapping[str, Any]) -> str:
        self.calls.append((name, dict(args)))
        if name == "submit":
            self.done, self.answer = True, str(args.get("answer"))
            return "submitted"
        return "code=42"


class ScriptedClient(Client):
    def __init__(self, replies: list[str | Exception], tmp: Path) -> None:
        super().__init__(ledger=Ledger(tmp / "l.jsonl"), cache_dir=tmp / "c", caps={})
        self.replies = replies
        self.requests: list[ChatRequest] = []

    def chat(
        self, request: ChatRequest, *, kind: str = "local", timeout: float = 600.0
    ) -> ChatResponse:
        self.requests.append(request)
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return ChatResponse(text=item, input_tokens=10, output_tokens=5, wall_s=0.1)


CFG = AgentConfig(backend="openai-local", model="m", max_steps=5)


def test_parse_action_formats() -> None:
    assert parse_action('{"tool": "a", "args": {"x": 1}}') == ("a", {"x": 1})
    assert parse_action('Sure!\n```json\n{"tool": "a", "args": {}}\n```') == ("a", {})
    assert parse_action('I will {broken then {"name": "b", "arguments": {"p": 2}}') == (
        "b",
        {"p": 2},
    )
    assert parse_action('{"tool": "a"}') == ("a", {})
    assert parse_action("no json here") is None
    assert parse_action('{"tool": 3, "args": {}} [1, 2]') is None
    assert parse_action('{"tool": "a", "args": "x"}') is None


def test_episode_success_with_fenced_and_bad_replies(tmp_path: Path) -> None:
    client = ScriptedClient(
        [
            "let me think",
            '```json\n{"tool": "read_file", "args": {"path": "/a"}}\n```',
            '{"tool": "delete_everything", "args": {}}',
            '{"tool": "submit", "args": {"answer": "42"}}',
        ],
        tmp_path,
    )
    env = FakeEnv()
    ep = run_episode(env, client, CFG)
    assert ep.status == "ok" and ep.steps == 4 and ep.parse_errors == 1
    assert env.answer == "42" and env.calls[0] == ("read_file", {"path": "/a"})
    assert [t["tool"] for t in ep.trace] == [None, "read_file", "delete_everything", "submit"]
    assert ep.input_tokens == 40 and ep.output_tokens == 20
    last = client.requests[-1]
    assert last.messages[0]["role"] == "system" and "read_file" in last.messages[0]["content"]
    assert [r.seed for r in client.requests] == [0, 1, 2, 3]
    assert "Observation:" in last.messages[-1]["content"]


def test_episode_step_limit_and_backend_error(tmp_path: Path) -> None:
    client = ScriptedClient(['{"tool": "read_file", "args": {}}'] * 5, tmp_path)
    ep = run_episode(FakeEnv(), client, CFG)
    assert ep.status == "timeout" and ep.steps == 5
    failing = ScriptedClient(['{"tool": "read_file", "args": {}}', BackendError("down")], tmp_path)
    ep2 = run_episode(FakeEnv(), failing, CFG)
    assert ep2.status == "infra_error" and ep2.steps == 1 and "down" in ep2.error


def test_wall_clock_limit(tmp_path: Path) -> None:
    client = ScriptedClient(['{"tool": "read_file", "args": {}}'] * 5, tmp_path)
    cfg = AgentConfig(backend="openai-local", model="m", wall_limit_s=-1.0)
    ep = run_episode(FakeEnv(), client, cfg)
    assert ep.status == "timeout" and ep.steps == 0


def test_system_prompt_lists_tools() -> None:
    text = system_prompt(FakeEnv().tool_specs())
    assert "- read_file: Read a file." in text and '"path"' in text


@pytest.mark.property
def test_parse_action_round_trip() -> None:
    from hypothesis import given, settings
    from hypothesis import strategies as st

    @settings(max_examples=100, deadline=None)
    @given(
        st.text(min_size=1, max_size=12),
        st.dictionaries(st.text(max_size=5), st.integers() | st.text(max_size=5), max_size=3),
    )
    def check(tool: str, args: dict[str, Any]) -> None:
        import json

        text = "prefix " + json.dumps({"tool": tool, "args": args}) + " suffix"
        assert parse_action(text) == (tool, args)

    check()
