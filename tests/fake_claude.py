"""A fake ``claude`` CLI for tests: emits headless stream-json from a JSON plan.

The plan is the file named by ``FAKE_CLAUDE_PLAN``:

* ``init``: fields merged into the init event (``apiKeySource``, ``tools``, ...);
* ``turns``: assistant turns, in order; each may have ``text``, ``tool_uses`` (a count),
  ``write`` ({relative path: content}, written into the working directory before the
  turn is printed), ``rate_limit`` (printed as a rate_limit_event after the turn),
  ``sleep`` (seconds, before the turn), and ``same_id`` (repeat the previous turn's id);
* ``result``: fields merged into the result event, or ``null`` for no result event;
* ``replies``: single-call mode (``--tools ""``); the reply is ``replies[n]`` where n is
  the number of ``<assistant>`` turns in the prompt;
* ``ignore_max_turns``: print every turn even past ``--max-turns``;
* ``exit``: the exit code.

Each invocation appends {argv, cwd, env flags} to ``FAKE_CLAUDE_LOG`` if set.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def emit(event: dict[str, object]) -> None:
    print(json.dumps(event), flush=True)


def arg(argv: list[str], name: str) -> str | None:
    return argv[argv.index(name) + 1] if name in argv else None


def main() -> int:
    argv = sys.argv[1:]
    plan = json.loads(Path(os.environ["FAKE_CLAUDE_PLAN"]).read_text())
    prompt = sys.stdin.read()
    log = os.environ.get("FAKE_CLAUDE_LOG")
    if log:
        names = ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "GH_TOKEN", "AWS_SECRET_ACCESS_KEY")
        record = {
            "argv": argv,
            "cwd": os.getcwd(),
            "prompt": prompt,
            "present": sorted(n for n in names if n in os.environ),
            "claude_mds_off": os.environ.get("CLAUDE_CODE_DISABLE_CLAUDE_MDS"),
            "auto_memory_off": os.environ.get("CLAUDE_CODE_DISABLE_AUTO_MEMORY"),
        }
        with open(log, "a") as fh:
            fh.write(json.dumps(record) + "\n")
    init = {"type": "system", "subtype": "init", "model": "claude-fake-1", "apiKeySource": "none"}
    emit({**init, **plan.get("init", {})})
    if arg(argv, "--tools") == "":
        reply = plan["replies"][min(prompt.count("<assistant>"), len(plan["replies"]) - 1)]
        emit(
            {
                "type": "assistant",
                "message": {"id": "m0", "content": [{"type": "text", "text": reply}]},
            }
        )
        emit(
            {
                "type": "result",
                "subtype": "success",
                "result": reply,
                "usage": {"input_tokens": 5, "output_tokens": 2},
                "total_cost_usd": 0.001,
            }
        )
        return int(plan.get("exit", 0))
    max_turns = int(arg(argv, "--max-turns") or 10**6)
    turns = plan.get("turns", [])
    capped = len(turns) > max_turns and not plan.get("ignore_max_turns")
    last_id = ""
    for n, turn in enumerate(turns[:max_turns] if capped else turns):
        time.sleep(float(turn.get("sleep", 0)))
        for rel, content in (turn.get("write") or {}).items():
            target = Path(rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        msg_id = last_id if turn.get("same_id") else f"msg-{n}"
        last_id = msg_id
        content: list[dict[str, object]] = [{"type": "text", "text": turn.get("text", "")}]
        content += [{"type": "tool_use", "name": "Read"}] * int(turn.get("tool_uses", 0))
        usage = {"input_tokens": 10, "cache_read_input_tokens": 5, "output_tokens": 3}
        emit({"type": "assistant", "message": {"id": msg_id, "content": content, "usage": usage}})
        if "rate_limit" in turn:
            emit({"type": "rate_limit_event", "rate_limit_info": turn["rate_limit"]})
    result = plan.get("result", {})
    if result is not None:
        base = {
            "type": "result",
            "subtype": "error_max_turns" if capped else "success",
            "is_error": capped,
            "result": "done",
            "num_turns": min(len(turns), max_turns),
            "total_cost_usd": 0.02,
            "usage": {"input_tokens": 30, "cache_creation_input_tokens": 7, "output_tokens": 9},
        }
        emit({**base, **result})
    return int(plan.get("exit", 0))


if __name__ == "__main__":
    sys.exit(main())
