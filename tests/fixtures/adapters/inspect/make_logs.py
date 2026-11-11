"""Generate tiny Inspect AI logs offline with the mockllm provider (mock data, not model results).

Usage: python make_logs.py <out_dir> <log_format: eval|json> [--invalidate]

Task `tiny_qa`: 4 samples, epochs=3, four scorers (match, letter_grade, closeness, has_answer).
The mock model answers each question with a fixed string, so every outcome is mock data.
- sample "loop" keeps generating until the task message_limit fires (a limit).
- sample "sum" raises a solver error on epoch 2 (an errored sample; fail_on_error=False).
- with --invalidate, sample "diff" epoch 1 is invalidated with inspect_ai.log.invalidate_samples.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from inspect_ai import Task, eval, task
from inspect_ai.dataset import Sample
from inspect_ai.log import ProvenanceData, invalidate_samples, write_eval_log
from inspect_ai.model import ChatMessageUser, ModelOutput, get_model
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    NOANSWER,
    PARTIAL,
    Score,
    Target,
    accuracy,
    match,
    mean,
    scorer,
)
from inspect_ai.solver import Generate, TaskState, solver

ANSWERS = {
    "What is 2 + 3?": "5",
    "What is 4 * 6?": "24 or 25",
    "What is 9 - 4?": "4",
    "Count upward forever.": "1",
}


def mock_output(input, tools, tool_choice, config):
    """Answer from the fixed table; the subtraction answer changes by epoch."""
    text = ""
    for message in input:
        if message.role == "user":
            text = message.text
    answer = ANSWERS.get(text, "1")
    if text == "What is 9 - 4?" and any("[epoch 2]" in m.text for m in input):
        answer = "5"
    if text == "What is 9 - 4?" and any("[epoch 3]" in m.text for m in input):
        answer = "I don't know."
    return ModelOutput.from_content(model="mockllm/model", content=answer)


@solver
def flaky_environment():
    """Raise on sample 'sum' epoch 2, to record an errored sample."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if state.sample_id == "sum" and state.epoch == 2:
            raise RuntimeError("mock environment failure (fixture)")
        state.messages.insert(0, ChatMessageUser(content=f"[epoch {state.epoch}]"))
        return state

    return solve


@solver
def keep_asking():
    """For sample 'loop', keep generating until the message limit ends the sample."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state = await generate(state)
        if state.sample_id == "loop":
            while True:
                state.messages.append(ChatMessageUser(content="Count upward forever."))
                state = await generate(state)
        return state

    return solve


def _number(text: str) -> float | None:
    found = re.findall(r"-?\d+(?:\.\d+)?", text)
    return float(found[0]) if found else None


@scorer(metrics=[accuracy()])
def letter_grade():
    """C if exact, P if the target is among several numbers, N for no answer, I otherwise."""

    async def score(state: TaskState, target: Target) -> Score:
        out = state.output.completion.strip()
        if "don't know" in out or not out:
            return Score(value=NOANSWER, answer=out)
        if out == target.text:
            return Score(value=CORRECT, answer=out)
        if target.text in re.findall(r"\d+", out):
            return Score(value=PARTIAL, answer=out)
        return Score(value=INCORRECT, answer=out)

    return score


@scorer(metrics=[mean()])
def closeness():
    """1 / (1 + |answer - target|) for numeric targets; unscored when no number is given."""

    async def score(state: TaskState, target: Target) -> Score:
        got, want = _number(state.output.completion), _number(target.text)
        if got is None or want is None:
            if not hasattr(Score, "unscored"):  # Inspect before Score.unscored existed
                return Score(value=0.0, explanation="no number to compare")
            return Score.unscored(reason="scoring_failed", explanation="no number to compare")
        return Score(value=1.0 / (1.0 + abs(got - want)), answer=str(got))

    return score


@scorer(metrics=[accuracy()])
def has_answer():
    """Boolean: True when the output is not a refusal to answer."""

    async def score(state: TaskState, target: Target) -> Score:
        out = state.output.completion.strip()
        return Score(value=bool(out) and "don't know" not in out)

    return score


@task
def tiny_qa() -> Task:
    """Four mock questions; mock data for adapter contract tests."""
    samples = [
        Sample(id="sum", input="What is 2 + 3?", target="5"),
        Sample(id="product", input="What is 4 * 6?", target="24"),
        Sample(id="diff", input="What is 9 - 4?", target="5"),
        Sample(id="loop", input="Count upward forever.", target="never"),
    ]
    return Task(
        dataset=samples,
        solver=[flaky_environment(), keep_asking()],
        scorer=[match(location="any"), letter_grade(), closeness(), has_answer()],
        message_limit=6,
    )


def main() -> None:
    out_dir, log_format = Path(sys.argv[1]), sys.argv[2]
    model = get_model("mockllm/model", custom_outputs=mock_output)
    [log] = eval(
        tiny_qa(),
        model=model,
        epochs=3,
        fail_on_error=False,
        log_dir=str(out_dir),
        log_format=log_format,
        max_samples=1,
        display="none",
    )
    print(log.location, log.status)
    if "--invalidate" in sys.argv:
        target = [s.uuid for s in log.samples or [] if s.id == "diff" and s.epoch == 1]
        log = invalidate_samples(
            log,
            [u for u in target if u],
            ProvenanceData(author="noise-floor fixture script", reason="mock invalidation"),
        )
        write_eval_log(log, log.location)
        print("invalidated", target)


if __name__ == "__main__":
    main()
