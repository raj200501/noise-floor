import json
from pathlib import Path

import pandas as pd
import pytest

from evalstats import EvalResults, SchemaError, read_csv, read_jsonl, read_results


def frame(**extra: list[object]) -> pd.DataFrame:
    base = {
        "model": ["a", "a", "b", "b"],
        "task": ["x", "y", "x", "y"],
        "run": [0, 0, 0, 0],
        "outcome": [1, 0, 0, 0],
    }
    base.update(extra)
    return pd.DataFrame(base)


def test_from_frame_validates_and_orders_columns() -> None:
    res = EvalResults.from_frame(frame(seed=[1, 2, 3, 4]))
    assert list(res.frame.columns) == ["model", "task", "run", "outcome", "seed"]
    assert res.models == ["a", "b"] and res.tasks == ["x", "y"]
    assert res.covariates == ["seed"] and res.is_binary and len(res) == 4
    assert res.frame["run"].tolist() == ["0", "0", "0", "0"]


def test_column_mapping_and_run_numbering() -> None:
    df = pd.DataFrame({"m": ["a", "a", "a"], "t": ["x", "x", "y"], "ok": [1, 0, 1]})
    res = EvalResults.from_frame(df, model="m", task="t", outcome="ok", run=None)
    assert res.frame["run"].tolist() == ["0", "1", "0"]
    assert res.run_counts("a").to_dict() == {"x": 2, "y": 1}


@pytest.mark.parametrize(
    ("df", "message"),
    [
        (frame().drop(columns="outcome"), "missing columns"),
        (frame(outcome=[1, 0, None, 0]), "non-finite"),
        (frame(outcome=[1, 0, 2, 0]), "outside"),
        (frame(outcome=["yes", 0, 0, 0]), "non-numeric"),
        (frame(model=["a", None, "b", "b"]), "missing values"),
        (frame(task=["x", "x", "x", "y"]), "duplicate"),
        (frame().iloc[0:0], "no rows"),
    ],
)
def test_validation_errors(df: pd.DataFrame, message: str) -> None:
    with pytest.raises(SchemaError, match=message):
        EvalResults.from_frame(df)


def test_rename_clash_is_rejected() -> None:
    df = frame().rename(columns={"model": "agent"}).assign(model="other")
    with pytest.raises(SchemaError, match="overwritten"):
        EvalResults.from_frame(df, model="agent")


def test_scores_pairing_and_matrix() -> None:
    df = pd.DataFrame(
        {
            "model": ["a", "a", "a", "b", "b"],
            "task": ["x", "x", "y", "x", "z"],
            "run": [0, 1, 0, 0, 0],
            "outcome": [1, 0, 1, 1, 0],
        }
    )
    res = EvalResults.from_frame(df)
    assert res.task_scores("a").to_dict() == {"x": 0.5, "y": 1.0}
    assert res.score("a") == pytest.approx(0.75)
    pair = res.paired("a", "b")
    assert pair.tasks == ("x",) and pair.differences.tolist() == [-0.5]
    assert pair.runs_a.tolist() == [2] and pair.runs_b.tolist() == [1]
    matrix = res.score_matrix()
    assert list(matrix.columns) == ["a", "b"] and matrix.loc["z", "a"] != matrix.loc["z", "a"]
    with pytest.raises(SchemaError, match="unknown model"):
        res.task_scores("c")
    with pytest.raises(SchemaError, match="share no tasks"):
        res.subset(tasks=["y", "z"]).paired("a", "b")


def test_subset() -> None:
    res = EvalResults.from_frame(frame())
    assert res.subset(models=["a"]).models == ["a"]
    assert res.subset(tasks=["x"]).tasks == ["x"]
    with pytest.raises(SchemaError, match="empty"):
        res.subset(models=["nope"])


def test_readers_round_trip(tmp_path: Path) -> None:
    df = frame()
    csv = tmp_path / "r.csv"
    df.to_csv(csv, index=False)
    assert read_csv(csv).frame.equals(EvalResults.from_frame(df).frame)
    tsv = tmp_path / "r.tsv"
    df.to_csv(tsv, sep="\t", index=False)
    assert read_results(tsv).frame.equals(EvalResults.from_frame(df).frame)
    jsonl = tmp_path / "r.jsonl"
    jsonl.write_text("\n".join(json.dumps(r) for r in df.to_dict("records")) + "\n\n")
    assert read_jsonl(jsonl).frame.equals(EvalResults.from_frame(df).frame)
    assert read_results(jsonl).models == ["a", "b"]
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"model": "a"\n')
    with pytest.raises(SchemaError, match="invalid JSON"):
        read_jsonl(bad)
    with pytest.raises(SchemaError, match="unsupported"):
        read_results(tmp_path / "r.parquet")
