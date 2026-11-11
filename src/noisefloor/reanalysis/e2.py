"""E2 and public-data E3 (prereg/0001, amended by prereg/0002).

Two stages, both deterministic:

* ``build``: from the raw fetches (outside the repository), write the committed
  snapshots under ``data/snapshots/`` and the manifests under ``artifacts/e2/<board>/``
  (source, revision or retrieval date, license, file hashes, raw-file hashes).
* ``analyze``: from the committed snapshots only, compute every number the
  preregistration asks for and write ``artifacts/e2/<board>/results.json`` and
  ``artifacts/e3/bounds.json``. ``verify`` recomputes and compares with the banked files.

Boards: ``tb2`` (Terminal-Bench 2.0 submissions) and ``epoch-swebench`` (Epoch AI,
SWE-bench Verified).
"""

from __future__ import annotations

import gzip
import hashlib
import itertools
import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from evalstats import (
    EvalResults,
    paired_difference,
    policy_bounds,
    rank_confidence,
    score_interval,
)
from evalstats.rank import complete_score_matrix
from noisefloor.reanalysis import epoch, tb2

REPO = Path(__file__).resolve().parents[3]
PREREG = "prereg/0001-e2-leaderboard-reanalysis.md"
AMENDMENT = "prereg/0002-e2-amend-epoch-reference-rule.md"
SNAPSHOTS = REPO / "data" / "snapshots"
ARTIFACTS = REPO / "artifacts"
LEVEL = 0.95
N_BOOT = 4000
SEED = 20261104
EXCLUDE_PREFIXES = ("OB-1", "pilot", "Forge")
RESOLVED_THRESHOLD = 0.5
AMBIGUOUS = ("infra_error", "timeout", "flagged")
BOARDS = ("tb2", "epoch-swebench")
TB2_SNAPSHOT = SNAPSHOTS / "tb2-572b2614"
EPOCH_SNAPSHOT = SNAPSHOTS / "epoch-swebench"


def sha256_file(path: Path) -> str:
    """SHA-256 of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path: Path) -> str:
    """Path relative to the repository root."""
    return str(path.resolve().relative_to(REPO))


def _canonical(obj: Any) -> str:
    return json.dumps(obj, indent=1, sort_keys=True, allow_nan=False) + "\n"


def _clean(value: Any) -> Any:
    """JSON-safe floats (NaN and infinities become None), recursively."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(v) for v in value]
    return value


# --------------------------------------------------------------------------- build


def job_datasets(config: Mapping[str, Any]) -> str:
    """The datasets a Harbor job configuration names, as ``name@version`` or ``path:<dir>``."""
    names = []
    for ds in config.get("datasets") or []:
        if not isinstance(ds, Mapping):
            continue
        if ds.get("name"):
            names.append(f"{ds.get('name')}@{ds.get('version')}")
        elif ds.get("path"):
            names.append(f"path:{ds.get('path')}")
    return ";".join(names)


def is_tb20(datasets: str) -> bool:
    """False if a job's datasets name a Terminal-Bench version other than 2.0 (2.1)."""
    low = datasets.lower()
    return not any(marker in low for marker in ("2-1", "2.1"))


def build_tb2(raw_dir: Path, out_dir: Path | None = None) -> dict[str, Any]:
    """Write the TB2 snapshot (trials, jobs) and its manifest from a raw fetch."""
    out_dir = out_dir or TB2_SNAPSHOT
    fetch_manifest = json.loads((raw_dir / "fetch_manifest.json").read_text())
    submissions = fetch_manifest.get("submissions") or tb2.list_submissions(
        str(fetch_manifest["revision"])
    )
    trials = tb2.load_trials(raw_dir)
    totals = tb2.load_job_totals(raw_dir)
    datasets = []
    for _, row in totals.iterrows():
        cfg_path = raw_dir / str(row["submission"]) / f"{row['job']}.config.json"
        try:
            config = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        except ValueError:
            config = {}
        datasets.append(job_datasets(config if isinstance(config, Mapping) else {}))
    totals["datasets"] = datasets
    totals["tb20"] = [is_tb20(d) for d in datasets]
    out_dir.mkdir(parents=True, exist_ok=True)
    trials_path, jobs_path = out_dir / "trials.csv.gz", out_dir / "jobs.csv.gz"
    trials_sha = tb2.write_table(trials, trials_path)
    jobs_sha = tb2.write_table(totals, jobs_path)
    manifest = {
        "board": "tb2",
        "prereg": PREREG,
        "confirmatory": True,
        "source_url": f"https://huggingface.co/datasets/{tb2.DATASET}",
        "revision": fetch_manifest["revision"],
        "retrieved": fetch_manifest.get("retrieved", ""),
        "license": tb2.LICENSE,
        "canary": tb2.CANARY,
        "snapshot": rel(trials_path),
        "sha256": trials_sha,
        "files": {rel(trials_path): trials_sha, rel(jobs_path): jobs_sha},
        "submissions": sorted(submissions),
        "raw_files": fetch_manifest["files"],
        "fetch_failures": fetch_manifest.get("failures", []),
        "command": "python -m noisefloor.reanalysis.tb2 fetch <raw>; "
        "python -m noisefloor.reanalysis build tb2 <raw>",
    }
    path = ARTIFACTS / "e2" / "tb2" / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical(manifest))
    return manifest


def build_epoch(raw_dir: Path, out_dir: Path | None = None) -> dict[str, Any]:
    """Write the Epoch snapshot (samples, logs) and its manifest from a raw fetch."""
    out_dir = out_dir or EPOCH_SNAPSHOT
    fetch_manifest = json.loads((raw_dir / "fetch_manifest.json").read_text())
    samples = epoch.load_samples(raw_dir)
    rows = epoch.read_rows(raw_dir / "benchmark_data.zip")
    published = {r.get("id", ""): r for r in rows}
    logs = pd.DataFrame(
        [
            {
                "log_id": log_id,
                "entry": f"{info['model']} [{log_id}]",
                "model_version": info["model"],
                "published_mean_score": published.get(log_id, {}).get("mean_score", ""),
                "started_at": published.get(log_id, {}).get("Started at", ""),
                "url": info["url"],
                "etag": info["etag"],
                "version_id": info["version_id"],
                "header_sha256": info["members"]["header.json"],
                "summaries_sha256": info["members"]["summaries.json"],
            }
            for log_id, info in sorted(fetch_manifest["logs"].items())
        ]
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_path, logs_path = out_dir / "samples.csv.gz", out_dir / "logs.csv.gz"
    samples_sha = epoch.write_table(samples, samples_path)
    logs_sha = epoch.write_table(logs, logs_path)
    manifest = {
        "board": "epoch-swebench",
        "prereg": PREREG,
        "amendment": AMENDMENT,
        "confirmatory": True,
        "source_url": epoch.ARCHIVE_URL,
        "archive_sha256": fetch_manifest["archive_sha256"],
        "archive_etag": fetch_manifest.get("archive_etag", ""),
        "retrieved": fetch_manifest["retrieved"],
        "license": epoch.LICENSE,
        "credit": epoch.CREDIT,
        "snapshot": rel(samples_path),
        "sha256": samples_sha,
        "files": {rel(samples_path): samples_sha, rel(logs_path): logs_sha},
        "fetch_failures": fetch_manifest.get("failures", []),
        "command": "python -m noisefloor.reanalysis.epoch fetch <raw>; "
        "python -m noisefloor.reanalysis build epoch-swebench <raw>",
    }
    path = ARTIFACTS / "e2" / "epoch-swebench" / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical(manifest))
    return manifest


# --------------------------------------------------------------------------- load


def read_gz_csv(path: Path) -> pd.DataFrame:
    """Read a snapshot table (comment lines start with #)."""
    with gzip.open(path, "rt") as fh:
        return pd.read_csv(
            fh,
            comment="#",
            keep_default_na=False,
            na_values=[""],
            dtype={"exception": str, "status_reason": str, "task": str},
        )


def tb2_tables(snapshot: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """TB2 trials and jobs from the committed snapshot."""
    snapshot = snapshot or TB2_SNAPSHOT
    return read_gz_csv(snapshot / "trials.csv.gz"), read_gz_csv(snapshot / "jobs.csv.gz")


def tb2_frame(
    trials: pd.DataFrame, jobs: pd.DataFrame, submissions: Sequence[str] = ()
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Included TB2 trials (long format with status) and the reason for each excluded entry.

    ``submissions`` lists every submission folder at the revision, so entries with no
    parseable job-level result are reported too (prereg/0001 exclusion 3).
    """
    excluded: dict[str, str] = {}
    job_key = jobs["submission"].astype(str) + "|" + jobs["job"].astype(str)
    non20 = set(job_key[~jobs["tb20"].astype(bool)])
    key = trials["submission"].astype(str) + "|" + trials["job"].astype(str)
    for sub in sorted(set(trials["submission"])):
        if sub.startswith(EXCLUDE_PREFIXES):
            excluded[sub] = "removed or rescored by the maintainers (Leaderboard Integrity Update)"
    keep = ~trials["submission"].isin(list(excluded)) & ~key.isin(non20)
    for sub in sorted(set(trials["submission"]) - set(excluded)):
        if not keep[trials["submission"] == sub].any():
            excluded[sub] = "every job ran a dataset other than Terminal-Bench 2.0"
    for sub in sorted(set(submissions) - set(trials["submission"])):
        excluded[sub] = "no job-level result.json in the documented format"
    frame = trials.loc[keep].copy()
    frame = frame.assign(model=frame["submission"], run=frame["trial"])
    return frame[["model", "task", "run", "outcome", "status", "submission", "job"]], excluded


def epoch_frame(samples: pd.DataFrame) -> pd.DataFrame:
    """Epoch samples in long format with status (entry label as model)."""
    frame = samples.assign(model=samples["entry"])
    return frame[["model", "task", "run", "outcome", "status", "status_reason"]].copy()


# --------------------------------------------------------------------------- analysis


def board_analysis(
    full: EvalResults,
    reference: EvalResults,
    *,
    pooled: Mapping[str, float],
    analysis_set: str,
) -> dict[str, Any]:
    """Q1 to Q3 for one analysis set (see prereg/0001 Methods)."""
    models = reference.models
    scores = {m: score_interval(reference, m, level=LEVEL) for m in models}
    order = sorted(models, key=lambda m: (-scores[m].interval.estimate, m))
    entries = [
        {
            "entry": m,
            "score": scores[m].interval.estimate,
            "lo": scores[m].interval.lo,
            "hi": scores[m].interval.hi,
            "n_tasks": scores[m].n_tasks,
            "runs_per_task": scores[m].runs_per_task,
            "pooled_score": pooled.get(m),
        }
        for m in order
    ]
    adjacent: list[dict[str, Any]] = []
    for a, b in itertools.pairwise(order):
        t = paired_difference(reference, a, b, method="cluster_t", level=LEVEL)
        bs = paired_difference(
            reference, a, b, method="bootstrap", level=LEVEL, n_boot=N_BOOT, seed=SEED
        )
        adjacent.append(
            {
                "higher": a,
                "lower": b,
                "difference": t.difference.estimate,
                "lo": t.difference.lo,
                "hi": t.difference.hi,
                "p_value": t.p_value,
                "n_tasks": t.n_tasks,
                "supported": t.supported,
                "bootstrap_lo": bs.difference.lo,
                "bootstrap_hi": bs.difference.hi,
                "bootstrap_supported": bs.supported,
            }
        )
    n_supported = sum(bool(p["supported"]) for p in adjacent)
    share = n_supported / len(adjacent) if adjacent else math.nan
    _, matrix, dropped = complete_score_matrix(reference)
    sim = rank_confidence(reference, level=LEVEL, simultaneous=True, n_boot=N_BOOT, seed=SEED)
    marg = rank_confidence(reference, level=LEVEL, simultaneous=False, n_boot=N_BOOT, seed=SEED)
    marginal = {r.model: r for r in marg.intervals}
    widths = sorted(r.worst_rank - r.best_rank + 1 for r in sim.intervals)
    bounds = policy_bounds(
        full, ambiguous=AMBIGUOUS, policy="as_scored", level=LEVEL, reference_order=order
    )
    result: dict[str, Any] = _clean(
        {
            "analysis_set": analysis_set,
            "n_entries": len(models),
            "n_runs": len(reference),
            "level": LEVEL,
            "entries": entries,
            "q1": {
                "method": "cluster_t paired difference on shared tasks",
                "n_pairs": len(adjacent),
                "n_supported": n_supported,
                "share_supported": share,
                "label": "mostly resolved" if share >= RESOLVED_THRESHOLD else "mostly unresolved",
                "n_supported_bootstrap": sum(p["bootstrap_supported"] for p in adjacent),
                "pairs": adjacent,
            },
            "q2": {
                "method": "simultaneous rank intervals, task bootstrap of max |t|",
                "n_boot": N_BOOT,
                "seed": SEED,
                "n_tasks": int(matrix.shape[0]),
                "dropped_tasks": dropped,
                "critical_value": sim.critical_value,
                "median_width": float(pd.Series(widths).median()),
                "n_could_be_first": len(sim.top_set),
                "top_set": list(sim.top_set),
                "intervals": [
                    {
                        "entry": r.model,
                        "rank": r.rank,
                        "best": r.best_rank,
                        "worst": r.worst_rank,
                        "marginal_best": marginal[r.model].best_rank,
                        "marginal_worst": marginal[r.model].worst_rank,
                    }
                    for r in sim.intervals
                ],
            },
            "q3": {
                "ambiguous": list(AMBIGUOUS),
                "n_pairs": len(bounds.adjacent),
                "n_identified": sum(p.identified for p in bounds.adjacent),
                "n_robust": sum(p.robust for p in bounds.adjacent),
                "ranking_changes": bounds.ranking_changes,
                "entries": [
                    {
                        "entry": b.model,
                        **{f"score_{k}": v for k, v in b.scores.items()},
                        "lower": b.lower,
                        "upper": b.upper,
                        "ci_lo": b.ci_lo,
                        "ci_hi": b.ci_hi,
                        "ambiguous_runs": b.ambiguous_runs,
                        "ambiguous_share": b.ambiguous_share,
                    }
                    for b in bounds.models
                ],
                "pairs": [
                    {
                        "higher": p.higher,
                        "lower": p.lower,
                        "bound_lo": p.bound_lo,
                        "bound_hi": p.bound_hi,
                        "ci_lo": p.ci_lo,
                        "ci_hi": p.ci_hi,
                        "identified": p.identified,
                        "robust": p.robust,
                    }
                    for p in bounds.adjacent
                ],
            },
        }
    )
    return result


def analyze_tb2(snapshot: Path | None = None) -> dict[str, Any]:
    """Results for the TB2 board (primary and secondary analysis sets)."""
    trials, jobs = tb2_tables(snapshot)
    manifest = json.loads((ARTIFACTS / "e2" / "tb2" / "manifest.json").read_text())
    frame, excluded = tb2_frame(trials, jobs, manifest.get("submissions", []))
    totals = jobs.loc[jobs["submission"].isin(set(frame["submission"]))]
    pooled_num = frame.groupby("submission")["outcome"].sum()
    pooled_den = totals.groupby("submission")["n_total_trials"].sum()
    pooled = {s: float(pooled_num[s] / pooled_den[s]) for s in pooled_num.index}
    tasks_per = frame.groupby("submission")["task"].nunique()
    n_tasks = int(frame["task"].nunique())
    full_cov = sorted(str(s) for s, n in tasks_per.items() if n == n_tasks)
    results = {}
    for name, subs in (("primary", full_cov), ("secondary", sorted(map(str, tasks_per.index)))):
        part = frame.loc[frame["submission"].isin(subs)]
        full = EvalResults.from_frame(part[["model", "task", "run", "outcome", "status"]])
        results[name] = board_analysis(full, full, pooled=pooled, analysis_set=name)
    return {
        "board": "tb2",
        "prereg": PREREG,
        "confirmatory": True,
        "reference_policy": "as_scored",
        "excluded_entries": excluded,
        "n_canonical_tasks": n_tasks,
        "n_unlisted_trials": int(totals["n_unlisted"].sum()),
        "sets": results,
    }


def analyze_epoch(snapshot: Path | None = None) -> dict[str, Any]:
    """Results for the Epoch SWE-bench Verified board (reference rule of prereg/0002)."""
    samples = read_gz_csv((snapshot or EPOCH_SNAPSHOT) / "samples.csv.gz")
    frame = epoch_frame(samples)
    full = EvalResults.from_frame(frame)
    reference = EvalResults.from_frame(frame.loc[frame["status"] != "infra_error"])
    pooled = {
        str(k): float(v) for k, v in reference.frame.groupby("model")["outcome"].mean().items()
    }
    return {
        "board": "epoch-swebench",
        "prereg": PREREG,
        "amendment": AMENDMENT,
        "confirmatory": True,
        "reference_policy": "drop infra_error samples (prereg/0002)",
        "excluded_entries": {},
        "sets": {"primary": board_analysis(full, reference, pooled=pooled, analysis_set="primary")},
    }


ANALYZERS: dict[str, Callable[[], dict[str, Any]]] = {
    "tb2": analyze_tb2,
    "epoch-swebench": analyze_epoch,
}


def e3_summary(results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The E3 bounds artifact: one entry per board (E1 entries are added by the E1 analysis)."""
    entries = []
    for board, res in results.items():
        q3 = res["sets"]["primary"]["q3"]
        entries.append({"source": board, "analysis_set": "primary", **q3})
    return {"prereg": PREREG, "confirmatory": True, "entries": entries}


def analyze(boards: Sequence[str] = BOARDS, write: bool = True) -> dict[str, dict[str, Any]]:
    """Compute results for the boards; write the artifacts unless ``write`` is False."""
    results = {b: ANALYZERS[b]() for b in boards}
    if write:
        for board, res in results.items():
            path = ARTIFACTS / "e2" / board / "results.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(_canonical(res))
        bounds_path = ARTIFACTS / "e3" / "bounds.json"
        bounds = json.loads(bounds_path.read_text()) if bounds_path.exists() else {}
        other = [e for e in bounds.get("entries", []) if e.get("source") not in results]
        summary = e3_summary(results)
        summary["entries"] = other + summary["entries"]
        bounds_path.parent.mkdir(parents=True, exist_ok=True)
        bounds_path.write_text(_canonical(summary))
    return results


def verify(boards: Sequence[str] = BOARDS) -> list[str]:
    """Problems found when recomputing the banked results (empty if they reproduce)."""
    problems = []
    for board in boards:
        manifest = json.loads((ARTIFACTS / "e2" / board / "manifest.json").read_text())
        for path, digest in manifest["files"].items():
            if sha256_file(REPO / path) != digest:
                problems.append(f"{board}: snapshot hash mismatch for {path}")
        banked = (ARTIFACTS / "e2" / board / "results.json").read_text()
        fresh = _canonical(ANALYZERS[board]())
        if fresh != banked:
            problems.append(f"{board}: recomputed results differ from the banked results.json")
    return problems
