"""evalstats: statistics for eval results.

Paired and clustered comparisons, bootstrap and mixed-model intervals, power and
minimum detectable effects, rank-confidence sets, variance components, and
policy-sensitivity bounds for model evaluation results.
"""

from __future__ import annotations

from evalstats.bounds import BoundedPair, BoundedScore, PolicyReport, policy_bounds
from evalstats.compare import PairedComparison, ScoreInterval, paired_difference, score_interval
from evalstats.glmm import Decomposition, VarianceComponent, decompose
from evalstats.intervals import (
    Interval,
    bca_interval,
    cluster_robust_mean,
    interval_for_mean,
    percentile_interval,
    t_interval,
)
from evalstats.power import PairedDesign, design_from_results, mde, power_at, tasks_needed
from evalstats.rank import AdjacentPair, RankConfidence, RankInterval, rank_confidence
from evalstats.readers import read_csv, read_jsonl, read_results
from evalstats.schema import EvalResults, PairedScores, SchemaError

__version__ = "0.0.1"

__all__ = [
    "AdjacentPair",
    "BoundedPair",
    "BoundedScore",
    "Decomposition",
    "EvalResults",
    "Interval",
    "PairedComparison",
    "PairedDesign",
    "PairedScores",
    "PolicyReport",
    "RankConfidence",
    "RankInterval",
    "SchemaError",
    "ScoreInterval",
    "VarianceComponent",
    "__version__",
    "bca_interval",
    "cluster_robust_mean",
    "decompose",
    "design_from_results",
    "interval_for_mean",
    "mde",
    "paired_difference",
    "percentile_interval",
    "policy_bounds",
    "power_at",
    "rank_confidence",
    "read_csv",
    "read_jsonl",
    "read_results",
    "score_interval",
    "t_interval",
    "tasks_needed",
]
