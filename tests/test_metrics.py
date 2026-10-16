"""Metric helpers: weighted AP / ROC-AUC match sklearn, top-k flags, bootstrap and calibration."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src import calibration
from src.metrics import (
    SortedScores,
    capacity_metrics,
    cluster_bootstrap,
    paired_differences,
    summarize_bootstrap,
    top_k_flags,
)

rng = np.random.default_rng(0)
N = 2000
Y = (rng.random(N) < 0.35).astype(int)
SCORE = Y * 0.5 + rng.random(N)
TIED = np.round(SCORE * 4)            # many ties, like the hour-count baseline
DAY = np.repeat(np.arange(4), N // 4)
STORE = rng.integers(0, 40, N)


def test_sorted_scores_match_sklearn_with_unit_weights():
    for s in (SCORE, TIED):
        ap, auc = SortedScores(Y, s).metrics(np.ones(N))
        assert abs(ap - average_precision_score(Y, s)) < 1e-12
        assert abs(auc - roc_auc_score(Y, s)) < 1e-12


def test_sorted_scores_match_sklearn_with_weights():
    w = rng.integers(0, 3, N).astype(float)
    ss = SortedScores(Y, TIED)
    ap, auc = ss.metrics(w[ss.order])
    assert abs(ap - average_precision_score(Y, TIED, sample_weight=w)) < 1e-12
    assert abs(auc - roc_auc_score(Y, TIED, sample_weight=w)) < 1e-12


def test_top_k_flags_per_day():
    f = top_k_flags(TIED, DAY, 0.10)
    counts = pd.Series(f).groupby(DAY).sum()
    assert (counts == round(N / 4 * 0.10)).all()
    # every flagged row scores at least as high as every unflagged row of the same day
    for d in range(4):
        m = DAY == d
        assert TIED[m & f].min() >= TIED[m & ~f].max()


def test_capacity_metrics_perfect_ranking():
    out = capacity_metrics(Y, Y + rng.random(N) * 0.1, DAY, ks=(0.10,))
    assert out["precision_at_10"] == 1.0
    assert 0 < out["recall_at_10"] < 1


def test_bootstrap_intervals_cover_point_and_pairs_are_paired():
    scores = {"a": SCORE, "b": TIED}
    draws = cluster_bootstrap(Y, scores, STORE, DAY, n_boot=200)
    point = pd.DataFrame({m: {"pr_auc": average_precision_score(Y, s)} for m, s in scores.items()}).T
    ci = summarize_bootstrap(draws, point, ["pr_auc"]).set_index("model")
    for m in scores:
        assert ci.loc[m, "pr_auc_lo"] <= ci.loc[m, "pr_auc"] <= ci.loc[m, "pr_auc_hi"]
    same = paired_differences(draws, point, [("a vs a", "a", "a")], ["pr_auc"]).iloc[0]
    assert same.ci_lo == same.ci_hi == 0 and not same.significant_95


def test_calibration_maps_are_monotone_and_reduce_brier():
    raw = 1 / (1 + np.exp(-(SCORE * 3 - 1)))   # miscalibrated probabilities
    for name, fit in calibration.FITTERS.items():
        cal = fit(raw, Y)
        p = calibration.apply(cal, np.sort(raw))
        assert np.all(np.diff(p) >= -1e-12), name
        assert np.mean((calibration.apply(cal, raw) - Y) ** 2) <= np.mean((raw - Y) ** 2)
    method, briers = calibration.choose_method(raw, Y, STORE)
    assert method in ("platt", "isotonic") and briers[method] <= briers["none"]
