"""Ranking, threshold and capacity metrics, plus the store-cluster bootstrap.

The capacity view matches how alerts are used: each day the team reviews the top k% of
store-product series by score. precision@k and recall@k are computed per day and then
averaged over days.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

SEED = 42
CAPACITY_KS = (0.05, 0.10, 0.20)


# ---------------------------------------------------------------- thresholds
def best_f1_threshold(y, score) -> float:
    p, r, t = precision_recall_curve(y, score)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    return float(t[int(np.nanargmax(f1))])


def evaluate(y, score, threshold: float) -> dict:
    """Threshold metrics for the stockout class plus the threshold-free AUCs."""
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "precision": precision_score(y, pred, pos_label=1, zero_division=0),
        "recall": recall_score(y, pred, pos_label=1),
        "f1": f1_score(y, pred, pos_label=1),
        "roc_auc": roc_auc_score(y, score),
        "pr_auc": average_precision_score(y, score),
        "alert_rate": pred.mean(),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


# ---------------------------------------------------------------- capacity
def tie_breaker(n: int, seed: int = SEED) -> np.ndarray:
    """Seeded random values that break score ties (the baselines score whole hours)."""
    return np.random.default_rng(seed).random(n)


def top_k_flags(score, day, k: float, seed: int = SEED) -> np.ndarray:
    """Boolean mask of the top round(k * n_day) rows by score within each day."""
    score = np.asarray(score, dtype=float)
    day = np.asarray(day)
    tie = tie_breaker(len(score), seed)
    # sort by day, then score descending, then tie value descending
    order = np.lexsort((-tie, -score, day))
    sorted_day = day[order]
    starts = np.r_[0, np.flatnonzero(sorted_day[1:] != sorted_day[:-1]) + 1]
    sizes = np.diff(np.r_[starts, len(day)])
    rank_in_day = np.arange(len(day)) - np.repeat(starts, sizes)
    limit = np.repeat(np.round(sizes * k).astype(int), sizes)
    flags = np.zeros(len(day), dtype=bool)
    flags[order] = rank_in_day < limit
    return flags


def capacity_metrics(y, score, day, ks=CAPACITY_KS, seed: int = SEED) -> dict:
    """precision@k and recall@k per day, averaged over days, for each capacity share k."""
    y = np.asarray(y, dtype=float)
    _, day_idx = np.unique(np.asarray(day), return_inverse=True)
    pos = np.bincount(day_idx, weights=y)
    out = {}
    for k in ks:
        f = top_k_flags(score, day, k, seed)
        tp = np.bincount(day_idx, weights=y * f)
        flagged = np.bincount(day_idx, weights=f.astype(float))
        # days where k rounds to zero alerts (tiny slices) or with no positives are skipped
        with np.errstate(invalid="ignore", divide="ignore"):
            prec = np.where(flagged > 0, tp / flagged, np.nan)
            rec = np.where(pos > 0, tp / pos, np.nan)
        out[f"precision_at_{int(round(k * 100))}"] = _nanmean(prec)
        out[f"recall_at_{int(round(k * 100))}"] = _nanmean(rec)
    return out


def _nanmean(x: np.ndarray) -> float:
    return float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")


def ranking_metrics(y, score, day, ks=CAPACITY_KS) -> dict:
    return {"pr_auc": average_precision_score(y, score), "roc_auc": roc_auc_score(y, score),
            **capacity_metrics(y, score, day, ks)}


# ---------------------------------------------------------------- weighted, pre-sorted metrics
class SortedScores:
    """A score vector sorted once, so weighted AP and ROC-AUC cost O(n) per bootstrap draw.

    Ties are grouped by distinct score values, which reproduces sklearn's
    average_precision_score and roc_auc_score when all weights are 1.
    """

    def __init__(self, y, score):
        score = np.asarray(score, dtype=float)
        self.order = np.argsort(-score, kind="mergesort")
        s = score[self.order]
        self.y = np.asarray(y, dtype=float)[self.order]
        # last index of each run of equal scores
        self.ends = np.r_[np.flatnonzero(s[1:] != s[:-1]), len(s) - 1]

    def metrics(self, w_sorted: np.ndarray) -> tuple[float, float]:
        wy = w_sorted * self.y
        tp = np.cumsum(wy)[self.ends]
        fp = np.cumsum(w_sorted - wy)[self.ends]
        p_tot, n_tot = tp[-1], fp[-1]
        recall = tp / p_tot
        precision = tp / np.maximum(tp + fp, 1e-12)
        ap = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
        tpr = np.r_[0.0, recall]
        fpr = np.r_[0.0, fp / n_tot]
        auc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))
        return ap, auc


def store_bootstrap_weights(store_ids, n_boot: int = 1000, seed: int = SEED):
    """Yield per-row weights: how many times each row's store is drawn in a resample of stores."""
    stores, inverse = np.unique(np.asarray(store_ids), return_inverse=True)
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(0, len(stores), len(stores)), minlength=len(stores))
        yield counts[inverse].astype(float)


def cluster_bootstrap(y, scores: dict[str, np.ndarray], store_ids, day, n_boot: int = 1000,
                      ks=CAPACITY_KS, seed: int = SEED) -> pd.DataFrame:
    """Store-cluster bootstrap of PR-AUC, ROC-AUC and precision@k / recall@k for every model.

    Stores are resampled with replacement; each row is weighted by its store's draw count.
    The daily top-k alert lists are fixed on the full test set, so precision@k measures the
    precision of that alert policy on the resampled stores. Every model sees the same draws,
    so differences between models can be read pairwise. Returns one row per draw and model.
    """
    y = np.asarray(y, dtype=float)
    day_codes, day_idx = np.unique(np.asarray(day), return_inverse=True)
    n_days = len(day_codes)
    sorted_scores = {m: SortedScores(y, s) for m, s in scores.items()}
    flags = {(m, k): top_k_flags(s, day, k, seed).astype(float) for m, s in scores.items() for k in ks}
    rows = []
    for b, w in enumerate(store_bootstrap_weights(store_ids, n_boot, seed)):
        wy = w * y
        pos_day = np.maximum(np.bincount(day_idx, weights=wy, minlength=n_days), 1e-12)
        for m, ss in sorted_scores.items():
            ap, auc = ss.metrics(w[ss.order])
            r = {"draw": b, "model": m, "pr_auc": ap, "roc_auc": auc}
            for k in ks:
                f = flags[(m, k)]
                tp_day = np.bincount(day_idx, weights=wy * f, minlength=n_days)
                fl_day = np.maximum(np.bincount(day_idx, weights=w * f, minlength=n_days), 1e-12)
                r[f"precision_at_{int(round(k * 100))}"] = float(np.mean(tp_day / fl_day))
                r[f"recall_at_{int(round(k * 100))}"] = float(np.mean(tp_day / pos_day))
            rows.append(r)
    return pd.DataFrame(rows)


def summarize_bootstrap(draws: pd.DataFrame, point: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """95% percentile intervals per model; `point` holds the full-sample value per model."""
    rows = []
    for m, g in draws.groupby("model", sort=False):
        r = {"model": m}
        for met in metrics:
            r[met] = float(point.loc[m, met])
            r[f"{met}_lo"] = float(g[met].quantile(0.025))
            r[f"{met}_hi"] = float(g[met].quantile(0.975))
        rows.append(r)
    return pd.DataFrame(rows)


def paired_differences(draws: pd.DataFrame, point: pd.DataFrame, pairs: list[tuple[str, str, str]],
                       metrics: list[str]) -> pd.DataFrame:
    """Paired bootstrap CIs for model A minus model B (same store resamples for both)."""
    rows = []
    for label, a, b in pairs:
        da = draws[draws.model == a].set_index("draw")
        db = draws[draws.model == b].set_index("draw")
        for met in metrics:
            diff = da[met] - db[met]
            lo, hi = diff.quantile(0.025), diff.quantile(0.975)
            rows.append({"comparison": label, "model_a": a, "model_b": b, "metric": met,
                         "difference": float(point.loc[a, met] - point.loc[b, met]),
                         "ci_lo": float(lo), "ci_hi": float(hi),
                         "share_draws_a_better": float((diff > 0).mean()),
                         "significant_95": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(rows)
