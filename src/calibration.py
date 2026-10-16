"""Probability calibration fitted on validation scores.

Two maps from a raw model score to a probability:
- Platt: a logistic regression on the logit of the score;
- isotonic: a monotone step function.

Both are stored as plain numbers so the saved model metadata (JSON) can apply them.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_platt(score, y) -> dict:
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(score).reshape(-1, 1), y)
    return {"method": "platt", "coef": float(lr.coef_[0, 0]), "intercept": float(lr.intercept_[0])}


def fit_isotonic(score, y) -> dict:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(score, y)
    return {"method": "isotonic", "x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}


def apply(cal: dict | None, score) -> np.ndarray:
    score = np.asarray(score, dtype=float)
    if cal is None or cal["method"] == "none":
        return score
    if cal["method"] == "platt":
        return 1 / (1 + np.exp(-(cal["coef"] * _logit(score) + cal["intercept"])))
    if cal["method"] == "isotonic":
        return np.interp(score, cal["x"], cal["y"])
    raise ValueError(f"unknown calibration method {cal['method']}")


FITTERS = {"platt": fit_platt, "isotonic": fit_isotonic}


def choose_method(score, y, groups, n_folds: int = 5, seed: int = 42) -> tuple[str, dict]:
    """Pick the calibration method by grouped cross-validation on validation rows only.

    Groups (stores) are split into folds; each method is fitted on the other folds and
    scored by Brier on the held-out fold. Returns the method with the lowest mean Brier
    and the per-method scores (raw scores included as "none").
    """
    score, y, groups = np.asarray(score), np.asarray(y), np.asarray(groups)
    uniq = np.unique(groups)
    fold_of = dict(zip(uniq, np.random.default_rng(seed).permutation(len(uniq)) % n_folds))
    fold = np.array([fold_of[g] for g in groups])
    briers = {"none": [], "platt": [], "isotonic": []}
    for k in range(n_folds):
        tr, te = fold != k, fold == k
        briers["none"].append(brier_score_loss(y[te], score[te]))
        for name, fit in FITTERS.items():
            briers[name].append(brier_score_loss(y[te], apply(fit(score[tr], y[tr]), score[te])))
    mean = {k: float(np.mean(v)) for k, v in briers.items()}
    best = min(("platt", "isotonic"), key=mean.get)
    return best, mean
