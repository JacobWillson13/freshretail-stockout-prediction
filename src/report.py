"""Test-set reporting: bootstrap CIs, capacity curves, calibration, slices and SHAP.

Reads what src/modeling.py wrote (test and validation predictions, results table, the
saved model) and never refits or re-selects a model, so the test set is still scored once.

Usage: python -m src.report
"""
from __future__ import annotations

import json

import duckdb
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, brier_score_loss  # noqa: E402

from src import calibration  # noqa: E402
from src import modeling as m  # noqa: E402
from src.data import WAREHOUSE  # noqa: E402
from src.metrics import (  # noqa: E402
    CAPACITY_KS,
    capacity_metrics,
    cluster_bootstrap,
    paired_differences,
    summarize_bootstrap,
)
from src.plots import NEUTRAL, SERIES, TEXT_2, save, style  # noqa: E402

TABLES = m.TABLES
N_BOOT = 1000
CI_METRICS = ["pr_auc", "roc_auc"] + [f"{p}_at_{int(round(k * 100))}" for k in CAPACITY_KS
                                      for p in ("precision", "recall")]
SHAP_SAMPLE = 20_000
METRIC_LABELS = {"pr_auc": "PR-AUC", "roc_auc": "ROC-AUC", "precision_at_10": "Precision@10%",
                 "recall_at_10": "Recall@10%"}
BASELINE_MARKERS = ["s", "^", "D", "v"]
ROLE_COLORS = {"actionable": SERIES[1], "response trigger": SERIES[0], "descriptive": NEUTRAL}


def load_outputs():
    preds = pd.read_parquet(m.PRED_PATH)
    preds_va = pd.read_parquet(m.VALID_PRED_PATH)
    results = pd.read_csv(TABLES / "model_results_all_splits.csv")
    meta = json.loads((m.MODELS_DIR / "best_model.json").read_text())
    return preds, preds_va, results, meta


def entries(preds: pd.DataFrame) -> list[str]:
    return [c.removeprefix("score_") for c in preds.columns if c.startswith("score_")]


def entry_label(e: str) -> str:
    model, config = m.describe_entry(e)
    return model if e.startswith("baseline") else f"{model} / {config}"


def best_baseline() -> str:
    bt = pd.read_csv(m.BACKTEST_SUMMARY)
    return bt[bt.entry.str.startswith("baseline")].sort_values("pr_auc_mean", ascending=False).entry.iloc[0]


# ---------------------------------------------------------------- confidence intervals
def bootstrap_tables(preds, results, meta, n_boot: int = N_BOOT):
    names = entries(preds)
    scores = {e: preds[f"score_{e}"].values for e in names}
    draws = cluster_bootstrap(preds.label.values, scores, preds.store_id.values, preds.dt.values,
                              n_boot=n_boot)
    point = results[results.split == "test"].set_index("entry")[CI_METRICS]
    ci = summarize_bootstrap(draws, point.loc[names], CI_METRICS)
    ci.insert(1, "label", ci.model.map(entry_label))
    ci = ci.rename(columns={"model": "entry"})
    best, base = meta["entry"], best_baseline()
    pairs = [(f"best ({best}) vs best baseline ({base})", best, base),
             (f"best ({best}) vs rule baseline (baseline_today)", best, "baseline_today"),
             ("full vs combined (LightGBM)", "lightgbm_full", "lightgbm_combined"),
             ("combined vs daily (LightGBM)", "lightgbm_combined", "lightgbm_daily"),
             ("full vs combined (logistic regression)", "logreg_full", "logreg_combined")]
    pairs = [p for i, p in enumerate(pairs) if p[1] != p[2] and p not in pairs[:i]]
    diffs = paired_differences(draws, point, pairs, CI_METRICS)
    return ci, diffs


def headline_table(ci: pd.DataFrame, meta: dict) -> pd.DataFrame:
    best, base = meta["entry"], best_baseline()
    rows = [("best baseline", base), ("rule baseline", "baseline_today"),
            ("daily", "lightgbm_daily"), ("combined", "lightgbm_combined"), ("best model", best)]
    seen, out = set(), []
    for role, e in rows:
        if e in seen:
            continue
        seen.add(e)
        r = ci.set_index("entry").loc[e]
        out.append({"role": role, "entry": e, "label": r.label, **{c: r[c] for c in ci.columns
                                                                    if c not in ("entry", "label")}})
    return pd.DataFrame(out)


def fmt_ci(r, met: str, pct: bool = False) -> str:
    if pct:
        return f"{r[met]:.1%} ({r[met + '_lo']:.1%} to {r[met + '_hi']:.1%})"
    return f"{r[met]:.3f} ({r[met + '_lo']:.3f} to {r[met + '_hi']:.3f})"


def headline_markdown(head: pd.DataFrame) -> str:
    lines = ["| Model | Precision@10% | Recall@10% | PR-AUC | ROC-AUC |", "|---|---|---|---|---|"]
    for _, r in head.iterrows():
        if r.entry.startswith("baseline"):
            name = f"{r.role.capitalize()}: {r.label.removeprefix('baseline: ')}"
        elif r.role == "best model":
            name = f"**Best model: {r.label.replace(' / ', ', ')} features**"
        else:
            name = f"{r.label.replace(' / ', ', ')} features"
        lines.append(f"| {name} | {fmt_ci(r, 'precision_at_10')} | "
                     f"{fmt_ci(r, 'recall_at_10')} | {fmt_ci(r, 'pr_auc')} | {fmt_ci(r, 'roc_auc')} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- capacity curve
def capacity_curve(preds, entries_: list[str], shares=None) -> pd.DataFrame:
    shares = shares if shares is not None else np.round(np.arange(0.01, 0.51, 0.01), 2)
    rows = []
    for e in entries_:
        out = capacity_metrics(preds.label.values, preds[f"score_{e}"].values, preds.dt.values, ks=shares)
        for k in shares:
            rows.append({"entry": e, "share": k, "precision": out[f"precision_at_{int(round(k * 100))}"],
                         "recall": out[f"recall_at_{int(round(k * 100))}"]})
    return pd.DataFrame(rows)


def plot_hero(curve: pd.DataFrame, best: str, base: str, ci: pd.DataFrame, prevalence: float):
    fig, ax = plt.subplots(figsize=(8, 4.4))
    base_label = m.describe_entry(base)[0].removeprefix("baseline: ")
    for e, color, label in [(best, SERIES[0], f"best model: {' / '.join(m.describe_entry(best))}"),
                            (base, NEUTRAL, f"best baseline: {base_label}")]:
        c = curve[curve.entry == e]
        ax.plot(c.share * 100, c.precision, color=color, label=label, lw=2.2 if e == best else 1.8,
                ls="-" if e == best else "--")
        r = ci.set_index("entry").loc[e]
        ax.errorbar([10], [r.precision_at_10], yerr=[[r.precision_at_10 - r.precision_at_10_lo],
                                                     [r.precision_at_10_hi - r.precision_at_10]],
                    color=color, marker="o", ms=6, capsize=4, zorder=5)
    ax.axvline(10, color=TEXT_2, lw=0.8, ls=":")
    rb, rs = ci.set_index("entry").loc[best], ci.set_index("entry").loc[base]
    ax.annotate(f"operating point: top 10% of series per day (5,000 alerts)\n"
                f"precision {rb.precision_at_10:.3f} vs {rs.precision_at_10:.3f}",
                xy=(10, rb.precision_at_10), xytext=(16, rb.precision_at_10 + 0.04), fontsize=8.5,
                color=TEXT_2, arrowprops=dict(arrowstyle="-", color=TEXT_2, lw=0.6))
    ax.axhline(prevalence, color="#c9c8c2", lw=1)
    ax.text(49, prevalence + 0.008, f"prevalence {prevalence:.1%}", ha="right", fontsize=8, color=TEXT_2)
    ax.set_xlabel("Daily review capacity (% of the 50,000 series flagged each day)")
    ax.set_ylabel("Precision of the alerts (mean over test days)")
    ax.set_title("Next-day stockout alerts: precision at each review capacity (test week)")
    ax.legend(loc="lower left", fontsize=8.5)
    ax.set_xlim(0, 50)
    save(fig, "hero_precision_at_k")
    plt.close(fig)


def plot_ci_forest(ci: pd.DataFrame, metric: str, fname: str, title: str):
    c = ci[~ci.entry.str.contains("minus")].sort_values(metric)
    fig, ax = plt.subplots(figsize=(8, 0.34 * len(c) + 1.2))
    colors = [NEUTRAL if e.startswith("baseline") else SERIES[0] if e.startswith("lightgbm") else SERIES[1]
              for e in c.entry]
    ax.errorbar(c[metric], range(len(c)), xerr=[c[metric] - c[f"{metric}_lo"], c[f"{metric}_hi"] - c[metric]],
                fmt="none", ecolor="#9a9994", capsize=3)
    ax.scatter(c[metric], range(len(c)), color=colors, zorder=5)
    ax.set_yticks(range(len(c)), c.label, fontsize=8)
    ax.set_xlabel(f"{METRIC_LABELS.get(metric, metric)} on test, 95% store-cluster bootstrap CI")
    ax.set_title(title)
    handles = [plt.Line2D([], [], marker="o", ls="none", color=col, label=lab)
               for lab, col in (("LightGBM", SERIES[0]), ("logistic regression", SERIES[1]),
                                ("baseline", NEUTRAL))]
    ax.legend(handles=handles, loc="lower right", fontsize=8.5, frameon=True, facecolor="white",
              edgecolor="#e6e5df")
    save(fig, fname)
    plt.close(fig)


# ---------------------------------------------------------------- calibration
def calibration_report(preds, preds_va, meta):
    e = meta["entry"]
    s_va, s_te = preds_va[f"score_{e}"].values, preds[f"score_{e}"].values
    y_va, y_te = preds_va.label.values, preds.label.values
    maps = {"raw": None, "platt": calibration.fit_platt(s_va, y_va),
            "isotonic": calibration.fit_isotonic(s_va, y_va)}
    rows, probs = [], {}
    for name, cal in maps.items():
        p = calibration.apply(cal, s_te)
        probs[name] = p
        bins = pd.qcut(p, 10, duplicates="drop")
        g = pd.DataFrame({"p": p, "y": y_te}).groupby(bins, observed=True)
        ece = float((g.size() / len(p) * (g.p.mean() - g.y.mean()).abs()).sum())
        rows.append({"calibration": name, "fitted_on": "validation" if cal else "",
                     "brier_test": brier_score_loss(y_te, p), "ece_10_bins_test": ece,
                     "mean_predicted_test": float(p.mean()), "observed_rate_test": float(y_te.mean()),
                     "pr_auc_test": average_precision_score(y_te, p),
                     "selected": name == meta["calibration"]["method"]})
    table = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(5.6, 5))
    ax.plot([0, 1], [0, 1], color="#c9c8c2", lw=1)
    for (name, p), color in zip(probs.items(), [NEUTRAL, SERIES[1], SERIES[0]]):
        bins = pd.qcut(p, 10, duplicates="drop")
        g = pd.DataFrame({"p": p, "y": y_te}).groupby(bins, observed=True).mean()
        b = table.set_index("calibration").loc[name, "brier_test"]
        ax.plot(g.p, g.y, marker="o", ms=4, color=color, label=f"{name} (Brier {b:.4f})")
    ax.set_xlabel("Mean predicted probability (test, 10 quantile bins)")
    ax.set_ylabel("Observed stockout rate")
    ax.set_title(f"Reliability on test, {' / '.join(m.describe_entry(e))}")
    ax.legend(fontsize=8)
    save(fig, "calibration_01_reliability")
    plt.close(fig)
    return table


# ---------------------------------------------------------------- slices
def slice_keys(con, preds) -> pd.DataFrame:
    """Slice attributes known before the test week (store volume and series history use train dates)."""
    keys = con.execute(
        """
        WITH pre AS (
          SELECT * FROM int_daily WHERE source = 'train'
        ),
        store_vol AS (
          SELECT store_id, AVG(sale_amount) AS store_mean_daily_sales FROM pre GROUP BY store_id
        ),
        series_hist AS (
          SELECT store_id, product_id, ANY_VALUE(first_category_id) AS first_category_id,
                 AVG(CASE WHEN oos_hours > 2 THEN 1.0 ELSE 0.0 END) AS series_train_stockout_share
          FROM pre GROUP BY store_id, product_id
        )
        SELECT h.*, v.store_mean_daily_sales
        FROM series_hist AS h JOIN store_vol AS v USING (store_id)
        """
    ).df()
    store = keys.drop_duplicates("store_id")[["store_id", "store_mean_daily_sales"]].copy()
    store["store_volume_decile"] = pd.qcut(store.store_mean_daily_sales.rank(method="first"), 10,
                                           labels=range(1, 11)).astype(int)
    keys = keys.merge(store[["store_id", "store_volume_decile"]], on="store_id")
    keys["series_history_decile"] = pd.qcut(keys.series_train_stockout_share, 10, duplicates="drop")
    keys["series_history_decile"] = keys.series_history_decile.astype(str)
    return preds.merge(keys, on=["store_id", "product_id"], how="left", validate="many_to_one")


def slice_table(df, entry: str, base: str) -> pd.DataFrame:
    df = df.assign(day=df.dt.astype(str))
    rows = []
    for dim in ["first_category_id", "store_volume_decile", "series_history_decile", "day"]:
        for value, g in df.groupby(dim, observed=True):
            r = {"slice": dim, "value": value, "n": len(g), "prevalence": g.label.mean()}
            for name, e in (("model", entry), ("baseline", base)):
                s = g[f"score_{e}"].values
                has_both = 0 < g.label.mean() < 1
                r[f"pr_auc_{name}"] = average_precision_score(g.label, s) if has_both else np.nan
                r[f"precision_at_10_{name}"] = capacity_metrics(g.label.values, s, g.dt.values,
                                                                ks=(0.10,))["precision_at_10"]
            r["pr_auc_lift"] = r["pr_auc_model"] - r["pr_auc_baseline"]
            r["pr_auc_over_prevalence"] = r["pr_auc_model"] - r["prevalence"]
            rows.append(r)
    return pd.DataFrame(rows)


def plot_slices(sl: pd.DataFrame):
    dims = ["first_category_id", "store_volume_decile", "series_history_decile", "day"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))
    for ax, dim in zip(axes.ravel(), dims):
        d = sl[sl.slice == dim]
        x = np.arange(len(d))
        ax.bar(x - 0.2, d.pr_auc_model, width=0.4, color=SERIES[0], label="best model")
        ax.bar(x + 0.2, d.pr_auc_baseline, width=0.4, color=NEUTRAL, label="best baseline")
        ax.plot(x, d.prevalence, color=SERIES[1], marker="_", ms=14, lw=0, label="prevalence")
        ax.set_xticks(x, d.value.astype(str), rotation=40, ha="right", fontsize=7)
        name = "prediction day D (label day D+1)" if dim == "day" else dim.replace("_", " ")
        ax.set_title(f"Test PR-AUC by {name}")
    axes[0, 0].legend(fontsize=8)
    save(fig, "slices_01_pr_auc")
    plt.close(fig)


# ---------------------------------------------------------------- SHAP
def shap_report(con, meta, n: int = SHAP_SAMPLE):
    import shap

    model = joblib.load(m.MODELS_DIR / "best_model.joblib")
    feats = meta["features"]
    if meta["kind"] != "lightgbm":
        return None
    X = m.load_dataset(con, where="WHERE split = 'test'")[feats]
    X = X.sample(n=min(n, len(X)), random_state=m.SEED)
    explainer = shap.TreeExplainer(model.booster_)
    sv = explainer.shap_values(X)
    sv = sv[1] if isinstance(sv, list) else sv
    imp = pd.DataFrame({"feature": feats, "mean_abs_shap": np.abs(sv).mean(axis=0)})
    imp["role"] = imp.feature.map(m.feature_role)
    imp["group"] = imp.feature.map(feature_group)
    imp = imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    plt.figure()
    shap.summary_plot(sv, X, max_display=20, show=False, plot_size=(9, 7.5))
    plt.title("SHAP summary, test sample of 20,000 rows (importance is not causation)", fontsize=10)
    plt.savefig(m.ROOT / "reports" / "figures" / "shap_01_summary.png", dpi=130, bbox_inches="tight")
    plt.close("all")
    top3 = imp.feature.head(3).tolist()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, f in zip(axes, top3):
        shap.dependence_plot(f, sv, X, interaction_index=None, ax=ax, show=False, alpha=0.3, dot_size=4)
        ax.set_title(f"{f} ({m.feature_role(f)})", fontsize=10)
    fig.suptitle("SHAP dependence for the top 3 features (associations, not causes)", fontweight="bold")
    save(fig, "shap_02_dependence_top3")
    plt.close("all")
    return imp


def feature_group(f: str) -> str:
    for g, feats in [("daily", m.DAILY), ("weekly", m.WEEKLY), ("category", m.CATEGORY),
                     ("calendar", m.CALENDAR), *m.NEW_GROUPS.items()]:
        if f in feats:
            return g
    return "other"


# ---------------------------------------------------------------- backtest and ablation figures
def plot_backtest():
    folds = pd.read_csv(TABLES / "backtest_folds.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4), sharex=True)
    order = folds.groupby("entry").pr_auc.mean().sort_values(ascending=False).index
    weeks = folds.drop_duplicates("fold").sort_values("fold")
    for ax, met in zip(axes, ["pr_auc", "precision_at_10"]):
        n_model = n_base = 0
        for e in order:
            d = folds[folds.entry == e].sort_values("fold")
            if e.startswith("baseline"):
                style_kw = dict(color=NEUTRAL, ls="--", lw=1.1, ms=5,
                                marker=BASELINE_MARKERS[n_base % len(BASELINE_MARKERS)])
                n_base += 1
            else:
                style_kw = dict(color=SERIES[n_model % len(SERIES)], ls="-", lw=1.6, ms=4, marker="o")
                n_model += 1
            ax.plot(d.fold, d[met], label=entry_label(e), **style_kw)
        ax.set_xticks(weeks.fold, [f"fold {f}\n{s[5:]}" for f, s in zip(weeks.fold, weeks.val_start)],
                      fontsize=8.5)
        ax.set_xlabel("validation week start (oldest first)")
        ax.set_title(f"Backtest {METRIC_LABELS.get(met, met)} by fold")
    axes[1].legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(fig, "backtest_01_folds")
    plt.close(fig)


def ablation_table(results: pd.DataFrame, ci: pd.DataFrame) -> pd.DataFrame:
    names = ["lightgbm_full"] + [m.entry_name("lightgbm", a) for a in m.ABLATIONS] + ["lightgbm_combined"]
    t = results[(results.entry.isin(names))].pivot_table(
        index="entry", columns="split", values=["pr_auc", "precision_at_10"]).reindex(names)
    t.columns = [f"{a}_{b}" for a, b in t.columns]
    full = t.loc["lightgbm_full"]
    for c in list(t.columns):
        t[f"{c}_change_vs_full"] = t[c] - full[c]
    t = t.join(ci.set_index("entry")[["pr_auc_lo", "pr_auc_hi"]].add_prefix("test_"))
    return t.reset_index()


AGGREGATION_ENTRIES = ["baseline_today"] + [f"{k}_{c}" for c in ("daily", "weekly", "category", "combined")
                                              for k in ("logreg", "lightgbm")]


def aggregation_study(results: pd.DataFrame) -> pd.DataFrame:
    """The aggregation grid (rule baseline plus 2 models x 4 aggregation configs) on test."""
    t = m.comparison_table(results, "test")
    return t[t.entry.isin(AGGREGATION_ENTRIES)].reset_index(drop=True)


# ---------------------------------------------------------------- main
def main():
    style()
    preds, preds_va, results, meta = load_outputs()
    con = duckdb.connect(str(WAREHOUSE), read_only=True)
    best, base = meta["entry"], best_baseline()
    print(f"best model {best}, best baseline {base}", flush=True)

    ci, diffs = bootstrap_tables(preds, results, meta)
    ci.to_csv(TABLES / "test_metrics_ci.csv", index=False)
    diffs.to_csv(TABLES / "test_paired_differences.csv", index=False)
    head = headline_table(ci, meta)
    head.to_csv(TABLES / "headline_test.csv", index=False)
    (TABLES / "headline_test.md").write_text(headline_markdown(head))
    print(headline_markdown(head))
    print(diffs[diffs.metric.isin(["pr_auc", "precision_at_10"])].round(4).to_string(index=False))

    curve = capacity_curve(preds, [best, base, "baseline_today", "lightgbm_daily", "lightgbm_combined"])
    curve.to_csv(TABLES / "capacity_curve_test.csv", index=False)
    plot_hero(curve, best, base, ci, float(preds.label.mean()))
    plot_ci_forest(ci, "precision_at_10", "ci_01_precision_at_10", "Precision@10% on test with 95% CIs")
    plot_ci_forest(ci, "pr_auc", "ci_02_pr_auc", "PR-AUC on test with 95% CIs")

    ablation_table(results, ci).to_csv(TABLES / "ablation_full.csv", index=False)
    aggregation_study(results).to_csv(TABLES / "aggregation_study_test.csv", index=False)
    if (TABLES / "backtest_folds.csv").exists():
        plot_backtest()

    cal = calibration_report(preds, preds_va, meta)
    cal.to_csv(TABLES / "calibration_test.csv", index=False)
    print(cal.round(4).to_string(index=False))

    sl = slice_table(slice_keys(con, preds), best, base)
    sl.to_csv(TABLES / "slices_test.csv", index=False)
    plot_slices(sl)

    imp = shap_report(con, meta)
    if imp is not None:
        imp.to_csv(TABLES / "shap_importance_test.csv", index=False)
        print(imp.head(15).round(4).to_string(index=False))
    con.close()


if __name__ == "__main__":
    main()
