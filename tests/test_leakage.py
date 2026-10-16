"""Leakage rule: every feature for day D is known at the end of D.

Method: rebuild the warehouse after scrambling every raw value dated after a cutoff
C. Features for rows with dt <= C must not change. The holiday flag is left intact
in the first test because tomorrow's holiday is an allowed calendar feature; the
second test shows it is the only feature that looks one day ahead.
"""
import numpy as np
import pandas as pd
import pytest

from src.modeling import CONFIGS, NEW_GROUPS, all_features
from tests.conftest import START, build, make_rows

FEATURES = all_features()
CUTOFF = (START + pd.Timedelta(days=24)).strftime("%Y-%m-%d")


def _scramble_after(df: pd.DataFrame, cutoff: str, keep_holiday: bool) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    out = df.copy()
    future = out.dt > cutoff
    n = int(future.sum())
    status = (rng.random((n, 24)) < 0.7).astype(int)
    sales = rng.gamma(2, 5, (n, 24))
    new_status, new_sale = iter(status.tolist()), iter(sales.tolist())
    out["hours_stock_status"] = [next(new_status) if f else v
                                 for f, v in zip(future, out.hours_stock_status)]
    out["hours_sale"] = [next(new_sale) if f else v for f, v in zip(future, out.hours_sale)]
    out.loc[future, "stock_hour6_22_cnt"] = status[:, 6:22].sum(axis=1)
    out.loc[future, "sale_amount"] = rng.gamma(2, 50, n)
    for col in ["discount", "precpt", "avg_temperature", "avg_humidity", "avg_wind_level"]:
        out.loc[future, col] = rng.uniform(0, 100, n)
    out.loc[future, "activity_flag"] = 1 - out.loc[future, "activity_flag"]
    if not keep_holiday:
        out.loc[future, "holiday_flag"] = 1 - out.loc[future, "holiday_flag"]
    return out


def _features(con) -> pd.DataFrame:
    return con.execute(
        f"SELECT store_id, product_id, dt, {', '.join(FEATURES)} FROM model_dataset "
        f"WHERE dt <= CAST('{CUTOFF}' AS DATE) ORDER BY store_id, product_id, dt"
    ).df()


@pytest.fixture(scope="module")
def base():
    df = make_rows(seed=5)
    return df, _features(build(df))


def test_features_ignore_future_data(base):
    df, before = base
    after = _features(build(_scramble_after(df, CUTOFF, keep_holiday=True)))
    assert len(before) > 0 and len(before) == len(after)
    pd.testing.assert_frame_equal(before, after)


def test_only_tomorrow_holiday_looks_ahead(base):
    df, before = base
    after = _features(build(_scramble_after(df, CUTOFF, keep_holiday=False)))
    changed = [c for c in FEATURES if not before[c].equals(after[c])]
    assert changed == ["holiday_tomorrow"]
    # and only on the cutoff day itself
    diff = before.holiday_tomorrow != after.holiday_tomorrow
    assert set(before.loc[diff, "dt"].astype(str)) == {CUTOFF}


def test_scramble_covers_every_new_feature():
    for feats in NEW_GROUPS.values():
        assert set(feats) <= set(FEATURES)
    assert set(CONFIGS["full"]) == set(FEATURES)


def test_dropped_date_index_feature():
    assert "hist_days_observed" not in FEATURES


@pytest.mark.parametrize("col", ["prod_oos_rate", "store_cat2_oos_rate"])
def test_leave_one_out_rates_ignore_the_rows_own_value(col):
    """Changing one series' own stockout on day D must not move its own product or category rate.

    The rates of the other stores (product) or other products (store x category) do move.
    """
    df = make_rows(seed=21, n_stores=3, n_products=4)
    df["second_category_id"] = df.product_id % 2
    day = (START + pd.Timedelta(days=20)).strftime("%Y-%m-%d")
    target = (df.store_id == 0) & (df.product_id == 1) & (df.dt == day)
    flipped = df.copy()
    idx = flipped.index[target][0]
    was_oos = flipped.at[idx, "stock_hour6_22_cnt"] > 2
    status = [0] * 24 if was_oos else [0] * 6 + [1] * 16 + [0] * 2
    flipped.at[idx, "hours_stock_status"] = status
    flipped.at[idx, "hours_sale"] = [0.0] * 24
    flipped.at[idx, "stock_hour6_22_cnt"] = int(sum(status[6:22]))
    q = f"SELECT store_id, product_id, {col} FROM features_entity WHERE dt = CAST('{day}' AS DATE) " \
        "ORDER BY store_id, product_id"
    before, after = build(df).execute(q).df(), build(flipped).execute(q).df()
    own = (before.store_id == 0) & (before.product_id == 1)
    assert before.loc[own, col].iloc[0] == after.loc[own, col].iloc[0]
    if col == "prod_oos_rate":
        peers = (before.product_id == 1) & (before.store_id != 0)
    else:
        peers = (before.store_id == 0) & (before.product_id % 2 == 1) & (before.product_id != 1)
    assert (before.loc[peers, col] != after.loc[peers, col]).all()


def test_label_is_not_a_feature():
    for feats in CONFIGS.values():
        assert not {"label", "next_oos_hours", "next_oos_hours_overnight", "label_dt"} & set(feats)
