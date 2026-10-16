"""Target: tomorrow has more than 2 out-of-stock hours in the 06:00 to 22:00 window."""
from tests.conftest import build, make_rows


def _day_status(window_oos: int, overnight_oos: int) -> list[int]:
    status = [0] * 24
    for h in range(6, 6 + window_oos):
        status[h] = 1
    overnight = [0, 1, 2, 3, 4, 5, 22, 23]
    for h in overnight[:overnight_oos]:
        status[h] = 1
    return status


def _with_days(cases):
    df = make_rows(seed=2, n_stores=1, n_products=1, n_days=10)
    for idx, (window_oos, overnight_oos) in cases.items():
        st = _day_status(window_oos, overnight_oos)
        df.at[idx, "hours_stock_status"] = st
        df.at[idx, "hours_sale"] = [0.0] * 24
        df.at[idx, "stock_hour6_22_cnt"] = window_oos
    return df


def test_label_threshold_and_overnight_excluded():
    # day index -> (window out-of-stock hours, overnight out-of-stock hours)
    cases = {1: (2, 0), 2: (3, 0), 3: (0, 8), 4: (2, 8), 5: (16, 0)}
    con = build(_with_days(cases))
    lab = con.execute("SELECT dt, label, next_oos_hours FROM int_target ORDER BY dt").df()
    by_day = dict(zip(range(len(lab)), lab.label))
    assert by_day[0] == 0   # tomorrow: 2 hours, not more than 2
    assert by_day[1] == 1   # tomorrow: 3 hours
    assert by_day[2] == 0   # tomorrow: 8 overnight hours only
    assert by_day[3] == 0   # tomorrow: 2 window hours plus overnight
    assert by_day[4] == 1   # tomorrow: fully out of stock
    assert lab.next_oos_hours[1] == 3


def test_last_day_has_no_label(warehouse, raw_df):
    n_series = raw_df.groupby(["store_id", "product_id"]).ngroups
    n_target = warehouse.execute("SELECT COUNT(*) FROM int_target").fetchone()[0]
    assert n_target == len(raw_df) - n_series
    last = warehouse.execute(
        "SELECT COUNT(*) FROM int_target WHERE dt = (SELECT MAX(dt) FROM stg_sales)"
    ).fetchone()[0]
    assert last == 0


def test_label_matches_next_day_count(warehouse):
    bad = warehouse.execute(
        """
        SELECT COUNT(*) FROM int_target t
        JOIN stg_sales s ON s.store_id = t.store_id AND s.product_id = t.product_id
          AND s.dt = t.label_dt
        WHERE t.label <> CASE WHEN s.stock_hour6_22_cnt > 2 THEN 1 ELSE 0 END
           OR t.label_dt <> t.dt + 1
        """
    ).fetchone()[0]
    assert bad == 0


def test_split_by_label_date(warehouse):
    r = warehouse.execute(
        """
        SELECT split, MIN(label_dt), MAX(label_dt) FROM model_dataset GROUP BY 1
        """
    ).fetchall()
    r = {split: (lo, hi) for split, lo, hi in r}
    eval_start = warehouse.execute(
        "SELECT MIN(dt) FROM stg_sales WHERE source = 'eval'"
    ).fetchone()[0]
    assert r["test"][0] == eval_start
    assert r["valid"][1] < eval_start
    assert (r["valid"][1] - r["valid"][0]).days == 13
    assert r["train"][1] < r["valid"][0]
