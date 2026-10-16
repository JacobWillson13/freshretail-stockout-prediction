"""Data checks on the raw tables (raw_train, raw_eval) and the staged data.

Every check returns a pandas DataFrame so the notebook and the PR can show it.
Usage: python -m src.checks  (writes reports/tables/data_checks_*.csv)
"""
from __future__ import annotations

import duckdb
import pandas as pd

from src.data import RAW_COLUMNS, ROOT

TABLES = ROOT / "reports" / "tables"
KEY = ["store_id", "product_id", "dt"]


def file_summary(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    rows = []
    for name in ["raw_train", "raw_eval"]:
        r = con.execute(
            f"""
            SELECT COUNT(*), COUNT(DISTINCT store_id), COUNT(DISTINCT product_id),
                   COUNT(DISTINCT (store_id, product_id)), COUNT(DISTINCT city_id),
                   COUNT(DISTINCT first_category_id), MIN(dt), MAX(dt), COUNT(DISTINCT dt)
            FROM {name}
            """
        ).fetchone()
        rows.append(dict(zip(
            ["table", "rows", "stores", "products", "store_products", "cities",
             "first_categories", "min_dt", "max_dt", "days"], [name, *r])))
    df = pd.DataFrame(rows)
    train_max = pd.Timestamp(df.loc[0, "max_dt"])
    eval_min = pd.Timestamp(df.loc[1, "min_dt"])
    df["eval_follows_train"] = [None, bool(eval_min > train_max)]
    df["gap_days"] = [None, (eval_min - train_max).days]
    return df


def series_overlap(con) -> pd.DataFrame:
    return con.execute(
        """
        SELECT
          (SELECT COUNT(*) FROM (SELECT DISTINCT store_id, product_id FROM raw_eval)) AS eval_series,
          (SELECT COUNT(*) FROM (SELECT DISTINCT store_id, product_id FROM raw_eval
             INTERSECT SELECT DISTINCT store_id, product_id FROM raw_train)) AS eval_series_in_train
        """
    ).df()


def missing_values(con) -> pd.DataFrame:
    rows = []
    for name in ["raw_train", "raw_eval"]:
        exprs = ", ".join(f"COUNT(*) - COUNT({c}) AS {c}" for c in RAW_COLUMNS)
        r = con.execute(f"SELECT {exprs} FROM {name}").df().T
        r.columns = [name]
        rows.append(r)
    out = pd.concat(rows, axis=1)
    out.index.name = "column"
    return out.reset_index()


def duplicate_keys(con) -> pd.DataFrame:
    rows = []
    for name in ["raw_train", "raw_eval"]:
        n = con.execute(
            f"SELECT COUNT(*) FROM (SELECT {', '.join(KEY)} FROM {name} "
            f"GROUP BY {', '.join(KEY)} HAVING COUNT(*) > 1)"
        ).fetchone()[0]
        rows.append({"table": name, "duplicate_keys": n})
    n = con.execute(
        "SELECT COUNT(*) FROM (SELECT store_id, product_id, dt FROM stg_sales "
        "GROUP BY 1, 2, 3 HAVING COUNT(*) > 1)"
    ).fetchone()[0]
    rows.append({"table": "train + eval combined", "duplicate_keys": n})
    return pd.DataFrame(rows)


def array_lengths(con) -> pd.DataFrame:
    rows = []
    for name in ["raw_train", "raw_eval"]:
        r = con.execute(
            f"""
            SELECT
              SUM(CASE WHEN len(hours_sale) <> 24 THEN 1 ELSE 0 END),
              SUM(CASE WHEN len(hours_stock_status) <> 24 THEN 1 ELSE 0 END),
              MIN(list_min(hours_stock_status)), MAX(list_max(hours_stock_status))
            FROM {name}
            """
        ).fetchone()
        rows.append(dict(zip(
            ["table", "hours_sale_len_not_24", "hours_stock_status_len_not_24",
             "stock_status_min", "stock_status_max"], [name, *r])))
    return pd.DataFrame(rows)


def series_contiguity(con) -> pd.DataFrame:
    """Each store-product should have one row per day with no gaps."""
    return con.execute(
        """
        WITH g AS (
          SELECT store_id, product_id, COUNT(*) AS n, MIN(dt) AS lo, MAX(dt) AS hi
          FROM stg_sales GROUP BY 1, 2
        )
        SELECT COUNT(*) AS series,
               SUM(CASE WHEN n = DATE_DIFF('day', lo, hi) + 1 THEN 1 ELSE 0 END) AS contiguous_series,
               MIN(n) AS min_days, MAX(n) AS max_days
        FROM g
        """
    ).df()


def window_match(con, table: str = "raw_train") -> pd.DataFrame:
    """Which hour range and which code reproduce stock_hour6_22_cnt?

    For candidate 0-based hour ranges [a, b], count the 1s and the 0s and report the
    share of rows where that count equals stock_hour6_22_cnt.
    """
    rows = []
    for a, b in [(6, 21), (6, 22), (5, 21), (7, 22), (5, 22), (6, 20)]:
        r = con.execute(
            f"""
            SELECT
              AVG(CASE WHEN list_sum(hours_stock_status[{a + 1}:{b + 1}]) = stock_hour6_22_cnt THEN 1.0 ELSE 0 END),
              AVG(CASE WHEN {b - a + 1} - list_sum(hours_stock_status[{a + 1}:{b + 1}]) = stock_hour6_22_cnt THEN 1.0 ELSE 0 END)
            FROM {table}
            """
        ).fetchone()
        rows.append({"hours (0-based, inclusive)": f"{a}..{b}", "n_hours": b - a + 1,
                     "share_rows_count_of_1_matches": r[0],
                     "share_rows_count_of_0_matches": r[1]})
    return pd.DataFrame(rows)


def status_vs_sales(con) -> pd.DataFrame:
    """Hourly sales by stock status: an out-of-stock hour should rarely record a sale."""
    return con.execute(
        """
        SELECT stock_status, COUNT(*) AS hours, AVG(sale) AS mean_hourly_sale,
               AVG(CASE WHEN sale > 0 THEN 1.0 ELSE 0 END) AS share_hours_with_sale
        FROM int_hourly
        WHERE in_window
        GROUP BY 1 ORDER BY 1
        """
    ).df()


def full_day_status(con) -> pd.DataFrame:
    """Days fully coded 1 in the window versus fully coded 0: daily sales differ sharply."""
    return con.execute(
        """
        SELECT CASE WHEN stock_hour6_22_cnt = 16 THEN 'all 16 window hours = 1'
                    WHEN stock_hour6_22_cnt = 0 THEN 'all 16 window hours = 0' END AS day_type,
               COUNT(*) AS days, AVG(sale_amount) AS mean_daily_sales,
               AVG(CASE WHEN sale_amount > 0 THEN 1.0 ELSE 0 END) AS share_days_with_sale
        FROM stg_sales
        WHERE stock_hour6_22_cnt IN (0, 16)
        GROUP BY 1 ORDER BY 1
        """
    ).df()


def target_prevalence(con) -> pd.DataFrame:
    """Share of labeled rows above alternative thresholds, by split."""
    return con.execute(
        """
        SELECT split, COUNT(*) AS rows,
               AVG(CASE WHEN next_oos_hours > 0 THEN 1.0 ELSE 0 END) AS gt0,
               AVG(CASE WHEN next_oos_hours > 2 THEN 1.0 ELSE 0 END) AS gt2_target,
               AVG(CASE WHEN next_oos_hours > 4 THEN 1.0 ELSE 0 END) AS gt4,
               AVG(CASE WHEN next_oos_hours > 8 THEN 1.0 ELSE 0 END) AS gt8,
               AVG(CASE WHEN next_oos_hours_overnight > 0 THEN 1.0 ELSE 0 END) AS overnight_gt0
        FROM model_dataset GROUP BY 1
        ORDER BY CASE split WHEN 'train' THEN 1 WHEN 'valid' THEN 2 ELSE 3 END
        """
    ).df()


def run_all(con) -> dict[str, pd.DataFrame]:
    out = {
        "files": file_summary(con),
        "series_overlap": series_overlap(con),
        "missing": missing_values(con),
        "duplicates": duplicate_keys(con),
        "arrays": array_lengths(con),
        "contiguity": series_contiguity(con),
        "window_match": window_match(con),
        "status_vs_sales": status_vs_sales(con),
        "full_day_status": full_day_status(con),
        "target_prevalence": target_prevalence(con),
    }
    TABLES.mkdir(parents=True, exist_ok=True)
    for name, df in out.items():
        df.to_csv(TABLES / f"data_checks_{name}.csv", index=False)
    return out


if __name__ == "__main__":
    from src.data import WAREHOUSE
    con = duckdb.connect(str(WAREHOUSE), read_only=True)
    pd.set_option("display.width", 200)
    for name, df in run_all(con).items():
        print(f"\n== {name}\n{df.to_string(index=False)}")
