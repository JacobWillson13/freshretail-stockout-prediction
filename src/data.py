"""Raw data loading.

The train set ships either as one file (train.parquet) or as two parts
(train_part1.parquet + train_part2.parquet, split only for GitHub size limits).
Both layouts produce the same raw_train table.
"""
from __future__ import annotations

from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
WAREHOUSE = ROOT / "data" / "warehouse.duckdb"

RAW_COLUMNS = [
    "city_id", "store_id", "management_group_id", "first_category_id",
    "second_category_id", "third_category_id", "product_id", "dt",
    "sale_amount", "hours_sale", "stock_hour6_22_cnt", "hours_stock_status",
    "discount", "holiday_flag", "activity_flag", "precpt", "avg_temperature",
    "avg_humidity", "avg_wind_level",
]


def train_files(raw_dir: Path = RAW_DIR) -> list[Path]:
    """Return the train parquet file(s), accepting either layout."""
    single = raw_dir / "train.parquet"
    if single.exists():
        return [single]
    parts = sorted(raw_dir.glob("train_part*.parquet"))
    if parts:
        return parts
    raise FileNotFoundError(
        f"No train.parquet or train_part*.parquet found in {raw_dir}"
    )


def eval_file(raw_dir: Path = RAW_DIR) -> Path:
    path = raw_dir / "eval.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return path


def _file_list_sql(paths: list[Path]) -> str:
    return "[" + ", ".join(f"'{p.as_posix()}'" for p in paths) + "]"


def connect(path: Path | str | None = WAREHOUSE) -> duckdb.DuckDBPyConnection:
    if path is None:
        return duckdb.connect()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    con.execute("SET preserve_insertion_order = false")
    return con


def load_raw(con: duckdb.DuckDBPyConnection, raw_dir: Path = RAW_DIR) -> None:
    """Create raw_train and raw_eval tables from parquet."""
    cols = ", ".join(RAW_COLUMNS)
    con.execute(
        f"CREATE OR REPLACE TABLE raw_train AS SELECT {cols} "
        f"FROM read_parquet({_file_list_sql(train_files(raw_dir))})"
    )
    con.execute(
        f"CREATE OR REPLACE TABLE raw_eval AS SELECT {cols} "
        f"FROM read_parquet({_file_list_sql([eval_file(raw_dir)])})"
    )


def register_raw_frames(con: duckdb.DuckDBPyConnection, train_df, eval_df) -> None:
    """Create raw tables from in-memory frames (used by tests)."""
    con.register("_train_df", train_df)
    con.register("_eval_df", eval_df)
    cols = ", ".join(RAW_COLUMNS)
    con.execute(f"CREATE OR REPLACE TABLE raw_train AS SELECT {cols} FROM _train_df")
    con.execute(f"CREATE OR REPLACE TABLE raw_eval AS SELECT {cols} FROM _eval_df")
    con.unregister("_train_df")
    con.unregister("_eval_df")
