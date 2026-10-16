"""Execute the SQL models in sql/ in file-name order.

Each file holds one model: a single SELECT whose result is materialized as a
table named after the file without its numeric prefix (01_stg_sales.sql ->
stg_sales). The SQL stays close to BigQuery Standard SQL; DuckDB-only lines carry
the BigQuery equivalent in a comment.

Usage: python -m src.run_sql
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import duckdb

from src.data import ROOT, connect, load_raw

SQL_DIR = ROOT / "sql"
_FILE_RE = re.compile(r"^(\d+)_(\w+)\.sql$")


def sql_models(sql_dir: Path = SQL_DIR) -> list[tuple[str, Path]]:
    models = []
    for path in sorted(sql_dir.glob("*.sql")):
        m = _FILE_RE.match(path.name)
        if m:
            models.append((m.group(2), path))
    return models


def run_models(
    con: duckdb.DuckDBPyConnection,
    sql_dir: Path = SQL_DIR,
    params: dict | None = None,
    verbose: bool = True,
) -> list[str]:
    """Materialize every model in order. `params` fills {placeholders} in the SQL."""
    params = params or {}
    built = []
    for name, path in sql_models(sql_dir):
        sql = path.read_text()
        for key, value in params.items():
            sql = sql.replace("{" + key + "}", str(value))
        start = time.time()
        con.execute(f"CREATE OR REPLACE TABLE {name} AS\n{sql}")
        if verbose:
            n = con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
            print(f"  {path.name:<36} -> {name:<28} {n:>12,} rows  {time.time() - start:6.1f}s")
        built.append(name)
    return built


def build_warehouse(db_path=None, raw_dir=None) -> duckdb.DuckDBPyConnection:
    kwargs = {} if db_path is None else {"path": db_path}
    con = connect(**kwargs)
    print("Loading raw parquet")
    if raw_dir is None:
        load_raw(con)
    else:
        load_raw(con, raw_dir)
    print("Running SQL models")
    run_models(con)
    return con


if __name__ == "__main__":
    build_warehouse()
    sys.exit(0)
