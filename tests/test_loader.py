"""The loader accepts either train.parquet or train_part*.parquet."""
from src.data import connect, load_raw, train_files
from tests.conftest import make_rows, split_frames


def _write(tmp_path, single: bool):
    train, eval_ = split_frames(make_rows(seed=3))
    if single:
        train.to_parquet(tmp_path / "train.parquet")
    else:
        half = len(train) // 2
        train.iloc[:half].to_parquet(tmp_path / "train_part1.parquet")
        train.iloc[half:].to_parquet(tmp_path / "train_part2.parquet")
    eval_.to_parquet(tmp_path / "eval.parquet")
    return len(train), len(eval_)


def test_single_and_parts_give_same_rows(tmp_path):
    counts = {}
    for single in (True, False):
        d = tmp_path / ("single" if single else "parts")
        d.mkdir()
        n_train, n_eval = _write(d, single)
        assert len(train_files(d)) == (1 if single else 2)
        con = connect(None)
        load_raw(con, d)
        counts[single] = con.execute(
            "SELECT (SELECT COUNT(*) FROM raw_train), (SELECT COUNT(*) FROM raw_eval)"
        ).fetchone()
        assert counts[single] == (n_train, n_eval)
    assert counts[True] == counts[False]
