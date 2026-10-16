"""The download script maps the Hugging Face file list to the layout src/data.py accepts."""
import pytest

from scripts.download_data import plan_layout


def test_single_train_file():
    files = ["README.md", "data/train.parquet", "data/eval.parquet"]
    assert plan_layout(files) == {"eval.parquet": "data/eval.parquet",
                                  "train.parquet": "data/train.parquet"}


def test_sharded_train_files():
    files = ["data/train-00001-of-00002.parquet", "data/train-00000-of-00002.parquet",
             "data/eval-00000-of-00001.parquet"]
    assert plan_layout(files) == {
        "eval.parquet": "data/eval-00000-of-00001.parquet",
        "train_part1.parquet": "data/train-00000-of-00002.parquet",
        "train_part2.parquet": "data/train-00001-of-00002.parquet",
    }


def test_unexpected_layout_raises():
    with pytest.raises(RuntimeError):
        plan_layout(["data/train.parquet"])
