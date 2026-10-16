"""Download FreshRetailNet-50K (train and eval parquet) from Hugging Face into data/raw/.

Source: the public dataset Dingdong-Inc/FreshRetailNet-50K. The script lists the
dataset files, picks the train and eval parquet files and writes the layout that
src/data.py accepts: train.parquet (one train file) or train_part1.parquet,
train_part2.parquet, ... (several train shards), plus eval.parquet.

Usage: python scripts/download_data.py [--out data/raw] [--force]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ID = "Dingdong-Inc/FreshRetailNet-50K"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "raw"


def plan_layout(repo_files: list[str]) -> dict[str, str]:
    """Map local file names to dataset file paths for the train and eval parquet files."""
    parquet = sorted(f for f in repo_files if f.endswith(".parquet"))
    name = lambda f: Path(f).name.lower()  # noqa: E731
    train = [f for f in parquet if name(f).startswith("train")]
    evals = [f for f in parquet if name(f).startswith(("eval", "test"))]
    if not train or len(evals) != 1:
        raise RuntimeError(
            f"Unexpected file layout in {REPO_ID}: train={train}, eval={evals}. "
            "Download the files by hand into data/raw/ (see README)."
        )
    layout = {"eval.parquet": evals[0]}
    if len(train) == 1:
        layout["train.parquet"] = train[0]
    else:
        for i, f in enumerate(train, start=1):
            layout[f"train_part{i}.parquet"] = f
    return layout


def download(out: Path = DEFAULT_OUT, force: bool = False) -> list[Path]:
    from huggingface_hub import hf_hub_download, list_repo_files

    out.mkdir(parents=True, exist_ok=True)
    existing = list(out.glob("train*.parquet")) + list(out.glob("eval.parquet"))
    if existing and not force:
        print(f"{out} already holds {sorted(p.name for p in existing)}; use --force to replace")
        return existing
    layout = plan_layout(list_repo_files(REPO_ID, repo_type="dataset"))
    written = []
    for local_name, repo_path in layout.items():
        print(f"  {repo_path} -> {out / local_name}")
        cached = hf_hub_download(REPO_ID, repo_path, repo_type="dataset")
        shutil.copyfile(cached, out / local_name)
        written.append(out / local_name)
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--force", action="store_true", help="replace files already in --out")
    args = ap.parse_args(argv)
    download(args.out, args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
