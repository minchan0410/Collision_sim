#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


def mat_preprocess_root() -> Path:
    return Path(__file__).resolve().parents[1]


def split_dataset(
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    suffix: str,
    use_ratio: float,
    seed: int | None,
    source_folder: str | None = None,
    source_dir: str | Path | None = None,
    out_root: str | Path | None = None,
) -> None:
    base_dir = mat_preprocess_root()
    if out_root is not None and str(out_root).strip():
        mat_txt_root = Path(str(out_root))
        if not mat_txt_root.is_absolute():
            mat_txt_root = mat_txt_root.resolve()
    else:
        mat_txt_root = base_dir / "mat_txt"

    total_ratio = train_ratio + val_ratio + test_ratio
    if total_ratio <= 0:
        raise ValueError("train/val/test ratio sum must be > 0.")

    if not (0 < use_ratio <= 1.0):
        raise ValueError("use_ratio must be in (0, 1.0].")

    if source_dir is not None and str(source_dir).strip():
        source_dir = Path(str(source_dir))
        if not source_dir.is_absolute():
            source_dir = source_dir.resolve()
    else:
        if not source_folder or not str(source_folder).strip():
            raise ValueError("Either source_folder or source_dir must be provided.")
        source_dir = (mat_txt_root / str(source_folder).strip()).resolve()

    if not source_dir.exists():
        raise FileNotFoundError(f"Source folder not found: {source_dir}")

    files = sorted(source_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found in source folder: {source_dir}")

    rng = random.Random(seed)
    rng.shuffle(files)

    used_total = int(len(files) * use_ratio)
    if used_total < 1:
        raise ValueError("Selected file count is 0. Increase use_ratio.")
    files = files[:used_total]

    train_frac = train_ratio / total_ratio
    val_frac = val_ratio / total_ratio
    train_end = int(used_total * train_frac)
    val_end = train_end + int(used_total * val_frac)

    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]

    dir_suffix = f"_{suffix}" if suffix else ""
    split_dirs = {
        "train": mat_txt_root / f"train{dir_suffix}",
        "val": mat_txt_root / f"val{dir_suffix}",
        "test": mat_txt_root / f"test{dir_suffix}",
    }

    for target_dir in split_dirs.values():
        if target_dir.exists():
            shutil.rmtree(target_dir)
            print(f"[INFO] Removed existing folder: {target_dir}")
        target_dir.mkdir(parents=True, exist_ok=True)

    def copy_files(file_list: list[Path], key: str) -> None:
        dst = split_dirs[key]
        for src in file_list:
            shutil.copy2(src, dst / src.name)
        print(f"[INFO] {key}: copied {len(file_list)} files -> {dst}")

    copy_files(train_files, "train")
    copy_files(val_files, "val")
    copy_files(test_files, "test")

    print("[DONE] split_data")
    print(f"  source: {source_dir}")
    print(f"  used files: {used_total} / {len(sorted(source_dir.glob('*.txt')))}")
    print(f"  train/val/test: {len(train_files)} / {len(val_files)} / {len(test_files)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split TXT dataset into train/val/test folders.")
    parser.add_argument("--train", type=float, default=8.0, help="Train ratio.")
    parser.add_argument("--val", type=float, default=1.0, help="Validation ratio.")
    parser.add_argument("--test", type=float, default=1.0, help="Test ratio.")
    parser.add_argument("--suffix", type=str, default="", help="Suffix for train/val/test folder names.")
    parser.add_argument("--use_ratio", type=float, default=1.0, help="Portion of source files to use.")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed.")
    parser.add_argument("--source", type=str, default=None, help="Source folder name under mat_preprocess/mat_txt.")
    parser.add_argument(
        "--source-dir",
        type=str,
        default=None,
        help="Absolute/relative source directory path (e.g., mat_preprocess/mat_txt/all).",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default=None,
        help="Root directory where train/val/test split folders are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_dataset(
        train_ratio=float(args.train),
        val_ratio=float(args.val),
        test_ratio=float(args.test),
        suffix=str(args.suffix).strip(),
        use_ratio=float(args.use_ratio),
        seed=int(args.seed) if args.seed is not None else None,
        source_folder=str(args.source).strip() if args.source is not None else None,
        source_dir=str(args.source_dir).strip() if args.source_dir is not None else None,
        out_root=str(args.out_root).strip() if args.out_root is not None else None,
    )


if __name__ == "__main__":
    main()
