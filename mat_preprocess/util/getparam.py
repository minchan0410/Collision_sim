#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from process_data_mat import (
    REPO_ROOT,
    estimate_standardization_from_folders,
    load_data_dt_from_mat_yaml,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate standardization stats from MAT TXT folders.")
    parser.add_argument(
        "--folders",
        nargs="+",
        default=["all"],
        help="Folder names under mat_preprocess/mat_txt used for estimation.",
    )
    parser.add_argument(
        "--min-points-per-track",
        type=int,
        default=3,
        help="Minimum points per track for estimation.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="",
        help="Optional output YAML path for estimated standardization.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dt = load_data_dt_from_mat_yaml()
    mat_txt_root = REPO_ROOT / "mat_preprocess" / "mat_txt"
    folders = [(mat_txt_root / name).resolve() for name in args.folders]

    print(f"[INFO] getparam dt={dt:.6f} sec (from configs/train.yaml)")
    print("[INFO] source folders:")
    for folder in folders:
        print(f"  - {folder}")

    standardization = estimate_standardization_from_folders(
        folders=folders,
        dt=dt,
        min_points_per_track=max(2, int(args.min_points_per_track)),
    )

    print("\n=== Estimated standardization (copy to process_data_mat if needed) ===")
    print(yaml.safe_dump(standardization, sort_keys=False, allow_unicode=False))

    if args.save:
        save_path = Path(args.save)
        if not save_path.is_absolute():
            save_path = (REPO_ROOT / save_path).resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(standardization, f, sort_keys=False, allow_unicode=False)
        print(f"[INFO] saved: {save_path}")


if __name__ == "__main__":
    main()
