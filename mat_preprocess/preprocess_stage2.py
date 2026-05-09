#!/usr/bin/env python3
"""
Stage-2 preprocessing wrapper.

Pipeline:
1) optional almost_collision.py
2) split_data.py
3) process_data_mat.py

Config is read from YAML (default: mat_preprocess/config/preprocess.yaml).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml


DEFAULT_CONFIG_PATH = Path("mat_preprocess/config/preprocess.yaml")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _quote_for_log(token: str) -> str:
    if " " in token or "\t" in token:
        return f"\"{token}\""
    return token


def _required(section: Dict[str, Any], section_name: str, key: str) -> Any:
    if key not in section:
        raise KeyError(f"Missing required key: {section_name}.{key}")
    return section[key]


def _resolve_path(path_value: Any, repo_root: Path) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "on"}:
        return True
    if text in {"false", "0", "no", "n", "off", ""}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def _load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML content in {config_path}")
    return cfg


def _build_commands(cfg: Dict[str, Any], repo_root: Path) -> Tuple[Optional[list[str]], list[str], list[str]]:
    split_cfg = cfg.get("split_data")
    proc_cfg = cfg.get("process_data_mat")
    if not isinstance(split_cfg, dict):
        raise KeyError("Missing section: split_data")
    if not isinstance(proc_cfg, dict):
        raise KeyError("Missing section: process_data_mat")

    split_train = float(_required(split_cfg, "split_data", "train"))
    split_val = float(_required(split_cfg, "split_data", "val"))
    split_test = float(_required(split_cfg, "split_data", "test"))
    split_suffix = str(_required(split_cfg, "split_data", "suffix"))
    split_use_ratio = float(_required(split_cfg, "split_data", "use_ratio"))
    split_seed = int(_required(split_cfg, "split_data", "seed"))
    if "in_root" in split_cfg:
        split_source_dir = _resolve_path(_required(split_cfg, "split_data", "in_root"), repo_root)
    elif "out_dir" in split_cfg:
        split_source_dir = _resolve_path(_required(split_cfg, "split_data", "out_dir"), repo_root)
    elif "source" in split_cfg:
        split_source = str(_required(split_cfg, "split_data", "source")).strip()
        if "/" in split_source or "\\" in split_source:
            split_source_dir = _resolve_path(split_source, repo_root)
        else:
            split_source_dir = _resolve_path(Path("mat_preprocess") / "mat_txt" / split_source, repo_root)
    else:
        raise KeyError("Missing required key: split_data.in_root (or legacy split_data.out_dir/source)")
    split_out_root = _resolve_path(split_cfg.get("out_root", "mat_preprocess/mat_txt"), repo_root)

    almost_cmd = None
    almost_cfg = cfg.get("almost_collision", {})
    if almost_cfg is None:
        almost_cfg = {}
    if not isinstance(almost_cfg, dict):
        raise TypeError("almost_collision must be a mapping.")

    if _as_bool(almost_cfg.get("enabled", False)):
        source_dir = _resolve_path(almost_cfg.get("in_root", split_source_dir), repo_root)
        almost_out_value = almost_cfg.get("out_root", almost_cfg.get("out_dir"))
        if almost_out_value:
            almost_out_dir = _resolve_path(almost_out_value, repo_root)
        else:
            almost_out_dir = source_dir.parent / f"{source_dir.name}_almost_collision"

        before_sec = float(_required(almost_cfg, "almost_collision", "before_sec"))
        after_sec = float(_required(almost_cfg, "almost_collision", "after_sec"))
        near_dist = float(almost_cfg.get("near_dist", 1.5))
        single_target_only = _as_bool(almost_cfg.get("single_target_only", False))
        if before_sec < 0 or after_sec < 0:
            raise ValueError("almost_collision.before_sec/after_sec must be >= 0.")
        if near_dist <= 0:
            raise ValueError("almost_collision.near_dist must be > 0.")

        almost_script = repo_root / "mat_preprocess" / "util" / "almost_collision.py"
        manifest_path = source_dir / "manifest.csv"
        almost_cmd = [
            sys.executable,
            str(almost_script),
            "--csv",
            str(manifest_path),
            "--txt-root",
            str(source_dir),
            "--out-dir",
            str(almost_out_dir),
            "--before-sec",
            str(before_sec),
            "--after-sec",
            str(after_sec),
            "--threshold-m",
            str(near_dist),
        ]
        if single_target_only:
            almost_cmd.append("--single-target-only")
        split_source_dir = almost_out_dir

    proc_suffix = str(proc_cfg.get("suffix", split_suffix))
    std_cfg = proc_cfg.get("standardization", {})
    if std_cfg is None:
        std_cfg = {}
    if not isinstance(std_cfg, dict):
        raise TypeError("process_data_mat.standardization must be a mapping.")

    std_mode = str(std_cfg.get("mode", "fixed"))
    std_scope = str(std_cfg.get("scope", "train"))
    std_min_points = int(std_cfg.get("min_points_per_track", 3))
    std_save_path = std_cfg.get("save_path", "")
    if std_mode not in {"fixed", "estimate"}:
        raise ValueError("process_data_mat.standardization.mode must be fixed or estimate.")
    if std_scope not in {"train", "all"}:
        raise ValueError("process_data_mat.standardization.scope must be train or all.")
    if std_min_points < 2:
        raise ValueError("process_data_mat.standardization.min_points_per_track must be >= 2.")

    proc_mat_txt_root = _resolve_path(proc_cfg.get("mat_txt_root", "mat_preprocess/mat_txt"), repo_root)
    proc_output_dir = _resolve_path(proc_cfg.get("output_dir", "mat_preprocess/processed_data"), repo_root)
    proc_num_aug = int(proc_cfg.get("num_train_augmentations", 4))
    if proc_num_aug < 0:
        raise ValueError("process_data_mat.num_train_augmentations must be >= 0.")

    process_script = repo_root / "mat_preprocess" / "util" / "process_data_mat.py"
    split_script = repo_root / "mat_preprocess" / "util" / "split_data.py"

    split_cmd = [
        sys.executable,
        str(split_script),
        "--train",
        str(split_train),
        "--val",
        str(split_val),
        "--test",
        str(split_test),
        "--suffix",
        split_suffix,
        "--use_ratio",
        str(split_use_ratio),
        "--seed",
        str(split_seed),
        "--source-dir",
        str(split_source_dir),
        "--out-root",
        str(split_out_root),
    ]

    process_cmd = [
        sys.executable,
        str(process_script),
        "--suffix",
        proc_suffix,
        "--mat-txt-root",
        str(proc_mat_txt_root),
        "--output-dir",
        str(proc_output_dir),
        "--standardization-mode",
        std_mode,
        "--standardization-scope",
        std_scope,
        "--standardization-min-points",
        str(std_min_points),
        "--num-train-augmentations",
        str(proc_num_aug),
    ]
    if std_save_path:
        process_cmd.extend(
            [
                "--standardization-save",
                str(_resolve_path(std_save_path, repo_root)),
            ]
        )

    return almost_cmd, split_cmd, process_cmd


def _run_command(command: list[str], stage_name: str) -> None:
    print(f"[RUN] {stage_name}")
    print("      " + " ".join(_quote_for_log(token) for token in command), flush=True)
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stage-2 preprocessing (split_data + process_data_mat) from YAML config.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=f"Path to YAML config. Default: {DEFAULT_CONFIG_PATH}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = _repo_root()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (repo_root / config_path).resolve()

    cfg = _load_config(config_path)
    almost_cmd, split_cmd, process_cmd = _build_commands(cfg, repo_root)

    if almost_cmd is not None:
        _run_command(almost_cmd, "1/3 almost_collision.py")
        _run_command(split_cmd, "2/3 split_data.py")
        _run_command(process_cmd, "3/3 process_data_mat.py")
    else:
        _run_command(split_cmd, "1/2 split_data.py")
        _run_command(process_cmd, "2/2 process_data_mat.py")
    print("[DONE] stage-2 preprocessing complete")


if __name__ == "__main__":
    main()
