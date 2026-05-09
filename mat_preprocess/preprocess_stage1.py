#!/usr/bin/env python3
"""
Stage-1 preprocessing wrapper.

Pipeline:
1) check.py
2) mat2txt.py

Config is read from YAML (default: mat_preprocess/config/preprocess.yaml).
Only selected options are exposed in YAML; other options are fixed here.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml


DEFAULT_CONFIG_PATH = Path("mat_preprocess/config/preprocess.yaml")

# Fixed options for check.py
CHECK_GLOB = "*.mat"
CHECK_PROGRESS_EVERY = 200

# Fixed options for mat2txt.py
MAT2TXT_RAW_FRAME_STEP = 10
MAT2TXT_JUMP_THRESHOLD_M = 20.0
MAT2TXT_MIN_POINTS_PER_TRACK = 2
MAT2TXT_PROGRESS_EVERY = 50


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


def _load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML content in {config_path}")
    return cfg


def _build_commands(cfg: Dict[str, Any], repo_root: Path) -> Tuple[list[str], list[str], Path]:
    check_cfg = cfg.get("check")
    mat2txt_cfg = cfg.get("mat2txt")
    if not isinstance(check_cfg, dict):
        raise KeyError("Missing section: check")
    if not isinstance(mat2txt_cfg, dict):
        raise KeyError("Missing section: mat2txt")

    check_in_root = _resolve_path(_required(check_cfg, "check", "in_root"), repo_root)
    check_out_dir = _resolve_path(_required(check_cfg, "check", "out_dir"), repo_root)
    check_workers = int(_required(check_cfg, "check", "workers"))
    check_target_var = str(check_cfg.get("target_var", "data"))
    check_required_fields = check_cfg.get("required_fields")
    if check_workers < 1:
        raise ValueError(f"check.workers must be >= 1, got {check_workers}")
    if check_required_fields is not None:
        if not isinstance(check_required_fields, list):
            raise TypeError("check.required_fields must be a list of field names.")
        check_required_fields = [str(x).strip() for x in check_required_fields if str(x).strip()]
        if not check_required_fields:
            raise ValueError("check.required_fields is empty.")

    mat2txt_in_root = _resolve_path(_required(mat2txt_cfg, "mat2txt", "in_root"), repo_root)
    mat2txt_pass_list = _resolve_path(_required(mat2txt_cfg, "mat2txt", "pass_list"), repo_root)
    mat2txt_out_dir = _resolve_path(_required(mat2txt_cfg, "mat2txt", "out_dir"), repo_root)
    sampling_time_sec = float(_required(mat2txt_cfg, "mat2txt", "sampling_time_sec"))
    if sampling_time_sec <= 0:
        raise ValueError(f"mat2txt.sampling_time_sec must be positive, got {sampling_time_sec}")

    check_script = repo_root / "mat_preprocess" / "util" / "check.py"
    mat2txt_script = repo_root / "mat_preprocess" / "util" / "mat2txt.py"

    check_cmd = [
        sys.executable,
        str(check_script),
        "--in_root",
        str(check_in_root),
        "--out_dir",
        str(check_out_dir),
        "--workers",
        str(check_workers),
        "--glob",
        CHECK_GLOB,
        "--progress_every",
        str(CHECK_PROGRESS_EVERY),
        "--target_var",
        check_target_var,
    ]
    if check_required_fields is not None:
        check_cmd.extend(["--required-fields", *check_required_fields])

    mat2txt_cmd = [
        sys.executable,
        str(mat2txt_script),
        "--in-root",
        str(mat2txt_in_root),
        "--pass-list",
        str(mat2txt_pass_list),
        "--out-dir",
        str(mat2txt_out_dir),
        "--target-dt",
        str(sampling_time_sec),
        "--raw-frame-step",
        str(MAT2TXT_RAW_FRAME_STEP),
        "--jump-threshold-m",
        str(MAT2TXT_JUMP_THRESHOLD_M),
        "--min-points-per-track",
        str(MAT2TXT_MIN_POINTS_PER_TRACK),
        "--progress-every",
        str(MAT2TXT_PROGRESS_EVERY),
    ]

    return check_cmd, mat2txt_cmd, mat2txt_pass_list


def _run_command(command: list[str], stage_name: str) -> None:
    print(f"[RUN] {stage_name}")
    print("      " + " ".join(_quote_for_log(token) for token in command), flush=True)
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stage-1 preprocessing (check + mat2txt) from YAML config.")
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
    check_cmd, mat2txt_cmd, pass_list_path = _build_commands(cfg, repo_root)

    _run_command(check_cmd, "1/2 check.py")

    if not pass_list_path.exists():
        raise FileNotFoundError(
            "pass_list file was not found after check stage: "
            f"{pass_list_path}"
        )

    _run_command(mat2txt_cmd, "2/2 mat2txt.py")
    print("[DONE] stage-1 preprocessing complete")


if __name__ == "__main__":
    main()
