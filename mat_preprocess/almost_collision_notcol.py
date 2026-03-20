#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

REQUIRED_COLUMNS = {
    "output_name",
    "status",
    "num_tracks",
    "sampled_steps",
    "original_steps",
    "original_dt",
}

def find_txt_file(txt_root: Path, output_name: str) -> Optional[Path]:
    direct = txt_root / output_name
    if direct.exists():
        return direct
    matches = list(txt_root.rglob(output_name))
    if matches:
        return matches[0]
    return None

def load_mat_params_from_yaml() -> Tuple[float, float, float]:
    config_path = Path("configs") / "mat.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Required config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML content in {config_path}")

    required = ["data_dt", "car_length", "car_width"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise KeyError(f"Missing keys in {config_path}: {missing}")

    dt = float(cfg["data_dt"])
    car_length = float(cfg["car_length"])
    car_width = float(cfg["car_width"])

    if dt <= 0:
        raise ValueError(f"'data_dt' must be positive in {config_path}, got {dt}")
    if car_length <= 0 or car_width <= 0:
        raise ValueError(
            f"'car_length'/'car_width' must be positive in {config_path}, "
            f"got car_length={car_length}, car_width={car_width}"
        )
    return dt, car_length, car_width

def compute_sampled_dt(row: pd.Series, fallback_dt: Optional[float]) -> float:
    if fallback_dt is not None:
        return float(fallback_dt)

    original_dt = float(row["original_dt"])
    original_steps = float(row["original_steps"])
    sampled_steps = float(row["sampled_steps"])

    if not math.isfinite(original_dt) or original_dt <= 0:
        raise ValueError("Invalid original_dt in manifest row.")
    if sampled_steps <= 0 or original_steps <= 0:
        raise ValueError("Invalid sampled_steps/original_steps in manifest row.")

    return original_dt * (original_steps / sampled_steps)

def load_txt(txt_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        txt_path,
        sep=r"\s+",
        header=None,
        engine="python",
    )

    if df.shape[1] >= 5:
        df = df.iloc[:, :5]
        df.columns = ["frame_id", "track_id", "x", "y", "yaw"]
    else:
        df = df.iloc[:, :4]
        df.columns = ["frame_id", "track_id", "x", "y"]
        df["yaw"] = np.nan

    if df.empty:
        raise ValueError("TXT file is empty.")

    df = df.sort_values(["frame_id", "track_id"]).reset_index(drop=True)
    return df

def _wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))

def _estimate_yaw_from_xy(x: np.ndarray, y: np.ndarray, speed_eps: float = 1.0e-6) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=float)
    if n == 1:
        return np.zeros((1,), dtype=float)

    dx = np.gradient(x)
    dy = np.gradient(y)
    speed = np.hypot(dx, dy)

    yaw = np.zeros((n,), dtype=float)
    prev = 0.0
    for i in range(n):
        if np.isfinite(dx[i]) and np.isfinite(dy[i]) and speed[i] > speed_eps:
            prev = float(np.arctan2(dy[i], dx[i]))
        yaw[i] = prev
    return _wrap_to_pi(yaw)

def _interp_angle_by_frame(frame: np.ndarray, yaw: np.ndarray) -> Optional[np.ndarray]:
    frame = np.asarray(frame, dtype=float)
    yaw = np.asarray(yaw, dtype=float)
    valid = np.isfinite(yaw)
    if valid.sum() < 2:
        return None

    f = frame[valid]
    # np.unwrap을 사용하여 각도의 불연속성을 해결한 후 보간 (수학적 오류 수정)
    yaw_unwrapped = np.unwrap(yaw[valid])
    yaw_interp = np.interp(frame, f, yaw_unwrapped)
    return _wrap_to_pi(yaw_interp)

def build_common_frames(df: pd.DataFrame) -> pd.DataFrame:
    track_ids = sorted(df["track_id"].unique().tolist())
    if len(track_ids) != 2:
        raise ValueError(f"Expected exactly 2 tracks, got {len(track_ids)}")

    a = df[df["track_id"] == track_ids[0]][["frame_id", "x", "y", "yaw"]].rename(
        columns={"x": "x1", "y": "y1", "yaw": "yaw1"}
    )
    b = df[df["track_id"] == track_ids[1]][["frame_id", "x", "y", "yaw"]].rename(
        columns={"x": "x2", "y": "y2", "yaw": "yaw2"}
    )

    merged = a.merge(b, on="frame_id", how="outer").sort_values("frame_id").reset_index(drop=True)
    # limit_direction="both" 추가하여 양 끝단 결측치 방지
    merged[["x1", "y1", "x2", "y2"]] = merged[["x1", "y1", "x2", "y2"]].interpolate(method="linear", limit_direction="both")
    merged = merged.dropna(subset=["x1", "y1", "x2", "y2"]).reset_index(drop=True)
    
    if merged.empty:
        raise ValueError("No valid common frames after interpolation.")

    frame = merged["frame_id"].to_numpy(dtype=float)

    yaw1_interp = _interp_angle_by_frame(frame, merged["yaw1"].to_numpy(dtype=float))
    yaw2_interp = _interp_angle_by_frame(frame, merged["yaw2"].to_numpy(dtype=float))

    if yaw1_interp is None:
        yaw1_interp = _estimate_yaw_from_xy(
            merged["x1"].to_numpy(dtype=float), merged["y1"].to_numpy(dtype=float)
        )
    if yaw2_interp is None:
        yaw2_interp = _estimate_yaw_from_xy(
            merged["x2"].to_numpy(dtype=float), merged["y2"].to_numpy(dtype=float)
        )

    merged["yaw1"] = _wrap_to_pi(yaw1_interp)
    merged["yaw2"] = _wrap_to_pi(yaw2_interp)
    merged["distance"] = ((merged["x1"] - merged["x2"]) ** 2 + (merged["y1"] - merged["y2"]) ** 2) ** 0.5
    return merged

def _vectorized_rect_corners(cx: np.ndarray, cy: np.ndarray, yaw: np.ndarray, length: float, width: float, rear_axle_offset: float) -> np.ndarray:
    # 후륜축 기준일 경우 정중앙(Center)으로 위치 보정
    if rear_axle_offset != 0.0:
        cx = cx + rear_axle_offset * np.cos(yaw)
        cy = cy + rear_axle_offset * np.sin(yaw)

    half_l = 0.5 * length
    half_w = 0.5 * width
    local = np.array([
        [half_l, half_w],
        [half_l, -half_w],
        [-half_l, -half_w],
        [-half_l, half_w],
    ])

    c = np.cos(yaw)
    s = np.sin(yaw)
    rot = np.empty((len(yaw), 2, 2))
    rot[:, 0, 0] = c
    rot[:, 0, 1] = -s
    rot[:, 1, 0] = s
    rot[:, 1, 1] = c

    # (N, 4, 2) 크기의 다각형 좌표 생성 (einsum 활용)
    rotated = np.einsum('kv,nwv->nkw', local, rot)
    centers = np.stack([cx, cy], axis=-1)[:, np.newaxis, :]
    return rotated + centers

def _vectorized_obb_overlap(corners_a: np.ndarray, corners_b: np.ndarray) -> np.ndarray:
    N = corners_a.shape[0]
    if N == 0:
        return np.array([], dtype=bool)

    # 각 다각형에서 서로 직교하는 2개의 변(Edge) 추출 -> 총 4개의 투영축
    edges_a = corners_a[:, 1:3, :] - corners_a[:, 0:2, :]
    edges_b = corners_b[:, 1:3, :] - corners_b[:, 0:2, :]

    axes_a = np.stack([-edges_a[:, :, 1], edges_a[:, :, 0]], axis=-1)
    axes_b = np.stack([-edges_b[:, :, 1], edges_b[:, :, 0]], axis=-1)
    axes = np.concatenate([axes_a, axes_b], axis=1) # Shape: (N, 4, 2)

    # 4개 축에 모든 꼭짓점 투영 (N, 4 꼭짓점, 4 축)
    proj_a = np.einsum('nkv,nav->nka', corners_a, axes)
    proj_b = np.einsum('nkv,nav->nka', corners_b, axes)

    min_a, max_a = proj_a.min(axis=1), proj_a.max(axis=1)
    min_b, max_b = proj_b.min(axis=1), proj_b.max(axis=1)

    # SAT: 모든 4개의 축에서 투영 구간이 겹치면 충돌로 간주
    overlap = (max_a >= min_b) & (max_b >= min_a)
    return overlap.all(axis=1)

def find_first_collision_index(merged: pd.DataFrame, car_length: float, car_width: float, rear_axle_offset: float) -> Optional[int]:
    x1, y1, yaw1 = merged["x1"].to_numpy(), merged["y1"].to_numpy(), merged["yaw1"].to_numpy()
    x2, y2, yaw2 = merged["x2"].to_numpy(), merged["y2"].to_numpy(), merged["yaw2"].to_numpy()

    c1 = _vectorized_rect_corners(x1, y1, yaw1, car_length, car_width, rear_axle_offset)
    c2 = _vectorized_rect_corners(x2, y2, yaw2, car_length, car_width, rear_axle_offset)

    overlap_mask = _vectorized_obb_overlap(c1, c2)
    indices = np.where(overlap_mask)[0]
    
    if len(indices) > 0:
        return int(merged.index[indices[0]])
    return None

def extract_window_frames(
    merged: pd.DataFrame,
    sampled_dt: float,
    before_sec: float,
    after_sec: float,
    threshold_m: float,
    require_full_window: bool,
    car_length: float,
    car_width: float,
    rear_axle_offset: float,
) -> Tuple[List[int], int, float, str, bool]:
    collision_idx = find_first_collision_index(merged, car_length, car_width, rear_axle_offset)

    if collision_idx is not None:
        raise RuntimeError("Collision detected (OBB overlap exists). This extractor keeps non-collision near-misses only.")

    near_mask = merged["distance"] <= threshold_m
    if not near_mask.any():
        raise RuntimeError("No near-collision moment under threshold.")
    anchor_idx = int(merged["distance"].idxmin())
    anchor_type = "closest_non_collision"

    anchor_frame = int(merged.loc[anchor_idx, "frame_id"])
    anchor_distance = float(merged.loc[anchor_idx, "distance"])

    frame_diffs = merged["frame_id"].diff().dropna()
    positive_diffs = frame_diffs[frame_diffs > 0]
    if positive_diffs.empty:
        frame_step = 10
    else:
        frame_step = int(round(float(positive_diffs.median())))
        if frame_step <= 0:
            frame_step = 10

    merged = merged.copy()
    merged["time_sec"] = ((merged["frame_id"] - merged["frame_id"].iloc[0]) / frame_step) * sampled_dt
    anchor_time = float(merged.loc[anchor_idx, "time_sec"])

    start_time = anchor_time - before_sec
    end_time = anchor_time + after_sec

    if require_full_window:
        if start_time < float(merged["time_sec"].iloc[0]) - 1e-9:
            raise RuntimeError("Not enough data before anchor point for full window.")
        if end_time > float(merged["time_sec"].iloc[-1]) + 1e-9:
            raise RuntimeError("Not enough data after anchor point for full window.")

    mask = (merged["time_sec"] >= start_time - 1e-9) & (merged["time_sec"] <= end_time + 1e-9)
    selected_frames = merged.loc[mask, "frame_id"].astype(int).tolist()
    if not selected_frames:
        raise RuntimeError("No frames selected for the requested time window.")

    return selected_frames, anchor_frame, anchor_distance, anchor_type, False

def remap_and_save(df: pd.DataFrame, selected_frames: List[int], save_path: Path) -> int:
    selected_frames_sorted = sorted(set(int(x) for x in selected_frames))
    frame0 = selected_frames_sorted[0]

    out_df = df[df["frame_id"].isin(selected_frames_sorted)].copy()
    if out_df.empty:
        raise RuntimeError("No rows to save.")

    out_df["frame_id"] = out_df["frame_id"].astype(int) - frame0
    out_df = out_df.sort_values(["frame_id", "track_id"]).reset_index(drop=True)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(save_path, sep="\t", header=False, index=False, float_format="%.6f")
    return len(out_df)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract near-collision (non-collision) windows from 2-track TXT files listed in manifest.csv."
    )
    parser.add_argument("--csv", default="mat_preprocess/mat_txt/all_002/manifest.csv", help="Manifest CSV path.")
    parser.add_argument("--txt-root", default="mat_preprocess/mat_txt/all_002", help="Root folder for input TXT files.")
    parser.add_argument("--out-dir", default="mat_preprocess/mat_txt/collision_notcol_002", help="Output folder.")
    parser.add_argument("--threshold-m", type=float, default=4.0, help="Near-collision threshold in meters (files with real collision are skipped).")
    parser.add_argument("--before-sec", type=float, default=1.5, help="Window seconds before anchor point.")
    parser.add_argument("--after-sec", type=float, default=0.1, help="Window seconds after anchor point.")
    parser.add_argument("--min-movement-m", type=float, default=2.0, help="Minimum movement distance per track in window.")
    # 후륜축 보정을 위한 인자 추가
    parser.add_argument("--rear-axle-offset", type=float, default=0.0, help="Offset from rear axle to true center (meters).")
    parser.add_argument(
        "--allow-truncated-window",
        action="store_true",
        help="Allow truncated windows near sequence boundaries.",
    )
    parser.add_argument(
        "--output-suffix",
        default="_notcol_window",
        help="Suffix for extracted TXT filename.",
    )
    parser.add_argument(
        "--summary-name",
        default="extraction_summary.csv",
        help="Output summary CSV filename.",
    )
    args = parser.parse_args()

    effective_sampled_dt, car_length, car_width = load_mat_params_from_yaml()

    csv_path = Path(args.csv)
    txt_root = Path(args.txt_root)
    out_dir = Path(args.out_dir)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(meta.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    candidates = meta[(meta["status"] == "OK") & (meta["num_tracks"] == 2)].copy()
    if candidates.empty:
        print("No candidates found. (status=OK, num_tracks=2)")
        return

    results: List[Dict[str, object]] = []

    for _, row in tqdm(candidates.iterrows(), total=len(candidates), desc="Extracting", unit="file"):
        output_name = str(row["output_name"])
        txt_path = find_txt_file(txt_root, output_name)
        record: Dict[str, object] = {
            "output_name": output_name,
            "txt_found": txt_path is not None,
            "status": "",
            "message": "",
            "saved_rows": 0,
            "closest_frame": "",
            "min_distance": "",
            "sampled_dt": "",
            "anchor_type": "",
            "collision_detected": False,
        }

        try:
            if txt_path is None:
                raise FileNotFoundError(f"TXT not found for {output_name}")

            sampled_dt = compute_sampled_dt(row, effective_sampled_dt)
            df = load_txt(txt_path)
            merged = build_common_frames(df)
            selected_frames, anchor_frame, min_distance, anchor_type, collision_detected = extract_window_frames(
                merged=merged,
                sampled_dt=sampled_dt,
                before_sec=args.before_sec,
                after_sec=args.after_sec,
                threshold_m=args.threshold_m,
                require_full_window=not args.allow_truncated_window,
                car_length=car_length,
                car_width=car_width,
                rear_axle_offset=args.rear_axle_offset,
            )

            window_df = df[df["frame_id"].isin(selected_frames)]
            for track_id, group in window_df.groupby("track_id"):
                dx = group["x"].max() - group["x"].min()
                dy = group["y"].max() - group["y"].min()
                move_dist = (dx ** 2 + dy ** 2) ** 0.5
                if move_dist < args.min_movement_m:
                    raise RuntimeError(
                        f"Track {track_id} movement ({move_dist:.2f}m) is below min_movement_m ({args.min_movement_m}m)."
                    )

            save_name = f"{Path(output_name).stem}{args.output_suffix}.txt"
            save_path = out_dir / save_name
            saved_rows = remap_and_save(df, selected_frames, save_path)

            record.update(
                {
                    "status": "SAVED",
                    "message": "",
                    "saved_rows": saved_rows,
                    "closest_frame": int(anchor_frame),
                    "min_distance": float(min_distance),
                    "sampled_dt": float(sampled_dt),
                    "saved_path": str(save_path),
                    "anchor_type": str(anchor_type),
                    "collision_detected": bool(collision_detected),
                }
            )
        except Exception as e:
            record.update({"status": "SKIPPED", "message": str(e)})

        results.append(record)

    summary_df = pd.DataFrame(results)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / args.summary_name
    summary_df.to_csv(summary_path, index=False)

    total = len(summary_df)
    saved = int((summary_df["status"] == "SAVED").sum())
    skipped = total - saved
    print(f"\nTotal: {total}")
    print(f"Saved: {saved}")
    print(f"Skipped: {skipped}")
    print(f"Summary CSV: {summary_path}")

if __name__ == "__main__":
    main()
