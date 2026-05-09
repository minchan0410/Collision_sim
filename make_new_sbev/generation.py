#!/usr/bin/env python3
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.patches import Polygon


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import generation_runtime as runtime  # noqa: E402
from mat_preprocess.util.mat2txt import build_yaw_series as build_mat_yaw_series  # noqa: E402
from mat_run import (  # noqa: E402
    COLOR_SETS,
    _build_axis_limits,
    _compute_yaw_from_positions,
    _draw_vehicle_box,
    _extract_dataset_epoch_from_name,
    _install_tensorboard_noop,
    _lighten_color,
    _load_scene_csv,
    _load_yaml_config,
    _resolve_path_yaws,
    _resolve_checkpoint,
)


DEFAULT_CONFIG = "configs/run.yaml"
DEFAULT_INPUT_DIR = "SBEV_Catalog_FNPrecrash/diffusion_input_csv"
DEFAULT_OUTPUT_ROOT = "SBEV_Catalog_FNPrecrash/diffusion_generation_result"
COLLISION_MODE_ORDER = (11, 12, 13, 21, 23, 31, 33, 41, 43, 51, 52, 53)


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def _build_runtime(cfg: dict):
    runtime = runtime.EasyDict()
    runtime.ckpt_dir = str(cfg.get("viz_ckpt_dir_pkl", cfg.get("viz_ckpt_dir", ""))).strip()
    runtime.ckpt_name = str(cfg.get("viz_ckpt_name", "")).strip()
    runtime.seed = int(cfg.get("viz_seed", cfg.get("seed", 123)))
    runtime.dataset = str(cfg.get("dataset", "")).strip()
    if runtime.ckpt_dir == "":
        raise ValueError("Set viz_ckpt_dir_pkl or viz_ckpt_dir in run.yaml.")
    if runtime.ckpt_name == "":
        raise ValueError("Set viz_ckpt_name in run.yaml.")
    return runtime


def _csv_sort_key(path: Path):
    info = _parse_input_info(path)
    return (info["scenario"], int(info["data_index"]) if str(info["data_index"]).isdigit() else 10**9, path.name)


def _collect_input_csvs(input_dir: Path, input_csv: Optional[Path]) -> List[Path]:
    if input_csv is not None:
        if not input_csv.exists():
            raise FileNotFoundError(f"Input CSV not found: {input_csv}")
        return [input_csv.resolve()]
    return sorted(input_dir.rglob("*_trajectory.csv"), key=_csv_sort_key)


def _parse_input_info(csv_path: Path) -> Dict[str, str]:
    scenario = csv_path.parent.name
    stem = csv_path.stem
    clean_stem = stem[:-len("_trajectory")] if stem.endswith("_trajectory") else stem
    clean_stem = clean_stem[:-len("_generation")] if clean_stem.endswith("_generation") else clean_stem
    match = re.match(r"^Image_(?P<label>\d+)_(?P<scenario>.+)_(?P<data_index>\d+)_(?P<image_frame>\d+)$", clean_stem)
    if match:
        return {
            "scenario": match.group("scenario"),
            "data_index": match.group("data_index"),
            "image_frame": match.group("image_frame"),
            "stem": clean_stem,
        }

    marker = f"_{scenario}_"
    data_index = "unknown"
    image_frame = "unknown"
    if clean_stem.startswith("Image_") and marker in clean_stem:
        tail = clean_stem[clean_stem.index(marker) + len(marker):]
        parts = tail.split("_")
        if len(parts) >= 2:
            data_index = parts[0]
            image_frame = parts[1]
    return {
        "scenario": scenario,
        "data_index": data_index,
        "image_frame": image_frame,
        "stem": clean_stem,
    }


def _make_output_path(input_csv: Path, output_root: Path) -> Path:
    info = _parse_input_info(input_csv)
    return output_root / info["scenario"] / info["stem"] / f"{info['stem']}_generation.csv"


def _yaw_from_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=float)
    if n == 1:
        return np.zeros((1,), dtype=float)
    dx = np.gradient(x)
    dy = np.gradient(y)
    yaw = np.arctan2(dy, dx)
    for idx in range(n):
        if not np.isfinite(yaw[idx]):
            yaw[idx] = yaw[idx - 1] if idx > 0 else 0.0
    return runtime._wrap_to_pi(yaw)


def _read_trajectory_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"agent", "sample", "frame", "x", "y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    df = df[df["sample"].astype(int) == 0].copy()
    if df.empty:
        raise ValueError(f"{path} has no sample=0 trajectory.")
    df["agent"] = df["agent"].astype(int)
    df["frame"] = df["frame"].astype(int)
    df["x"] = df["x"].astype(float)
    df["y"] = df["y"].astype(float)
    if "yaw" in df.columns:
        df["yaw"] = pd.to_numeric(df["yaw"], errors="coerce").astype(float)
    return df.sort_values(["agent", "frame"]).reset_index(drop=True)


def _read_generation_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"agent", "sample", "frame", "x", "y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    df = df.copy()
    df["agent"] = df["agent"].astype(int)
    df["sample"] = df["sample"].astype(int)
    df["frame"] = df["frame"].astype(int)
    df["x"] = df["x"].astype(float)
    df["y"] = df["y"].astype(float)
    return df.sort_values(["agent", "sample", "frame"]).reset_index(drop=True)


def _infer_input_end_frame(df: pd.DataFrame, tol: float = 1e-5) -> int:
    history_ends: List[int] = []
    for agent in sorted(df["agent"].unique()):
        gt = df[(df["agent"] == agent) & (df["sample"] == 0)].set_index("frame")[["x", "y"]]
        if gt.empty:
            continue
        sample_ids = sorted(df[(df["agent"] == agent) & (df["sample"] > 0)]["sample"].unique())
        for sample_id in sample_ids:
            pred = df[(df["agent"] == agent) & (df["sample"] == sample_id)].set_index("frame")[["x", "y"]]
            common = sorted(set(gt.index).intersection(set(pred.index)))
            if len(common) <= 1:
                continue
            for idx, frame in enumerate(common):
                dist = float(np.linalg.norm(gt.loc[frame].to_numpy(dtype=float) - pred.loc[frame].to_numpy(dtype=float)))
                if dist > tol:
                    history_ends.append(int(common[max(0, idx - 1)]))
                    break
    if history_ends:
        return int(min(history_ends))
    return int(df["frame"].max())


def _xy_until(df: pd.DataFrame, agent: int, sample: int, max_frame: int, min_frame: Optional[int] = None) -> np.ndarray:
    sub = df[(df["agent"] == agent) & (df["sample"] == sample) & (df["frame"] <= max_frame)]
    if min_frame is not None:
        sub = sub[sub["frame"] >= min_frame]
    return sub.sort_values("frame")[["x", "y"]].to_numpy(dtype=float)


def _last_xy_until(df: pd.DataFrame, agent: int, sample: int, max_frame: int, min_frame: Optional[int] = None) -> Optional[np.ndarray]:
    arr = _xy_until(df, agent, sample, max_frame=max_frame, min_frame=min_frame)
    if arr.shape[0] == 0:
        return None
    return arr[-1]


def _build_compare_axis_limits(df: pd.DataFrame) -> Tuple[float, float, float, float]:
    pts = df[["x", "y"]].to_numpy(dtype=float)
    if pts.shape[0] == 0:
        return -10.0, 10.0, -10.0, 10.0
    x_min, y_min = np.min(pts, axis=0)
    x_max, y_max = np.max(pts, axis=0)
    center_x = 0.5 * (x_min + x_max)
    center_y = 0.5 * (y_min + y_max)
    max_range = max(15.0, float(x_max - x_min), float(y_max - y_min))
    half = 0.60 * max_range
    return center_x - half, center_x + half, center_y - half, center_y + half


def _draw_path(ax, xy: np.ndarray, *, color, linewidth: float, markersize: float, alpha: float, label: Optional[str] = None, zorder: float = 2.0) -> None:
    if xy.shape[0] == 0:
        return
    if xy.shape[0] == 1:
        ax.scatter([xy[0, 0]], [xy[0, 1]], s=max(10.0, markersize * 8.0), color=color, alpha=alpha, label=label, zorder=zorder)
        return
    ax.plot(
        xy[:, 0],
        xy[:, 1],
        "-o",
        color=color,
        linewidth=linewidth,
        markersize=markersize,
        alpha=alpha,
        label=label,
        zorder=zorder,
    )


def _draw_vehicle_at(ax, xy_path: np.ndarray, *, car_width: float, car_length: float, color, alpha: float, zorder: float) -> None:
    if xy_path.shape[0] == 0:
        return
    yaws = _compute_yaw_from_positions(xy_path)
    yaw = float(yaws[-1]) if yaws.shape[0] > 0 else 0.0
    _draw_vehicle_box(
        ax=ax,
        center_xy=xy_path[-1],
        yaw=yaw,
        car_length=car_length,
        car_width=car_width,
        face_color=color,
        face_alpha=alpha,
        zorder=zorder,
    )


def _draw_sbev_anchor_vehicle_box(
    ax,
    anchor_xy: np.ndarray,
    *,
    yaw: float,
    role: str,
    car_length: float,
    car_width: float,
    ego_cg_to_front: float,
    face_color,
    face_alpha: float,
    zorder: float,
) -> None:
    world = _sbev_anchor_vehicle_corners(
        anchor_xy,
        yaw=yaw,
        role=role,
        car_length=car_length,
        car_width=car_width,
        ego_cg_to_front=ego_cg_to_front,
    )
    patch = Polygon(
        world,
        closed=True,
        facecolor=face_color,
        edgecolor=(0.15, 0.15, 0.15, min(1.0, float(face_alpha) + 0.15)),
        linewidth=0.8,
        alpha=float(face_alpha),
        zorder=zorder,
    )
    ax.add_patch(patch)


def _sbev_anchor_vehicle_corners(
    anchor_xy: np.ndarray,
    *,
    yaw: float,
    role: str,
    car_length: float,
    car_width: float,
    ego_cg_to_front: float,
) -> np.ndarray:
    half_w = 0.5 * float(car_width)
    role_lower = str(role).strip().lower()
    if "ego" in role_lower:
        x0, x1 = -float(car_length), 0.0
    elif "target" in role_lower:
        x0, x1 = 0.0, float(car_length)
    else:
        half_l = 0.5 * float(car_length)
        x0, x1 = -half_l, half_l

    local = np.asarray(
        [
            [x1, half_w],
            [x1, -half_w],
            [x0, -half_w],
            [x0, half_w],
        ],
        dtype=np.float32,
    )
    c = np.cos(float(yaw))
    s = np.sin(float(yaw))
    rot = np.asarray([[c, -s], [s, c]], dtype=np.float32)
    anchor = np.asarray(anchor_xy, dtype=np.float32).reshape(2)
    if "ego" in role_lower:
        anchor = anchor + np.asarray([c, s], dtype=np.float32) * float(ego_cg_to_front)
    return local @ rot.T + anchor


def _ego_local_to_world(local_xy: np.ndarray, *, ego_xy: np.ndarray, yaw: float, ego_cg_to_front: float) -> np.ndarray:
    local_xy = np.asarray(local_xy, dtype=np.float32)
    c = np.cos(float(yaw))
    s = np.sin(float(yaw))
    rot = np.asarray([[c, -s], [s, c]], dtype=np.float32)
    ego_front = np.asarray(ego_xy, dtype=np.float32).reshape(2) + np.asarray([c, s], dtype=np.float32) * float(ego_cg_to_front)
    return local_xy @ rot.T + ego_front


def _world_to_ego_local(world_xy: np.ndarray, *, ego_xy: np.ndarray, yaw: float, ego_cg_to_front: float) -> np.ndarray:
    world_xy = np.asarray(world_xy, dtype=np.float32)
    c = np.cos(float(yaw))
    s = np.sin(float(yaw))
    rot = np.asarray([[c, -s], [s, c]], dtype=np.float32)
    ego_front = np.asarray(ego_xy, dtype=np.float32).reshape(2) + np.asarray([c, s], dtype=np.float32) * float(ego_cg_to_front)
    return (world_xy - ego_front) @ rot


def _draw_ego_collision_mode_grid(
    ax,
    ego_xy: np.ndarray,
    *,
    yaw: float,
    car_length: float,
    car_width: float,
    ego_cg_to_front: float,
    zorder: float,
) -> None:
    half_w = 0.5 * float(car_width)
    length = float(car_length)
    width = float(car_width)

    for row_idx in range(6):
        x_val = -length * float(row_idx) / 5.0
        pts = np.asarray([[x_val, -half_w], [x_val, half_w]], dtype=np.float32)
        world = _ego_local_to_world(pts, ego_xy=ego_xy, yaw=yaw, ego_cg_to_front=ego_cg_to_front)
        ax.plot(world[:, 0], world[:, 1], color="white", linewidth=0.45, alpha=0.80, zorder=zorder)

    for col_idx in range(4):
        y_val = half_w - width * float(col_idx) / 3.0
        pts = np.asarray([[-length, y_val], [0.0, y_val]], dtype=np.float32)
        world = _ego_local_to_world(pts, ego_xy=ego_xy, yaw=yaw, ego_cg_to_front=ego_cg_to_front)
        ax.plot(world[:, 0], world[:, 1], color="white", linewidth=0.45, alpha=0.80, zorder=zorder)

    for mode in COLLISION_MODE_ORDER:
        row = int(mode // 10)
        col = int(mode % 10)
        label_local = np.asarray(
            [[-length * (float(row) - 0.5) / 5.0, half_w - width * (float(col) - 0.5) / 3.0]],
            dtype=np.float32,
        )
        label_world = _ego_local_to_world(label_local, ego_xy=ego_xy, yaw=yaw, ego_cg_to_front=ego_cg_to_front)[0]
        ax.text(
            float(label_world[0]),
            float(label_world[1]),
            str(mode),
            color="yellow",
            fontsize=5.5,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=zorder + 0.1,
        )


def _append_sbev_vehicle_extents(
    out_points: List[np.ndarray],
    xy: Optional[np.ndarray],
    yaw_arr: Optional[np.ndarray],
    *,
    role: str,
    car_length: float,
    car_width: float,
    ego_cg_to_front: float,
) -> None:
    if xy is None:
        return
    xy = np.asarray(xy, dtype=np.float32)
    if xy.ndim != 2 or xy.shape[0] == 0 or xy.shape[1] < 2:
        return
    xy = xy[:, :2]
    out_points.append(xy)
    yaws = _resolve_path_yaws(xy, yaw_arr)
    for idx in range(xy.shape[0]):
        out_points.append(
            _sbev_anchor_vehicle_corners(
                xy[idx],
                yaw=float(yaws[idx]) if yaws.shape[0] > idx else 0.0,
                role=role,
                car_length=car_length,
                car_width=car_width,
                ego_cg_to_front=ego_cg_to_front,
            )
        )


def _build_sbev_video_axis_limits(
    agent_records: Sequence[dict],
    *,
    car_length: float,
    car_width: float,
    ego_cg_to_front: float,
) -> Tuple[float, float, float, float]:
    points: List[np.ndarray] = []
    for rec in agent_records:
        role = str(rec.get("role", ""))
        _append_sbev_vehicle_extents(
            points,
            rec.get("gt"),
            rec.get("gt_yaw"),
            role=role,
            car_length=car_length,
            car_width=car_width,
            ego_cg_to_front=ego_cg_to_front,
        )
        pred_yaws = rec.get("pred_yaw", {})
        for sample_idx, pred_arr in rec.get("pred", {}).items():
            _append_sbev_vehicle_extents(
                points,
                pred_arr,
                pred_yaws.get(sample_idx),
                role=role,
                car_length=car_length,
                car_width=car_width,
                ego_cg_to_front=ego_cg_to_front,
            )

    if not points:
        return _build_axis_limits(agent_records, "all")

    pts = np.vstack(points)
    x_min, y_min = np.min(pts, axis=0)
    x_max, y_max = np.max(pts, axis=0)
    x_range = max(1.0, float(x_max - x_min))
    y_range = max(1.0, float(y_max - y_min))
    x_margin = max(2.0, 0.08 * x_range, 0.75 * float(car_width))
    y_margin = max(3.0, 0.12 * y_range, 0.75 * float(car_length))
    x0 = float(x_min - x_margin)
    x1 = float(x_max + x_margin)
    y0 = float(y_min - y_margin)
    y1 = float(y_max + y_margin)

    min_side_ratio = 0.85
    x_span = max(1.0e-6, x1 - x0)
    y_span = max(1.0e-6, y1 - y0)
    if x_span < y_span * min_side_ratio:
        center = 0.5 * (x0 + x1)
        half = 0.5 * y_span * min_side_ratio
        x0 = center - half
        x1 = center + half
    elif y_span < x_span * min_side_ratio:
        center = 0.5 * (y0 + y1)
        half = 0.5 * x_span * min_side_ratio
        y0 = center - half
        y1 = center + half

    return (x0, x1, y0, y1)


def _convex_polygons_overlap(poly_a: np.ndarray, poly_b: np.ndarray, eps: float = 1.0e-7) -> bool:
    poly_a = np.asarray(poly_a, dtype=np.float32)
    poly_b = np.asarray(poly_b, dtype=np.float32)
    if poly_a.ndim != 2 or poly_b.ndim != 2 or poly_a.shape[0] < 3 or poly_b.shape[0] < 3:
        return False
    for poly in (poly_a, poly_b):
        edges = np.roll(poly, -1, axis=0) - poly
        for edge in edges:
            axis = np.asarray([-edge[1], edge[0]], dtype=np.float32)
            norm = float(np.linalg.norm(axis))
            if norm <= eps:
                continue
            axis = axis / norm
            proj_a = poly_a @ axis
            proj_b = poly_b @ axis
            if float(np.max(proj_a)) < float(np.min(proj_b)) - eps:
                return False
            if float(np.max(proj_b)) < float(np.min(proj_a)) - eps:
                return False
    return True


def _clip_polygon_axis(poly: np.ndarray, axis: int, bound: float, *, keep_greater: bool, eps: float = 1.0e-8) -> np.ndarray:
    poly = np.asarray(poly, dtype=np.float32)
    if poly.ndim != 2 or poly.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)

    def _inside(point: np.ndarray) -> bool:
        value = float(point[axis])
        return value >= float(bound) - eps if keep_greater else value <= float(bound) + eps

    out: List[np.ndarray] = []
    prev = poly[-1]
    prev_inside = _inside(prev)
    for curr in poly:
        curr_inside = _inside(curr)
        if curr_inside != prev_inside:
            denom = float(curr[axis] - prev[axis])
            if abs(denom) > eps:
                alpha = (float(bound) - float(prev[axis])) / denom
                alpha = float(np.clip(alpha, 0.0, 1.0))
                out.append(prev + alpha * (curr - prev))
        if curr_inside:
            out.append(curr)
        prev = curr
        prev_inside = curr_inside
    if not out:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(out, dtype=np.float32)


def _clip_polygon_to_rect(poly: np.ndarray, *, x_min: float, x_max: float, y_min: float, y_max: float) -> np.ndarray:
    out = _clip_polygon_axis(poly, 0, x_min, keep_greater=True)
    out = _clip_polygon_axis(out, 0, x_max, keep_greater=False)
    out = _clip_polygon_axis(out, 1, y_min, keep_greater=True)
    out = _clip_polygon_axis(out, 1, y_max, keep_greater=False)
    return out


def _polygon_area(poly: np.ndarray) -> float:
    poly = np.asarray(poly, dtype=np.float32)
    if poly.ndim != 2 or poly.shape[0] < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _ego_collision_mode_from_target_box(
    *,
    ego_xy: np.ndarray,
    ego_yaw: float,
    target_box_world: np.ndarray,
    car_length: float,
    car_width: float,
    ego_cg_to_front: float,
) -> int:
    target_local = _world_to_ego_local(
        target_box_world,
        ego_xy=ego_xy,
        yaw=ego_yaw,
        ego_cg_to_front=ego_cg_to_front,
    )
    half_w = 0.5 * float(car_width)
    length = float(car_length)
    width = float(car_width)

    best_mode = 0
    best_area = 0.0
    best_tie_distance = float("inf")
    target_center = np.mean(target_local, axis=0)
    for mode in COLLISION_MODE_ORDER:
        row = int(mode // 10)
        col = int(mode % 10)
        x_max = -length * float(row - 1) / 5.0
        x_min = -length * float(row) / 5.0
        y_max = half_w - width * float(col - 1) / 3.0
        y_min = half_w - width * float(col) / 3.0
        clipped = _clip_polygon_to_rect(target_local, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
        area = _polygon_area(clipped)
        cell_center = np.asarray([0.5 * (x_min + x_max), 0.5 * (y_min + y_max)], dtype=np.float32)
        tie_distance = float(np.linalg.norm(cell_center - target_center))
        if area > best_area + 1.0e-7 or (abs(area - best_area) <= 1.0e-7 and area > 1.0e-7 and tie_distance < best_tie_distance):
            best_area = area
            best_mode = int(mode)
            best_tie_distance = tie_distance
    return int(best_mode) if best_area > 1.0e-7 else 0


def _path_frame_map(arr: np.ndarray, yaw_arr: Optional[np.ndarray], frame_arr: Optional[np.ndarray]) -> Dict[int, Tuple[np.ndarray, float]]:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return {}
    if frame_arr is None:
        frames = np.arange(arr.shape[0], dtype=np.int32)
    else:
        frames = np.asarray(frame_arr, dtype=np.int32).reshape(-1)
    limit = min(arr.shape[0], frames.shape[0])
    yaws = _resolve_path_yaws(arr[:limit], yaw_arr)
    return {
        int(frames[idx]): (arr[idx, :2], float(yaws[idx]) if yaws.shape[0] > idx else 0.0)
        for idx in range(limit)
    }


def _first_collision_frame_for_sample(
    agent_records: Sequence[dict],
    sample_idx: int,
    *,
    car_length: float,
    car_width: float,
    ego_cg_to_front: float,
) -> Optional[Tuple[int, int]]:
    ego_rec = next((rec for rec in agent_records if int(rec.get("agent_index", -1)) == 0), None)
    if ego_rec is None:
        return None

    ego_arr = ego_rec.get("pred", {}).get(sample_idx, ego_rec.get("gt"))
    ego_yaw = ego_rec.get("pred_yaw", {}).get(sample_idx, ego_rec.get("gt_yaw"))
    ego_frame = ego_rec.get("pred_frame", {}).get(sample_idx, ego_rec.get("gt_frame"))
    ego_map = _path_frame_map(ego_arr, ego_yaw, ego_frame)
    if not ego_map:
        return None

    first_collision: Optional[int] = None
    first_mode = 0
    for target_rec in agent_records:
        if int(target_rec.get("agent_index", -1)) == 0:
            continue
        target_arr = target_rec.get("gt") if int(sample_idx) == 0 else target_rec.get("pred", {}).get(sample_idx)
        if target_arr is None:
            continue
        target_yaw = target_rec.get("gt_yaw") if int(sample_idx) == 0 else target_rec.get("pred_yaw", {}).get(sample_idx)
        target_frame = target_rec.get("gt_frame") if int(sample_idx) == 0 else target_rec.get("pred_frame", {}).get(sample_idx)
        target_map = _path_frame_map(target_arr, target_yaw, target_frame)
        if not target_map:
            continue

        target_frames = np.asarray(target_frame, dtype=np.int32).reshape(-1) if target_frame is not None else np.arange(len(target_arr), dtype=np.int32)
        hist_len = max(1, int(target_rec.get("history_len", 1)))
        future_start_idx = min(hist_len, max(0, int(target_frames.shape[0]) - 1))
        future_start_frame = int(target_frames[future_start_idx])

        for frame in sorted(set(ego_map.keys()).intersection(target_map.keys())):
            if frame < future_start_frame:
                continue
            ego_xy, ego_heading = ego_map[frame]
            target_xy, target_heading = target_map[frame]
            ego_box = _sbev_anchor_vehicle_corners(
                ego_xy,
                yaw=ego_heading,
                role="Ego",
                car_length=car_length,
                car_width=car_width,
                ego_cg_to_front=ego_cg_to_front,
            )
            target_box = _sbev_anchor_vehicle_corners(
                target_xy,
                yaw=target_heading,
                role=str(target_rec.get("role", "Target")),
                car_length=car_length,
                car_width=car_width,
                ego_cg_to_front=ego_cg_to_front,
            )
            if _convex_polygons_overlap(ego_box, target_box):
                mode = _ego_collision_mode_from_target_box(
                    ego_xy=ego_xy,
                    ego_yaw=ego_heading,
                    target_box_world=target_box,
                    car_length=car_length,
                    car_width=car_width,
                    ego_cg_to_front=ego_cg_to_front,
                )
                if mode == 0:
                    continue
                if first_collision is None or frame < first_collision:
                    first_collision = int(frame)
                    first_mode = int(mode)
                break

    if first_collision is None:
        return None
    return int(first_collision), int(first_mode)


def _truncate_generated_csv_at_render_collisions(path: Path, cfg: dict) -> Dict[int, Tuple[int, int]]:
    df = pd.read_csv(path)
    if df.empty or "sample" not in df.columns:
        return {}
    if "cm" in df.columns:
        df = df.drop(columns=["cm"])

    agent_records = _load_scene_csv(path)
    car_width = float(cfg.get("car_width", 1.825))
    car_length = float(cfg.get("car_length", 4.650))
    ego_cg_to_front = float(cfg.get("ego_cg_to_front_bumper", 0.5 * car_length))
    sample_ids = sorted(int(sample_id) for sample_id in df["sample"].unique())

    collision_info: Dict[int, Tuple[int, int]] = {}
    for sample_idx in sample_ids:
        collision = _first_collision_frame_for_sample(
            agent_records,
            sample_idx,
            car_length=car_length,
            car_width=car_width,
            ego_cg_to_front=ego_cg_to_front,
        )
        if collision is not None:
            collision_frame, collision_mode = collision
            collision_info[int(sample_idx)] = (int(collision_frame), int(collision_mode))

    keep = np.ones((len(df),), dtype=bool)
    cm_values = np.zeros((len(df),), dtype=np.int32)
    for sample_idx, (collision_frame, collision_mode) in collision_info.items():
        sample_mask = df["sample"].astype(int).to_numpy() == int(sample_idx)
        future_mask = df["frame"].astype(int).to_numpy() > int(collision_frame)
        if int(sample_idx) > 0:
            keep &= ~(sample_mask & future_mask)
        cm_values[sample_mask] = int(collision_mode)

    out = df.loc[keep].copy()
    out.insert(2, "cm", cm_values[keep].astype(int))
    for int_col in ("agent", "sample", "frame"):
        if int_col in out.columns:
            out[int_col] = out[int_col].astype(int)
    out["cm"] = out["cm"].astype(int)
    ordered = ["agent", "sample", "cm", "frame", "x", "y"]
    if "yaw" in out.columns:
        ordered.append("yaw")
    ordered += [col for col in out.columns if col not in ordered]
    out = out[ordered].sort_values(["agent", "sample", "frame"]).reset_index(drop=True)
    out.to_csv(path, index=False)
    return collision_info


def _render_compare_video(csv_path: Path, cfg: dict, fmt_override: str = "") -> Path:
    agent_records = _load_scene_csv(csv_path)
    if not agent_records:
        raise ValueError(f"{csv_path} has no renderable agent records.")

    fps = int(cfg.get("viz_video_fps", 10))
    dpi = int(cfg.get("viz_video_dpi", 140))
    end_hold_sec = float(cfg.get("viz_video_end_hold_sec", 1.5))
    fmt = str(fmt_override or cfg.get("viz_video_format", "auto")).strip().lower()
    bbox_alpha = float(cfg.get("viz_video_bbox_alpha", 0.22))
    car_width = float(cfg.get("car_width", 1.825))
    car_length = float(cfg.get("car_length", 4.650))
    ego_cg_to_front = float(cfg.get("ego_cg_to_front_bumper", 0.5 * car_length))

    max_frames = 1
    for rec in agent_records:
        max_frames = max(max_frames, int(rec["gt"].shape[0]))
        for pred_arr in rec["pred"].values():
            max_frames = max(max_frames, int(pred_arr.shape[0]))

    hold_frames = max(0, int(round(end_hold_sec * max(1, fps))))
    total_frames = max_frames + hold_frames
    x0, x1, y0, y1 = _build_sbev_video_axis_limits(
        agent_records,
        car_length=car_length,
        car_width=car_width,
        ego_cg_to_front=ego_cg_to_front,
    )
    pred_box_alpha = float(np.clip(max(0.45, bbox_alpha * 2.0), 0.40, 0.70))

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)

    scene_frame0 = min(
        int(np.min(rec["gt_frame"])) if rec.get("gt_frame") is not None else 0
        for rec in agent_records
        if rec.get("gt") is not None and rec["gt"].shape[0] > 0
    )
    info = _parse_input_info(csv_path)
    try:
        raw_error_frame = int(info["image_frame"])
    except Exception:
        raw_error_frame = None

    data_dt = float(cfg.get("data_dt", 0.02))
    source_dt = float(cfg.get("sbev_source_dt", 0.01))
    source_dt = source_dt if source_dt > 0 else 0.01
    raw_frames_per_scene_step = max(1, int(round(data_dt / source_dt)))
    history_steps = max(1, int(round(float(cfg.get("history_sec", 0.6)) / data_dt)))
    error_steps = max(1, int(round(float(cfg.get("default_error_sec", 0.3)) / data_dt)))
    error_scene_frame = scene_frame0 + history_steps + 1 + error_steps

    def _display_raw_frame(traj_frame_idx: int) -> Optional[int]:
        if raw_error_frame is None:
            return None
        current_scene_frame = scene_frame0 + int(traj_frame_idx)
        return int(raw_error_frame + (current_scene_frame - error_scene_frame) * raw_frames_per_scene_step)

    def _style_axis(ax, title: str, traj_frame_idx: int) -> None:
        raw_frame = _display_raw_frame(traj_frame_idx)
        if raw_frame is None:
            frame_text = f"frame {min(traj_frame_idx + 1, max_frames)}/{max_frames}"
        else:
            frame_text = f"SBEV frame {raw_frame} | {min(traj_frame_idx + 1, max_frames)}/{max_frames}"
        ax.set_title(f"{title} | {frame_text}")
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.3)

    def _zorders(rec: dict) -> Tuple[float, float, float]:
        role_lower = str(rec.get("role", "")).strip().lower()
        if "target" in role_lower:
            return 60.0, 40.0, 20.0
        if "ego" in role_lower:
            return 50.0, 30.0, 10.0
        return 45.0, 25.0, 15.0

    def _plot_segment(ax, arr: np.ndarray, start: int, end: int, *, color, linewidth: float, markersize: float, alpha: float, zorder: float) -> None:
        if arr.shape[0] == 0:
            return
        start = max(0, int(start))
        end = min(int(end), int(arr.shape[0]) - 1)
        if end < start:
            return
        seg = arr[start: end + 1]
        if seg.shape[0] == 1:
            ax.scatter([seg[0, 0]], [seg[0, 1]], color=color, s=max(10.0, markersize * 8.0), alpha=alpha, zorder=zorder)
            return
        ax.plot(seg[:, 0], seg[:, 1], "-o", color=color, linewidth=linewidth, markersize=markersize, alpha=alpha, zorder=zorder)

    def _draw_box_at_index(ax, arr: np.ndarray, idx: int, *, role: str, color, alpha: float, zorder: float, yaw_arr=None) -> None:
        if arr.shape[0] == 0:
            return
        idx = min(max(0, int(idx)), int(arr.shape[0]) - 1)
        yaws = _resolve_path_yaws(arr, yaw_arr)
        _draw_sbev_anchor_vehicle_box(
            ax=ax,
            anchor_xy=arr[idx],
            yaw=float(yaws[idx]) if yaws.shape[0] > idx else 0.0,
            role=role,
            car_length=car_length,
            car_width=car_width,
            ego_cg_to_front=ego_cg_to_front,
            face_color=color,
            face_alpha=alpha,
            zorder=zorder,
        )

    def _draw_gt_agent(ax, rec: dict, traj_frame_idx: int, *, show_label: bool) -> None:
        agent_index = int(rec["agent_index"])
        colors = COLOR_SETS[agent_index % len(COLOR_SETS)]
        traj_z, _pred_box_z, gt_box_z = _zorders(rec)
        gt = rec["gt"]
        hist_len = int(rec["history_len"])
        hist_end = min(traj_frame_idx, max(0, hist_len - 1), int(gt.shape[0]) - 1)
        _plot_segment(
            ax,
            gt,
            0,
            hist_end,
            color=colors["hist"],
            linewidth=3.2,
            markersize=3,
            alpha=1.0,
            zorder=traj_z + 0.20,
        )
        gt_start = max(0, hist_len - 1)
        if traj_frame_idx >= gt_start:
            _plot_segment(
                ax,
                gt,
                gt_start,
                traj_frame_idx,
                color=colors["gt"],
                linewidth=3.2,
                markersize=3,
                alpha=1.0,
                zorder=traj_z + 0.05,
            )
        curr_idx = min(traj_frame_idx, int(gt.shape[0]) - 1)
        _draw_box_at_index(ax, gt, curr_idx, role=rec.get("role", ""), yaw_arr=rec.get("gt_yaw"), color=colors["pred"], alpha=1.0, zorder=gt_box_z)
        if show_label:
            yaws = _resolve_path_yaws(gt, rec.get("gt_yaw"))
            curr_yaw = float(yaws[curr_idx]) if yaws.shape[0] > curr_idx else 0.0
            role_text = str(rec.get("role", ""))
            if "ego" in role_text.strip().lower():
                _draw_ego_collision_mode_grid(
                    ax,
                    gt[curr_idx],
                    yaw=curr_yaw,
                    car_length=car_length,
                    car_width=car_width,
                    ego_cg_to_front=ego_cg_to_front,
                    zorder=traj_z + 12.0,
                )
                return
            label_corners = _sbev_anchor_vehicle_corners(
                gt[curr_idx],
                yaw=curr_yaw,
                role=role_text,
                car_length=car_length,
                car_width=car_width,
                ego_cg_to_front=ego_cg_to_front,
            )
            label_xy = np.mean(label_corners, axis=0)
            ax.text(
                float(label_xy[0]),
                float(label_xy[1]),
                role_text,
                color="white",
                fontsize=9,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=traj_z + 10.0,
            )

    def _draw_diffusion_agent(ax, rec: dict, traj_frame_idx: int) -> None:
        agent_index = int(rec["agent_index"])
        colors = COLOR_SETS[agent_index % len(COLOR_SETS)]
        traj_z, pred_box_z, gt_box_z = _zorders(rec)
        gt = rec["gt"]
        hist_len = int(rec["history_len"])
        pred_items = sorted(rec["pred"].items(), key=lambda item: item[0])

        if len(pred_items) == 0:
            _draw_gt_agent(ax, rec, traj_frame_idx, show_label=False)
            return

        hist_end = min(traj_frame_idx, max(0, hist_len - 1), int(gt.shape[0]) - 1)
        _plot_segment(
            ax,
            gt,
            0,
            hist_end,
            color=colors["hist"],
            linewidth=3.2,
            markersize=3,
            alpha=1.0,
            zorder=traj_z + 0.20,
        )
        if traj_frame_idx <= max(0, hist_len - 1):
            _draw_box_at_index(ax, gt, hist_end, role=rec.get("role", ""), yaw_arr=rec.get("gt_yaw"), color=colors["pred"], alpha=0.82, zorder=gt_box_z)

        pred_start = max(0, hist_len - 1)
        pred_yaw_map = rec.get("pred_yaw", {})
        for _sample_idx, pred_arr in pred_items:
            if pred_arr.shape[0] == 0:
                continue
            pred_end = min(traj_frame_idx, int(pred_arr.shape[0]) - 1)
            if pred_end >= pred_start:
                _plot_segment(
                    ax,
                    pred_arr,
                    pred_start,
                    pred_end,
                    color=colors["gt"],
                    linewidth=1.25,
                    markersize=1.6,
                    alpha=0.96,
                    zorder=traj_z + 0.10,
                )
            _draw_box_at_index(
                ax,
                pred_arr,
                pred_end,
                role=rec.get("role", ""),
                yaw_arr=pred_yaw_map.get(_sample_idx),
                color=_lighten_color(colors["gt"], amount=0.35),
                alpha=pred_box_alpha,
                zorder=pred_box_z,
            )

    def _draw_frame(frame_idx: int):
        traj_frame_idx = min(int(frame_idx), max_frames - 1)
        left_ax, right_ax = axes
        left_ax.clear()
        right_ax.clear()

        for rec in agent_records:
            _draw_gt_agent(left_ax, rec, traj_frame_idx, show_label=True)
            _draw_diffusion_agent(right_ax, rec, traj_frame_idx)

        import matplotlib.lines as mlines
        left_handles = [
            mlines.Line2D([], [], color="black", linestyle="-", marker="o", markersize=4, linewidth=3.2, label="GT"),
            mlines.Line2D([], [], color="white", marker="s", markerfacecolor="gray", markersize=8, label="Current box"),
        ]
        right_handles = [
            mlines.Line2D([], [], color="black", linestyle="-", marker="o", markersize=4, linewidth=3.2, label="Input"),
            mlines.Line2D([], [], color="slategray", linestyle="-", marker="o", markersize=2.8, linewidth=1.55, label="Diffusion"),
        ]
        left_ax.legend(handles=left_handles, loc="best", fontsize=8)
        right_ax.legend(handles=right_handles, loc="best", fontsize=8)
        _style_axis(left_ax, "GT sample 0", traj_frame_idx)
        _style_axis(right_ax, "Diffusion samples", traj_frame_idx)
        fig.suptitle(csv_path.stem, fontsize=12)

    ani = FuncAnimation(fig, _draw_frame, frames=total_frames, interval=1000.0 / max(1, fps), blit=False, repeat=False)

    if fmt == "gif":
        out_path = csv_path.with_name(f"{csv_path.stem}_compare.gif")
        writer = PillowWriter(fps=max(1, fps))
    elif fmt == "mp4":
        out_path = csv_path.with_name(f"{csv_path.stem}_compare.mp4")
        writer = FFMpegWriter(fps=max(1, fps), bitrate=2600)
    else:
        if FFMpegWriter.isAvailable():
            out_path = csv_path.with_name(f"{csv_path.stem}_compare.mp4")
            writer = FFMpegWriter(fps=max(1, fps), bitrate=2600)
        else:
            out_path = csv_path.with_name(f"{csv_path.stem}_compare.gif")
            writer = PillowWriter(fps=max(1, fps))

    if out_path.exists():
        out_path.unlink()
    ani.save(str(out_path), writer=writer, dpi=max(80, int(dpi)))
    plt.close(fig)
    return out_path


def _build_scene_from_csv(csv_path: Path, cfg: dict, standardization: dict):
    runtime._load_runtime_symbols()

    df = _read_trajectory_csv(csv_path)
    dt = float(cfg.get("data_dt", 0.02))
    if dt <= 0:
        raise ValueError(f"data_dt must be positive, got {dt}")

    center = df[["x", "y"]].to_numpy(dtype=float).mean(axis=0)
    env = runtime.Environment(node_type_list=["PEDESTRIAN"], standardization=standardization)
    env.attention_radius = {(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN): 50}

    data_columns = runtime._make_data_columns()
    scene = runtime.Scene(timesteps=int(df["frame"].max()) + 1, dt=dt, name=csv_path.stem)

    agent_ranges: Dict[int, Tuple[int, int]] = {}
    for agent_id, group in df.groupby("agent", sort=True):
        group = group.sort_values("frame")
        frames = group["frame"].to_numpy(dtype=int)
        full_frames = np.arange(int(frames.min()), int(frames.max()) + 1, dtype=int)
        aligned = group.set_index("frame").reindex(full_frames)
        if aligned[["x", "y"]].isna().any().any():
            raise ValueError(f"{csv_path} has missing frames for agent={agent_id}.")

        x_world = aligned["x"].to_numpy(dtype=float)
        y_world = aligned["y"].to_numpy(dtype=float)
        x_model = x_world - float(center[0])
        y_model = y_world - float(center[1])
        vx = runtime.derivative_of(x_model, dt)
        vy = runtime.derivative_of(y_model, dt)
        ax = runtime.derivative_of(vx, dt)
        ay = runtime.derivative_of(vy, dt)
        valid_xy = np.isfinite(x_world) & np.isfinite(y_world)
        if "yaw" in aligned.columns:
            raw_yaw = pd.to_numeric(aligned["yaw"], errors="coerce").to_numpy(dtype=float)
            yaw = build_mat_yaw_series(x_world, y_world, valid_xy, raw_yaw=raw_yaw)
        else:
            yaw = _yaw_from_xy(x_world, y_world)
        yaw_rate = runtime.derivative_of(np.unwrap(yaw), dt)

        node_data = pd.DataFrame(
            {
                ("position", "x"): x_model,
                ("position", "y"): y_model,
                ("velocity", "x"): vx,
                ("velocity", "y"): vy,
                ("acceleration", "x"): ax,
                ("acceleration", "y"): ay,
                ("heading", "yaw"): yaw,
                ("heading", "yaw_rate"): yaw_rate,
            },
            columns=data_columns,
        )
        description = "ego" if int(agent_id) == 0 else f"target_{int(agent_id)}"
        node = runtime.Node(
            node_type=env.NodeType.PEDESTRIAN,
            node_id=str(int(agent_id) + 1),
            data=node_data,
            first_timestep=int(full_frames[0]),
            description=description,
        )
        scene.nodes.append(node)
        agent_ranges[int(agent_id)] = (int(full_frames[0]), int(full_frames[-1]))

    env.scenes = [scene]
    meta = {
        "center": np.asarray(center, dtype=np.float64),
        "target_dt": dt,
        "agent_ranges": agent_ranges,
        "input_csv": csv_path,
    }
    return env, scene, meta


def _infer_t_and_horizon(csv_path: Path, cfg: dict, agent_ranges: Dict[int, Tuple[int, int]]) -> Tuple[int, int]:
    runtime._load_runtime_symbols()
    config = runtime.EasyDict(dict(cfg))
    hyperparams = runtime._apply_run_timebase(runtime.get_traj_hypers(config), config)
    max_hl = int(hyperparams["maximum_history_length"])
    run_ph = int(hyperparams["prediction_horizon"])

    first_frame = max(start for start, _end in agent_ranges.values())
    last_frame = min(end for _start, end in agent_ranges.values())
    t = int(first_frame + max_hl)
    available_future = int(last_frame - t)
    if available_future <= 0:
        raise ValueError(
            f"{csv_path} does not contain future points after inferred input_end={t}."
        )
    ph = min(run_ph, available_future)
    if ph <= 0:
        raise ValueError(f"Invalid prediction horizon for {csv_path}: {ph}")
    return t, ph


def _nodes_at_t(scene, t: int, env, min_hl: int):
    nodes = runtime._present_nodes(scene, t, env.NodeType.PEDESTRIAN, min_hl)
    ego_node = runtime._resolve_ego_node(scene)
    return sorted(nodes, key=lambda n: (0 if n is ego_node else 1, *runtime._node_sort_tuple(n)))


def _process_one_csv(
    *,
    input_csv: Path,
    output_csv: Path,
    cfg: dict,
    config_path: Path,
    ckpt_path: Path,
    ckpt_dataset: str,
    ckpt_epoch: int,
    standardization: dict,
    model,
) -> None:
    env, scene, meta = _build_scene_from_csv(input_csv, cfg, standardization)
    t, ph = _infer_t_and_horizon(input_csv, cfg, meta["agent_ranges"])

    config = runtime.EasyDict(dict(cfg))
    config.config = str(config_path)
    config.exp_name = str(output_csv.parent)
    config.dataset = str(config.get("dataset", "") or ckpt_dataset or "")
    config.eval_mode = True
    config.eval_at = int(ckpt_epoch)
    config.prediction_horizon = int(ph)

    model.encoder.set_environment(env)
    model.encoder.set_annealing_params()
    model.config = config

    hyperparams = runtime._apply_run_timebase(runtime.get_traj_hypers(config), config)
    min_hl = int(hyperparams["minimum_history_length"])
    nodes = _nodes_at_t(scene, t, env, min_hl)
    if not nodes:
        raise RuntimeError(f"No nodes present at t={t}: {input_csv}")

    labels = runtime._role_labels(scene, nodes)
    records = runtime._generate_records(
        model=model,
        env=env,
        scene=scene,
        nodes_at_t=nodes,
        labels=labels,
        config=config,
        hyperparams=hyperparams,
        t=t,
        center=meta["center"],
    )
    runtime._write_trajectory_csv(output_csv, records, t)
    _truncate_generated_csv_at_render_collisions(output_csv, cfg)


def main() -> None:
    config_path = _resolve_path(DEFAULT_CONFIG)
    cfg = _load_yaml_config(config_path)

    input_dir = _resolve_path(DEFAULT_INPUT_DIR)
    output_root = _resolve_path(DEFAULT_OUTPUT_ROOT)
    output_root.mkdir(parents=True, exist_ok=True)

    csv_paths = _collect_input_csvs(input_dir, None)
    if not csv_paths:
        raise FileNotFoundError(f"No input trajectory CSVs found in {input_dir}")

    runtime = _build_runtime(cfg)
    ckpt_path = _resolve_checkpoint(runtime)
    ckpt_dataset, ckpt_epoch = _extract_dataset_epoch_from_name(ckpt_path.name)
    if ckpt_epoch is None:
        raise ValueError(f"Checkpoint filename must match '<dataset>_epoch<number>.pt': {ckpt_path.name}")

    np.random.seed(runtime.seed)
    torch.manual_seed(runtime.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(runtime.seed)

    device = str(cfg.get("generation_device", "cuda")).strip().lower()
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Set generation_device: cpu in run.yaml or run in a CUDA env.")

    runtime._load_runtime_symbols()
    standardization = runtime._load_standardization(cfg)

    first_env, _first_scene, _first_meta = _build_scene_from_csv(csv_paths[0], cfg, standardization)
    first_t, first_ph = _infer_t_and_horizon(csv_paths[0], cfg, _first_meta["agent_ranges"])
    model_cfg = runtime.EasyDict(dict(cfg))
    model_cfg.config = str(config_path)
    model_cfg.exp_name = str(output_root)
    model_cfg.dataset = str(runtime.dataset or ckpt_dataset or "")
    model_cfg.eval_mode = True
    model_cfg.eval_at = int(ckpt_epoch)
    model_cfg.prediction_horizon = int(first_ph)

    _install_tensorboard_noop()
    original_file_handler = logging.FileHandler
    logging.FileHandler = lambda *a, **k: logging.NullHandler()
    try:
        model, _model_hyperparams = runtime._load_model_no_pkl(
            model_cfg,
            ckpt_path,
            first_env,
            output_root,
            device,
        )
    finally:
        logging.FileHandler = original_file_handler

    ok_count = 0
    print(f"[Info] Config: {config_path}")
    print(f"[Info] Checkpoint: {ckpt_path}")
    print(f"[Info] Inputs: {len(csv_paths)} CSV(s)")
    print(f"[Info] First inferred input_end={first_t}, horizon={first_ph}")

    video_count = 0
    for idx, path in enumerate(csv_paths, start=1):
        out_csv = _make_output_path(path, output_root)
        try:
            if out_csv.exists():
                out_csv.unlink()
            _process_one_csv(
                input_csv=path,
                output_csv=out_csv,
                cfg=cfg,
                config_path=config_path,
                ckpt_path=ckpt_path,
                ckpt_dataset=str(runtime.dataset or ckpt_dataset or ""),
                ckpt_epoch=int(ckpt_epoch),
                standardization=standardization,
                model=model,
            )
            ok_count += 1
            print(f"[{idx}/{len(csv_paths)}] OK: {out_csv}")
            out_video = _render_compare_video(out_csv, cfg)
            video_count += 1
            print(f"[{idx}/{len(csv_paths)}] VIDEO OK: {out_video}")
        except Exception as exc:
            print(f"[{idx}/{len(csv_paths)}] ERROR: {path} ({type(exc).__name__}: {exc})")

    print(f"[Done] generated={ok_count}, videos={video_count}, failed={len(csv_paths) - ok_count}")
    print(f"[Done] output root: {output_root}")


if __name__ == "__main__":
    main()
