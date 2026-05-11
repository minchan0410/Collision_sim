#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG = SCRIPT_DIR / "sbev_catalog_config.yaml"
IMAGE_NAME_RE = re.compile(r"^Image_(?P<label>\d+)_(?P<scenario>.+)_(?P<data_index>\d+)_(?P<frame>\d+)$")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mat_preprocess.util.mat2txt import (  # noqa: E402
    TRAFFIC_TX_RE,
    as_numeric_1d,
    build_sample_indices,
    build_yaw_series,
    load_mat_accessor,
)


@dataclass(frozen=True)
class CatalogImage:
    path: Path
    scenario: str
    collision_label: int
    data_index: int
    image_frame_1based: int


@dataclass
class MatSeries:
    time: np.ndarray
    sample_idx: np.ndarray
    sample_times: np.ndarray
    ego_x: np.ndarray
    ego_y: np.ndarray
    ego_yaw: np.ndarray
    traffic: List[Tuple[int, np.ndarray, np.ndarray, np.ndarray, float, float]]


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config: {path}")
    return cfg


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def _positive_float(value, name: str) -> float:
    out = float(value)
    if out <= 0 or not math.isfinite(out):
        raise ValueError(f"{name} must be positive, got {value}")
    return out


def _load_mat_timing(catalog_cfg: dict) -> Tuple[float, float, float]:
    data_dt = _positive_float(catalog_cfg.get("data_dt", 0.02), "data_dt")
    history_sec = _positive_float(catalog_cfg.get("history_sec", 0.6), "history_sec")
    prediction_sec = _positive_float(catalog_cfg.get("prediction_sec", 0.9), "prediction_sec")
    return data_dt, history_sec, prediction_sec


def _find_carmaker_test_run_file(root: Optional[Path], scenario: str) -> Optional[Path]:
    if root is None:
        return None
    candidates = [
        root / scenario,
        root / "TestRun" / scenario,
        root / f"{scenario}.trf",
        root / "TestRun" / f"{scenario}.trf",
    ]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    try:
        matches = sorted(path for path in root.rglob(scenario) if path.is_file())
    except Exception:
        matches = []
    return matches[0].resolve() if matches else None


def _parse_carmaker_traffic_dimensions(path: Optional[Path]) -> Dict[int, Tuple[float, float]]:
    dims: Dict[int, Tuple[float, float]] = {}
    if path is None or not path.exists():
        return dims
    pattern = re.compile(
        r"^Traffic\.(?P<idx>\d+)\.Basics\.Dimension\s*=\s*"
        r"(?P<length>[-+0-9.eE]+)\s+(?P<width>[-+0-9.eE]+)"
    )
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = pattern.match(line.strip())
            if not match:
                continue
            length = float(match.group("length"))
            width = float(match.group("width"))
            if math.isfinite(width) and math.isfinite(length) and width > 0 and length > 0:
                dims[int(match.group("idx"))] = (width, length)
    return dims


def _find_carmaker_vehicle_file(test_run_file: Optional[Path], vehicle_root: Optional[Path]) -> Optional[Path]:
    if test_run_file is None or not test_run_file.exists():
        return None
    vehicle_name = ""
    with test_run_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = re.match(r"^Vehicle\s*=\s*(?P<name>\S+)", line.strip())
            if match:
                vehicle_name = match.group("name").strip()
                break
    if not vehicle_name:
        return None
    roots: List[Path] = []
    if vehicle_root is not None:
        roots.append(vehicle_root)
    data_root = None
    for parent in test_run_file.parents:
        if parent.name.lower() == "data":
            data_root = parent
            break
    if data_root is not None:
        roots.append(data_root / "Vehicle")
    for root in roots:
        for candidate in (root / vehicle_name, root / f"{vehicle_name}.veh"):
            if candidate.is_file():
                return candidate.resolve()
        try:
            matches = sorted(path for path in root.rglob(vehicle_name) if path.is_file())
        except Exception:
            matches = []
        if matches:
            return matches[0].resolve()
    return None


def _parse_carmaker_ego_dimensions(path: Optional[Path], default_width: float, default_length: float) -> Tuple[float, float]:
    if path is None or not path.exists():
        return float(default_width), float(default_length)
    text = path.read_text(encoding="utf-8", errors="ignore")

    outer_match = re.search(r"^Vehicle\.OuterSkin\s*=\s*(.+)$", text, flags=re.MULTILINE)
    if outer_match:
        values = [float(v) for v in re.findall(r"[-+0-9.eE]+", outer_match.group(1))]
        if len(values) >= 5:
            length = abs(values[3] - values[0])
            width = abs(values[4] - values[1])
            if math.isfinite(width) and math.isfinite(length) and width > 0 and length > 0:
                return width, length

    width_match = re.search(r"^CarGen\.Vehicle\.Width\s*=\s*([-+0-9.eE]+)", text, flags=re.MULTILINE)
    length_match = re.search(r"^CarGen\.Vehicle\.Length\s*=\s*([-+0-9.eE]+)", text, flags=re.MULTILINE)
    if width_match and length_match:
        width = float(width_match.group(1)) / 1000.0
        length = float(length_match.group(1)) / 1000.0
        if math.isfinite(width) and math.isfinite(length) and width > 0 and length > 0:
            return width, length

    return float(default_width), float(default_length)


def _parse_catalog_image(path: Path) -> Optional[CatalogImage]:
    match = IMAGE_NAME_RE.match(path.stem)
    if match is None:
        return None

    return CatalogImage(
        path=path,
        scenario=match.group("scenario"),
        collision_label=int(match.group("label")),
        data_index=int(match.group("data_index")),
        image_frame_1based=int(match.group("frame")),
    )


def _iter_catalog_images(catalog_dir: Path) -> List[CatalogImage]:
    images: List[CatalogImage] = []
    for path in catalog_dir.rglob("Image_*.png"):
        parsed = _parse_catalog_image(path)
        if parsed is not None:
            images.append(parsed)
    images.sort(key=lambda item: (item.scenario, item.data_index, item.image_frame_1based, item.path.name))
    return images


def _find_mat_file(mat_root: Path, image: CatalogImage) -> Optional[Path]:
    direct = mat_root / image.scenario / f"{image.scenario}_data_{image.data_index}.mat"
    if direct.exists():
        return direct.resolve()

    pattern = f"{image.scenario}_data_{image.data_index}.mat"
    matches = sorted((mat_root / image.scenario).rglob(pattern)) if (mat_root / image.scenario).exists() else []
    if matches:
        return matches[0].resolve()
    return None


def _get_first_numeric(accessor, keys: Iterable[str], sample_idx: np.ndarray, label: str) -> np.ndarray:
    for key in keys:
        if accessor.has(key):
            return as_numeric_1d(accessor.get(key), name=key)[sample_idx]
    raise KeyError(f"Missing {label}: tried {', '.join(keys)}")


def _get_optional_numeric(accessor, keys: Iterable[str], sample_idx: np.ndarray) -> Optional[np.ndarray]:
    for key in keys:
        if accessor.has(key):
            try:
                return as_numeric_1d(accessor.get(key), name=key)[sample_idx]
            except Exception:
                pass
    return None


def _load_mat_series(
    mat_path: Path,
    data_dt: float,
    *,
    traffic_dimensions: Optional[Dict[int, Tuple[float, float]]] = None,
    default_target_width: float = 1.825,
    default_target_length: float = 4.650,
) -> MatSeries:
    accessor = load_mat_accessor(mat_path)
    try:
        time = as_numeric_1d(accessor.get("Time"), name="Time")
        sample_idx, _original_dt = build_sample_indices(time, target_dt=data_dt)
        if sample_idx.size < 2:
            raise ValueError("MAT is too short after sampling.")

        ego_x = _get_first_numeric(accessor, ("Car_Con_tx", "Car_tx"), sample_idx, "ego x")
        ego_y = _get_first_numeric(accessor, ("Car_Con_ty", "Car_ty"), sample_idx, "ego y")
        ego_valid = np.isfinite(ego_x) & np.isfinite(ego_y)
        ego_raw_yaw = _get_optional_numeric(accessor, ("Car_Yaw", "Vhcl_Yaw"), sample_idx)
        ego_yaw = build_yaw_series(ego_x, ego_y, ego_valid, raw_yaw=ego_raw_yaw)

        n_objs = None
        if accessor.has("Traffic_nObjs"):
            try:
                n_objs = as_numeric_1d(accessor.get("Traffic_nObjs"), name="Traffic_nObjs")[sample_idx]
            except Exception:
                n_objs = None

        traffic_slots = []
        for key in accessor.keys():
            match = TRAFFIC_TX_RE.match(key)
            if not match:
                continue
            slot_token = match.group(1)
            slot_idx = int(slot_token)
            ty_key = f"Traffic_T{slot_token}_ty"
            if accessor.has(ty_key):
                traffic_slots.append((slot_idx, slot_token, key, ty_key))
        traffic_slots.sort(key=lambda item: item[0])

        traffic: List[Tuple[int, np.ndarray, np.ndarray, np.ndarray, float, float]] = []
        traffic_dimensions = traffic_dimensions or {}
        for slot_idx, slot_token, tx_key, ty_key in traffic_slots:
            tx = as_numeric_1d(accessor.get(tx_key), name=tx_key)[sample_idx]
            ty = as_numeric_1d(accessor.get(ty_key), name=ty_key)[sample_idx]
            valid = np.isfinite(tx) & np.isfinite(ty)

            state_key = tx_key.replace("_tx", "_State")
            if accessor.has(state_key):
                try:
                    state = as_numeric_1d(accessor.get(state_key), name=state_key)[sample_idx]
                    valid &= state > 0
                except Exception:
                    pass

            if n_objs is not None:
                valid &= n_objs > slot_idx

            raw_yaw = None
            for yaw_key in (f"Traffic_T{slot_token}_rz", f"Traffic_T{slot_token}_Yaw", f"Traffic_T{slot_token}_yaw"):
                if accessor.has(yaw_key):
                    try:
                        raw_yaw = as_numeric_1d(accessor.get(yaw_key), name=yaw_key)[sample_idx]
                        break
                    except Exception:
                        raw_yaw = None

            yaw = build_yaw_series(tx, ty, valid, raw_yaw=raw_yaw)
            tx = tx.astype(float)
            ty = ty.astype(float)
            yaw = yaw.astype(float)
            tx[~valid] = np.nan
            ty[~valid] = np.nan
            yaw[~valid] = np.nan
            width, length = traffic_dimensions.get(
                slot_idx,
                (float(default_target_width), float(default_target_length)),
            )
            traffic.append((slot_idx, tx, ty, yaw, float(width), float(length)))

        return MatSeries(
            time=time,
            sample_idx=sample_idx,
            sample_times=time[sample_idx],
            ego_x=ego_x.astype(float),
            ego_y=ego_y.astype(float),
            ego_yaw=ego_yaw.astype(float),
            traffic=traffic,
        )
    finally:
        close = getattr(accessor, "close", None)
        if callable(close):
            close()


def _select_target_slots(
    series: MatSeries,
    *,
    scene_frame: int,
    max_target_distance_m: float,
    all_targets: bool,
) -> List[Tuple[int, np.ndarray, np.ndarray, np.ndarray, float, float]]:
    candidates = []
    ego_xy = np.array([series.ego_x[scene_frame], series.ego_y[scene_frame]], dtype=float)
    for slot_idx, tx, ty, yaw, width, length in series.traffic:
        target_xy = np.array([tx[scene_frame], ty[scene_frame]], dtype=float)
        if not np.isfinite(target_xy).all() or not np.isfinite(ego_xy).all():
            continue
        distance = float(np.linalg.norm(target_xy - ego_xy))
        if distance <= max_target_distance_m:
            candidates.append((distance, slot_idx, tx, ty, yaw, width, length))

    candidates.sort(key=lambda item: (item[0], item[1]))
    if all_targets:
        return [(slot_idx, tx, ty, yaw, width, length) for _distance, slot_idx, tx, ty, yaw, width, length in candidates]
    return [(candidates[0][1], candidates[0][2], candidates[0][3], candidates[0][4], candidates[0][5], candidates[0][6])] if candidates else []


def _window_for_error(
    series: MatSeries,
    *,
    image_frame_1based: int,
    error_sec: float,
    data_dt: float,
    history_sec: float,
    prediction_sec: float,
) -> Dict[str, int | float]:
    raw_frame0 = max(0, int(image_frame_1based) - 1)
    if raw_frame0 >= len(series.time):
        raise ValueError(f"image frame out of MAT range: {image_frame_1based}")

    e_scene = int(np.argmin(np.abs(series.sample_idx - raw_frame0)))
    error_steps = max(1, int(round(float(error_sec) / float(data_dt))))
    history_steps = max(1, int(round(float(history_sec) / float(data_dt))))
    prediction_steps = max(1, int(round(float(prediction_sec) / float(data_dt))))

    future_start = e_scene - error_steps
    input_end = future_start - 1
    input_start = input_end - history_steps
    future_end = input_end + prediction_steps

    if input_start < 0:
        raise ValueError("window starts before MAT sequence.")
    if future_start <= input_end:
        raise ValueError("invalid future start.")
    if not (future_start < e_scene < future_end):
        raise ValueError("error frame is not inside generated future window.")
    if future_end >= len(series.sample_idx):
        raise ValueError("window ends after MAT sequence.")

    return {
        "raw_error_frame_0based": raw_frame0,
        "error_scene_frame": e_scene,
        "input_start": input_start,
        "input_end": input_end,
        "future_start": future_start,
        "future_end": future_end,
        "error_time_sec": float(series.time[raw_frame0]),
        "data_dt": float(data_dt),
        "history_sec": float(history_sec),
        "prediction_sec": float(prediction_sec),
    }


def _write_trajectory_csv(
    path: Path,
    *,
    series: MatSeries,
    window: Dict[str, int | float],
    targets: List[Tuple[int, np.ndarray, np.ndarray, np.ndarray, float, float]],
    ego_width: float,
    ego_length: float,
) -> int:
    rows = []
    start = int(window["input_start"])
    end = int(window["future_end"])
    frames = range(start, end + 1)

    for frame in frames:
        rows.append(
            {
                "agent": 0,
                "sample": 0,
                "frame": frame,
                "x": float(series.ego_x[frame]),
                "y": float(series.ego_y[frame]),
                "yaw": float(series.ego_yaw[frame]),
                "width": float(ego_width),
                "length": float(ego_length),
            }
        )

    for agent_idx, (_slot_idx, tx, ty, yaw, width, length) in enumerate(targets, start=1):
        for frame in frames:
            if not (np.isfinite(tx[frame]) and np.isfinite(ty[frame]) and np.isfinite(yaw[frame])):
                continue
            rows.append(
                {
                    "agent": agent_idx,
                    "sample": 0,
                    "frame": frame,
                    "x": float(tx[frame]),
                    "y": float(ty[frame]),
                    "yaw": float(yaw[frame]),
                    "width": float(width),
                    "length": float(length),
                }
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["agent", "sample", "frame", "x", "y", "yaw", "width", "length"])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _safe_stem(image: CatalogImage) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", image.path.stem)


def _process_one(
    image: CatalogImage,
    *,
    mat_root: Path,
    carmaker_test_run_root: Optional[Path],
    carmaker_vehicle_root: Optional[Path],
    out_dir: Path,
    data_dt: float,
    history_sec: float,
    prediction_sec: float,
    error_sec: float,
    max_target_distance_m: float,
    all_targets: bool,
    ego_width: float,
    ego_length: float,
    default_target_width: float,
    default_target_length: float,
) -> Dict[str, object]:
    mat_path = _find_mat_file(mat_root, image)
    base = {
        "image_name": image.path.name,
        "image_path": str(image.path),
        "scenario": image.scenario,
        "collision_label": image.collision_label,
        "data_index": image.data_index,
        "image_frame_1based": image.image_frame_1based,
        "mat_path": str(mat_path) if mat_path else "",
        "status": "OK",
        "message": "",
    }
    if mat_path is None:
        base.update({"status": "MISSING_MAT", "message": "MAT file not found"})
        return base

    try:
        test_run_file = _find_carmaker_test_run_file(carmaker_test_run_root, image.scenario)
        traffic_dimensions = _parse_carmaker_traffic_dimensions(test_run_file)
        vehicle_file = _find_carmaker_vehicle_file(test_run_file, carmaker_vehicle_root)
        actual_ego_width, actual_ego_length = _parse_carmaker_ego_dimensions(
            vehicle_file,
            default_width=ego_width,
            default_length=ego_length,
        )
        series = _load_mat_series(
            mat_path,
            data_dt=data_dt,
            traffic_dimensions=traffic_dimensions,
            default_target_width=default_target_width,
            default_target_length=default_target_length,
        )
        window = _window_for_error(
            series,
            image_frame_1based=image.image_frame_1based,
            error_sec=error_sec,
            data_dt=data_dt,
            history_sec=history_sec,
            prediction_sec=prediction_sec,
        )
        targets = _select_target_slots(
            series,
            scene_frame=int(window["error_scene_frame"]),
            max_target_distance_m=max_target_distance_m,
            all_targets=all_targets,
        )
        if not targets:
            raise ValueError("no target in distance gate")

        out_csv = out_dir / image.scenario / f"{_safe_stem(image)}_trajectory.csv"
        row_count = _write_trajectory_csv(
            out_csv,
            series=series,
            window=window,
            targets=targets,
            ego_width=actual_ego_width,
            ego_length=actual_ego_length,
        )
        base.update(
            {
                **window,
                "target_slots": ";".join(str(slot_idx) for slot_idx, _tx, _ty, _yaw, _width, _length in targets),
                "target_sizes": ";".join(f"{width:.3f}x{length:.3f}" for _slot_idx, _tx, _ty, _yaw, width, length in targets),
                "carmaker_test_run": str(test_run_file) if test_run_file else "",
                "carmaker_vehicle": str(vehicle_file) if vehicle_file else "",
                "ego_size": f"{actual_ego_width:.3f}x{actual_ego_length:.3f}",
                "num_targets": len(targets),
                "num_rows": row_count,
                "trajectory_csv": str(out_csv),
            }
        )
    except Exception as exc:
        base.update({"status": "ERROR", "message": f"{type(exc).__name__}: {exc}"})
    return base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build diffusion-style trajectory CSVs from FN SBEV image names and MAT files.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Config YAML path.")
    parser.add_argument("--error", type=float, default=None, help="Seconds from generated future start to FN SBEV frame.")
    parser.add_argument("--image-name", default="", help="Process only one image filename.")
    parser.add_argument("--scenario", default="", help="Process one scenario.")
    parser.add_argument("--data-index", type=int, default=None, help="Process one data index.")
    parser.add_argument("--image-frame", type=int, default=None, help="Process one DSM image frame.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of images to process.")
    parser.add_argument("--all-targets", action="store_true", help="Write every nearby target instead of closest target only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _load_config(Path(args.config).resolve())

    catalog_dir = _resolve_repo_path(
        cfg.get("wrong_sbev_dir", cfg.get("catalog_dir", "SBEV_Catalog_FNPrecrash"))
    )
    mat_root = _resolve_repo_path(cfg.get("mat_root", "mat_preprocess/mat"))
    out_dir = _resolve_repo_path(cfg.get("trajectory_csv_dir", "SBEV_Catalog_FNPrecrash/diffusion_input_csv"))
    carmaker_root_value = cfg.get(
        "carmaker_test_run_root",
        r"C:\Users\VIC_26\Desktop\Collision-prediction\CarMaker Project\Data\TestRun",
    )
    carmaker_test_run_root = _resolve_repo_path(carmaker_root_value) if str(carmaker_root_value).strip() else None
    vehicle_root_value = cfg.get(
        "carmaker_vehicle_root",
        r"C:\Users\VIC_26\Desktop\Collision-prediction\CarMaker Project\Data\Vehicle",
    )
    carmaker_vehicle_root = _resolve_repo_path(vehicle_root_value) if str(vehicle_root_value).strip() else None

    data_dt, history_sec, prediction_sec = _load_mat_timing(cfg)
    error_sec = float(args.error if args.error is not None else cfg.get("default_error_sec", 0.3))
    max_target_distance_m = float(cfg.get("max_target_distance_m", 60.0))
    ego_width = float(cfg.get("ego_width", cfg.get("car_width", 1.825)))
    ego_length = float(cfg.get("ego_length", cfg.get("car_length", 4.650)))
    default_target_width = float(cfg.get("target_width", cfg.get("car_width", ego_width)))
    default_target_length = float(cfg.get("target_length", cfg.get("car_length", ego_length)))

    images = _iter_catalog_images(catalog_dir)
    if args.image_name:
        images = [image for image in images if image.path.name == args.image_name]
    if args.scenario:
        images = [image for image in images if image.scenario == args.scenario]
    if args.data_index is not None:
        images = [image for image in images if image.data_index == int(args.data_index)]
    if args.image_frame is not None:
        images = [image for image in images if image.image_frame_1based == int(args.image_frame)]
    if args.limit is not None:
        images = images[: max(0, int(args.limit))]

    rows = [
        _process_one(
            image,
            mat_root=mat_root,
            carmaker_test_run_root=carmaker_test_run_root,
            carmaker_vehicle_root=carmaker_vehicle_root,
            out_dir=out_dir,
            data_dt=data_dt,
            history_sec=history_sec,
            prediction_sec=prediction_sec,
            error_sec=error_sec,
            max_target_distance_m=max_target_distance_m,
            all_targets=bool(args.all_targets),
            ego_width=ego_width,
            ego_length=ego_length,
            default_target_width=default_target_width,
            default_target_length=default_target_length,
        )
        for image in images
    ]

    ok_count = sum(1 for row in rows if row.get("status") == "OK")
    print(f"[Done] processed={len(rows)}, ok={ok_count}, error={len(rows) - ok_count}")
    print(f"[Done] output dir: {out_dir}")


if __name__ == "__main__":
    main()
