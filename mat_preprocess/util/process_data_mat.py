#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import dill
import numpy as np
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

Environment = None
Scene = None
Node = None
derivative_of = None


def _load_environment_symbols() -> None:
    global Environment, Scene, Node, derivative_of
    if Environment is not None and Scene is not None and Node is not None and derivative_of is not None:
        return

    from environment import Environment as _Environment  # type: ignore
    from environment import Node as _Node  # type: ignore
    from environment import Scene as _Scene  # type: ignore
    from environment import derivative_of as _derivative_of  # type: ignore

    Environment = _Environment
    Scene = _Scene
    Node = _Node
    derivative_of = _derivative_of


DEFAULT_STANDARDIZATION: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {
    "PEDESTRIAN": {
        "position": {"x": {"mean": 0.0, "std": 3.91}, "y": {"mean": 0.0, "std": 13.49}},
        "velocity": {"x": {"mean": 0.0, "std": 5.06}, "y": {"mean": 0.0, "std": 9.61}},
        "acceleration": {"x": {"mean": 0.0, "std": 3.93}, "y": {"mean": 0.0, "std": 3.30}},
        "heading": {"yaw": {"mean": 0.0, "std": 0.57}, "yaw_rate": {"mean": 0.0, "std": 0.14}},
    }
}


def mat_preprocess_root() -> Path:
    return Path(__file__).resolve().parents[1]


def make_data_columns() -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples(
        [
            ("position", "x"),
            ("position", "y"),
            ("velocity", "x"),
            ("velocity", "y"),
            ("acceleration", "x"),
            ("acceleration", "y"),
            ("heading", "yaw"),
            ("heading", "yaw_rate"),
        ]
    )


def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def build_yaw_series(
    raw_yaw: np.ndarray | None,
    x: np.ndarray,
    y: np.ndarray,
    speed_eps: float = 1.0e-3,
) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    n = x.shape[0]
    yaw = np.full((n,), np.nan, dtype=float)

    if raw_yaw is not None:
        raw = np.asarray(raw_yaw, dtype=float).reshape(-1)
        if raw.shape[0] == n:
            finite_raw = np.isfinite(raw)
            yaw[finite_raw] = wrap_to_pi(raw[finite_raw])

    if n >= 2:
        dx = np.diff(x)
        dy = np.diff(y)
        speed = np.hypot(dx, dy)
        motion = np.full((n,), np.nan, dtype=float)
        valid = np.isfinite(dx) & np.isfinite(dy) & (speed > speed_eps)
        motion[:-1][valid] = np.arctan2(dy[valid], dx[valid])
        motion[-1] = motion[-2]

        missing = ~np.isfinite(yaw)
        yaw[missing] = motion[missing]

    finite = np.isfinite(yaw)
    if not np.any(finite):
        return np.zeros((n,), dtype=float)

    first_valid = int(np.flatnonzero(finite)[0])
    yaw[:first_valid] = yaw[first_valid]
    prev = float(yaw[first_valid])
    for i in range(first_valid + 1, n):
        if np.isfinite(yaw[i]):
            prev = float(yaw[i])
        else:
            yaw[i] = prev
    return wrap_to_pi(yaw)


def maybe_makedirs(path_to_create: Path) -> None:
    path_to_create.mkdir(parents=True, exist_ok=True)


def load_data_dt_from_mat_yaml() -> float:
    config_path = REPO_ROOT / "configs" / "train.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Required config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML content in {config_path}")
    if "data_dt" not in cfg:
        raise KeyError(f"'data_dt' is missing in {config_path}")

    dt_cfg = float(cfg["data_dt"])
    if dt_cfg <= 0:
        raise ValueError(f"'data_dt' must be positive in {config_path}, got {dt_cfg}")
    return dt_cfg


def augment_scene(scene, angle: float):
    def rotate_pc(pc: np.ndarray, alpha: float) -> np.ndarray:
        mat = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
        return mat @ pc

    data_columns = make_data_columns()
    scene_aug = Scene(timesteps=scene.timesteps, dt=scene.dt, name=scene.name)
    alpha = angle * np.pi / 180.0

    for node in scene.nodes:
        x_src = np.array(node.data.position.x, dtype=float)
        y_src = np.array(node.data.position.y, dtype=float)
        x, y = rotate_pc(np.array([x_src, y_src]), alpha)
        vx = derivative_of(x, scene.dt)
        vy = derivative_of(y, scene.dt)
        ax = derivative_of(vx, scene.dt)
        ay = derivative_of(vy, scene.dt)

        try:
            yaw_src = np.array(node.data[:, ("heading", "yaw")], dtype=float).reshape(-1)
        except Exception:
            yaw_src = build_yaw_series(None, x_src, y_src)
        yaw = wrap_to_pi(yaw_src + alpha)
        yaw_rate = derivative_of(np.unwrap(yaw), scene.dt)

        data_dict = {
            ("position", "x"): x,
            ("position", "y"): y,
            ("velocity", "x"): vx,
            ("velocity", "y"): vy,
            ("acceleration", "x"): ax,
            ("acceleration", "y"): ay,
            ("heading", "yaw"): yaw,
            ("heading", "yaw_rate"): yaw_rate,
        }

        node_data = pd.DataFrame(data_dict, columns=data_columns)
        scene_aug.nodes.append(
            Node(
                node_type=node.type,
                node_id=node.id,
                data=node_data,
                first_timestep=node.first_timestep,
            )
        )
    return scene_aug


def augment(scene):
    choices = [scene] + (scene.augmented if hasattr(scene, "augmented") else [])
    scene_aug = np.random.choice(choices)
    scene_aug.temporal_scene_graph = scene.temporal_scene_graph
    return scene_aug


def load_txt_dataframe(full_data_path: Path) -> pd.DataFrame:
    data = pd.read_csv(full_data_path, sep=r"\s+", index_col=False, header=None)
    if data.shape[1] >= 5:
        data = data.iloc[:, :5]
        data.columns = ["frame_id", "track_id", "pos_x", "pos_y", "yaw"]
    else:
        data = data.iloc[:, :4]
        data.columns = ["frame_id", "track_id", "pos_x", "pos_y"]
        data["yaw"] = np.nan

    data["frame_id"] = pd.to_numeric(data["frame_id"], downcast="integer")
    data["track_id"] = pd.to_numeric(data["track_id"], downcast="integer")
    data["yaw"] = pd.to_numeric(data["yaw"], errors="coerce")
    data["frame_id"] -= int(data["frame_id"].min())
    data["node_type"] = "PEDESTRIAN"
    data["node_id"] = data["track_id"].astype(str)
    data.sort_values(["track_id", "frame_id"], inplace=True)

    data["pos_x"] = data["pos_x"] - data["pos_x"].mean()
    data["pos_y"] = data["pos_y"] - data["pos_y"].mean()
    return data


def iter_txt_files(folder: Path) -> Iterable[Path]:
    for p in sorted(folder.rglob("*.txt")):
        if p.is_file():
            yield p


def _std_or_fallback(values: List[float], fallback: float) -> float:
    if not values:
        return float(fallback)
    return float(np.std(np.asarray(values, dtype=float)))


def estimate_standardization_from_folders(
    folders: List[Path],
    dt: float,
    min_points_per_track: int = 3,
) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    all_positions_x: List[float] = []
    all_positions_y: List[float] = []
    all_velocities_x: List[float] = []
    all_velocities_y: List[float] = []
    all_accelerations_x: List[float] = []
    all_accelerations_y: List[float] = []
    all_heading_yaw: List[float] = []
    all_heading_yaw_rate: List[float] = []

    scanned_files = 0
    for folder in folders:
        if not folder.exists():
            print(f"[WARN] standardization source folder not found: {folder}")
            continue

        for txt_file in iter_txt_files(folder):
            scanned_files += 1
            data = load_txt_dataframe(txt_file)

            for _, group in data.groupby("track_id"):
                if len(group) < min_points_per_track:
                    continue

                pos_x = group["pos_x"].to_numpy(dtype=float)
                pos_y = group["pos_y"].to_numpy(dtype=float)
                vel_x = np.gradient(pos_x, dt)
                vel_y = np.gradient(pos_y, dt)
                acc_x = np.gradient(vel_x, dt)
                acc_y = np.gradient(vel_y, dt)
                yaw = build_yaw_series(group["yaw"].to_numpy(dtype=float), pos_x, pos_y)
                yaw_rate = np.gradient(np.unwrap(yaw), dt)

                all_positions_x.extend(pos_x.tolist())
                all_positions_y.extend(pos_y.tolist())
                all_velocities_x.extend(vel_x.tolist())
                all_velocities_y.extend(vel_y.tolist())
                all_accelerations_x.extend(acc_x.tolist())
                all_accelerations_y.extend(acc_y.tolist())
                all_heading_yaw.extend(yaw.tolist())
                all_heading_yaw_rate.extend(yaw_rate.tolist())

    if scanned_files == 0:
        raise RuntimeError("No TXT files scanned while estimating standardization.")

    std_pos_x = _std_or_fallback(all_positions_x, 1.0)
    std_pos_y = _std_or_fallback(all_positions_y, 1.0)
    std_vel_x = _std_or_fallback(all_velocities_x, 2.0)
    std_vel_y = _std_or_fallback(all_velocities_y, 2.0)
    std_acc_x = _std_or_fallback(all_accelerations_x, 1.0)
    std_acc_y = _std_or_fallback(all_accelerations_y, 1.0)
    std_yaw = _std_or_fallback(all_heading_yaw, float(np.pi))
    std_yaw_rate = _std_or_fallback(all_heading_yaw_rate, 1.0)

    print("[INFO] estimated standardization from TXT:")
    print(f"       position std(x,y)=({std_pos_x:.4f}, {std_pos_y:.4f})")
    print(f"       velocity std(x,y)=({std_vel_x:.4f}, {std_vel_y:.4f})")
    print(f"       accel    std(x,y)=({std_acc_x:.4f}, {std_acc_y:.4f})")
    print(f"       heading  std(yaw,yaw_rate)=({std_yaw:.4f}, {std_yaw_rate:.4f})")

    return {
        "PEDESTRIAN": {
            "position": {"x": {"mean": 0.0, "std": std_pos_x}, "y": {"mean": 0.0, "std": std_pos_y}},
            "velocity": {"x": {"mean": 0.0, "std": std_vel_x}, "y": {"mean": 0.0, "std": std_vel_y}},
            "acceleration": {"x": {"mean": 0.0, "std": std_acc_x}, "y": {"mean": 0.0, "std": std_acc_y}},
            "heading": {"yaw": {"mean": 0.0, "std": std_yaw}, "yaw_rate": {"mean": 0.0, "std": std_yaw_rate}},
        }
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MAT pkl datasets from split TXT folders.")
    parser.add_argument("--suffix", type=str, default="", help="Suffix for split folders and output names.")
    parser.add_argument(
        "--mat-txt-root",
        type=str,
        default="mat_preprocess/mat_txt",
        help="Root folder containing split TXT folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="mat_preprocess/processed_data",
        help="Output folder for pkl files.",
    )
    parser.add_argument(
        "--standardization-mode",
        type=str,
        choices=["fixed", "estimate"],
        default="fixed",
        help="Use fixed defaults or estimate from TXT (getparam-style).",
    )
    parser.add_argument(
        "--standardization-scope",
        type=str,
        choices=["train", "all"],
        default="train",
        help="When estimating, use only train split or all train/val/test splits.",
    )
    parser.add_argument(
        "--standardization-min-points",
        type=int,
        default=3,
        help="Minimum points per track for standardization estimation.",
    )
    parser.add_argument(
        "--standardization-save",
        type=str,
        default="",
        help="Optional path to save estimated standardization as YAML.",
    )
    parser.add_argument(
        "--num-train-augmentations",
        type=int,
        default=4,
        help="Number of random rotations per train scene.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _load_environment_symbols()
    dt = load_data_dt_from_mat_yaml()
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    print(f"[INFO] process_data_mat dt={dt:.6f} sec (from configs/train.yaml)")

    mat_txt_root = Path(args.mat_txt_root)
    if not mat_txt_root.is_absolute():
        mat_txt_root = (REPO_ROOT / mat_txt_root).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()
    maybe_makedirs(output_dir)

    suffix = str(args.suffix).strip()
    dir_suffix = f"_{suffix}" if suffix else ""
    split_folders = {
        "train": mat_txt_root / f"train{dir_suffix}",
        "val": mat_txt_root / f"val{dir_suffix}",
        "test": mat_txt_root / f"test{dir_suffix}",
    }

    if args.standardization_mode == "estimate":
        if args.standardization_scope == "train":
            std_folders = [split_folders["train"]]
        else:
            std_folders = [split_folders["train"], split_folders["val"], split_folders["test"]]
        standardization_cfg = estimate_standardization_from_folders(
            folders=std_folders,
            dt=dt,
            min_points_per_track=int(args.standardization_min_points),
        )
        if args.standardization_save:
            save_path = Path(args.standardization_save)
            if not save_path.is_absolute():
                save_path = (REPO_ROOT / save_path).resolve()
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with save_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(standardization_cfg, f, sort_keys=False, allow_unicode=False)
            print(f"[INFO] saved estimated standardization: {save_path}")
    else:
        standardization_cfg = DEFAULT_STANDARDIZATION

    data_columns = make_data_columns()
    num_aug = max(0, int(args.num_train_augmentations))

    for split_name in ("train", "val", "test"):
        target_dir = split_folders[split_name]
        if suffix:
            output_name = f"mat_{suffix}_{split_name}"
        else:
            output_name = f"mat_{split_name}"
        data_dict_path = output_dir / f"{output_name}.pkl"

        if not target_dir.exists():
            print(f"[WARN] split folder missing, skip: {target_dir}")
            continue

        print(f"[{split_name.upper()}] processing: {target_dir}")
        env = Environment(node_type_list=["PEDESTRIAN"], standardization=standardization_cfg)
        env.attention_radius = {(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN): 50}

        scenes: List[Scene] = []
        for txt_path in iter_txt_files(target_dir):
            print(f"  - {txt_path}")
            data = load_txt_dataframe(txt_path)
            max_timesteps = int(data["frame_id"].max())
            scene = Scene(
                timesteps=max_timesteps + 1,
                dt=dt,
                name=output_name,
                aug_func=augment if split_name == "train" else None,
            )

            for node_id in pd.unique(data["node_id"]):
                node_df = data[data["node_id"] == node_id]
                node_values = node_df[["pos_x", "pos_y"]].to_numpy(dtype=float)
                if node_values.shape[0] < 2:
                    continue

                new_first_idx = int(node_df["frame_id"].iloc[0])
                x = node_values[:, 0]
                y = node_values[:, 1]
                vx = derivative_of(x, scene.dt)
                vy = derivative_of(y, scene.dt)
                ax = derivative_of(vx, scene.dt)
                ay = derivative_of(vy, scene.dt)
                yaw_raw = node_df["yaw"].to_numpy(dtype=float)
                yaw = build_yaw_series(yaw_raw, x, y)
                yaw_rate = derivative_of(np.unwrap(yaw), scene.dt)

                data_dict = {
                    ("position", "x"): x,
                    ("position", "y"): y,
                    ("velocity", "x"): vx,
                    ("velocity", "y"): vy,
                    ("acceleration", "x"): ax,
                    ("acceleration", "y"): ay,
                    ("heading", "yaw"): yaw,
                    ("heading", "yaw_rate"): yaw_rate,
                }
                node_data = pd.DataFrame(data_dict, columns=data_columns)
                node = Node(node_type=env.NodeType.PEDESTRIAN, node_id=node_id, data=node_data)
                node.first_timestep = new_first_idx
                scene.nodes.append(node)

            if split_name == "train" and num_aug > 0:
                scene.augmented = []
                angles = np.random.uniform(-180.0, 180.0, size=num_aug)
                for angle in angles:
                    scene.augmented.append(augment_scene(scene, float(angle)))

            scenes.append(scene)

        env.scenes = scenes
        print(f"[INFO] scenes built: {len(scenes)} ({output_name})")
        if scenes:
            with data_dict_path.open("wb") as f:
                dill.dump(env, f, protocol=dill.HIGHEST_PROTOCOL)
            print(f"[INFO] saved: {data_dict_path}")

    print("[DONE] process_data_mat complete")


if __name__ == "__main__":
    main()
