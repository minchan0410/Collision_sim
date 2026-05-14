#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm.auto import tqdm

from mat_run import (
    EasyDict,
    _build_mid_expected_ckpt_path,
    _build_noop_filehandler_patch,
    _build_torch_load_redirect,
    _extract_dataset_epoch_from_name,
    _install_tensorboard_noop,
    _resolve_checkpoint,
)


DEFAULT_CONFIG = "configs/run.yaml"
DEFAULT_EVAL_DIR = "experiments/eval_runtime"
DEFAULT_SAMPLES = 20
DEFAULT_SAMPLING = "ddpm"
DEFAULT_DIFFUSION_STRIDE = 1
DEFAULT_TIMESTEP_STRIDE = 10
DEFAULT_TIMESTEP_BATCH = 10
DEFAULT_SEED = 123
DEFAULT_BEST_OF = True
DEFAULT_BICYCLE_ROLLOUT = False

MODE_ORDER = [
    {"mode": "pure", "dynamics": False, "interaction": False},
    {"mode": "dynamics", "dynamics": True, "interaction": False},
    {"mode": "interaction", "dynamics": False, "interaction": True},
    {"mode": "dynamics_interaction", "dynamics": True, "interaction": True},
]


def _load_yaml(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML config: {path}")
    return cfg


def _resolve_path(path_value: Union[str, Path]) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    return (Path.cwd() / path).resolve()


def _runtime_from_cfg(cfg: Dict[str, object]) -> EasyDict:
    runtime = EasyDict()
    runtime.ckpt_dir = str(cfg.get("viz_ckpt_dir_pkl", cfg.get("viz_ckpt_dir", ""))).strip()
    runtime.ckpt_name = str(cfg.get("viz_ckpt_name", "")).strip()
    if runtime.ckpt_dir == "":
        raise ValueError("Set viz_ckpt_dir_pkl or viz_ckpt_dir in config.")
    if runtime.ckpt_name == "":
        raise ValueError("Set viz_ckpt_name in config.")
    return runtime


def _prepare_eval_config(
    base_cfg: Dict[str, object],
    config_path: Path,
    checkpoint_path: Path,
    runtime_dir: Path,
    dataset_override: str,
    epoch_override: Optional[int],
) -> EasyDict:
    ckpt_dataset, ckpt_epoch = _extract_dataset_epoch_from_name(checkpoint_path.name)
    if ckpt_epoch is None:
        if epoch_override is None:
            raise ValueError(
                "Checkpoint filename must be '<dataset>_epoch<number>.pt'."
            )
        ckpt_epoch = int(epoch_override)

    dataset = dataset_override or str(base_cfg.get("dataset", "") or "").strip() or str(ckpt_dataset or "")
    if dataset == "":
        raise ValueError("Could not infer dataset. Set dataset in config.")

    cfg = dict(base_cfg)
    cfg["config"] = str(config_path)
    cfg["exp_name"] = str(runtime_dir.resolve())
    cfg["dataset"] = dataset
    cfg["eval_mode"] = True
    cfg["eval_at"] = int(epoch_override if epoch_override is not None else ckpt_epoch)
    return EasyDict(cfg)


def _parse_scene_limit(argv: List[str]) -> Tuple[Optional[int], List[str]]:
    scene_limit = None
    rest: List[str] = []
    for arg in argv:
        if re.fullmatch(r"--\d+", arg):
            if scene_limit is not None:
                raise SystemExit("Only one scene limit like --20 may be provided.")
            scene_limit = max(0, int(arg[2:]))
        else:
            rest.append(arg)
    return scene_limit, rest


def parse_args() -> argparse.Namespace:
    scene_limit, rest = _parse_scene_limit(sys.argv[1:])
    parser = argparse.ArgumentParser(
        description="Evaluate diffusion modes with optional dynamics and interaction guidance."
    )
    parser.add_argument(
        "--dynamics-guidance",
        choices=("on", "off"),
        default="off",
        help="Turn dynamics guidance on/off for this run. Default: off.",
    )
    parser.add_argument(
        "--interaction-guidance",
        choices=("on", "off"),
        default="off",
        help="Turn GT-conditioned collision/not-collision guidance on/off. Default: off.",
    )
    parser.add_argument(
        "--all-modes",
        action="store_true",
        help="Run pure, dynamics, interaction, and dynamics_interaction in sequence.",
    )
    parser.add_argument(
        "--viz-examples",
        type=int,
        default=None,
        help="Number of fixed target examples to save as images.",
    )
    parser.add_argument(
        "--render-videos",
        action="store_true",
        help="Render mat_run-style MP4/GIF videos for saved visualization examples.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Limit evaluated target items with a deterministic round-robin selection across selected scenes.",
    )
    parser.add_argument(
        "--scene-limit",
        type=int,
        default=None,
        help="Optional normal-form scene limit. The legacy --20 form is still supported.",
    )
    args = parser.parse_args(rest)
    if scene_limit is not None and args.scene_limit is not None:
        raise SystemExit("Use either --20 style or --scene-limit, not both.")
    args.scene_limit = scene_limit if scene_limit is not None else args.scene_limit
    return args


def _mode_from_flags(dynamics_guidance: str, interaction_guidance: str) -> Dict[str, object]:
    dynamics = dynamics_guidance == "on"
    interaction = interaction_guidance == "on"
    if dynamics and interaction:
        mode = "dynamics_interaction"
    elif dynamics:
        mode = "dynamics"
    elif interaction:
        mode = "interaction"
    else:
        mode = "pure"
    return {"mode": mode, "dynamics": dynamics, "interaction": interaction}


def _requested_modes(args: argparse.Namespace) -> List[Dict[str, object]]:
    if args.all_modes:
        return [dict(mode) for mode in MODE_ORDER]
    return [_mode_from_flags(args.dynamics_guidance, args.interaction_guidance)]


def _set_random_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _select_scenes(agent, max_scenes: int | None, seed: int) -> List[object]:
    scenes = list(agent.eval_scenes)
    if max_scenes is None:
        return scenes

    scene_limit = max(0, int(max_scenes))
    if scene_limit < len(scenes):
        rng = np.random.default_rng(int(seed))
        scene_indices = rng.choice(len(scenes), size=scene_limit, replace=False)
        return [scenes[int(idx)] for idx in scene_indices]
    return scenes[:scene_limit]


def _set_eval_mode(config: EasyDict, mode_spec: Dict[str, object]) -> None:
    interaction_enabled = bool(mode_spec["interaction"])
    config.dynamics_guidance_enabled = bool(mode_spec["dynamics"])
    # Keep the interaction sampling-loop enabled; per-target override chooses
    # collision vs not-collision from GT collision status.
    config.collision_guidance_enabled = interaction_enabled
    config.not_collision_guidance_enabled = False
    config.heading_guidance_enabled = False
    config.bicycle_rollout_enabled = bool(DEFAULT_BICYCLE_ROLLOUT)


def _clean_xy(traj: np.ndarray, expected_len: int | None = None) -> Optional[np.ndarray]:
    traj = np.asarray(traj, dtype=np.float32)
    if traj.ndim != 2 or traj.shape[1] < 2:
        return None
    traj = traj[:, :2]
    if expected_len is not None and traj.shape[0] != int(expected_len):
        return None
    if np.isnan(traj).any() or not np.isfinite(traj).all():
        return None
    return traj


def _future_yaws(agent, history: np.ndarray, future: np.ndarray) -> Optional[np.ndarray]:
    history = _clean_xy(history)
    future = _clean_xy(future)
    if future is None:
        return None
    if history is not None and history.shape[0] > 0:
        full = np.vstack((history[-1:], future))
        return agent._compute_yaw_from_positions(full)[1:]
    return agent._compute_yaw_from_positions(future)


def _gt_collision(agent, item: Dict[str, object]) -> bool:
    event = agent._find_first_collision_event(
        traj_a=np.asarray(item["ego_future"], dtype=np.float32),
        yaws_a=np.asarray(item["ego_yaw"], dtype=np.float32),
        traj_b=np.asarray(item["target_future"], dtype=np.float32),
        yaws_b=np.asarray(item["target_yaw"], dtype=np.float32),
        car_length=float(getattr(agent.config, "car_length", 4.65)),
        car_width=float(getattr(agent.config, "car_width", 1.825)),
    )
    return event is not None


def _build_eval_items(
    agent,
    *,
    scenes: List[object],
    timestep_stride: int,
    timestep_batch: int,
) -> List[Dict[str, object]]:
    node_type = "PEDESTRIAN"
    ph = int(agent.ph)
    min_hl = int(agent.min_hl)
    max_hl = int(agent.max_hl)
    items: List[Dict[str, object]] = []
    item_index = 0

    for scene_idx, scene in enumerate(scenes, start=1):
        ego_node = agent._resolve_scene_ego_node(scene)
        if ego_node is None:
            print(f"[Warn] Skipping scene without ego: {scene.name}")
            continue

        print(f"[items] Scene {scene_idx}/{len(scenes)}: {scene.name}")
        for t0 in tqdm(range(0, scene.timesteps, int(timestep_stride)), ncols=90):
            timesteps = np.arange(t0, t0 + int(timestep_batch))
            nodes_per_ts = scene.present_nodes(
                timesteps,
                type=node_type,
                min_history_timesteps=min_hl,
                min_future_timesteps=ph,
                return_robot=not agent.hyperparams["incl_robot_node"],
            )

            for timestep in sorted(nodes_per_ts.keys()):
                present_nodes = sorted(nodes_per_ts[timestep], key=agent._node_sort_tuple)
                if ego_node not in present_nodes:
                    continue

                ego_history = _clean_xy(
                    ego_node.get(np.array([timestep - max_hl, timestep]), {"position": ["x", "y"]})
                )
                ego_future = _clean_xy(
                    ego_node.get(np.array([timestep + 1, timestep + ph]), {"position": ["x", "y"]}),
                    expected_len=ph,
                )
                ego_yaw = _future_yaws(agent, ego_history, ego_future)
                if ego_history is None or ego_future is None or ego_yaw is None or ego_yaw.shape[0] != ph:
                    continue

                for target_node in present_nodes:
                    if target_node is ego_node:
                        continue

                    target_history = _clean_xy(
                        target_node.get(np.array([timestep - max_hl, timestep]), {"position": ["x", "y"]})
                    )
                    target_future = _clean_xy(
                        target_node.get(np.array([timestep + 1, timestep + ph]), {"position": ["x", "y"]}),
                        expected_len=ph,
                    )
                    target_yaw = _future_yaws(agent, target_history, target_future)
                    if target_history is None or target_future is None or target_yaw is None or target_yaw.shape[0] != ph:
                        continue

                    item = {
                        "index": item_index,
                        "scene_index": int(scene_idx),
                        "scene_name": str(scene.name),
                        "scene": scene,
                        "timestep": int(timestep),
                        "target_node": target_node,
                        "target_id": str(getattr(target_node, "id", "")),
                        "ego_history": ego_history,
                        "ego_future": ego_future,
                        "ego_yaw": ego_yaw,
                        "target_history": target_history,
                        "target_future": target_future,
                        "target_yaw": target_yaw,
                    }
                    item["gt_collided"] = _gt_collision(agent, item)
                    items.append(item)
                    item_index += 1

    return items


def _limit_items_round_robin(items: List[Dict[str, object]], max_items: Optional[int]) -> List[Dict[str, object]]:
    if max_items is None:
        return items
    limit = max(0, int(max_items))
    if limit == 0 or len(items) <= limit:
        return items[:limit]

    by_scene: Dict[int, List[Dict[str, object]]] = {}
    for item in items:
        by_scene.setdefault(int(item["scene_index"]), []).append(item)

    selected: List[Dict[str, object]] = []
    cursor = 0
    scene_keys = sorted(by_scene)
    while len(selected) < limit:
        added = False
        for scene_key in scene_keys:
            scene_items = by_scene[scene_key]
            if cursor < len(scene_items):
                selected.append(scene_items[cursor])
                added = True
                if len(selected) >= limit:
                    break
        if not added:
            break
        cursor += 1

    for new_index, item in enumerate(selected):
        item["index"] = int(new_index)
    return selected


def _build_batch_for_item(agent, item: Dict[str, object]):
    from dataset import collate, get_node_timestep_data

    target_node = item["target_node"]
    scene = item["scene"]
    timestep = int(item["timestep"])
    scene_graph = scene.get_scene_graph(
        timestep,
        agent.eval_env.attention_radius,
        agent.hyperparams["edge_addition_filter"],
        agent.hyperparams["edge_removal_filter"],
    )
    edge_types = [
        edge_type for edge_type in agent.eval_env.get_edge_types()
        if edge_type[0] == target_node.type
    ]
    data = get_node_timestep_data(
        agent.eval_env,
        scene,
        timestep,
        target_node,
        agent.hyperparams["state"],
        agent.hyperparams["pred_state"],
        edge_types,
        int(agent.max_hl),
        int(agent.ph),
        agent.hyperparams,
        scene_graph=scene_graph,
    )
    return collate([data])


def _guidance_for_item(item: Dict[str, object], interaction_enabled: bool) -> Tuple[str, Optional[Dict[str, object]]]:
    if not interaction_enabled:
        return "none", None
    selected = "collision" if bool(item["gt_collided"]) else "not_collision"
    return selected, {
        "collision_enabled": selected == "collision",
        "not_collision_enabled": selected == "not_collision",
        "collision_reference_positions": np.asarray(item["ego_future"], dtype=np.float32),
    }


def _dataset_scaled_metrics(config: EasyDict, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if config.dataset == "eth":
        return values / 0.6
    if config.dataset == "sdd":
        return values * 50.0
    return values


def _safe_name(value: object) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "item"


def _plot_sample_image(
    *,
    agent,
    item: Dict[str, object],
    mode: str,
    selected_guidance: str,
    pred_pos: np.ndarray,
    pred_yaw: np.ndarray,
    best_idx: int,
    out_dir: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))

    ego_path = np.vstack((item["ego_history"][-1:], item["ego_future"]))
    ego_yaw_full = np.concatenate(([item["ego_yaw"][0]], item["ego_yaw"]))
    target_gt = np.vstack((item["target_history"][-1:], item["target_future"]))
    target_yaw_full = np.concatenate(([item["target_yaw"][0]], item["target_yaw"]))

    ax.plot(ego_path[:, 0], ego_path[:, 1], color="#1f77b4", linewidth=2.2, label="ego GT")
    ax.plot(target_gt[:, 0], target_gt[:, 1], color="#2ca02c", linewidth=2.2, label="target GT")
    ax.scatter([ego_path[0, 0]], [ego_path[0, 1]], color="#1f77b4", s=35, zorder=4)
    ax.scatter([target_gt[0, 0]], [target_gt[0, 1]], color="#2ca02c", s=35, zorder=4)

    for sample_idx in range(pred_pos.shape[0]):
        pred_path = np.vstack((item["target_history"][-1:], pred_pos[sample_idx]))
        alpha = 0.28 if sample_idx != best_idx else 0.95
        lw = 0.9 if sample_idx != best_idx else 2.4
        color = "#ff7f0e" if sample_idx != best_idx else "#d62728"
        label = "diffusion samples" if sample_idx == 0 else None
        if sample_idx == best_idx:
            label = "best ADE sample"
        ax.plot(pred_path[:, 0], pred_path[:, 1], color=color, alpha=alpha, linewidth=lw, label=label)

    stride = max(1, int(round((int(agent.ph) + 1) / 8)))
    agent._draw_vehicle_rectangles(
        ax,
        ego_path,
        ego_yaw_full,
        float(getattr(agent.config, "car_length", 4.65)),
        float(getattr(agent.config, "car_width", 1.825)),
        color="#1f77b4",
        alpha=0.18,
        stride=stride,
        show_center=False,
        show_yaw_arrow=False,
    )
    agent._draw_vehicle_rectangles(
        ax,
        target_gt,
        target_yaw_full,
        float(getattr(agent.config, "car_length", 4.65)),
        float(getattr(agent.config, "car_width", 1.825)),
        color="#2ca02c",
        alpha=0.18,
        stride=stride,
        show_center=False,
        show_yaw_arrow=False,
    )
    best_path = np.vstack((item["target_history"][-1:], pred_pos[best_idx]))
    best_yaw_full = np.concatenate(([pred_yaw[best_idx, 0]], pred_yaw[best_idx]))
    agent._draw_vehicle_rectangles(
        ax,
        best_path,
        best_yaw_full,
        float(getattr(agent.config, "car_length", 4.65)),
        float(getattr(agent.config, "car_width", 1.825)),
        color="#d62728",
        alpha=0.20,
        stride=stride,
        show_center=False,
        show_yaw_arrow=False,
    )

    all_pts = [ego_path[:, :2], target_gt[:, :2], pred_pos.reshape(-1, 2)]
    pts = np.vstack(all_pts)
    finite = pts[np.isfinite(pts).all(axis=1)]
    if finite.shape[0] > 0:
        pad = max(2.0, float(np.ptp(finite[:, 0]) + np.ptp(finite[:, 1])) * 0.08)
        ax.set_xlim(float(np.min(finite[:, 0]) - pad), float(np.max(finite[:, 0]) + pad))
        ax.set_ylim(float(np.min(finite[:, 1]) - pad), float(np.max(finite[:, 1]) + pad))

    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    ax.set_title(
        f"{mode} | scene={item['scene_name']} t={item['timestep']} "
        f"target={item['target_id']} gt_collision={item['gt_collided']} guide={selected_guidance}",
        fontsize=9,
    )

    filename = (
        f"item_{int(item['index']):04d}_sceneidx_{int(item['scene_index']):03d}_"
        f"scene_{_safe_name(item['scene_name'])}_t{int(item['timestep'])}_"
        f"target_{_safe_name(item['target_id'])}_"
        f"gtcol_{int(bool(item['gt_collided']))}_guide_{selected_guidance}.png"
    )
    path = out_dir / filename
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_item_video_csv(
    *,
    agent,
    item: Dict[str, object],
    pred_pos: np.ndarray,
    pred_yaw: np.ndarray,
    out_dir: Path,
    mode: str,
    selected_guidance: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"item_{int(item['index']):04d}_sceneidx_{int(item['scene_index']):03d}_"
        f"scene_{_safe_name(item['scene_name'])}_t{int(item['timestep'])}_"
        f"target_{_safe_name(item['target_id'])}_"
        f"gtcol_{int(bool(item['gt_collided']))}_guide_{selected_guidance}.csv"
    )
    path = out_dir / filename

    car_width = float(getattr(agent.config, "car_width", 1.825))
    car_length = float(getattr(agent.config, "car_length", 4.650))
    fieldnames = ["agent", "sample", "frame", "x", "y", "yaw", "width", "length"]
    rows: List[Dict[str, object]] = []

    def append_path(agent_idx: int, sample_idx: int, xy: np.ndarray, yaw: np.ndarray) -> None:
        xy = np.asarray(xy, dtype=np.float32)
        yaw = np.asarray(yaw, dtype=np.float32).reshape(-1)
        frame0 = int(item["timestep"]) - int(np.asarray(item["target_history"]).shape[0]) + 1
        for point_idx, point in enumerate(xy[:, :2]):
            yaw_value = float(yaw[min(point_idx, yaw.shape[0] - 1)]) if yaw.shape[0] > 0 else ""
            rows.append(
                {
                    "agent": int(agent_idx),
                    "sample": int(sample_idx),
                    "frame": int(frame0 + point_idx),
                    "x": float(point[0]),
                    "y": float(point[1]),
                    "yaw": yaw_value,
                    "width": car_width,
                    "length": car_length,
                }
            )

    ego_path = np.vstack((item["ego_history"], item["ego_future"]))
    ego_yaw = agent._compute_yaw_from_positions(ego_path)
    target_gt_path = np.vstack((item["target_history"], item["target_future"]))
    target_gt_yaw = agent._compute_yaw_from_positions(target_gt_path)
    target_hist_yaw = agent._compute_yaw_from_positions(np.asarray(item["target_history"], dtype=np.float32))

    append_path(0, 0, ego_path, ego_yaw)
    append_path(1, 0, target_gt_path, target_gt_yaw)
    for sample_idx in range(pred_pos.shape[0]):
        sample_path = np.vstack((item["target_history"], pred_pos[sample_idx]))
        sample_yaw = np.concatenate((target_hist_yaw, pred_yaw[sample_idx]))
        append_path(1, sample_idx + 1, sample_path, sample_yaw)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _render_item_video(csv_path: Path, out_dir: Path, config: EasyDict) -> Path:
    from mat_run import _load_scene_csv, _render_scene_video

    fps = int(config.get("viz_video_fps", 10))
    dpi = int(config.get("viz_video_dpi", 140))
    end_hold_sec = float(config.get("viz_video_end_hold_sec", 1.5))
    fmt = str(config.get("viz_video_format", "auto"))
    bbox_alpha = float(config.get("viz_video_bbox_alpha", 0.22))
    records = _load_scene_csv(csv_path)
    if len(records) == 0:
        raise RuntimeError(f"No valid trajectory rows for video: {csv_path}")
    return _render_scene_video(
        csv_path=csv_path,
        out_dir=out_dir,
        agent_records=records,
        pred_sample_idx="all",
        fps=fps,
        dpi=dpi,
        car_width=float(config.get("car_width", 1.825)),
        car_length=float(config.get("car_length", 4.650)),
        end_hold_sec=end_hold_sec,
        bbox_alpha=bbox_alpha,
        fmt=fmt,
    )


def _evaluate_mode(
    agent,
    *,
    mode_spec: Dict[str, object],
    items: List[Dict[str, object]],
    viz_item_indices: set[int],
    output_dir: Path,
    samples: int,
    sampling: str,
    diffusion_stride: int,
    best_of: bool,
    seed: int,
    render_videos: bool,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    mode = str(mode_spec["mode"])
    _set_eval_mode(agent.config, mode_spec)
    agent.model.config = agent.config
    agent.model.eval()

    rows: List[Dict[str, object]] = []
    mode_dir = output_dir / mode
    image_count = 0
    video_count = 0

    with torch.no_grad():
        for item in tqdm(items, ncols=90, desc=mode):
            item_seed = int(seed) + int(item["index"])
            _set_random_seed(item_seed)

            selected_guidance, guidance_override = _guidance_for_item(item, bool(mode_spec["interaction"]))
            batch = _build_batch_for_item(agent, item)
            pred_pack = agent.model.generate(
                batch,
                "PEDESTRIAN",
                num_points=int(agent.ph),
                sample=int(samples),
                bestof=True,
                sampling=sampling,
                step=int(diffusion_stride),
                return_dynamics=True,
                guidance_override=guidance_override,
            )

            pred_pos = np.asarray(pred_pack["position"][:, 0], dtype=np.float32)
            pred_yaw = np.asarray(pred_pack["yaw"][:, 0], dtype=np.float32)
            target_future = np.asarray(item["target_future"], dtype=np.float32)
            horizon = min(pred_pos.shape[1], target_future.shape[0])
            if horizon <= 0:
                continue

            errors = np.linalg.norm(pred_pos[:, :horizon] - target_future[:horizon], axis=-1)
            ade_by_sample = np.mean(errors, axis=1)
            fde_by_sample = errors[:, -1]
            best_idx = int(np.argmin(ade_by_sample)) if bool(best_of) else 0

            image_path = ""
            video_csv_path = ""
            video_path = ""
            if int(item["index"]) in viz_item_indices:
                image_path = str(_plot_sample_image(
                    agent=agent,
                    item=item,
                    mode=mode,
                    selected_guidance=selected_guidance,
                    pred_pos=pred_pos[:, :horizon],
                    pred_yaw=pred_yaw[:, :horizon],
                    best_idx=best_idx,
                    out_dir=mode_dir,
                ))
                image_count += 1
                if render_videos:
                    video_csv = _write_item_video_csv(
                        agent=agent,
                        item=item,
                        pred_pos=pred_pos[:, :horizon],
                        pred_yaw=pred_yaw[:, :horizon],
                        out_dir=mode_dir / "csv",
                        mode=mode,
                        selected_guidance=selected_guidance,
                    )
                    rendered = _render_item_video(video_csv, mode_dir / "vids", agent.config)
                    video_csv_path = str(video_csv)
                    video_path = str(rendered)
                    video_count += 1

            rows.append(
                {
                    "scene": item["scene_name"],
                    "scene_index": int(item["scene_index"]),
                    "timestep": int(item["timestep"]),
                    "target_id": item["target_id"],
                    "item_index": int(item["index"]),
                    "item_seed": item_seed,
                    "mode": mode,
                    "dynamics_guidance": bool(mode_spec["dynamics"]),
                    "interaction_guidance": bool(mode_spec["interaction"]),
                    "selected_interaction": selected_guidance,
                    "gt_collided": bool(item["gt_collided"]),
                    "best_sample_index": best_idx,
                    "ade": float(ade_by_sample[best_idx]),
                    "fde": float(fde_by_sample[best_idx]),
                    "image_path": image_path,
                    "video_csv_path": video_csv_path,
                    "video_path": video_path,
                }
            )

    ade_values = _dataset_scaled_metrics(agent.config, np.asarray([row["ade"] for row in rows], dtype=float))
    fde_values = _dataset_scaled_metrics(agent.config, np.asarray([row["fde"] for row in rows], dtype=float))
    selected = [str(row["selected_interaction"]) for row in rows]
    gt_collisions = [bool(row["gt_collided"]) for row in rows]

    if len(rows) == 0:
        summary = {
            "mode": mode,
            "targets": 0,
            "ade_mean": "",
            "ade_median": "",
            "fde_mean": "",
            "fde_median": "",
            "gt_collision_count": 0,
            "collision_guidance_count": 0,
            "not_collision_guidance_count": 0,
            "images": image_count,
            "videos": video_count,
        }
    else:
        summary = {
            "mode": mode,
            "targets": int(len(rows)),
            "ade_mean": float(np.mean(ade_values)),
            "ade_median": float(np.median(ade_values)),
            "fde_mean": float(np.mean(fde_values)),
            "fde_median": float(np.median(fde_values)),
            "gt_collision_count": int(sum(gt_collisions)),
            "collision_guidance_count": int(sum(value == "collision" for value in selected)),
            "not_collision_guidance_count": int(sum(value == "not_collision" for value in selected)),
            "images": image_count,
            "videos": video_count,
        }
    return summary, rows


def _fmt(value: object) -> str:
    if value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _print_summary(rows: List[Dict[str, object]]) -> None:
    columns = [
        ("mode", "mode"),
        ("targets", "targets"),
        ("ade_mean", "ADE mean"),
        ("ade_median", "ADE med"),
        ("fde_mean", "FDE mean"),
        ("fde_median", "FDE med"),
        ("gt_collision_count", "GT col"),
        ("collision_guidance_count", "col guide"),
        ("not_collision_guidance_count", "notcol guide"),
        ("images", "images"),
        ("videos", "videos"),
    ]
    table = [[_fmt(row.get(key, "")) for key, _ in columns] for row in rows]
    headers = [label for _, label in columns]
    widths = [
        max(len(headers[idx]), *(len(row[idx]) for row in table))
        for idx in range(len(headers))
    ]
    print("\n=== eval_diffusion summary ===")
    print(" | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for row in table:
        print(" | ".join(row[idx].ljust(widths[idx]) for idx in range(len(row))))


def _write_csv(path: Path, rows: List[Dict[str, object]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _xlsx_cell(col_idx: int, row_idx: int, value: object) -> str:
    col = ""
    idx = col_idx
    while idx > 0:
        idx, rem = divmod(idx - 1, 26)
        col = chr(65 + rem) + col
    ref = f"{col}{row_idx}"
    if value is None or value == "":
        return f'<c r="{ref}"/>'
    if isinstance(value, bool):
        text = "TRUE" if value else "FALSE"
        return f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>'
    if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(float(value)):
        return f'<c r="{ref}"><v>{float(value) if isinstance(value, (float, np.floating)) else int(value)}</v></c>'
    text = escape(str(value))
    return f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>'


def _xlsx_sheet_xml(rows: List[List[object]]) -> str:
    sheet_rows = []
    for row_idx, row in enumerate(rows, start=1):
        cells = "".join(_xlsx_cell(col_idx, row_idx, value) for col_idx, value in enumerate(row, start=1))
        sheet_rows.append(f'<row r="{row_idx}">{cells}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(sheet_rows)}</sheetData>'
        '</worksheet>'
    )


def _write_xlsx(path: Path, sheets: List[Tuple[str, List[Dict[str, object]], List[str]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            + "".join(
                f'<Override PartName="/xl/worksheets/sheet{idx}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                for idx in range(1, len(sheets) + 1)
            )
            + "</Types>",
        )
        zf.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            "</Relationships>",
        )
        zf.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets>'
            + "".join(
                f'<sheet name="{escape(name[:31])}" sheetId="{idx}" r:id="rId{idx}"/>'
                for idx, (name, _, _) in enumerate(sheets, start=1)
            )
            + "</sheets></workbook>",
        )
        zf.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + "".join(
                f'<Relationship Id="rId{idx}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{idx}.xml"/>'
                for idx in range(1, len(sheets) + 1)
            )
            + "</Relationships>",
        )
        for idx, (_, rows, fields) in enumerate(sheets, start=1):
            sheet_rows: List[List[object]] = [fields]
            sheet_rows.extend([[row.get(field, "") for field in fields] for row in rows])
            zf.writestr(f"xl/worksheets/sheet{idx}.xml", _xlsx_sheet_xml(sheet_rows))


def _write_outputs(
    output_dir: Path,
    config: EasyDict,
    summary_rows: List[Dict[str, object]],
    item_rows: List[Dict[str, object]],
) -> Tuple[Path, Path, Path]:
    modes = "_".join(_safe_name(row["mode"]) for row in summary_rows)
    prefix = f"eval_diffusion_{_safe_name(config.dataset)}_epoch{int(config.eval_at)}_{modes}"
    summary_path = output_dir / f"{prefix}_summary.csv"
    items_path = output_dir / f"{prefix}_per_item.csv"
    xlsx_path = output_dir / f"{prefix}.xlsx"
    summary_fields = [
        "mode",
        "targets",
        "ade_mean",
        "ade_median",
        "fde_mean",
        "fde_median",
        "gt_collision_count",
        "collision_guidance_count",
        "not_collision_guidance_count",
        "images",
        "videos",
    ]
    item_fields = [
        "scene",
        "scene_index",
        "timestep",
        "target_id",
        "item_index",
        "item_seed",
        "mode",
        "dynamics_guidance",
        "interaction_guidance",
        "selected_interaction",
        "gt_collided",
        "best_sample_index",
        "ade",
        "fde",
        "image_path",
        "video_csv_path",
        "video_path",
    ]
    _write_csv(
        summary_path,
        summary_rows,
        summary_fields,
    )
    _write_csv(
        items_path,
        item_rows,
        item_fields,
    )
    _write_xlsx(
        xlsx_path,
        [
            ("summary", summary_rows, summary_fields),
            ("per_item", item_rows, item_fields),
        ],
    )
    return summary_path, items_path, xlsx_path


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This evaluation path uses MID's CUDA model. Run in a CUDA-enabled environment.")

    config_path = _resolve_path(DEFAULT_CONFIG)
    base_cfg = _load_yaml(config_path)

    runtime = _runtime_from_cfg(base_cfg)
    checkpoint_path = _resolve_checkpoint(runtime)

    runtime_dir = _resolve_path(DEFAULT_EVAL_DIR)
    runtime_dir.mkdir(parents=True, exist_ok=True)

    config = _prepare_eval_config(
        base_cfg=base_cfg,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        runtime_dir=runtime_dir,
        dataset_override="",
        epoch_override=None,
    )

    expected_ckpt_path = _build_mid_expected_ckpt_path(runtime_dir, config.dataset, int(config.eval_at))
    original_torch_load, patched_torch_load = _build_torch_load_redirect(expected_ckpt_path, checkpoint_path)
    original_file_handler, noop_file_handler = _build_noop_filehandler_patch()

    _install_tensorboard_noop()
    from mid import MID

    print(f"[Info] Config: {config_path}")
    print(f"[Info] Dataset: {config.dataset}")
    print(f"[Info] Checkpoint: {checkpoint_path}")
    print(f"[Info] Expected ckpt redirected: {expected_ckpt_path}")

    torch.load = patched_torch_load
    logging.FileHandler = noop_file_handler
    try:
        agent = MID(config)
    finally:
        torch.load = original_torch_load
        logging.FileHandler = original_file_handler

    modes = _requested_modes(args)
    viz_examples = args.viz_examples
    if viz_examples is None:
        viz_examples = int(base_cfg.get("viz_num_examples", 1) or 1)
    viz_examples = max(0, int(viz_examples))

    _set_random_seed(DEFAULT_SEED)
    scenes = _select_scenes(agent, max_scenes=args.scene_limit, seed=DEFAULT_SEED)
    print(f"[Info] Selected scenes: {len(scenes)}")
    items = _build_eval_items(
        agent,
        scenes=scenes,
        timestep_stride=DEFAULT_TIMESTEP_STRIDE,
        timestep_batch=DEFAULT_TIMESTEP_BATCH,
    )
    if args.max_items is not None:
        original_item_count = len(items)
        items = _limit_items_round_robin(items, args.max_items)
        print(f"[Info] Target items limited: {len(items)} / {original_item_count}")
    print(f"[Info] Target items: {len(items)}")
    if len(items) == 0:
        raise RuntimeError("No valid target items found for evaluation.")

    viz_item_indices = {int(item["index"]) for item in items[:viz_examples]}
    output_dir = runtime_dir / "eval_diffusion_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summary_rows: List[Dict[str, object]] = []
    all_item_rows: List[Dict[str, object]] = []
    for mode_spec in modes:
        _set_random_seed(DEFAULT_SEED)
        summary, rows = _evaluate_mode(
            agent,
            mode_spec=mode_spec,
            items=items,
            viz_item_indices=viz_item_indices,
            output_dir=output_dir,
            samples=DEFAULT_SAMPLES,
            sampling=DEFAULT_SAMPLING,
            diffusion_stride=DEFAULT_DIFFUSION_STRIDE,
            best_of=DEFAULT_BEST_OF,
            seed=DEFAULT_SEED,
            render_videos=bool(args.render_videos),
        )
        all_summary_rows.append(summary)
        all_item_rows.extend(rows)
        print(f"{summary['mode']} ADE: {summary['ade_mean']}")
        print(f"{summary['mode']} FDE: {summary['fde_mean']}")

    _print_summary(all_summary_rows)
    summary_path, items_path, xlsx_path = _write_outputs(output_dir, config, all_summary_rows, all_item_rows)
    print(f"\n[Info] Summary CSV: {summary_path}")
    print(f"[Info] Per-item CSV: {items_path}")
    print(f"[Info] Excel workbook: {xlsx_path}")
    print(f"[Info] Images root: {output_dir}")


if __name__ == "__main__":
    main()
