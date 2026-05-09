#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

try:
    from easydict import EasyDict
except Exception:
    class EasyDict(dict):
        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError as e:
                raise AttributeError(key) from e

        def __setattr__(self, key, value):
            self[key] = value


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


Environment = None
Node = None
Scene = None
derivative_of = None
collate = None
get_node_timestep_data = None
AutoEncoder = None
Trajectron = None
ModelRegistrar = None
get_traj_hypers = None


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def _load_runtime_symbols() -> None:
    global Environment, Node, Scene, derivative_of
    global collate, get_node_timestep_data
    global AutoEncoder, Trajectron, ModelRegistrar, get_traj_hypers

    if Environment is not None:
        return

    from dataset import collate as _collate
    from dataset import get_node_timestep_data as _get_node_timestep_data
    from environment import Environment as _Environment
    from environment import Node as _Node
    from environment import Scene as _Scene
    from environment import derivative_of as _derivative_of
    from models.autoencoder import AutoEncoder as _AutoEncoder
    from models.trajectron import Trajectron as _Trajectron
    from utils.model_registrar import ModelRegistrar as _ModelRegistrar
    from utils.trajectron_hypers import get_traj_hypers as _get_traj_hypers

    Environment = _Environment
    Node = _Node
    Scene = _Scene
    derivative_of = _derivative_of
    collate = _collate
    get_node_timestep_data = _get_node_timestep_data
    AutoEncoder = _AutoEncoder
    Trajectron = _Trajectron
    ModelRegistrar = _ModelRegistrar
    get_traj_hypers = _get_traj_hypers

def _load_standardization(cfg: dict) -> dict:
    std_path_value = str(cfg.get("standardization_path", "") or "").strip()
    if std_path_value == "":
        candidates = [
            REPO_ROOT / "mat_preprocess" / "processed_data" / "standardization_collision.yaml",
            REPO_ROOT / "processed_data" / "standardization_collision.yaml",
            REPO_ROOT / "mat_preprocess" / "config" / "standardization_collision.yaml",
        ]
        std_path = next((p for p in candidates if p.exists()), None)
    else:
        std_path = _resolve_path(std_path_value)

    if std_path is not None and std_path.exists():
        with std_path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        if isinstance(loaded, dict):
            print(f"[Info] Standardization: {std_path}")
            return loaded

    from mat_preprocess.util.process_data_mat import DEFAULT_STANDARDIZATION

    print("[Warn] Standardization YAML not found. Using process_data_mat DEFAULT_STANDARDIZATION.")
    return DEFAULT_STANDARDIZATION

def _make_data_columns() -> pd.MultiIndex:
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

def _wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi

def _numeric_node_id(node) -> Optional[int]:
    raw_id = getattr(node, "id", node)
    if isinstance(raw_id, (int, np.integer)):
        return int(raw_id)
    if isinstance(raw_id, float) and float(raw_id).is_integer():
        return int(raw_id)
    raw = str(raw_id).strip()
    try:
        value = float(raw)
        if value.is_integer():
            return int(value)
    except Exception:
        pass
    match = re.search(r"-?\d+", raw)
    return int(match.group()) if match else None

def _node_sort_tuple(node) -> Tuple[int, int, str]:
    numeric_id = _numeric_node_id(node)
    if numeric_id is None:
        return (1, 10**9, str(getattr(node, "id", "")))
    return (0, numeric_id, str(getattr(node, "id", "")))

def _resolve_ego_node(scene: Scene):
    nodes = list(getattr(scene, "nodes", []))
    explicit = [n for n in nodes if "ego" in str(getattr(n, "description", "") or "").lower()]
    if explicit:
        return sorted(explicit, key=_node_sort_tuple)[0]
    id_one = [n for n in nodes if _numeric_node_id(n) == 1]
    if id_one:
        return sorted(id_one, key=_node_sort_tuple)[0]
    return sorted(nodes, key=_node_sort_tuple)[0] if nodes else None

def _role_labels(scene: Scene, nodes_at_t: Sequence[Node]) -> Dict[Node, str]:
    ego = _resolve_ego_node(scene)
    labels: Dict[Node, str] = {}
    if ego in nodes_at_t:
        labels[ego] = "Ego"
    others = sorted([n for n in nodes_at_t if n is not ego], key=_node_sort_tuple)
    for idx, node in enumerate(others, start=1):
        labels[node] = "Target" if len(others) == 1 else f"Opponent {idx}"
    for node in nodes_at_t:
        labels.setdefault(node, f"ID {node.id}")
    return labels

def _apply_run_timebase(hyperparams: dict, cfg: dict) -> dict:
    out = dict(hyperparams)
    dt = float(cfg.get("data_dt", out.get("dt", 0.02)))
    out["dt"] = dt

    if cfg.get("prediction_horizon", None) is not None:
        out["prediction_horizon"] = max(1, int(round(float(cfg["prediction_horizon"]))))
    elif cfg.get("prediction_sec", None) is not None:
        out["prediction_horizon"] = max(1, int(round(float(cfg["prediction_sec"]) / dt)))

    if cfg.get("minimum_history_length", None) is not None:
        out["minimum_history_length"] = max(1, int(round(float(cfg["minimum_history_length"]))))
    if cfg.get("maximum_history_length", None) is not None:
        out["maximum_history_length"] = max(1, int(round(float(cfg["maximum_history_length"]))))
    elif cfg.get("history_sec", None) is not None:
        h = max(1, int(round(float(cfg["history_sec"]) / dt)))
        out["minimum_history_length"] = h
        out["maximum_history_length"] = h

    if out["minimum_history_length"] > out["maximum_history_length"]:
        out["minimum_history_length"] = out["maximum_history_length"]
    return out

def _load_model_no_pkl(config: EasyDict, ckpt_path: Path, env: Environment, model_dir: Path, device: str):
    hyperparams = _apply_run_timebase(get_traj_hypers(config), config)
    hyperparams["enc_rnn_dim_edge"] = config.encoder_dim // 2
    hyperparams["enc_rnn_dim_edge_influence"] = config.encoder_dim // 2
    hyperparams["enc_rnn_dim_history"] = config.encoder_dim // 2
    hyperparams["enc_rnn_dim_future"] = config.encoder_dim // 2

    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    registrar = ModelRegistrar(str(model_dir), device)
    registrar.load_models(checkpoint["encoder"])
    encoder = Trajectron(registrar, hyperparams, device)
    encoder.set_environment(env)
    encoder.set_annealing_params()
    model = AutoEncoder(config, encoder).to(device)
    model.load_state_dict(checkpoint["ddpm"])
    model.eval()
    return model, hyperparams

def _present_nodes(scene: Scene, t: int, node_type, min_hl: int) -> List[Node]:
    present = scene.present_nodes(
        np.array([t]),
        type=node_type,
        min_history_timesteps=min_hl,
        min_future_timesteps=0,
    )
    return list(present.get(t, []))

def _state_history(node: Node, t: int, max_hl: int, state, center: np.ndarray):
    arr = node.get(np.array([t - max_hl, t]), state)
    scene_ts = np.arange(t - max_hl, t + 1, dtype=int)
    valid = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1])
    arr = arr[valid]
    scene_ts = scene_ts[valid]
    out = []
    for idx, row in enumerate(arr):
        out.append(
            {
                "point_idx": int(idx),
                "scene_t": int(scene_ts[idx]),
                "x_model": float(row[0]),
                "y_model": float(row[1]),
                "x": float(row[0] + center[0]),
                "y": float(row[1] + center[1]),
                "vx": float(row[2]) if row.shape[0] > 2 else float("nan"),
                "vy": float(row[3]) if row.shape[0] > 3 else float("nan"),
                "ax": float(row[4]) if row.shape[0] > 4 else float("nan"),
                "ay": float(row[5]) if row.shape[0] > 5 else float("nan"),
                "yaw": float(row[6]) if row.shape[0] > 6 else float("nan"),
                "yaw_rate": float(row[7]) if row.shape[0] > 7 else float("nan"),
            }
        )
    return out

def _future_xy(node: Node, t: int, ph: int, center: np.ndarray) -> np.ndarray:
    fut = node.get(np.array([t + 1, t + ph]), {"position": ["x", "y"]})
    valid = np.isfinite(fut[:, 0]) & np.isfinite(fut[:, 1])
    fut = fut[valid]
    if fut.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    fut = np.asarray(fut, dtype=np.float32)
    fut[:, 0] += float(center[0])
    fut[:, 1] += float(center[1])
    return fut

def _build_collision_guidance_override(config: EasyDict, reference_future_xy_model: Optional[np.ndarray]):
    collision_enabled = bool(getattr(config, "collision_guidance_enabled", False))
    not_collision_enabled = bool(getattr(config, "not_collision_guidance_enabled", False))
    if (not collision_enabled) and (not not_collision_enabled):
        return None
    if reference_future_xy_model is None:
        return None
    ref = np.asarray(reference_future_xy_model, dtype=np.float32)
    if ref.ndim != 2 or ref.shape[0] <= 0 or ref.shape[1] != 2:
        return None
    return {
        "collision_enabled": collision_enabled,
        "not_collision_enabled": not_collision_enabled,
        "collision_reference_positions": ref,
    }

def _merge_guidance_overrides(*overrides):
    merged = {}
    for override in overrides:
        if isinstance(override, dict) and len(override) > 0:
            merged.update(override)
    return merged if merged else None

def _last_finite_history_value(history_state: Sequence[dict], key: str) -> Optional[float]:
    for item in reversed(history_state):
        try:
            value = float(item.get(key, float("nan")))
        except Exception:
            continue
        if np.isfinite(value):
            return value
    return None

def _build_heading_guidance_override(config: EasyDict, history_state: Sequence[dict]):
    if not bool(getattr(config, "heading_guidance_enabled", False)):
        return None

    target_yaw = _last_finite_history_value(history_state, "yaw")
    if target_yaw is None:
        return None

    stationary_only = bool(getattr(config, "heading_guidance_stationary_only", True))
    if stationary_only:
        speed_threshold = max(float(getattr(config, "heading_guidance_stationary_speed", 0.30)), 0.0)
        displacement_threshold = max(float(getattr(config, "heading_guidance_stationary_displacement", 0.35)), 0.0)
        dt = max(float(getattr(config, "data_dt", 0.02)), 1.0e-6)

        speeds = []
        xy = []
        scene_t = []
        for item in history_state:
            try:
                vx = float(item.get("vx", float("nan")))
                vy = float(item.get("vy", float("nan")))
                if np.isfinite(vx) and np.isfinite(vy):
                    speeds.append(float(np.hypot(vx, vy)))
                x = float(item.get("x_model", item.get("x", float("nan"))))
                y = float(item.get("y_model", item.get("y", float("nan"))))
                if np.isfinite(x) and np.isfinite(y):
                    xy.append((x, y))
                    scene_t.append(int(item.get("scene_t", len(scene_t))))
            except Exception:
                continue

        stationary_by_speed = False
        if len(speeds) > 0:
            speeds_arr = np.asarray(speeds, dtype=np.float32)
            stationary_by_speed = float(np.nanmedian(speeds_arr)) <= speed_threshold

        stationary_by_disp = False
        if len(xy) >= 2:
            xy_arr = np.asarray(xy, dtype=np.float32)
            displacement = float(np.linalg.norm(xy_arr[-1] - xy_arr[0]))
            duration = max(float(max(scene_t[-1] - scene_t[0], 1)) * dt, dt)
            displacement_speed = displacement / duration
            stationary_by_disp = displacement <= displacement_threshold or displacement_speed <= speed_threshold

        if not (stationary_by_speed or stationary_by_disp):
            return None

    return {
        "heading_enabled": True,
        "heading_target_yaw": float(_wrap_to_pi(np.asarray([target_yaw], dtype=np.float32))[0]),
    }

def _write_history_csv(path: Path, records: list) -> None:
    fieldnames = [
        "agent_index",
        "agent_id",
        "role",
        "point_idx",
        "scene_t",
        "x",
        "y",
        "x_model",
        "y_model",
        "vx",
        "vy",
        "ax",
        "ay",
        "yaw",
        "yaw_rate",
        "is_current",
    ]
    rows = []
    for rec in records:
        hist = rec["history_state"]
        current_idx = max((h["point_idx"] for h in hist), default=-1)
        for h in hist:
            rows.append(
                {
                    "agent_index": rec["agent_index"],
                    "agent_id": rec["agent_id"],
                    "role": rec["role"],
                    **h,
                    "is_current": int(h["point_idx"] == current_idx),
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def _write_diffusion_csv(path: Path, records: list, center: np.ndarray, t: int, dt: float) -> None:
    fieldnames = [
        "agent_index",
        "agent_id",
        "role",
        "source",
        "sample_idx",
        "point_idx",
        "scene_t",
        "x",
        "y",
        "x_model",
        "y_model",
        "vx",
        "vy",
        "speed",
        "yaw",
        "relative_distance",
        "relative_speed",
        "closing_speed",
        "ttc",
    ]
    rows = []
    for rec in records:
        dyn = rec.get("dynamics")
        if dyn is None:
            continue
        pos = np.asarray(dyn["position"], dtype=np.float32)
        vel = np.asarray(dyn.get("velocity", np.full_like(pos, np.nan)), dtype=np.float32)
        speed = np.asarray(dyn.get("speed", np.full(pos.shape[:2], np.nan)), dtype=np.float32)
        yaw = np.asarray(dyn.get("yaw", np.full(pos.shape[:2], np.nan)), dtype=np.float32)
        ref = rec.get("reference_future_model", None)
        ref_vel = None
        if ref is not None:
            ref = np.asarray(ref, dtype=np.float32)
            if ref.ndim == 2 and ref.shape[0] > 0 and ref.shape[1] >= 2:
                ref = ref[:, :2]
                ref_vel = np.zeros_like(ref, dtype=np.float32)
                if ref.shape[0] > 1:
                    ref_vel[:-1] = (ref[1:] - ref[:-1]) / max(float(dt), 1.0e-6)
                    ref_vel[-1] = ref_vel[-2]
            else:
                ref = None
        for sample_idx in range(pos.shape[0]):
            for point_idx in range(pos.shape[1]):
                x_model = float(pos[sample_idx, point_idx, 0])
                y_model = float(pos[sample_idx, point_idx, 1])
                relative_distance = ""
                relative_speed = ""
                closing_speed = ""
                ttc = ""
                if ref is not None and ref_vel is not None and point_idx < ref.shape[0]:
                    rel_pos = pos[sample_idx, point_idx, :2] - ref[point_idx, :2]
                    rel_vel = vel[sample_idx, point_idx, :2] - ref_vel[point_idx, :2]
                    dist = float(np.linalg.norm(rel_pos))
                    rel_spd = float(np.linalg.norm(rel_vel))
                    close = float(-np.dot(rel_pos, rel_vel) / max(dist, 1.0e-6))
                    relative_distance = dist
                    relative_speed = rel_spd
                    closing_speed = close
                    ttc = (dist / close) if close > 1.0e-6 else float("inf")
                rows.append(
                    {
                        "agent_index": rec["agent_index"],
                        "agent_id": rec["agent_id"],
                        "role": rec["role"],
                        "source": rec.get("source", "diffusion"),
                        "sample_idx": int(sample_idx + 1),
                        "point_idx": int(point_idx),
                        "scene_t": int(t + point_idx + 1),
                        "x": float(x_model + center[0]),
                        "y": float(y_model + center[1]),
                        "x_model": x_model,
                        "y_model": y_model,
                        "vx": float(vel[sample_idx, point_idx, 0]),
                        "vy": float(vel[sample_idx, point_idx, 1]),
                        "speed": float(speed[sample_idx, point_idx]),
                        "yaw": float(yaw[sample_idx, point_idx]),
                        "relative_distance": relative_distance,
                        "relative_speed": relative_speed,
                        "closing_speed": closing_speed,
                        "ttc": ttc,
                    }
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def _write_trajectory_csv(path: Path, records: list, t: int) -> None:
    fieldnames = ["agent", "sample", "frame", "x", "y", "yaw"]

    def _path_yaw(xy: np.ndarray, raw_yaw=None, default_yaw: float = 0.0) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float32)
        n = int(xy.shape[0]) if xy.ndim == 2 else 0
        if n <= 0:
            return np.zeros((0,), dtype=np.float32)
        yaw = np.full((n,), float(default_yaw), dtype=np.float32)
        prev = float(default_yaw)
        if n > 1:
            delta = np.diff(xy[:, :2], axis=0)
            for idx in range(n - 1):
                if np.linalg.norm(delta[idx]) > 1.0e-4:
                    prev = float(np.arctan2(delta[idx, 1], delta[idx, 0]))
                yaw[idx] = prev
            yaw[-1] = prev
        if raw_yaw is not None:
            raw = np.asarray(raw_yaw, dtype=np.float32).reshape(-1)
            limit = min(n, int(raw.shape[0]))
            valid = np.isfinite(raw[:limit])
            valid_idx = np.nonzero(valid)[0]
            yaw[valid_idx] = raw[:limit][valid_idx]
        return _wrap_to_pi(yaw)

    rows = []
    for rec in records:
        history_xy = np.asarray([[h["x"], h["y"]] for h in rec["history_state"]], dtype=np.float32)
        history_frames = [int(h["scene_t"]) for h in rec["history_state"]]
        history_yaw_raw = np.asarray([float(h.get("yaw", float("nan"))) for h in rec["history_state"]], dtype=np.float32)
        finite_history_yaw = history_yaw_raw[np.isfinite(history_yaw_raw)]
        default_yaw = float(finite_history_yaw[-1]) if finite_history_yaw.size > 0 else 0.0
        history_yaw = _path_yaw(history_xy, raw_yaw=history_yaw_raw, default_yaw=default_yaw)
        future_gt = np.asarray(rec.get("future_gt", np.zeros((0, 2))), dtype=np.float32)
        if history_xy.ndim != 2 or history_xy.shape[0] == 0:
            continue
        if future_gt.ndim == 2 and future_gt.shape[0] > 0:
            gt = np.vstack((history_xy, future_gt[:, :2]))
            gt_yaw_raw = np.concatenate((history_yaw, np.full((future_gt.shape[0],), np.nan, dtype=np.float32)))
            gt_yaw = _path_yaw(gt, raw_yaw=gt_yaw_raw, default_yaw=default_yaw)
            gt_frames = history_frames + [int(t + i + 1) for i in range(future_gt.shape[0])]
            for point_idx, point in enumerate(gt):
                rows.append(
                    {
                        "agent": rec["agent_index"],
                        "sample": 0,
                        "frame": int(gt_frames[point_idx]),
                        "x": float(point[0]),
                        "y": float(point[1]),
                        "yaw": float(gt_yaw[point_idx]),
                    }
                )

        dyn = rec.get("dynamics")
        if dyn is None:
            continue
        pos = np.asarray(dyn["position_world"], dtype=np.float32)
        dyn_yaw = np.asarray(dyn.get("yaw", np.full(pos.shape[:2], np.nan)), dtype=np.float32)
        for sample_idx, pred_future in enumerate(pos, start=1):
            pred = np.vstack((history_xy, pred_future[:, :2]))
            future_yaw = dyn_yaw[sample_idx - 1] if dyn_yaw.ndim == 2 and sample_idx - 1 < dyn_yaw.shape[0] else np.full((pred_future.shape[0],), np.nan, dtype=np.float32)
            pred_yaw_raw = np.concatenate((history_yaw, future_yaw.reshape(-1)[: pred_future.shape[0]]))
            pred_yaw = _path_yaw(pred, raw_yaw=pred_yaw_raw, default_yaw=default_yaw)
            pred_frames = history_frames + [int(t + i + 1) for i in range(pred_future.shape[0])]
            for point_idx, point in enumerate(pred):
                rows.append(
                    {
                        "agent": rec["agent_index"],
                        "sample": int(sample_idx),
                        "frame": int(pred_frames[point_idx]),
                        "x": float(point[0]),
                        "y": float(point[1]),
                        "yaw": float(pred_yaw[point_idx]),
                    }
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def _generate_records(
    *,
    model,
    env: Environment,
    scene: Scene,
    nodes_at_t: Sequence[Node],
    labels: Dict[Node, str],
    config: EasyDict,
    hyperparams: dict,
    t: int,
    center: np.ndarray,
) -> list:
    ph = int(hyperparams["prediction_horizon"])
    max_hl = int(hyperparams["maximum_history_length"])
    num_samples = int(getattr(config, "viz_num_samples", 1))
    sampling = str(getattr(config, "sampling", "ddpm"))
    flexibility = float(getattr(config, "viz_sampling_flexibility", 0.0))
    predict_agents = int(getattr(config, "viz_predict_agents", 2))
    predict_target_only = predict_agents == 1
    ego_node = _resolve_ego_node(scene)

    records = []
    ego_reference_future_model = None
    scene_graph = scene.get_scene_graph(
        t,
        env.attention_radius,
        hyperparams["edge_addition_filter"],
        hyperparams["edge_removal_filter"],
    )

    for agent_index, node in enumerate(nodes_at_t):
        role = labels.get(node, f"ID {node.id}")
        history_state = _state_history(node, t, max_hl, hyperparams["state"][node.type], center)
        future_gt = _future_xy(node, t, ph, center)

        rec = {
            "agent_index": int(agent_index),
            "agent_id": str(node.id),
            "role": role,
            "node": node,
            "history_state": history_state,
            "future_gt": future_gt,
            "dynamics": None,
            "source": "diffusion",
        }

        is_ego = node is ego_node
        if predict_target_only and is_ego:
            if future_gt.shape[0] > 0:
                ego_reference_future_model = future_gt.copy()
                ego_reference_future_model[:, 0] -= float(center[0])
                ego_reference_future_model[:, 1] -= float(center[1])
            records.append(rec)
            continue

        edge_types = [edge_type for edge_type in env.get_edge_types() if edge_type[0] == node.type]
        item = get_node_timestep_data(
            env,
            scene,
            t,
            node,
            hyperparams["state"],
            hyperparams["pred_state"],
            edge_types,
            max_hl,
            ph,
            hyperparams,
            scene_graph=scene_graph,
        )
        batch = collate([item])
        guidance_override = None
        if not is_ego:
            guidance_override = _merge_guidance_overrides(
                _build_collision_guidance_override(config, ego_reference_future_model),
                _build_heading_guidance_override(config, history_state),
            )
        if (not is_ego) and ego_reference_future_model is not None:
            rec["reference_future_model"] = np.asarray(ego_reference_future_model, dtype=np.float32).copy()

        with torch.no_grad():
            pred_pack = model.generate(
                batch,
                node.type,
                num_points=ph,
                sample=num_samples,
                bestof=True,
                flexibility=flexibility,
                sampling=sampling,
                step=1,
                return_dynamics=True,
                guidance_override=guidance_override,
            )

        pos_model = np.asarray(pred_pack["position"][:, 0], dtype=np.float32)
        pos_world = pos_model.copy()
        pos_world[:, :, 0] += float(center[0])
        pos_world[:, :, 1] += float(center[1])
        rec["dynamics"] = {
            "position": pos_model,
            "position_world": pos_world,
            "velocity": np.asarray(pred_pack["velocity"][:, 0], dtype=np.float32),
            "speed": np.asarray(pred_pack["speed"][:, 0], dtype=np.float32),
            "yaw": np.asarray(pred_pack["yaw"][:, 0], dtype=np.float32),
        }
        records.append(rec)

        if is_ego and pos_model.shape[0] > 0:
            ego_reference_future_model = pos_model[0]

    return records

