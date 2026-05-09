import argparse
import csv
import logging
import math
import re
import shutil
import sys
import types
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.patches import Polygon

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


USER_DEFAULTS = {
    "config": "configs/run.yaml",
}

COLOR_SETS = [
    {"hist": "#011f4b", "gt": "#005b96", "pred": "#2f6f95", "curr": "#011f4b"},
    {"hist": "#4d0000", "gt": "#990000", "pred": "#b03030", "curr": "#4d0000"},
    {"hist": "#003300", "gt": "#1a8c1a", "pred": "#3f8f3f", "curr": "#003300"},
    {"hist": "#2a0a4d", "gt": "#6a3d9a", "pred": "#7d52b3", "curr": "#2a0a4d"},
    {"hist": "#4d1a00", "gt": "#cc5200", "pred": "#d9872e", "curr": "#4d1a00"},
    {"hist": "#003333", "gt": "#008080", "pred": "#3f9f9f", "curr": "#003333"},
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize MAT checkpoint predictions using settings from YAML."
    )
    parser.add_argument("--config", default=USER_DEFAULTS["config"], help="Path to YAML config (default: configs/run.yaml)")
    return parser.parse_args()


def _load_yaml_config(config_path: Path):
    last_error = None
    for encoding in ("utf-8", "utf-8-sig", "cp949", "euc-kr"):
        try:
            with open(config_path, "r", encoding=encoding) as f:
                cfg = yaml.safe_load(f)
            break
        except UnicodeDecodeError as exc:
            last_error = exc
    else:
        raise last_error
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML content: {config_path}")
    return cfg


def _to_optional_int(value, key_name: str):
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception as e:
        raise ValueError(f"'{key_name}' must be an integer or null, got: {value}") from e


def _build_viz_runtime(config_path: Path, cfg: dict, output_mode: str = "direct"):
    runtime = EasyDict()
    runtime.config_path = str(config_path)
    runtime.ckpt_dir = str(cfg.get("viz_ckpt_dir", cfg.get("viz_ckpt_dir_pkl", ""))).strip()
    runtime.output_dir = str(cfg.get("viz_output_dir", cfg.get("viz_output_dir_pkl", "mat_run")))
    runtime.ckpt_name = str(cfg.get("viz_ckpt_name", "")).strip()
    runtime.seed = int(cfg.get("viz_seed", cfg.get("seed", 123)))
    runtime.viz_epoch_tag = _to_optional_int(cfg.get("viz_epoch_tag", None), "viz_epoch_tag")
    runtime.dataset = str(cfg.get("dataset", "")).strip()

    if runtime.ckpt_dir == "":
        raise ValueError("Set 'viz_ckpt_dir' in YAML.")
    if runtime.ckpt_name == "":
        raise ValueError("Set 'viz_ckpt_name' in YAML.")

    ckpt_name_path = Path(runtime.ckpt_name)
    if ckpt_name_path.is_absolute() or ckpt_name_path.name != runtime.ckpt_name:
        raise ValueError(
            "'viz_ckpt_name' must be a filename only (no path). "
            "Use 'viz_ckpt_dir' for folder and 'viz_ckpt_name' for file name."
        )
    return runtime


def _ensure_output_dir(output_dir: Path):
    if output_dir.exists():
        return output_dir.resolve(), None
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir.resolve(), f"[Info] output_dir did not exist. Created: {output_dir.resolve()}"


def _extract_dataset_epoch_from_name(ckpt_name: str):
    m = re.match(r"^(?P<dataset>.+)_epoch(?P<epoch>\d+)\.pt$", ckpt_name)
    if m is None:
        return None, None
    return m.group("dataset"), int(m.group("epoch"))


def _resolve_checkpoint(runtime):
    ckpt_dir = Path(runtime.ckpt_dir).resolve()
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"viz_ckpt_dir does not exist: {ckpt_dir}")
    runtime.ckpt_dir = str(ckpt_dir)

    ckpt_path = (ckpt_dir / runtime.ckpt_name).resolve()
    if not ckpt_path.exists():
        available = sorted([p.name for p in ckpt_dir.glob("*.pt")])
        sample = ", ".join(available[:8]) if available else "(none)"
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path} "
            f"(viz_ckpt_dir={runtime.ckpt_dir}, viz_ckpt_name={runtime.ckpt_name}). "
            f"Available in folder: {sample}"
        )
    if ckpt_path.suffix.lower() != ".pt":
        raise ValueError(f"viz_ckpt_name must point to a .pt file, got: {runtime.ckpt_name}")
    return ckpt_path


def _build_config(base_cfg: dict, runtime, ckpt_path: Path):
    cfg = dict(base_cfg)
    output_root = Path(runtime.output_dir)
    output_root, msg = _ensure_output_dir(output_root)
    if msg:
        print(msg)
    try_dir = _allocate_try_dir(output_root.resolve())
    # MID builds model_dir via os.path.join('./experiments', exp_name).
    # Using an absolute exp_name makes model_dir resolve to the absolute output_dir path.
    exp_name = str(try_dir)

    ckpt_dataset, ckpt_epoch = _extract_dataset_epoch_from_name(ckpt_path.name)
    if ckpt_epoch is None:
        raise ValueError(
            f"Checkpoint filename must match '<dataset>_epoch<number>.pt'. Got: {ckpt_path.name}"
        )

    dataset_name = runtime.dataset if runtime.dataset else ckpt_dataset
    if not dataset_name:
        raise ValueError("Could not infer dataset name. Set 'dataset' in YAML or use '<dataset>_epochN.pt' checkpoint name.")

    cfg["config"] = runtime.config_path
    cfg["exp_name"] = exp_name
    cfg["dataset"] = dataset_name
    cfg["eval_mode"] = True
    cfg["eval_at"] = int(ckpt_epoch)

    return EasyDict(cfg), int(ckpt_epoch), try_dir


def _allocate_try_dir(run_root_dir: Path) -> Path:
    max_try_idx = 0
    for p in run_root_dir.iterdir():
        if not p.is_dir():
            continue
        m = re.fullmatch(r"try_(\d+)", p.name)
        if m is None:
            continue
        max_try_idx = max(max_try_idx, int(m.group(1)))
    next_try_idx = max_try_idx + 1
    try_dir = run_root_dir / f"try_{next_try_idx}"
    try_dir.mkdir(parents=True, exist_ok=False)
    return try_dir.resolve()


def _build_mid_expected_ckpt_path(model_dir: Path, dataset: str, epoch: int) -> Path:
    return (model_dir / f"{dataset}_epoch{int(epoch)}.pt").resolve()


def _build_torch_load_redirect(expected_ckpt_path: Path, selected_ckpt_path: Path):
    original_torch_load = torch.load
    expected_resolved = expected_ckpt_path.resolve()
    selected_resolved = selected_ckpt_path.resolve()

    def _torch_load_redirect(path_or_obj, *args, **kwargs):
        try:
            if isinstance(path_or_obj, (str, Path)):
                requested = Path(path_or_obj).resolve()
                if requested == expected_resolved and requested != selected_resolved:
                    return original_torch_load(str(selected_resolved), *args, **kwargs)
        except Exception:
            pass
        return original_torch_load(path_or_obj, *args, **kwargs)

    return original_torch_load, _torch_load_redirect


def _install_tensorboard_noop():
    class _SummaryWriter:
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def add_histogram(self, *args, **kwargs):
            pass

        def close(self):
            pass

    tbx = types.ModuleType("tensorboardX")
    tbx.SummaryWriter = _SummaryWriter
    sys.modules["tensorboardX"] = tbx


def _build_noop_filehandler_patch():
    original_file_handler = logging.FileHandler

    class _NoopFileHandler(logging.Handler):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def emit(self, record):
            pass

    return original_file_handler, _NoopFileHandler


def _cleanup_mid_viz_dirs(model_dir: Path):
    try:
        shutil.rmtree(str(model_dir / "viz_outputs"), ignore_errors=True)
    except Exception:
        pass


def _compute_yaw_from_positions(traj, default_yaw=0.0, speed_eps=1e-4):
    traj = np.asarray(traj, dtype=np.float32)
    if traj.ndim != 2 or traj.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    n = int(traj.shape[0])
    yaws = np.full((n,), float(default_yaw), dtype=np.float32)
    if n == 1:
        return yaws

    deltas = np.diff(traj, axis=0)
    speeds = np.linalg.norm(deltas, axis=1)
    prev_yaw = float(default_yaw)
    for i in range(n - 1):
        if speeds[i] > speed_eps:
            prev_yaw = float(np.arctan2(deltas[i, 1], deltas[i, 0]))
        yaws[i] = prev_yaw
    yaws[-1] = prev_yaw
    return yaws


def _resolve_path_yaws(traj, yaw_values=None, default_yaw=0.0):
    yaws = _compute_yaw_from_positions(traj, default_yaw=default_yaw)
    if yaw_values is None:
        return yaws
    raw = np.asarray(yaw_values, dtype=np.float32).reshape(-1)
    if raw.size == 0 or yaws.size == 0:
        return yaws
    limit = min(int(raw.size), int(yaws.size))
    valid = np.isfinite(raw[:limit])
    valid_idx = np.nonzero(valid)[0]
    yaws[valid_idx] = raw[:limit][valid_idx]
    return yaws


def _draw_vehicle_box(ax, center_xy, yaw, car_length, car_width, face_color, face_alpha, zorder=2.2):
    half_l = 0.5 * float(car_length)
    half_w = 0.5 * float(car_width)
    local = np.array(
        [
            [half_l, half_w],
            [half_l, -half_w],
            [-half_l, -half_w],
            [-half_l, half_w],
        ],
        dtype=np.float32,
    )
    c = math.cos(float(yaw))
    s = math.sin(float(yaw))
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    center_xy = np.asarray(center_xy, dtype=np.float32).reshape(2)
    world = local @ rot.T + center_xy
    patch = Polygon(
        world,
        closed=True,
        facecolor=face_color,
        edgecolor=(0.15, 0.15, 0.15, min(1.0, face_alpha + 0.15)),
        linewidth=0.8,
        alpha=float(face_alpha),
        zorder=float(zorder),
    )
    ax.add_patch(patch)


def _lighten_color(color, amount=0.5):
    rgb = np.asarray(mcolors.to_rgb(color), dtype=np.float32)
    amount = float(np.clip(amount, 0.0, 1.0))
    out = rgb * (1.0 - amount) + amount
    return tuple(np.clip(out, 0.0, 1.0).tolist())


def _load_scene_csv(csv_path):
    groups = {}

    def _parse_optional_float(row, key, default=float("nan")):
        if key not in row:
            return default
        value = row.get(key, "")
        if value is None or str(value).strip() == "":
            return default
        try:
            return float(value)
        except Exception:
            return default

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        simple_format = {"agent", "sample", "frame", "x", "y"}.issubset(fieldnames)
        for row in reader:
            if simple_format:
                agent_index = int(row["agent"])
                agent_id = str(agent_index)
                role = "Ego" if agent_index == 0 else ("Target" if agent_index == 1 else f"Opponent {agent_index}")
                sample_idx = int(row["sample"])
                traj_type = "input_plus_gt" if sample_idx == 0 else "input_plus_pred"
                point_idx = int(row["frame"])
                yaw = _parse_optional_float(row, "yaw")
            else:
                agent_index = int(row["agent_index"])
                agent_id = str(row["agent_id"])
                role = str(row["role"])
                traj_type = str(row["trajectory_type"])
                sample_idx = int(row["sample_idx"])
                point_idx = int(row["point_idx"])
                yaw = _parse_optional_float(row, "yaw")
            x = float(row["x"])
            y = float(row["y"])
            key = (agent_index, agent_id, role, traj_type, sample_idx)
            if key not in groups:
                groups[key] = []
            groups[key].append((point_idx, x, y, yaw))

    agents = {}
    for key, pts in groups.items():
        agent_index, agent_id, role, traj_type, sample_idx = key
        pts_sorted = sorted(pts, key=lambda x: x[0])
        frame_arr = np.asarray([p[0] for p in pts_sorted], dtype=np.int32)
        arr = np.asarray([[p[1], p[2]] for p in pts_sorted], dtype=np.float32)
        yaw_arr = np.asarray([p[3] for p in pts_sorted], dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] != 2:
            continue
        if agent_index not in agents:
            agents[agent_index] = {
                "agent_index": int(agent_index),
                "agent_id": agent_id,
                "role": role,
                "gt": None,
                "gt_yaw": None,
                "gt_frame": None,
                "pred": {},
                "pred_yaw": {},
                "pred_frame": {},
            }
        if traj_type == "input_plus_gt" and sample_idx == 0:
            agents[agent_index]["gt"] = arr
            agents[agent_index]["gt_yaw"] = yaw_arr
            agents[agent_index]["gt_frame"] = frame_arr
        elif traj_type == "input_plus_pred" and sample_idx > 0:
            agents[agent_index]["pred"][sample_idx] = arr
            agents[agent_index]["pred_yaw"][sample_idx] = yaw_arr
            agents[agent_index]["pred_frame"][sample_idx] = frame_arr

    out = []
    for agent_index in sorted(agents.keys()):
        rec = agents[agent_index]
        if rec["gt"] is None:
            continue
        all_paths = [rec["gt"]] + [rec["pred"][k] for k in sorted(rec["pred"].keys())]
        rec["history_len"] = _infer_common_history_len(all_paths)
        out.append(rec)
    return out


def _infer_common_history_len(paths, tol=1e-6):
    if len(paths) == 0:
        return 1
    min_len = min(int(p.shape[0]) for p in paths)
    if min_len <= 1:
        return 1
    hist_len = 1
    for i in range(min_len):
        ref = paths[0][i]
        ok = True
        for p in paths[1:]:
            if not np.allclose(ref, p[i], atol=tol, rtol=0.0):
                ok = False
                break
        if ok:
            hist_len = i + 1
        else:
            break
    return max(1, hist_len)


def _build_axis_limits(agent_records, pred_sample_idx):
    all_points = []
    for rec in agent_records:
        gt = rec["gt"]
        all_points.append(gt)
        if pred_sample_idx == "all":
            for _, pred in sorted(rec["pred"].items()):
                all_points.append(pred)
        elif pred_sample_idx is not None and pred_sample_idx in rec["pred"]:
            all_points.append(rec["pred"][pred_sample_idx])
    pts = np.vstack(all_points)
    x_min, y_min = np.min(pts, axis=0)
    x_max, y_max = np.max(pts, axis=0)
    center_x = (x_max + x_min) / 2.0
    center_y = (y_max + y_min) / 2.0
    max_range = max(15.0, x_max - x_min, y_max - y_min)
    margin = max_range * 0.55
    return (center_x - margin, center_x + margin, center_y - margin, center_y + margin)


def _video_stem(csv_path, pred_sample_idx):
    base = csv_path.stem
    if pred_sample_idx == "all":
        return f"{base}"
    if pred_sample_idx is None:
        return f"{base}_gt"
    return f"{base}_pred_s{int(pred_sample_idx):03d}"


def _render_scene_video(
    csv_path,
    out_dir,
    agent_records,
    pred_sample_idx,
    fps,
    dpi,
    car_width,
    car_length,
    end_hold_sec,
    bbox_alpha,
    fmt,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    video_stem = _video_stem(csv_path, pred_sample_idx)
    x0, x1, y0, y1 = _build_axis_limits(agent_records, pred_sample_idx)

    max_frames = 1
    for rec in agent_records:
        n = rec["gt"].shape[0]
        if pred_sample_idx == "all":
            for _, pred_arr in rec["pred"].items():
                n = max(n, pred_arr.shape[0])
        elif pred_sample_idx is not None and pred_sample_idx in rec["pred"]:
            n = max(n, rec["pred"][pred_sample_idx].shape[0])
        max_frames = max(max_frames, int(n))
    hold_frames = max(0, int(round(float(end_hold_sec) * max(1, int(fps)))))
    total_frames = max_frames + hold_frames

    fig, ax = plt.subplots(figsize=(10, 10))
    pred_enabled = pred_sample_idx is not None
    draw_all_preds = (pred_sample_idx == "all")
    gt_box_alpha = 1.0
    pred_box_alpha = float(np.clip(max(0.45, bbox_alpha * 2.0), 0.40, 0.70))
    pred_linewidth = 1.25
    pred_markersize = 1.6
    pred_box_lighten = 0.35

    def _draw_frame(frame_idx):
        ax.clear()
        traj_frame_idx = min(int(frame_idx), max_frames - 1)

        for rec in agent_records:
            agent_index = rec["agent_index"]
            colors = COLOR_SETS[agent_index % len(COLOR_SETS)]
            role_lower = str(rec.get("role", "")).strip().lower()
            is_ego = ("ego" in role_lower)
            is_target = ("target" in role_lower)
            if is_target:
                traj_z = 60.0
                pred_box_z = 40.0
                gt_box_z = 20.0
            elif is_ego:
                traj_z = 50.0
                pred_box_z = 30.0
                gt_box_z = 10.0
            else:
                traj_z = 45.0
                pred_box_z = 25.0
                gt_box_z = 15.0
            hist_len = int(rec["history_len"])
            gt = rec["gt"]
            pred = rec["pred"].get(pred_sample_idx, None) if (pred_enabled and (not draw_all_preds)) else None

            hist_end = min(traj_frame_idx, max(0, hist_len - 1), gt.shape[0] - 1)
            if hist_end >= 1:
                hseg = gt[: hist_end + 1]
                ax.plot(
                    hseg[:, 0],
                    hseg[:, 1],
                    "-o",
                    color=colors["hist"],
                    linewidth=3.2,
                    markersize=3,
                    zorder=traj_z + 0.20,
                )

            gt_start = max(0, hist_len - 1)
            if traj_frame_idx >= gt_start and gt.shape[0] >= 2:
                gt_end = min(traj_frame_idx, gt.shape[0] - 1)
                if gt_end > gt_start:
                    gt_seg = gt[gt_start: gt_end + 1]
                    ax.plot(
                        gt_seg[:, 0],
                        gt_seg[:, 1],
                        "-o",
                        color=colors["hist"],
                        linewidth=3.2,
                        markersize=3,
                        alpha=1.0,
                        zorder=traj_z + 0.05,
                    )

            if draw_all_preds and traj_frame_idx >= gt_start:
                pred_items = sorted(rec["pred"].items(), key=lambda x: x[0])
                if len(pred_items) > 0:
                    for _, pred_arr in pred_items:
                        if pred_arr.shape[0] == 0:
                            continue
                        pred_end = min(traj_frame_idx, pred_arr.shape[0] - 1)
                        if pred_end > gt_start:
                            pred_seg = pred_arr[gt_start: pred_end + 1]
                            ax.plot(
                                pred_seg[:, 0],
                                pred_seg[:, 1],
                                "-o",
                                color=colors["gt"],
                                alpha=0.96,
                                linewidth=pred_linewidth,
                                markersize=pred_markersize,
                                zorder=traj_z + 0.10,
                            )
            elif pred is not None and pred.shape[0] > 0 and traj_frame_idx >= gt_start:
                pred_end = min(traj_frame_idx, pred.shape[0] - 1)
                if pred_end > gt_start:
                    pred_seg = pred[gt_start: pred_end + 1]
                    ax.plot(
                        pred_seg[:, 0],
                        pred_seg[:, 1],
                        "-o",
                        color=colors["gt"],
                        alpha=0.96,
                        linewidth=pred_linewidth,
                        markersize=pred_markersize,
                        zorder=traj_z + 0.10,
                    )

            curr_idx = min(traj_frame_idx, gt.shape[0] - 1)
            curr = gt[curr_idx]
            ax.scatter(
                [curr[0]],
                [curr[1]],
                color=colors["curr"],
                s=60,
                edgecolors="white",
                linewidths=1.5,
                zorder=traj_z + 0.30,
            )
            ax.text(
                float(curr[0]) + 2.0,
                float(curr[1]) + 1.0,
                rec["role"],
                color=colors["hist"],
                fontsize=9,
                fontweight="bold",
                zorder=traj_z + 0.35,
            )

            if draw_all_preds:
                for pred_sample_idx_inner, pred_arr in sorted(rec["pred"].items(), key=lambda x: x[0]):
                    if pred_arr.shape[0] == 0:
                        continue
                    pidx = min(traj_frame_idx, pred_arr.shape[0] - 1)
                    pyaws = _resolve_path_yaws(pred_arr, rec.get("pred_yaw", {}).get(pred_sample_idx_inner))
                    _draw_vehicle_box(
                        ax=ax,
                        center_xy=pred_arr[pidx],
                        yaw=float(pyaws[pidx]) if pyaws.shape[0] > 0 else 0.0,
                        car_length=car_length,
                        car_width=car_width,
                        face_color=_lighten_color(colors["gt"], amount=pred_box_lighten),
                        face_alpha=pred_box_alpha,
                        zorder=pred_box_z,
                    )
            elif pred is not None and pred.shape[0] > 0:
                pidx = min(traj_frame_idx, pred.shape[0] - 1)
                pyaws = _resolve_path_yaws(pred, rec.get("pred_yaw", {}).get(pred_sample_idx))
                _draw_vehicle_box(
                    ax=ax,
                    center_xy=pred[pidx],
                    yaw=float(pyaws[pidx]) if pyaws.shape[0] > 0 else 0.0,
                    car_length=car_length,
                    car_width=car_width,
                    face_color=_lighten_color(colors["gt"], amount=pred_box_lighten),
                    face_alpha=pred_box_alpha,
                    zorder=pred_box_z,
                )

            gt_yaws = _resolve_path_yaws(gt, rec.get("gt_yaw"))
            _draw_vehicle_box(
                ax=ax,
                center_xy=curr,
                yaw=float(gt_yaws[curr_idx]) if gt_yaws.shape[0] > 0 else 0.0,
                car_length=car_length,
                car_width=car_width,
                face_color=colors["pred"],
                face_alpha=gt_box_alpha,
                zorder=gt_box_z,
            )

        import matplotlib.lines as mlines
        hist_line = mlines.Line2D([], [], color="black", linestyle="-", marker="o", markersize=4, linewidth=3.2, label="History")
        gt_line = mlines.Line2D([], [], color="black", linestyle="-", marker="o", markersize=4, linewidth=3.2, label="GT Future")
        pred_line = mlines.Line2D([], [], color="slategray", linestyle="-", marker="o", markersize=2.8, linewidth=1.55, label="Predictions (-o)")
        curr_point = mlines.Line2D([], [], color="white", marker="o", markerfacecolor="black", markersize=8, label="Current Pos")
        ax.legend(handles=[hist_line, gt_line, pred_line, curr_point], loc="best")

        mode_text = "GT + PRED (all)" if pred_sample_idx == "all" else "GT"
        title_frame = min(int(frame_idx) + 1, max_frames)
        ax.set_title(f"{csv_path.stem} | {mode_text} | frame {title_frame}/{max_frames}")
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.3)

    ani = FuncAnimation(fig, _draw_frame, frames=total_frames, interval=1000.0 / max(1, fps), blit=False, repeat=False)

    if fmt == "gif":
        out_path = (out_dir / video_stem).with_suffix(".gif")
        writer = PillowWriter(fps=max(1, fps))
    elif fmt == "mp4":
        out_path = (out_dir / video_stem).with_suffix(".mp4")
        writer = FFMpegWriter(fps=max(1, fps), bitrate=2200)
    else:
        if FFMpegWriter.isAvailable():
            out_path = (out_dir / video_stem).with_suffix(".mp4")
            writer = FFMpegWriter(fps=max(1, fps), bitrate=2200)
        else:
            out_path = (out_dir / video_stem).with_suffix(".gif")
            writer = PillowWriter(fps=max(1, fps))

    ani.save(str(out_path), writer=writer, dpi=max(80, int(dpi)))
    plt.close(fig)
    return out_path


def _render_videos_from_csv(csv_dir: Path, vids_dir: Path, config) -> int:
    """
    Render one video per CSV scene.
    """
    csv_dir = csv_dir.resolve()
    vids_dir = vids_dir.resolve()
    vids_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted([p.resolve() for p in csv_dir.glob("*.csv") if p.is_file()])
    if len(csv_files) == 0:
        print(f"[Warn] No CSV files found for video rendering: {csv_dir}")
        return 0

    fps = int(config.get("viz_video_fps", 10))
    dpi = int(config.get("viz_video_dpi", 140))
    end_hold_sec = float(config.get("viz_video_end_hold_sec", 1.5))
    fmt = str(config.get("viz_video_format", "auto"))
    bbox_alpha = float(config.get("viz_video_bbox_alpha", 0.22))
    car_width = float(config.get("car_width", 1.825))
    car_length = float(config.get("car_length", 4.650))

    saved = 0
    for csv_path in csv_files:
        agent_records = _load_scene_csv(csv_path)
        if len(agent_records) == 0:
            print(f"[Skip] No valid trajectory rows: {csv_path}")
            continue
        out_path = _render_scene_video(
            csv_path=csv_path,
            out_dir=vids_dir,
            agent_records=agent_records,
            pred_sample_idx="all",
            fps=fps,
            dpi=dpi,
            car_width=car_width,
            car_length=car_length,
            end_hold_sec=end_hold_sec,
            bbox_alpha=bbox_alpha,
            fmt=fmt,
        )
        saved += 1
        print(f"[Saved] {out_path}")

    return saved


def main():
    args = parse_args()
    required_run_path = Path(USER_DEFAULTS["config"]).resolve()
    if not required_run_path.exists():
        raise FileNotFoundError(
            f"Required run config not found: {required_run_path}. "
            "Create configs/run.yaml before running mat_run.py."
        )

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    yaml_cfg = _load_yaml_config(config_path)
    runtime = _build_viz_runtime(config_path, yaml_cfg, output_mode="pkl")

    ckpt_path = _resolve_checkpoint(runtime)

    config, ckpt_epoch, output_dir = _build_config(yaml_cfg, runtime, ckpt_path)

    # Reproducible random scene/timestep selection in visualization.
    np.random.seed(runtime.seed)
    torch.manual_seed(runtime.seed)

    print(f"[Info] Config: {config_path}")
    print(f"[Info] Checkpoint: {ckpt_path}")
    print(f"[Info] Dataset: {config.dataset}")
    print(f"[Info] Checkpoint dir: {runtime.ckpt_dir}")
    print(f"[Info] Output dir: {output_dir}")
    expected_mid_ckpt = _build_mid_expected_ckpt_path(output_dir, config.dataset, ckpt_epoch)
    csv_dir = (output_dir / "csv").resolve()
    vids_dir = (output_dir / "vids").resolve()
    csv_dir.mkdir(parents=True, exist_ok=True)
    vids_dir.mkdir(parents=True, exist_ok=True)
    config["viz_csv_dir"] = str(csv_dir)
    config["viz_save_images"] = False

    # Disable tensorboard/event logs in visualization-only runs.
    _install_tensorboard_noop()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in the current Python environment. "
            "Please run mat_run.py in the same GPU environment used for training (e.g., csim)."
        )

    from mid import MID

    original_torch_load, patched_torch_load = _build_torch_load_redirect(expected_mid_ckpt, ckpt_path)
    original_file_handler, noop_file_handler = _build_noop_filehandler_patch()
    torch.load = patched_torch_load
    logging.FileHandler = noop_file_handler
    try:
        agent = MID(config)
    finally:
        torch.load = original_torch_load
        logging.FileHandler = original_file_handler

    viz_epoch = int(runtime.viz_epoch_tag) if runtime.viz_epoch_tag is not None else int(ckpt_epoch)

    out_dir = output_dir
    video_count = 0
    try:
        print("[Info] Trajectory CSV generation started...")
        agent._visualize_epoch(viz_epoch)
        print("[Info] CSV generation finished. Rendering videos from CSV...")
        video_count = _render_videos_from_csv(csv_dir=csv_dir, vids_dir=vids_dir, config=config)
        print(f"[Info] Video rendering finished. videos={video_count}")
    finally:
        _cleanup_mid_viz_dirs(Path(agent.model_dir))

    print(f"[Done] Outputs saved to: {out_dir} (csv_dir={csv_dir}, videos={video_count})")


if __name__ == "__main__":
    main()
