import argparse
import logging
import re
import shutil
import sys
import types
from datetime import datetime
from pathlib import Path

import numpy as np
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


USER_DEFAULTS = {
    "config": "configs/run.yaml",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize MAT checkpoint predictions using settings from YAML."
    )
    parser.add_argument("--config", default=USER_DEFAULTS["config"], help="Path to YAML config (default: configs/run.yaml)")
    return parser.parse_args()


def _load_yaml_config(config_path: Path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
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


def _build_viz_runtime(config_path: Path, cfg: dict):
    runtime = EasyDict()
    runtime.config_path = str(config_path)
    runtime.ckpt_dir = str(cfg.get("viz_ckpt_dir", "")).strip()
    runtime.ckpt_name = str(cfg.get("viz_ckpt_name", "")).strip()
    runtime.output_dir = str(cfg.get("viz_output_dir", "results/mat_viz"))
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
    run_root_dir = (output_root / ckpt_path.stem).resolve()
    run_root_dir.mkdir(parents=True, exist_ok=True)
    try_dir = _allocate_try_dir(run_root_dir)
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


def _make_unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    idx = 1
    while True:
        candidate = path.with_name(f"{stem}_{idx}{suffix}")
        if not candidate.exists():
            return candidate
        idx += 1


def _build_savefig_redirect(target_dir: Path):
    from matplotlib.figure import Figure

    original_savefig = Figure.savefig
    target_dir = target_dir.resolve()

    def _savefig_redirect(fig, fname, *args, **kwargs):
        redirected = fname
        try:
            if isinstance(fname, (str, Path)):
                src_path = Path(fname)
                if src_path.suffix.lower() == ".png":
                    dst_path = _make_unique_path(target_dir / src_path.name)
                    redirected = str(dst_path)
        except Exception:
            pass
        return original_savefig(fig, redirected, *args, **kwargs)

    return Figure, original_savefig, _savefig_redirect


def _cleanup_mid_viz_dirs(model_dir: Path):
    try:
        shutil.rmtree(str(model_dir / "viz_outputs"), ignore_errors=True)
    except Exception:
        pass


def _collect_prefixed_items(cfg, prefix: str):
    items = []
    for k in sorted(cfg.keys()):
        if str(k).startswith(prefix):
            items.append((str(k), cfg[k]))
    return items


def _write_try_param_log(
    try_dir: Path,
    runtime,
    config_path: Path,
    ckpt_path: Path,
    config,
    ckpt_epoch: int,
    viz_epoch: int,
    status: str,
    image_count: int = 0,
    error_message: str = "",
):
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append(f"generated_at: {now_str}")
    lines.append(f"status: {status}")
    lines.append(f"image_count: {int(image_count)}")
    if error_message:
        lines.append(f"error: {error_message}")
    lines.append("")

    lines.append("[paths]")
    lines.append(f"config_path: {Path(config_path).resolve()}")
    lines.append(f"checkpoint_path: {Path(ckpt_path).resolve()}")
    lines.append(f"viz_ckpt_dir: {runtime.ckpt_dir}")
    lines.append(f"viz_ckpt_name: {runtime.ckpt_name}")
    lines.append(f"try_dir: {try_dir.resolve()}")
    lines.append("")

    lines.append("[core]")
    lines.append(f"dataset: {config.get('dataset', '')}")
    lines.append(f"checkpoint_epoch: {int(ckpt_epoch)}")
    lines.append(f"visualized_epoch: {int(viz_epoch)}")
    lines.append(f"viz_seed: {runtime.seed}")
    lines.append(f"sampling: {config.get('sampling', 'ddpm')}")
    lines.append(f"sampling_xt_temperature: {config.get('sampling_xt_temperature', '')}")
    lines.append(f"viz_sampling_flexibility: {config.get('viz_sampling_flexibility', '')}")
    lines.append(f"viz_predict_agents: {config.get('viz_predict_agents', '')}")
    lines.append(f"viz_num_examples: {config.get('viz_num_examples', '')}")
    lines.append(f"viz_num_samples: {config.get('viz_num_samples', '')}")
    lines.append(f"car_width: {config.get('car_width', '')}")
    lines.append(f"car_length: {config.get('car_length', '')}")
    lines.append(f"data_dt: {config.get('data_dt', '')}")
    lines.append(f"history_sec: {config.get('history_sec', '')}")
    lines.append(f"prediction_sec: {config.get('prediction_sec', '')}")
    lines.append(f"tf_layer: {config.get('tf_layer', '')}")
    lines.append(f"encoder_dim: {config.get('encoder_dim', '')}")
    lines.append(f"diffnet: {config.get('diffnet', '')}")
    lines.append("")

    pref_sections = [
        "dynamics_guidance_",
        "collision_guidance_",
        "not_collision_guidance_",
        "viz_vehicle_box_",
        "yaw_loss_",
        "bicycle_rollout_",
    ]
    for pref in pref_sections:
        items = _collect_prefixed_items(config, pref)
        lines.append(f"[{pref}*]")
        if len(items) == 0:
            lines.append("(none)")
        else:
            for k, v in items:
                lines.append(f"{k}: {v}")
        lines.append("")

    lines.append("[full_effective_config]")
    for k in sorted(config.keys()):
        lines.append(f"{k}: {config[k]}")
    lines.append("")

    out_path = try_dir / "run_params.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_path


def main():
    args = parse_args()
    required_run_path = Path(USER_DEFAULTS["config"]).resolve()
    if not required_run_path.exists():
        raise FileNotFoundError(
            f"Required run config not found: {required_run_path}. "
            "Create configs/run.yaml before running mat_viz.py."
        )

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    yaml_cfg = _load_yaml_config(config_path)
    runtime = _build_viz_runtime(config_path, yaml_cfg)

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
    img_dir = (output_dir / "img").resolve()
    csv_dir = (output_dir / "csv").resolve()
    img_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    config["viz_csv_dir"] = str(csv_dir)

    # Disable tensorboard/event logs in visualization-only runs.
    _install_tensorboard_noop()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in the current Python environment. "
            "Please run mat_viz.py in the same GPU environment used for training (e.g., csim)."
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
    _write_try_param_log(
        try_dir=output_dir,
        runtime=runtime,
        config_path=config_path,
        ckpt_path=ckpt_path,
        config=config,
        ckpt_epoch=ckpt_epoch,
        viz_epoch=viz_epoch,
        status="running",
        image_count=0,
        error_message="",
    )

    out_dir = output_dir
    run_error = None
    figure_cls = None
    original_savefig = None
    try:
        figure_cls, original_savefig, patched_savefig = _build_savefig_redirect(img_dir)
        figure_cls.savefig = patched_savefig
        print("[Info] Visualization started...")
        agent._visualize_epoch(viz_epoch)
        print("[Info] Visualization finished.")
    except Exception as e:
        run_error = str(e)
        raise
    finally:
        if figure_cls is not None and original_savefig is not None:
            figure_cls.savefig = original_savefig
        _cleanup_mid_viz_dirs(Path(agent.model_dir))
        final_image_count = len(list(img_dir.glob("*.png")))
        _write_try_param_log(
            try_dir=output_dir,
            runtime=runtime,
            config_path=config_path,
            ckpt_path=ckpt_path,
            config=config,
            ckpt_epoch=ckpt_epoch,
            viz_epoch=viz_epoch,
            status="failed" if run_error else "completed",
            image_count=final_image_count,
            error_message=run_error or "",
        )

    print(f"[Done] Visualization saved to: {out_dir}")


if __name__ == "__main__":
    main()
