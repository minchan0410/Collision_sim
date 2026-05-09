#!/usr/bin/env python3
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

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
DEFAULT_MODES = ["pure"]
DEFAULT_EVAL_DIR = "experiments/eval_runtime"
DEFAULT_SAMPLES = 20
DEFAULT_SAMPLING = "ddpm"
DEFAULT_DIFFUSION_STRIDE = 1
DEFAULT_TIMESTEP_STRIDE = 10
DEFAULT_TIMESTEP_BATCH = 10
DEFAULT_SEED = 123
DEFAULT_BEST_OF = True
DEFAULT_BICYCLE_ROLLOUT = False


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


def _set_eval_mode(config: EasyDict, mode: str, bicycle_rollout: bool) -> None:
    if mode == "pure":
        config.dynamics_guidance_enabled = False
        config.collision_guidance_enabled = False
        config.not_collision_guidance_enabled = False
    elif mode == "dynamics":
        config.dynamics_guidance_enabled = True
        config.collision_guidance_enabled = False
        config.not_collision_guidance_enabled = False
    else:
        raise ValueError(f"Unknown eval mode: {mode}")

    config.bicycle_rollout_enabled = bool(bicycle_rollout)


def _metric_row(
    *,
    mode: str,
    epoch: int,
    sampling: str,
    diffusion_stride: int,
    samples: int,
    best_of: bool,
    bicycle_rollout: bool,
    scene_count: int,
    batch_count: int,
    prediction_count: int,
    ade_values: np.ndarray,
    fde_values: np.ndarray,
) -> Dict[str, object]:
    if ade_values.size == 0 or fde_values.size == 0:
        return {
            "mode": mode,
            "epoch": int(epoch),
            "sampling": sampling,
            "diffusion_stride": int(diffusion_stride),
            "samples": int(samples),
            "best_of": bool(best_of),
            "bicycle_rollout": bool(bicycle_rollout),
            "scenes": int(scene_count),
            "batches": int(batch_count),
            "predictions": int(prediction_count),
            "ade_mean": "",
            "fde_mean": "",
            "ade_median": "",
            "fde_median": "",
        }

    return {
        "mode": mode,
        "epoch": int(epoch),
        "sampling": sampling,
        "diffusion_stride": int(diffusion_stride),
        "samples": int(samples),
        "best_of": bool(best_of),
        "bicycle_rollout": bool(bicycle_rollout),
        "scenes": int(scene_count),
        "batches": int(batch_count),
        "predictions": int(prediction_count),
        "ade_mean": float(np.mean(ade_values)),
        "fde_mean": float(np.mean(fde_values)),
        "ade_median": float(np.median(ade_values)),
        "fde_median": float(np.median(fde_values)),
    }


def _evaluate_mode(
    agent,
    *,
    mode: str,
    samples: int,
    sampling: str,
    diffusion_stride: int,
    timestep_stride: int,
    timestep_batch: int,
    max_scenes: int | None,
    best_of: bool,
    bicycle_rollout: bool,
    seed: int,
) -> Dict[str, object]:
    from dataset import get_timesteps_data
    import evaluation

    _set_eval_mode(agent.config, mode, bicycle_rollout=bicycle_rollout)
    agent.model.config = agent.config
    agent.model.eval()

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    node_type = "PEDESTRIAN"
    ph = int(agent.ph)
    min_hl = int(agent.min_hl)
    max_hl = int(agent.max_hl)

    scenes = list(agent.eval_scenes)
    if max_scenes is not None:
        scene_limit = max(0, int(max_scenes))
        if scene_limit < len(scenes):
            rng = np.random.default_rng(int(seed))
            scene_indices = rng.choice(len(scenes), size=scene_limit, replace=False)
            scenes = [scenes[int(idx)] for idx in scene_indices]
        else:
            scenes = scenes[:scene_limit]

    eval_ade: List[float] = []
    eval_fde: List[float] = []
    batch_count = 0

    with torch.no_grad():
        for scene_idx, scene in enumerate(scenes, start=1):
            print(f"[{mode}] Scene {scene_idx}/{len(scenes)}: {scene.name}")
            for t in tqdm(range(0, scene.timesteps, int(timestep_stride)), ncols=90):
                timesteps = np.arange(t, t + int(timestep_batch))
                batch = get_timesteps_data(
                    env=agent.eval_env,
                    scene=scene,
                    t=timesteps,
                    node_type=node_type,
                    state=agent.hyperparams["state"],
                    pred_state=agent.hyperparams["pred_state"],
                    edge_types=agent.eval_env.get_edge_types(),
                    min_ht=min_hl,
                    max_ht=max_hl,
                    min_ft=ph,
                    max_ft=ph,
                    hyperparams=agent.hyperparams,
                )
                if batch is None:
                    continue

                test_batch = batch[0]
                nodes = batch[1]
                timesteps_o = batch[2]

                traj_pred = agent.model.generate(
                    test_batch,
                    node_type,
                    num_points=ph,
                    sample=int(samples),
                    bestof=True,
                    sampling=sampling,
                    step=int(diffusion_stride),
                )

                predictions_dict = {}
                for j, ts in enumerate(timesteps_o):
                    predictions_dict.setdefault(ts, {})
                    predictions_dict[ts][nodes[j]] = np.transpose(traj_pred[:, [j]], (1, 0, 2, 3))

                batch_error_dict = evaluation.compute_batch_statistics(
                    predictions_dict,
                    scene.dt,
                    max_hl=max_hl,
                    ph=ph,
                    node_type_enum=agent.eval_env.NodeType,
                    kde=False,
                    map=None,
                    best_of=bool(best_of),
                    prune_ph_to_future=True,
                )

                eval_ade.extend(batch_error_dict[node_type]["ade"])
                eval_fde.extend(batch_error_dict[node_type]["fde"])
                batch_count += 1

    ade_values = np.asarray(eval_ade, dtype=float)
    fde_values = np.asarray(eval_fde, dtype=float)

    if agent.config.dataset == "eth":
        ade_values = ade_values / 0.6
        fde_values = fde_values / 0.6
    elif agent.config.dataset == "sdd":
        ade_values = ade_values * 50
        fde_values = fde_values * 50

    return _metric_row(
        mode=mode,
        epoch=int(agent.config.eval_at),
        sampling=sampling,
        diffusion_stride=diffusion_stride,
        samples=samples,
        best_of=best_of,
        bicycle_rollout=bicycle_rollout,
        scene_count=len(scenes),
        batch_count=batch_count,
        prediction_count=int(ade_values.size),
        ade_values=ade_values,
        fde_values=fde_values,
    )


def parse_scene_limit() -> Optional[int]:
    args = sys.argv[1:]
    if len(args) == 0:
        return None
    if len(args) == 1 and args[0].startswith("--") and args[0][2:].isdigit():
        return max(0, int(args[0][2:]))
    raise SystemExit(
        "Usage:\n"
        "  python eval_diffusion.py        # evaluate all scenes\n"
        "  python eval_diffusion.py --20   # evaluate first 20 scenes"
    )


def main() -> None:
    max_scenes = parse_scene_limit()

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

    for mode in DEFAULT_MODES:
        row = _evaluate_mode(
            agent,
            mode=mode,
            samples=DEFAULT_SAMPLES,
            sampling=DEFAULT_SAMPLING,
            diffusion_stride=DEFAULT_DIFFUSION_STRIDE,
            timestep_stride=DEFAULT_TIMESTEP_STRIDE,
            timestep_batch=DEFAULT_TIMESTEP_BATCH,
            max_scenes=max_scenes,
            best_of=DEFAULT_BEST_OF,
            bicycle_rollout=DEFAULT_BICYCLE_ROLLOUT,
            seed=DEFAULT_SEED,
        )
        print(f"{mode} ADE: {row['ade_mean']}")
        print(f"{mode} FDE: {row['fde_mean']}")


if __name__ == "__main__":
    main()
