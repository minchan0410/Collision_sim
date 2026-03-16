import argparse
import os
import re
import shutil
import sys
import types
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize MAT checkpoint predictions using MID._visualize_epoch"
    )
    parser.add_argument("--config", default="configs/mat.yaml", help="Path to yaml config")
    parser.add_argument("--exp_dir", default="experiments/mat", help="Experiment directory containing .pt files")
    parser.add_argument("--dataset", default="", help="Dataset name override (e.g., mat_collision)")
    parser.add_argument("--ckpt", default="", help="Checkpoint path or filename under exp_dir")
    parser.add_argument(
        "--ckpt_index",
        type=int,
        default=None,
        help="Select checkpoint by index from --list_ckpts output (0 = latest)",
    )
    parser.add_argument("--list_ckpts", action="store_true", help="List checkpoints in exp_dir and exit")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for scene/timestep sampling")
    parser.add_argument("--viz_num_examples", type=int, default=None, help="Override mat.yaml viz_num_examples")
    parser.add_argument("--viz_num_samples", type=int, default=None, help="Override mat.yaml viz_num_samples")
    parser.add_argument("--sampling", default="", help="Override sampling method (ddpm/ddim)")
    parser.add_argument(
        "--viz_epoch_tag",
        type=int,
        default=None,
        help="Epoch tag used in output folder name epoch_XXXX (default: checkpoint epoch)",
    )
    return parser.parse_args()


def _list_checkpoints(exp_dir: Path):
    ckpts = sorted(exp_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return ckpts


def _find_checkpoint_dirs():
    roots = [Path("experiments"), Path("result")]
    dirs = []
    for root in roots:
        if not root.exists():
            continue
        for pt in root.rglob("*.pt"):
            parent = pt.parent.resolve()
            if parent not in dirs:
                dirs.append(parent)
    return sorted(dirs, key=lambda p: str(p))


def _resolve_exp_dir_with_fallback(exp_dir: Path):
    if exp_dir.exists():
        return exp_dir.resolve(), None
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir.resolve(), f"[Info] exp_dir did not exist. Created: {exp_dir.resolve()}"


def _extract_dataset_epoch_from_name(ckpt_name: str):
    m = re.match(r"^(?P<dataset>.+)_epoch(?P<epoch>\d+)\.pt$", ckpt_name)
    if m is None:
        return None, None
    return m.group("dataset"), int(m.group("epoch"))


def _resolve_checkpoint(args):
    exp_dir = Path(args.exp_dir)
    exp_dir, msg = _resolve_exp_dir_with_fallback(exp_dir)
    if msg:
        print(msg)
    if exp_dir is None:
        print("[Info] Available checkpoint directories:")
        ckpt_dirs = _find_checkpoint_dirs()
        if len(ckpt_dirs) == 0:
            print("  (none found under ./experiments or ./result)")
        else:
            for d in ckpt_dirs[:20]:
                print(f"  - {d}")
            if len(ckpt_dirs) > 20:
                print(f"  ... and {len(ckpt_dirs) - 20} more")
        return None
    args.exp_dir = str(exp_dir)

    if args.ckpt:
        ckpt_path = Path(args.ckpt)
        if ckpt_path.is_absolute():
            ckpt_path = ckpt_path.resolve()
        else:
            cand_in_exp = (exp_dir / ckpt_path).resolve()
            cand_local = ckpt_path.resolve()
            if cand_in_exp.exists():
                ckpt_path = cand_in_exp
            elif cand_local.exists():
                ckpt_path = cand_local
            else:
                print(f"[Warn] Checkpoint not found: {args.ckpt}")
                print("[Info] Available checkpoint directories:")
                ckpt_dirs = _find_checkpoint_dirs()
                if len(ckpt_dirs) == 0:
                    print("  (none found under ./experiments or ./result)")
                else:
                    for d in ckpt_dirs[:20]:
                        print(f"  - {d}")
                    if len(ckpt_dirs) > 20:
                        print(f"  ... and {len(ckpt_dirs) - 20} more")
                return None

        if not ckpt_path.exists():
            print(f"[Warn] Checkpoint not found: {ckpt_path}")
            return None
        return ckpt_path

    ckpts = _list_checkpoints(exp_dir)
    if len(ckpts) == 0:
        # Also allow checkpoints nested under subdirectories.
        ckpts = sorted(exp_dir.rglob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if len(ckpts) == 0:
        print(f"[Warn] No .pt checkpoints found under: {exp_dir}")
        print("[Info] Available checkpoint directories:")
        ckpt_dirs = _find_checkpoint_dirs()
        if len(ckpt_dirs) == 0:
            print("  (none found under ./experiments or ./result)")
        else:
            for d in ckpt_dirs[:20]:
                print(f"  - {d}")
            if len(ckpt_dirs) > 20:
                print(f"  ... and {len(ckpt_dirs) - 20} more")
        return None

    if args.list_ckpts:
        print(f"[Checkpoints] {exp_dir}")
        for idx, ckpt in enumerate(ckpts):
            mtime = ckpt.stat().st_mtime
            print(f"{idx:>2}: {ckpt.name}  (mtime={mtime:.0f})")
        return None

    if args.ckpt_index is not None:
        if args.ckpt_index < 0 or args.ckpt_index >= len(ckpts):
            print(f"[Warn] ckpt_index out of range: {args.ckpt_index} (available: 0..{len(ckpts) - 1})")
            return None
        return ckpts[args.ckpt_index].resolve()

    # default: latest
    return ckpts[0].resolve()


def _build_config(args, ckpt_path: Path):
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        cfg = {}

    exp_dir = Path(args.exp_dir).resolve()
    # MID builds model_dir via os.path.join('./experiments', exp_name).
    # Using an absolute exp_name makes model_dir resolve to the absolute exp_dir path.
    exp_name = str(exp_dir)

    ckpt_dataset, ckpt_epoch = _extract_dataset_epoch_from_name(ckpt_path.name)
    if ckpt_epoch is None:
        raise ValueError(
            f"Checkpoint filename must match '<dataset>_epoch<number>.pt'. Got: {ckpt_path.name}"
        )

    dataset_name = args.dataset.strip() if args.dataset.strip() else ckpt_dataset
    if not dataset_name:
        raise ValueError("Could not infer dataset name. Please pass --dataset explicitly.")

    cfg["config"] = args.config
    cfg["exp_name"] = exp_name
    cfg["dataset"] = dataset_name
    cfg["eval_mode"] = True
    cfg["eval_at"] = int(ckpt_epoch)

    if args.viz_num_examples is not None:
        cfg["viz_num_examples"] = int(args.viz_num_examples)
    if args.viz_num_samples is not None:
        cfg["viz_num_samples"] = int(args.viz_num_samples)
    if args.sampling:
        cfg["sampling"] = args.sampling

    return EasyDict(cfg), int(ckpt_epoch), exp_dir


def main():
    args = parse_args()

    ckpt_path = _resolve_checkpoint(args)
    if ckpt_path is None:
        return

    config, ckpt_epoch, exp_dir = _build_config(args, ckpt_path)

    # Reproducible random scene/timestep selection in visualization.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"[Info] Config: {args.config}")
    print(f"[Info] Checkpoint: {ckpt_path}")
    print(f"[Info] Dataset: {config.dataset}")
    print(f"[Info] Exp dir: {exp_dir}")

    expected_ckpt = exp_dir / ckpt_path.name
    if ckpt_path.resolve() != expected_ckpt.resolve():
        exp_dir.mkdir(parents=True, exist_ok=True)
        if (not expected_ckpt.exists()) or (expected_ckpt.stat().st_size != ckpt_path.stat().st_size):
            shutil.copy2(str(ckpt_path), str(expected_ckpt))
            print(f"[Info] Copied checkpoint to exp_dir: {expected_ckpt}")
        else:
            print(f"[Info] Reusing existing checkpoint in exp_dir: {expected_ckpt}")

    # mid.py imports tensorboardX at module import time.
    # Provide a fallback so visualization can run even if tensorboardX is absent.
    try:
        import tensorboardX  # noqa: F401
    except Exception:
        try:
            from torch.utils.tensorboard import SummaryWriter as _SummaryWriter
        except Exception:
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

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in the current Python environment. "
            "Please run mat_viz.py in the same GPU environment used for training (e.g., csim)."
        )

    from mid import MID

    agent = MID(config)

    # Ensure output goes to the selected exp_dir.
    agent.model_dir = str(exp_dir)
    os.makedirs(agent.model_dir, exist_ok=True)

    viz_epoch = int(args.viz_epoch_tag) if args.viz_epoch_tag is not None else int(ckpt_epoch)
    agent._visualize_epoch(viz_epoch)

    out_dir = Path(agent.model_dir) / "viz_outputs" / f"epoch_{viz_epoch:04d}"
    print(f"[Done] Visualization saved to: {out_dir}")


if __name__ == "__main__":
    main()
