import argparse
import os
import re
import shutil
import sys
import types
from pathlib import Path
from typing import Optional

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
    # Run with no args: python mat_viz.py
    # Latest checkpoint is selected from ckpt_dir.
    "config": "configs/mat.yaml",
    "ckpt_dir": "experiments/mat",
    "output_dir": "results/mat_viz/",
    "dataset": "",
    "seed": 123,
    # Fixed counts unless overridden by CLI.
    "viz_num_examples": 30,
    "viz_num_samples": 1,
    "sampling": "",
    "viz_epoch_tag": None,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize MAT checkpoint predictions using MID._visualize_epoch"
    )
    parser.add_argument("--config", default=USER_DEFAULTS["config"], help="Path to yaml config")
    parser.add_argument(
        "--ckpt_dir",
        default=USER_DEFAULTS["ckpt_dir"],
        help="Checkpoint directory (.pt files). Latest is used by default.",
    )
    parser.add_argument(
        "--output_dir",
        default=USER_DEFAULTS["output_dir"],
        help="Output directory root for generated images.",
    )
    # Backward compatibility: legacy one-dir option for both ckpt/input and output.
    parser.add_argument("--exp_dir", default="", help=argparse.SUPPRESS)
    parser.add_argument("--dataset", default=USER_DEFAULTS["dataset"], help="Dataset name override (e.g., mat_collision)")
    parser.add_argument("--ckpt", default="", help="Checkpoint path or filename under ckpt_dir")
    parser.add_argument(
        "--ckpt_index",
        type=int,
        default=None,
        help="Select checkpoint by index from --list_ckpts output (0 = latest)",
    )
    parser.add_argument("--list_ckpts", action="store_true", help="List checkpoints in ckpt_dir and exit")
    parser.add_argument("--seed", type=int, default=USER_DEFAULTS["seed"], help="Random seed for scene/timestep sampling")
    parser.add_argument(
        "--viz_num_examples",
        type=int,
        default=USER_DEFAULTS["viz_num_examples"],
        help="Number of scenes to visualize (fixed default unless overridden).",
    )
    parser.add_argument(
        "--viz_num_samples",
        type=int,
        default=USER_DEFAULTS["viz_num_samples"],
        help="Number of predicted samples per scene (fixed default unless overridden).",
    )
    parser.add_argument("--sampling", default=USER_DEFAULTS["sampling"], help="Override sampling method (ddpm/ddim)")
    parser.add_argument(
        "--viz_epoch_tag",
        type=int,
        default=USER_DEFAULTS["viz_epoch_tag"],
        help="Epoch tag used in output folder name epoch_XXXX (default: checkpoint epoch)",
    )
    return parser.parse_args()


def _extract_epoch_from_ckpt_name(ckpt_name: str) -> Optional[int]:
    m = re.search(r"_epoch(\d+)\.pt$", ckpt_name)
    if m is None:
        return None
    return int(m.group(1))


def _ckpt_sort_key(path: Path):
    # Prefer larger epoch numbers when available; otherwise fall back to mtime.
    epoch = _extract_epoch_from_ckpt_name(path.name)
    if epoch is None:
        return (0, -1, path.stat().st_mtime)
    return (1, epoch, path.stat().st_mtime)


def _list_checkpoints(exp_dir: Path):
    ckpts = sorted(exp_dir.glob("*.pt"), key=_ckpt_sort_key, reverse=True)
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


def _resolve_checkpoint(args):
    ckpt_dir = Path(args.ckpt_dir).resolve()
    if not ckpt_dir.exists():
        print(f"[Warn] ckpt_dir does not exist: {ckpt_dir}")
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
    args.ckpt_dir = str(ckpt_dir)

    if args.ckpt:
        ckpt_path = Path(args.ckpt)
        if ckpt_path.is_absolute():
            ckpt_path = ckpt_path.resolve()
        else:
            cand_in_exp = (ckpt_dir / ckpt_path).resolve()
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

    ckpts = _list_checkpoints(ckpt_dir)
    if len(ckpts) == 0:
        # Also allow checkpoints nested under subdirectories.
        ckpts = sorted(ckpt_dir.rglob("*.pt"), key=_ckpt_sort_key, reverse=True)
    if len(ckpts) == 0:
        print(f"[Warn] No .pt checkpoints found under: {ckpt_dir}")
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
        print(f"[Checkpoints] {ckpt_dir}")
        for idx, ckpt in enumerate(ckpts):
            mtime = ckpt.stat().st_mtime
            epoch = _extract_epoch_from_ckpt_name(ckpt.name)
            epoch_txt = f"epoch={epoch}" if epoch is not None else "epoch=?"
            print(f"{idx:>2}: {ckpt.name}  ({epoch_txt}, mtime={mtime:.0f})")
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

    output_dir = Path(args.output_dir)
    output_dir, msg = _ensure_output_dir(output_dir)
    if msg:
        print(msg)
    # MID builds model_dir via os.path.join('./experiments', exp_name).
    # Using an absolute exp_name makes model_dir resolve to the absolute output_dir path.
    exp_name = str(output_dir)

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

    return EasyDict(cfg), int(ckpt_epoch), output_dir


def main():
    args = parse_args()

    if args.exp_dir:
        # Legacy behavior: one directory used for both checkpoint lookup and output.
        args.ckpt_dir = args.exp_dir
        args.output_dir = args.exp_dir

    ckpt_path = _resolve_checkpoint(args)
    if ckpt_path is None:
        return

    config, ckpt_epoch, output_dir = _build_config(args, ckpt_path)

    # Reproducible random scene/timestep selection in visualization.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"[Info] Config: {args.config}")
    print(f"[Info] Checkpoint: {ckpt_path}")
    print(f"[Info] Dataset: {config.dataset}")
    print(f"[Info] Checkpoint dir: {args.ckpt_dir}")
    print(f"[Info] Output dir: {output_dir}")

    expected_ckpt = output_dir / ckpt_path.name
    if ckpt_path.resolve() != expected_ckpt.resolve():
        output_dir.mkdir(parents=True, exist_ok=True)
        if (not expected_ckpt.exists()) or (expected_ckpt.stat().st_size != ckpt_path.stat().st_size):
            shutil.copy2(str(ckpt_path), str(expected_ckpt))
            print(f"[Info] Copied checkpoint to output_dir: {expected_ckpt}")
        else:
            print(f"[Info] Reusing existing checkpoint in output_dir: {expected_ckpt}")

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

    # Ensure output goes to the selected output_dir.
    agent.model_dir = str(output_dir)
    os.makedirs(agent.model_dir, exist_ok=True)

    viz_epoch = int(args.viz_epoch_tag) if args.viz_epoch_tag is not None else int(ckpt_epoch)
    agent._visualize_epoch(viz_epoch)

    out_dir = Path(agent.model_dir) / "viz_outputs" / f"epoch_{viz_epoch:04d}"
    print(f"[Done] Visualization saved to: {out_dir}")


if __name__ == "__main__":
    main()
