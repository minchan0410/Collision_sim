#!/usr/bin/env python3
"""
Standalone visualization script for MID checkpoints.

This script is intentionally separate from train/eval. It loads a checkpoint
produced during training (e.g. experiments/<exp_name>/<dataset>_epoch30.pt),
loads a dataset split (train by default), picks 1~2 clear trajectory examples,
and saves visualization images.

Example:
    python visualize_checkpoint.py \
        --config configs/baseline.yaml \
        --dataset eth \
        --checkpoint experiments/baseline/eth_epoch30.pt \
        --split train \
        --num_examples 2 \
        --num_samples 20 \
        --sampling ddim \
        --num_diffusion_steps 5
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import dill
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from dataset import collate, get_node_timestep_data
from models.autoencoder import AutoEncoder
from models.trajectron import Trajectron
from utils.model_registrar import ModelRegistrar
from utils.trajectron_hypers import get_traj_hypers

POSITION_STATE = {"position": ["x", "y"]}
TARGET_NODE_TYPE = "PEDESTRIAN"


@dataclass
class ExampleCandidate:
    scene_index: int
    scene_name: str
    timestep: int
    node_id: str
    score: float


class MIDCheckpointVisualizer:
    def __init__(self, config: SimpleNamespace, checkpoint_path: Path, split: str, device: torch.device):
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.split = split
        self.device = device
        self.model_dir = checkpoint_path.parent

        self.hyperparams = self._build_hyperparams(config)
        self.env = self._load_environment(split)
        self._build_model()

        self.max_hl = self.hyperparams["maximum_history_length"]
        self.ph = self.hyperparams["prediction_horizon"]

    @staticmethod
    def _yes_no(value, default):
        if value is None:
            return default
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, str):
            value_lower = value.strip().lower()
            if value_lower in {"yes", "true", "1"}:
                return "yes"
            if value_lower in {"no", "false", "0"}:
                return "no"
        return default

    def _build_hyperparams(self, config: SimpleNamespace) -> Dict:
        """
        Start from the repo defaults, then apply the minimal overrides that are
        needed for compatibility with the user's config/checkpoint.
        """
        hypers = get_traj_hypers()
        hypers["enc_rnn_dim_edge"] = config.encoder_dim // 2
        hypers["enc_rnn_dim_edge_influence"] = config.encoder_dim // 2
        hypers["enc_rnn_dim_history"] = config.encoder_dim // 2
        hypers["enc_rnn_dim_future"] = config.encoder_dim // 2

        # Optional compatibility overrides.
        if hasattr(config, "k_eval"):
            hypers["k_eval"] = int(config.k_eval)
        if hasattr(config, "override_attention_radius"):
            hypers["override_attention_radius"] = list(config.override_attention_radius)
        if hasattr(config, "incl_robot_node"):
            hypers["incl_robot_node"] = bool(config.incl_robot_node)
        if hasattr(config, "map_encoding"):
            hypers["use_map_encoding"] = bool(config.map_encoding)
        if hasattr(config, "no_edge_encoding"):
            hypers["edge_encoding"] = not bool(config.no_edge_encoding)
        if hasattr(config, "dynamic_edges"):
            hypers["dynamic_edges"] = self._yes_no(config.dynamic_edges, hypers["dynamic_edges"])
        if hasattr(config, "offline_scene_graph"):
            hypers["offline_scene_graph"] = self._yes_no(config.offline_scene_graph, hypers["offline_scene_graph"])
        if hasattr(config, "scene_freq_mult_train"):
            hypers["scene_freq_mult_train"] = bool(config.scene_freq_mult_train)
        if hasattr(config, "scene_freq_mult_eval"):
            hypers["scene_freq_mult_eval"] = bool(config.scene_freq_mult_eval)
        if hasattr(config, "node_freq_mult_train"):
            hypers["node_freq_mult_train"] = bool(config.node_freq_mult_train)
        if hasattr(config, "node_freq_mult_eval"):
            hypers["node_freq_mult_eval"] = bool(config.node_freq_mult_eval)
        if hasattr(config, "augment"):
            hypers["augment"] = bool(config.augment)
        return hypers

    def _load_environment(self, split: str):
        data_path = Path(self.config.data_dir) / f"{self.config.dataset}_{split}.pkl"
        if not data_path.exists():
            raise FileNotFoundError(
                f"Could not find dataset file: {data_path}\n"
                f"Check --data_dir. In this repo, process_data.py may create ETH/UCY files under "
                f"'processed_data_noise', so you may need to pass --data_dir processed_data_noise."
            )

        with data_path.open("rb") as f:
            env = dill.load(f, encoding="latin1")

        for attention_radius_override in getattr(self.config, "override_attention_radius", []):
            node_type1, node_type2, attention_radius = attention_radius_override.split(" ")
            env.attention_radius[(node_type1, node_type2)] = float(attention_radius)

        if env.robot_type is None and self.hyperparams["incl_robot_node"]:
            env.robot_type = env.NodeType[0]
            for scene in env.scenes:
                scene.add_robot_from_nodes(env.robot_type)

        return env

    @staticmethod
    def _safe_torch_load(path: Path, map_location):
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=map_location)

    def _build_model(self):
        checkpoint = self._safe_torch_load(self.checkpoint_path, map_location=self.device)

        self.registrar = ModelRegistrar(str(self.model_dir), self.device)
        self.registrar.load_models(checkpoint["encoder"])
        self.registrar.to(self.device)

        self.encoder = Trajectron(self.registrar, self.hyperparams, self.device)
        self.encoder.set_environment(self.env)
        self.encoder.set_annealing_params()

        self.model = AutoEncoder(self.config, encoder=self.encoder).to(self.device)
        self.model.load_state_dict(checkpoint["ddpm"])
        self.model.eval()

    @staticmethod
    def _clean_traj(traj: np.ndarray) -> np.ndarray:
        if traj.ndim == 1:
            traj = traj.reshape(1, -1)
        return traj[~np.isnan(traj).any(axis=1)]

    @staticmethod
    def _traj_length(traj: np.ndarray) -> float:
        if len(traj) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum())

    def _validate_manual_target(self, scene_index: int, timestep: int, node_id: str):
        if scene_index < 0 or scene_index >= len(self.env.scenes):
            raise IndexError(f"scene_index must be in [0, {len(self.env.scenes) - 1}], got {scene_index}")

        scene = self.env.scenes[scene_index]
        node = scene.get_node_by_id(node_id)
        if node is None:
            raise ValueError(f"Node id '{node_id}' does not exist in scene {scene_index} ({scene.name}).")

        valid_nodes = scene.present_nodes(
            np.array([timestep]),
            type=TARGET_NODE_TYPE,
            min_history_timesteps=self.max_hl,
            min_future_timesteps=self.ph,
        ).get(timestep, [])

        if node not in valid_nodes:
            raise ValueError(
                f"Node '{node_id}' is not valid at timestep {timestep}. "
                f"It needs at least {self.max_hl} history steps and {self.ph} future steps."
            )

        return ExampleCandidate(
            scene_index=scene_index,
            scene_name=scene.name,
            timestep=timestep,
            node_id=node_id,
            score=math.inf,
        )

    def auto_pick_examples(self, num_examples: int, search_stride: int) -> List[ExampleCandidate]:
        best_per_node: Dict[Tuple[int, str], ExampleCandidate] = {}

        for scene_index, scene in enumerate(self.env.scenes):
            timesteps = np.arange(0, scene.timesteps, max(1, search_stride), dtype=int)
            present = scene.present_nodes(
                timesteps,
                type=TARGET_NODE_TYPE,
                min_history_timesteps=self.max_hl,
                min_future_timesteps=self.ph,
            )

            for timestep, nodes in present.items():
                context_count = max(0, len(nodes) - 1)
                for node in nodes:
                    history = self._clean_traj(node.get(np.array([timestep - self.max_hl, timestep]), POSITION_STATE))
                    future = self._clean_traj(node.get(np.array([timestep + 1, timestep + self.ph]), POSITION_STATE))
                    if len(history) < 2 or len(future) < 2:
                        continue

                    displacement = float(np.linalg.norm(future[-1] - history[-1]))
                    future_length = self._traj_length(future)
                    score = displacement + 0.35 * future_length + 0.05 * context_count

                    key = (scene_index, node.id)
                    candidate = ExampleCandidate(
                        scene_index=scene_index,
                        scene_name=scene.name,
                        timestep=int(timestep),
                        node_id=node.id,
                        score=score,
                    )
                    if key not in best_per_node or candidate.score > best_per_node[key].score:
                        best_per_node[key] = candidate

        candidates = sorted(best_per_node.values(), key=lambda item: item.score, reverse=True)
        if not candidates:
            raise RuntimeError(
                "Could not find any valid target node. "
                "Check the dataset split or lower the search constraints."
            )
        return candidates[:num_examples]

    def _prepare_single_batch(self, scene, timestep: int, node):
        scene_graph = scene.get_scene_graph(
            timestep,
            self.env.attention_radius,
            self.hyperparams["edge_addition_filter"],
            self.hyperparams["edge_removal_filter"],
        )
        edge_types = [edge_type for edge_type in self.env.get_edge_types() if edge_type[0] == node.type]
        item = get_node_timestep_data(
            env=self.env,
            scene=scene,
            t=timestep,
            node=node,
            state=self.hyperparams["state"],
            pred_state=self.hyperparams["pred_state"],
            edge_types=edge_types,
            max_ht=self.max_hl,
            max_ft=self.ph,
            hyperparams=self.hyperparams,
            scene_graph=scene_graph,
        )
        return collate([item])

    def _collect_context_histories(self, scene, timestep: int, focal_node, radius: float, max_context_nodes: int = 12):
        context_nodes = scene.present_nodes(
            np.array([timestep]),
            type=TARGET_NODE_TYPE,
            min_history_timesteps=1,
            min_future_timesteps=1,
        ).get(timestep, [])
        focal_now = self._clean_traj(focal_node.get(np.array([timestep, timestep]), POSITION_STATE))[-1]

        context_items = []
        for node in context_nodes:
            if node == focal_node:
                continue
            history = self._clean_traj(node.get(np.array([max(0, timestep - self.max_hl), timestep]), POSITION_STATE))
            if len(history) == 0:
                continue
            distance = float(np.linalg.norm(history[-1] - focal_now))
            if distance <= radius:
                context_items.append((distance, history))

        context_items.sort(key=lambda item: item[0])
        return [history for _, history in context_items[:max_context_nodes]]

    @staticmethod
    def _compute_errors(predictions: np.ndarray, future: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # predictions: [num_samples, ph, 2], future: [ph, 2]
        error = np.linalg.norm(predictions - future[None, :, :], axis=-1)
        ade = error.mean(axis=-1)
        fde = error[:, -1]
        return ade, fde

    def predict_for_candidate(
        self,
        candidate: ExampleCandidate,
        num_samples: int,
        sampling: str,
        num_diffusion_steps: int,
        bestof: bool,
        context_radius: float,
    ):
        total_steps = int(self.model.diffusion.var_sched.num_steps)
        if total_steps % num_diffusion_steps != 0:
            raise ValueError(
                f"num_diffusion_steps ({num_diffusion_steps}) must divide total diffusion steps ({total_steps})."
            )
        stride = total_steps // num_diffusion_steps

        scene = self.env.scenes[candidate.scene_index]
        node = scene.get_node_by_id(candidate.node_id)
        if node is None:
            raise ValueError(f"Node '{candidate.node_id}' no longer exists in scene {candidate.scene_index}.")

        batch = self._prepare_single_batch(scene, candidate.timestep, node)
        history = self._clean_traj(node.get(np.array([candidate.timestep - self.max_hl, candidate.timestep]), POSITION_STATE))
        future = self._clean_traj(node.get(np.array([candidate.timestep + 1, candidate.timestep + self.ph]), POSITION_STATE))
        context_histories = self._collect_context_histories(scene, candidate.timestep, node, context_radius)

        with torch.no_grad():
            predictions = self.model.generate(
                batch=batch,
                node_type=TARGET_NODE_TYPE,
                num_points=self.ph,
                sample=num_samples,
                bestof=bestof,
                sampling=sampling,
                step=stride,
            )

        predictions = predictions[:, 0]  # [num_samples, ph, 2]
        ade, fde = self._compute_errors(predictions, future)
        best_idx = int(np.argmin(ade))

        return {
            "scene": scene,
            "node": node,
            "history": history,
            "future": future,
            "predictions": predictions,
            "best_idx": best_idx,
            "best_ade": float(ade[best_idx]),
            "best_fde": float(fde[best_idx]),
            "context_histories": context_histories,
            "num_diffusion_steps": num_diffusion_steps,
        }

    @staticmethod
    def _sanitize_filename(text: str) -> str:
        return re.sub(r"[^a-zA-Z0-9._-]+", "_", text)

    def render_prediction(
        self,
        candidate: ExampleCandidate,
        prediction_pack: Dict,
        output_dir: Path,
        dpi: int,
    ) -> Dict:
        scene = prediction_pack["scene"]
        history = prediction_pack["history"]
        future = prediction_pack["future"]
        predictions = prediction_pack["predictions"]
        best_idx = prediction_pack["best_idx"]
        best_pred = predictions[best_idx]
        context_histories = prediction_pack["context_histories"]

        fig, ax = plt.subplots(figsize=(7.5, 7.5))

        # Nearby pedestrians (light gray for context only)
        for ctx_history in context_histories:
            ax.plot(
                ctx_history[:, 0],
                ctx_history[:, 1],
                linewidth=1.2,
                alpha=0.45,
                color="#b0b0b0",
                zorder=1,
            )
            ax.scatter(
                ctx_history[-1, 0],
                ctx_history[-1, 1],
                s=18,
                alpha=0.45,
                color="#b0b0b0",
                zorder=1,
            )

        # All predicted samples for the focal pedestrian.
        for sample_idx, pred in enumerate(predictions):
            label = "Pred samples" if sample_idx == 0 else None
            ax.plot(
                pred[:, 0],
                pred[:, 1],
                linewidth=1.4,
                alpha=0.18,
                color="#1f77b4",
                label=label,
                zorder=2,
            )

        # Best sample / GT / observed trajectory.
        ax.plot(
            history[:, 0],
            history[:, 1],
            "-o",
            linewidth=2.4,
            markersize=3,
            color="black",
            label="Observed",
            zorder=4,
        )
        ax.plot(
            future[:, 0],
            future[:, 1],
            "-o",
            linewidth=2.6,
            markersize=3,
            color="#2ca02c",
            label="Ground truth",
            zorder=5,
        )
        ax.plot(
            best_pred[:, 0],
            best_pred[:, 1],
            "-o",
            linewidth=2.6,
            markersize=3,
            color="#d62728",
            label="Best sample",
            zorder=6,
        )
        ax.scatter(
            history[-1, 0],
            history[-1, 1],
            s=55,
            color="black",
            edgecolors="white",
            linewidths=0.8,
            label="Current position",
            zorder=7,
        )

        all_points = [history, future, best_pred] + list(predictions) + list(context_histories)
        stacked = np.concatenate([pts for pts in all_points if len(pts) > 0], axis=0)
        x_min, y_min = stacked.min(axis=0)
        x_max, y_max = stacked.max(axis=0)
        pad_x = max(0.8, (x_max - x_min) * 0.18)
        pad_y = max(0.8, (y_max - y_min) * 0.18)

        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(y_min - pad_y, y_max + pad_y)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(loc="best", fontsize=9, frameon=True)
        ax.set_title(
            f"{self.config.dataset.upper()} | split={self.split} | scene={candidate.scene_index}:{scene.name}\n"
            f"node={candidate.node_id}, t={candidate.timestep}, best ADE={prediction_pack['best_ade']:.3f}, "
            f"best FDE={prediction_pack['best_fde']:.3f}",
            fontsize=11,
        )
        fig.tight_layout()

        scene_name = self._sanitize_filename(scene.name)
        node_name = self._sanitize_filename(candidate.node_id)
        filename = (
            f"{self.config.dataset}_{self.split}_scene{candidate.scene_index}_{scene_name}_"
            f"t{candidate.timestep}_node{node_name}.png"
        )
        output_path = output_dir / filename
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        return {
            "scene_index": int(candidate.scene_index),
            "scene_name": candidate.scene_name,
            "timestep": int(candidate.timestep),
            "node_id": candidate.node_id,
            "score": None if math.isinf(candidate.score) else float(candidate.score),
            "best_ade": float(prediction_pack["best_ade"]),
            "best_fde": float(prediction_pack["best_fde"]),
            "num_context_nodes": int(len(context_histories)),
            "output_file": str(output_path),
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize 1~2 MID trajectories from a trained checkpoint.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config used for training.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g. eth / hotel / univ / zara1 / zara2 / sdd")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to a train-generated checkpoint (.pt).")
    parser.add_argument("--epoch", type=int, default=None, help="If --checkpoint is omitted, load <dataset>_epoch{epoch}.pt.")
    parser.add_argument("--data_dir", type=str, default=None, help="Override the data_dir in the YAML config.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train", help="Which split to visualize from.")
    parser.add_argument("--output_dir", type=str, default="viz_outputs", help="Directory for saved images and summary json.")
    parser.add_argument("--num_examples", type=int, default=2, help="How many examples to save.")
    parser.add_argument("--num_samples", type=int, default=20, help="How many trajectory samples to draw per example.")
    parser.add_argument("--sampling", choices=["ddpm", "ddim"], default="ddpm", help="Sampling method.")
    parser.add_argument("--num_diffusion_steps", type=int, default=100, help="User-facing diffusion step count. 100 = full steps, 5 = fast 5-step sampling.")
    parser.add_argument("--no_bestof", action="store_true", help="Disable best-of style random x_T initialization.")
    parser.add_argument("--device", type=str, default="auto", help="cuda / cpu / auto")
    parser.add_argument("--seed", type=int, default=None, help="Random seed. Defaults to the training config seed if present.")
    parser.add_argument("--search_stride", type=int, default=5, help="Only used in auto mode. Search every N timesteps.")
    parser.add_argument("--scene_index", type=int, default=None, help="Manual target scene index.")
    parser.add_argument("--timestep", type=int, default=None, help="Manual target timestep.")
    parser.add_argument("--node_id", type=str, default=None, help="Manual target node id.")
    parser.add_argument("--context_radius", type=float, default=8.0, help="Plot nearby pedestrians within this radius.")
    parser.add_argument("--dpi", type=int, default=220, help="Saved image DPI.")
    return parser.parse_args()


def load_config(config_path: Path, dataset: str, data_dir_override: Optional[str]) -> SimpleNamespace:
    with config_path.open("r") as f:
        config_dict = yaml.safe_load(f) or {}

    config_dict = dict(config_dict)
    config_dict["dataset"] = dataset.rstrip("/")
    config_dict["exp_name"] = config_path.stem
    if data_dir_override is not None:
        config_dict["data_dir"] = data_dir_override

    # Minimal defaults used by the visualizer.
    config_dict.setdefault("diffnet", "TransformerConcatLinear")
    config_dict.setdefault("encoder_dim", 256)
    config_dict.setdefault("tf_layer", 3)
    config_dict.setdefault("data_dir", "processed_data")
    config_dict.setdefault("seed", 123)
    config_dict.setdefault("override_attention_radius", [])
    config_dict.setdefault("incl_robot_node", False)
    config_dict.setdefault("map_encoding", False)
    config_dict.setdefault("no_edge_encoding", False)
    return SimpleNamespace(**config_dict)


def resolve_checkpoint_path(config: SimpleNamespace, checkpoint_arg: str, epoch: Optional[int]) -> Path:
    if checkpoint_arg:
        checkpoint_path = Path(checkpoint_arg)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return checkpoint_path

    model_dir = Path("experiments") / config.exp_name
    if epoch is not None:
        checkpoint_path = model_dir / f"{config.dataset}_epoch{epoch}.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return checkpoint_path

    pattern = f"{config.dataset}_epoch*.pt"
    candidates = list(model_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint was found under {model_dir} matching {pattern}. "
            f"Pass --checkpoint explicitly or specify --epoch."
        )

    def extract_epoch(path: Path) -> int:
        match = re.search(r"_epoch(\d+)\.pt$", path.name)
        return int(match.group(1)) if match else -1

    candidates.sort(key=extract_epoch)
    return candidates[-1]


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    device = choose_device(args.device)
    config = load_config(Path(args.config), args.dataset, args.data_dir)
    seed = config.seed if args.seed is None else args.seed
    set_seed(int(seed))

    checkpoint_path = resolve_checkpoint_path(config, args.checkpoint, args.epoch)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    visualizer = MIDCheckpointVisualizer(
        config=config,
        checkpoint_path=checkpoint_path,
        split=args.split,
        device=device,
    )

    manual_fields = [args.scene_index, args.timestep, args.node_id]
    manual_mode = any(field is not None for field in manual_fields)
    if manual_mode and not all(field is not None for field in manual_fields):
        raise ValueError("Manual mode requires --scene_index, --timestep, and --node_id together.")

    if manual_mode:
        candidates = [visualizer._validate_manual_target(args.scene_index, args.timestep, args.node_id)]
    else:
        candidates = visualizer.auto_pick_examples(args.num_examples, args.search_stride)

    summary = {
        "config": str(Path(args.config).resolve()),
        "dataset": config.dataset,
        "split": args.split,
        "checkpoint": str(checkpoint_path.resolve()),
        "device": str(device),
        "sampling": args.sampling,
        "num_diffusion_steps": int(args.num_diffusion_steps),
        "num_samples": int(args.num_samples),
        "bestof": bool(not args.no_bestof),
        "seed": int(seed),
        "examples": [],
    }

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Dataset split: {config.dataset}_{args.split}.pkl")
    print(f"Saving images to: {output_dir.resolve()}")

    for index, candidate in enumerate(candidates, start=1):
        prediction_pack = visualizer.predict_for_candidate(
            candidate=candidate,
            num_samples=args.num_samples,
            sampling=args.sampling,
            num_diffusion_steps=args.num_diffusion_steps,
            bestof=not args.no_bestof,
            context_radius=args.context_radius,
        )
        item = visualizer.render_prediction(candidate, prediction_pack, output_dir, dpi=args.dpi)
        summary["examples"].append(item)
        print(
            f"[{index}/{len(candidates)}] scene={candidate.scene_index}, node={candidate.node_id}, "
            f"t={candidate.timestep}, best_ADE={item['best_ade']:.3f}, best_FDE={item['best_fde']:.3f}"
        )
        print(f"    -> {item['output_file']}")

    summary_path = output_dir / f"{config.dataset}_{args.split}_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
