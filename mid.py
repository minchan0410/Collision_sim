import os
import argparse
import torch
import dill
import pdb
import numpy as np
import os.path as osp
import logging
import time
import re
from torch import nn, optim, utils
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm.auto import tqdm
import pickle

from dataset import EnvironmentDataset, collate, get_timesteps_data, get_node_timestep_data, restore
from models.autoencoder import AutoEncoder
from models.trajectron import Trajectron
from utils.model_registrar import ModelRegistrar
from utils.trajectron_hypers import get_traj_hypers
import evaluation

import matplotlib
matplotlib.use("Agg")  # Save images without display server
import matplotlib.pyplot as plt


class MID():
    def __init__(self, config):
        self.config = config
        torch.backends.cudnn.benchmark = True
        self._build()

    def _cache_and_validate_hyperparams(self):
        self.ph = int(self.hyperparams['prediction_horizon'])
        self.min_hl = int(self.hyperparams['minimum_history_length'])
        self.max_hl = int(self.hyperparams['maximum_history_length'])

        if self.ph <= 0:
            raise ValueError(f"prediction_horizon must be positive, got {self.ph}")
        if self.min_hl <= 0:
            raise ValueError(f"minimum_history_length must be positive, got {self.min_hl}")
        if self.max_hl <= 0:
            raise ValueError(f"maximum_history_length must be positive, got {self.max_hl}")
        if self.min_hl > self.max_hl:
            raise ValueError(
                f"minimum_history_length({self.min_hl}) cannot be larger than "
                f"maximum_history_length({self.max_hl})"
            )


    @staticmethod
    def _extract_numeric_node_id(node):
        raw_id = getattr(node, "id", node)

        if isinstance(raw_id, (int, np.integer)):
            return int(raw_id)

        if isinstance(raw_id, float) and float(raw_id).is_integer():
            return int(raw_id)

        raw_id_str = str(raw_id).strip()

        try:
            raw_id_float = float(raw_id_str)
            if raw_id_float.is_integer():
                return int(raw_id_float)
        except Exception:
            pass

        match = re.search(r"-?\d+", raw_id_str)
        if match:
            return int(match.group())

        return None

    def _node_sort_tuple(self, node):
        numeric_id = self._extract_numeric_node_id(node)
        if numeric_id is not None:
            return (0, numeric_id, str(getattr(node, "id", "")))
        return (1, float("inf"), str(getattr(node, "id", "")))

    def _resolve_scene_ego_node(self, scene):
        scene_nodes = list(getattr(scene, "nodes", []))

        explicit_ego_nodes = [
            node for node in scene_nodes
            if "ego" in str(getattr(node, "description", "") or "").lower()
        ]
        if explicit_ego_nodes:
            return sorted(explicit_ego_nodes, key=self._node_sort_tuple)[0]

        numeric_nodes = []
        for node in scene_nodes:
            numeric_id = self._extract_numeric_node_id(node)
            if numeric_id is not None:
                numeric_nodes.append((numeric_id, node))

        if not numeric_nodes:
            return None

        # mat_preprocess/mat2txt.py 기준:
        #   next_track_id = 1 에서 시작하고 Ego를 먼저 rows에 기록한 뒤,
        #   그 다음에 Traffic(agent) track_id를 순차적으로 부여한다.
        # 따라서 현재 파이프라인(process_data_mat.py -> MID)에서는
        # 가장 작은 track_id(보통 1)가 Ego 차량이다.
        ego_candidates = [node for numeric_id, node in numeric_nodes if numeric_id == 1]
        if ego_candidates:
            return sorted(ego_candidates, key=self._node_sort_tuple)[0]

        return min(numeric_nodes, key=lambda item: (item[0], self._node_sort_tuple(item[1])))[1]

    def _resolve_visual_role_labels(self, scene, nodes_at_t):
        nodes_at_t = list(nodes_at_t)
        if len(nodes_at_t) == 0:
            return {}

        ego_node = self._resolve_scene_ego_node(scene)
        role_labels = {}

        if ego_node in nodes_at_t:
            role_labels[ego_node] = f"Ego"

        opponents = [node for node in nodes_at_t if node is not ego_node]
        opponents = sorted(opponents, key=self._node_sort_tuple)

        only_one_opponent = len(opponents) == 1
        for opp_idx, node in enumerate(opponents, start=1):
            prefix = "Target" if only_one_opponent else f"Opponent {opp_idx}"
            role_labels[node] = f"{prefix}"

        # Ego가 현재 timestep에 없고 단일 차량만 존재하는 경우에는
        # 잘못 Ego로 단정하지 않고 원래 id를 유지한다.
        if ego_node not in nodes_at_t and len(opponents) == 1:
            only_node = opponents[0]
            role_labels[only_node] = f"ID {only_node.id}"

        return role_labels

    def _get_vehicle_color_set(self, node_idx):
        """
        차량별 색상 세트를 반환합니다.
        요구사항: 명도(Darkness) 순서가 hist > gt > pred 순으로 어두워야 함.
        (hist가 가장 어둡고, 고쳐진 pred는 기존 pred보다 어두워야 함)
        """
        color_sets = [
            # 세트 1 (파랑 계열)
            {
                "hist": "#011f4b",  # Darkest Navy (가장 어둠)
                "gt":   "#005b96",  # Dark Blue
                "pred": "#6497b1",  # Medium Blue (기존 파스텔보다 어둡지만 gt보단 밝음)
                "curr": "#011f4b",  # hist와 동일하게 유지
            },
            # 세트 2 (빨강 계열)
            {
                "hist": "#4d0000",  # Darkest Maroon
                "gt":   "#990000",  # Dark Red
                "pred": "#cc4c4c",  # Medium Red
                "curr": "#4d0000",
            },
            # 세트 3 (초록 계열)
            {
                "hist": "#003300",  # Darkest Deep Green
                "gt":   "#1a8c1a",  # Dark Green
                "pred": "#66b366",  # Medium Green
                "curr": "#003300",
            },
            # 세트 4 (보라 계열)
            {
                "hist": "#2a0a4d",  # Darkest Indigo/Purple
                "gt":   "#6a3d9a",  # Dark Purple
                "pred": "#9966cc",  # Medium Purple
                "curr": "#2a0a4d",
            },
            # 세트 5 (오렌지/브라운 계열)
            {
                "hist": "#4d1a00",  # Darkest Brown
                "gt":   "#cc5200",  # Dark Orange/Rust
                "pred": "#ffb366",  # Medium Orange
                "curr": "#4d1a00",
            },
            # 세트 6 (청록/틸 계열)
            {
                "hist": "#003333",  # Darkest Teal
                "gt":   "#008080",  # Teal
                "pred": "#66cccc",  # Medium Teal
                "curr": "#003333",
            },
        ]
        return color_sets[node_idx % len(color_sets)]

    def train(self):
        for epoch in range(1, self.config.epochs + 1):
            self.train_dataset.augment = self.config.augment
            for node_type, data_loader in self.train_data_loader.items():
                pbar = tqdm(data_loader, ncols=80)
                for batch in pbar:
                    self.optimizer.zero_grad()
                    train_loss = self.model.get_loss(batch, node_type)
                    pbar.set_description(f"Epoch {epoch}, {node_type} MSE: {train_loss.item():.2f}")
                    train_loss.backward()
                    self.optimizer.step()

            self.scheduler.step()
            self.train_dataset.augment = False

            # Evaluation logic (runs only when eval_every > 0)
            if getattr(self.config, 'eval_every', 0) > 0 and epoch % self.config.eval_every == 0:
                self.model.eval()

                node_type = "PEDESTRIAN"
                eval_ade_batch_errors = []
                eval_fde_batch_errors = []

                ph = self.ph
                min_hl = self.min_hl
                max_hl = self.max_hl

                for i, scene in enumerate(self.eval_scenes):
                    print(f"----- Evaluating Scene {i + 1}/{len(self.eval_scenes)}")
                    for t in tqdm(range(0, scene.timesteps, 10)):
                        timesteps = np.arange(t, t + 10)
                        batch = get_timesteps_data(
                            env=self.eval_env,
                            scene=scene,
                            t=timesteps,
                            node_type=node_type,
                            state=self.hyperparams['state'],
                            pred_state=self.hyperparams['pred_state'],
                            edge_types=self.eval_env.get_edge_types(),
                            min_ht=min_hl,
                            max_ht=max_hl,
                            min_ft=ph,
                            max_ft=ph,
                            hyperparams=self.hyperparams
                        )
                        if batch is None:
                            continue

                        test_batch = batch[0]
                        nodes = batch[1]
                        timesteps_o = batch[2]

                        traj_pred = self.model.generate(
                            test_batch,
                            node_type,
                            num_points=ph,
                            sample=20,
                            bestof=True,
                            sampling="ddpm",
                            step=1
                        )

                        predictions = traj_pred
                        predictions_dict = {}
                        for j, ts in enumerate(timesteps_o):
                            if ts not in predictions_dict:
                                predictions_dict[ts] = dict()
                            predictions_dict[ts][nodes[j]] = np.transpose(predictions[:, [j]], (1, 0, 2, 3))

                        batch_error_dict = evaluation.compute_batch_statistics(
                            predictions_dict,
                            scene.dt,
                            max_hl=max_hl,
                            ph=ph,
                            node_type_enum=self.eval_env.NodeType,
                            kde=False,
                            map=None,
                            best_of=True,
                            prune_ph_to_future=True
                        )

                        eval_ade_batch_errors = np.hstack((eval_ade_batch_errors, batch_error_dict[node_type]['ade']))
                        eval_fde_batch_errors = np.hstack((eval_fde_batch_errors, batch_error_dict[node_type]['fde']))

                ade = np.mean(eval_ade_batch_errors)
                fde = np.mean(eval_fde_batch_errors)

                print(f"Epoch {epoch} Best Of 20: ADE: {ade:.4f} FDE: {fde:.4f}")
                self.log.info(f"Best of 20: Epoch {epoch} ADE: {ade:.4f} FDE: {fde:.4f}")

                self.model.train()

            # Checkpoint + visualization logic controlled by save_pt_every
            save_pt_every = getattr(self.config, 'save_pt_every', 1)
            if save_pt_every > 0 and epoch % save_pt_every == 0:
                checkpoint = {
                    'encoder': self.registrar.model_dict,
                    'ddpm': self.model.state_dict()
                }
                torch.save(checkpoint, osp.join(self.model_dir, f"{self.config.dataset}_epoch{epoch}.pt"))
                print(f"> Epoch {epoch} Checkpoint Saved successfully!")

                if getattr(self.config, 'viz_enabled', True):
                    self._visualize_epoch(epoch)

    def _visualize_epoch(self, epoch):
        print(f"--- Generating Random Scene Visualizations for Epoch {epoch} ---")

        viz_root_dir = osp.join(self.model_dir, "viz_outputs")
        os.makedirs(viz_root_dir, exist_ok=True)

        epoch_viz_dir = osp.join(viz_root_dir, f"epoch_{epoch:04d}")
        os.makedirs(epoch_viz_dir, exist_ok=True)

        num_samples = getattr(self.config, "viz_num_samples", 50)
        sampling = getattr(self.config, "sampling", "ddpm")

        ph = self.ph
        min_hl = self.min_hl
        max_hl = self.max_hl
        node_type = "PEDESTRIAN"

        if len(self.eval_scenes) == 0:
            print("  [Warn] 평가 씬이 없어 시각화를 건너뜁니다.")
            return

        num_scenes_to_viz = getattr(self.config, "viz_num_examples", 2)
        num_scenes_to_viz = min(num_scenes_to_viz, len(self.eval_scenes))

        scene_indices = np.random.choice(
            len(self.eval_scenes),
            size=num_scenes_to_viz,
            replace=False
        )

        self.model.eval()

        for scene_order, scene_idx in enumerate(scene_indices, start=1):
            scene = self.eval_scenes[int(scene_idx)]

            valid_timesteps = []
            for t_cand in range(min_hl, scene.timesteps - ph):
                present_cand = scene.present_nodes(
                    np.array([t_cand]),
                    type=node_type,
                    min_history_timesteps=min_hl,
                    min_future_timesteps=ph
                )
                if t_cand in present_cand and len(present_cand[t_cand]) > 0:
                    valid_timesteps.append(t_cand)

            if len(valid_timesteps) == 0:
                print(f"  [Warn] 씬 '{scene.name}'에는 조건을 만족하는 timestep이 없어 스킵합니다.")
                continue

            t = int(np.random.choice(valid_timesteps))

            present = scene.present_nodes(
                np.array([t]),
                type=node_type,
                min_history_timesteps=min_hl,
                min_future_timesteps=ph
            )

            if t not in present or len(present[t]) == 0:
                print(f"  [Warn] 씬 '{scene.name}'의 t={t}에서 유효 차량이 없어 스킵합니다.")
                continue

            raw_nodes_at_t = list(present[t])
            role_labels = self._resolve_visual_role_labels(scene, raw_nodes_at_t)
            ego_node = self._resolve_scene_ego_node(scene)

            nodes_at_t = sorted(
                raw_nodes_at_t,
                key=lambda n: (
                    0 if n is ego_node else 1,
                    *self._node_sort_tuple(n)
                )
            )
            print(f"  -> [{scene_order}/{num_scenes_to_viz}] 씬 '{scene.name}'에서 랜덤 시점 t={t}, 차량 {len(nodes_at_t)}대를 시각화합니다.")

            fig, ax = plt.subplots(figsize=(10, 10))
            all_points_for_scale = []

            for node_idx, node in enumerate(nodes_at_t):
                colors = self._get_vehicle_color_set(node_idx)

                history = node.get(np.array([t - max_hl, t]), {'position': ['x', 'y']})
                future = node.get(np.array([t + 1, t + ph]), {'position': ['x', 'y']})

                history = history[~np.isnan(history).any(axis=1)]
                future = future[~np.isnan(future).any(axis=1)]

                if len(history) < 2 or len(future) < 2:
                    continue

                scene_graph = scene.get_scene_graph(
                    t,
                    self.eval_env.attention_radius,
                    self.hyperparams['edge_addition_filter'],
                    self.hyperparams['edge_removal_filter']
                )

                edge_types = [
                    edge_type for edge_type in self.eval_env.get_edge_types()
                    if edge_type[0] == node.type
                ]

                item = get_node_timestep_data(
                    self.eval_env,
                    scene,
                    t,
                    node,
                    self.hyperparams['state'],
                    self.hyperparams['pred_state'],
                    edge_types,
                    max_hl,
                    ph,
                    self.hyperparams,
                    scene_graph=scene_graph
                )
                batch = collate([item])

                with torch.no_grad():
                    predictions = self.model.generate(
                        batch,
                        node.type,
                        num_points=ph,
                        sample=num_samples,
                        bestof=True,
                        sampling=sampling,
                        step=1
                    )
                    predictions = predictions[:, 0]

                for pred in predictions:
                    pred_traj = np.vstack((history[-1:], pred))
                    ax.plot(
                        pred_traj[:, 0],
                        pred_traj[:, 1],
                        '-',
                        color=colors["pred"],
                        alpha=1,
                        linewidth=1.5
                        # markersize=3
                    )

                ax.plot(
                    history[:, 0],
                    history[:, 1],
                    '-o',
                    color=colors["hist"],
                    linewidth=2.2,
                    markersize=3
                )

                gt_traj = np.vstack((history[-1:], future))
                ax.plot(
                    gt_traj[:, 0],
                    gt_traj[:, 1],
                    '-o',
                    color=colors["gt"],
                    linewidth=1.5,
                    markersize=2,
                    alpha=0.8
                )

                ax.scatter(
                    history[-1, 0],
                    history[-1, 1],
                    color=colors["curr"],
                    s=90,
                    edgecolors='white',
                    linewidths=1.5,
                    zorder=5
                )

                display_label = role_labels.get(node, f"ID {node.id}")

                ax.text(
                    history[-1, 0] + 2,
                    history[-1, 1] + 1,
                    display_label,
                    color=colors["hist"],
                    fontsize=9,
                    fontweight='bold'
                )

                all_points_for_scale.extend(history)
                all_points_for_scale.extend(future)
                all_points_for_scale.extend(predictions.reshape(-1, 2))

            if len(all_points_for_scale) == 0:
                plt.close(fig)
                continue

            all_points = np.array(all_points_for_scale)
            x_min, y_min = np.min(all_points, axis=0)
            x_max, y_max = np.max(all_points, axis=0)

            center_x = (x_max + x_min) / 2.0
            center_y = (y_max + y_min) / 2.0
            max_range = max(15.0, x_max - x_min, y_max - y_min)
            margin = max_range * 0.55

            ax.set_xlim(center_x - margin, center_x + margin)
            ax.set_ylim(center_y - margin, center_y + margin)
            ax.set_aspect('equal', adjustable='box')
            ax.grid(alpha=0.3)

            import matplotlib.lines as mlines
            hist_line = mlines.Line2D([], [], color='black', marker='o', markersize=4, label='History')
            gt_line = mlines.Line2D([], [], color='dimgray', marker='o', markersize=4, label='GT Future')
            pred_line = mlines.Line2D([], [], color='lightgray', marker='o', markersize=3, label='Predictions')
            curr_point = mlines.Line2D([], [], color='white', marker='o', markerfacecolor='black', markersize=8, label='Current Pos')
            ax.legend(handles=[hist_line, gt_line, pred_line, curr_point], loc='best')

            ax.set_title(
                f"Epoch {epoch} | Scene #{scene_order} (idx={int(scene_idx)}) | "
                f"{scene.name} | t={t}"
            )

            safe_scene_name = str(scene.name).replace("/", "_").replace("\\", "_")
            out_file = osp.join(
                epoch_viz_dir,
                f"scene_{scene_order:02d}_idx_{int(scene_idx):04d}_{safe_scene_name}_t_{t}.png"
            )
            fig.savefig(out_file, dpi=200, bbox_inches='tight')
            plt.close(fig)

            print(f"  -> Saved: {out_file}")

        self.model.train()

    def eval(self, sampling, step):
        epoch = self.config.eval_at

        self.log.info(f"Sampling: {sampling} Stride: {step}")

        node_type = "PEDESTRIAN"
        eval_ade_batch_errors = []
        eval_fde_batch_errors = []

        ph = self.ph
        min_hl = self.min_hl
        max_hl = self.max_hl

        for i, scene in enumerate(self.eval_scenes):
            print(f"----- Evaluating Scene {i + 1}/{len(self.eval_scenes)}")
            for t in tqdm(range(0, scene.timesteps, 10)):
                timesteps = np.arange(t, t + 10)
                batch = get_timesteps_data(
                    env=self.eval_env,
                    scene=scene,
                    t=timesteps,
                    node_type=node_type,
                    state=self.hyperparams['state'],
                    pred_state=self.hyperparams['pred_state'],
                    edge_types=self.eval_env.get_edge_types(),
                    min_ht=min_hl,
                    max_ht=max_hl,
                    min_ft=ph,
                    max_ft=ph,
                    hyperparams=self.hyperparams
                )
                if batch is None:
                    continue

                test_batch = batch[0]
                nodes = batch[1]
                timesteps_o = batch[2]

                traj_pred = self.model.generate(
                    test_batch,
                    node_type,
                    num_points=ph,
                    sample=20,
                    bestof=True,
                    sampling=sampling,
                    step=step
                )

                predictions = traj_pred
                predictions_dict = {}
                for j, ts in enumerate(timesteps_o):
                    if ts not in predictions_dict:
                        predictions_dict[ts] = dict()
                    predictions_dict[ts][nodes[j]] = np.transpose(predictions[:, [j]], (1, 0, 2, 3))

                batch_error_dict = evaluation.compute_batch_statistics(
                    predictions_dict,
                    scene.dt,
                    max_hl=max_hl,
                    ph=ph,
                    node_type_enum=self.eval_env.NodeType,
                    kde=False,
                    map=None,
                    best_of=True,
                    prune_ph_to_future=True
                )

                eval_ade_batch_errors = np.hstack((eval_ade_batch_errors, batch_error_dict[node_type]['ade']))
                eval_fde_batch_errors = np.hstack((eval_fde_batch_errors, batch_error_dict[node_type]['fde']))

        ade = np.mean(eval_ade_batch_errors)
        fde = np.mean(eval_fde_batch_errors)

        if self.config.dataset == "eth":
            ade = ade / 0.6
            fde = fde / 0.6
        elif self.config.dataset == "sdd":
            ade = ade * 50
            fde = fde * 50

        print(f"Sampling: {sampling} Stride: {step}")
        print(f"Epoch {epoch} Best Of 20: ADE: {ade} FDE: {fde}")

    def _build(self):
        self._build_dir()
        self._build_encoder_config()
        self._build_encoder()
        self._build_model()
        self._build_train_loader()
        self._build_eval_loader()
        self._build_optimizer()

        # self._build_offline_scene_graph()
        print("> Everything built. Have fun :)")

    def _build_dir(self):
        self.model_dir = osp.join("./experiments", self.config.exp_name)
        self.log_writer = SummaryWriter(log_dir=self.model_dir)
        os.makedirs(self.model_dir, exist_ok=True)

        log_name = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
        log_name = f"{self.config.dataset}_{log_name}"

        log_dir = osp.join(self.model_dir, log_name)
        self.log = logging.getLogger()
        self.log.setLevel(logging.INFO)

        handler = logging.FileHandler(log_dir)
        handler.setLevel(logging.INFO)
        self.log.addHandler(handler)

        self.log.info("Config:")
        self.log.info(self.config)
        self.log.info("\n")
        self.log.info("Eval on:")
        self.log.info(self.config.dataset)
        self.log.info("\n")

        self.train_data_path = osp.join(self.config.data_dir, self.config.dataset + "_train.pkl")
        self.eval_data_path = osp.join(self.config.data_dir, self.config.dataset + "_test.pkl")
        print("> Directory built!")

    def _build_optimizer(self):
        self.optimizer = optim.Adam(
            [
                {'params': self.registrar.get_all_but_name_match('map_encoder').parameters()},
                {'params': self.model.parameters()}
            ],
            lr=self.config.lr
        )
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.98)
        print("> Optimizer built!")

    def _build_encoder_config(self):
        self.hyperparams = get_traj_hypers()
        self.hyperparams['enc_rnn_dim_edge'] = self.config.encoder_dim // 2
        self.hyperparams['enc_rnn_dim_edge_influence'] = self.config.encoder_dim // 2
        self.hyperparams['enc_rnn_dim_history'] = self.config.encoder_dim // 2
        self.hyperparams['enc_rnn_dim_future'] = self.config.encoder_dim // 2

        self._cache_and_validate_hyperparams()

        self.registrar = ModelRegistrar(self.model_dir, "cuda")

        if self.config.eval_mode:
            epoch = self.config.eval_at
            self.checkpoint = torch.load(
                osp.join(self.model_dir, f"{self.config.dataset}_epoch{epoch}.pt"),
                map_location="cpu"
            )
            self.registrar.load_models(self.checkpoint['encoder'])

        with open(self.train_data_path, 'rb') as f:
            self.train_env = dill.load(f, encoding='latin1')
        with open(self.eval_data_path, 'rb') as f:
            self.eval_env = dill.load(f, encoding='latin1')

    def _build_encoder(self):
        self.encoder = Trajectron(self.registrar, self.hyperparams, "cuda")
        self.encoder.set_environment(self.train_env)
        self.encoder.set_annealing_params()

    def _build_model(self):
        config = self.config
        model = AutoEncoder(config, encoder=self.encoder)

        self.model = model.cuda()
        if self.config.eval_mode:
            self.model.load_state_dict(self.checkpoint['ddpm'])

        print("> Model built!")

    def _build_train_loader(self):
        config = self.config
        self.train_scenes = []

        with open(self.train_data_path, 'rb') as f:
            train_env = dill.load(f, encoding='latin1')

        for attention_radius_override in config.override_attention_radius:
            node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
            train_env.attention_radius[(node_type1, node_type2)] = float(attention_radius)

        self.train_scenes = self.train_env.scenes
        self.train_scenes_sample_probs = self.train_env.scenes_freq_mult_prop if config.scene_freq_mult_train else None

        self.train_dataset = EnvironmentDataset(
            train_env,
            self.hyperparams['state'],
            self.hyperparams['pred_state'],
            scene_freq_mult=self.hyperparams['scene_freq_mult_train'],
            node_freq_mult=self.hyperparams['node_freq_mult_train'],
            hyperparams=self.hyperparams,
            min_history_timesteps=self.min_hl,
            min_future_timesteps=self.ph,
            return_robot=not self.config.incl_robot_node
        )

        self.train_data_loader = dict()
        for node_type_data_set in self.train_dataset:
            node_type_dataloader = utils.data.DataLoader(
                node_type_data_set,
                collate_fn=collate,
                pin_memory=True,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.preprocess_workers
            )
            self.train_data_loader[node_type_data_set.node_type] = node_type_dataloader

    def _build_eval_loader(self):
        config = self.config
        self.eval_scenes = []
        eval_scenes_sample_probs = None

        if config.eval_every is not None:
            with open(self.eval_data_path, 'rb') as f:
                self.eval_env = dill.load(f, encoding='latin1')

            for attention_radius_override in config.override_attention_radius:
                node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
                self.eval_env.attention_radius[(node_type1, node_type2)] = float(attention_radius)

            if self.eval_env.robot_type is None and self.hyperparams['incl_robot_node']:
                self.eval_env.robot_type = self.eval_env.NodeType[0]
                for scene in self.eval_env.scenes:
                    scene.add_robot_from_nodes(self.eval_env.robot_type)

            self.eval_scenes = self.eval_env.scenes

            eval_scenes_sample_probs = self.eval_env.scenes_freq_mult_prop if config.scene_freq_mult_eval else None

            self.eval_dataset = EnvironmentDataset(
                self.eval_env,
                self.hyperparams['state'],
                self.hyperparams['pred_state'],
                scene_freq_mult=self.hyperparams['scene_freq_mult_eval'],
                node_freq_mult=self.hyperparams['node_freq_mult_eval'],
                hyperparams=self.hyperparams,
                min_history_timesteps=self.min_hl,
                min_future_timesteps=self.ph,
                return_robot=not config.incl_robot_node
            )

            self.eval_data_loader = dict()
            for node_type_data_set in self.eval_dataset:
                node_type_dataloader = utils.data.DataLoader(
                    node_type_data_set,
                    collate_fn=collate,
                    pin_memory=True,
                    batch_size=config.eval_batch_size,
                    shuffle=True,
                    num_workers=config.preprocess_workers
                )
                self.eval_data_loader[node_type_data_set.node_type] = node_type_dataloader

        print("> Dataset built!")

    def _build_offline_scene_graph(self):
        if self.hyperparams['offline_scene_graph'] == 'yes':
            print(f"Offline calculating scene graphs")
            for i, scene in enumerate(self.train_scenes):
                scene.calculate_scene_graph(
                    self.train_env.attention_radius,
                    self.hyperparams['edge_addition_filter'],
                    self.hyperparams['edge_removal_filter']
                )
                print(f"Created Scene Graph for Training Scene {i}")

            for i, scene in enumerate(self.eval_scenes):
                scene.calculate_scene_graph(
                    self.eval_env.attention_radius,
                    self.hyperparams['edge_addition_filter'],
                    self.hyperparams['edge_removal_filter']
                )
                print(f"Created Scene Graph for Evaluation Scene {i}")