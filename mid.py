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
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon


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
                "pred": "#2f6f95",  # Darker Blue
                "curr": "#011f4b",  # hist와 동일하게 유지
            },
            # 세트 2 (빨강 계열)
            {
                "hist": "#4d0000",  # Darkest Maroon
                "gt":   "#990000",  # Dark Red
                "pred": "#b03030",  # Darker Red
                "curr": "#4d0000",
            },
            # 세트 3 (초록 계열)
            {
                "hist": "#003300",  # Darkest Deep Green
                "gt":   "#1a8c1a",  # Dark Green
                "pred": "#3f8f3f",  # Darker Green
                "curr": "#003300",
            },
            # 세트 4 (보라 계열)
            {
                "hist": "#2a0a4d",  # Darkest Indigo/Purple
                "gt":   "#6a3d9a",  # Dark Purple
                "pred": "#7d52b3",  # Darker Purple
                "curr": "#2a0a4d",
            },
            # 세트 5 (오렌지/브라운 계열)
            {
                "hist": "#4d1a00",  # Darkest Brown
                "gt":   "#cc5200",  # Dark Orange/Rust
                "pred": "#d9872e",  # Darker Orange
                "curr": "#4d1a00",
            },
            # 세트 6 (청록/틸 계열)
            {
                "hist": "#003333",  # Darkest Teal
                "gt":   "#008080",  # Teal
                "pred": "#3f9f9f",  # Darker Teal
                "curr": "#003333",
            },
        ]
        return color_sets[node_idx % len(color_sets)]

    @staticmethod
    def _compute_yaw_from_positions(traj, default_yaw=0.0, speed_eps=1e-4):
        """
        Estimate yaw (rad) from a position trajectory.
        Low-speed segments keep the previous yaw to avoid unstable flips.
        """
        traj = np.asarray(traj, dtype=np.float32)
        if traj.ndim != 2 or traj.shape[0] == 0:
            return np.zeros((0,), dtype=np.float32)

        n = traj.shape[0]
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

    @staticmethod
    def _draw_vehicle_rectangles(
        ax,
        traj,
        yaws,
        car_length,
        car_width,
        color,
        alpha=0.2,
        linewidth=0.6,
        zorder=2,
        stride=1,
        edge_style="dark",
        overlap_darkening=False,
    ):
        """
        Draw oriented vehicle rectangles at each trajectory point.
        """
        traj = np.asarray(traj, dtype=np.float32)
        yaws = np.asarray(yaws, dtype=np.float32)
        if traj.ndim != 2 or traj.shape[0] == 0:
            return

        stride = max(1, int(stride))
        half_l = 0.5 * float(car_length)
        half_w = 0.5 * float(car_width)

        local_corners = np.array(
            [
                [half_l, half_w],
                [half_l, -half_w],
                [-half_l, -half_w],
                [-half_l, half_w],
            ],
            dtype=np.float32,
        )
        base_rgb = np.array(mcolors.to_rgb(color), dtype=np.float32)
        blend = float(np.clip(alpha, 0.0, 1.0))

        if overlap_darkening:
            fill_color = base_rgb
            patch_alpha = blend
        else:
            # Avoid overlap darkening by drawing opaque pre-blended color.
            fill_color = base_rgb * blend + (1.0 - blend)
            patch_alpha = 1.0

        if edge_style == "none":
            edge_color = "none"
            edge_width = 0.0
        else:
            edge_color = np.clip(base_rgb * 0.45, 0.0, 1.0)
            edge_width = max(0.40, float(linewidth) * 0.70)

        for i in range(0, traj.shape[0], stride):
            x, y = traj[i]
            yaw = float(yaws[i]) if i < yaws.shape[0] else float(yaws[-1]) if yaws.size > 0 else 0.0

            c = np.cos(yaw)
            s = np.sin(yaw)
            rot = np.array([[c, -s], [s, c]], dtype=np.float32)
            world_corners = local_corners @ rot.T + np.array([x, y], dtype=np.float32)

            patch = Polygon(
                world_corners,
                closed=True,
                facecolor=fill_color,
                edgecolor=edge_color,
                alpha=patch_alpha,
                linewidth=edge_width,
                zorder=zorder,
            )
            ax.add_patch(patch)

    @staticmethod
    def _vehicle_box_corners(center_xy, yaw, car_length, car_width):
        half_l = 0.5 * float(car_length)
        half_w = 0.5 * float(car_width)
        local_corners = np.array(
            [
                [half_l, half_w],
                [half_l, -half_w],
                [-half_l, -half_w],
                [-half_l, half_w],
            ],
            dtype=np.float32,
        )
        c = np.cos(float(yaw))
        s = np.sin(float(yaw))
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)
        center_xy = np.asarray(center_xy, dtype=np.float32).reshape(2)
        return local_corners @ rot.T + center_xy

    @staticmethod
    def _obb_intersect(corners_a, corners_b, eps=1e-6):
        corners_a = np.asarray(corners_a, dtype=np.float32).reshape(4, 2)
        corners_b = np.asarray(corners_b, dtype=np.float32).reshape(4, 2)

        axes = []
        for corners in (corners_a, corners_b):
            for i in range(4):
                edge = corners[(i + 1) % 4] - corners[i]
                normal = np.array([-edge[1], edge[0]], dtype=np.float32)
                norm = np.linalg.norm(normal)
                if norm > eps:
                    axes.append(normal / norm)

        for axis in axes:
            proj_a = corners_a @ axis
            proj_b = corners_b @ axis
            if (proj_a.max() < proj_b.min() - eps) or (proj_b.max() < proj_a.min() - eps):
                return False
        return True

    def _find_first_collision_timestep(
        self,
        traj_a,
        yaws_a,
        traj_b,
        yaws_b,
        car_length,
        car_width,
    ):
        traj_a = np.asarray(traj_a, dtype=np.float32)
        traj_b = np.asarray(traj_b, dtype=np.float32)
        yaws_a = np.asarray(yaws_a, dtype=np.float32).reshape(-1)
        yaws_b = np.asarray(yaws_b, dtype=np.float32).reshape(-1)

        n = min(len(traj_a), len(traj_b), len(yaws_a), len(yaws_b))
        if n <= 0:
            return None

        for t_idx in range(n):
            box_a = self._vehicle_box_corners(traj_a[t_idx], yaws_a[t_idx], car_length, car_width)
            box_b = self._vehicle_box_corners(traj_b[t_idx], yaws_b[t_idx], car_length, car_width)
            if self._obb_intersect(box_a, box_b):
                return t_idx
        return None

    @staticmethod
    def _collision_mode_local_point(mode_id, car_length, car_width):
        try:
            mode = int(mode_id)
        except (TypeError, ValueError):
            return None

        col = mode // 10
        row = mode % 10
        valid = (1 <= col <= 5) and ((row in (1, 3)) or (row == 2 and col in (1, 5)))
        if not valid:
            return None

        # User-defined convention (vehicle forward direction basis):
        # 11 12 13
        # 21    23
        # 31    33
        # 41    43
        # 51 52 53
        # first digit: front(1) -> rear(5), second digit: left(1), center(2), right(3)
        x_local = (3.0 - float(col)) * (float(car_length) / 4.0)
        y_local = (2.0 - float(row)) * (float(car_width) / 2.0)
        return np.array([x_local, y_local], dtype=np.float32)

    def _estimate_collision_mode_from_pair(
        self,
        ego_center_xy,
        ego_yaw,
        opp_center_xy,
        car_length,
        car_width,
        eps=1e-6,
    ):
        """
        Estimate ego-side collision mode from ego/opponent centers at collision timestep.
        We project the vector toward opponent center onto ego box boundary, then pick
        the closest valid mode point among {11,12,13,21,23,31,33,41,43,51,52,53}.
        """
        ego_center = np.asarray(ego_center_xy, dtype=np.float32).reshape(2)
        opp_center = np.asarray(opp_center_xy, dtype=np.float32).reshape(2)

        # World -> ego local.
        c = np.cos(float(ego_yaw))
        s = np.sin(float(ego_yaw))
        rot_t = np.array([[c, s], [-s, c]], dtype=np.float32)  # R^T
        rel_local = rot_t @ (opp_center - ego_center)

        dx = float(rel_local[0])
        dy = float(rel_local[1])
        half_l = 0.5 * float(car_length)
        half_w = 0.5 * float(car_width)

        # If almost same center, fallback to front-center candidate.
        if (abs(dx) + abs(dy)) < eps:
            dx, dy = 1.0, 0.0

        sx = (half_l / max(abs(dx), eps)) if abs(dx) > eps else float("inf")
        sy = (half_w / max(abs(dy), eps)) if abs(dy) > eps else float("inf")
        scale = min(sx, sy)
        contact_local = np.array([dx * scale, dy * scale], dtype=np.float32)
        contact_local[0] = np.clip(contact_local[0], -half_l, half_l)
        contact_local[1] = np.clip(contact_local[1], -half_w, half_w)

        valid_modes = (11, 12, 13, 21, 23, 31, 33, 41, 43, 51, 52, 53)
        best_mode = None
        best_dist = None
        for mode in valid_modes:
            pt = self._collision_mode_local_point(mode, car_length, car_width)
            if pt is None:
                continue
            d2 = float(np.sum((pt - contact_local) ** 2))
            if (best_dist is None) or (d2 < best_dist):
                best_dist = d2
                best_mode = int(mode)

        return best_mode

    @staticmethod
    def _mirror_collision_mode(mode_id):
        try:
            mode = int(mode_id)
        except (TypeError, ValueError):
            return None
        col = mode // 10
        row = mode % 10
        mirror_col = 6 - col
        mirror_mode = mirror_col * 10 + row
        return mirror_mode

    def _collision_mode_world_point(self, center_xy, yaw, mode_id, car_length, car_width):
        local_pt = self._collision_mode_local_point(mode_id, car_length, car_width)
        if local_pt is None:
            return None
        c = np.cos(float(yaw))
        s = np.sin(float(yaw))
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)
        center_xy = np.asarray(center_xy, dtype=np.float32).reshape(2)
        return local_pt @ rot.T + center_xy

    @staticmethod
    def _collision_mode_cell_indices(mode_id):
        try:
            mode = int(mode_id)
        except (TypeError, ValueError):
            return None

        col = mode // 10
        row = mode % 10
        valid = (1 <= col <= 5) and ((row in (1, 3)) or (row == 2 and col in (1, 5)))
        if not valid:
            return None

        # Keep first digit front->rear, flip second digit so 1=left, 3=right.
        return int(col - 1), int(3 - row)

    def _draw_collision_mode_grid(
        self,
        ax,
        center_xy,
        yaw,
        car_length,
        car_width,
        mode_id,
        cell_color="#ff2d2d",
        cell_alpha=0.65,
        grid_color="black",
        grid_alpha=0.62,
        grid_linewidth=0.55,
        zorder=6.1,
    ):
        """
        Draw a 5x3 collision grid on the ego vehicle and highlight the selected collision-mode cell.
        """
        cell_idx = self._collision_mode_cell_indices(mode_id)
        if cell_idx is None:
            return False

        col_idx, row_idx = cell_idx
        half_l = 0.5 * float(car_length)
        half_w = 0.5 * float(car_width)
        dx = float(car_length) / 5.0
        dy = float(car_width) / 3.0

        center_xy = np.asarray(center_xy, dtype=np.float32).reshape(2)
        c = np.cos(float(yaw))
        s = np.sin(float(yaw))
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)

        def to_world(local_pts):
            return local_pts @ rot.T + center_xy

        # Collision-mode cell mapping:
        # col(first digit): 1->front ... 5->rear
        # row(second digit): 1->left, 2->center(front/rear only), 3->right
        x_high = half_l - float(col_idx) * dx
        x_low = x_high - dx
        y_low = -half_w + float(row_idx) * dy
        y_high = y_low + dy

        cell_local = np.array(
            [
                [x_high, y_high],
                [x_high, y_low],
                [x_low, y_low],
                [x_low, y_high],
            ],
            dtype=np.float32,
        )
        cell_world = to_world(cell_local)
        ax.add_patch(
            Polygon(
                cell_world,
                closed=True,
                facecolor=cell_color,
                edgecolor="none",
                alpha=float(np.clip(cell_alpha, 0.0, 1.0)),
                zorder=zorder,
            )
        )

        # Draw 5x3 grid lines on the oriented vehicle box.
        for i in range(6):
            x = -half_l + i * dx
            seg_local = np.array([[x, -half_w], [x, half_w]], dtype=np.float32)
            seg_world = to_world(seg_local)
            ax.plot(
                seg_world[:, 0],
                seg_world[:, 1],
                color=grid_color,
                alpha=grid_alpha,
                linewidth=grid_linewidth,
                zorder=zorder + 0.05,
            )

        for j in range(4):
            y = -half_w + j * dy
            seg_local = np.array([[-half_l, y], [half_l, y]], dtype=np.float32)
            seg_world = to_world(seg_local)
            ax.plot(
                seg_world[:, 0],
                seg_world[:, 1],
                color=grid_color,
                alpha=grid_alpha,
                linewidth=grid_linewidth,
                zorder=zorder + 0.05,
            )

        return True

    def train(self):
        grad_clip = float(getattr(self.config, "grad_clip", self.hyperparams.get("grad_clip", 1.0)))

        for epoch in range(1, self.config.epochs + 1):
            self.train_dataset.augment = self.config.augment
            for node_type, data_loader in self.train_data_loader.items():
                pbar = tqdm(data_loader, ncols=80)
                for batch in pbar:
                    self.optimizer.zero_grad()
                    train_loss = self.model.get_loss(batch, node_type)
                    if not torch.isfinite(train_loss):
                        pbar.set_description(f"Epoch {epoch}, {node_type} Loss: non-finite (skip)")
                        self.optimizer.zero_grad(set_to_none=True)
                        continue

                    pbar.set_description(f"Epoch {epoch}, {node_type} Loss: {train_loss.item():.2f}")
                    train_loss.backward()

                    if grad_clip > 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                        if not torch.isfinite(grad_norm):
                            pbar.set_description(f"Epoch {epoch}, {node_type} grad-norm non-finite (skip)")
                            self.optimizer.zero_grad(set_to_none=True)
                            continue

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
        car_width = float(getattr(self.config, "car_width", 1.825))
        car_length = float(getattr(self.config, "car_length", 4.650))
        draw_vehicle_boxes = bool(getattr(self.config, "viz_vehicle_boxes_enabled", True))
        box_stride = max(1, int(getattr(self.config, "viz_vehicle_box_stride", 1)))
        viz_collision_mode_guidance = bool(getattr(self.config, "viz_collision_mode_guidance_enabled", False))
        viz_collision_mode_guidance_apply_to_ego = bool(
            getattr(self.config, "viz_collision_mode_guidance_apply_to_ego", False)
        )
        viz_collision_mode_id = int(
            getattr(
                self.config,
                "viz_collision_mode_guidance_mode",
                getattr(self.config, "collision_mode_guidance_mode", 11),
            )
        )

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
            node_viz_data = []
            ego_guidance_state = None
            if viz_collision_mode_guidance and (ego_node is not None):
                ego_hist = ego_node.get(np.array([t - max_hl, t]), {'position': ['x', 'y']})
                ego_hist = ego_hist[~np.isnan(ego_hist).any(axis=1)]
                if len(ego_hist) >= 1:
                    ego_hist_yaw = self._compute_yaw_from_positions(ego_hist)
                    ego_yaw_now = float(ego_hist_yaw[-1]) if len(ego_hist_yaw) > 0 else 0.0
                    if len(ego_hist) >= 2:
                        dt_scene_default = float(self.config.data_dt)
                        dt_scene = max(float(getattr(scene, "dt", dt_scene_default)), 1e-6)
                        ego_vel_now = (ego_hist[-1] - ego_hist[-2]) / dt_scene
                    else:
                        ego_vel_now = np.zeros((2,), dtype=np.float32)
                    ego_guidance_state = {
                        "pos": np.asarray(ego_hist[-1], dtype=np.float32),
                        "yaw": float(ego_yaw_now),
                        "vel": np.asarray(ego_vel_now, dtype=np.float32),
                    }

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

                guidance_override = None
                if viz_collision_mode_guidance and (ego_guidance_state is not None):
                    if (node is not ego_node) or viz_collision_mode_guidance_apply_to_ego:
                        guidance_override = {
                            "enabled": True,
                            "collision_mode_enabled": True,
                            "collision_mode_id": viz_collision_mode_id,
                            "collision_mode_target_position": ego_guidance_state["pos"],
                            "collision_mode_target_yaw": ego_guidance_state["yaw"],
                            "collision_mode_target_velocity": ego_guidance_state["vel"],
                            "collision_mode_target_length": car_length,
                            "collision_mode_target_width": car_width,
                        }

                with torch.no_grad():
                    pred_pack = self.model.generate(
                        batch,
                        node.type,
                        num_points=ph,
                        sample=num_samples,
                        bestof=True,
                        sampling=sampling,
                        step=1,
                        return_dynamics=True,
                        guidance_override=guidance_override,
                    )

                    if isinstance(pred_pack, dict):
                        predictions = pred_pack["position"][:, 0]    # [S, T, 2]
                        pred_yaws = pred_pack["yaw"][:, 0]           # [S, T]
                    else:
                        predictions = pred_pack[:, 0]                # fallback for legacy behavior
                        pred_yaws = None

                history_yaw = self._compute_yaw_from_positions(history)
                history_last_yaw = float(history_yaw[-1]) if len(history_yaw) > 0 else 0.0

                pred_items = []
                for pred_idx, pred in enumerate(predictions):
                    pred_traj = np.vstack((history[-1:], pred))
                    if pred_yaws is not None:
                        pred_yaw_full = np.concatenate(
                            [np.array([history_last_yaw], dtype=np.float32), pred_yaws[pred_idx]],
                            axis=0,
                        )
                    else:
                        pred_yaw_full = self._compute_yaw_from_positions(pred_traj, default_yaw=history_last_yaw)
                    pred_items.append((pred_traj, pred_yaw_full))

                # Swap GT/Prediction intensity roles for path rendering.
                pred_draw_color = colors["gt"]
                gt_draw_color = colors["pred"]
                shared_path_lw = 1.55

                for pred_traj, _ in pred_items:
                    ax.plot(
                        pred_traj[:, 0],
                        pred_traj[:, 1],
                        '-o',
                        color=pred_draw_color,
                        alpha=0.96,
                        linewidth=shared_path_lw,
                        markersize=2.0,
                        zorder=3,
                    )

                ax.plot(
                    history[:, 0],
                    history[:, 1],
                    '-o',
                    color=colors["hist"],
                    linewidth=3.2,
                    markersize=3,
                    zorder=4,
                )

                gt_traj = np.vstack((history[-1:], future))
                ax.plot(
                    gt_traj[:, 0],
                    gt_traj[:, 1],
                    '-',
                    color=gt_draw_color,
                    linewidth=shared_path_lw,
                    alpha=0.94,
                    zorder=4,
                )

                node_viz_data.append(
                    {
                        "node": node,
                        "colors": colors,
                        "pred_items": pred_items,
                    }
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

            if draw_vehicle_boxes and len(node_viz_data) > 0:
                representative = []
                for rec in node_viz_data:
                    pred_items = rec.get("pred_items", [])
                    if len(pred_items) == 0:
                        continue
                    rep_traj, rep_yaw = pred_items[0]
                    representative.append(
                        {
                            "node": rec["node"],
                            "colors": rec["colors"],
                            "traj": np.asarray(rep_traj, dtype=np.float32),
                            "yaw": np.asarray(rep_yaw, dtype=np.float32),
                        }
                    )

                if len(representative) > 0:
                    # Enforce exactly one box timestep per vehicle to avoid mixed-time rendering.
                    draw_t_by_idx = [None] * len(representative)
                    collision_event = None

                    ego_idx = None
                    for i, rec in enumerate(representative):
                        if rec["node"] is ego_node:
                            ego_idx = i
                            break
                    if ego_idx is None:
                        ego_idx = 0

                    if len(representative) == 1:
                        draw_t_by_idx[0] = int(max(0, len(representative[0]["traj"]) - 1))
                    else:
                        ego_rec = representative[ego_idx]
                        pair_stats = []
                        for i, rec in enumerate(representative):
                            if i == ego_idx:
                                continue
                            min_len = min(len(ego_rec["traj"]), len(ego_rec["yaw"]), len(rec["traj"]), len(rec["yaw"]))
                            if min_len <= 0:
                                continue
                            t_collision = self._find_first_collision_timestep(
                                ego_rec["traj"][:min_len],
                                ego_rec["yaw"][:min_len],
                                rec["traj"][:min_len],
                                rec["yaw"][:min_len],
                                car_length,
                                car_width,
                            )
                            pair_stats.append(
                                {
                                    "opp_idx": i,
                                    "t_collision": None if t_collision is None else int(t_collision),
                                    "t_last": int(min_len - 1),
                                }
                            )

                        collided_pairs = [p for p in pair_stats if p["t_collision"] is not None]
                        if len(collided_pairs) > 0:
                            # Use one global collision timestep (earliest) so ego/track are aligned in time.
                            first_collision = min(collided_pairs, key=lambda p: p["t_collision"])
                            collision_event = (first_collision["opp_idx"], first_collision["t_collision"])
                            t_collision = int(first_collision["t_collision"])
                            draw_t_by_idx[ego_idx] = t_collision
                            draw_t_by_idx[first_collision["opp_idx"]] = t_collision

                            # Non-colliding tracks: draw at each track's last available timestep.
                            for p in pair_stats:
                                opp_idx = int(p["opp_idx"])
                                if draw_t_by_idx[opp_idx] is None:
                                    draw_t_by_idx[opp_idx] = int(p["t_last"])
                        else:
                            # No collision in scene: draw all boxes at the last timestep.
                            for i, rec in enumerate(representative):
                                draw_t_by_idx[i] = int(max(0, min(len(rec["traj"]), len(rec["yaw"])) - 1))

                    # Fallback for any unresolved index.
                    for i, rec in enumerate(representative):
                        if draw_t_by_idx[i] is None:
                            draw_t_by_idx[i] = int(max(0, min(len(rec["traj"]), len(rec["yaw"])) - 1))

                    # Draw only selected timestep boxes (collision timestep or last timestep).
                    for i, rec in enumerate(representative):
                        max_idx = int(max(0, min(len(rec["traj"]), len(rec["yaw"])) - 1))
                        t_box = int(np.clip(draw_t_by_idx[i], 0, max_idx))
                        self._draw_vehicle_rectangles(
                            ax=ax,
                            traj=rec["traj"][t_box:t_box + 1],
                            yaws=rec["yaw"][t_box:t_box + 1],
                            car_length=car_length,
                            car_width=car_width,
                            color=rec["colors"]["pred"],
                            alpha=0.28,
                            linewidth=1.1,
                            zorder=2.2,
                            stride=1,
                            edge_style="dark",
                            overlap_darkening=False,
                        )

                    # Draw ego collision-mode grid highlight when collision is detected.
                    if collision_event is not None:
                        opp_idx, t_draw = collision_event
                        ego_rec = representative[ego_idx]
                        opp_rec = representative[opp_idx]
                        t_e = int(np.clip(t_draw, 0, min(len(ego_rec["traj"]), len(ego_rec["yaw"])) - 1))
                        t_o = int(np.clip(t_draw, 0, min(len(opp_rec["traj"]), len(opp_rec["yaw"])) - 1))
                        inferred_mode = self._estimate_collision_mode_from_pair(
                            ego_center_xy=ego_rec["traj"][t_e],
                            ego_yaw=ego_rec["yaw"][t_e],
                            opp_center_xy=opp_rec["traj"][t_o],
                            car_length=car_length,
                            car_width=car_width,
                        )
                        mode_to_draw = int(inferred_mode) if inferred_mode is not None else int(viz_collision_mode_id)
                        _ = self._draw_collision_mode_grid(
                            ax=ax,
                            center_xy=ego_rec["traj"][t_e],
                            yaw=ego_rec["yaw"][t_e],
                            car_length=car_length,
                            car_width=car_width,
                            mode_id=mode_to_draw,
                            cell_color="#ff2d2d",
                            cell_alpha=0.66,
                            grid_color="black",
                            grid_alpha=0.62,
                            grid_linewidth=0.55,
                            zorder=6.1,
                        )

                        ego_label = role_labels.get(ego_rec["node"], f"ID {ego_rec['node'].id}")
                        opp_label = role_labels.get(opp_rec["node"], f"ID {opp_rec['node'].id}")
                        print(
                            f"    [collision-box] {ego_label} & {opp_label} "
                            f"at pred_idx={int(t_draw)} (scene_t={int(t + t_draw)}), "
                            f"requested_mode={int(viz_collision_mode_id)}, inferred_mode={int(mode_to_draw)}"
                        )

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
            hist_line = mlines.Line2D([], [], color='black', linestyle='-', marker='o', markersize=4, linewidth=3.2, label='History')
            gt_line = mlines.Line2D([], [], color='dimgray', linestyle='-', linewidth=1.55, label='GT Future (solid)')
            pred_line = mlines.Line2D([], [], color='slategray', linestyle='-', marker='o', markersize=2.8, linewidth=1.55, label='Predictions (-o)')
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
        self.hyperparams = get_traj_hypers(self.config)
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
