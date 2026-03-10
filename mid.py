import os
import argparse
import torch
import dill
import pdb
import numpy as np
import os.path as osp
import logging
import time
from torch import nn, optim, utils
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm.auto import tqdm
import pickle

from dataset import EnvironmentDataset, collate, get_timesteps_data, get_node_timestep_data, restore
# from dataset import EnvironmentDataset, collate, get_timesteps_data, restore
from models.autoencoder import AutoEncoder
from models.trajectron import Trajectron
from utils.model_registrar import ModelRegistrar
from utils.trajectron_hypers import get_traj_hypers
import evaluation

import matplotlib
matplotlib.use("Agg") # 서버 터미널 환경에서 에러 없이 이미지를 저장하기 위함
import matplotlib.pyplot as plt

class MID():
    def __init__(self, config):
        self.config = config
        torch.backends.cudnn.benchmark = True
        self._build()

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
            
            # --- 수정된 부분 1: 평가 로직 (eval_every > 0 일 때만 실행) ---
            if getattr(self.config, 'eval_every', 0) > 0 and epoch % self.config.eval_every == 0:
                self.model.eval()

                node_type = "PEDESTRIAN"
                eval_ade_batch_errors = []
                eval_fde_batch_errors = []

                ph = self.hyperparams['prediction_horizon']
                max_hl = self.hyperparams['maximum_history_length']

                for i, scene in enumerate(self.eval_scenes):
                    print(f"----- Evaluating Scene {i + 1}/{len(self.eval_scenes)}")
                    for t in tqdm(range(0, scene.timesteps, 10)):
                        timesteps = np.arange(t, t+10)
                        batch = get_timesteps_data(env=self.eval_env, scene=scene, t=timesteps, node_type=node_type, state=self.hyperparams['state'],
                                       pred_state=self.hyperparams['pred_state'], edge_types=self.eval_env.get_edge_types(),
                                       min_ht=7, max_ht=self.hyperparams['maximum_history_length'], min_ft=ph,
                                       max_ft=ph, hyperparams=self.hyperparams)
                        if batch is None:
                            continue
                        test_batch = batch[0]
                        nodes = batch[1]
                        timesteps_o = batch[2]
                        traj_pred = self.model.generate(test_batch, node_type, num_points=ph, sample=20, bestof=True, sampling="ddpm", step=1) 

                        predictions = traj_pred
                        predictions_dict = {}
                        for j, ts in enumerate(timesteps_o):
                            if ts not in predictions_dict.keys():
                                predictions_dict[ts] = dict()
                            predictions_dict[ts][nodes[j]] = np.transpose(predictions[:, [j]], (1, 0, 2, 3))

                        batch_error_dict = evaluation.compute_batch_statistics(predictions_dict,
                                                                               scene.dt,
                                                                               max_hl=max_hl,
                                                                               ph=ph,
                                                                               node_type_enum=self.eval_env.NodeType,
                                                                               kde=False,
                                                                               map=None,
                                                                               best_of=True,
                                                                               prune_ph_to_future=True)

                        eval_ade_batch_errors = np.hstack((eval_ade_batch_errors, batch_error_dict[node_type]['ade']))
                        eval_fde_batch_errors = np.hstack((eval_fde_batch_errors, batch_error_dict[node_type]['fde']))

                ade = np.mean(eval_ade_batch_errors)
                fde = np.mean(eval_fde_batch_errors)

                print(f"Epoch {epoch} Best Of 20: ADE: {ade:.4f} FDE: {fde:.4f}")
                self.log.info(f"Best of 20: Epoch {epoch} ADE: {ade:.4f} FDE: {fde:.4f}")

                self.model.train()

            # --- 수정된 부분 2: save_pt_every 주기에 맞춘 저장 및 시각화 로직 ---
            save_pt_every = getattr(self.config, 'save_pt_every', 1) # yaml에 없으면 기본값 1
            if save_pt_every > 0 and epoch % save_pt_every == 0:
                checkpoint = {
                    'encoder': self.registrar.model_dict,
                    'ddpm': self.model.state_dict()
                 }
                torch.save(checkpoint, osp.join(self.model_dir, f"{self.config.dataset}_epoch{epoch}.pt"))
                print(f"> Epoch {epoch} Checkpoint Saved successfully!")

                # 체크포인트 저장 시 시각화 함께 수행
                if getattr(self.config, 'viz_enabled', True):
                    self._visualize_epoch(epoch)


    # --- 새로 추가된 부분: viz.py의 기능을 내장한 시각화 메서드 ---
    def _visualize_epoch(self, epoch):
        print(f"--- Generating Full Scene Visualizations for Epoch {epoch} ---")
        viz_dir = osp.join(self.model_dir, "viz_outputs")
        os.makedirs(viz_dir, exist_ok=True)
        
        num_samples = getattr(self.config, "viz_num_samples", 50)
        sampling = getattr(self.config, "sampling", "ddpm")
        
        ph = self.hyperparams['prediction_horizon']
        max_hl = self.hyperparams['maximum_history_length']
        node_type = "PEDESTRIAN"
        
        # 전체 씬을 모두 그리면 시간이 오래 걸리므로, 설정된 개수(예: 2개)의 씬만 평가
        num_scenes_to_viz = getattr(self.config, "viz_num_examples", 2)
        
        self.model.eval()
        for scene_idx, scene in enumerate(self.eval_scenes[:num_scenes_to_viz]):
            # 씬의 중간 지점 시간(t)을 하나 선택
            t = scene.timesteps // 2
            if t < max_hl or t > scene.timesteps - ph:
                t = max_hl # 안전한 t값으로 보정
                
            # 해당 시점 t에 존재하는 모든 차량(Node) 가져오기
            present = scene.present_nodes(np.array([t]), type=node_type, min_history_timesteps=max_hl, min_future_timesteps=ph)
            
            if t not in present or len(present[t]) == 0:
                continue
                
            # 하나의 큰 캔버스 생성
            fig, ax = plt.subplots(figsize=(10, 10))
            all_points_for_scale = []
            
            # 씬에 있는 '모든 차량'을 순회하며 하나의 Plot에 겹쳐 그리기
            for node in present[t]:
                history = node.get(np.array([t - max_hl, t]), {'position': ['x', 'y']})
                future = node.get(np.array([t + 1, t + ph]), {'position': ['x', 'y']})
                
                history = history[~np.isnan(history).any(axis=1)]
                future = future[~np.isnan(future).any(axis=1)]
                if len(history) < 2 or len(future) < 2: 
                    continue
                    
                scene_graph = scene.get_scene_graph(t, self.eval_env.attention_radius, self.hyperparams['edge_addition_filter'], self.hyperparams['edge_removal_filter'])
                edge_types = [edge_type for edge_type in self.eval_env.get_edge_types() if edge_type[0] == node.type]
                
                item = get_node_timestep_data(self.eval_env, scene, t, node, self.hyperparams['state'], self.hyperparams['pred_state'], edge_types, max_hl, ph, self.hyperparams, scene_graph=scene_graph)
                batch = collate([item])
                
                with torch.no_grad():
                    predictions = self.model.generate(batch, node.type, num_points=ph, sample=num_samples, bestof=True, sampling=sampling, step=1)
                    predictions = predictions[:, 0]
                
                # 예측 분포 (파란 궤적들)
                for pred in predictions:
                    pred_traj = np.vstack((history[-1:], pred))
                    ax.plot(pred_traj[:, 0], pred_traj[:, 1], '-o', color='#1f77b4', alpha=0.3, linewidth=1.5, markersize=3)
                
                # 과거 궤적 (검은 선)
                ax.plot(history[:, 0], history[:, 1], '-o', color='black', linewidth=2, markersize=4)
                
                # 정답 미래 궤적 (초록 선)
                gt_traj = np.vstack((history[-1:], future))
                ax.plot(gt_traj[:, 0], gt_traj[:, 1], '-o', color='#2ca02c', linewidth=2, markersize=4, alpha=0.7)
                
                # 현재 위치
                ax.scatter(history[-1, 0], history[-1, 1], color='black', s=80, edgecolors='white', linewidths=1.5, zorder=5)
                
                # 스케일 조정을 위해 포인트 수집
                all_points_for_scale.extend(history)
                all_points_for_scale.extend(future)
                all_points_for_scale.extend(predictions.reshape(-1, 2))
                
            if len(all_points_for_scale) == 0:
                plt.close(fig)
                continue
                
            # 화면에 모든 차량이 다 들어오도록 전체 스케일 동적 조정
            all_points = np.array(all_points_for_scale)
            x_min, y_min = np.min(all_points, axis=0)
            x_max, y_max = np.max(all_points, axis=0)
            center_x = (x_max + x_min) / 2.0
            center_y = (y_max + y_min) / 2.0
            max_range = max(15.0, x_max - x_min, y_max - y_min) # 최소 반경 15m 보장
            margin = max_range * 0.55
            
            ax.set_xlim(center_x - margin, center_x + margin)
            ax.set_ylim(center_y - margin, center_y + margin)
            ax.set_aspect('equal', adjustable='box')
            ax.grid(alpha=0.3)
            
            # 범례는 하나만 표시 (중복 방지)
            import matplotlib.lines as mlines
            hist_line = mlines.Line2D([], [], color='black', marker='o', markersize=4, label='History (Observed)')
            gt_line = mlines.Line2D([], [], color='#2ca02c', marker='o', markersize=4, label='Future (GT)')
            pred_line = mlines.Line2D([], [], color='#1f77b4', marker='o', markersize=3, label='Predictions')
            curr_point = mlines.Line2D([], [], color='white', marker='o', markerfacecolor='black', markersize=8, label='Current Pos')
            ax.legend(handles=[hist_line, gt_line, pred_line, curr_point], loc='best')
            
            ax.set_title(f"Epoch {epoch} | Full Scene: {scene.name} | t={t}")
            
            # 이미지 저장
            out_file = osp.join(viz_dir, f"epoch_{epoch}_full_scene_{scene.name}_t_{t}.png")
            fig.savefig(out_file, dpi=200, bbox_inches='tight')
            plt.close(fig)
            print(f"  -> Saved full scene visualization: {out_file}")
            
        self.model.train()


    def eval(self, sampling, step):
        epoch = self.config.eval_at

        self.log.info(f"Sampling: {sampling} Stride: {step}")

        node_type = "PEDESTRIAN"
        eval_ade_batch_errors = []
        eval_fde_batch_errors = []
        ph = self.hyperparams['prediction_horizon']
        max_hl = self.hyperparams['maximum_history_length']


        for i, scene in enumerate(self.eval_scenes):
            print(f"----- Evaluating Scene {i + 1}/{len(self.eval_scenes)}")
            for t in tqdm(range(0, scene.timesteps, 10)):
                timesteps = np.arange(t,t+10)
                batch = get_timesteps_data(env=self.eval_env, scene=scene, t=timesteps, node_type=node_type, state=self.hyperparams['state'],
                               pred_state=self.hyperparams['pred_state'], edge_types=self.eval_env.get_edge_types(),
                               min_ht=7, max_ht=self.hyperparams['maximum_history_length'], min_ft=12,
                               max_ft=12, hyperparams=self.hyperparams)
                if batch is None:
                    continue
                test_batch = batch[0]
                nodes = batch[1]
                timesteps_o = batch[2]
                traj_pred = self.model.generate(test_batch, node_type, num_points=12, sample=20,bestof=True, sampling=sampling, step=step) # B * 20 * 12 * 2

                predictions = traj_pred
                predictions_dict = {}
                for j, ts in enumerate(timesteps_o):
                    if ts not in predictions_dict.keys():
                        predictions_dict[ts] = dict()
                    predictions_dict[ts][nodes[j]] = np.transpose(predictions[:, [j]], (1, 0, 2, 3))



                batch_error_dict = evaluation.compute_batch_statistics(predictions_dict,
                                                                       scene.dt,
                                                                       max_hl=max_hl,
                                                                       ph=ph,
                                                                       node_type_enum=self.eval_env.NodeType,
                                                                       kde=False,
                                                                       map=None,
                                                                       best_of=True,
                                                                       prune_ph_to_future=True)

                eval_ade_batch_errors = np.hstack((eval_ade_batch_errors, batch_error_dict[node_type]['ade']))
                eval_fde_batch_errors = np.hstack((eval_fde_batch_errors, batch_error_dict[node_type]['fde']))



        ade = np.mean(eval_ade_batch_errors)
        fde = np.mean(eval_fde_batch_errors)

        if self.config.dataset == "eth":
            ade = ade/0.6
            fde = fde/0.6
        elif self.config.dataset == "sdd":
            ade = ade * 50
            fde = fde * 50
        print(f"Sampling: {sampling} Stride: {step}")
        print(f"Epoch {epoch} Best Of 20: ADE: {ade} FDE: {fde}")
        #self.log.info(f"Best of 20: Epoch {epoch} ADE: {ade} FDE: {fde}")


    def _build(self):
        self._build_dir()

        self._build_encoder_config()
        self._build_encoder()
        self._build_model()
        self._build_train_loader()
        self._build_eval_loader()

        self._build_optimizer()

        #self._build_offline_scene_graph()
        #pdb.set_trace()
        print("> Everything built. Have fun :)")

    def _build_dir(self):
        self.model_dir = osp.join("./experiments",self.config.exp_name)
        self.log_writer = SummaryWriter(log_dir=self.model_dir)
        os.makedirs(self.model_dir,exist_ok=True)
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

        self.train_data_path = osp.join(self.config.data_dir,self.config.dataset + "_train.pkl")
        self.eval_data_path = osp.join(self.config.data_dir,self.config.dataset + "_test.pkl")
        print("> Directory built!")

    def _build_optimizer(self):
        self.optimizer = optim.Adam([{'params': self.registrar.get_all_but_name_match('map_encoder').parameters()},
                                     {'params': self.model.parameters()}
                                    ],
                                    lr=self.config.lr)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer,gamma=0.98)
        print("> Optimizer built!")

    def _build_encoder_config(self):

        self.hyperparams = get_traj_hypers()
        self.hyperparams['enc_rnn_dim_edge'] = self.config.encoder_dim//2
        self.hyperparams['enc_rnn_dim_edge_influence'] = self.config.encoder_dim//2
        self.hyperparams['enc_rnn_dim_history'] = self.config.encoder_dim//2
        self.hyperparams['enc_rnn_dim_future'] = self.config.encoder_dim//2
        # registar
        self.registrar = ModelRegistrar(self.model_dir, "cuda")

        if self.config.eval_mode:
            epoch = self.config.eval_at
            checkpoint_dir = osp.join(self.model_dir, f"{self.config.dataset}_epoch{epoch}.pt")
            self.checkpoint = torch.load(osp.join(self.model_dir, f"{self.config.dataset}_epoch{epoch}.pt"), map_location = "cpu")

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
        """ Define Model """
        config = self.config
        model = AutoEncoder(config, encoder = self.encoder)

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

        self.train_dataset = EnvironmentDataset(train_env,
                                           self.hyperparams['state'],
                                           self.hyperparams['pred_state'],
                                           scene_freq_mult=self.hyperparams['scene_freq_mult_train'],
                                           node_freq_mult=self.hyperparams['node_freq_mult_train'],
                                           hyperparams=self.hyperparams,
                                           min_history_timesteps=1,
                                           min_future_timesteps=self.hyperparams['prediction_horizon'],
                                           return_robot=not self.config.incl_robot_node)
        self.train_data_loader = dict()
        for node_type_data_set in self.train_dataset:
            node_type_dataloader = utils.data.DataLoader(node_type_data_set,
                                                         collate_fn=collate,
                                                         pin_memory = True,
                                                         batch_size=self.config.batch_size,
                                                         shuffle=True,
                                                         num_workers=self.config.preprocess_workers)
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
                self.eval_env.robot_type = self.eval_env.NodeType[0]  # TODO: Make more general, allow the user to specify?
                for scene in self.eval_env.scenes:
                    scene.add_robot_from_nodes(self.eval_env.robot_type)

            # --- 수정된 부분 3: 559개 대신 맨 앞 5개 씬만 평가하도록 슬라이싱 적용 ---
            self.eval_scenes = self.eval_env.scenes[:5] 
            
            eval_scenes_sample_probs = self.eval_env.scenes_freq_mult_prop if config.scene_freq_mult_eval else None
            self.eval_dataset = EnvironmentDataset(self.eval_env,
                                              self.hyperparams['state'],
                                              self.hyperparams['pred_state'],
                                              scene_freq_mult=self.hyperparams['scene_freq_mult_eval'],
                                              node_freq_mult=self.hyperparams['node_freq_mult_eval'],
                                              hyperparams=self.hyperparams,
                                              min_history_timesteps=self.hyperparams['minimum_history_length'],
                                              min_future_timesteps=self.hyperparams['prediction_horizon'],
                                              return_robot=not config.incl_robot_node)
            self.eval_data_loader = dict()
            for node_type_data_set in self.eval_dataset:
                node_type_dataloader = utils.data.DataLoader(node_type_data_set,
                                                             collate_fn=collate,
                                                             pin_memory=True,
                                                             batch_size=config.eval_batch_size,
                                                             shuffle=True,
                                                             num_workers=config.preprocess_workers)
                self.eval_data_loader[node_type_data_set.node_type] = node_type_dataloader

        print("> Dataset built!")

    def _build_offline_scene_graph(self):
        if self.hyperparams['offline_scene_graph'] == 'yes':
            print(f"Offline calculating scene graphs")
            for i, scene in enumerate(self.train_scenes):
                scene.calculate_scene_graph(self.train_env.attention_radius,
                                            self.hyperparams['edge_addition_filter'],
                                            self.hyperparams['edge_removal_filter'])
                print(f"Created Scene Graph for Training Scene {i}")

            for i, scene in enumerate(self.eval_scenes):
                scene.calculate_scene_graph(self.eval_env.attention_radius,
                                            self.hyperparams['edge_addition_filter'],
                                            self.hyperparams['edge_removal_filter'])
                print(f"Created Scene Graph for Evaluation Scene {i}")