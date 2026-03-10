import torch
from torch.nn import Module
import torch.nn as nn
from .encoders.trajectron import Trajectron
from .encoders import dynamics as dynamic_module
import models.diffusion as diffusion
from models.diffusion import DiffusionTraj,VarianceSchedule
import pdb

class AutoEncoder(Module):

    def __init__(self, config, encoder):
        super().__init__()
        self.config = config
        self.encoder = encoder
        self.diffnet = getattr(diffusion, config.diffnet)

        self.diffusion = DiffusionTraj(
            net = self.diffnet(point_dim=2, context_dim=config.encoder_dim, tf_layer=config.tf_layer, residual=False),
            var_sched = VarianceSchedule(
                num_steps=100,
                beta_T=5e-2,
                mode='linear'

            )
        )

    def encode(self, batch,node_type):
        z = self.encoder.get_latent(batch, node_type)
        return z
    
    def generate(self, batch, node_type, num_points, sample, bestof, flexibility=0.0, ret_traj=False, sampling="ddpm", step=1):
        dynamics = self.encoder.node_models_dict[node_type].dynamic
        encoded_x = self.encoder.get_latent(batch, node_type)
        
        # 1. Diffusion 모델은 이제 정규화된(-1 ~ 1) 스케일의 속도를 예측함
        predicted_y_st_vel = self.diffusion.sample(num_points, encoded_x, sample, bestof, flexibility=flexibility, ret_traj=ret_traj, sampling=sampling, step=step)
        
        # 2. 정규화 복원 (Destandardization) 로직 추가
        env = self.encoder.env
        pred_state = self.encoder.pred_state
        _, std = env.get_standardize_params(pred_state[node_type], node_type)
        
        # 텐서 연산을 위해 std를 GPU 텐서로 변환
        std_tensor = torch.tensor(std, device=predicted_y_st_vel.device, dtype=torch.float32)
        
        # 예측된 정규화 속도에 표준편차를 곱해 실제 미터(m/s) 스케일로 복원
        predicted_y_vel = predicted_y_st_vel * std_tensor
        
        # 3. 실제 스케일의 속도를 적분하여 위치로 변환
        predicted_y_pos = dynamics.integrate_samples(predicted_y_vel)
        return predicted_y_pos.cpu().detach().numpy()

    def get_loss(self, batch, node_type):
        (first_history_index,
         x_t, y_t, x_st_t, y_st_t,
         neighbors_data_st,
         neighbors_edge_value,
         robot_traj_st_t,
         map) = batch

        feat_x_encoded = self.encode(batch,node_type) 
        # [확실하게 틀렸던 부분 수정] y_t 대신 y_st_t를 사용해야 함!
        loss = self.diffusion.get_loss(y_st_t.cuda(), feat_x_encoded)
        return loss
