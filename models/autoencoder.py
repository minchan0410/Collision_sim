import math
import torch
from torch.nn import Module

import models.diffusion as diffusion
from models.diffusion import DiffusionTraj, VarianceSchedule


class AutoEncoder(Module):
    def __init__(self, config, encoder):
        super().__init__()
        self.config = config
        self.encoder = encoder
        self.diffnet = getattr(diffusion, config.diffnet)

        self.diffusion = DiffusionTraj(
            net=self.diffnet(point_dim=2, context_dim=config.encoder_dim, tf_layer=config.tf_layer, residual=False),
            var_sched=VarianceSchedule(
                num_steps=100,
                beta_T=5e-2,
                mode='linear'
            )
        )

    def _build_dynamics_guidance(self, dynamics, velocity_std):
        dynamics_enabled = bool(getattr(self.config, "dynamics_guidance_enabled", False))
        collision_enabled = bool(getattr(self.config, "collision_guidance_enabled", False))
        not_collision_enabled = bool(getattr(self.config, "not_collision_guidance_enabled", False))
        heading_enabled = bool(getattr(self.config, "heading_guidance_enabled", False))
        guidance_enabled = dynamics_enabled or collision_enabled or not_collision_enabled or heading_enabled
        if not guidance_enabled:
            return {"enabled": False}

        initial_velocity = None
        initial_position = None
        if hasattr(dynamics, "initial_conditions") and isinstance(dynamics.initial_conditions, dict):
            initial_velocity = dynamics.initial_conditions.get("vel", None)
            initial_position = dynamics.initial_conditions.get("pos", None)

        dyn_w = 1.0 if dynamics_enabled else 0.0
        if dynamics_enabled:
            guidance_scale = float(getattr(self.config, "dynamics_guidance_scale", 0.03))
            guidance_start_ratio = 0.0
            guidance_inner_steps = int(getattr(self.config, "dynamics_guidance_inner_steps", 1))
            guidance_max_grad_norm = float(getattr(self.config, "dynamics_guidance_max_grad_norm", 1.5))
        elif heading_enabled:
            guidance_scale = float(getattr(self.config, "heading_guidance_scale", 0.03))
            guidance_start_ratio = float(getattr(self.config, "heading_guidance_start_ratio", 0.0))
            guidance_inner_steps = int(getattr(self.config, "heading_guidance_inner_steps", 1))
            guidance_max_grad_norm = float(getattr(self.config, "heading_guidance_max_grad_norm", 4.0))
        else:
            # Allow interaction-only guidance even when dynamics_guidance_enabled=False.
            guidance_scale = float(getattr(self.config, "ttc_guidance_sampling_scale", 0.12))
            guidance_start_ratio = float(getattr(self.config, "ttc_guidance_start_ratio", 0.0))
            guidance_inner_steps = int(getattr(self.config, "ttc_guidance_inner_steps", 1))
            guidance_max_grad_norm = float(getattr(self.config, "ttc_guidance_max_grad_norm", 6.0))

        return {
            "enabled": True,
            "scale": guidance_scale,
            "start_ratio": guidance_start_ratio,
            "inner_steps": guidance_inner_steps,
            "max_grad_norm": guidance_max_grad_norm,
            "dt": float(getattr(dynamics, "dt", 1.0)),
            "velocity_std": velocity_std,
            "initial_velocity": initial_velocity,
            "initial_position": initial_position,
            "eps": float(getattr(self.config, "dynamics_guidance_eps", 1e-6)),
            "min_speed": float(getattr(self.config, "dynamics_guidance_min_speed", 0.5)),
            "max_accel": float(getattr(self.config, "dynamics_guidance_max_accel", 6.0)),
            "max_steer_deg": float(getattr(self.config, "dynamics_guidance_max_steer_deg", 32.0)),
            "max_steer_rate_deg_s": float(getattr(self.config, "dynamics_guidance_max_steer_rate_deg_s", 120.0)),
            "weight_accel": dyn_w * float(getattr(self.config, "dynamics_guidance_weight_accel", 1.0)),
            "weight_steer": dyn_w * float(getattr(self.config, "dynamics_guidance_weight_steer", 1.0)),
            "weight_steer_rate": dyn_w * float(getattr(self.config, "dynamics_guidance_weight_steer_rate", 0.25)),
            "wheelbase": float(getattr(self.config, "dynamics_guidance_wheelbase", getattr(self.config, "car_length", 4.65) * 0.58)),
            # Directly suppress bicycle heading spin in near-stop regime.
            "low_speed_yaw_weight": dyn_w * float(getattr(self.config, "dynamics_guidance_low_speed_yaw_weight", 0.6)),
            "low_speed_yaw_threshold": float(getattr(self.config, "dynamics_guidance_low_speed_yaw_threshold", 0.3)),
            # Heading alignment guidance is activated per target by passing heading_target_yaw.
            "heading_enabled": False,
            "heading_weight": float(getattr(self.config, "heading_guidance_weight", 0.0)),
            "heading_min_speed": float(getattr(self.config, "heading_guidance_min_speed", 0.03)),
            "heading_focus_ratio": float(getattr(self.config, "heading_guidance_focus_ratio", 1.0)),
            # TTC interaction guidance, active only when a reference trajectory is provided.
            # If not-collision guidance is enabled, collision guidance is disabled to avoid conflicts.
            "collision_enabled": bool(getattr(self.config, "collision_guidance_enabled", False))
                                and not bool(getattr(self.config, "not_collision_guidance_enabled", False)),
            "not_collision_enabled": bool(getattr(self.config, "not_collision_guidance_enabled", False)),
            "ttc_focus_ratio": float(getattr(self.config, "ttc_guidance_focus_ratio", 1.0)),
            "ttc_collision_distance": float(getattr(self.config, "ttc_collision_distance", 0.75)),
            "ttc_collision_time": float(getattr(self.config, "ttc_collision_time", 0.6)),
            "ttc_collision_min_closing_speed": float(getattr(self.config, "ttc_collision_min_closing_speed", 0.75)),
            "ttc_safe_distance": float(getattr(self.config, "ttc_safe_distance", 4.0)),
            "ttc_safe_time": float(getattr(self.config, "ttc_safe_time", 2.0)),
            "ttc_safe_max_closing_speed": float(getattr(self.config, "ttc_safe_max_closing_speed", 0.25)),
            "ttc_safe_rel_speed": float(getattr(self.config, "ttc_safe_rel_speed", 0.0)),
            "ttc_weight_distance": float(getattr(self.config, "ttc_weight_distance", 1.0)),
            "ttc_weight_time": float(getattr(self.config, "ttc_weight_time", 1.0)),
            "ttc_weight_rel_speed": float(getattr(self.config, "ttc_weight_rel_speed", 0.5)),
            "ttc_collision_scale": float(
                getattr(
                    self.config,
                    "ttc_collision_scale",
                    getattr(self.config, "collision_guidance_global_scale", 1.0),
                )
            ),
            "ttc_not_collision_scale": float(
                getattr(
                    self.config,
                    "ttc_not_collision_scale",
                    getattr(self.config, "not_collision_guidance_global_scale", 1.0),
                )
            ),
            "ttc_scale_jitter": float(getattr(self.config, "ttc_guidance_scale_jitter", 0.0)),
            "ttc_distance_softmin_temp": float(getattr(self.config, "ttc_distance_softmin_temp", 0.25)),
            "ttc_time_softmin_temp": float(getattr(self.config, "ttc_time_softmin_temp", 0.15)),
            "ttc_speed_softmax_temp": float(getattr(self.config, "ttc_speed_softmax_temp", 0.25)),
            "ttc_near_distance_temp": float(getattr(self.config, "ttc_near_distance_temp", 0.5)),
            "ttc_closing_gate_temp": float(getattr(self.config, "ttc_closing_gate_temp", 0.25)),
            "ttc_closing_softplus_beta": float(getattr(self.config, "ttc_closing_softplus_beta", 5.0)),
            "safesim_ttc_time_bandwidth": float(getattr(self.config, "safesim_ttc_time_bandwidth", 1.0)),
            "safesim_ttc_distance_bandwidth": float(getattr(self.config, "safesim_ttc_distance_bandwidth", 1.0)),
            "safesim_ttc_min_velocity_diff": float(getattr(self.config, "safesim_ttc_min_velocity_diff", 0.1)),
            "safesim_collision_rel_speed_target": float(getattr(self.config, "safesim_collision_rel_speed_target", 0.0)),
            "safesim_not_collision_danger_margin": float(getattr(self.config, "safesim_not_collision_danger_margin", 0.15)),
            "xT_temperature": float(getattr(self.config, "sampling_xt_temperature", 1.0)),
        }

    @staticmethod
    def _merge_guidance(base_guidance, override_guidance):
        if not isinstance(base_guidance, dict):
            base_guidance = {}
        if not isinstance(override_guidance, dict) or len(override_guidance) == 0:
            return base_guidance
        merged = dict(base_guidance)
        merged.update(override_guidance)
        return merged

    def encode(self, batch, node_type):
        z = self.encoder.get_latent(batch, node_type)
        return z

    @staticmethod
    def _compute_yaw_from_velocity(velocity, initial_velocity=None, speed_eps=1e-4):
        """
        Convert velocity vectors to yaw angles (rad), while keeping yaw stable
        when speed is near zero by carrying forward the previous valid yaw.

        velocity: [S, B, T, 2]
        initial_velocity: [B, 2] or None
        """
        raw_yaw = torch.atan2(velocity[..., 1], velocity[..., 0])  # [S, B, T]
        speed = torch.norm(velocity, dim=-1)  # [S, B, T]

        samples, batch, horizon = raw_yaw.shape
        yaw = raw_yaw.clone()

        if initial_velocity is not None:
            if not torch.is_tensor(initial_velocity):
                initial_velocity = torch.tensor(initial_velocity, device=velocity.device, dtype=velocity.dtype)
            else:
                initial_velocity = initial_velocity.to(device=velocity.device, dtype=velocity.dtype)
            if initial_velocity.dim() == 1:
                initial_velocity = initial_velocity.unsqueeze(0)
            if initial_velocity.size(0) == 1 and batch > 1:
                initial_velocity = initial_velocity.expand(batch, -1)
            elif initial_velocity.size(0) != batch:
                initial_velocity = initial_velocity[:batch]

            init_yaw = torch.atan2(initial_velocity[:, 1], initial_velocity[:, 0]).unsqueeze(0).expand(samples, -1)
        else:
            init_yaw = raw_yaw[:, :, 0]

        prev_yaw = init_yaw
        for t in range(horizon):
            low_speed = speed[:, :, t] < speed_eps
            current = yaw[:, :, t]
            current = torch.where(low_speed, prev_yaw, current)
            yaw[:, :, t] = current
            prev_yaw = current

        return yaw

    @staticmethod
    def _wrap_to_pi(angle):
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    def _rollout_bicycle_projection(self, predicted_y_vel, dynamics, guidance=None):
        """
        Project velocity samples to a kinematically feasible bicycle-like rollout.
        This keeps the current model output space (vx, vy) but enforces turn-rate
        consistency with wheelbase/steering bounds during trajectory generation.
        """
        if not hasattr(dynamics, "initial_conditions") or not isinstance(dynamics.initial_conditions, dict):
            return None

        init_pos = dynamics.initial_conditions.get("pos", None)
        if init_pos is None:
            return None

        device = predicted_y_vel.device
        dtype = predicted_y_vel.dtype
        eps = 1e-6
        dt = max(float(getattr(dynamics, "dt", 1.0)), eps)

        wheelbase = float(getattr(self.config, "dynamics_guidance_wheelbase", getattr(self.config, "car_length", 4.65) * 0.58))
        wheelbase = max(wheelbase, eps)

        max_steer_deg = float(getattr(self.config, "dynamics_guidance_max_steer_deg", 32.0))
        max_steer_rad = abs(max_steer_deg) * (math.pi / 180.0)
        kappa_max = abs(math.tan(max_steer_rad)) / wheelbase

        low_speed_th = float(getattr(self.config, "dynamics_guidance_low_speed_yaw_threshold", 0.3))

        samples, batch, horizon, _ = predicted_y_vel.shape
        speed = torch.norm(predicted_y_vel, dim=-1).clamp_min(0.0)
        raw_yaw = torch.atan2(predicted_y_vel[..., 1], predicted_y_vel[..., 0])

        init_pos = torch.as_tensor(init_pos, device=device, dtype=dtype)
        if init_pos.dim() == 1:
            init_pos = init_pos.unsqueeze(0)
        if init_pos.size(0) == 1 and batch > 1:
            init_pos = init_pos.expand(batch, -1)
        elif init_pos.size(0) != batch:
            init_pos = init_pos[:batch]

        init_pos = init_pos.unsqueeze(0).expand(samples, -1, -1).clone()  # [S, B, 2]
        pos_prev = init_pos

        guided_init_yaw = None
        if isinstance(guidance, dict) and bool(guidance.get("heading_enabled", False)):
            target_yaw = guidance.get("heading_target_yaw", None)
            if target_yaw is not None:
                guided_init_yaw = torch.as_tensor(target_yaw, device=device, dtype=dtype)
                if guided_init_yaw.dim() == 0:
                    guided_init_yaw = guided_init_yaw.view(1)
                else:
                    guided_init_yaw = guided_init_yaw.reshape(guided_init_yaw.size(0), -1)[:, -1]
                if guided_init_yaw.size(0) == 1 and batch > 1:
                    guided_init_yaw = guided_init_yaw.expand(batch)
                elif guided_init_yaw.size(0) != batch:
                    guided_init_yaw = guided_init_yaw[:batch]

        init_vel = dynamics.initial_conditions.get("vel", None)
        if init_vel is not None:
            init_vel = torch.as_tensor(init_vel, device=device, dtype=dtype)
            if init_vel.dim() == 1:
                init_vel = init_vel.unsqueeze(0)
            if init_vel.size(0) == 1 and batch > 1:
                init_vel = init_vel.expand(batch, -1)
            elif init_vel.size(0) != batch:
                init_vel = init_vel[:batch]
            init_speed = torch.norm(init_vel, dim=-1)
            init_yaw = torch.atan2(init_vel[:, 1], init_vel[:, 0])
            if guided_init_yaw is not None:
                init_yaw = torch.where(init_speed < low_speed_th, guided_init_yaw, init_yaw)
            yaw_prev = init_yaw.unsqueeze(0).expand(samples, -1).clone()
        elif guided_init_yaw is not None:
            yaw_prev = guided_init_yaw.unsqueeze(0).expand(samples, -1).clone()
        else:
            yaw_prev = raw_yaw[:, :, 0].clone()

        proj_pos = []
        proj_vel = []
        proj_yaw = []

        for t in range(horizon):
            sp_t = speed[:, :, t]
            target_yaw = raw_yaw[:, :, t]
            yaw_delta = self._wrap_to_pi(target_yaw - yaw_prev)

            max_delta = sp_t * kappa_max * dt

            # Tensor-wise clamp for older PyTorch compatibility.
            yaw_delta = torch.maximum(torch.minimum(yaw_delta, max_delta), -max_delta)
            yaw_t = yaw_prev + yaw_delta

            if low_speed_th > 0:
                low_speed_mask = sp_t < low_speed_th
                yaw_t = torch.where(low_speed_mask, yaw_prev, yaw_t)

            vel_t = torch.stack([sp_t * torch.cos(yaw_t), sp_t * torch.sin(yaw_t)], dim=-1)
            pos_prev = pos_prev + vel_t * dt

            proj_vel.append(vel_t)
            proj_pos.append(pos_prev)
            proj_yaw.append(yaw_t)
            yaw_prev = yaw_t

        projected_vel = torch.stack(proj_vel, dim=2)  # [S, B, T, 2]
        projected_pos = torch.stack(proj_pos, dim=2)  # [S, B, T, 2]
        projected_yaw = torch.stack(proj_yaw, dim=2)  # [S, B, T]
        return projected_pos, projected_vel, projected_yaw

    def _compute_yaw_aux_loss(self, pred_vel_phys, gt_vel_phys):
        min_speed = float(getattr(self.config, "yaw_loss_min_speed", 0.5))
        speed_eps = float(getattr(self.config, "yaw_loss_speed_eps", 1e-6))

        pred_speed = torch.norm(pred_vel_phys, dim=-1)  # [B, T]
        gt_speed = torch.norm(gt_vel_phys, dim=-1)      # [B, T]

        # Use AND mask to avoid unstable gradients when one side speed is near zero.
        moving_mask = ((pred_speed > min_speed) & (gt_speed > min_speed)).float()

        if moving_mask.sum().item() <= 0:
            return pred_vel_phys.new_tensor(0.0)

        pred_speed_safe = pred_speed.clamp_min(min_speed + speed_eps)
        gt_speed_safe = gt_speed.clamp_min(min_speed + speed_eps)
        pred_dir = pred_vel_phys / pred_speed_safe.unsqueeze(-1)
        gt_dir = gt_vel_phys / gt_speed_safe.unsqueeze(-1)

        # Equivalent to 1 - cos(delta_yaw), but numerically more stable than atan2-based delta.
        cos_delta = torch.sum(pred_dir * gt_dir, dim=-1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        yaw_error = 1.0 - cos_delta
        return (yaw_error * moving_mask).sum() / moving_mask.sum().clamp_min(speed_eps)

    def generate(
        self,
        batch,
        node_type,
        num_points,
        sample,
        bestof,
        flexibility=0.0,
        ret_traj=False,
        sampling="ddpm",
        step=1,
        return_dynamics=False,
        guidance_override=None,
    ):
        dynamics = self.encoder.node_models_dict[node_type].dynamic
        encoded_x = self.encoder.get_latent(batch, node_type)

        env = self.encoder.env
        pred_state = self.encoder.pred_state
        _, std = env.get_standardize_params(pred_state[node_type], node_type)
        std_tensor = torch.as_tensor(std, device=encoded_x.device, dtype=torch.float32)

        guidance_cfg = self._build_dynamics_guidance(dynamics, std_tensor)
        guidance_cfg = self._merge_guidance(guidance_cfg, guidance_override)
        predicted_y_st_vel = self.diffusion.sample(
            num_points,
            encoded_x,
            sample,
            bestof,
            flexibility=flexibility,
            ret_traj=ret_traj,
            sampling=sampling,
            step=step,
            guidance=guidance_cfg,
        )

        predicted_y_vel = predicted_y_st_vel * std_tensor
        predicted_y_pos = dynamics.integrate_samples(predicted_y_vel)

        projected_yaw = None
        if bool(getattr(self.config, "bicycle_rollout_enabled", True)):
            rollout = self._rollout_bicycle_projection(predicted_y_vel, dynamics, guidance=guidance_cfg)
            if rollout is not None:
                predicted_y_pos, predicted_y_vel, projected_yaw = rollout

        if not return_dynamics:
            return predicted_y_pos.cpu().detach().numpy()

        initial_velocity = None
        if hasattr(dynamics, "initial_conditions") and isinstance(dynamics.initial_conditions, dict):
            initial_velocity = dynamics.initial_conditions.get("vel", None)

        speed = torch.norm(predicted_y_vel, dim=-1)  # [S, B, T]
        if projected_yaw is not None:
            yaw = projected_yaw
        else:
            yaw = self._compute_yaw_from_velocity(predicted_y_vel, initial_velocity=initial_velocity)

        return {
            "position": predicted_y_pos.cpu().detach().numpy(),   # [S, B, T, 2]
            "velocity": predicted_y_vel.cpu().detach().numpy(),   # [S, B, T, 2]
            "speed": speed.cpu().detach().numpy(),                # [S, B, T]
            "yaw": yaw.cpu().detach().numpy(),                    # [S, B, T] (rad)
        }

    def get_loss(self, batch, node_type):
        (
            first_history_index,
            x_t, y_t, x_st_t, y_st_t,
            neighbors_data_st,
            neighbors_edge_value,
            robot_traj_st_t,
            map,
        ) = batch

        feat_x_encoded = self.encode(batch, node_type)
        y_st_t = y_st_t.to(device=feat_x_encoded.device, dtype=feat_x_encoded.dtype)

        diff_out = self.diffusion.get_loss(y_st_t, feat_x_encoded, return_details=True)
        diffusion_loss = diff_out["loss"]

        if not bool(getattr(self.config, "yaw_loss_enabled", False)):
            return diffusion_loss

        env = self.encoder.env
        pred_state = self.encoder.pred_state
        _, std = env.get_standardize_params(pred_state[node_type], node_type)
        std_tensor = torch.as_tensor(std, device=y_st_t.device, dtype=y_st_t.dtype).view(1, 1, -1)

        pred_vel_phys = diff_out["x0_hat"] * std_tensor
        gt_vel_phys = y_st_t * std_tensor
        yaw_aux_loss = self._compute_yaw_aux_loss(pred_vel_phys, gt_vel_phys)

        yaw_weight = float(getattr(self.config, "yaw_loss_weight", 0.1))
        return diffusion_loss + yaw_weight * yaw_aux_loss
