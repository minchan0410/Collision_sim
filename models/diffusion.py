import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList
import numpy as np
import torch.nn as nn
from .common import *
import pdb


class VarianceSchedule(Module):

    def __init__(self, num_steps, mode='linear', beta_1=1e-4, beta_T=5e-2, cosine_s=8e-3):
        super().__init__()
        assert mode in ('linear', 'cosine')
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == 'linear':
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)
        elif mode == 'cosine':
            timesteps = torch.arange(num_steps + 1) / num_steps + cosine_s
            alphas = timesteps / (1 + cosine_s) * math.pi / 2
            alphas = torch.cos(alphas).pow(2)
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = betas.clamp(max=0.999)

        betas = torch.cat([torch.zeros([1]), betas], dim=0)  # Padding

        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i - 1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex)
        self.register_buffer('sigmas_inflex', sigmas_inflex)

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps + 1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas


class DiffusionTraj(Module):

    def __init__(self, net, var_sched: VarianceSchedule):
        super().__init__()
        self.net = net
        self.var_sched = var_sched

    def get_loss(self, x_0, context, t=None, return_details=False):
        """
        Paper-faithful conditioning:
            epsilon_theta(y_k, k, x)

        Diffusion coefficients still use beta_k / alpha_bar_k,
        but the network time embedding receives timestep index k itself.
        """
        batch_size, _, point_dim = x_0.size()
        device = x_0.device

        if t is None:
            t = self.var_sched.uniform_sample_t(batch_size)

        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=torch.long, device=device)
        else:
            t = t.to(device=device, dtype=torch.long)

        alpha_bar = self.var_sched.alpha_bars[t].to(device)

        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)       # (B, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)   # (B, 1, 1)

        e_rand = torch.randn_like(x_0)  # (B, N, d)
        timestep = t.float() / self.var_sched.num_steps

        x_t = c0 * x_0 + c1 * e_rand
        e_theta = self.net(x_t, timestep=timestep, context=context)
        loss = F.mse_loss(e_theta.view(-1, point_dim), e_rand.view(-1, point_dim), reduction='mean')
        if not return_details:
            return loss

        # DDPM-style x0 reconstruction used by auxiliary training objectives.
        x0_hat = (x_t - c1 * e_theta) / c0
        return {
            "loss": loss,
            "x0_hat": x0_hat,
        }

    @staticmethod
    def _masked_mean(value, mask=None, eps=1e-6):
        if value.numel() == 0:
            return value.new_tensor(0.0)
        if mask is None:
            return value.mean()
        weighted = value * mask
        denom = mask.sum().clamp_min(eps)
        return weighted.sum() / denom

    @staticmethod
    def _wrap_to_pi(angle):
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    def _soft_limit_penalty(self, quantity, limit, mask=None):
        if quantity.numel() == 0:
            return quantity.new_tensor(0.0)
        if limit <= 0:
            return quantity.new_tensor(0.0)
        violation = F.relu(quantity - limit).pow(2)
        return self._masked_mean(violation, mask=mask)

    @staticmethod
    def _as_tensor(value, device, dtype):
        if value is None:
            return None
        if torch.is_tensor(value):
            return value.to(device=device, dtype=dtype)
        return torch.tensor(value, device=device, dtype=dtype)

    @staticmethod
    def _expand_batch_tensor(value, batch_size):
        if value is None:
            return None
        if value.size(0) == batch_size:
            return value
        if value.size(0) == 1 and batch_size > 1:
            return value.expand(batch_size, *value.shape[1:])
        return value[:batch_size]

    @staticmethod
    def _collision_mode_local_geometry(mode_id, length, width):
        """
        Collision mode definition (from Fig.2, Lee et al. IEEE Access 2024):
            first digit: front(1) ... rear(5)
            second digit: left(1), center(2; only for 1x/5x), right(3)
        Valid classes: 11,12,13,21,23,31,33,41,43,51,52,53
        """
        col = mode_id // 10
        row = mode_id % 10
        valid = (1 <= col <= 5) and ((row in (1, 3)) or (row == 2 and col in (1, 5)))
        if not valid:
            return None, None

        # col: 1..5 -> +L/2, +L/4, 0, -L/4, -L/2
        x_local = (3.0 - float(col)) * (length / 4.0)
        # row: 1,2,3 -> +W/2(left), 0, -W/2(right)
        y_local = (2.0 - float(row)) * (width / 2.0)
        local_point = torch.stack([x_local, y_local], dim=-1)

        # Outward normal in ego-local frame.
        nx = torch.zeros_like(x_local)
        ny = torch.zeros_like(y_local)
        if col == 1:
            nx = nx + 1.0
        elif col == 5:
            nx = nx - 1.0

        if row == 1:
            ny = ny + 1.0
        elif row == 3:
            ny = ny - 1.0

        local_normal = torch.stack([nx, ny], dim=-1)
        local_normal = local_normal / torch.norm(local_normal, dim=-1, keepdim=True).clamp_min(1e-6)
        return local_point, local_normal

    def _compute_collision_mode_objective(self, vel_phys, guidance):
        if not guidance or not bool(guidance.get("collision_mode_enabled", False)):
            return vel_phys.new_tensor(0.0)

        eps = float(guidance.get("eps", 1e-6))
        dt = max(float(guidance.get("dt", 1.0)), eps)
        batch_size, horizon, _ = vel_phys.shape
        if horizon <= 0:
            return vel_phys.new_tensor(0.0)

        weight = float(guidance.get("collision_mode_weight", 0.0))
        if weight <= 0:
            return vel_phys.new_tensor(0.0)

        mode_raw = guidance.get("collision_mode_id", 11)
        try:
            mode_id = int(mode_raw)
        except (TypeError, ValueError):
            return vel_phys.new_tensor(0.0)

        initial_pos = self._as_tensor(guidance.get("initial_position", None), vel_phys.device, vel_phys.dtype)
        if initial_pos is None:
            return vel_phys.new_tensor(0.0)
        if initial_pos.dim() == 1:
            initial_pos = initial_pos.unsqueeze(0)
        initial_pos = self._expand_batch_tensor(initial_pos, batch_size)
        if initial_pos.size(-1) != 2:
            return vel_phys.new_tensor(0.0)

        adv_pos = initial_pos.unsqueeze(1) + torch.cumsum(vel_phys * dt, dim=1)  # [B, T, 2]

        target_positions = self._as_tensor(guidance.get("collision_mode_target_positions", None), vel_phys.device, vel_phys.dtype)
        if target_positions is not None:
            if target_positions.dim() == 2:
                target_positions = target_positions.unsqueeze(0)
            target_positions = self._expand_batch_tensor(target_positions, batch_size)
            if target_positions.size(1) < horizon:
                pad = target_positions[:, -1:, :].expand(batch_size, horizon - target_positions.size(1), 2)
                target_positions = torch.cat([target_positions, pad], dim=1)
            else:
                target_positions = target_positions[:, :horizon, :]
        else:
            target_pos0 = self._as_tensor(guidance.get("collision_mode_target_position", None), vel_phys.device, vel_phys.dtype)
            if target_pos0 is None:
                return vel_phys.new_tensor(0.0)
            if target_pos0.dim() == 1:
                target_pos0 = target_pos0.unsqueeze(0)
            target_pos0 = self._expand_batch_tensor(target_pos0, batch_size)
            if target_pos0.size(-1) != 2:
                return vel_phys.new_tensor(0.0)

            target_vel = self._as_tensor(guidance.get("collision_mode_target_velocity", None), vel_phys.device, vel_phys.dtype)
            if target_vel is None:
                target_vel = torch.zeros_like(target_pos0)
            if target_vel.dim() == 1:
                target_vel = target_vel.unsqueeze(0)
            target_vel = self._expand_batch_tensor(target_vel, batch_size)
            if target_vel.size(-1) != 2:
                target_vel = torch.zeros_like(target_pos0)

            t_axis = torch.arange(1, horizon + 1, device=vel_phys.device, dtype=vel_phys.dtype).view(1, horizon, 1) * dt
            target_positions = target_pos0.unsqueeze(1) + target_vel.unsqueeze(1) * t_axis

        target_yaws = self._as_tensor(guidance.get("collision_mode_target_yaws", None), vel_phys.device, vel_phys.dtype)
        if target_yaws is not None:
            if target_yaws.dim() == 1:
                target_yaws = target_yaws.unsqueeze(0)
            target_yaws = self._expand_batch_tensor(target_yaws, batch_size)
            if target_yaws.size(1) < horizon:
                pad = target_yaws[:, -1:].expand(batch_size, horizon - target_yaws.size(1))
                target_yaws = torch.cat([target_yaws, pad], dim=1)
            else:
                target_yaws = target_yaws[:, :horizon]
        else:
            target_yaw = self._as_tensor(guidance.get("collision_mode_target_yaw", None), vel_phys.device, vel_phys.dtype)
            if target_yaw is None:
                # If yaw is not given, infer from target velocity when available.
                target_vel = self._as_tensor(guidance.get("collision_mode_target_velocity", None), vel_phys.device, vel_phys.dtype)
                if target_vel is None:
                    target_yaw = torch.zeros((batch_size,), device=vel_phys.device, dtype=vel_phys.dtype)
                else:
                    if target_vel.dim() == 1:
                        target_vel = target_vel.unsqueeze(0)
                    target_vel = self._expand_batch_tensor(target_vel, batch_size)
                    target_yaw = torch.atan2(target_vel[:, 1], target_vel[:, 0])
            if target_yaw.dim() == 0:
                target_yaw = target_yaw.view(1)
            if target_yaw.dim() == 1:
                target_yaw = target_yaw.unsqueeze(0) if target_yaw.size(0) == horizon else target_yaw.unsqueeze(1)
            if target_yaw.dim() == 2 and target_yaw.size(0) != batch_size:
                target_yaw = self._expand_batch_tensor(target_yaw, batch_size)
            if target_yaw.dim() == 2 and target_yaw.size(1) == 1:
                target_yaws = target_yaw.expand(batch_size, horizon)
            elif target_yaw.dim() == 2 and target_yaw.size(1) >= horizon:
                target_yaws = target_yaw[:, :horizon]
            elif target_yaw.dim() == 2 and target_yaw.size(1) < horizon:
                pad = target_yaw[:, -1:].expand(batch_size, horizon - target_yaw.size(1))
                target_yaws = torch.cat([target_yaw, pad], dim=1)
            else:
                target_yaws = target_yaw.reshape(batch_size, 1).expand(batch_size, horizon)

        length = self._as_tensor(guidance.get("collision_mode_target_length", 4.65), vel_phys.device, vel_phys.dtype)
        width = self._as_tensor(guidance.get("collision_mode_target_width", 1.825), vel_phys.device, vel_phys.dtype)
        if length.dim() == 0:
            length = length.view(1)
        if width.dim() == 0:
            width = width.view(1)
        length = self._expand_batch_tensor(length, batch_size).reshape(batch_size)
        width = self._expand_batch_tensor(width, batch_size).reshape(batch_size)

        local_point, local_normal = self._collision_mode_local_geometry(mode_id, length, width)
        if local_point is None or local_normal is None:
            return vel_phys.new_tensor(0.0)

        cos_yaw = torch.cos(target_yaws)
        sin_yaw = torch.sin(target_yaws)

        lp_x = local_point[:, 0].unsqueeze(1)
        lp_y = local_point[:, 1].unsqueeze(1)
        offset_x = lp_x * cos_yaw - lp_y * sin_yaw
        offset_y = lp_x * sin_yaw + lp_y * cos_yaw
        target_contact = target_positions + torch.stack([offset_x, offset_y], dim=-1)  # [B, T, 2]

        ln_x = local_normal[:, 0].unsqueeze(1)
        ln_y = local_normal[:, 1].unsqueeze(1)
        normal_x = ln_x * cos_yaw - ln_y * sin_yaw
        normal_y = ln_x * sin_yaw + ln_y * cos_yaw
        normal_world = torch.stack([normal_x, normal_y], dim=-1)
        normal_world = normal_world / torch.norm(normal_world, dim=-1, keepdim=True).clamp_min(eps)

        dist2 = torch.sum((adv_pos - target_contact).pow(2), dim=-1)  # [B, T]
        sigma = max(float(guidance.get("collision_mode_sigma", 1.5)), eps)
        use_softmin = bool(guidance.get("collision_mode_use_softmin", True))
        softmin_tau = max(float(guidance.get("collision_mode_softmin_temperature", 1.0)), eps)

        if use_softmin:
            time_weights = torch.softmax(-dist2 / softmin_tau, dim=1)  # [B, T]
            # Soft-argmin weighted distance (always non-negative and stable).
            dist2_reduced = torch.sum(time_weights * dist2, dim=1)
            vel_sel = torch.sum(time_weights.unsqueeze(-1) * vel_phys, dim=1)
            normal_sel = torch.sum(time_weights.unsqueeze(-1) * normal_world, dim=1)
            target_vel_sel = None
            target_vel_hist = (target_positions[:, 1:] - target_positions[:, :-1]) / dt if horizon > 1 else None
            if target_vel_hist is not None:
                target_vel_hist = torch.cat([target_vel_hist[:, :1], target_vel_hist], dim=1)
                target_vel_sel = torch.sum(time_weights.unsqueeze(-1) * target_vel_hist, dim=1)
        else:
            time_index = int(guidance.get("collision_mode_time_index", -1))
            if time_index < 0:
                time_index = horizon + time_index
            time_index = max(0, min(horizon - 1, time_index))
            dist2_reduced = dist2[:, time_index]
            vel_sel = vel_phys[:, time_index]
            normal_sel = normal_world[:, time_index]
            target_vel_sel = None
            if horizon > 1:
                if time_index == 0:
                    target_vel_sel = (target_positions[:, 1] - target_positions[:, 0]) / dt
                else:
                    target_vel_sel = (target_positions[:, time_index] - target_positions[:, time_index - 1]) / dt

        objective = weight * torch.mean(dist2_reduced / (sigma * sigma))

        # Optional approach-alignment term (adversary should move toward impact area inward normal).
        weight_approach = float(guidance.get("collision_mode_weight_approach", 0.0))
        if weight_approach > 0:
            approach_cos = float(guidance.get("collision_mode_approach_cos", 0.0))
            vel_dir = vel_sel / torch.norm(vel_sel, dim=-1, keepdim=True).clamp_min(eps)
            inward_normal = -normal_sel
            inward_normal = inward_normal / torch.norm(inward_normal, dim=-1, keepdim=True).clamp_min(eps)
            align = torch.sum(vel_dir * inward_normal, dim=-1)
            speed = torch.norm(vel_sel, dim=-1)
            moving_mask = (speed > max(float(guidance.get("min_speed", 0.0)), 0.05)).float()
            objective = objective + weight_approach * self._masked_mean(F.relu(approach_cos - align).pow(2), moving_mask)

        # Optional closing-speed term.
        weight_closing = float(guidance.get("collision_mode_weight_closing", 0.0))
        if weight_closing > 0:
            min_closing = float(guidance.get("collision_mode_min_closing_speed", 0.5))
            if target_vel_sel is None:
                target_vel_sel = torch.zeros_like(vel_sel)
            rel_vel = vel_sel - target_vel_sel
            inward_normal = -normal_sel
            inward_normal = inward_normal / torch.norm(inward_normal, dim=-1, keepdim=True).clamp_min(eps)
            closing_speed = torch.sum(rel_vel * inward_normal, dim=-1)
            objective = objective + weight_closing * torch.mean(F.relu(min_closing - closing_speed).pow(2))

        return objective

    def _compute_dynamics_objective(self, vel_phys, guidance):
        eps = float(guidance.get("eps", 1e-6))
        dt = max(float(guidance.get("dt", 1.0)), eps)
        min_speed = max(float(guidance.get("min_speed", 0.2)), eps)

        max_accel = float(guidance.get("max_accel", 6.0))
        max_jerk = float(guidance.get("max_jerk", 8.0))
        max_yaw_rate = float(guidance.get("max_yaw_rate", 0.6))
        max_curvature = float(guidance.get("max_curvature", 0.25))
        max_lateral_accel = float(guidance.get("max_lateral_accel", 4.5))
        max_slip_ratio = float(guidance.get("max_slip_ratio", 3.0))
        reverse_tolerance = float(guidance.get("reverse_tolerance", 0.25))

        weight_accel = float(guidance.get("weight_accel", 1.0))
        weight_jerk = float(guidance.get("weight_jerk", 0.6))
        weight_yaw_rate = float(guidance.get("weight_yaw_rate", 0.8))
        weight_curvature = float(guidance.get("weight_curvature", 1.0))
        weight_lateral_accel = float(guidance.get("weight_lateral_accel", 0.9))
        weight_slip = float(guidance.get("weight_slip", 0.4))
        weight_reverse = float(guidance.get("weight_reverse", 0.5))
        low_speed_yaw_weight = float(guidance.get("low_speed_yaw_weight", 0.0))
        low_speed_yaw_threshold = max(float(guidance.get("low_speed_yaw_threshold", min_speed)), eps)

        use_bicycle_curvature = bool(guidance.get("use_bicycle_curvature", False))
        curvature_limit = max_curvature
        if use_bicycle_curvature:
            wheelbase = max(float(guidance.get("wheelbase", 0.0)), eps)
            max_steer_deg = float(guidance.get("max_steer_deg", 32.0))
            max_steer_rad = abs(max_steer_deg) * (math.pi / 180.0)
            bicycle_kappa_max = abs(math.tan(max_steer_rad)) / wheelbase
            if max_curvature > 0:
                curvature_limit = min(max_curvature, bicycle_kappa_max)
            else:
                curvature_limit = bicycle_kappa_max

        speed = torch.norm(vel_phys, dim=-1).clamp_min(eps)  # [B, T]
        speed_pair = 0.5 * (speed[:, 1:] + speed[:, :-1]) if speed.size(1) > 1 else speed.new_zeros(speed.size(0), 0)
        moving_mask = (speed_pair > min_speed).float()

        objective = vel_phys.new_tensor(0.0)

        # Acceleration and jerk limits.
        acc = (vel_phys[:, 1:] - vel_phys[:, :-1]) / dt if vel_phys.size(1) > 1 else vel_phys.new_zeros(vel_phys.size(0), 0, vel_phys.size(2))
        acc_mag = torch.norm(acc, dim=-1) if acc.numel() > 0 else speed.new_zeros(speed.size(0), 0)
        if weight_accel > 0:
            objective = objective + weight_accel * self._soft_limit_penalty(acc_mag, max_accel, mask=moving_mask)

        if acc.size(1) > 1:
            jerk = (acc[:, 1:] - acc[:, :-1]) / dt
            jerk_mag = torch.norm(jerk, dim=-1)
            jerk_mask = (moving_mask[:, 1:] * moving_mask[:, :-1]) if moving_mask.size(1) > 1 else None
            if weight_jerk > 0:
                objective = objective + weight_jerk * self._soft_limit_penalty(jerk_mag, max_jerk, mask=jerk_mask)

        # Yaw-rate, curvature, and lateral acceleration.
        if vel_phys.size(1) > 1:
            heading = torch.atan2(vel_phys[..., 1], vel_phys[..., 0])
            yaw_delta = self._wrap_to_pi(heading[:, 1:] - heading[:, :-1])
            yaw_rate = torch.abs(yaw_delta) / dt

            if weight_yaw_rate > 0:
                static_yaw_limit = torch.full_like(yaw_rate, max(max_yaw_rate, 0.0))
                if curvature_limit > 0:
                    dynamic_yaw_limit = speed_pair * curvature_limit
                    yaw_limit = torch.minimum(static_yaw_limit, dynamic_yaw_limit)
                else:
                    yaw_limit = static_yaw_limit

                # Apply without moving mask so near-stop spin also gets penalized.
                yaw_violation = F.relu(yaw_rate - yaw_limit).pow(2)
                objective = objective + weight_yaw_rate * self._masked_mean(yaw_violation, mask=None)

            if weight_curvature > 0:
                curvature = yaw_rate / speed_pair.clamp_min(min_speed)
                objective = objective + weight_curvature * self._soft_limit_penalty(curvature, curvature_limit, mask=moving_mask)

            if weight_lateral_accel > 0:
                lateral_accel = yaw_rate * speed_pair
                objective = objective + weight_lateral_accel * self._soft_limit_penalty(
                    lateral_accel, max_lateral_accel, mask=moving_mask
                )

            if low_speed_yaw_weight > 0:
                # Direct near-stop anti-spin penalty.
                low_speed_mask = (speed_pair < low_speed_yaw_threshold).float()
                objective = objective + low_speed_yaw_weight * self._masked_mean(yaw_rate.pow(2), mask=low_speed_mask)

        # Slip-like proxy: lateral/longitudinal acceleration ratio.
        if acc.numel() > 0 and weight_slip > 0:
            vel_ref = vel_phys[:, 1:]
            vel_hat = vel_ref / torch.norm(vel_ref, dim=-1, keepdim=True).clamp_min(min_speed)
            accel_long = torch.sum(acc * vel_hat, dim=-1)
            accel_lat = torch.norm(acc - accel_long.unsqueeze(-1) * vel_hat, dim=-1)
            slip_ratio = accel_lat / (torch.abs(accel_long) + eps)
            objective = objective + weight_slip * self._soft_limit_penalty(slip_ratio, max_slip_ratio, mask=moving_mask)

        # Backward-motion proxy against current heading direction.
        if weight_reverse > 0:
            initial_velocity = guidance.get("initial_velocity", None)
            if initial_velocity is None:
                initial_velocity = vel_phys[:, 0]
            elif not torch.is_tensor(initial_velocity):
                initial_velocity = torch.tensor(initial_velocity, device=vel_phys.device, dtype=vel_phys.dtype)
            else:
                initial_velocity = initial_velocity.to(device=vel_phys.device, dtype=vel_phys.dtype)

            if initial_velocity.dim() == 1:
                initial_velocity = initial_velocity.unsqueeze(0)
            if initial_velocity.size(0) == 1 and vel_phys.size(0) > 1:
                initial_velocity = initial_velocity.expand(vel_phys.size(0), -1)
            elif initial_velocity.size(0) != vel_phys.size(0):
                initial_velocity = initial_velocity[:vel_phys.size(0)]

            initial_speed = torch.norm(initial_velocity, dim=-1, keepdim=True)
            fallback_velocity = vel_phys[:, 0]
            fallback_speed = torch.norm(fallback_velocity, dim=-1, keepdim=True)
            fallback_dir = fallback_velocity / fallback_speed.clamp_min(min_speed)
            initial_dir = initial_velocity / initial_speed.clamp_min(min_speed)
            use_fallback = (initial_speed < min_speed).float()
            travel_dir = use_fallback * fallback_dir + (1.0 - use_fallback) * initial_dir
            travel_dir = travel_dir / torch.norm(travel_dir, dim=-1, keepdim=True).clamp_min(min_speed)

            forward_speed = torch.sum(vel_phys * travel_dir.unsqueeze(1), dim=-1)
            reverse_violation = F.relu(-(forward_speed + reverse_tolerance)).pow(2)
            reverse_mask = (speed > min_speed).float()
            objective = objective + weight_reverse * self._masked_mean(reverse_violation, reverse_mask)

        collision_objective = self._compute_collision_mode_objective(vel_phys, guidance)
        if torch.isfinite(collision_objective):
            objective = objective + collision_objective

        return objective

    def _apply_dynamics_guidance(self, x_next, guidance):
        if not guidance or not guidance.get("enabled", False):
            return x_next

        scale = float(guidance.get("scale", 0.0))
        if scale <= 0:
            return x_next

        velocity_std = guidance.get("velocity_std", None)
        if velocity_std is None:
            return x_next
        if not torch.is_tensor(velocity_std):
            velocity_std = torch.tensor(velocity_std, device=x_next.device, dtype=x_next.dtype)
        velocity_std = velocity_std.to(device=x_next.device, dtype=x_next.dtype).view(1, 1, -1)

        inner_steps = max(1, int(guidance.get("inner_steps", 1)))
        max_grad_norm = float(guidance.get("max_grad_norm", 0.0))

        x_guided = x_next
        for _ in range(inner_steps):
            with torch.enable_grad():
                x_var = x_guided.detach().requires_grad_(True)
                vel_phys = x_var * velocity_std
                objective = self._compute_dynamics_objective(vel_phys, guidance)
                if not torch.isfinite(objective):
                    return x_guided
                grad = torch.autograd.grad(objective, x_var, create_graph=False, retain_graph=False)[0]

            if grad is None:
                return x_guided

            if max_grad_norm > 0:
                grad_flat = grad.reshape(grad.size(0), -1)
                grad_norm = torch.norm(grad_flat, dim=-1, keepdim=True).clamp_min(1e-12)
                clip = torch.clamp(max_grad_norm / grad_norm, max=1.0)
                grad = grad * clip.view(-1, 1, 1)

            x_guided = (x_var - scale * grad).detach()

        return x_guided

    def sample(self, num_points, context, sample, bestof, point_dim=2,
               flexibility=0.0, ret_traj=False, sampling="ddpm", step=100, guidance=None):
        guidance = guidance or {}
        guidance_enabled = bool(guidance.get("enabled", False))
        start_ratio = float(guidance.get("start_ratio", 0.0)) if guidance_enabled else 0.0
        start_ratio = min(max(start_ratio, 0.0), 1.0)
        guidance_start_t = max(1, int(round(self.var_sched.num_steps * (1.0 - start_ratio))))

        traj_list = []
        for _ in range(sample):
            batch_size = context.size(0)
            device = context.device

            # if bestof:
            #     x_T = torch.randn([batch_size, num_points, point_dim], device=device)
            # else:
            #     x_T = torch.zeros([batch_size, num_points, point_dim], device=device)

            x_T = torch.randn([batch_size, num_points, point_dim], device=device)

            traj = {self.var_sched.num_steps: x_T}
            stride = step

            for t in range(self.var_sched.num_steps, 0, -stride):
                x_t = traj[t]
                z = torch.randn_like(x_t) if t > 1 else torch.zeros_like(x_t)

                alpha = self.var_sched.alphas[t].to(device)
                alpha_bar = self.var_sched.alpha_bars[t].to(device)
                alpha_bar_next = self.var_sched.alpha_bars[t - stride].to(device)
                sigma = self.var_sched.get_sigmas(t, flexibility).to(device)

                c0 = 1.0 / torch.sqrt(alpha)
                c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

                timestep = torch.full((batch_size,), float(t) / self.var_sched.num_steps, device=device)
                e_theta = self.net(x_t, timestep=timestep, context=context)

                if sampling == "ddpm":
                    x_next = c0 * (x_t - c1 * e_theta) + sigma * z
                elif sampling == "ddim":
                    x0_t = (x_t - e_theta * torch.sqrt(1 - alpha_bar)) / torch.sqrt(alpha_bar)
                    x_next = torch.sqrt(alpha_bar_next) * x0_t + torch.sqrt(1 - alpha_bar_next) * e_theta
                else:
                    raise ValueError(f"Unknown sampling method: {sampling}")

                if guidance_enabled and t <= guidance_start_t:
                    x_next = self._apply_dynamics_guidance(x_next, guidance)

                traj[t - stride] = x_next.detach()  # Stop gradient and save trajectory.
                traj[t] = traj[t].cpu()             # Move previous output to CPU memory.
                if not ret_traj:
                    del traj[t]

            if ret_traj:
                traj_list.append(traj)
            else:
                traj_list.append(traj[0])

        return torch.stack(traj_list)


class TrajNet(Module):

    def __init__(self, point_dim, context_dim, residual):
        super().__init__()
        self.act = F.leaky_relu
        self.residual = residual
        self.layers = ModuleList([
            ConcatSquashLinear(point_dim, 128, context_dim + 3),
            ConcatSquashLinear(128, 256, context_dim + 3),
            ConcatSquashLinear(256, 512, context_dim + 3),
            ConcatSquashLinear(512, 256, context_dim + 3),
            ConcatSquashLinear(256, 128, context_dim + 3),
            ConcatSquashLinear(128, point_dim, context_dim + 3),
        ])

    def forward(self, x, timestep, context):
        """
        Args:
            x: noisy trajectory y_k, (B, N, d)
            timestep: diffusion step index k, (B,)
            context: encoded history feature f, (B, F)
        """
        batch_size = x.size(0)
        timestep = timestep.float().view(batch_size, 1, 1)   # (B, 1, 1)
        context = context.view(batch_size, 1, -1)            # (B, 1, F)

        time_emb = torch.cat([timestep, torch.sin(timestep), torch.cos(timestep)], dim=-1)  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)     # (B, 1, F+3)

        out = x
        for i, layer in enumerate(self.layers):
            out = layer(ctx=ctx_emb, x=out)
            if i < len(self.layers) - 1:
                out = self.act(out)

        if self.residual:
            return x + out
        else:
            return out


class TransformerConcatLinear(Module):

    def __init__(self, point_dim, context_dim, tf_layer, residual):
        super().__init__()
        self.residual = residual
        self.pos_emb = PositionalEncoding(d_model=2 * context_dim, dropout=0.1, max_len=24)
        self.concat1 = ConcatSquashLinear(point_dim, 2 * context_dim, context_dim + 3)
        self.layer = nn.TransformerEncoderLayer(
            d_model=2 * context_dim, nhead=4, dim_feedforward=4 * context_dim
        )
        self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=tf_layer)
        self.concat3 = ConcatSquashLinear(2 * context_dim, context_dim, context_dim + 3)
        self.concat4 = ConcatSquashLinear(context_dim, context_dim // 2, context_dim + 3)
        self.linear = ConcatSquashLinear(context_dim // 2, point_dim, context_dim + 3)

    def forward(self, x, timestep, context):
        batch_size = x.size(0)
        timestep = timestep.float().view(batch_size, 1, 1)   # (B, 1, 1)
        context = context.view(batch_size, 1, -1)            # (B, 1, F)

        time_emb = torch.cat([timestep, torch.sin(timestep), torch.cos(timestep)], dim=-1)  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)     # (B, 1, F+3)

        x = self.concat1(ctx_emb, x)
        final_emb = x.permute(1, 0, 2)
        final_emb = self.pos_emb(final_emb)

        trans = self.transformer_encoder(final_emb).permute(1, 0, 2)
        trans = self.concat3(ctx_emb, trans)
        trans = self.concat4(ctx_emb, trans)
        return self.linear(ctx_emb, trans)


class TransformerLinear(Module):

    def __init__(self, point_dim, context_dim, residual):
        super().__init__()
        self.residual = residual

        self.pos_emb = PositionalEncoding(d_model=128, dropout=0.1, max_len=24)
        self.y_up = nn.Linear(point_dim, 128)
        self.ctx_up = nn.Linear(context_dim + 3, 128)
        self.layer = nn.TransformerEncoderLayer(d_model=128, nhead=2, dim_feedforward=512)
        self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=3)
        self.linear = nn.Linear(128, point_dim)

    def forward(self, x, timestep, context):
        batch_size = x.size(0)
        timestep = timestep.float().view(batch_size, 1, 1)   # (B, 1, 1)
        context = context.view(batch_size, 1, -1)            # (B, 1, F)

        time_emb = torch.cat([timestep, torch.sin(timestep), torch.cos(timestep)], dim=-1)  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)     # (B, 1, F+3)

        ctx_emb = self.ctx_up(ctx_emb)
        emb = self.y_up(x)
        final_emb = torch.cat([ctx_emb, emb], dim=1).permute(1, 0, 2)
        final_emb = self.pos_emb(final_emb)

        trans = self.transformer_encoder(final_emb)   # (1+N) * B * 128
        trans = trans[1:].permute(1, 0, 2)            # B * N * 128
        return self.linear(trans)


class LinearDecoder(Module):
    def __init__(self):
        super().__init__()
        self.act = F.leaky_relu
        self.layers = ModuleList([
            nn.Linear(32, 64),
            nn.Linear(64, 128),
            nn.Linear(128, 256),
            nn.Linear(256, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 12)
        ])

    def forward(self, code):
        out = code
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i < len(self.layers) - 1:
                out = self.act(out)
        return out
