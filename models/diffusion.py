import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList
import numpy as np
import torch.nn as nn
from .common import *
from . import safesim_guidance
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

    def _compute_heading_objective(self, vel_phys, guidance):
        if not bool(guidance.get("heading_enabled", False)):
            return vel_phys.new_tensor(0.0)

        target_yaw = guidance.get("heading_target_yaw", None)
        if target_yaw is None:
            return vel_phys.new_tensor(0.0)

        weight = float(guidance.get("heading_weight", 0.0))
        if weight <= 0.0:
            return vel_phys.new_tensor(0.0)

        eps = float(guidance.get("eps", 1e-6))
        min_speed = max(float(guidance.get("heading_min_speed", 0.03)), eps)
        focus_ratio = float(guidance.get("heading_focus_ratio", 1.0))
        focus_ratio = min(max(focus_ratio, 0.0), 1.0)
        horizon = vel_phys.size(1)
        focus_steps = max(1, int(round(horizon * focus_ratio))) if horizon > 0 else 0
        if focus_steps <= 0:
            return vel_phys.new_tensor(0.0)

        yaw = target_yaw
        if not torch.is_tensor(yaw):
            yaw = torch.tensor(yaw, device=vel_phys.device, dtype=vel_phys.dtype)
        else:
            yaw = yaw.to(device=vel_phys.device, dtype=vel_phys.dtype)
        if yaw.dim() == 0:
            yaw = yaw.view(1)
        else:
            yaw = yaw.reshape(yaw.size(0), -1)[:, -1]

        batch_size = vel_phys.size(0)
        if yaw.size(0) == 1 and batch_size > 1:
            yaw = yaw.expand(batch_size)
        elif yaw.size(0) != batch_size:
            yaw = yaw[:batch_size]

        forward = torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=-1).view(-1, 1, 2)
        vel_focus = vel_phys[:, :focus_steps, :]
        speed = torch.norm(vel_focus, dim=-1).clamp_min(eps)
        cos_align = (torch.sum(vel_focus * forward, dim=-1) / speed).clamp(-1.0, 1.0)
        direction_loss = 1.0 - cos_align

        moving_mask = (speed > min_speed).float()
        return weight * self._masked_mean(direction_loss, mask=moving_mask)

    @staticmethod
    def _broadcast_batch_matrix(value, batch_size, device, dtype):
        if value is None:
            return None
        if not torch.is_tensor(value):
            value = torch.tensor(value, device=device, dtype=dtype)
        else:
            value = value.to(device=device, dtype=dtype)
        if value.dim() == 1:
            value = value.unsqueeze(0)
        if value.size(0) == 1 and batch_size > 1:
            value = value.expand(batch_size, -1)
        elif value.size(0) != batch_size:
            value = value[:batch_size]
        return value

    def _compute_collision_objective(self, vel_phys, guidance):
        collision_enabled = bool(guidance.get("collision_enabled", False))
        not_collision_enabled = bool(guidance.get("not_collision_enabled", False))

        # Avoid conflicting attraction/repulsion terms in the same run.
        if collision_enabled and not_collision_enabled:
            collision_enabled = False

        if (not collision_enabled) and (not not_collision_enabled):
            return vel_phys.new_tensor(0.0)

        interaction_guidance_scale = max(float(guidance.get("interaction_guidance_scale", 1.0)), 0.0)
        if interaction_guidance_scale <= 0.0:
            return vel_phys.new_tensor(0.0)

        ref_pos = guidance.get("collision_reference_positions", None)
        if ref_pos is None:
            return vel_phys.new_tensor(0.0)

        eps = float(guidance.get("eps", 1e-6))
        dt = max(float(guidance.get("dt", 1.0)), eps)
        batch_size = vel_phys.size(0)
        horizon = vel_phys.size(1)

        if not torch.is_tensor(ref_pos):
            ref_pos = torch.tensor(ref_pos, device=vel_phys.device, dtype=vel_phys.dtype)
        else:
            ref_pos = ref_pos.to(device=vel_phys.device, dtype=vel_phys.dtype)

        if ref_pos.dim() == 2 and ref_pos.size(-1) == 2:
            ref_pos = ref_pos.unsqueeze(0)
        if ref_pos.dim() != 3 or ref_pos.size(-1) != 2 or horizon <= 0:
            return vel_phys.new_tensor(0.0)

        if ref_pos.size(0) == 1 and batch_size > 1:
            ref_pos = ref_pos.expand(batch_size, -1, -1)
        elif ref_pos.size(0) != batch_size:
            ref_pos = ref_pos[:batch_size]
        if ref_pos.size(0) == 0:
            return vel_phys.new_tensor(0.0)

        initial_position = guidance.get("initial_position", None)
        initial_position = self._broadcast_batch_matrix(initial_position, batch_size, vel_phys.device, vel_phys.dtype)
        if initial_position is None:
            initial_position = vel_phys.new_zeros((batch_size, 2))

        pred_pos = initial_position.unsqueeze(1) + torch.cumsum(vel_phys * dt, dim=1)
        min_len = min(pred_pos.size(1), ref_pos.size(1))
        if min_len <= 0:
            return vel_phys.new_tensor(0.0)
        pred_pos = pred_pos[:, :min_len]
        ref_pos = ref_pos[:, :min_len]

        objective = vel_phys.new_tensor(0.0)
        ref_vel = vel_phys.new_zeros(ref_pos.shape)
        if ref_pos.size(1) > 1:
            ref_vel[:, :-1] = (ref_pos[:, 1:] - ref_pos[:, :-1]) / dt
            ref_vel[:, -1] = ref_vel[:, -2]

        pair = safesim_guidance.compute_pair_kinematics(
            pred_pos=pred_pos,
            ref_pos=ref_pos,
            pred_vel=vel_phys[:, :min_len],
            ref_vel=ref_vel,
            eps=eps,
        )
        if pair["rel_pos"].numel() == 0:
            return vel_phys.new_tensor(0.0)

        danger, _, _ = safesim_guidance.compute_safesim_ttc_score(
            pair["rel_pos"],
            pair["rel_vel"],
            time_bandwidth=float(guidance.get("ttc_time_bandwidth", guidance.get("safesim_ttc_time_bandwidth", 1.0))),
            distance_bandwidth=float(guidance.get("ttc_distance_bandwidth", guidance.get("safesim_ttc_distance_bandwidth", 1.0))),
            min_velocity_diff=float(guidance.get("ttc_min_velocity_diff", guidance.get("safesim_ttc_min_velocity_diff", 0.1))),
            eps=eps,
        )

        if collision_enabled:
            pred_speed = torch.norm(vel_phys[:, :min_len], dim=-1)
            ref_speed = torch.norm(ref_vel, dim=-1)
            objective = objective + safesim_guidance.collision_guidance_loss(
                danger=danger,
                distance=pair["distance"],
                ego_speed=ref_speed,
                ctrl_speed=pred_speed,
                ttc_loss_timesteps=guidance.get("ttc_loss_timesteps", None),
                ttc_loss_scale=float(guidance.get("ttc_loss_scale", 1.0)),
                causecollision_loss_timesteps=guidance.get("causecollision_loss_timesteps", None),
                causecollision_loss_scale=float(guidance.get("causecollision_loss_scale", 1.0)),
                causecollision_adv_term_weight=guidance.get("causecollision_adv_term_weight", None),
                causecollision_adv_bound=float(guidance.get("causecollision_adv_bound", 30.0)),
                causecollision_speed_diff=float(guidance.get("causecollision_speed_diff", 2.0)),
                causecollision_interact_dist_thresh=float(guidance.get("causecollision_interact_dist_thresh", 100.0)),
                reduction="mean",
            )

        if not_collision_enabled:
            objective = objective + safesim_guidance.local_noncollision_loss(
                pair["distance"],
                safe_distance=float(guidance.get("local_noncollision_safe_distance", 4.0)),
                loss_timesteps=guidance.get("local_noncollision_loss_timesteps", None),
                loss_scale=float(guidance.get("local_noncollision_loss_scale", 1.0)),
                reduction="mean",
            )

        return objective * interaction_guidance_scale

    def _compute_dynamics_objective(self, vel_phys, guidance):
        eps = float(guidance.get("eps", 1e-6))
        dt = max(float(guidance.get("dt", 1.0)), eps)
        min_speed = max(float(guidance.get("min_speed", 0.2)), eps)

        max_accel = float(guidance.get("max_accel", 6.0))
        max_steer_deg = float(guidance.get("max_steer_deg", 32.0))
        max_steer_rate_deg_s = float(guidance.get("max_steer_rate_deg_s", 120.0))

        weight_accel = float(guidance.get("weight_accel", 1.0))
        weight_steer = float(guidance.get("weight_steer", 1.0))
        weight_steer_rate = float(guidance.get("weight_steer_rate", 0.25))
        low_speed_yaw_weight = float(guidance.get("low_speed_yaw_weight", 0.0))
        low_speed_yaw_threshold = max(float(guidance.get("low_speed_yaw_threshold", min_speed)), eps)

        wheelbase = max(float(guidance.get("wheelbase", 0.0)), eps)
        max_steer_rad = abs(max_steer_deg) * (math.pi / 180.0)
        max_steer_rate_rad_s = abs(max_steer_rate_deg_s) * (math.pi / 180.0)

        speed = torch.norm(vel_phys, dim=-1).clamp_min(eps)  # [B, T]
        speed_pair = 0.5 * (speed[:, 1:] + speed[:, :-1]) if speed.size(1) > 1 else speed.new_zeros(speed.size(0), 0)
        moving_mask = (speed_pair > min_speed).float()

        objective = vel_phys.new_tensor(0.0)

        # Longitudinal/total acceleration bound.
        acc = (vel_phys[:, 1:] - vel_phys[:, :-1]) / dt if vel_phys.size(1) > 1 else vel_phys.new_zeros(vel_phys.size(0), 0, vel_phys.size(2))
        acc_mag = torch.norm(acc, dim=-1) if acc.numel() > 0 else speed.new_zeros(speed.size(0), 0)
        if weight_accel > 0:
            objective = objective + weight_accel * self._soft_limit_penalty(acc_mag, max_accel, mask=moving_mask)

        # Bicycle geometry: kappa = yaw_rate / speed, delta = atan(L * kappa).
        if vel_phys.size(1) > 1:
            heading = torch.atan2(vel_phys[..., 1], vel_phys[..., 0])
            yaw_delta = self._wrap_to_pi(heading[:, 1:] - heading[:, :-1])
            yaw_rate = torch.abs(yaw_delta) / dt
            signed_yaw_rate = yaw_delta / dt
            curvature = signed_yaw_rate / speed_pair.clamp_min(min_speed)
            steer_angle = torch.atan(curvature * wheelbase)

            if weight_steer > 0:
                steer_violation = F.relu(steer_angle.abs() - max_steer_rad).pow(2)
                objective = objective + weight_steer * self._masked_mean(steer_violation, mask=moving_mask)

            if steer_angle.size(1) > 1 and weight_steer_rate > 0 and max_steer_rate_rad_s > 0:
                steer_rate = torch.abs(steer_angle[:, 1:] - steer_angle[:, :-1]) / dt
                steer_rate_mask = (moving_mask[:, 1:] * moving_mask[:, :-1]) if moving_mask.size(1) > 1 else None
                objective = objective + weight_steer_rate * self._soft_limit_penalty(
                    steer_rate,
                    max_steer_rate_rad_s,
                    mask=steer_rate_mask,
                )

            if low_speed_yaw_weight > 0:
                # Near-stop anti-spin penalty. Steering angle is ill-defined when speed is near zero.
                low_speed_mask = (speed_pair < low_speed_yaw_threshold).float()
                objective = objective + low_speed_yaw_weight * self._masked_mean(yaw_rate.pow(2), mask=low_speed_mask)

        objective = objective + self._compute_collision_objective(vel_phys, guidance)
        objective = objective + self._compute_heading_objective(vel_phys, guidance)

        return objective

    @staticmethod
    def _sample_guidance_variant(guidance, device):
        if not isinstance(guidance, dict) or len(guidance) == 0:
            return {}
        g = dict(guidance)

        collision_enabled = bool(g.get("collision_enabled", False))
        not_collision_enabled = bool(g.get("not_collision_enabled", False))
        if collision_enabled and not_collision_enabled:
            collision_enabled = False
            g["collision_enabled"] = False

        if (not collision_enabled) and (not not_collision_enabled):
            return g

        return g

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
                if not objective.requires_grad:
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

        traj_list = []
        for _ in range(sample):
            batch_size = context.size(0)
            device = context.device
            sample_guidance = self._sample_guidance_variant(guidance, device=device)
            guidance_enabled = bool(sample_guidance.get("enabled", False))
            start_ratio = float(sample_guidance.get("start_ratio", 0.0)) if guidance_enabled else 0.0
            start_ratio = min(max(start_ratio, 0.0), 1.0)
            guidance_start_t = max(1, int(round(self.var_sched.num_steps * (1.0 - start_ratio))))
            xT_temperature = max(float(sample_guidance.get("xT_temperature", 1.0)), 1e-6)

            x_T = torch.randn([batch_size, num_points, point_dim], device=device) * xT_temperature

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
                    x_next = self._apply_dynamics_guidance(x_next, sample_guidance)

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
