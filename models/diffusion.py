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

    def get_loss(self, x_0, context, t=None):
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

        e_theta = self.net(c0 * x_0 + c1 * e_rand, timestep=timestep, context=context)
        loss = F.mse_loss(e_theta.view(-1, point_dim), e_rand.view(-1, point_dim), reduction='mean')
        return loss

    def sample(self, num_points, context, sample, bestof, point_dim=2,
               flexibility=0.0, ret_traj=False, sampling="ddpm", step=100):
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