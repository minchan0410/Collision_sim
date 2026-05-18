import torch


def compute_pair_kinematics(pred_pos, ref_pos, pred_vel, ref_vel, eps=1.0e-6):
    rel_pos = pred_pos - ref_pos
    rel_vel = pred_vel - ref_vel
    distance = torch.norm(rel_pos, dim=-1).clamp_min(eps)
    rel_speed = torch.norm(rel_vel, dim=-1)
    closing_speed = -torch.sum(rel_pos * rel_vel, dim=-1) / distance
    return {
        "rel_pos": rel_pos,
        "rel_vel": rel_vel,
        "distance": distance,
        "rel_speed": rel_speed,
        "closing_speed": closing_speed,
    }


def compute_safesim_ttc_score(
    rel_pos,
    rel_vel,
    time_bandwidth=1.0,
    distance_bandwidth=1.0,
    min_velocity_diff=0.1,
    eps=1.0e-6,
):
    time_bandwidth = max(float(time_bandwidth), eps)
    distance_bandwidth = max(float(distance_bandwidth), eps)
    min_velocity_diff = max(float(min_velocity_diff), eps)

    raw_speed_sq = torch.sum(rel_vel * rel_vel, dim=-1)
    min_speed_sq = min_velocity_diff * min_velocity_diff
    speed_sq = raw_speed_sq.clamp_min(min_speed_sq)
    raw_t_col = -torch.sum(rel_pos * rel_vel, dim=-1) / speed_sq
    t_col = torch.relu(raw_t_col)

    cross = rel_vel[..., 0] * rel_pos[..., 1] - rel_vel[..., 1] * rel_pos[..., 0]
    closest_distance = torch.abs(cross) / torch.sqrt(speed_sq).clamp_min(eps)
    current_distance = torch.norm(rel_pos, dim=-1)
    use_current_distance = (raw_t_col < 0.0) | (raw_speed_sq < min_speed_sq)
    d_col = torch.where(use_current_distance, current_distance, closest_distance)
    d_col = torch.relu(d_col)

    danger = torch.exp(
        -t_col.pow(2) / (2.0 * time_bandwidth)
        -d_col.pow(2) / (2.0 * distance_bandwidth)
    )
    return danger, t_col, d_col


def _select_timesteps(value, timesteps):
    if timesteps is None:
        return value
    horizon = value.size(-1)
    if horizon <= 0:
        return value

    if isinstance(timesteps, int):
        if timesteps <= 0:
            return value[..., :0]
        return value[..., : min(timesteps, horizon)]

    if isinstance(timesteps, float):
        return _select_timesteps(value, int(timesteps))

    if isinstance(timesteps, (list, tuple)):
        if len(timesteps) == 0:
            return value
        index = torch.as_tensor(timesteps, device=value.device, dtype=torch.long)
        index = torch.clamp(index, 0, horizon - 1)
        return torch.index_select(value, dim=-1, index=index)

    return value


def _reduce_loss(value, reduction="mean"):
    if value.numel() == 0:
        return value.new_tensor(0.0)
    if reduction == "none":
        return value
    if reduction == "sum":
        return value.sum()
    return value.mean()


def _per_sample_mean(value):
    if value.numel() == 0:
        return value.new_zeros(value.shape[:-1])
    return value.mean(dim=-1)


def safesim_ttc_loss(danger, loss_timesteps=None, reduction="mean"):
    """SafeSim TTC loss sign: minimizing this increases the TTC danger score."""
    danger = _select_timesteps(danger, loss_timesteps)
    loss = -danger
    return _reduce_loss(loss, reduction=reduction)


def causecollision_loss(distance, loss_timesteps=None, reduction="mean"):
    """
    Gradient-descent form of SafeSim J_coll = -sum_t d(t).

    The diffusion guidance step minimizes objectives, so the paper's collision
    score is sign-flipped into mean_t d(t). This intentionally uses only the
    timestep-wise distance term.
    """
    distance = _select_timesteps(distance, loss_timesteps)
    if distance.numel() == 0:
        return distance.new_tensor(0.0)
    return _reduce_loss(distance, reduction=reduction)


def relative_speed_loss(
    distance,
    ego_speed,
    adv_speed,
    speed_diff=0.0,
    distance_threshold=5.0,
    loss_timesteps=None,
    reduction="mean",
):
    """SafeSim supplementary J_v = |v_ego - v_adv - v_diff| 1{d < d_col}."""
    distance = _select_timesteps(distance, loss_timesteps)
    ego_speed = _select_timesteps(ego_speed, loss_timesteps)
    adv_speed = _select_timesteps(adv_speed, loss_timesteps)
    if distance.numel() == 0:
        return distance.new_tensor(0.0)

    mask = distance < float(distance_threshold)
    loss = torch.abs(ego_speed - adv_speed - float(speed_diff)) * mask.float()
    return _reduce_loss(loss, reduction=reduction)


def collision_guidance_loss(
    danger,
    distance,
    ego_speed,
    adv_speed,
    ttc_loss_timesteps=None,
    causecollision_loss_timesteps=None,
    relative_speed_loss_timesteps=None,
    relative_speed_diff=0.0,
    relative_speed_distance_threshold=5.0,
    reduction="mean",
):
    ttc = safesim_ttc_loss(
        danger,
        loss_timesteps=ttc_loss_timesteps,
        reduction=reduction,
    )
    cause = causecollision_loss(
        distance,
        loss_timesteps=causecollision_loss_timesteps,
        reduction=reduction,
    )
    rel_speed = relative_speed_loss(
        distance,
        ego_speed,
        adv_speed,
        speed_diff=relative_speed_diff,
        distance_threshold=relative_speed_distance_threshold,
        loss_timesteps=relative_speed_loss_timesteps,
        reduction=reduction,
    )
    return cause + rel_speed + ttc


def collision_sample_score(
    danger,
    distance,
    ego_speed,
    adv_speed,
    ttc_filter_timesteps=None,
    causecollision_filter_timesteps=None,
    relative_speed_filter_timesteps=None,
    relative_speed_diff=0.0,
    relative_speed_distance_threshold=5.0,
):
    ttc = safesim_ttc_loss(
        danger,
        loss_timesteps=ttc_filter_timesteps,
        reduction="none",
    )
    cause = causecollision_loss(
        distance,
        loss_timesteps=causecollision_filter_timesteps,
        reduction="none",
    )
    rel_speed = relative_speed_loss(
        distance,
        ego_speed,
        adv_speed,
        speed_diff=relative_speed_diff,
        distance_threshold=relative_speed_distance_threshold,
        loss_timesteps=relative_speed_filter_timesteps,
        reduction="none",
    )
    return _per_sample_mean(cause) + _per_sample_mean(rel_speed) + _per_sample_mean(ttc)
