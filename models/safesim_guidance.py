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


def safesim_ttc_loss(danger, loss_timesteps=None, loss_scale=1.0, reduction="mean"):
    """SafeSim TTC loss sign: minimizing this increases the TTC danger score."""
    danger = _select_timesteps(danger, loss_timesteps)
    loss = -danger
    return _reduce_loss(loss, reduction=reduction) * float(loss_scale)


def calculate_exact_speed_penalty(
    distance,
    ego_speed,
    ctrl_speed,
    exact_diff,
    distance_threshold=5.0,
    margin=0.0,
):
    """SafeSim helper: penalize ctrl speed away from ego_speed + exact_diff when close."""
    close_enough = distance < float(distance_threshold)
    target_speed = ego_speed + float(exact_diff)
    speed_difference = torch.abs(ctrl_speed - target_speed)
    outside_margin = speed_difference > float(margin)
    return (speed_difference - float(margin)) * close_enough.float() * outside_margin.float()


def causecollision_loss(
    distance,
    ego_speed,
    ctrl_speed,
    adv_term_weight=None,
    adv_bound=30.0,
    speed_diff=2.0,
    interact_dist_thresh=100.0,
    loss_timesteps=None,
    loss_scale=1.0,
    reduction="mean",
):
    """
    Local ego-target version of SafeSim causecollision.

    Official SafeSim computes this over ego/control indices inside a multi-agent
    batch. Here distance is already the selected ego-target distance [B,T].
    """
    if adv_term_weight is None:
        adv_term_weight = {}
    distance_weight = float(adv_term_weight.get("distance", 1.0))
    filtered_weight = float(adv_term_weight.get("filtered_distance", 0.1))
    speed_weight = float(adv_term_weight.get("speed_penalty", 0.0))

    distance = _select_timesteps(distance, loss_timesteps)
    ego_speed = _select_timesteps(ego_speed, loss_timesteps)
    ctrl_speed = _select_timesteps(ctrl_speed, loss_timesteps)
    if distance.numel() == 0:
        return distance.new_tensor(0.0)

    min_distances = torch.min(distance, dim=-1, keepdim=True).values
    interaction_mask = min_distances < float(interact_dist_thresh)
    interaction_mask = interaction_mask.expand_as(distance)

    speed_penalty = calculate_exact_speed_penalty(distance, ego_speed, ctrl_speed, speed_diff)
    distance_term = distance_weight * min_distances.expand_as(distance)
    speed_term = speed_weight * speed_penalty
    filtered_term = filtered_weight * distance

    loss = torch.zeros_like(distance)
    loss[interaction_mask] = (
        distance_term[interaction_mask]
        + speed_term[interaction_mask]
        + filtered_term[interaction_mask]
    )
    loss = torch.clamp(loss, 0.0, float(adv_bound))
    return _reduce_loss(loss, reduction=reduction) * float(loss_scale)


def collision_guidance_loss(
    danger,
    distance,
    ego_speed,
    ctrl_speed,
    ttc_loss_timesteps=None,
    ttc_loss_scale=1.0,
    causecollision_loss_timesteps=None,
    causecollision_loss_scale=1.0,
    causecollision_adv_term_weight=None,
    causecollision_adv_bound=30.0,
    causecollision_speed_diff=2.0,
    causecollision_interact_dist_thresh=100.0,
    reduction="mean",
):
    ttc = safesim_ttc_loss(
        danger,
        loss_timesteps=ttc_loss_timesteps,
        loss_scale=ttc_loss_scale,
        reduction=reduction,
    )
    cause = causecollision_loss(
        distance,
        ego_speed,
        ctrl_speed,
        adv_term_weight=causecollision_adv_term_weight,
        adv_bound=causecollision_adv_bound,
        speed_diff=causecollision_speed_diff,
        interact_dist_thresh=causecollision_interact_dist_thresh,
        loss_timesteps=causecollision_loss_timesteps,
        loss_scale=causecollision_loss_scale,
        reduction=reduction,
    )
    return cause + ttc


def collision_sample_score(
    danger,
    distance,
    ego_speed,
    ctrl_speed,
    ttc_filter_timesteps=None,
    ttc_loss_scale=1.0,
    causecollision_filter_timesteps=None,
    causecollision_loss_scale=1.0,
    causecollision_adv_term_weight=None,
    causecollision_adv_bound=30.0,
    causecollision_speed_diff=2.0,
    causecollision_interact_dist_thresh=100.0,
):
    ttc = safesim_ttc_loss(
        danger,
        loss_timesteps=ttc_filter_timesteps,
        loss_scale=ttc_loss_scale,
        reduction="none",
    )
    cause = causecollision_loss(
        distance,
        ego_speed,
        ctrl_speed,
        adv_term_weight=causecollision_adv_term_weight,
        adv_bound=causecollision_adv_bound,
        speed_diff=causecollision_speed_diff,
        interact_dist_thresh=causecollision_interact_dist_thresh,
        loss_timesteps=causecollision_filter_timesteps,
        loss_scale=causecollision_loss_scale,
        reduction="none",
    )
    return _per_sample_mean(cause) + _per_sample_mean(ttc)
