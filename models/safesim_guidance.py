import torch


def compute_pair_kinematics(pred_pos, ref_pos, pred_vel, ref_vel, eps=1.0e-6):
    rel_pos = pred_pos - ref_pos
    rel_vel = pred_vel - ref_vel
    distance = torch.norm(rel_pos, dim=-1)
    rel_speed = torch.norm(rel_vel, dim=-1)
    closing_speed = -torch.sum(rel_pos * rel_vel, dim=-1) / distance.clamp_min(eps)
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


def _softmin(values, temperature, eps=1.0e-6):
    temperature = max(float(temperature), eps)
    weights = torch.softmax(-values / temperature, dim=-1)
    return torch.sum(values * weights, dim=-1)


def _softmax(values, temperature, eps=1.0e-6):
    temperature = max(float(temperature), eps)
    weights = torch.softmax(values / temperature, dim=-1)
    return torch.sum(values * weights, dim=-1)


def collision_objective(
    distance,
    rel_speed,
    danger,
    collision_distance=0.75,
    rel_speed_target=0.75,
    weight_distance=1.0,
    weight_ttc=1.0,
    weight_rel_speed=0.5,
    distance_temp=0.25,
    danger_temp=0.15,
    rel_speed_temp=0.25,
    eps=1.0e-6,
):
    softmin_distance = _softmin(distance, distance_temp, eps=eps)
    softmax_danger = _softmax(danger, danger_temp, eps=eps)

    distance_penalty = torch.relu(softmin_distance - float(collision_distance)).pow(2).mean()
    danger_reward = -softmax_danger.mean()

    speed_penalty = distance.new_tensor(0.0)
    if float(rel_speed_target) > 0.0 and float(weight_rel_speed) != 0.0:
        softmax_rel_speed = _softmax(rel_speed, rel_speed_temp, eps=eps)
        speed_penalty = torch.relu(float(rel_speed_target) - softmax_rel_speed).pow(2).mean()

    return (
        float(weight_distance) * distance_penalty
        + float(weight_ttc) * danger_reward
        + float(weight_rel_speed) * speed_penalty
    )


def not_collision_objective(
    distance,
    rel_speed,
    closing_speed,
    danger,
    safe_distance=4.0,
    safe_max_closing_speed=0.25,
    rel_speed_limit=0.0,
    danger_margin=0.15,
    weight_distance=1.0,
    weight_ttc=1.0,
    weight_rel_speed=0.5,
    near_distance_temp=0.5,
    closing_gate_temp=0.25,
):
    near_temp = max(float(near_distance_temp), 1.0e-6)
    closing_temp = max(float(closing_gate_temp), 1.0e-6)
    near_gate = torch.sigmoid((float(safe_distance) - distance) / near_temp)
    closing_gate = torch.sigmoid(closing_speed / closing_temp)

    distance_penalty = torch.relu(float(safe_distance) - distance).pow(2).mean()
    danger_penalty = torch.relu(danger - float(danger_margin)).pow(2).mean()
    closing_penalty = (
        torch.relu(closing_speed - float(safe_max_closing_speed)).pow(2)
        * near_gate
        * closing_gate
    ).mean()

    rel_speed_penalty = distance.new_tensor(0.0)
    if float(rel_speed_limit) > 0.0:
        rel_speed_penalty = (torch.relu(rel_speed - float(rel_speed_limit)).pow(2) * near_gate).mean()

    return (
        float(weight_distance) * distance_penalty
        + float(weight_ttc) * danger_penalty
        + float(weight_rel_speed) * (closing_penalty + rel_speed_penalty)
    )
