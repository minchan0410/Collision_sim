import torch

from models.safesim_guidance import (
    calculate_exact_speed_penalty,
    causecollision_loss,
    collision_sample_score,
    compute_pair_kinematics,
    compute_safesim_ttc_score,
    local_noncollision_loss,
    local_noncollision_sample_score,
    safesim_ttc_loss,
)


def _score(pred_pos, ref_pos, pred_vel, ref_vel):
    pair = compute_pair_kinematics(pred_pos, ref_pos, pred_vel, ref_vel)
    danger, t_col, d_col = compute_safesim_ttc_score(
        pair["rel_pos"],
        pair["rel_vel"],
        time_bandwidth=1.0,
        distance_bandwidth=1.0,
        min_velocity_diff=0.1,
    )
    return pair, danger, t_col, d_col


def test_head_on_approach_has_high_danger():
    ref_pos = torch.zeros(1, 1, 2)
    pred_pos = torch.tensor([[[2.0, 0.0]]])
    ref_vel = torch.zeros(1, 1, 2)
    pred_vel = torch.tensor([[[-2.0, 0.0]]])

    _, danger, t_col, d_col = _score(pred_pos, ref_pos, pred_vel, ref_vel)

    assert torch.isfinite(danger).all()
    assert danger.item() > 0.6
    assert torch.allclose(t_col, torch.tensor([[1.0]]), atol=1.0e-5)
    assert torch.allclose(d_col, torch.zeros_like(d_col), atol=1.0e-5)


def test_receding_vehicle_is_finite_and_lower_danger():
    ref_pos = torch.zeros(1, 1, 2)
    pred_pos = torch.tensor([[[2.0, 0.0]]])
    ref_vel = torch.zeros(1, 1, 2)
    pred_vel = torch.tensor([[[2.0, 0.0]]])

    _, danger, t_col, d_col = _score(pred_pos, ref_pos, pred_vel, ref_vel)

    assert torch.isfinite(danger).all()
    assert torch.allclose(t_col, torch.zeros_like(t_col))
    assert torch.allclose(d_col, torch.tensor([[2.0]]), atol=1.0e-5)
    assert danger.item() < 0.2


def test_nearly_equal_velocity_uses_current_distance():
    ref_pos = torch.zeros(1, 1, 2)
    pred_pos = torch.tensor([[[2.0, 0.0]]])
    ref_vel = torch.tensor([[[1.0, 0.0]]])
    pred_vel = torch.tensor([[[1.0, 0.0]]])

    _, danger, t_col, d_col = _score(pred_pos, ref_pos, pred_vel, ref_vel)

    assert torch.isfinite(danger).all()
    assert torch.allclose(t_col, torch.zeros_like(t_col))
    assert torch.allclose(d_col, torch.tensor([[2.0]]), atol=1.0e-5)
    assert danger.item() < 0.2


def test_crossing_approach_responds_to_closest_approach():
    ref_pos = torch.zeros(1, 1, 2)
    pred_pos = torch.tensor([[[1.0, -1.0]]])
    ref_vel = torch.zeros(1, 1, 2)
    pred_vel = torch.tensor([[[0.0, 1.0]]])

    _, danger, t_col, d_col = _score(pred_pos, ref_pos, pred_vel, ref_vel)

    assert danger.item() > 0.3
    assert torch.allclose(t_col, torch.tensor([[1.0]]), atol=1.0e-5)
    assert torch.allclose(d_col, torch.tensor([[1.0]]), atol=1.0e-5)


def test_ttc_collision_loss_prefers_dangerous_close_pair():
    close_danger = torch.tensor([[0.8, 0.9]])

    far_danger = torch.tensor([[0.0, 0.0]])

    close_obj = safesim_ttc_loss(close_danger)
    far_obj = safesim_ttc_loss(far_danger)

    assert close_obj < far_obj


def test_causecollision_prefers_closer_ego_target_trajectories():
    close_distance = torch.tensor([[1.0, 1.5, 1.2]])
    far_distance = torch.tensor([[6.0, 7.0, 8.0]])
    ego_speed = torch.zeros_like(close_distance)
    ctrl_speed = torch.zeros_like(close_distance)
    weights = {"distance": 1.0, "speed_penalty": 0.0, "filtered_distance": 0.1}

    close_obj = causecollision_loss(close_distance, ego_speed, ctrl_speed, adv_term_weight=weights)
    far_obj = causecollision_loss(far_distance, ego_speed, ctrl_speed, adv_term_weight=weights)

    assert close_obj < far_obj


def test_speed_penalty_targets_ego_speed_plus_speed_diff_when_close():
    distance = torch.tensor([[2.0, 2.0, 6.0]])
    ego_speed = torch.tensor([[10.0, 10.0, 10.0]])
    matching_ctrl_speed = torch.tensor([[12.0, 12.0, 12.0]])
    slow_ctrl_speed = torch.tensor([[10.0, 10.0, 10.0]])

    matching = calculate_exact_speed_penalty(distance, ego_speed, matching_ctrl_speed, exact_diff=2.0)
    slow = calculate_exact_speed_penalty(distance, ego_speed, slow_ctrl_speed, exact_diff=2.0)

    assert torch.allclose(matching, torch.zeros_like(matching))
    assert torch.allclose(slow[:, :2], torch.full((1, 2), 2.0))
    assert torch.allclose(slow[:, 2:], torch.zeros(1, 1))


def test_local_noncollision_penalizes_only_inside_safe_distance():
    distance = torch.tensor([[2.0, 4.0, 5.0]])
    loss = local_noncollision_loss(distance, safe_distance=4.0, reduction="none")

    assert torch.allclose(loss, torch.tensor([[4.0, 0.0, 0.0]]))


def test_sample_scores_order_collision_and_noncollision_oppositely():
    danger = torch.tensor([[[0.9, 0.9]], [[0.0, 0.0]]])
    distance = torch.tensor([[[1.0, 1.0]], [[6.0, 6.0]]])
    ego_speed = torch.zeros_like(distance)
    ctrl_speed = torch.zeros_like(distance)

    collision_score = collision_sample_score(
        danger,
        distance,
        ego_speed,
        ctrl_speed,
        causecollision_adv_term_weight={"distance": 1.0, "speed_penalty": 0.0, "filtered_distance": 0.1},
    )
    noncollision_score = local_noncollision_sample_score(distance, safe_distance=4.0)

    assert torch.argsort(collision_score[:, 0]).tolist() == [0, 1]
    assert torch.argsort(noncollision_score[:, 0]).tolist() == [1, 0]
