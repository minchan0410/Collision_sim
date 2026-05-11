import torch

from models.safesim_guidance import (
    collision_objective,
    compute_pair_kinematics,
    compute_safesim_ttc_score,
    not_collision_objective,
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


def test_collision_objective_prefers_dangerous_close_pair():
    close_distance = torch.tensor([[0.5, 0.4]])
    close_rel_speed = torch.tensor([[1.0, 1.2]])
    close_danger = torch.tensor([[0.8, 0.9]])

    far_distance = torch.tensor([[5.0, 4.5]])
    far_rel_speed = torch.tensor([[0.1, 0.1]])
    far_danger = torch.tensor([[0.0, 0.0]])

    close_obj = collision_objective(close_distance, close_rel_speed, close_danger)
    far_obj = collision_objective(far_distance, far_rel_speed, far_danger)

    assert close_obj < far_obj


def test_not_collision_objective_penalizes_dangerous_close_pair():
    close_distance = torch.tensor([[0.5, 0.4]])
    close_rel_speed = torch.tensor([[1.0, 1.2]])
    close_closing = torch.tensor([[1.0, 1.2]])
    close_danger = torch.tensor([[0.8, 0.9]])

    far_distance = torch.tensor([[5.0, 4.5]])
    far_rel_speed = torch.tensor([[0.1, 0.1]])
    far_closing = torch.tensor([[-0.2, -0.2]])
    far_danger = torch.tensor([[0.0, 0.0]])

    close_obj = not_collision_objective(close_distance, close_rel_speed, close_closing, close_danger)
    far_obj = not_collision_objective(far_distance, far_rel_speed, far_closing, far_danger)

    assert close_obj > far_obj
