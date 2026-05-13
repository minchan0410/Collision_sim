import torch

from models.diffusion import DiffusionTraj
from models.diffusion_planner_collision_avoidance import (
    batch_signed_distance_rect,
    center_rect_to_points,
    diffusion_planner_collision_energy,
)
from models.safesim_guidance import (
    calculate_exact_speed_penalty,
    causecollision_loss,
    collision_sample_score,
    compute_pair_kinematics,
    compute_safesim_ttc_score,
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


def _rect(x, y, length=4.0, width=2.0):
    return torch.tensor([[x, y, 1.0, 0.0, length, width]], dtype=torch.float32)


def test_dp_signed_distance_rect_signs():
    ego = center_rect_to_points(_rect(0.0, 0.0))
    far = center_rect_to_points(_rect(10.0, 0.0))
    overlap = center_rect_to_points(_rect(1.0, 0.0))

    far_distance = batch_signed_distance_rect(ego, far)
    overlap_distance = batch_signed_distance_rect(ego, overlap)

    assert far_distance.item() > 0.0
    assert overlap_distance.item() < 0.0


def test_dp_noncollision_far_has_small_gradient():
    gen = torch.zeros(1, 5, 2, requires_grad=True)
    neighbor = torch.zeros(1, 5, 2)
    neighbor[..., 0] = 20.0

    loss = diffusion_planner_collision_energy(
        gen,
        neighbor,
        gen_size=(4.0, 2.0),
        neighbor_size=(4.0, 2.0),
        r=1.0,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(gen.grad).all()
    assert gen.grad.abs().max() < 1.0e-4


def test_dp_noncollision_overlap_has_finite_gradient():
    gen = torch.zeros(1, 5, 2, requires_grad=True)
    neighbor = torch.zeros(1, 5, 2)
    neighbor[..., 0] = 0.5
    neighbor[..., 1] = 0.25

    loss = diffusion_planner_collision_energy(
        gen,
        neighbor,
        gen_size=(4.0, 2.0),
        neighbor_size=(4.0, 2.0),
        r=1.0,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(gen.grad).all()
    assert gen.grad.abs().max() > 0.0


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
    close = torch.zeros(1, 2, 2)
    far = torch.zeros(1, 2, 2)
    far[..., 0] = 20.0
    gen_pos = torch.cat([close, far], dim=0)
    ref_pos = torch.zeros_like(gen_pos)
    noncollision_score = diffusion_planner_collision_energy(
        gen_pos,
        ref_pos,
        gen_size=(4.0, 2.0),
        neighbor_size=(4.0, 2.0),
        reduction="none",
    )

    assert torch.argsort(collision_score[:, 0]).tolist() == [0, 1]
    assert torch.argsort(noncollision_score).tolist() == [1, 0]


def test_interaction_guidance_scale_multiplies_collision_and_noncollision_objectives():
    diffusion = DiffusionTraj(net=None, var_sched=None)
    vel_phys = torch.zeros(1, 2, 2)
    ref_pos = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])

    base_guidance = {
        "collision_reference_positions": ref_pos,
        "dt": 1.0,
        "eps": 1.0e-6,
        "causecollision_adv_term_weight": {"distance": 1.0, "speed_penalty": 0.0, "filtered_distance": 0.1},
        "dp_noncol_gen_size": (4.0, 2.0),
        "dp_noncol_neighbor_size": (4.0, 2.0),
    }

    for collision_enabled, not_collision_enabled in ((True, False), (False, True)):
        guidance = dict(
            base_guidance,
            collision_enabled=collision_enabled,
            not_collision_enabled=not_collision_enabled,
            interaction_guidance_scale=1.0,
        )
        base = diffusion._compute_collision_objective(vel_phys, guidance)

        guidance["interaction_guidance_scale"] = 4.0
        scaled = diffusion._compute_collision_objective(vel_phys, guidance)

        assert torch.allclose(scaled, base * 4.0)
