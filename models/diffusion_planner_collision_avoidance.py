import torch


def center_rect_to_points(rect):
    """
    Convert center-format oriented rectangles to corner points.

    rect: [N, 6] = (x, y, cos_h, sin_h, length, width)
    return: [N, 4, 2]
    """
    if rect.numel() == 0:
        return rect.new_zeros((0, 4, 2))

    xy = rect[:, :2]
    cos_h = rect[:, 2]
    sin_h = rect[:, 3]
    lw = rect[:, 4:]

    rot = torch.stack([cos_h, -sin_h, sin_h, cos_h], dim=1).reshape(-1, 2, 2)
    base = rect.new_tensor(
        [[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]]
    ) / 2.0

    local = torch.einsum("nd,kd->nkd", lw, base)
    rotated = torch.einsum("nkd,ndh->nkh", local, rot)
    return xy[:, None, :] + rotated


def batch_signed_distance_rect(rect1, rect2, eps=1.0e-6):
    """
    Signed separating-axis distance between two batches of oriented rectangles.

    rect1, rect2: [N, 4, 2]
    return: [N], positive when separated and negative when overlapped.
    """
    if rect1.numel() == 0 or rect2.numel() == 0:
        return rect1.new_zeros((0,))

    axes = torch.stack(
        [
            rect1[:, 0] - rect1[:, 1],
            rect1[:, 1] - rect1[:, 2],
            rect2[:, 0] - rect2[:, 1],
            rect2[:, 1] - rect2[:, 2],
        ],
        dim=1,
    )
    axes = axes / torch.norm(axes, dim=2, keepdim=True).clamp_min(float(eps))

    proj1 = torch.einsum("nij,nkj->nik", axes, rect1)
    proj2 = torch.einsum("nij,nkj->nik", axes, rect2)
    proj1_min, proj1_max = proj1.min(dim=2).values, proj1.max(dim=2).values
    proj2_min, proj2_max = proj2.min(dim=2).values, proj2.max(dim=2).values

    overlap = torch.cat([proj1_min - proj2_max, proj2_min - proj1_max], dim=1)
    positive_distance = torch.where(overlap < 0.0, torch.full_like(overlap, 1.0e5), overlap)
    is_overlap = (overlap < 0.0).all(dim=1)
    return torch.where(is_overlap, overlap.max(dim=1).values, positive_distance.min(dim=1).values)


def heading_from_positions(pos, eps=1.0e-6):
    """
    Estimate heading unit vectors from positions with stable low-motion fallback.

    pos: [..., T, 2]
    return: [..., T, 2] = (cos_h, sin_h)
    """
    heading = torch.zeros_like(pos)
    if pos.size(-2) <= 0:
        return heading

    delta = torch.zeros_like(pos)
    if pos.size(-2) > 1:
        delta[..., :-1, :] = pos[..., 1:, :] - pos[..., :-1, :]
        delta[..., -1, :] = delta[..., -2, :]
    else:
        delta[..., 0, 0] = 1.0

    speed = torch.norm(delta, dim=-1)
    prev = torch.zeros_like(pos[..., 0, :])
    prev[..., 0] = 1.0

    for t in range(pos.size(-2)):
        valid = speed[..., t] > float(eps)
        current = delta[..., t, :] / speed[..., t].unsqueeze(-1).clamp_min(float(eps))
        current = torch.where(valid.unsqueeze(-1), current, prev)
        heading[..., t, :] = current
        prev = current

    return heading


def _size_tensor(size, device, dtype, default):
    if size is None:
        size = default
    value = torch.as_tensor(size, device=device, dtype=dtype).flatten()
    if value.numel() < 2:
        value = torch.as_tensor(default, device=device, dtype=dtype).flatten()
    return value[:2]


def _align_neighbor_pos(gen_pos, neighbor_pos):
    if neighbor_pos.dim() == 3:
        neighbor_pos = neighbor_pos.unsqueeze(1)
    if neighbor_pos.dim() != 4 or neighbor_pos.size(-1) != 2:
        raise ValueError("neighbor_pos must have shape [B,T,2] or [B,M,T,2]")

    batch = gen_pos.size(0)
    if neighbor_pos.size(0) == 1 and batch > 1:
        neighbor_pos = neighbor_pos.expand(batch, -1, -1, -1)
    elif neighbor_pos.size(0) != batch:
        min_batch = min(batch, neighbor_pos.size(0))
        gen_pos = gen_pos[:min_batch]
        neighbor_pos = neighbor_pos[:min_batch]

    return gen_pos, neighbor_pos


def _align_valid_mask(valid_mask, batch, neighbors, horizon, device, dtype):
    if valid_mask is None:
        return torch.ones((batch, neighbors, horizon), device=device, dtype=dtype)

    valid_mask = torch.as_tensor(valid_mask, device=device, dtype=dtype)
    if valid_mask.dim() == 2:
        valid_mask = valid_mask.unsqueeze(1)
    if valid_mask.dim() != 3:
        return torch.ones((batch, neighbors, horizon), device=device, dtype=dtype)

    if valid_mask.size(0) == 1 and batch > 1:
        valid_mask = valid_mask.expand(batch, -1, -1)
    elif valid_mask.size(0) != batch:
        if batch % valid_mask.size(0) == 0:
            valid_mask = valid_mask.repeat(batch // valid_mask.size(0), 1, 1)
        else:
            valid_mask = valid_mask[:batch]

    if valid_mask.size(1) == 1 and neighbors > 1:
        valid_mask = valid_mask.expand(-1, neighbors, -1)
    elif valid_mask.size(1) != neighbors:
        valid_mask = valid_mask[:, :neighbors]

    return valid_mask[:, :, :horizon]


def diffusion_planner_collision_energy(
    gen_pos,
    neighbor_pos,
    gen_size=None,
    neighbor_size=None,
    r=1.0,
    omega_c=1.0,
    eps=1.0e-6,
    valid_mask=None,
    bbox_inflation=0.0,
    reduction="mean",
):
    """
    Paper-faithful Diffusion Planner Eq. (9) collision-avoidance energy.

    gen_pos: [B, T, 2]
    neighbor_pos: [B, T, 2] or [B, M, T, 2]
    """
    if gen_pos.dim() != 3 or gen_pos.size(-1) != 2:
        raise ValueError("gen_pos must have shape [B,T,2]")

    gen_pos, neighbor_pos = _align_neighbor_pos(gen_pos, neighbor_pos)
    if gen_pos.size(0) == 0:
        return gen_pos.new_tensor(0.0)

    batch, neighbors, horizon, _ = neighbor_pos.shape
    horizon = min(gen_pos.size(1), horizon)
    if horizon <= 0 or neighbors <= 0:
        return gen_pos.new_tensor(0.0)

    gen_pos = gen_pos[:, :horizon]
    neighbor_pos = neighbor_pos[:, :, :horizon]

    gen_heading = heading_from_positions(gen_pos, eps=eps)
    neighbor_heading = heading_from_positions(neighbor_pos, eps=eps)

    default_size = (4.8, 2.0)
    gen_lw = _size_tensor(gen_size, gen_pos.device, gen_pos.dtype, default_size)
    nbr_lw = _size_tensor(neighbor_size, gen_pos.device, gen_pos.dtype, default_size)
    gen_lw = gen_lw + float(bbox_inflation)
    nbr_lw = nbr_lw + float(bbox_inflation)

    gen_lw = gen_lw.view(1, 1, 2).expand(batch, horizon, 2)
    nbr_lw = nbr_lw.view(1, 1, 1, 2).expand(batch, neighbors, horizon, 2)

    gen_rect = torch.cat([gen_pos, gen_heading, gen_lw], dim=-1)
    nbr_rect = torch.cat([neighbor_pos, neighbor_heading, nbr_lw], dim=-1)

    gen_corners = center_rect_to_points(gen_rect.reshape(batch * horizon, 6))
    gen_corners = gen_corners.reshape(batch, 1, horizon, 4, 2).expand(batch, neighbors, horizon, 4, 2)
    nbr_corners = center_rect_to_points(nbr_rect.reshape(batch * neighbors * horizon, 6))
    nbr_corners = nbr_corners.reshape(batch, neighbors, horizon, 4, 2)

    signed_distance = batch_signed_distance_rect(
        gen_corners.reshape(batch * neighbors * horizon, 4, 2),
        nbr_corners.reshape(batch * neighbors * horizon, 4, 2),
        eps=eps,
    ).reshape(batch, neighbors, horizon)

    valid_mask = _align_valid_mask(
        valid_mask,
        batch=batch,
        neighbors=neighbors,
        horizon=horizon,
        device=gen_pos.device,
        dtype=gen_pos.dtype,
    )

    r = max(float(r), float(eps))
    omega_c = max(float(omega_c), float(eps))

    c = torch.relu(1.0 - signed_distance / r)
    psi = torch.exp(omega_c * c) - (omega_c * c)

    pos_mask = (signed_distance > 0.0).to(gen_pos.dtype) * valid_mask
    neg_mask = (signed_distance < 0.0).to(gen_pos.dtype) * valid_mask

    pos_energy = (pos_mask * psi).sum(dim=(1, 2)) / (pos_mask.sum(dim=(1, 2)) + float(eps))
    neg_energy = (neg_mask * psi).sum(dim=(1, 2)) / (neg_mask.sum(dim=(1, 2)) + float(eps))
    per_batch = (pos_energy + neg_energy) / omega_c

    if reduction == "none":
        return per_batch
    if reduction == "sum":
        return per_batch.sum()
    return per_batch.mean()
