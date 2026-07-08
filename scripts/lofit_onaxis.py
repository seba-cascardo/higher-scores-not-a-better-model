"""P9 — on-axis projection penalty toward v_inf, for training-time on-axis-constrained LoFiT.

The constructive flip of the project's central negative: force the trainable offset alpha*theta
to live ON the functional inference axis v_inf by penalizing its off-axis mass. Added to the
task-margin loss in train_v7_lofit.py as `L = L_margin + lam * onaxis_penalty(...)`.

Math (per (layer, head) pair i, with unit v_hat_i = v_inf_i / ||v_inf_i||):
    theta_eff = alpha[:,None] * theta                      # the applied offset (penalize this,
                                                           #   not theta, so alpha can't game it)
    theta_perp = theta_eff - (theta_eff·v_hat) v_hat       # off-axis component
    penalty   = mean_i mask_i * ||theta_perp_i||^2

Provenance (verified 2026-07-08): for Gemma, v_inf lives in runs/functional_directions.pt as
(50 captured layers, 32 heads, 256); captured_indices maps rows->model layer ids and covers all
12 offset layers [30-40, 49, 55] -> 48/48 pairs map (no blind spot on the late L49/L55 drivers).
"""
from __future__ import annotations

import torch


def map_vinf_to_pairs(func_dirs, layer_head_pairs):
    """Map v_inf (captured-layer rows) onto the offset's (layer, head) pairs.

    func_dirs: the loaded functional_directions.pt dict — needs `v_inf` (n_captured, n_heads,
        head_dim) and `captured_indices` (n_captured,) mapping row -> model layer id.
    layer_head_pairs: list of (model_layer, head) from the offset blob.
    Returns (v_inf_perpair (n_pairs, head_dim), mask (n_pairs,)); masked pairs (no captured
    v_inf for that layer) get a zero v_inf row and mask 0.
    """
    v_inf = torch.as_tensor(func_dirs["v_inf"], dtype=torch.float32)
    captured = [int(x) for x in torch.as_tensor(func_dirs["captured_indices"]).reshape(-1).tolist()]
    layer_to_row = {L: r for r, L in enumerate(captured)}
    head_dim = int(v_inf.shape[-1])

    n_pairs = len(layer_head_pairs)
    v_per = torch.zeros(n_pairs, head_dim, dtype=torch.float32)
    mask = torch.zeros(n_pairs, dtype=torch.float32)
    for i, (li, hi) in enumerate(layer_head_pairs):
        li, hi = int(li), int(hi)
        row = layer_to_row.get(li)
        if row is not None:
            v_per[i] = v_inf[row, hi]
            mask[i] = 1.0
    return v_per, mask


def onaxis_penalty(alpha, theta, v_inf_perpair, mask, eps=1e-8):
    """Mean off-axis energy of the applied offset alpha*theta, projected against v_inf.

    alpha: (n_pairs,) — trainable scale. theta: (n_pairs, head_dim) — trainable direction.
    v_inf_perpair: (n_pairs, head_dim). mask: (n_pairs,) 1 where a v_inf cell exists.
    Differentiable in alpha and theta. Averaged over ALL pairs (masked ones contribute 0).
    """
    alpha = torch.as_tensor(alpha, dtype=torch.float32) if not torch.is_tensor(alpha) else alpha
    theta = torch.as_tensor(theta, dtype=torch.float32) if not torch.is_tensor(theta) else theta
    v = v_inf_perpair.to(theta.dtype)
    m = mask.to(theta.dtype)

    theta_eff = alpha.unsqueeze(1) * theta                      # (n_pairs, head_dim)
    v_hat = v / (v.norm(dim=1, keepdim=True) + eps)             # unit; zero rows stay ~zero
    proj_coeff = (theta_eff * v_hat).sum(dim=1, keepdim=True)   # (n_pairs, 1)
    theta_perp = theta_eff - proj_coeff * v_hat                 # off-axis component
    perp_energy = (theta_perp ** 2).sum(dim=1)                  # (n_pairs,)
    return (m * perp_energy).sum() / m.numel()


def frac_perp(alpha, theta, v_inf_perpair, mask, eps=1e-8):
    """Diagnostic: masked-mean ||theta_perp||^2 / ||theta_eff||^2 — how off-axis the offset is
    (1 = fully off-axis, 0 = fully on-axis). Log this during the sweep."""
    theta_eff = alpha.unsqueeze(1) * theta
    v = v_inf_perpair.to(theta.dtype)
    m = mask.to(theta.dtype)
    v_hat = v / (v.norm(dim=1, keepdim=True) + eps)
    proj_coeff = (theta_eff * v_hat).sum(dim=1, keepdim=True)
    perp = ((theta_eff - proj_coeff * v_hat) ** 2).sum(dim=1)
    tot = (theta_eff ** 2).sum(dim=1) + eps
    return float((m * perp / tot).sum() / (m.sum() + eps))
