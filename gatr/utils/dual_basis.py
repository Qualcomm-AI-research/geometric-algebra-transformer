# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
from functools import lru_cache

import numpy as np
import torch
from clifford import pga as PGA
from clifford.pga import layout

e1, e2, e3, e123 = PGA.blades["e1"], PGA.blades["e2"], PGA.blades["e3"], PGA.blades["e123"]


def point_r3_to_mv(v):
    """Helper function to construct basis for reference_dual()."""
    return e123 + sum(v * [e1.dual(), e2.dual(), e3.dual()])


def point_r3_expand(p_r3):
    """Helper function to construct basis for reference_dual()."""
    p_sq = np.einsum("...i,...j->...ij", p_r3, p_r3).reshape(*p_r3.shape[:-1], 9)
    p_exp = np.concatenate([p_sq, p_r3, np.ones((*p_r3.shape[:-1], 1))], -1)
    return p_exp


def sw(u, x):
    """Helper function to construct basis for reference_dual()."""
    grades = set(g % 2 for g in u.grades())
    assert len(grades) == 1, f"{u} is mixed odd/even"
    if grades.pop() == 0:
        return u * x * u.shirokov_inverse()
    return u * x.gradeInvol() * u.shirokov_inverse()


def dual(p, x):
    """Helper function to construct basis for reference_dual()."""
    p = p(3)  # Take the trivector of p, making it a point
    s = p | -e123  # The scale to make it a proper point
    p = p / s  # Normalize so that e123 dim is 1
    t = 0.5 - (e123 * p) / 2  # Translation from p to origin
    assert sw(t, p) == e123  # Assert this works
    x_t = sw(t, x)  # Move x to origin
    y = x_t.dual()  # Dualize at origin
    y_t = sw(t.inv(), y)  # Move back to p
    return y_t * s  # Re-apply scale of p


def compute_dual_basis():
    """Helper function to construct basis for reference_dual()."""
    num_points = 100
    p_r3 = np.random.rand(num_points, 3)
    p_mv = [point_r3_to_mv(v) for v in p_r3]
    p_exp = point_r3_expand(p_r3)
    ws = []
    for i in range(16):
        x = layout.MultiVector(np.eye(16)[i])
        ys = np.stack([dual(p, x).value for p in p_mv])
        w = (np.linalg.pinv(p_exp) @ ys).T
        ws.append(w)
    w = np.stack(ws, 1)
    w_rounded = (w * 2).round() / 2  # Round to halves
    np.testing.assert_allclose(w, w_rounded, atol=1e-10)
    w = w_rounded
    return w


@lru_cache(5)
def compute_dual_basis_torch(device):
    """Helper function to construct basis for reference_dual()."""
    w = compute_dual_basis()
    return torch.tensor(w, dtype=torch.float32, device=device)
