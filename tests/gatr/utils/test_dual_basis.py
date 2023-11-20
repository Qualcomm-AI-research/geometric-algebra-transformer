# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
import numpy as np
from clifford.pga import layout

from gatr.utils.dual_basis import compute_dual_basis, dual, point_r3_expand, point_r3_to_mv


def test_dual_basis():
    """Tests correctnes sof dual basis."""
    w = compute_dual_basis()
    x = layout.MultiVector(np.random.rand(16))
    p_r3 = np.random.rand(3)
    p = point_r3_to_mv(p_r3)
    p_exp = point_r3_expand(p_r3)
    dual1 = dual(p, x).value
    dual2 = np.einsum("ijc,c,j->i", w, p_exp, x.value)
    np.testing.assert_allclose(dual1, dual2)
