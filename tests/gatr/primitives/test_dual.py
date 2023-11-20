# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
"""Unit tests of bilinear primitives."""

import pytest
import torch

from gatr.primitives.dual import (
    dual,
    efficient_equivariant_join,
    equivariant_join,
    explicit_equivariant_join,
    join_norm,
)
from gatr.primitives.invariants import norm
from tests.helpers import (
    BATCH_DIMS,
    TOLERANCES,
    check_consistence_with_dual,
    check_pin_equivariance,
)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_dual_correctness(batch_dims):
    """Tests that dual() computes the correct dual."""
    check_consistence_with_dual(dual, batch_dims)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_join_norm_correctness(batch_dims):
    """Tests the join_norm() primitive for correctness."""
    x = torch.randn(*batch_dims, 16)
    y = torch.randn(*batch_dims, 16)
    reference = torch.randn(*batch_dims, 16)

    true_result = norm(equivariant_join(x, y, reference) / torch.abs(reference[..., [14]]))
    result = join_norm(x, y)
    torch.testing.assert_close(result, true_result)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_join_equivariance(batch_dims):
    """Tests the join() primitive for equivariance."""
    check_pin_equivariance(equivariant_join, 3, batch_dims=batch_dims, **TOLERANCES, num_checks=10)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_efficient_join_implementation_equivalence(batch_dims):
    """Tests that efficient_reference_dual() and explicit_reference_dual() agree."""
    x = torch.randn(*batch_dims, 16)
    y = torch.randn(*batch_dims, 16)
    reference = torch.randn(*batch_dims, 16)

    result_expl = explicit_equivariant_join(x, y, reference)
    result_eff = efficient_equivariant_join(x, y, reference)
    torch.testing.assert_close(result_expl, result_eff)
