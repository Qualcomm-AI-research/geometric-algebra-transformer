# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
"""Unit tests of bilinear primitives."""

import pytest
import torch

from gatr.primitives import inner_product, norm, pin_invariants
from tests.helpers import BATCH_DIMS, TOLERANCES, check_pin_invariance


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_norm_correctness(batch_dims):
    """Checks that norm() is consistent with inner_product()."""
    inputs = torch.randn(*batch_dims, 16)
    norms = norm(inputs)
    true_norms = torch.sqrt(inner_product(inputs, inputs))
    torch.testing.assert_close(norms, true_norms)


@pytest.mark.parametrize(
    "vector,true_norm",
    [
        ((0.0, 0.0, 0.0), 0.0),
        ((1.0, 0.0, 0.0), 1.0),
        ((0.0, -2.0, 0.0), 2.0),
        ((0.0, 0.0, 3.141), 3.141),
        (None, None),
    ],
)
def test_norm_of_vector(vector, true_norm):
    """Computes the norm of a pure vector and compares against a known result."""

    # If vector is None, randomly sample it
    if vector is None:
        vector = torch.randn(3)
        true_norm = torch.linalg.norm(vector)

    # Construct multivector
    inputs = torch.zeros(16)
    inputs[2:5] = torch.Tensor(vector)

    # Compute norm
    result = norm(inputs)

    # Validate result
    true_norm = torch.Tensor([true_norm])

    torch.testing.assert_close(result, true_norm)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_inner_product_invariance(batch_dims):
    """Tests the innner_product() primitive for equivariance."""
    check_pin_invariance(inner_product, 2, batch_dims=batch_dims, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_norm_invariance(batch_dims):
    """Tests the norm() primitive for equivariance."""
    check_pin_invariance(norm, 1, batch_dims=batch_dims, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_pin_invariants_invariance(batch_dims):
    """Tests the pin_invariants() primitive for equivariance."""
    check_pin_invariance(pin_invariants, 1, batch_dims=batch_dims, **TOLERANCES)
