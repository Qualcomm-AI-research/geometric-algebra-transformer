# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
"""Unit tests of nonlinearity primitives."""

import pytest
import torch

from gatr.primitives import equi_layer_norm, norm
from tests.helpers import TOLERANCES, check_pin_equivariance


@pytest.mark.parametrize("batch_dims", [(7, 9)])
@pytest.mark.parametrize("scale", [0.1, 1.0, 50.0])
def test_equi_layer_norm_correctness(batch_dims, scale):
    """Tests whether the output of equi_layer_norm has the correct variance."""
    inputs = scale * torch.randn(*batch_dims, 16)
    normalized_inputs = equi_layer_norm(inputs, gain=1.0, epsilon=1e-9)
    variance = torch.mean(norm(normalized_inputs) ** 2)
    torch.testing.assert_close(variance, torch.ones_like(variance), **TOLERANCES)


@pytest.mark.parametrize("batch_dims", [(7, 9)])
def test_equi_layer_norm_equivariance(batch_dims):
    """Tests equi_layer_norm() primitive for equivariance."""
    check_pin_equivariance(equi_layer_norm, 1, batch_dims=batch_dims, **TOLERANCES)
