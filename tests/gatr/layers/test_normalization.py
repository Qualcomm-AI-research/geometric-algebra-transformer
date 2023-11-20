# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
import pytest
import torch

from gatr.layers.layer_norm import EquiLayerNorm
from gatr.primitives import norm
from tests.helpers import TOLERANCES, check_pin_equivariance


@pytest.mark.parametrize("batch_dims", [(7, 9)])
@pytest.mark.parametrize("num_scalars", [9])
def test_equi_layer_norm_layer_correctness(batch_dims, num_scalars):
    """Tests whether the output of EquiLayerNorm has the correct variance."""
    inputs = torch.randn(*batch_dims, 16)
    scalars = torch.randn(*batch_dims, num_scalars)
    layer = EquiLayerNorm(epsilon=1e-9)
    normalized_inputs, _ = layer(inputs, scalars=scalars)
    dims = tuple(range(1, len(batch_dims) + 1))
    variance = torch.mean(norm(normalized_inputs) ** 2, dim=dims)
    torch.testing.assert_close(variance, torch.ones_like(variance), **TOLERANCES)


@pytest.mark.parametrize("batch_dims", [(7, 9)])
@pytest.mark.parametrize("num_scalars", [9])
def test_equi_layer_norm_layer_equivariance(batch_dims, num_scalars):
    """Tests EquiLayerNorm() for equivariance."""
    layer = EquiLayerNorm()
    scalars = torch.randn(*batch_dims, num_scalars)
    check_pin_equivariance(
        layer, 1, batch_dims=batch_dims, fn_kwargs=dict(scalars=scalars), **TOLERANCES
    )
