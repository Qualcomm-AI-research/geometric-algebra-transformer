# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
import pytest
import torch

from gatr.layers.mlp.nonlinearities import ScalarGatedNonlinearity
from tests.helpers import BATCH_DIMS, TOLERANCES, check_pin_equivariance


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("num_scalars", [9])
@pytest.mark.parametrize("activation", ["gelu", "relu", "sigmoid"])
def test_scalar_gated_nonlinearity_layer_equivariance(activation, batch_dims, num_scalars):
    """Tests ScalarGatedNonlinearity() for equivariance."""
    layer = ScalarGatedNonlinearity(nonlinearity=activation)
    scalars = torch.randn(*batch_dims, num_scalars)
    check_pin_equivariance(
        layer, 1, batch_dims=batch_dims, fn_kwargs=dict(scalars=scalars), **TOLERANCES
    )
