# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
import pytest
import torch

from gatr.layers.mlp.geometric_bilinears import GeometricBilinear
from tests.helpers import BATCH_DIMS, TOLERANCES, check_pin_equivariance


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("in_mv_channels", [8])
@pytest.mark.parametrize("out_mv_channels", [10])
@pytest.mark.parametrize("in_s_channels", [3])
@pytest.mark.parametrize("out_s_channels", [5])
def test_geometric_bilinears_equivariance(
    batch_dims, in_mv_channels, out_mv_channels, in_s_channels, out_s_channels
):
    """Tests GeometricBilinear() for equivariance."""

    layer = GeometricBilinear(
        in_mv_channels,
        out_mv_channels,
        in_s_channels=in_s_channels,
        out_s_channels=out_s_channels,
    )
    data_dims = tuple(list(batch_dims) + [in_mv_channels])
    scalars = torch.randn(*batch_dims, in_s_channels)
    reference_mv = torch.randn(16)

    # Because of the fixed reference MV, we only test Spin equivariance
    check_pin_equivariance(
        layer,
        1,
        fn_kwargs=dict(scalars=scalars, reference_mv=reference_mv),
        batch_dims=data_dims,
        spin=True,
        **TOLERANCES,
    )
