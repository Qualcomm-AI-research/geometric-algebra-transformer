# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
import pytest
import torch

from gatr.layers import GeoMLP
from gatr.layers.mlp.config import MLPConfig
from tests.helpers import BATCH_DIMS, TOLERANCES, check_pin_equivariance

_CHANNELS = [((5,), (12,)), ((4, 8, 7), (10, 9, 3))]


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("activation", ["gelu"])
@pytest.mark.parametrize("mv_channels,s_channels", _CHANNELS)
def test_geo_mlp_shape(batch_dims, mv_channels, s_channels, activation):
    """Tests the output shape of GeoMLP()."""

    inputs = torch.randn(*batch_dims, mv_channels[0], 16)
    scalars = None if s_channels is None else torch.randn(*batch_dims, s_channels[0])
    reference_mv = torch.randn(16)

    net = GeoMLP(MLPConfig(mv_channels=mv_channels, s_channels=s_channels, activation=activation))
    outputs, outputs_scalars = net(inputs, reference_mv=reference_mv, scalars=scalars)
    assert outputs.shape == (*batch_dims, mv_channels[-1], 16)
    assert outputs_scalars.shape == (*batch_dims, s_channels[-1])


@pytest.mark.parametrize("batch_dims", [[100]])
@pytest.mark.parametrize("activation", ["gelu"])
@pytest.mark.parametrize("mv_channels,s_channels", _CHANNELS)
def test_geo_mlp_equivariance(batch_dims, mv_channels, s_channels, activation):
    """Tests GeoMLP() for Pin equivariance."""
    net = GeoMLP(MLPConfig(mv_channels=mv_channels, s_channels=s_channels, activation=activation))
    data_dims = tuple(list(batch_dims) + [mv_channels[0]])
    scalars = torch.randn(*batch_dims, s_channels[0])
    reference_mv = torch.randn(16)

    # Because of the fixed reference MV, we only test Spin equivariance
    check_pin_equivariance(
        net,
        1,
        batch_dims=data_dims,
        fn_kwargs=dict(scalars=scalars, reference_mv=reference_mv),
        spin=True,
        **TOLERANCES,
    )
