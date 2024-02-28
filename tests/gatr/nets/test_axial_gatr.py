# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
import pytest
import torch

from gatr.layers.attention.config import SelfAttentionConfig
from gatr.layers.mlp.config import MLPConfig
from gatr.nets import AxialGATr
from tests.helpers import BATCH_DIMS, MILD_TOLERANCES, check_pin_equivariance

S_CHANNELS = [(None, None, 7, [False, False]), (4, 5, 6, [True, True])]


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize(
    "num_items1,num_items2,in_mv_channels,out_mv_channels,hidden_mv_channels", [(8, 11, 6, 4, 8)]
)
@pytest.mark.parametrize("num_heads", [4])
@pytest.mark.parametrize("num_blocks", [1])
@pytest.mark.parametrize("in_s_channels,out_s_channels,hidden_s_channels,pos_encoding", S_CHANNELS)
@pytest.mark.parametrize("join_reference", ["data", "canonical"])
def test_axial_gatr_shape(
    batch_dims,
    num_items1,
    num_items2,
    in_mv_channels,
    out_mv_channels,
    hidden_mv_channels,
    num_blocks,
    num_heads,
    in_s_channels,
    out_s_channels,
    hidden_s_channels,
    pos_encoding,
    join_reference,
):
    """Tests the output shape of AxialGATr."""
    inputs = torch.randn(*batch_dims, num_items1, num_items2, in_mv_channels, 16)
    scalars = (
        None
        if in_s_channels is None
        else torch.randn(*batch_dims, num_items1, num_items2, in_s_channels)
    )
    net = AxialGATr(
        in_mv_channels,
        out_mv_channels,
        hidden_mv_channels,
        attention=SelfAttentionConfig(num_heads=num_heads),
        in_s_channels=in_s_channels,
        out_s_channels=out_s_channels,
        hidden_s_channels=hidden_s_channels,
        num_blocks=num_blocks,
        mlp=MLPConfig(),
        pos_encodings=pos_encoding,
    )

    outputs, output_scalars = net(inputs, scalars=scalars, join_reference=join_reference)

    assert outputs.shape == (*batch_dims, num_items1, num_items2, out_mv_channels, 16)
    if in_s_channels is not None:
        assert output_scalars.shape == (*batch_dims, num_items1, num_items2, out_s_channels)


@pytest.mark.parametrize("batch_dims", [(64,)])
@pytest.mark.parametrize(
    "num_items1,num_items2,in_mv_channels,out_mv_channels,hidden_mv_channels", [(3, 4, 5, 6, 10)]
)
@pytest.mark.parametrize("num_heads", [4])
@pytest.mark.parametrize("num_blocks", [1])
@pytest.mark.parametrize("in_s_channels,out_s_channels,hidden_s_channels,pos_encoding", S_CHANNELS)
def test_axial_gatr_equivariance(
    batch_dims,
    num_items1,
    num_items2,
    in_mv_channels,
    out_mv_channels,
    hidden_mv_channels,
    num_blocks,
    num_heads,
    in_s_channels,
    out_s_channels,
    hidden_s_channels,
    pos_encoding,
):
    """Tests AxialGATr for equivariance."""
    try:
        net = AxialGATr(
            in_mv_channels,
            out_mv_channels,
            hidden_mv_channels,
            attention=SelfAttentionConfig(num_heads=num_heads),
            in_s_channels=in_s_channels,
            out_s_channels=out_s_channels,
            hidden_s_channels=hidden_s_channels,
            num_blocks=num_blocks,
            mlp=MLPConfig(),
            pos_encodings=pos_encoding,
        )
    except NotImplementedError:
        # Some features require scalar inputs, and failing without them is fine
        return

    scalars = (
        None
        if in_s_channels is None
        else torch.randn(*batch_dims, num_items1, num_items2, in_s_channels)
    )
    data_dims = tuple(list(batch_dims) + [num_items1, num_items2, in_mv_channels])

    # We only test for Spin, not Pin, equivariance b/c AxialTransformer currently has some manual breaking of the
    # mirror symmetry
    check_pin_equivariance(
        net, 1, batch_dims=data_dims, fn_kwargs=dict(scalars=scalars), **MILD_TOLERANCES, spin=True
    )
