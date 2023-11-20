# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
import pytest
import torch

from gatr.baselines.transformer import BaselineAxialTransformer, BaselineTransformer
from tests.helpers import BATCH_DIMS


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("num_items", [5])
@pytest.mark.parametrize("in_channels,hidden_channels,out_channels", [(7, 8, 9)])
@pytest.mark.parametrize("num_heads", [4])
@pytest.mark.parametrize("num_blocks", [3])
@pytest.mark.parametrize("pos_encoding", [False, True])
def test_transformer_shape(
    batch_dims,
    num_items,
    in_channels,
    hidden_channels,
    out_channels,
    num_heads,
    num_blocks,
    pos_encoding,
):
    """Tests the output shape of BaselineTransformer."""
    inputs = torch.randn(*batch_dims, num_items, in_channels)
    net = BaselineTransformer(
        in_channels,
        out_channels,
        hidden_channels,
        num_blocks=num_blocks,
        num_heads=num_heads,
        pos_encoding=pos_encoding,
    )
    outputs = net(inputs)

    assert outputs.shape == (*batch_dims, num_items, out_channels)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("num_items1,num_items2", [(5, 11)])
@pytest.mark.parametrize("in_channels,hidden_channels,out_channels", [(7, 8, 9)])
@pytest.mark.parametrize("num_heads", [4])
@pytest.mark.parametrize("num_blocks", [3])
@pytest.mark.parametrize("pos_encodings", [(False, False), (True, False), (True, True)])
def test_axial_transformer_shape(
    batch_dims,
    num_items1,
    num_items2,
    in_channels,
    hidden_channels,
    out_channels,
    num_heads,
    num_blocks,
    pos_encodings,
):
    """Tests the output shape of BaselineAxialTransformer."""
    inputs = torch.randn(*batch_dims, num_items1, num_items2, in_channels)
    net = BaselineAxialTransformer(
        in_channels,
        out_channels,
        hidden_channels,
        num_blocks=num_blocks,
        num_heads=num_heads,
        pos_encodings=pos_encodings,
    )
    outputs = net(inputs)

    assert outputs.shape == (*batch_dims, num_items1, num_items2, out_channels)
