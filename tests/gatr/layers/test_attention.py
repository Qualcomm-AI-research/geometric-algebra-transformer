# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
import pytest
import torch

from gatr.layers import SelfAttention, SelfAttentionConfig
from tests.helpers import BATCH_DIMS, TOLERANCES, check_pin_equivariance


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize(
    "num_items,in_channels,out_channels,increase_hidden_channels", [(2, 4, 4, 2)]
)
@pytest.mark.parametrize("in_s_channels,out_s_channels", [(17, 13)])
@pytest.mark.parametrize("num_heads", [4, 1])
@pytest.mark.parametrize("multi_query", [True, False])
@pytest.mark.parametrize("pos_encoding", [False, True])
def test_attention_equivariance(
    batch_dims,
    num_items,
    in_channels,
    out_channels,
    num_heads,
    in_s_channels,
    out_s_channels,
    pos_encoding,
    multi_query,
    increase_hidden_channels,
):
    """Tests the SelfAttention layer for Pin equivariance, with scalar inputs."""

    config = SelfAttentionConfig(
        in_mv_channels=in_channels,
        out_mv_channels=out_channels,
        in_s_channels=in_s_channels,
        out_s_channels=out_s_channels,
        num_heads=num_heads,
        pos_encoding=pos_encoding,
        multi_query=multi_query,
        increase_hidden_channels=increase_hidden_channels,
    )
    layer = SelfAttention(config)

    data_dims = tuple(list(batch_dims) + [num_items, in_channels])
    scalars = torch.randn(*batch_dims, num_items, in_s_channels)
    check_pin_equivariance(
        layer,
        1,
        batch_dims=data_dims,
        fn_kwargs=dict(scalars=scalars),
        spin=True,
        **TOLERANCES,
    )
