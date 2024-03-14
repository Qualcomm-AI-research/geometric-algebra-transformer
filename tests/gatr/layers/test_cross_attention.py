# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
import pytest
import torch
from xformers.ops.fmha.attn_bias import BlockDiagonalMask

from gatr.layers import SelfAttention, SelfAttentionConfig
from gatr.layers.attention.cross_attention import CrossAttention
from gatr.utils.clifford import SlowRandomPinTransform
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


@pytest.mark.parametrize("block_attention", [True, False])
def test_cross_attention(block_attention):
    """Test cross attention shapes."""
    if block_attention:
        attn_mask = BlockDiagonalMask.from_seqlens([31, 29, 40], [3, 7, 21])
        num_batch = 1
    else:
        attn_mask = None
        num_batch = 2

    device = torch.device("cuda")

    config = SelfAttentionConfig(
        in_mv_channels=5,
        out_mv_channels=6,
        in_s_channels=2,
        out_s_channels=4,
        num_heads=5,
        increase_hidden_channels=3,
    )
    layer = CrossAttention(config, in_q_mv_channels=6, in_q_s_channels=6)
    layer.to(device)

    num_kv = 10
    num_q = 7
    mv_in = torch.randn(num_batch, num_kv, 5, 16, device=device)
    s_in = torch.randn(num_batch, num_kv, 2, device=device)

    mv_in_q = torch.randn(num_batch, num_q, 6, 16, device=device)
    s_in_q = torch.randn(num_batch, num_q, 6, device=device)

    t = SlowRandomPinTransform()

    mv_out1, s_out1 = layer.forward(mv_in, mv_in_q, s_in, s_in_q, attention_mask=attn_mask)
    mv_out1 = t(mv_out1)

    mv_out2, s_out2 = layer.forward(t(mv_in), t(mv_in_q), s_in, s_in_q, attention_mask=attn_mask)

    assert mv_out1.shape == (num_batch, num_q, 6, 16)
    assert s_out1.shape == (num_batch, num_q, 4)

    torch.testing.assert_close(mv_out1, mv_out2)
    torch.testing.assert_close(s_out1, s_out2)
