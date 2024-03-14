# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
import pytest
import torch
from xformers.ops.fmha.attn_bias import BlockDiagonalMask

from gatr.interface import embed_point, embed_scalar
from gatr.primitives import geometric_attention, pga_attention, sdp_attention
from gatr.primitives.attention import _lin_square_normalizer
from tests.helpers import BATCH_DIMS, TOLERANCES, check_pin_equivariance


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("num_s_channels_out", [11, 1])
@pytest.mark.parametrize("num_s_channels_in", [17, 1])
@pytest.mark.parametrize("num_mv_channels_out", [3, 1])
@pytest.mark.parametrize("num_mv_channels_in", [2, 1])
@pytest.mark.parametrize("num_tokens_out", [5, 1])
@pytest.mark.parametrize("num_tokens_in", [7, 1])
def test_scalar_attention_shape(
    batch_dims,
    num_tokens_in,
    num_tokens_out,
    num_mv_channels_in,
    num_mv_channels_out,
    num_s_channels_in,
    num_s_channels_out,
):
    """Tests that outputs of scalar_attention() have correct shape."""
    # Generate inputs
    q_mv = torch.randn(*batch_dims, num_tokens_out, num_mv_channels_in, 16)
    k_mv = torch.randn(*batch_dims, num_tokens_in, num_mv_channels_in, 16)
    v_mv = torch.randn(*batch_dims, num_tokens_in, num_mv_channels_out, 16)
    q_s = torch.randn(*batch_dims, num_tokens_out, num_s_channels_in)
    k_s = torch.randn(*batch_dims, num_tokens_in, num_s_channels_in)
    v_s = torch.randn(*batch_dims, num_tokens_in, num_s_channels_out)

    # Compute attention outputs
    outputs, outputs_scalar = sdp_attention(q_mv, k_mv, v_mv, q_s, k_s, v_s)

    # Check shape of outputs
    assert outputs.shape == (*batch_dims, num_tokens_out, num_mv_channels_out, 16)
    assert outputs_scalar.shape == (*batch_dims, num_tokens_out, num_s_channels_out)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("num_s_channels_out", [11, 1])
@pytest.mark.parametrize("num_s_channels_in", [17, 1])
@pytest.mark.parametrize("num_mv_channels_out", [3, 1])
@pytest.mark.parametrize("num_mv_channels_in", [2, 1])
@pytest.mark.parametrize("num_tokens_out", [5, 1])
@pytest.mark.parametrize("num_tokens_in", [7, 1])
def test_pga_attention_shape(
    batch_dims,
    num_tokens_in,
    num_tokens_out,
    num_mv_channels_in,
    num_mv_channels_out,
    num_s_channels_in,
    num_s_channels_out,
):
    """Tests that outputs of pga_attention() have correct shape."""
    # Generate inputs
    q_mv = torch.randn(*batch_dims, num_tokens_out, num_mv_channels_in, 16)
    k_mv = torch.randn(*batch_dims, num_tokens_in, num_mv_channels_in, 16)
    v_mv = torch.randn(*batch_dims, num_tokens_in, num_mv_channels_out, 16)
    q_s = torch.randn(*batch_dims, num_tokens_out, num_s_channels_in)
    k_s = torch.randn(*batch_dims, num_tokens_in, num_s_channels_in)
    v_s = torch.randn(*batch_dims, num_tokens_in, num_s_channels_out)
    weights = (  # Weights need to be non-negative
        torch.exp(torch.randn(num_mv_channels_in)),
        torch.exp(torch.randn(num_mv_channels_in)),
        torch.exp(torch.randn(num_s_channels_in)),
    )

    # Compute attention outputs
    outputs, outputs_scalar = pga_attention(q_mv, k_mv, v_mv, q_s, k_s, v_s, weights=weights)

    # Check shape of outputs
    assert outputs.shape == (*batch_dims, num_tokens_out, num_mv_channels_out, 16)
    assert outputs_scalar.shape == (*batch_dims, num_tokens_out, num_s_channels_out)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("num_s_channels_out", [11, 1])
@pytest.mark.parametrize("num_s_channels_in", [17, 1])
@pytest.mark.parametrize("num_mv_channels_out", [3, 1])
@pytest.mark.parametrize("num_mv_channels_in", [2, 1])
@pytest.mark.parametrize("num_tokens_out", [5, 1])
@pytest.mark.parametrize("num_tokens_in", [7, 1])
def test_geometric_attention_shape(
    batch_dims,
    num_tokens_in,
    num_tokens_out,
    num_mv_channels_in,
    num_mv_channels_out,
    num_s_channels_in,
    num_s_channels_out,
):
    """Tests that outputs of geometric_attention() have correct shape."""
    # Generate inputs
    q_mv = torch.randn(*batch_dims, num_tokens_out, num_mv_channels_in, 16)
    k_mv = torch.randn(*batch_dims, num_tokens_in, num_mv_channels_in, 16)
    v_mv = torch.randn(*batch_dims, num_tokens_in, num_mv_channels_out, 16)
    q_s = torch.randn(*batch_dims, num_tokens_out, num_s_channels_in)
    k_s = torch.randn(*batch_dims, num_tokens_in, num_s_channels_in)
    v_s = torch.randn(*batch_dims, num_tokens_in, num_s_channels_out)

    # Compute attention outputs
    outputs, outputs_scalar = geometric_attention(
        q_mv, k_mv, v_mv, q_s, k_s, v_s, normalizer=_lin_square_normalizer
    )

    # Check shape of outputs
    assert outputs.shape == (*batch_dims, num_tokens_out, num_mv_channels_out, 16)
    assert outputs_scalar.shape == (*batch_dims, num_tokens_out, num_s_channels_out)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("num_scalars", [5])
@pytest.mark.parametrize("key_dim", [2])
@pytest.mark.parametrize("item_dim", [3])
def test_scalar_attention_equivariance(batch_dims, key_dim, item_dim, num_scalars):
    """Tests scalar_attention() for Pin equivariance."""
    data_dims = tuple(list(batch_dims) + [item_dim, key_dim])
    queries_scalar = torch.randn(*batch_dims, item_dim, num_scalars)
    keys_scalar = torch.randn(*batch_dims, item_dim, num_scalars)
    values_scalar = torch.randn(*batch_dims, item_dim, num_scalars)
    kwargs = dict(q_s=queries_scalar, k_s=keys_scalar, v_s=values_scalar)
    check_pin_equivariance(sdp_attention, 3, batch_dims=data_dims, fn_kwargs=kwargs, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("num_scalars", [5])
@pytest.mark.parametrize("key_dim", [2])
@pytest.mark.parametrize("item_dim", [3])
def test_pga_attention_equivariance(batch_dims, key_dim, item_dim, num_scalars):
    """Tests pga_attention() for Pin equivariance."""
    data_dims = tuple(list(batch_dims) + [item_dim, key_dim])
    queries_scalar = torch.randn(*batch_dims, item_dim, num_scalars)
    keys_scalar = torch.randn(*batch_dims, item_dim, num_scalars)
    values_scalar = torch.randn(*batch_dims, item_dim, num_scalars)
    weights = (  # Weights need to be non-negative
        torch.exp(torch.randn(key_dim)),
        torch.exp(torch.randn(key_dim)),
        torch.exp(torch.randn(num_scalars)),
    )
    kwargs = dict(q_s=queries_scalar, k_s=keys_scalar, v_s=values_scalar, weights=weights)
    check_pin_equivariance(pga_attention, 3, batch_dims=data_dims, fn_kwargs=kwargs, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("num_scalars", [5])
@pytest.mark.parametrize("key_dim", [2])
@pytest.mark.parametrize("item_dim", [3])
def test_geometric_attention_equivariance(batch_dims, key_dim, item_dim, num_scalars):
    """Tests geometric_attention() for Pin equivariance."""
    data_dims = tuple(list(batch_dims) + [item_dim, key_dim])
    queries_scalar = torch.randn(*batch_dims, item_dim, num_scalars)
    keys_scalar = torch.randn(*batch_dims, item_dim, num_scalars)
    values_scalar = torch.randn(*batch_dims, item_dim, num_scalars)
    kwargs = dict(q_s=queries_scalar, k_s=keys_scalar, v_s=values_scalar, normalizer=lambda v: 1.0)
    check_pin_equivariance(
        geometric_attention, 3, batch_dims=data_dims, fn_kwargs=kwargs, **TOLERANCES
    )


def test_pga_attention_proximity():
    """Tests that with the right weights, pga_attention attends to nearby points."""

    # Data
    q_mv = embed_point(torch.zeros(1, 1, 3))  # (1, 1, 16), point at the origin
    k_mv = embed_point(
        torch.tensor([[[1.0, 0.0, 0.0]], [[0.0, 100.0, 0.0]]])
    )  # (2, 1, 16), close point and far point
    v_mv = embed_scalar(
        torch.tensor([[[1.0], [0.0]], [[0.0], [1.0]]])
    )  # (2, 2, 16), one-hot encoding of what was attended to
    q_s = torch.zeros(1, 1)  # we don't care about scalars
    k_s = torch.zeros(2, 1)
    v_s = torch.zeros(2, 1)
    weights = torch.tensor(
        [0.0, 1.0, 0.0]
    )  # Only attend based on join - not inner products or aux scalars

    # Compute attention
    out_mv, out_s = pga_attention(q_mv, k_mv, v_mv, q_s, k_s, v_s, weights)

    # Check that we attended to closer one
    torch.testing.assert_close(out_mv, v_mv[[0]], **TOLERANCES)
    torch.testing.assert_close(out_s, v_s[[0]], **TOLERANCES)


def test_geometric_attention_proximity():
    """Tests that with the right weights, geometric_attention attends to nearby points."""

    # Data
    q_mv = embed_point(torch.zeros(1, 1, 3))  # (1, 1, 16), point at the origin
    k_mv = embed_point(
        torch.tensor([[[1.0, 0.0, 0.0]], [[0.0, 100.0, 0.0]]])
    )  # (2, 1, 16), close point and far point
    v_mv = embed_scalar(
        torch.tensor([[[1.0], [0.0]], [[0.0], [1.0]]])
    )  # (2, 2, 16), one-hot enc of attn target
    q_s = torch.zeros(1, 1)  # we don't care about scalars
    k_s = torch.zeros(2, 1)
    v_s = torch.zeros(2, 1)

    # Compute attention
    out_mv, out_s = geometric_attention(
        q_mv, k_mv, v_mv, q_s, k_s, v_s, normalizer=_lin_square_normalizer
    )

    # Check that we attended to closer one
    torch.testing.assert_close(out_mv, v_mv[[0]], **TOLERANCES)
    torch.testing.assert_close(out_s, v_s[[0]], **TOLERANCES)


@pytest.mark.parametrize("block_attention", [True, False])
def test_geometric_cross_attention(block_attention):
    """Test cross attention shapes."""
    num_q = 100
    num_kv = 51
    num_heads = 2
    mv_ch_qk = 3
    mv_ch_v = 4
    s_ch_qk = 3
    s_ch_v = 4
    multi_query = True
    num_heads_kv = 1 if multi_query else num_heads
    device = torch.device("cuda")

    if block_attention:
        attn_mask = BlockDiagonalMask.from_seqlens([31, 29, 40], [3, 7, 21])
        num_batch = 1
    else:
        attn_mask = None
        num_batch = 2

    # Data
    q_mv = torch.randn(num_batch, num_heads, num_q, mv_ch_qk, 16, device=device)
    k_mv = torch.randn(num_batch, num_heads_kv, num_kv, mv_ch_qk, 16, device=device)
    v_mv = torch.randn(num_batch, num_heads_kv, num_kv, mv_ch_v, 16, device=device)
    q_s = torch.randn(num_batch, num_heads, num_q, s_ch_qk, device=device)
    k_s = torch.randn(num_batch, num_heads_kv, num_kv, s_ch_qk, device=device)
    v_s = torch.randn(num_batch, num_heads_kv, num_kv, s_ch_v, device=device)

    out_mv, out_s = geometric_attention(
        q_mv,
        k_mv,
        v_mv,
        q_s,
        k_s,
        v_s,
        normalizer=_lin_square_normalizer,
        attn_mask=attn_mask,
    )

    assert out_mv.shape == (num_batch, num_heads, num_q, mv_ch_v, 16)
    assert out_s.shape == (num_batch, num_heads, num_q, s_ch_v)
