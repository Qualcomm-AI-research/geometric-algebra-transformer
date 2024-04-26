# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
import math
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
from einops import rearrange
from torch import Tensor
from torch.nn.functional import scaled_dot_product_attention as torch_sdpa
from xformers.ops import AttentionBias, memory_efficient_attention

from gatr.primitives.dual import join_norm
from gatr.primitives.invariants import inner_product
from gatr.utils.einsum import gatr_cache, gatr_einsum
from gatr.utils.misc import minimum_autocast_precision
from gatr.utils.tensors import expand_pairwise, to_nd

# When computing the normalization factor in attention weights, multivectors contribute with the
# following factor:
_MV_SIZE_FACTOR = 8

# Multivector indices that contribute to inner product and trivectors
# All components that contribute to the inner product:
_INNER_PRODUCT_IDX = [0, 2, 3, 4, 8, 9, 10, 14]
# Scalar, non-ideal part of vector and bivector; no trivectors:
_INNER_PRODUCT_WO_TRI_IDX = [0, 2, 3, 4, 8, 9, 10]
# Trivector indices (ideal part first):
_TRIVECTOR_IDX = [11, 12, 13, 14]

# Masked out attention logits are set to this constant (a finite replacement for -inf):
_MASKED_OUT = float("-inf")

# Force the use of xformers attention, even when no xformers attention mask is provided:
FORCE_XFORMERS = False


def sdp_attention(
    q_mv: Tensor,
    k_mv: Tensor,
    v_mv: Tensor,
    q_s: Tensor,
    k_s: Tensor,
    v_s: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Equivariant geometric attention based on scaled dot products.

    Expects both multivector and scalar queries, keys, and values as inputs.
    Then this function computes multivector and scalar outputs in the following way:

    ```
    attn_weights[..., i, j] = softmax_j[
        pga_inner_product(q_mv[..., i, :, :], k_mv[..., j, :, :])
        + euclidean_inner_product(q_s[..., i, :], k_s[..., j, :])
    ]
    out_mv[..., i, c, :] = sum_j attn_weights[..., i, j] v_mv[..., j, c, :] / norm
    out_s[..., i, c] = sum_j attn_weights[..., i, j] v_s[..., j, c] / norm
    ```

    Parameters
    ----------
    q_mv : Tensor with shape (..., num_items_out, num_mv_channels_in, 16)
        Queries, multivector part.
    k_mv : Tensor with shape (..., num_items_in, num_mv_channels_in, 16)
        Keys, multivector part.
    v_mv : Tensor with shape (..., num_items_in, num_mv_channels_out, 16)
        Values, multivector part.
    q_s : Tensor with shape (..., num_items_out, num_s_channels_in)
        Queries, scalar part.
    k_s : Tensor with shape (..., num_items_in, num_s_channels_in)
        Keys, scalar part.
    v_s : Tensor with shape (..., num_items_in, num_s_channels_out)
        Values, scalar part.

    Returns
    -------
    outputs_mv : Tensor with shape (..., num_items_out, num_mv_channels_out, 16)
        Result, multivector part
    outputs_s : Tensor with shape (..., num_items_out, num_s_channels_out)
        Result, scalar part
    """

    # Construct queries and keys by concatenating relevant MV components and aux scalars
    q = torch.cat([rearrange(q_mv[..., _INNER_PRODUCT_IDX], "... c x -> ... (c x)"), q_s], -1)
    k = torch.cat([rearrange(k_mv[..., _INNER_PRODUCT_IDX], "... c x -> ... (c x)"), k_s], -1)

    num_channels_out = v_mv.shape[-2]
    v = torch.cat([rearrange(v_mv, "... c x -> ... (c x)"), v_s], -1)

    v_out = scaled_dot_product_attention(q, k, v)

    v_out_mv = rearrange(v_out[..., : num_channels_out * 16], "... (c x) -> ...  c x", x=16)
    v_out_s = v_out[..., num_channels_out * 16 :]

    return v_out_mv, v_out_s


def pga_attention(
    q_mv: Tensor,
    k_mv: Tensor,
    v_mv: Tensor,
    q_s: Tensor,
    k_s: Tensor,
    v_s: Tensor,
    weights: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
    attention_mask=None,
) -> Tuple[Tensor, Tensor]:
    """Equivariant geometric attention based on scaled dot products and the equivariant join.

    Expects both multivector and scalar queries, keys, and values as inputs.
    Then this function computes multivector and scalar outputs in the following way:

    ```
    attn_weights[..., i, j] = softmax_j[
        pga_inner_product(q_mv[..., i, :, :], k_mv[..., j, :, :])
        + norm(join(q_mv[..., i, :, :], k_mv[..., j, :, :]))
        + euclidean_inner_product(q_s[..., i, :], k_s[..., j, :])
    ]
    out_mv[..., i, c, :] = sum_j attn_weights[..., i, j] v_mv[..., j, c, :] / norm
    out_s[..., i, c] = sum_j attn_weights[..., i, j] v_s[..., j, c] / norm
    ```

    Optionally, the three contributions are weighted with `weights`.

    This is not used in GATr, because it does not reduce to dot-product attention and thus does not
    benefit from efficient implementations like `geometric_attention()` does.

    Parameters
    ----------
    q_mv : Tensor with shape (..., num_items_out, num_mv_channels_in, 16)
        Queries, multivector part.
    k_mv : Tensor with shape (..., num_items_in, num_mv_channels_in, 16)
        Keys, multivector part.
    v_mv : Tensor with shape (..., num_items_in, num_mv_channels_out, 16)
        Values, multivector part.
    q_s : Tensor with shape (..., num_items_out, num_s_channels_in)
        Queries, scalar part.
    k_s : Tensor with shape (..., num_items_in, num_s_channels_in)
        Keys, scalar part.
    v_s : Tensor with shape (..., num_items_in, num_s_channels_out)
        Values, scalar part.
    weights : None, or tuple of three Tensors
        Weights for the combination of the inner product, join, and aux scalar parts
    attention_mask: None or Tensor with shape (..., num_items, num_items)
        Optional attention mask

    Returns
    -------
    outputs_mv : Tensor with shape (..., num_items_out, num_mv_channels_out, 16)
        Result, multivector part.
    outputs_s : Tensor with shape (..., num_items_out, num_s_channels_out)
        Result, scalar part.
    """

    # Negative weights are trouble
    if weights is not None:
        for weight in weights:
            assert torch.min(weight) >= 0.0

    # Compute attention weights, first through the inner product between multivectors...
    q_mv = rearrange(q_mv, "... items_out channels x -> ... items_out 1 channels x")
    k_mv = rearrange(k_mv, "... items_in channels x -> ... 1 items_in channels x")
    h = inner_product(q_mv, k_mv)[..., 0]  # (..., items_out, items_in, channels)
    if weights is not None:
        h = weights[0] * h
    attn_weights = torch.sum(h, dim=-1)  # (..., items_out, items_in)

    # ... then through the join...
    h = -join_norm(
        q_mv, k_mv, channel_sum=True, channel_weights=weights[1] if weights is not None else None
    )[
        ..., 0
    ]  # (..., items_out, items_in)
    attn_weights = attn_weights + h

    # ... and finally from auxiliary scalars
    q_s = rearrange(q_s, "... items_out channels -> ... items_out 1 channels")
    k_s = rearrange(k_s, "... items_in channels -> ... 1 items_in channels")
    h = q_s * k_s  # (..., items_out, items_in, channels)
    if weights is not None:
        h = weights[2] * h
    attn_weights = attn_weights + torch.sum(h, dim=-1)  # (..., items_out, items_in)

    # Attention mask
    if attention_mask is not None:
        attn_weights.masked_fill_(~attention_mask, _MASKED_OUT)

    # Combine and weight
    attn_weights = attn_weights / np.sqrt(2 * q_mv.shape[-2] * _MV_SIZE_FACTOR + q_s.shape[-1])

    # Softmax
    attn_weights = attn_weights.softmax(dim=-1)  # Softmax over items_in

    # Compute attention output
    outputs_mv = torch.einsum(
        "... j i, ... i c x -> ... j c x", attn_weights, v_mv
    )  # (..., items_out, channels, 16)
    outputs_s = torch.einsum(
        "... j i, ... i c -> ... j c", attn_weights, v_s
    )  # (..., items_out, channels)

    return outputs_mv, outputs_s


@gatr_cache
def _build_dist_basis(device, dtype) -> Tuple[Tensor, Tensor]:
    """Compute basis features for queries and keys in the geometric SDP attention.

    Parameters
    ----------
    device: torch.device
        Device.
    dtype: torch.dtype
        Dtype.

    Returns
    -------
    basis_q : Tensor with shape (4, 4, 5)
        Basis features for queries.
    basis_k : Tensor with shape (4, 4, 5)
        Basis features for keys.
    """
    r3 = torch.arange(3, device=device)
    basis_q = torch.zeros((4, 4, 5), device=device, dtype=dtype)
    basis_k = torch.zeros((4, 4, 5), device=device, dtype=dtype)

    # -sum_i (q_i^2) * k_0^2
    basis_q[r3, r3, 0] = 1
    basis_k[3, 3, 0] = -1

    # -q_0^2 * sum_i (k_i^2)
    basis_q[3, 3, 1] = 1
    basis_k[r3, r3, 1] = -1

    # sum_i 2 q_0 q_i k_0 k_i
    basis_q[r3, 3, 2 + r3] = 1
    basis_k[r3, 3, 2 + r3] = 2

    return basis_q, basis_k


def _build_dist_vec(tri: Tensor, basis: Tensor, normalizer: Callable[[Tensor], Tensor]) -> Tensor:
    """Build 5D vector whose inner product with another such vector computes the squared distance.

    Parameters
    ----------
    tri: Tensor
        Batch of multivectors, only trivector part is used.
    basis: Tensor
        One of the bases from _build_dist_basis.
    normalizer: Callable[[Tensor], Tensor]
        A normalization function.

    Returns
    -------
    outputs : Tensor
        Batch of 5D vectors
    """
    tri_normed = tri * normalizer(tri[..., [3]])
    vec = gatr_einsum("xyz,...x,...y->...z", basis, tri_normed, tri_normed)
    return vec


@minimum_autocast_precision(torch.float32, output="low")
def _lin_square_normalizer(v: Tensor, epsilon=0.001) -> Tensor:
    """Apply linear square normalization to the input tensor.

    Parameters
    ----------
    v : Tensor
        Input tensor.
    epsilon : float, optional
        Small constant added to the denominator to avoid division by zero.
        Default is 0.001.

    Returns
    -------
    normalized_v : Tensor
        Normalized tensor after applying linear square normalization.
    """
    return v / (v.pow(2) + epsilon)


def geometric_attention(
    q_mv: Tensor,
    k_mv: Tensor,
    v_mv: Tensor,
    q_s: Tensor,
    k_s: Tensor,
    v_s: Tensor,
    normalizer: Callable[[Tensor], Tensor],
    weights: Optional[Tensor] = None,
    attn_mask: Optional[Union[AttentionBias, Tensor]] = None,
) -> Tuple[Tensor, Tensor]:
    """Equivariant geometric attention based on scaled dot products and nonlinear aux features.

    This is the main attention mechanism used in GATr. Thanks to the nonlinear features, the
    scaled-dot-product attention takes into account the Euclidean distance.

    Expects both multivector and scalar queries, keys, and values as inputs.
    Then this function computes multivector and scalar outputs in the following way:

    ```
    attn_weights[..., i, j] = softmax_j[
        pga_inner_product(q_mv[..., i, :, :], k_mv[..., j, :, :])
        + euclidean_inner_product(q_s[..., i, :], k_s[..., j, :])
        + inner_product(phi(q_s[..., i, :]), psi(k_s[..., j, :]))
    ]
    out_mv[..., i, c, :] = sum_j attn_weights[..., i, j] v_mv[..., j, c, :] / norm
    out_s[..., i, c] = sum_j attn_weights[..., i, j] v_s[..., j, c] / norm
    ```

    Optionally, the three contributions are weighted with `weights`.

    Parameters
    ----------
    q_mv : Tensor with shape (..., num_items_out, num_mv_channels_in, 16)
        Queries, multivector part.
    k_mv : Tensor with shape (..., num_items_in, num_mv_channels_in, 16)
        Keys, multivector part.
    v_mv : Tensor with shape (..., num_items_in, num_mv_channels_out, 16)
        Values, multivector part.
    q_s : Tensor with shape (..., heads, num_items_out, num_s_channels_in)
        Queries, scalar part.
    k_s : Tensor with shape (..., heads, num_items_in, num_s_channels_in)
        Keys, scalar part.
    v_s : Tensor with shape (..., heads, num_items_in, num_s_channels_out)
        Values, scalar part.
    normalizer : callable
        Normalization function.
    weights: Optional[Tensor] with shape (..., 1, num_channels_in)
        Weights for the combination of the inner product, nonlinear distance-aware features, and
        scalar parts.
    attn_mask: None or AttentionBias or Tensor with shape (..., num_items_in, num_items_out)
        Optional attention mask. If provided as a tensor, it should be of either of shape
        `(num_items_in, num_items_out)`, `(..., 1, num_items_in, num_items_out)`, or
        `(..., num_heads, num_items_in, num_items_out)`.

    Returns
    -------
    outputs_mv : Tensor with shape (..., heads, num_items_out, num_channels_out, 16)
        Result, multivector part.
    outputs_s : Tensor with shape (..., heads, num_items_out, num_s_channels_out)
        Result, scalar part.
    """
    bh_shape = q_mv.shape[:-3]
    q_mv = to_nd(q_mv, 5)
    k_mv = to_nd(k_mv, 5)
    v_mv = to_nd(v_mv, 5)
    q_s = to_nd(q_s, 4)
    k_s = to_nd(k_s, 4)
    v_s = to_nd(v_s, 4)

    if isinstance(attn_mask, Tensor) and len(attn_mask.shape) > 2:
        # Attention mask tensors should be reshaped to [-1, heads or 1, q_tokens, k_tokens]
        attn_mask = attn_mask.view(-1, *attn_mask.shape[-3:])

    num_mv_channels_v = v_mv.shape[-2]
    num_s_channels_v = v_s.shape[-1]
    num_mv_channels_qk = q_mv.shape[-2]
    num_s_channels_qk = q_s.shape[-1]

    q_tri = q_mv[..., _TRIVECTOR_IDX]
    k_tri = k_mv[..., _TRIVECTOR_IDX]

    basis_q, basis_k = _build_dist_basis(q_tri.device, q_tri.dtype)

    q_dist = _build_dist_vec(q_tri, basis_q, normalizer)
    k_dist = _build_dist_vec(k_tri, basis_k, normalizer)
    if weights is not None:
        q_dist = q_dist * weights[..., None].to(q_dist.dtype)

    device = q_mv.device
    dtype = q_mv.dtype

    num_channels_qk = num_mv_channels_qk * (7 + 5) + num_s_channels_qk
    num_channels_v = num_mv_channels_v * 16 + num_s_channels_v
    num_channels = max(num_channels_qk, num_channels_v)
    num_channels = 8 * -(-num_channels // 8)  # Ceil to multiple of 8

    q = torch.cat(
        [
            rearrange(q_mv[..., _INNER_PRODUCT_WO_TRI_IDX], "... c x -> ... (c x)"),
            rearrange(q_dist, "... c d -> ... (c d)"),
            q_s,
            torch.zeros(*q_s.shape[:3], num_channels - num_channels_qk, device=device, dtype=dtype),
        ],
        -1,
    )
    k = torch.cat(
        [
            rearrange(k_mv[..., _INNER_PRODUCT_WO_TRI_IDX], "... c x -> ... (c x)"),
            rearrange(k_dist, "... c d -> ... (c d)"),
            k_s,
            torch.zeros(*k_s.shape[:3], num_channels - num_channels_qk, device=device, dtype=dtype),
        ],
        -1,
    )

    v = torch.cat(
        [
            rearrange(v_mv, "... c x -> ... (c x)"),
            v_s,
            torch.zeros(*v_s.shape[:3], num_channels - num_channels_v, device=device, dtype=dtype),
        ],
        -1,
    )
    k = k * math.sqrt(num_channels / num_channels_qk)  # Correct for zero padding
    q, k, v_out = _sdpa_graph_breaking(q, k, v, attn_mask=attn_mask)

    v_out_mv = rearrange(v_out[..., : num_mv_channels_v * 16], "... (c x) -> ...  c x", x=16)
    v_out_s = v_out[..., num_mv_channels_v * 16 : num_mv_channels_v * 16 + num_s_channels_v]

    v_out_mv = v_out_mv.view(*bh_shape, *v_out_mv.shape[-3:])
    v_out_s = v_out_s.view(*bh_shape, *v_out_s.shape[-2:])

    return v_out_mv, v_out_s


@torch.compiler.disable
def _sdpa_graph_breaking(q, k, v, attn_mask):
    """A helper function to isolate the graph-breaking parts of the attention (cf. decorator).

    TODO: This function can be dissolved once we get expand_pairwise to not break the graph;
    then we can simply compiler.disable the xformers attention.

    """
    q, k, v = expand_pairwise(q, k, v, exclude_dims=(-2,))  # Don't expand along token dimension)
    v_out = scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    return q, k, v_out


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Optional[Union[AttentionBias, Tensor]] = None,
) -> Tensor:
    """Execute (vanilla) scaled dot-product attention.

    Dynamically dispatch to xFormers if attn_mask is an instance of xformers.ops.AttentionBias
    or FORCE_XFORMERS is set, use torch otherwise.

    Parameters
    ----------
    query : Tensor
        of shape [batch, head, item, d]
    key : Tensor
        of shape [batch, head, item, d]
    value : Tensor
        of shape [batch, head, item, d]
    attn_mask : Optional[Union[AttentionBias, Tensor]]
        Attention mask

    Returns
    -------
    Tensor
        of shape [batch, head, item, d]
    """
    if FORCE_XFORMERS or isinstance(attn_mask, AttentionBias):
        # [batch, head, item, d] -> [batch, item, head, d]
        query = query.transpose(1, 2).contiguous()
        key = key.transpose(1, 2).contiguous()
        value = value.transpose(1, 2).contiguous()
        out = memory_efficient_attention(
            query.contiguous(), key.contiguous(), value, attn_bias=attn_mask
        )
        out = out.transpose(1, 2)  # [batch, item, head, d] -> [batch, head, item, d]
        return out
    return torch_sdpa(query, key, value, attn_mask=attn_mask)
