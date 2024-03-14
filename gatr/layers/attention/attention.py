# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
"""Self-attention layers."""

from functools import partial

import torch
from torch import nn

from gatr.layers.attention.config import SelfAttentionConfig
from gatr.primitives.attention import _lin_square_normalizer, geometric_attention


class GeometricAttention(nn.Module):
    """Geometric attention layer.

    This is the main attention mechanism used in GATr. Thanks to the nonlinear features, the
    scaled-dot-product attention takes into account the Euclidean distance.

    Given multivector and scalar queries, keys, and values, this layer computes:

    ```
    attn_weights[..., i, j] = softmax_j[
        weights[0] * pga_inner_product(q_mv[..., i, :, :], k_mv[..., j, :, :])
        + weights[1] * inner_product(phi(q_s[..., i, :]), psi(k_s[..., j, :]))
        + weights[2] * euclidean_inner_product(q_s[..., i, :], k_s[..., j, :])
    ]
    out_mv[..., i, c, :] = sum_j attn_weights[..., i, j] v_mv[..., j, c, :] / norm
    out_s[..., i, c] = sum_j attn_weights[..., i, j] v_s[..., j, c] / norm
    ```

    Parameters
    ----------
    config : SelfAttentionConfig
        Attention configuration.
    """

    def __init__(self, config: SelfAttentionConfig) -> None:
        super().__init__()

        self.normalizer = partial(_lin_square_normalizer, epsilon=config.normalizer_eps)
        self.log_weights = nn.Parameter(
            torch.zeros((config.num_heads, 1, config.hidden_mv_channels))
        )

    def forward(self, q_mv, k_mv, v_mv, q_s, k_s, v_s, attention_mask=None):
        """Forward pass through geometric attention.

        Given multivector and scalar queries, keys, and values, this forward pass computes:

        ```
        attn_weights[..., i, j] = softmax_j[
            weights[0] * pga_inner_product(q_mv[..., i, :, :], k_mv[..., j, :, :])
            + weights[1] * inner_product(phi(q_s[..., i, :]), psi(k_s[..., j, :]))
            + weights[2] * euclidean_inner_product(q_s[..., i, :], k_s[..., j, :])
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
        q_s : Tensor with shape (..., heads, num_items_out, num_s_channels_in)
            Queries, scalar part.
        k_s : Tensor with shape (..., heads, num_items_in, num_s_channels_in)
            Keys, scalar part.
        v_s : Tensor with shape (..., heads, num_items_in, num_s_channels_out)
            Values, scalar part.
        attention_mask: None or Tensor or AttentionBias
            Optional attention mask.
        """

        weights = self.log_weights.exp()
        h_mv, h_s = geometric_attention(
            q_mv,
            k_mv,
            v_mv,
            q_s,
            k_s,
            v_s,
            normalizer=self.normalizer,
            weights=weights,
            attn_mask=attention_mask,
        )

        return h_mv, h_s
