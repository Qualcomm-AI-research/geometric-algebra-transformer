# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
"""Self-attention layers."""

from typing import Optional, Tuple

import torch
from einops import rearrange
from torch import nn

from gatr.layers.attention.attention import GeometricAttention
from gatr.layers.attention.config import SelfAttentionConfig
from gatr.layers.attention.positional_encoding import ApplyRotaryPositionalEncoding
from gatr.layers.attention.qkv import MultiQueryQKVModule, QKVModule
from gatr.layers.dropout import GradeDropout
from gatr.layers.linear import EquiLinear


class SelfAttention(nn.Module):
    """Geometric self-attention layer.

    Constructs queries, keys, and values, computes attention, and projects linearly to outputs.

    Parameters
    ----------
    config : SelfAttentionConfig
        Attention configuration.
    """

    def __init__(self, config: SelfAttentionConfig) -> None:
        super().__init__()

        # Store settings
        self.config = config

        # QKV computation
        self.qkv_module = MultiQueryQKVModule(config) if config.multi_query else QKVModule(config)

        # Output projection
        self.out_linear = EquiLinear(
            in_mv_channels=config.hidden_mv_channels * config.num_heads,
            out_mv_channels=config.out_mv_channels,
            in_s_channels=(
                None
                if config.in_s_channels is None
                else config.hidden_s_channels * config.num_heads
            ),
            out_s_channels=config.out_s_channels,
            initialization=config.output_init,
        )

        # Optional positional encoding
        self.pos_encoding: nn.Module
        if config.pos_encoding:
            self.pos_encoding = ApplyRotaryPositionalEncoding(
                config.hidden_s_channels, item_dim=-2, base=config.pos_enc_base
            )
        else:
            self.pos_encoding = nn.Identity()

        # Attention
        self.attention = GeometricAttention(config)

        # Dropout
        self.dropout: Optional[nn.Module]
        if config.dropout_prob is not None:
            self.dropout = GradeDropout(config.dropout_prob)
        else:
            self.dropout = None

    def forward(
        self,
        multivectors: torch.Tensor,
        additional_qk_features_mv: Optional[torch.Tensor] = None,
        scalars: Optional[torch.Tensor] = None,
        additional_qk_features_s: Optional[torch.Tensor] = None,
        attention_mask=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes forward pass on inputs with shape `(..., items, channels, 16)`.

        The result is the following:

        ```
        # For each head
        queries = linear_channels(inputs)
        keys = linear_channels(inputs)
        values = linear_channels(inputs)
        hidden = attention_items(queries, keys, values, biases=biases)
        head_output = linear_channels(hidden)

        # Combine results
        output = concatenate_heads head_output
        ```

        Parameters
        ----------
        multivectors : torch.Tensor with shape (..., num_items, channels_in, 16)
            Input multivectors.
        additional_qk_features_mv : None or torch.Tensor with shape
            (..., num_items, add_qk_mv_channels, 16)
            Additional Q/K features, multivector part.
        scalars : None or torch.Tensor with shape (..., num_items, num_items, in_scalars)
            Optional input scalars
        additional_qk_features_s : None or torch.Tensor with shape
            (..., num_items, add_qk_mv_channels, 16)
            Additional Q/K features, scalar part.
        scalars : None or torch.Tensor with shape (..., num_items, num_items, in_scalars)
            Optional input scalars
        attention_mask: None or torch.Tensor with shape (..., num_items, num_items)
            Optional attention mask

        Returns
        -------
        outputs_mv : torch.Tensor with shape (..., num_items, channels_out, 16)
            Output multivectors.
        output_scalars : torch.Tensor with shape (..., num_items, channels_out, out_scalars)
            Output scalars, if scalars are provided. Otherwise None.
        """
        # Compute Q, K, V
        q_mv, k_mv, v_mv, q_s, k_s, v_s = self.qkv_module(
            multivectors, scalars, additional_qk_features_mv, additional_qk_features_s
        )

        # Rotary positional encoding
        q_s = self.pos_encoding(q_s)
        k_s = self.pos_encoding(k_s)

        # Attention layer
        h_mv, h_s = self.attention(q_mv, k_mv, v_mv, q_s, k_s, v_s, attention_mask=attention_mask)

        h_mv = rearrange(
            h_mv, "... n_heads n_items hidden_channels x -> ... n_items (n_heads hidden_channels) x"
        )
        h_s = rearrange(
            h_s, "... n_heads n_items hidden_channels -> ... n_items (n_heads hidden_channels)"
        )

        # Transform linearly one more time
        outputs_mv, outputs_s = self.out_linear(h_mv, scalars=h_s)

        # Dropout
        if self.dropout is not None:
            outputs_mv, outputs_s = self.dropout(outputs_mv, outputs_s)

        return outputs_mv, outputs_s
