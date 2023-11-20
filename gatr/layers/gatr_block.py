# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
from dataclasses import replace
from typing import Optional, Tuple

import torch
from torch import nn

from gatr.layers import SelfAttention, SelfAttentionConfig
from gatr.layers.layer_norm import EquiLayerNorm
from gatr.layers.mlp.config import MLPConfig
from gatr.layers.mlp.mlp import GeoMLP


class GATrBlock(nn.Module):
    """Equivariant transformer block for GATr.

    This is the biggest building block of GATr.

    Inputs are first processed by a block consisting of LayerNorm, multi-head geometric
    self-attention, and a residual connection. Then the data is processed by a block consisting of
    another LayerNorm, an item-wise two-layer geometric MLP with GeLU activations, and another
    residual connection.

    Parameters
    ----------
    mv_channels : int
        Number of input and output multivector channels
    s_channels: int
        Number of input and output scalar channels
    attention: SelfAttentionConfig
        Attention configuration
    mlp: MLPConfig
        MLP configuration
    dropout_prob : float or None
        Dropout probability
    """

    def __init__(
        self,
        mv_channels: int,
        s_channels: int,
        attention: SelfAttentionConfig,
        mlp: MLPConfig,
        dropout_prob: Optional[float] = None,
    ) -> None:
        super().__init__()

        # Normalization layer (stateless, so we can use the same layer for both normalization
        # instances)
        self.norm = EquiLayerNorm()

        # Self-attention layer
        attention = replace(
            attention,
            in_mv_channels=mv_channels,
            out_mv_channels=mv_channels,
            in_s_channels=s_channels,
            out_s_channels=s_channels,
            output_init="small",
            dropout_prob=dropout_prob,
        )
        self.attention = SelfAttention(attention)

        # MLP block
        mlp = replace(
            mlp,
            mv_channels=(mv_channels, 2 * mv_channels, mv_channels),
            s_channels=(s_channels, 2 * s_channels, s_channels),
            dropout_prob=dropout_prob,
        )
        self.mlp = GeoMLP(mlp)

    def forward(
        self,
        multivectors: torch.Tensor,
        scalars: torch.Tensor,
        reference_mv=None,
        additional_qk_features_mv=None,
        additional_qk_features_s=None,
        attention_mask=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the transformer block.

        Inputs are first processed by a block consisting of LayerNorm, multi-head geometric
        self-attention, and a residual connection. Then the data is processed by a block consisting
        of another LayerNorm, an item-wise two-layer geometric MLP with GeLU activations, and
        another residual connection.

        Parameters
        ----------
        multivectors : torch.Tensor with shape (..., items, channels, 16)
            Input multivectors.
        scalars : torch.Tensor with shape (..., s_channels)
            Input scalars.
        reference_mv : torch.Tensor with shape (..., 16) or None
            Reference multivector for the equivariant join operation in the MLP.
        additional_qk_features_mv : None or torch.Tensor with shape
            (..., num_items, add_qk_mv_channels, 16)
            Additional Q/K features, multivector part.
        additional_qk_features_s : None or torch.Tensor with shape
            (..., num_items, add_qk_mv_channels, 16)
            Additional Q/K features, scalar part.
        attention_mask: None or torch.Tensor or AttentionBias
            Optional attention mask.

        Returns
        -------
        outputs_mv : torch.Tensor with shape (..., items, channels, 16).
            Output multivectors
        output_scalars : torch.Tensor with shape (..., s_channels)
            Output scalars
        """

        # Attention block: layer norm
        h_mv, h_s = self.norm(multivectors, scalars=scalars)

        # Attention block: self attention
        h_mv, h_s = self.attention(
            h_mv,
            scalars=h_s,
            additional_qk_features_mv=additional_qk_features_mv,
            additional_qk_features_s=additional_qk_features_s,
            attention_mask=attention_mask,
        )

        # Attention block: skip connection
        outputs_mv = multivectors + h_mv
        outputs_s = scalars + h_s

        # MLP block: layer norm
        h_mv, h_s = self.norm(outputs_mv, scalars=outputs_s)

        # MLP block: MLP
        h_mv, h_s = self.mlp(h_mv, scalars=h_s, reference_mv=reference_mv)

        # MLP block: skip connection
        outputs_mv = outputs_mv + h_mv
        outputs_s = outputs_s + h_s

        return outputs_mv, outputs_s
