# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All rights reserved.
from dataclasses import replace
from typing import Literal, Optional, Sequence, Tuple, Union

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as checkpoint_
from xformers.ops import AttentionBias

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
    checkpoint : None or sequence of "mlp", "attention"
        Which components to apply gradient checkpointing to
    """

    def __init__(
        self,
        mv_channels: int,
        s_channels: int,
        attention: SelfAttentionConfig,
        mlp: MLPConfig,
        dropout_prob: Optional[float] = None,
        checkpoint: Optional[Sequence[Literal["mlp", "attention"]]] = None,
    ) -> None:
        super().__init__()

        # Gradient checkpointing settings
        if checkpoint is not None:
            for key in checkpoint:
                assert key in ["mlp", "attention"]
        self._checkpoint_mlp = checkpoint is not None and "mlp" in checkpoint
        self._checkpoint_attn = checkpoint is not None and "attention" in checkpoint

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
        reference_mv: Optional[torch.Tensor] = None,
        additional_qk_features_mv: Optional[torch.Tensor] = None,
        additional_qk_features_s: Optional[torch.Tensor] = None,
        attention_mask: Optional[Union[AttentionBias, torch.Tensor]] = None,
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

        # Attention block
        attn_kwargs = dict(
            multivectors=multivectors,
            scalars=scalars,
            additional_qk_features_mv=additional_qk_features_mv,
            additional_qk_features_s=additional_qk_features_s,
            attention_mask=attention_mask,
        )
        if self._checkpoint_attn:
            h_mv, h_s = checkpoint_(self._attention_block, use_reentrant=False, **attn_kwargs)
        else:
            h_mv, h_s = self._attention_block(**attn_kwargs)

        # Skip connection
        outputs_mv = multivectors + h_mv
        outputs_s = scalars + h_s

        # MLP block
        mlp_kwargs = dict(multivectors=outputs_mv, scalars=outputs_s, reference_mv=reference_mv)
        if self._checkpoint_mlp:
            h_mv, h_s = checkpoint_(self._mlp_block, use_reentrant=False, **mlp_kwargs)
        else:
            h_mv, h_s = self._mlp_block(outputs_mv, scalars=outputs_s, reference_mv=reference_mv)

        # Skip connection
        outputs_mv = outputs_mv + h_mv
        outputs_s = outputs_s + h_s

        return outputs_mv, outputs_s

    def _attention_block(
        self,
        multivectors: torch.Tensor,
        scalars: torch.Tensor,
        additional_qk_features_mv: Optional[torch.Tensor] = None,
        additional_qk_features_s: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Attention block."""

        h_mv, h_s = self.norm(multivectors, scalars=scalars)
        h_mv, h_s = self.attention(
            h_mv,
            scalars=h_s,
            additional_qk_features_mv=additional_qk_features_mv,
            additional_qk_features_s=additional_qk_features_s,
            attention_mask=attention_mask,
        )
        return h_mv, h_s

    def _mlp_block(
        self,
        multivectors: torch.Tensor,
        scalars: torch.Tensor,
        reference_mv: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """MLP block."""

        h_mv, h_s = self.norm(multivectors, scalars=scalars)
        h_mv, h_s = self.mlp(h_mv, scalars=h_s, reference_mv=reference_mv)
        return h_mv, h_s
