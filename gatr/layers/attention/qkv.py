# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
import torch
from einops import rearrange
from torch import nn

from gatr.layers.attention.config import SelfAttentionConfig
from gatr.layers.linear import EquiLinear


class QKVModule(nn.Module):
    """Compute (multivector and scalar) queries, keys, and values via multi-head attention.

    Parameters
    ----------
    config: SelfAttentionConfig
        Attention configuration
    """

    def __init__(self, config: SelfAttentionConfig):
        super().__init__()
        self.in_linear = EquiLinear(
            in_mv_channels=config.in_mv_channels + config.additional_qk_mv_channels,
            out_mv_channels=3 * config.hidden_mv_channels * config.num_heads,
            in_s_channels=config.in_s_channels + config.additional_qk_s_channels,
            out_s_channels=None
            if config.in_s_channels is None
            else 3 * config.hidden_s_channels * config.num_heads,
        )
        self.config = config

    def forward(
        self, inputs, scalars, additional_qk_features_mv=None, additional_qk_features_s=None
    ):
        """Forward pass.

        Parameters
        ----------
        inputs : torch.Tensor
            Multivector inputs
        scalars : torch.Tensor
            Scalar inputs
        additional_qk_features_mv : None or torch.Tensor
            Additional multivector features that should be provided for the Q/K computation (e.g.
            positions of objects)
        additional_qk_features_s : None or torch.Tensor
            Additional scalar features that should be provided for the Q/K computation (e.g.
            object types)

        Returns
        -------
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
        """

        # Additional inputs
        if additional_qk_features_mv is not None:
            inputs = torch.cat((inputs, additional_qk_features_mv), dim=-2)
        if additional_qk_features_s is not None:
            scalars = torch.cat((scalars, additional_qk_features_s), dim=-1)

        qkv_mv, qkv_s = self.in_linear(
            inputs, scalars
        )  # (..., num_items, 3 * hidden_channels * num_heads, 16)
        qkv_mv = rearrange(
            qkv_mv,
            "... items (qkv hidden num_heads) x -> qkv ... num_heads items hidden x",
            num_heads=self.config.num_heads,
            hidden=self.config.hidden_mv_channels,
            qkv=3,
        )
        q_mv, k_mv, v_mv = qkv_mv  # each: (..., num_heads, num_items, num_channels, 16)

        # Same, for optional scalar components
        if qkv_s is not None:
            qkv_s = rearrange(
                qkv_s,
                "... items (qkv hidden num_heads) -> qkv ... num_heads items hidden",
                num_heads=self.config.num_heads,
                hidden=self.config.hidden_s_channels,
                qkv=3,
            )
            q_s, k_s, v_s = qkv_s  # each: (..., num_heads, num_items, num_channels)
        else:
            q_s, k_s, v_s = None, None, None

        return q_mv, k_mv, v_mv, q_s, k_s, v_s


class MultiQueryQKVModule(nn.Module):
    """Compute (multivector and scalar) queries, keys, and values via multi-query attention.

    Parameters
    ----------
    config: SelfAttentionConfig
        Attention configuration
    """

    def __init__(self, config: SelfAttentionConfig):
        super().__init__()

        # Q projection
        self.q_linear = EquiLinear(
            in_mv_channels=config.in_mv_channels + config.additional_qk_mv_channels,
            out_mv_channels=config.hidden_mv_channels * config.num_heads,
            in_s_channels=config.in_s_channels + config.additional_qk_s_channels,
            out_s_channels=config.hidden_s_channels * config.num_heads,
        )

        # Key and value projections (shared between heads)
        self.k_linear = EquiLinear(
            in_mv_channels=config.in_mv_channels + config.additional_qk_mv_channels,
            out_mv_channels=config.hidden_mv_channels,
            in_s_channels=config.in_s_channels + config.additional_qk_s_channels,
            out_s_channels=config.hidden_s_channels,
        )
        self.v_linear = EquiLinear(
            in_mv_channels=config.in_mv_channels,
            out_mv_channels=config.hidden_mv_channels,
            in_s_channels=config.in_s_channels,
            out_s_channels=config.hidden_s_channels,
        )
        self.config = config

    def forward(
        self, inputs, scalars, additional_qk_features_mv=None, additional_qk_features_s=None
    ):
        """Forward pass.

        Parameters
        ----------
        inputs : torch.Tensor
            Multivector inputs
        scalars : torch.Tensor
            Scalar inputs
        additional_qk_features_mv : None or torch.Tensor
            Additional multivector features that should be provided for the Q/K computation (e.g.
            positions of objects)
        additional_qk_features_s : None or torch.Tensor
            Additional scalar features that should be provided for the Q/K computation (e.g.
            object types)

        Returns
        -------
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
        """

        # Additional inputs
        if additional_qk_features_mv is not None:
            qk_inputs = torch.cat((inputs, additional_qk_features_mv), dim=-2)
        else:
            qk_inputs = inputs
        if scalars is not None and additional_qk_features_s is not None:
            qk_scalars = torch.cat((scalars, additional_qk_features_s), dim=-1)
        else:
            qk_scalars = scalars

        # Project to queries, keys, and values (multivector reps)
        q_mv, q_s = self.q_linear(
            qk_inputs, qk_scalars
        )  # (..., num_items, hidden_channels * num_heads, 16)
        k_mv, k_s = self.k_linear(qk_inputs, qk_scalars)  # (..., num_items, hidden_channels, 16)
        v_mv, v_s = self.v_linear(inputs, scalars)  # (..., num_items, hidden_channels, 16)

        # Rearrange to (..., heads, items, channels, 16) shape
        q_mv = rearrange(
            q_mv,
            "... items (hidden_channels num_heads) x -> ... num_heads items hidden_channels x",
            num_heads=self.config.num_heads,
            hidden_channels=self.config.hidden_mv_channels,
        )
        k_mv = rearrange(k_mv, "... items hidden_channels x -> ... 1 items hidden_channels x")
        v_mv = rearrange(v_mv, "... items hidden_channels x -> ... 1 items hidden_channels x")

        # Same for scalars
        if q_s is not None:
            q_s = rearrange(
                q_s,
                "... items (hidden_channels num_heads) -> ... num_heads items hidden_channels",
                num_heads=self.config.num_heads,
                hidden_channels=self.config.hidden_s_channels,
            )
            k_s = rearrange(k_s, "... items hidden_channels -> ... 1 items hidden_channels")
            v_s = rearrange(v_s, "... items hidden_channels -> ... 1 items hidden_channels")
        else:
            q_s, k_s, v_s = None, None, None

        return q_mv, k_mv, v_mv, q_s, k_s, v_s
