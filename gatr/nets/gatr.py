# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All rights reserved.
"""Equivariant transformer for multivector data."""

from dataclasses import replace
from typing import Literal, Optional, Sequence, Tuple, Union
from warnings import warn

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint as checkpoint_

from gatr.layers.attention.config import SelfAttentionConfig
from gatr.layers.gatr_block import GATrBlock
from gatr.layers.linear import EquiLinear
from gatr.layers.mlp.config import MLPConfig
from gatr.utils.tensors import construct_reference_multivector


class GATr(nn.Module):
    """GATr network for a data with a single token dimension.

    This, together with gatr.nets.axial_gatr.AxialGATr, is the main architecture proposed in our
    paper.

    It combines `num_blocks` GATr transformer blocks, each consisting of geometric self-attention
    layers, a geometric MLP, residual connections, and normalization layers. In addition, there
    are initial and final equivariant linear layers.

    Assumes input has shape `(..., items, in_channels, 16)`, output has shape
    `(..., items, out_channels, 16)`, will create hidden representations with shape
    `(..., items, hidden_channels, 16)`.

    Parameters
    ----------
    in_mv_channels : int
        Number of input multivector channels.
    out_mv_channels : int
        Number of output multivector channels.
    hidden_mv_channels : int
        Number of hidden multivector channels.
    in_s_channels : None or int
        If not None, sets the number of scalar input channels.
    out_s_channels : None or int
        If not None, sets the number of scalar output channels.
    hidden_s_channels : None or int
        If not None, sets the number of scalar hidden channels.
    attention: Dict
        Data for SelfAttentionConfig
    mlp: Dict
        Data for MLPConfig
    num_blocks : int
        Number of transformer blocks.
    checkpoint_blocks : bool
        Deprecated option to specify gradient checkpointing. Use `checkpoint=["block"]` instead
    dropout_prob : float or None
        Dropout probability
    checkpoint : None or sequence of "mlp", "attention", "block"
        Which components to apply gradient checkpointing to
    """

    def __init__(
        self,
        in_mv_channels: int,
        out_mv_channels: int,
        hidden_mv_channels: int,
        in_s_channels: Optional[int],
        out_s_channels: Optional[int],
        hidden_s_channels: Optional[int],
        attention: SelfAttentionConfig,
        mlp: MLPConfig,
        num_blocks: int = 10,
        reinsert_mv_channels: Optional[Tuple[int]] = None,
        reinsert_s_channels: Optional[Tuple[int]] = None,
        checkpoint_blocks: bool = False,
        dropout_prob: Optional[float] = None,
        checkpoint: Union[
            None, Sequence[Literal["block"]], Sequence[Literal["mlp", "attention"]]
        ] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        # Gradient checkpointing settings
        if checkpoint_blocks:
            # The checkpoint_blocks keyword was deprecated in v1.4.0.
            if checkpoint is not None:
                raise ValueError(
                    "Both checkpoint_blocks and checkpoint were specified. Please only use"
                    "checkpoint."
                )
            warn(
                'The checkpoint_blocks keyword is deprecated since v1.4.0. Use checkpoint=["block"]'
                "instead.",
                category=DeprecationWarning,
            )
            checkpoint = ["block"]
        if checkpoint is not None:
            for key in checkpoint:
                assert key in ["block", "mlp", "attention"]
        if checkpoint is not None and "block" in checkpoint:
            self._checkpoint_blocks = True
            if "mlp" in checkpoint or "attention" in checkpoint:
                raise ValueError(
                    "Checkpointing both on the block level and the MLP / attention"
                    'level is not sensible. Please use either checkpoint=["block"] or '
                    f'checkpoint=["attention", "mlp"]. Found checkpoint={checkpoint}.'
                )
            checkpoint = None
        else:
            self._checkpoint_blocks = False

        self.linear_in = EquiLinear(
            in_mv_channels,
            hidden_mv_channels,
            in_s_channels=in_s_channels,
            out_s_channels=hidden_s_channels,
        )
        attention = replace(
            SelfAttentionConfig.cast(attention),  # convert duck typing to actual class
            additional_qk_mv_channels=(
                0 if reinsert_mv_channels is None else len(reinsert_mv_channels)
            ),
            additional_qk_s_channels=0 if reinsert_s_channels is None else len(reinsert_s_channels),
        )
        mlp = MLPConfig.cast(mlp)
        self.blocks = nn.ModuleList(
            [
                GATrBlock(
                    mv_channels=hidden_mv_channels,
                    s_channels=hidden_s_channels,
                    attention=attention,
                    mlp=mlp,
                    dropout_prob=dropout_prob,
                    checkpoint=checkpoint,
                )
                for _ in range(num_blocks)
            ]
        )
        self.linear_out = EquiLinear(
            hidden_mv_channels,
            out_mv_channels,
            in_s_channels=hidden_s_channels,
            out_s_channels=out_s_channels,
        )
        self._reinsert_s_channels = reinsert_s_channels
        self._reinsert_mv_channels = reinsert_mv_channels

    def forward(
        self,
        multivectors: torch.Tensor,
        scalars: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        join_reference: Union[Tensor, str] = "data",
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """Forward pass of the network.

        Parameters
        ----------
        multivectors : torch.Tensor with shape (..., in_mv_channels, 16)
            Input multivectors.
        scalars : None or torch.Tensor with shape (..., in_s_channels)
            Optional input scalars.
        attention_mask: None or torch.Tensor with shape (..., num_items, num_items)
            Optional attention mask
        join_reference : Tensor with shape (..., 16) or {"data", "canonical"}
            Reference multivector for the equivariant joint operation. If "data", a
            reference multivector is constructed from the mean of the input multivectors. If
            "canonical", a constant canonical reference multivector is used instead.

        Returns
        -------
        outputs_mv : torch.Tensor with shape (..., out_mv_channels, 16)
            Output multivectors.
        outputs_s : None or torch.Tensor with shape (..., out_s_channels)
            Output scalars, if scalars are provided. Otherwise None.
        """

        # Reference multivector and channels that will be re-inserted in any query / key computation
        reference_mv = construct_reference_multivector(join_reference, multivectors)
        additional_qk_features_mv, additional_qk_features_s = self._construct_reinserted_channels(
            multivectors, scalars
        )

        # Pass through the blocks
        h_mv, h_s = self.linear_in(multivectors, scalars=scalars)
        for block in self.blocks:
            if self._checkpoint_blocks:
                h_mv, h_s = checkpoint_(
                    block,
                    h_mv,
                    use_reentrant=False,
                    scalars=h_s,
                    reference_mv=reference_mv,
                    additional_qk_features_mv=additional_qk_features_mv,
                    additional_qk_features_s=additional_qk_features_s,
                    attention_mask=attention_mask,
                )
            else:
                h_mv, h_s = block(
                    h_mv,
                    scalars=h_s,
                    reference_mv=reference_mv,
                    additional_qk_features_mv=additional_qk_features_mv,
                    additional_qk_features_s=additional_qk_features_s,
                    attention_mask=attention_mask,
                )

        outputs_mv, outputs_s = self.linear_out(h_mv, scalars=h_s)

        return outputs_mv, outputs_s

    def _construct_reinserted_channels(self, multivectors, scalars):
        """Constructs input features that will be reinserted in every attention layer."""

        if self._reinsert_mv_channels is None:
            additional_qk_features_mv = None
        else:
            additional_qk_features_mv = multivectors[..., self._reinsert_mv_channels, :]

        if self._reinsert_s_channels is None:
            additional_qk_features_s = None
        else:
            assert scalars is not None
            additional_qk_features_s = scalars[..., self._reinsert_s_channels]

        return additional_qk_features_mv, additional_qk_features_s
