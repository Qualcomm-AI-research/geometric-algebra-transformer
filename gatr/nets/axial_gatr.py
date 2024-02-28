# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
from dataclasses import replace
from typing import Optional, Tuple, Union

import torch
from einops import rearrange
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from gatr.layers.attention.config import SelfAttentionConfig
from gatr.layers.gatr_block import GATrBlock
from gatr.layers.linear import EquiLinear
from gatr.layers.mlp.config import MLPConfig
from gatr.utils.tensors import construct_reference_multivector

# Default rearrange patterns for odd blocks
_MV_REARRANGE_PATTERN = "... i j c x -> ... j i c x"
_S_REARRANGE_PATTERN = "... i j c -> ... j i c"


class AxialGATr(nn.Module):  # pylint: disable=duplicate-code
    """Axial GATr network for two token dimensions.

    This, together with gatr.nets.gatr.GATr, is the main architecture proposed in our paper.

    It combines `num_blocks` GATr transformer blocks, each consisting of geometric self-attention
    layers, a geometric MLP, residual connections, and normalization layers. In addition, there
    are initial and final equivariant linear layers.

    Assumes input data with shape `(..., num_items_1, num_items_2, num_channels, 16)`.

    The first, third, fifth, ... block computes attention over the `items_2` axis. The other blocks
    compute attention over the `items_1` axis. Positional encoding can be specified separately for
    both axes.

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
    pos_encodings : tuple of bool
        Whether to apply rotary positional embeddings along the item dimensions to the scalar keys
        and queries. The first element in the tuple determines whether positional embeddings
        are applied to the first item dimension, the second element the same for the second item
        dimension.
    collapse_dims : tuple of bool
        Whether batch and token dimensions will be collapsed to support xformers block attention.
        The first element of this tuple describes the behaviour in the 0th, 2nd, ... block, the
        second element in the 1st, 3rd, ... block (where the axes are reversed).
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
        num_blocks: int = 20,
        checkpoint_blocks: bool = False,
        pos_encodings: Tuple[bool, bool] = (False, False),
        collapse_dims: Tuple[bool, bool] = (False, False),
        **kwargs,
    ) -> None:
        super().__init__()
        self.linear_in = EquiLinear(
            in_mv_channels,
            hidden_mv_channels,
            in_s_channels=in_s_channels,
            out_s_channels=hidden_s_channels,
        )
        attention = SelfAttentionConfig.cast(attention)
        mlp = MLPConfig.cast(mlp)
        self.blocks = nn.ModuleList(
            [
                GATrBlock(
                    mv_channels=hidden_mv_channels,
                    s_channels=hidden_s_channels,
                    attention=replace(
                        attention,
                        pos_encoding=pos_encodings[(block + 1) % 2],
                    ),
                    mlp=mlp,
                )
                for block in range(num_blocks)
            ]
        )
        self.linear_out = EquiLinear(
            hidden_mv_channels,
            out_mv_channels,
            in_s_channels=hidden_s_channels,
            out_s_channels=out_s_channels,
        )
        self._checkpoint_blocks = checkpoint_blocks
        self._collapse_dims_for_even_blocks, self._collapse_dims_for_odd_blocks = collapse_dims

    def forward(
        self,
        multivectors: Tensor,
        scalars: Optional[Tensor] = None,
        attention_mask: Optional[Tuple] = None,
        join_reference: Union[Tensor, str] = "data",
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward pass of the network.

        Parameters
        ----------
        multivectors : Tensor with shape (..., num_items_1, num_items_2, in_mv_channels, 16)
            Input multivectors.
        scalars : None or Tensor with shape (..., num_items_1, num_items_2, in_s_channels)
            Optional input scalars.
        attention_mask : None or tuple of Tensor
            Optional attention masks.
        join_reference : Tensor with shape (..., 16) or {"data", "canonical"}
            Reference multivector for the equivariant joint operation. If "data", a
            reference multivector is constructed from the mean of the input multivectors. If
            "canonical", a constant canonical reference multivector is used instead.

        Returns
        -------
        outputs_mv : Tensor with shape (..., num_items_1, num_items_2, out_mv_channels, 16)
            Output multivectors.
        outputs_s : None or Tensor with shape
            (..., num_items_1, num_items_2, out_mv_channels, 16)
            Output scalars, if scalars are provided. Otherwise None.
        """

        # Reference MV for equivariant join
        reference_mv = construct_reference_multivector(join_reference, multivectors)

        # Pass through the blocks
        h_mv, h_s = self.linear_in(multivectors, scalars=scalars)
        for i, block in enumerate(self.blocks):
            # For first, third, ... block, we want to perform attention over the first token
            # dimension. We implement this by transposing the two item dimensions.
            if i % 2 == 1:
                h_mv, h_s, input_batch_dims = self._reshape_data_before_odd_blocks(h_mv, h_s)
            else:
                h_mv, h_s, input_batch_dims = self._reshape_data_before_even_blocks(h_mv, h_s)

            # Attention masks will also be different
            if attention_mask is None:
                this_attention_mask = None
            else:
                this_attention_mask = attention_mask[(i + 1) % 2]

            if self._checkpoint_blocks:
                h_mv, h_s = checkpoint(
                    block,
                    h_mv,
                    use_reentrant=False,
                    scalars=h_s,
                    reference_mv=reference_mv,
                    attention_mask=this_attention_mask,
                )
            else:
                h_mv, h_s = block(
                    h_mv,
                    scalars=h_s,
                    reference_mv=reference_mv,
                    attention_mask=this_attention_mask,
                )

            # Transposing back to standard axis order
            if i % 2 == 1:
                h_mv, h_s = self._reshape_data_after_odd_blocks(h_mv, h_s, input_batch_dims)
            else:
                h_mv, h_s = self._reshape_data_after_even_blocks(h_mv, h_s, input_batch_dims)

        outputs_mv, outputs_s = self.linear_out(h_mv, scalars=h_s)

        return outputs_mv, outputs_s

    @staticmethod
    def _construct_join_reference(inputs: Tensor):
        """Constructs a reference vector for dualization from the inputs."""

        # When using torch-geometric-style batching, this code should be adapted to perform the
        # mean over the items in each batch, but not over the batch dimension.
        # We leave this as an exercise for the practitioner :)
        mean_dim = tuple(range(1, len(inputs.shape) - 1))
        return torch.mean(inputs, dim=mean_dim, keepdim=True)  # (batch, 1, ..., 1, 16)

    def _reshape_data_before_odd_blocks(self, multivector, scalar):
        # Transpose token axes
        multivector = rearrange(multivector, _MV_REARRANGE_PATTERN)  # (axis2, axis1, ...)
        scalar = rearrange(scalar, _S_REARRANGE_PATTERN)  # (axis2, axis1, ...)

        # Prepare uncollapsing
        input_batch_dims = multivector.shape[:2]
        assert scalar.shape[:2] == input_batch_dims

        # Collapse dimensions
        if self._collapse_dims_for_odd_blocks:
            multivector = multivector.reshape(-1, *multivector.shape[2:])  # (axis2 * axis1, ...)
            scalar = scalar.reshape(-1, *scalar.shape[2:])  # (axis2 * axis1, ...)

        return multivector, scalar, input_batch_dims

    def _reshape_data_after_odd_blocks(self, multivector, scalar, input_batch_dims):
        # Uncollapse
        if self._collapse_dims_for_odd_blocks:
            multivector = multivector.reshape(
                *input_batch_dims, *multivector.shape[1:]
            )  # (axis2, axis1, ...)
            scalar = scalar.reshape(*input_batch_dims, *scalar.shape[1:])  # (axis2, axis1, ...)

        # Re-transpose token axes
        multivector = rearrange(multivector, _MV_REARRANGE_PATTERN)  # (axis1, axis2, ...)
        scalar = rearrange(scalar, _S_REARRANGE_PATTERN)  # (axis1, axis2, ...)

        return multivector, scalar

    def _reshape_data_before_even_blocks(self, multivector, scalar):
        # Prepare un-collapsing
        input_batch_dims = multivector.shape[:2]
        assert scalar.shape[:2] == input_batch_dims

        # Collapse
        if self._collapse_dims_for_even_blocks:
            multivector = multivector.reshape(-1, *multivector.shape[2:])  # (axis2 * axis1, ...)
            scalar = scalar.reshape(-1, *scalar.shape[2:])  # (axis2 * axis1, ...)

        return multivector, scalar, input_batch_dims

    def _reshape_data_after_even_blocks(self, multivector, scalar, input_batch_dims):
        # Uncollapse
        if self._collapse_dims_for_even_blocks:
            multivector = multivector.reshape(
                *input_batch_dims, *multivector.shape[1:]
            )  # (axis2, axis1, ...)
            scalar = scalar.reshape(*input_batch_dims, *scalar.shape[1:])  # (axis2, axis1, ...)

        return multivector, scalar
