# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
"""Equivariant normalization layers."""

from typing import Tuple

import torch
from torch import nn

from gatr.primitives import equi_layer_norm


class EquiLayerNorm(nn.Module):
    """Equivariant LayerNorm for multivectors.

    Rescales input such that `mean_channels |inputs|^2 = 1`, where the norm is the GA norm and the
    mean goes over the channel dimensions.

    In addition, the layer performs a regular LayerNorm operation on auxiliary scalar inputs.

    Parameters
    ----------
    mv_channel_dim : int
        Channel dimension index for multivector inputs. Defaults to the second-last entry (last are
        the multivector components).
    scalar_channel_dim : int
        Channel dimension index for scalar inputs. Defaults to the last entry.
    epsilon : float
        Small numerical factor to avoid instabilities. We use a reasonably large number to balance
        issues that arise from some multivector components not contributing to the norm.
    """

    def __init__(self, mv_channel_dim=-2, scalar_channel_dim=-1, epsilon: float = 0.01):
        super().__init__()
        self.mv_channel_dim = mv_channel_dim
        self.epsilon = epsilon

        if scalar_channel_dim != -1:
            raise NotImplementedError(
                "Currently, only scalar_channel_dim = -1 is implemented, but found"
                f" {scalar_channel_dim}"
            )

    def forward(
        self, multivectors: torch.Tensor, scalars: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass. Computes equivariant LayerNorm for multivectors.

        Parameters
        ----------
        multivectors : torch.Tensor with shape (..., 16)
            Multivector inputs
        scalars : torch.Tensor with shape (..., self.in_channels, self.in_scalars)
            Scalar inputs

        Returns
        -------
        outputs_mv : torch.Tensor with shape (..., 16)
            Normalized multivectors
        output_scalars : torch.Tensor with shape (..., self.out_channels, self.in_scalars)
            Normalized scalars.
        """

        outputs_mv = equi_layer_norm(
            multivectors, channel_dim=self.mv_channel_dim, epsilon=self.epsilon
        )
        normalized_shape = scalars.shape[-1:]
        outputs_s = torch.nn.functional.layer_norm(scalars, normalized_shape=normalized_shape)

        return outputs_mv, outputs_s
