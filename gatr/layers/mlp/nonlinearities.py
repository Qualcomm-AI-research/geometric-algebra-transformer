# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
from typing import Tuple

import torch
from torch import nn

from gatr.primitives.nonlinearities import gated_gelu, gated_relu, gated_sigmoid


class ScalarGatedNonlinearity(nn.Module):
    """Gated nonlinearity, where the gate is simply given by the scalar component of the input.

    Given multivector input x, computes f(x_0) * x, where f can either be ReLU, sigmoid, or GeLU.

    Auxiliary scalar inputs are simply processed with ReLU, sigmoid, or GeLU, without gating.

    Parameters
    ----------
    nonlinearity : {"relu", "sigmoid", "gelu"}
        Non-linearity type
    """

    def __init__(self, nonlinearity: str = "relu", **kwargs) -> None:
        super().__init__()

        gated_fn_dict = dict(relu=gated_relu, gelu=gated_gelu, sigmoid=gated_sigmoid)
        scalar_fn_dict = dict(
            relu=nn.functional.relu, gelu=nn.functional.gelu, sigmoid=nn.functional.sigmoid
        )
        try:
            self.gated_nonlinearity = gated_fn_dict[nonlinearity]
            self.scalar_nonlinearity = scalar_fn_dict[nonlinearity]
        except KeyError as exc:
            raise ValueError(
                f"Unknown nonlinearity {nonlinearity} for options {list(gated_fn_dict.keys())}"
            ) from exc

    def forward(
        self, multivectors: torch.Tensor, scalars: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes f(x_0) * x for multivector x, where f is GELU, ReLU, or sigmoid.

        f is chosen depending on self.nonlinearity.

        Parameters
        ----------
        multivectors : torch.Tensor with shape (..., self.in_channels, 16)
            Input multivectors
        scalars : None or torch.Tensor with shape (..., self.in_channels, self.in_scalars)
            Input scalars

        Returns
        -------
        outputs_mv : torch.Tensor with shape (..., self.out_channels, 16)
            Output multivectors
        output_scalars : torch.Tensor with shape (..., self.out_channels, self.in_scalars)
            Output scalars
        """

        gates = multivectors[..., [0]]
        outputs_mv = self.gated_nonlinearity(multivectors, gates=gates)
        outputs_s = self.scalar_nonlinearity(scalars)

        return outputs_mv, outputs_s
