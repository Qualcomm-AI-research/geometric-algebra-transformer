# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
"""Equivariant dropout layer."""

from typing import Tuple

import torch
from torch import nn

from gatr.primitives import grade_dropout


class GradeDropout(nn.Module):
    """Grade dropout for multivectors (and regular dropout for auxiliary scalars).

    Parameters
    ----------
    p : float
        Dropout probability.
    """

    def __init__(self, p: float = 0.0):
        super().__init__()
        self._dropout_prob = p

    def forward(
        self, multivectors: torch.Tensor, scalars: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass. Applies dropout.

        Parameters
        ----------
        multivectors : torch.Tensor with shape (..., 16)
            Multivector inputs.
        scalars : torch.Tensor
            Scalar inputs.

        Returns
        -------
        outputs_mv : torch.Tensor with shape (..., 16)
            Multivector inputs with dropout applied.
        output_scalars : torch.Tensor
            Scalar inputs with dropout applied.
        """

        out_mv = grade_dropout(multivectors, p=self._dropout_prob, training=self.training)
        out_s = torch.nn.functional.dropout(scalars, p=self._dropout_prob, training=self.training)

        return out_mv, out_s
