# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
import torch

from gatr.primitives.linear import grade_project


def grade_dropout(x: torch.Tensor, p: float, training: bool = True) -> torch.Tensor:
    """Multivector dropout, dropping out grades independently.

    Parameters
    ----------
    x : torch.Tensor with shape (..., 16)
        Input data.
    p : float
        Dropout probability (assumed the same for each grade).
    training : bool
        Switches between train-time and test-time behaviour.

    Returns
    -------
    outputs : torch.Tensor with shape (..., 16)
        Inputs with dropout applied.
    """

    # Project to grades
    x = grade_project(x)

    # Apply standard 1D dropout
    # For whatever reason, that only works with a single batch dimension, so let's reshape a bit
    h = x.view(-1, 5, 16)
    h = torch.nn.functional.dropout1d(h, p=p, training=training, inplace=False)
    h = h.view(x.shape)

    # Combine grades again
    h = torch.sum(h, dim=-2)

    return h
