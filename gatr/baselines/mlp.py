# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
from typing import List

import numpy as np
import torch
from torch import nn


class MLP(nn.Module):
    """A simple baseline MLP.

    Flattens all dimensions except batch and uses GELU nonlinearities.
    """

    def __init__(self, in_shape, out_shape, hidden_channels, hidden_layers):
        super().__init__()

        if not hidden_layers > 0:
            raise NotImplementedError("Only supports > 0 hidden layers")

        self.in_shape = in_shape
        self.out_shape = out_shape

        layers: List[nn.Module] = [nn.Linear(np.product(in_shape), hidden_channels)]
        for _ in range(hidden_layers - 1):
            layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_channels, hidden_channels))

        layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_channels, np.product(self.out_shape)))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor):
        """Forward pass of baseline MLP."""
        batchsize = inputs.shape[0]
        inputs = inputs.reshape(batchsize, -1)
        outputs = self.mlp(inputs)
        outputs = outputs.reshape(batchsize, *self.out_shape)
        return outputs
