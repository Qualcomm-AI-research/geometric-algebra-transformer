# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.

"""Adapted from the below.

https://github.com/RobDHess/Steerable-E3-GNN/blob/1b95898f6f18204b510ae127d7f38cd29f610f4d/nbody/train_nbody.py
by Johannes Brandstetter, Rob Hesselink, Erik Bekkers at
https://github.com/RobDHess/Steerable-E3-GNN

MIT License

Copyright (c) 2021 Johannes Brandstetter, Rob Hesselink, Erik Bekkers

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from e3nn.o3 import Irreps
from segnn.balanced_irreps import BalancedIrreps, WeightBalancedIrreps
from segnn.segnn.segnn import SEGNN
from torch import nn


def _make_segnn_irreps(vec_channels, scalar_channels):
    """Creates an Irreps object for SEGNNModel given a number of O(3) vectors and scalars."""
    assert vec_channels > 0 or scalar_channels > 0
    irreps = []
    if vec_channels > 0:
        irreps.append(f"{vec_channels}x1o")
    if scalar_channels > 0:
        irreps.append(f"{scalar_channels}x0e")
    return Irreps(" + ".join(irreps))


class SEGNNModel(nn.Module):
    """SEGNN baseline.

    We restrict ourselves to vector and scalar inputs, as those were sufficient for our experiments.

    References
    ----------
    J. Brandstetter et al, "Geometric and Physical Quantities Improve E(3) Equivariant Message
    Passing", ICLR 2022
    """

    def __init__(
        self,
        input_vec_channels=2,
        input_s_channels=1,
        output_vec_channels=1,
        output_s_channels=0,
        additional_message_s_channels=2,
        hidden_features=128,
        lmax_h=2,
        lmax_attr=3,
        subspace_type="weightbalanced",
        layers=7,
        norm="instance",
        pool="avg",
        task="node",
        **kwargs,
    ):
        super().__init__()

        input_irreps = _make_segnn_irreps(input_vec_channels, input_s_channels)
        output_irreps = _make_segnn_irreps(output_vec_channels, output_s_channels)
        edge_attr_irreps = Irreps.spherical_harmonics(lmax_attr)
        node_attr_irreps = Irreps.spherical_harmonics(lmax_attr)
        additional_message_irreps = _make_segnn_irreps(0, additional_message_s_channels)

        if subspace_type == "weightbalanced":
            hidden_irreps = WeightBalancedIrreps(
                _make_segnn_irreps(0, hidden_features), node_attr_irreps, sh=True, lmax=lmax_h
            )
        elif subspace_type == "balanced":
            hidden_irreps = BalancedIrreps(lmax_h, hidden_features, True)
        else:
            raise ValueError(f"Unknown subspace type {subspace_type}")

        self.model = SEGNN(
            input_irreps,
            hidden_irreps,
            output_irreps,
            edge_attr_irreps,
            node_attr_irreps,
            num_layers=layers,
            norm=norm,
            pool=pool,
            task=task,
            additional_message_irreps=additional_message_irreps,
        )

    def forward(self, *inputs, **kwargs):
        """Forward pass."""
        return self.model(*inputs, **kwargs)
