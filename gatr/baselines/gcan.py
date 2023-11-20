# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.

"""Adapted from code by David Ruhe.

MIT License

Copyright (c) Microsoft Corporation.

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

import numpy as np
import torch
from cliffordlayers.cliffordalgebra import CliffordAlgebra
from cliffordlayers.nn.modules.gcan import MultiVectorAct, PGAConjugateLinear
from torch import nn
from torch_geometric.nn import MessagePassing


class GCAMLP(nn.Module):
    """GCA-MLP model.

    As described in D. Ruhe et al, "Geometric Clifford Algebra Networks".
    Based on layers from the official
    [cliffordlayers code](https://github.com/microsoft/cliffordlayers/tree/main).

    This network uses the same projective geometric algebra representations as GATr, but is not
    E(3)-equivariant.

    References
    ----------
    D. Ruhe et al, "Geometric Clifford Algebra Networks", arXiv:2302.06594

    Parameters
    ----------
    in_shape : tuple of int
        Shape expected of input tensors, excluding the 16 components of the multivectors.
    out_shape : tuple of int
        Shape expected for output tensors, excluding the 16 components of the multivectors.
    hidden_channels : int
        Number of hidden channels.
    hidden_layers : int
        Number of hidden layers.
    act_agg : {"linear", "sum", "mean"}
        Aggregation function.
    flatten : bool
        If True, the network flattens the input along all dimensions except the first (batch)
        and last (multivector component) dimension. Otherwise, all dimensions except for the last
        two are treated as batch dimensions.
    """

    def __init__(
        self,
        in_shape,
        out_shape,
        hidden_channels,
        hidden_layers,
        act_agg="linear",
        flatten=True,
        **kwargs,
    ):
        super().__init__()

        if not hidden_layers > 0:
            raise NotImplementedError("Only supports > 0 hidden layers")

        in_channels = np.product(in_shape) if flatten else in_shape[-1]
        out_channels = np.product(out_shape) if flatten else out_shape[-1]
        self.flatten = flatten
        self.in_shape = in_shape
        self.out_shape = out_shape

        self.pga = CliffordAlgebra((0, 1, 1, 1))
        self.model = self._make_net(
            in_channels, out_channels, hidden_channels, hidden_layers, act_agg
        )

    def _make_net(self, in_channels, out_channels, hidden_channels, hidden_layers, act_agg):
        """PGA-MLP code as sent by David Ruhe."""

        all_blades = list(range(16))
        action_blades = (0, 5, 6, 7, 8, 9, 10, 15)  # From David; using all blades results in NaNs

        linear_in = PGAConjugateLinear(
            in_channels,
            hidden_channels,
            self.pga,
            input_blades=all_blades,
            action_blades=action_blades,
        )
        linear_out = nn.Sequential(
            MultiVectorAct(
                hidden_channels,
                self.pga,
                input_blades=all_blades,
                agg=act_agg,
            ),
            PGAConjugateLinear(
                hidden_channels,
                out_channels,
                self.pga,
                input_blades=all_blades,
                action_blades=action_blades,
            ),
        )
        mlp = [
            nn.Sequential(
                MultiVectorAct(
                    hidden_channels,
                    self.pga,
                    input_blades=all_blades,
                    agg=act_agg,
                ),
                PGAConjugateLinear(
                    hidden_channels,
                    hidden_channels,
                    self.pga,
                    input_blades=all_blades,
                    action_blades=action_blades,
                ),
            )
            for _ in range(hidden_layers - 1)
        ]
        return nn.Sequential(linear_in, *mlp, linear_out)

    def forward(self, inputs: torch.Tensor):
        """Forward pass.

        Parameters
        ----------
        inputs : torch.Tensor with shape (..., 16)
            Input multivectors.

        Returns
        -------
        outputs : torch.Tensor with shape (..., 16)
            Output multivectors.
        """
        input_shape = inputs.shape
        if self.flatten:
            # Flatten all non-batch dimensions
            inputs = inputs.reshape(input_shape[0], -1, 16)
        outputs = self.model(inputs)
        if self.flatten:
            outputs = outputs.reshape(input_shape[0], *self.out_shape, 16)
        return outputs


class GCAGNNLayer(MessagePassing):  # pylint: disable=abstract-method
    """GCA-GNN layer as described in D. Ruhe et al.

    This was described in "Geometric Clifford Algebra Networks" by D. Ruhe et. al,
    and in private communication from D. Ruhe.
    Uses the GCAN layers in the official cliffordlayers library, pieces them together based on the
    information in the paper. Follows the PyG message-passing schema.

    References
    ----------
    D. Ruhe et al, "Geometric Clifford Algebra Networks", arXiv:2302.06594

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    message_channels : int
        Number of channels in the messages.
    mlp_hidden_channels : int
        Number of hidden channels in the MLPs.
    mlp_hidden_layers : int
        Number of hidden layers in the MLPs.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        message_channels,
        mlp_hidden_channels,
        mlp_hidden_layers,
    ):
        super().__init__(aggr="add", flow="source_to_target", node_dim=-3)

        # GCA-MLPs for input, output, message, and update functions
        shared_kwargs = dict(
            hidden_channels=mlp_hidden_channels, hidden_layers=mlp_hidden_layers, flatten=False
        )
        self.message_mlp = GCAMLP((2 * in_channels,), (message_channels,), **shared_kwargs)
        self.update_mlp = GCAMLP(
            (in_channels + message_channels,), (out_channels,), **shared_kwargs
        )

    def forward(self, x, edge_index):  # pylint: disable=arguments-differ
        """Forward pass."""
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):  # pylint: disable=arguments-differ
        """Constructs message."""
        features = torch.cat([x_i, x_j], dim=1)  # shape (edges, 2*node_channels, 16)
        message = self.message_mlp(features)  # shape (edges, message_channels, 16)
        return message

    def update(self, aggr_out, x):  # pylint: disable=arguments-differ
        """Updates node states with aggregated messages."""

        features = torch.cat(
            [x, aggr_out], dim=1
        )  # shape (nodes, node_channels + message_channels, 16)
        x = self.update_mlp(features)  # shape (nodes, node_channels, 16)
        return x


class GCAGNN(nn.Module):
    """GCA-GNN model as described in D. Ruhe et al.

    The model was described in "Geometric Clifford Algebra Networks" by D.Ruhe et al.,
    and in private communication from D. Ruhe.

    Combines multiple GCAGNNLayers.

    This network uses the same projective geometric algebra representations as GATr, but is not
    E(3)-equivariant.

    References
    ----------
    D. Ruhe et al, "Geometric Clifford Algebra Networks", arXiv:2302.06594

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    node_channels : int
        Number of channels in the hidden representation for each node.
    message_channels : int
        Number of channels in the messages.
    mlp_hidden_channels : int
        Number of hidden channels in the MLPs.
    mlp_hidden_layers : int
        Number of hidden layers in the MLPs.
    message_passing_steps : int
        Number of message-passing steps / GCAGNNLayer blocks.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        node_channels,
        message_channels,
        mlp_hidden_channels,
        mlp_hidden_layers,
        message_passing_steps,
        **kwargs,
    ):
        super().__init__()

        # Construct layers
        self.layers = nn.ModuleList([])
        shared_kwargs = dict(
            mlp_hidden_channels=mlp_hidden_channels,
            mlp_hidden_layers=mlp_hidden_layers,
            message_channels=message_channels,
        )

        # Initial step: in_channels to node_channels
        self.layers.append(GCAGNNLayer(in_channels, node_channels, **shared_kwargs))

        # Intermediate steps / layers
        for _ in range(message_passing_steps - 2):
            self.layers.append(GCAGNNLayer(node_channels, node_channels, **shared_kwargs))

        # Final step: node_channels to out_channels
        self.layers.append(GCAGNNLayer(node_channels, out_channels, **shared_kwargs))

    def forward(self, x, edge_index):
        """Forward pass."""
        for layer in self.layers:
            x = layer(x, edge_index=edge_index)
        return x
