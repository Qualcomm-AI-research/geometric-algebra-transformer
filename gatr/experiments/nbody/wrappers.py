# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
import dgl
import numpy as np
import torch
from e3nn.o3 import Irreps, spherical_harmonics
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from torch_scatter import scatter

from gatr.baselines.gcan import GCAGNN
from gatr.baselines.transformer import BaselineAxialTransformer, BaselineTransformer
from gatr.experiments.base_wrapper import BaseWrapper
from gatr.interface import (
    embed_point,
    embed_scalar,
    embed_translation,
    extract_point,
    extract_point_embedding_reg,
)
from gatr.utils.misc import make_full_edge_index


def embed_nbody_data_in_pga(inputs):
    """Represent the n-body initial state in PGA multivectors.

    Masses are represented as scalars, positions as trivectors, and velocities as bivectors
    (like translations).  All three are summed (this is equivalent to concatenation, as an equi
    linear layer can easily separate the grades again).

    This function is used both by the GATr and by the GCAN wrappers.

    Parameters
    ----------
    inputs : torch.Tensor with shape (batchsize, objects, 7)
        n-body initial state: a concatenation of masses, initial positions, and initial
        velocities along the feature dimension.

    Returns
    -------
    multivector : torch.Tensor with shape (batchsize, objects, 1, 16)
        GA embedding.
    """

    # Build one multivector holding masses, points, and velocities for each object
    masses = inputs[:, :, [0]]  # (batchsize, objects, 1)
    masses = embed_scalar(masses)  # (batchsize, objects, 16)
    points = inputs[:, :, 1:4]  # (batchsize, objects, 3)
    points = embed_point(points)  # (batchsize, objects, 16)
    velocities = inputs[:, :, 4:7]  # (batchsize, objects, 3)
    velocities = embed_translation(velocities)  # (batchsize, objects, 16)
    multivector = masses + points + velocities  # (batchsize, objects, 16)

    # Insert channel dimension
    multivector = multivector.unsqueeze(2)  # (batchsize, objects, 1, 16)

    return multivector


class NBodyGATrWrapper(BaseWrapper):
    """Wraps around GATr for the n-body prediction experiment.

    Parameters
    ----------
    net : torch.nn.Module
        GATr model that accepts inputs with 1 multivector channel and 1 scalar channel, and
        returns outputs with 1 multivector channel and 1 scalar channel.
    """

    def __init__(self, net):
        super().__init__(net, scalars=True, return_other=True)
        self.supports_variable_items = True

    def embed_into_ga(self, inputs):
        """Embeds raw inputs into the geometric algebra (+ scalar) representation.

        Parameters
        ----------
        inputs : torch.Tensor with shape (batchsize, objects, 7)
            n-body initial state: a concatenation of masses, initial positions, and initial
            velocities along the feature dimension.

        Returns
        -------
        mv_inputs : torch.Tensor
            Multivector representation of masses, positions, and velocities.
        scalar_inputs : torch.Tensor or None
            Dummy auxiliary scalars, containing no information.
        """
        batchsize, num_objects, _ = inputs.shape

        # Build one multivector holding masses, positions, and velocities for each object
        multivector = embed_nbody_data_in_pga(inputs)

        # Scalar inputs are not really needed here
        scalars = torch.zeros((batchsize, num_objects, 1), device=inputs.device)

        return multivector, scalars

    def extract_from_ga(self, multivector, scalars):
        """Extracts raw outputs from the GATr multivector + scalar outputs.

        We parameterize the predicted final positions as points.

        Parameters
        ----------
        multivector : torch.Tensor
            Multivector outputs from GATr.
        scalars : torch.Tensor or None
            Scalar outputs from GATr.

        Returns
        -------
        outputs : torch.Tensor
            Predicted final-state positions.
        other : torch.Tensor
            Regularization terms.
        """

        # Check channels of inputs. Batchsize and object numbers are free.
        assert multivector.shape[2:] == (1, 16)
        assert scalars.shape[2:] == (1,)

        # Extract position
        points = extract_point(multivector[:, :, 0, :])

        # Extract non-point components and compute regularization
        other = extract_point_embedding_reg(multivector[:, :, 0, :])
        reg = torch.sum(other**2, dim=[1, 2])
        if self.scalars:
            reg = reg + torch.sum(scalars**2, dim=[1, 2])

        return points, reg


class NBodyBaselineWrapper(nn.Module):
    """Wraps around simple baselines (MLP or Transformer) for the n-body prediction experiment.

    Parameters
    ----------
    net : torch.nn.Module
        Model that accepts inputs with 7 channels and returns outputs with 3 channels.
    """

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.supports_variable_items = isinstance(
            net, (BaselineTransformer, BaselineAxialTransformer)
        )

    def forward(self, inputs):
        """Wrapped forward pass.

        Parameters
        ----------
        inputs : torch.Tensor
            Raw inputs, as given by dataset.

        Returns
        -------
        outputs : torch.Tensor
            Raw outputs, as expected in dataset.
        other : torch.Tensor
            Dummy term, since the baselines do not require regularization.
        """
        batchsize = inputs.shape[0]
        return self.net(inputs), torch.zeros(batchsize, device=inputs.device)


class NBodySEGNNWrapper(nn.Module):
    """Wraps around the SEGNN baseline for the n-body prediction experiment.

    Parameters
    ----------
    net : torch.nn.Module
        SEGNN model that accepts inputs with inputs with 2 vector channels and 1 scalar channel,
        and returns outputs with 1 vector channel.
    """

    def __init__(self, net, neighbors, lmax_attr, canonicalize_mode="com"):
        super().__init__()

        self.net = net

        self.canonicalize_mode = canonicalize_mode
        self.neighbors = neighbors
        self.transform_attr_irreps = Irreps.spherical_harmonics(lmax_attr)
        self.supports_variable_items = True

    def forward(self, inputs):
        """Wrapped forward pass.

        Parameters
        ----------
        inputs : torch.Tensor
            Raw inputs, as given by dataset.

        Returns
        -------
        outputs : torch.Tensor
            Raw outputs, as expected in dataset.
        other : torch.Tensor
            Dummy term, since the baselines do not require regularization.

        Raises
        ------
        ValueError
            If `self.canonicalize_mode` is invalid.
        """
        batchsize, num_objects, _ = inputs.shape

        # Separate into scalars and vectors
        masses = inputs[:, :, [0]]  # (batchsize, objects, 1)
        locations = inputs[:, :, 1:4]  # (batchsize, objects, 3)
        velocities = inputs[:, :, 4:7]  # (batchsize, objects, 3)

        # Canonicalize
        if self.canonicalize_mode == "com":
            weights = masses
        elif self.canonicalize_mode == "heaviest":
            weights = torch.exp(2.0 * masses.double())
        elif self.canonicalize_mode == "even":
            weights = torch.ones_like(masses)
        else:
            raise ValueError(f"Unknown canonicalization mode {self.canonicalize_mode}")

        com = torch.sum(
            weights / torch.sum(weights, dim=-2, keepdim=True) * locations.double(),
            dim=-2,
            keepdim=True,
        ).float()
        locations = locations - com

        # Represent as graph
        graph = Data(pos=locations.view(-1, 3), vel=velocities.view(-1, 3), mass=masses.view(-1, 1))
        batch = torch.arange(0, batchsize, device=inputs.device)
        graph.batch = batch.repeat_interleave(num_objects).to(inputs.device, torch.long)
        graph.edge_index = knn_graph(locations.view(-1, 3), self.neighbors, graph.batch)

        graph = self._augment_gravity_graph(graph)  # Add O3 attributes

        # Push through model
        pred_shift = self.net(graph)
        pred_shift = pred_shift.view(batchsize, num_objects, 3)
        predictions = (
            locations + pred_shift
        )  # The model predicts the shift, not the final positions

        # Undo canonicalization
        predictions = predictions + com

        return predictions, torch.zeros(batchsize, device=inputs.device)

    def _augment_gravity_graph(self, graph):
        """SEGNN feature engineering for n-body experiments.

        Constructs node features (position relative to mean position, velocity embedding, absolute
        velocity) and edge features (pairwise distances, product of charges / masses).
        """

        pos = graph.pos
        vel = graph.vel
        mass = graph.mass

        prod_mass = mass[graph.edge_index[0]] * mass[graph.edge_index[1]]
        rel_pos = pos[graph.edge_index[0]] - pos[graph.edge_index[1]]
        edge_dist = torch.sqrt(rel_pos.pow(2).sum(1, keepdims=True))

        graph.edge_attr = spherical_harmonics(
            self.transform_attr_irreps, rel_pos, normalize=True, normalization="integral"
        )
        vel_embedding = spherical_harmonics(
            self.transform_attr_irreps, vel, normalize=True, normalization="integral"
        )
        graph.node_attr = (
            scatter(graph.edge_attr, graph.edge_index[1], dim=0, reduce="mean") + vel_embedding
        )

        vel_abs = torch.sqrt(vel.pow(2).sum(1, keepdims=True))

        graph.x = torch.cat((pos, vel, vel_abs), 1)  # Note that pos is here already canonicalized
        graph.additional_message_features = torch.cat((edge_dist, prod_mass), dim=-1)
        return graph


class NBodySE3TransformerWrapper(nn.Module):
    """Wraps around the SE3-Transformer baseline for the n-body prediction experiment.

    Parameters
    ----------
    net : torch.nn.Module
        SE3-Transformer model.
    """

    def __init__(self, net, canonicalize_to_com=True, canonicalize_mode="com"):
        super().__init__()

        self.net = net
        self.canonicalize_to_com = canonicalize_to_com
        self.canonicalize_mode = canonicalize_mode
        self.supports_variable_items = True

    def forward(self, inputs):
        """Wrapped forward pass.

        Parameters
        ----------
        inputs : torch.Tensor
            Raw inputs, as given by dataset.

        Returns
        -------
        outputs : torch.Tensor
            Raw outputs, as expected in dataset.
        other : torch.Tensor
            Dummy term, since the baselines do not require regularization.

        Raises
        ------
        ValueError
            If `self.canonicalize_mode` is invalid.
        """
        batchsize, num_objects, _ = inputs.shape

        # Separate into scalars and vectors
        masses = inputs[:, :, [0]]  # (batchsize, objects, 1)
        locations = inputs[:, :, 1:4]  # (batchsize, objects, 3)
        velocities = inputs[:, :, 4:7]  # (batchsize, objects, 3)

        # Canonicalize to center-of-mass frame if requested
        if self.canonicalize_to_com:
            if self.canonicalize_mode == "com":
                weights = masses
            elif self.canonicalize_mode == "heaviest":
                weights = torch.exp(2.0 * masses.double())
            else:
                raise ValueError(f"Unknown canonicalization mode {self.canonicalize_mode}")
            com = torch.sum(
                weights / torch.sum(weights, dim=-2, keepdim=True) * locations.double(),
                dim=-2,
                keepdim=True,
            ).float()
            locations = locations - com
        else:
            com = torch.zeros_like(locations)

        # Represent as graph
        graphs = self._build_graphs(locations, velocities, masses)

        # Push through model
        predictions = self.net(graphs)
        predictions = predictions[:, 0, :]  # Only positions, not velocities
        predictions = predictions.view(batchsize, num_objects, 3)
        predictions = (
            locations + predictions
        )  # Model predicts positions relative to initial pos, make it absolute

        # Undo canonicalization
        if self.canonicalize_to_com:
            predictions = predictions + com

        return predictions, torch.zeros(batchsize, device=inputs.device)

    def _build_graphs(self, locations, velocities, masses):
        """Builds graph for a full batch."""
        graphs = [
            self._build_graph(loc, vel, m) for loc, vel, m in zip(locations, velocities, masses)
        ]
        graphs = dgl.batch(graphs)

        return graphs

    def _build_graph(self, locations, velocities, masses):
        """Builds graph for a single sample."""
        n_points = len(locations)
        indices_src, indices_dst = self._fully_connected_idx(n_points)
        graph = dgl.DGLGraph((indices_src, indices_dst)).to(locations.device)
        graph.ndata["x"] = torch.unsqueeze(locations, dim=1)  # [N, 1, 3]
        graph.ndata["v"] = torch.unsqueeze(velocities, dim=1)  # [N, 1, 3]
        graph.ndata["c"] = torch.unsqueeze(masses, dim=1)  # [N, 1, 1]
        graph.edata["d"] = locations[indices_dst] - locations[indices_src]  # relative postions
        graph.edata["w"] = masses[indices_dst] * masses[indices_src]
        return graph

    @staticmethod
    def _fully_connected_idx(num_atoms):
        """Creates source and destination indices for a fully connected graph."""
        src = []
        dst = []
        for i in range(num_atoms):
            for j in range(num_atoms):
                if i != j:
                    src.append(i)
                    dst.append(j)
        return np.array(src), np.array(dst)


class NBodyGCANWrapper(nn.Module):
    """Wraps around GCA-MLP and GCA-GNN baselines for the n-body experiment.

    Parameters
    ----------
    net : torch.nn.Module
        GCAN model that accepts inputs with multivector inputs with 1 channel and
        returns multivector outputs with 1 channel.
    """

    def __init__(self, net, geometric_batching=False):
        super().__init__()
        self.net = net
        self._geometric_batching = geometric_batching
        self.supports_variable_items = isinstance(net, GCAGNN)

    def forward(self, inputs: torch.Tensor):
        """Wrapped forward pass.

        Parameters
        ----------
        inputs : torch.Tensor
            Raw inputs, as given by dataset.

        Returns
        -------
        outputs : torch.Tensor
            Raw outputs, as expected in dataset.
        other : torch.Tensor
            Dummy term, since the baselines do not require regularization.
        """

        multivector = embed_nbody_data_in_pga(inputs)

        if self._geometric_batching:
            # PyG-style batching, with a single batch-token dimension and edge list
            edge_index = self.make_edge_index(inputs)
            multivector = multivector.view(-1, *multivector.shape[2:])
            multivector_outputs = self.net(multivector, edge_index=edge_index)
            multivector_outputs = multivector_outputs.view(
                inputs.shape[0], -1, *multivector_outputs.shape[1:]
            )
        else:
            # Standard batching, with (batch, token, ...) tensors
            multivector_outputs = self.net(multivector)

        outputs, reg = self.extract_from_ga(multivector_outputs)
        return outputs, reg

    @staticmethod
    def extract_from_ga(multivector):
        """Extracts predicted positions from PGA multivectors."""
        # Check channels of inputs. Batchsize and object numbers are free.
        assert multivector.shape[2:] == (1, 16)

        # Extract position
        points = extract_point(multivector[:, :, 0, :])

        # Extract non-point components and compute regularization
        other = extract_point_embedding_reg(multivector[:, :, 0, :])
        reg = torch.sum(other**2, dim=[1, 2])
        return points, reg

    @staticmethod
    def make_edge_index(inputs):
        """Constructs an edge index for fully connected graph."""
        batchsize, num_items, _ = inputs.shape
        return make_full_edge_index(
            num_items, batchsize=batchsize, self_loops=False, device=inputs.device
        )

    def to(self, *args, **kwargs):
        """Send to device.

        Overwritten to also move the CliffordAlgebra object, which is not an nn.Module.
        """
        resursively_move_gcan_pga(self, *args, **kwargs)
        super().to(*args, **kwargs)
        return self


def resursively_move_gcan_pga(net, *args, **kwargs):
    """Moves all PGA instances in GCAN component to device / dtype."""
    try:
        net.pga.to(*args, **kwargs)
    except AttributeError:
        pass

    for child in net.children():
        resursively_move_gcan_pga(child, *args, **kwargs)
