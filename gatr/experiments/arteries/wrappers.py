# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All rights reserved.
import torch
from torch import nn
from xformers.ops.fmha import BlockDiagonalMask

from gatr.experiments.base_wrapper import BaseWrapper
from gatr.interface import embed_oriented_plane, embed_point, embed_scalar, extract_oriented_plane


def build_attention_mask(inputs):
    """Construct attention mask from pytorch geometric batch.

    Parameters
    ----------
    inputs : torch_geometric.data.Batch
        Data batch.

    Returns
    -------
    attention_mask : xformers.ops.fmha.BlockDiagonalMask
        Block-diagonal attention mask: within each sample, each token can attend to each other
        token.
    """
    return BlockDiagonalMask.from_seqlens(torch.bincount(inputs.batch).tolist())


class ArteryBaselineWrapper(nn.Module):
    """Wraps around simple baselines (like a Transformer) for the artery experiment.

    We use the following parameterization of the artery mesh:
    - the node positions of the arterial mesh
    - the mesh normals
    - the geodesic distance to the inlet

    Parameters
    ----------
    net : torch.nn.Module
        Model that accepts inputs with 7 channels and returns outputs with 3 channels.
    """

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, inputs):
        """Wrapped forward pass."""
        mask = build_attention_mask(inputs)
        x = torch.cat([inputs.pos, inputs.norm, inputs.geo[:, None]], 1)[None]
        return self.net(x, attention_mask=mask)


class ArteryGATrWrapper(BaseWrapper):
    """Wraps around GATr for the arterial wall-shear-stress prediction experiment.

    Parameters
    ----------
    net : torch.nn.Module
        GATr model that accepts inputs with 3 multivector channels and 1 scalar channel, and
        returns outputs with 1 multivector channel and 1 scalar channel.
    """

    def __init__(self, net):
        super().__init__(net, scalars=True, return_other=False)

    def build_attention_mask(
        self,
        inputs,
        mv=None,
        s=None,
    ):
        """Construct block-diagonal attention mask."""
        return build_attention_mask(inputs)

    @torch.compiler.disable
    def embed_into_ga(self, inputs):
        """Embeds raw artery data into the geometric algebra.

        We use the following parameterization of the artery mesh:
        - the node positions of the arterial mesh, represented as points
        - the mesh normals, represented as oriented planes
        - the geodesic distance to the inlet, represented as scalars

        Parameters
        ----------
        inputs : torch_geometric.data.Batch
            Raw artery data.

        Returns
        -------
        multivectors : torch.Tensor with shape (batchsize * num_items, num_mv_channels, 16)
            Multivector embeddings of the geometric information in the dataset.
        scalars : torch.Tensor with shape (batchsize * num_items, num_s_channels)
            Auxiliary scalars, which in this dataset are just dummy placeholders.
        """

        # Get interesting features from PyG object
        num_items = len(inputs.pos)
        pos = inputs.pos.reshape(1, num_items, 1, 3)  # (batch, items, channels, 3)
        mesh_normals = inputs.norm.reshape(1, num_items, 1, 3)  # (batch, items, channels, 3)
        inlet_distance = inputs.geo.reshape(1, num_items, 1, 1)  # (batch, items, channels, 1)

        # NaN debugging
        assert torch.all(torch.isfinite(pos))
        assert torch.all(torch.isfinite(mesh_normals))
        assert torch.all(torch.isfinite(inlet_distance))

        # Embed in GA
        positions = embed_point(pos)
        mesh_normals = embed_oriented_plane(mesh_normals, pos)
        inlet_distance = embed_scalar(inlet_distance)

        # NaN debugging
        assert torch.all(torch.isfinite(positions))
        assert torch.all(torch.isfinite(mesh_normals))
        assert torch.all(torch.isfinite(inlet_distance))

        # Concatenate along channel dimension
        mv = torch.concatenate(
            [positions, mesh_normals, inlet_distance], dim=2
        )  # (batch, items, 3, 16)

        # Dummy scalar
        s = torch.zeros(1, num_items, 1, device=mv.device, dtype=mv.dtype)

        return mv, s

    def extract_from_ga(self, multivector, scalars):
        """Extracts the predicted wall shear stress from the output multivectors.

        We parameterize the wall shear stress as translation-invariant vectors.

        Parameters
        ----------
        multivector : torch.Tensor with shape (1, batchsize * num_items, num_mv_channels, 16)
            Multivector outputs.
        scalars : torch.Tensor with shape (1, batchsize * num_items, num_s_channels)
            Scalar outputs.

        Returns
        -------
        wss : torch.Tensor with shape (1, batchsize * num_items, 3)
            Predicted wall shear stress vectors.
        regularization : None
            Regularization term. As there is no regularizer in this experiment, we return None.
        """
        # Check channels of inputs. Object numbers are free.
        assert multivector.shape[0] == scalars.shape[0] == 1
        assert multivector.shape[2:] == (1, 16)

        # Extract wall shear stress
        wss = extract_oriented_plane(multivector[:, :, 0, :])  # (1, objects, 3)

        return wss, None
