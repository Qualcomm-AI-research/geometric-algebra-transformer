# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All rights reserved.
"""Functions that embed points in the geometric algebra."""


import warnings

import torch

from gatr.utils.warning import GATrDeprecationWarning


def embed_point(coordinates: torch.Tensor) -> torch.Tensor:
    """Embeds 3D points in multivectors.

    We follow the convention used in the reference below and map points to tri-vectors.

    References
    ----------
    Leo Dorst, "A Guided Tour to the Plane-Based Geometric Algebra PGA",
    https://geometricalgebra.org/downloads/PGA4CS.pdf

    Parameters
    ----------
    coordinates : torch.Tensor with shape (..., 3)
        3D coordinates

    Returns
    -------
    multivector : torch.Tensor with shape (..., 16)
        Embedding into multivector.
    """

    # Create multivector tensor with same batch shape, same device, same dtype as input
    batch_shape = coordinates.shape[:-1]
    multivector = torch.zeros(*batch_shape, 16, dtype=coordinates.dtype, device=coordinates.device)

    # Embedding into trivectors
    # Homogeneous coordinates: unphysical component / embedding dim, x_123
    multivector[..., 14] = 1.0
    multivector[..., 13] = -coordinates[..., 0]  # x-coordinate embedded in x_023
    multivector[..., 12] = coordinates[..., 1]  # y-coordinate embedded in x_013
    multivector[..., 11] = -coordinates[..., 2]  # z-coordinate embedded in x_012

    return multivector


def extract_point(
    multivector: torch.Tensor, divide_by_embedding_dim: bool = True, threshold: float = 1e-3
) -> torch.Tensor:
    """Given a multivector, extract any potential 3D point from the trivector components.

    Nota bene: if the output is interpreted a regular R^3 point,
    this function is only equivariant if divide_by_embedding_dim=True
    (or if the e_123 component is guaranteed to equal 1)!

    References
    ----------
    Leo Dorst, "A Guided Tour to the Plane-Based Geometric Algebra PGA",
        https://geometricalgebra.org/downloads/PGA4CS.pdf

    Parameters
    ----------
    multivector : torch.Tensor with shape (..., 16)
        Multivector.
    divide_by_embedding_dim : bool
        Whether to divice by the embedding dim. Proper PGA etiquette would have us do this, but it
        may not be good for NN training. If set the False, this function is not equivariant for all
        inputs!
    threshold : float
        Minimum value of the additional, unphysical component. Necessary to avoid exploding values
        or NaNs when this unphysical component of the homogeneous coordinates becomes small.

    Returns
    -------
    coordinates : torch.Tensor with shape (..., 3)
        3D coordinates corresponding to the trivector components of the multivector.
    """
    if not divide_by_embedding_dim:
        warnings.warn(
            'Calling "extract_point" with divide_by_embedding_dim=False is deprecated, '
            "because it is not equivariant.",
            GATrDeprecationWarning,
            2,
        )

    coordinates = torch.cat(
        [-multivector[..., [13]], multivector[..., [12]], -multivector[..., [11]]], dim=-1
    )

    # Divide by embedding dim
    if divide_by_embedding_dim:
        embedding_dim = multivector[
            ..., [14]
        ]  # Embedding dimension / scale of homogeneous coordinates
        embedding_dim = torch.where(torch.abs(embedding_dim) > threshold, embedding_dim, threshold)
        coordinates = coordinates / embedding_dim

    return coordinates


def extract_point_embedding_reg(multivector: torch.Tensor) -> torch.Tensor:
    """Given a multivector x, returns |x_{123}| - 1.

    Put differently, this is the deviation of the norm of a pseudoscalar component from 1.
    This can be used as a regularization term when predicting point positions, to avoid x_123 to
    be too close to 0.

    Parameters
    ----------
    multivector : torch.Tensor with shape (..., 16)
        Multivector.

    Returns
    -------
    regularization : torch.Tensor with shape (..., 1)
        |multivector_123| - 1.
    """

    return torch.abs(multivector[..., [14]]) - 1.0
