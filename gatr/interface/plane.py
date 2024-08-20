# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All rights reserved.
import torch
from torch import Tensor

from gatr.interface.translation import embed_translation
from gatr.primitives.bilinear import geometric_product


def embed_oriented_plane(normal: Tensor, position: Tensor) -> Tensor:
    """Embeds an (oriented plane) in the PGA.

    Following L. Dorst, the plane is represent as PGA vectors.

    References
    ----------
    Leo Dorst, "A Guided Tour to the Plane-Based Geometric Algebra PGA",
        https://geometricalgebra.org/downloads/PGA4CS.pdf

    Parameters
    ----------
    normal : torch.Tensor with shape (..., 3)
        Normal to the plane.
    position : torch.Tensor with shape (..., 3)
        One position on the plane.

    Returns
    -------
    multivector : torch.Tensor with shape (..., 16)
        Embedding into multivector.
    """

    # Create multivector tensor with same batch shape, same device, same dtype as input
    batch_shape = normal.shape[:-1]
    multivector = torch.zeros(*batch_shape, 16, dtype=normal.dtype, device=normal.device)

    # Embedding a plane through origin into vectors
    multivector[..., 2:5] = normal[..., :]

    # Shift away from origin by translating
    translation = embed_translation(position)
    inverse_translation = embed_translation(-position)
    multivector = geometric_product(
        geometric_product(translation, multivector), inverse_translation
    )

    return multivector


def extract_oriented_plane(multivector: torch.Tensor) -> torch.Tensor:
    """Extracts the normal on an oriented plane from a multivector.

    Currently, this function does *not* extract a support point for the plane (or the distance to
    the origin).

    This is an equivariant function producing a 3-vector that rotates, but is invariant to
    translation. In fact, this is the only way of extracting such a vector from a multivector
    with a linear function.

    References
    ----------
    Leo Dorst, "A Guided Tour to the Plane-Based Geometric Algebra PGA",
        https://geometricalgebra.org/downloads/PGA4CS.pdf

    Parameters
    ----------
    multivector : torch.Tensor with shape (..., 16)
        Embedding into multivector.

    Returns
    -------
    normal : torch.Tensor with shape (..., 3)
        Normal to the plane.
    """

    return multivector[..., 2:5]
