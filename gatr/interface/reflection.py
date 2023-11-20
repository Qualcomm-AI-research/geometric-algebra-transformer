# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
"""Functions that create multivectors representing elements of the Pin group."""


from typing import Optional

import torch

from gatr.interface.plane import embed_oriented_plane, extract_oriented_plane


def embed_reflection(normal: torch.Tensor, position: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Embeds the reflection on a plane in multivectors.

    Following L. Dorst, this is represented as a PGA vector.

    References
    ----------
    Leo Dorst, "A Guided Tour to the Plane-Based Geometric Algebra PGA",
        https://geometricalgebra.org/downloads/PGA4CS.pdf

    Parameters
    ----------
    position : torch.Tensor with shape (..., 3) or None
        One position on the plane. If None, the plane goes through the origin.
    normal : torch.Tensor with shape (..., 3)
        Normal to the plane.

    Returns
    -------
    multivector : torch.Tensor with shape (..., 16)
        Embedding into multivector.
    """

    return embed_oriented_plane(normal, position)


def extract_reflection(multivector: torch.Tensor) -> torch.Tensor:
    """Extracts the normal on an reflection plane from a multivector.

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

    return extract_oriented_plane(multivector)
