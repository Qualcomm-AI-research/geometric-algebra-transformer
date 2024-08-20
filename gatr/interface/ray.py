# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All rights reserved.
import torch
from torch import Tensor


def embed_pluecker_ray(pluecker_ray: Tensor) -> Tensor:
    """Embed ray in Plücker coordinates as a multivector.

    Plücker coords are (v, o x v) for ray through o in direction v.

    Args:
        pluecker_ray (Tensor): of shape [..., 6]

    Returns:
        Tensor: of shape [..., 16]
    """
    mv = torch.zeros(
        *pluecker_ray.shape[:-1], 16, device=pluecker_ray.device, dtype=pluecker_ray.dtype
    )
    mv[..., 5:8] = pluecker_ray[..., 3:6]
    mv[..., 8] = pluecker_ray[..., 2]
    mv[..., 9] = -pluecker_ray[..., 1]
    mv[..., 10] = pluecker_ray[..., 0]
    return mv


def extract_pluecker_ray(multivector: Tensor) -> Tensor:
    """Extract ray in Plücker coordinates from a multivector.

    Plücker coords are (v, o x v) for ray through o in direction v.

    Args:
        multivector: Tensor of shape [..., 16]

    Returns:
        pluecker_ray (Tensor): of shape [..., 6]
    """
    return torch.cat(
        [
            multivector[..., 10, None],
            -multivector[..., 9, None],
            multivector[..., 8, None],
            multivector[..., 5:8],
        ],
        -1,
    )
