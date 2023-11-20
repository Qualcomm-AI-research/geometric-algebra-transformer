# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
import torch


def embed_rotation(quaternion: torch.Tensor) -> torch.Tensor:
    """Embeds 3D rotations in multivectors.

    We follow the convention used in Leo Dorst, "A Guided Tour to the Plane-Based Geometric Algebra
    PGA", and map rotations around the origin to bivectors.

    For quaternions we use the "Hamilton" convention, where ijk = -1 (*not* the JPL convention
    where ijk = 1). For details, see "Why and How to Avoid the Flipped Quaternion Multiplication"
    by Sommer et al. (2018)

    References
    ----------
    Leo Dorst, "A Guided Tour to the Plane-Based Geometric Algebra PGA",
        https://geometricalgebra.org/downloads/PGA4CS.pdf
    Sommer et al., "Why and How to Avoid the Flipped Quaternion Multiplication", arXiv:1801.07478

    Parameters
    ----------
    quaternion : torch.Tensor with shape (..., 4)
        Quaternions in ijkw order and Hamilton convention (ijk=-1)

    Returns
    -------
    multivector : torch.Tensor with shape (..., 16)
        Embedding into multivector.
    """
    # Create multivector tensor with same batch shape, same device, same dtype as input
    batch_shape = quaternion.shape[:-1]
    multivector = torch.zeros(*batch_shape, 16, dtype=quaternion.dtype, device=quaternion.device)

    # Embedding into bivectors
    # w component of quaternion is the scalar component of the multivector
    multivector[..., 0] = quaternion[..., 3]
    multivector[..., 8] = -quaternion[..., 2]  # k component of quaternion is the bivector -e12
    multivector[..., 9] = quaternion[..., 1]  # j component of quaternion is the bivector e13
    multivector[..., 10] = -quaternion[..., 0]  # i component of quaternion is the bivector -e23

    return multivector


def extract_rotation(multivector: torch.Tensor, normalize: bool = False) -> torch.Tensor:
    """Extracts the represented rotation quaternion from the multivector's bivector components.

    Parameters
    ----------
    multivector : torch.Tensor with shape (..., 16)
        Multivector.
    normalize : bool
        Whether to normalize the quaternion to unit norm.

    Returns
    -------
    quaternion : torch.Tensor with shape (..., 4)
        quaternion in ijkw order and Hamilton convention (ijk = -1)
    """

    quaternion = torch.cat(
        [
            -multivector[..., [10]],
            multivector[..., [9]],
            -multivector[..., [8]],
            multivector[..., [0]],
        ],
        dim=-1,
    )

    if normalize:
        quaternion = quaternion / torch.linalg.norm(quaternion, dim=-1, keepdim=True)

    return quaternion
