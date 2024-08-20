# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All rights reserved.
from typing import Optional

import torch

from gatr.interface.plane import embed_oriented_plane, extract_oriented_plane
from gatr.interface.point import embed_point, extract_point
from gatr.interface.rotation import embed_rotation, extract_rotation
from gatr.interface.translation import embed_translation
from gatr.primitives.bilinear import geometric_product, outer_product
from gatr.primitives.linear import reverse


def embed_3d_object(
    position: torch.Tensor, orientation: torch.Tensor, explicit_point: bool = False
) -> torch.Tensor:
    """Embeds the pose of a 3D object PGA multivectors.

    The pose is given by a position and a orientation quaternion. The latter is assumed to follow
    the pybullet conventions: to be given in Hamilton convention, where ijk = -1, and with the
    scalar component last (ijkw order).

    The embedding contains three plane multivectors v_i with
    ```
    v_i = T(p) R(q) e_i reverse(T(p) R(q))
    ```
    where `T()` is the translation embedding, `R()` is the rotation embedding, and `e_i` are the PGA
    basis vectors.

    If `explicit_point`, then a fourth multivector represents the position itself.

    References
    ----------
    Sommer et al., "Why and How to Avoid the Flipped Quaternion Multiplication", arXiv:1801.07478

    Parameters
    ----------
    position : torch.Tensor with shape (..., 1, 3)
        Position in 3D space.
    orientation : torch.Tensor with shape (..., 1, 4)
        Quaternions in ijkw order and Hamilton convention (ijk=-1)
    explicit_point : bool
        If True, in addition to the three planes we provide the explicit point representation
        `P(p)`.

    Returns
    -------
    multivector : torch.Tensor with shape (..., n_channels, 16)
        Embedding into multivectors, with `n_channels = 4 if explicit_point else 3`.
    """

    # Check inputs
    assert position.shape[-2:] == (1, 3)
    assert orientation.shape[-2:] == (1, 4)

    # Basis planes
    basis = torch.zeros(3, 16, dtype=position.dtype, device=position.device)  # (3, 16)
    basis[:, 2:5] = torch.eye(
        3, dtype=position.dtype, device=position.device
    )  # basis[i] has e_i = 1

    # Translation and rotation
    t = embed_translation(position)  # (..., 1, 16)
    r = embed_rotation(orientation)  # (..., 1, 16)
    tr = geometric_product(t, r)  # (..., 1, 16)

    # Construct embedding
    embedding = geometric_product(geometric_product(tr, basis), reverse(tr))  # (..., 3, 16)

    # Optional point embedding
    if explicit_point:
        point = embed_point(position)
        embedding = torch.cat((embedding, point), dim=-2)  # (..., 4, 16)

    return embedding


def extract_3d_object(
    multivector: torch.Tensor,
    normalize_quaternion: bool = True,
    sign_reference: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Extracts 3D position and quaternion orientation from the multivector embedding of an object.

    This function inverts embed_3d_object().

    The extraction starts from three (plane) MVs v_i and is based on the magic equation
    ```
    P(p) = v_1 ^ v_2 ^ v_3
    q = (1 + sum_i <v_i>_E e_i)
    ```
    where `<v_i>_E` is the projection of v_i to the Euclidean vector components.

    Parameters
    ----------
    multivector : torch.Tensor with shape (..., 3 or 4, 16)
        Object embedding.
    normalize_quaternion : bool
        Whether to normalize quaternions.
    sign_reference : None or torch.Tensor
        If provided, used to fix the (ambiguous) sign of the orientation quaternion.

    Returns
    -------
    position : torch.Tensor with shape (..., 1, 3)
        Position in 3D space.
    orientation : torch.Tensor with shape (..., 1, 4)
        Quaternions in ijkw order and Hamilton convention (ijk=-1).
    """

    # Split inputs
    v = [multivector[..., [i], :] for i in range(3)]  # (..., 1, 16) each

    # Basis vectors
    basis = torch.zeros(3, 16, dtype=multivector.dtype, device=multivector.device)  # (3, 16)
    basis[:, 2:5] = torch.eye(
        3, dtype=multivector.dtype, device=multivector.device
    )  # basis[i] has e_i = 1
    projection = torch.sum(basis, dim=0)  # (16)

    # Extract position
    if multivector.shape[-2] == 4:
        position = extract_point(multivector[..., [3], :])  # (..., 1, 3)
    else:
        position = outer_product(outer_product(v[0], v[1]), v[2])  # (..., 1, 16)
        position = extract_point(position)  # (..., 1, 3)

    # Extract orientation
    orientation = torch.zeros(16, dtype=multivector.dtype, device=multivector.device)
    orientation[0] = 1.0
    for vi, ei in zip(v, basis):
        vi_proj = projection * vi  # Project to Euclidean vector components
        orientation = orientation + geometric_product(vi_proj, ei)  # Magic equation

    quat = extract_rotation(orientation, normalize=normalize_quaternion)

    # Fix quaternion sign
    if sign_reference is not None:
        sign = torch.sign(torch.sum(sign_reference * quat, dim=-1, keepdim=True))
        quat = sign * quat

    return position, quat


def embed_3d_object_two_vec(position: torch.Tensor, orientation: torch.Tensor) -> torch.Tensor:
    """Embeds the pose of a 3D object in two-vec parameterization into the PGA.

    The inputs consist of the position of an object and two vectors that specify its orientation.
    These two vectors can be generated from a rotation matrix by taking the first two column
    vectors.

    The embedding contains three planes and a point, just like embed_3d_object().

    Parameters
    ----------
    position : torch.Tensor with shape (..., 1, 3)
        Position in 3D space.
    orientation : torch.Tensor with shape (..., 1, 6)
        Orientation in two-vec parameterization

    Returns
    -------
    multivector : torch.Tensor with shape (..., 4, 16)
        Embedding into four multivectors.
    """

    # Check inputs
    assert position.shape[-2:] == (1, 3)
    assert orientation.shape[-2:] == (1, 6)

    # Point
    point_embedding = embed_point(position)  # (..., 1, 16)

    # Two-vec -> three vectors
    orientation = orientation.reshape(*orientation.shape[:-2], 2, 3)
    third_vec = torch.linalg.cross(orientation[..., [0], :], orientation[..., [1], :])
    orientation = torch.cat((orientation, third_vec), dim=-2)

    # Embed orientation as planes
    orientation_embedding = embed_oriented_plane(orientation, position)  # (..., 3, 16)

    embedding = torch.cat((point_embedding, orientation_embedding), dim=-2)  # (..., 4, 16)
    return embedding


def extract_3d_object_two_vec(multivector: torch.Tensor) -> torch.Tensor:
    """Extracts 3D position and two-vec orientation from the multivector embedding of an object.

    This inverts embed_3d_object_two_vec().

    Parameters
    ----------
    multivector : torch.Tensor with shape (..., 4, 16)
        Object embedding.

    Returns
    -------
    position : torch.Tensor with shape (..., 1, 3)
        Position in 3D space.
    orientation : torch.Tensor with shape (..., 1, 6)
        Two-vec param of orientation
    """

    # Position
    position = extract_point(multivector[..., [0], :])  # (..., 1, 3)

    # Orientation
    orientation = extract_oriented_plane(multivector[..., [1, 2], :])  # (..., 2, 3)
    orientation = orientation.reshape(*multivector.shape[:-2], 1, 6)  # (..., 1, 6)

    return position, orientation
