# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
import pytest
import torch

from gatr.interface import (
    embed_oriented_plane,
    embed_point,
    embed_pseudoscalar,
    embed_rotation,
    embed_scalar,
    extract_oriented_plane,
    extract_point,
    extract_pseudoscalar,
    extract_rotation,
    extract_scalar,
)
from gatr.primitives.bilinear import geometric_product
from gatr.primitives.linear import reverse
from gatr.utils.quaternions import quaternion_to_rotation_matrix, random_quaternion
from tests.helpers import BATCH_DIMS, TOLERANCES


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_rotation_embedding_consistency(batch_dims):
    """Tests whether rotation embeddings into multivectors are cycle consistent."""

    quaternions = torch.randn(*batch_dims, 4)
    multivectors = embed_rotation(quaternions)
    quaternions_reencoded = extract_rotation(multivectors)
    torch.testing.assert_close(quaternions, quaternions_reencoded, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_rotation_on_point(batch_dims):
    """Tests whether applying rotations to points via matrix-vector multiplication equals rotation
    via geometric product."""

    # Sample random quaternions and points:
    q = random_quaternion(batch_dims)
    x = torch.randn(*batch_dims, 3)

    # Method 1: Convert to orthogonal matrix and do matrix-vector multiplication
    r = quaternion_to_rotation_matrix(q)
    rx = torch.einsum("... i j, ... j -> ... i", r, x)

    # Method 2: Embed as multivectors and use sandwich product
    m = embed_rotation(q)
    p = embed_point(x)
    mpm = geometric_product(geometric_product(m, p), reverse(m))
    rxp = embed_point(rx)
    mpmx = extract_point(mpm)

    torch.testing.assert_close(mpm, rxp, **TOLERANCES)
    torch.testing.assert_close(mpmx, rx, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_rotation_on_plane(batch_dims):
    """Tests whether applying rotations to planes via matrix-vector multiplication equals rotation
    via geometric product."""

    # Sample random quaternions and plane normals:
    q = random_quaternion(batch_dims)
    normals = torch.randn(*batch_dims, 3)
    normals /= torch.linalg.norm(normals, dim=-1).unsqueeze(-1)

    # Method 1: Convert to orthogonal matrix and do matrix-vector multiplication
    r = quaternion_to_rotation_matrix(q)
    rx = torch.einsum("... i j, ... j -> ... i", r, normals)

    # Method 2: Embed as multivectors and use sandwich product
    m = embed_rotation(q)
    p = embed_oriented_plane(normals, torch.zeros_like(normals))
    mpm = geometric_product(geometric_product(m, p), reverse(m))
    rxp = embed_oriented_plane(rx, torch.zeros_like(normals))
    mpmx = extract_oriented_plane(mpm)

    torch.testing.assert_close(mpm, rxp, **TOLERANCES)
    torch.testing.assert_close(mpmx, rx, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_rotation_on_scalar(batch_dims):
    """Tests whether rotations act on scalars as expected."""

    # Scalar
    s = torch.randn(*batch_dims, 1)
    x = embed_scalar(s)

    # Rotation
    q = random_quaternion(batch_dims)
    u = embed_rotation(q)

    # Compute rotation with sandwich product
    rx = geometric_product(geometric_product(u, x), reverse(u))
    rs = extract_scalar(rx)

    # Verify that scalar didn't change
    torch.testing.assert_close(rs, s, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_rotation_on_pseudoscalar(batch_dims):
    """Tests whether rotations act on pseudoscalars as expected."""

    # Scalar
    ps = torch.randn(*batch_dims, 1)
    x = embed_pseudoscalar(ps)

    # Rotation
    q = random_quaternion(batch_dims)
    u = embed_rotation(q)

    # Compute rotation with sandwich product
    rx = geometric_product(geometric_product(u, x), reverse(u))
    rps = extract_pseudoscalar(rx)

    # Verify that pseudoscalar didn't change
    torch.testing.assert_close(rps, ps, **TOLERANCES)
