# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All rights reserved.
import torch

from gatr.interface import embed_pluecker_ray, embed_point, extract_pluecker_ray
from gatr.primitives.dual import equivariant_join


def test_embed_pluecker_ray():
    """Test whether ray matches the join construction."""
    camera = torch.randn(100, 3)
    pixel = torch.randn(100, 3)

    direction = pixel - camera
    pluecker_ray = torch.cat([direction, torch.linalg.cross(pixel, pixel - camera)], 1)

    mv = embed_pluecker_ray(pluecker_ray)
    mv2 = equivariant_join(embed_point(camera), embed_point(pixel), reference=torch.ones(16))

    torch.testing.assert_close(mv, mv2)


def test_embed_extract_pluecker_ray():
    """Test whether embed and extract is the identity."""
    pluecker = torch.randn(100, 6)
    mv = embed_pluecker_ray(pluecker)
    recon = extract_pluecker_ray(mv)
    torch.testing.assert_close(pluecker, recon)
