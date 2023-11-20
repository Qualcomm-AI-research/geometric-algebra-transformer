# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
import pytest
import torch

from gatr.interface import embed_oriented_plane, extract_oriented_plane
from tests.helpers import BATCH_DIMS, TOLERANCES


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("include_position", [False, True])
def test_plane_embedding_consistency(batch_dims, include_position):
    """Tests whether plane embeddings into multivectors are cycle consistent."""
    pos = torch.randn(*batch_dims, 3) if include_position else None
    normal = torch.randn(*batch_dims, 3)
    normal = normal / torch.linalg.norm(normal, dim=-1, keepdim=True)
    multivectors = embed_oriented_plane(normal, position=pos)
    normal_extracted = extract_oriented_plane(multivectors)
    torch.testing.assert_close(normal, normal_extracted, **TOLERANCES)
