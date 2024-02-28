# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
import pytest
import torch

from gatr.interface import embed_oriented_plane, extract_oriented_plane
from tests.helpers import BATCH_DIMS, TOLERANCES


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_plane_embedding_consistency(batch_dims):
    """Tests whether plane embeddings into multivectors are cycle consistent."""
    pos = torch.randn(*batch_dims, 3)
    normal = torch.randn(*batch_dims, 3)
    normal = normal / torch.linalg.norm(normal, dim=-1, keepdim=True)
    multivectors = embed_oriented_plane(normal, position=pos)
    normal_extracted = extract_oriented_plane(multivectors)
    torch.testing.assert_close(normal, normal_extracted, **TOLERANCES)
