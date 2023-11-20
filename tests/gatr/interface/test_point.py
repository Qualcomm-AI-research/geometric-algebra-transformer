# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
import numpy as np
import pytest
import torch

from gatr.interface import embed_point, extract_point, extract_point_embedding_reg
from tests.helpers import BATCH_DIMS, TOLERANCES


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_point_embedding_consistency(batch_dims):
    """Tests whether 3D point embeddings into multivectors are cycle consistent."""
    points = torch.randn(*batch_dims, 3)
    multivectors = embed_point(points)
    points_reencoded = extract_point(multivectors)
    other_mv_components = extract_point_embedding_reg(multivectors)
    torch.testing.assert_close(points, points_reencoded, **TOLERANCES)
    torch.testing.assert_close(
        other_mv_components, torch.zeros_like(other_mv_components), **TOLERANCES
    )


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("rescaling", [2.0, -10.0, None])
def test_point_scaling_invariance(batch_dims, rescaling):
    """Tests whether 3D points extracted from multivector are invariant under rescalings of the
    multivector."""

    # If rescaling is None, sample a number (but avoid zero)
    if rescaling is None:
        while rescaling is None or np.abs(rescaling) < 0.1:
            rescaling = np.random.randn()

    multivectors = torch.randn(*batch_dims, 16)

    # Make sure the homogeneous coordinate is not too small
    multivectors[..., 14] = torch.where(
        abs(multivectors[..., 14]) < 0.1, 0.1, multivectors[..., 14]
    )

    rescaled_multivectors = rescaling * multivectors
    points = extract_point(multivectors)
    points_from_rescaled = extract_point(rescaled_multivectors)
    torch.testing.assert_close(points, points_from_rescaled, **TOLERANCES)
