# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
import numpy as np
import pytest
import torch
from scipy.stats import ortho_group

from gatr.interface import (
    embed_3d_object,
    embed_3d_object_two_vec,
    extract_3d_object,
    extract_3d_object_two_vec,
)
from tests.helpers import BATCH_DIMS, TOLERANCES


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("explicit_point", [True, False])
def test_3d_object_embedding_consistency(batch_dims, explicit_point):
    """Tests whether 3D point embeddings into multivectors are cycle consistent."""
    points = torch.randn(*batch_dims, 1, 3)
    quaternions = torch.randn(*batch_dims, 1, 4)
    quaternions = quaternions / torch.linalg.norm(quaternions, dim=-1, keepdim=True)

    multivectors = embed_3d_object(points, quaternions, explicit_point=explicit_point)
    points_reencoded, quats_reencoded = extract_3d_object(multivectors, sign_reference=quaternions)

    torch.testing.assert_close(points, points_reencoded, **TOLERANCES)
    torch.testing.assert_close(quaternions, quats_reencoded, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_3d_object_two_vec_embedding_consistency(batch_dims):
    """Tests whether 3D point embeddings in the two-vec parameterizations into multivectors are
    cycle consistent."""
    point = torch.randn(*batch_dims, 1, 3)
    orientation = torch.from_numpy(ortho_group.rvs(3, size=np.product(batch_dims))).to(
        torch.float32
    )[..., :2, :]
    orientation = orientation.reshape((*batch_dims, 1, 6))
    multivector = embed_3d_object_two_vec(point, orientation=orientation)
    point_reencoded, orientation_reencoded = extract_3d_object_two_vec(multivector)

    torch.testing.assert_close(point, point_reencoded, **TOLERANCES)
    torch.testing.assert_close(orientation, orientation_reencoded, **TOLERANCES)
