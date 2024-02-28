# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
import pytest
import torch

from gatr.interface import (
    embed_oriented_plane,
    embed_point,
    embed_pseudoscalar,
    embed_reflection,
    embed_scalar,
    extract_oriented_plane,
    extract_point,
    extract_point_embedding_reg,
    extract_pseudoscalar,
    extract_reflection,
    extract_scalar,
)
from gatr.primitives import geometric_product, grade_involute
from tests.helpers import BATCH_DIMS, TOLERANCES


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_reflection_embedding_consistency(batch_dims):
    """Tests whether reflection embeddings into multivectors are cycle consistent."""
    reflection_normals = torch.randn(*batch_dims, 3)
    reflection_normals /= torch.linalg.norm(reflection_normals, dim=-1).unsqueeze(-1)
    reflection_pos = torch.randn(*batch_dims, 3)
    multivectors = embed_reflection(reflection_normals, reflection_pos)
    reflection_normals_reencoded = extract_reflection(multivectors)
    torch.testing.assert_close(reflection_normals, reflection_normals_reencoded, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("reflection_normal_to_axis", [0, 1, 2])
def test_reflection_on_point(batch_dims, reflection_normal_to_axis):
    """Tests whether the embedding of a reflection along the plane normal to a given coordinate axis
    acts on point embeddings as expected."""

    # Generate input data
    points = torch.randn(*batch_dims, 3)
    reflection_normal = torch.zeros(3)
    reflection_normal[reflection_normal_to_axis] = 1.0

    # True reflections
    reflected_true = torch.clone(points)
    reflected_true[..., reflection_normal_to_axis] = -points[..., reflection_normal_to_axis]

    # Embed points and translations into MV
    points_embedding = embed_point(points)
    reflection_embedding = embed_reflection(reflection_normal, torch.zeros_like(reflection_normal))

    # Compute translation in multivector space with sandwich product
    reflected_sandwich = geometric_product(reflection_embedding, points_embedding)
    reflected_sandwich = geometric_product(reflected_sandwich, reflection_embedding)
    reflected_sandwich = grade_involute(reflected_sandwich)
    other_components_sandwich = extract_point_embedding_reg(reflected_sandwich)
    reflected_sandwich = extract_point(reflected_sandwich)

    # Verify that all translation results are consistent
    torch.testing.assert_close(reflected_sandwich, reflected_true, **TOLERANCES)

    # Verify that output MVs have other components at zero
    torch.testing.assert_close(
        other_components_sandwich, torch.zeros_like(other_components_sandwich), **TOLERANCES
    )


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("reflection_normal_to_axis", [0, 1, 2])
def test_reflection_on_plane(batch_dims, reflection_normal_to_axis):
    """Tests whether reflection embeddings act on oriented plane embeddings as expected."""

    # Plane
    plane_normals = torch.randn(*batch_dims, 3)
    plane_normals /= torch.linalg.norm(plane_normals, dim=-1).unsqueeze(-1)
    plane_pos = torch.zeros_like(plane_normals)
    plane = embed_oriented_plane(plane_normals, plane_pos)

    # Reflection
    reflection_normal = torch.zeros(3)
    reflection_normal[reflection_normal_to_axis] = 1.0
    reflection_pos = torch.zeros_like(reflection_normal)
    reflection = embed_reflection(reflection_normal, reflection_pos)

    # Reflected plane from sandwich product
    reflected_plane = grade_involute(
        geometric_product(geometric_product(reflection, plane), reflection)
    )
    reflected_normals = extract_oriented_plane(reflected_plane)

    # True reflected plane
    reflected_true = torch.clone(plane_normals)
    reflected_true[..., reflection_normal_to_axis] = -plane_normals[..., reflection_normal_to_axis]

    # Verify that normals didn't change
    torch.testing.assert_close(reflected_normals, reflected_true, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_reflection_on_scalar(batch_dims):
    """Tests whether reflections act on scalars as expected."""

    # Generate input data
    inputs = torch.randn(*batch_dims, 1)
    reflection_normals = torch.randn(*batch_dims, 3)
    reflection_normals /= torch.linalg.norm(reflection_normals, dim=-1).unsqueeze(-1)
    reflection_pos = torch.zeros_like(reflection_normals)

    # Embed planes and translations into MV
    inputs_embedding = embed_scalar(inputs)
    reflection_embedding = embed_reflection(reflection_normals, reflection_pos)

    # Compute translation in multivector space with sandwich product
    reflected_inputs = geometric_product(reflection_embedding, inputs_embedding)
    reflected_inputs = geometric_product(reflected_inputs, reflection_embedding)
    reflected_inputs = grade_involute(reflected_inputs)
    reflected_inputs = extract_scalar(reflected_inputs)

    # Verify that normals didn't change
    torch.testing.assert_close(reflected_inputs, inputs, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_reflection_on_pseudoscalar(batch_dims):
    """Tests whether reflections act on pseudoscalars as expected."""

    # Generate input data
    inputs = torch.randn(*batch_dims, 1)
    reflection_normals = torch.randn(*batch_dims, 3)
    reflection_normals /= torch.linalg.norm(reflection_normals, dim=-1).unsqueeze(-1)
    reflection_pos = torch.zeros_like(reflection_normals)

    # Embed planes and translations into MV
    inputs_embedding = embed_pseudoscalar(inputs)
    reflection_embedding = embed_reflection(reflection_normals, reflection_pos)
    inv_reflection_embedding = reflection_embedding

    # Compute translation in multivector space with sandwich product
    reflected_inputs = geometric_product(reflection_embedding, inputs_embedding)
    reflected_inputs = geometric_product(reflected_inputs, inv_reflection_embedding)
    reflected_inputs = grade_involute(reflected_inputs)
    reflected_inputs = extract_pseudoscalar(reflected_inputs)

    # Verify that normals didn't change
    torch.testing.assert_close(reflected_inputs, -inputs, **TOLERANCES)
