# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All rights reserved.
import pytest
import torch

from gatr.interface import (
    embed_oriented_plane,
    embed_point,
    embed_pseudoscalar,
    embed_scalar,
    embed_translation,
    extract_oriented_plane,
    extract_point,
    extract_point_embedding_reg,
    extract_pseudoscalar,
    extract_scalar,
    extract_translation,
)
from gatr.primitives import geometric_product, reverse
from gatr.utils.warning import GATrDeprecationWarning
from tests.helpers import BATCH_DIMS, TOLERANCES


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_translation_embedding_consistency(batch_dims):
    """Tests whether translation embeddings into multivectors are cycle consistent."""
    translations = torch.randn(*batch_dims, 3)
    multivectors = embed_translation(translations)
    with pytest.warns(GATrDeprecationWarning):  # ensure deprecationwarning raised
        translations_reencoded = extract_translation(multivectors)
    torch.testing.assert_close(translations, translations_reencoded, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_translation_on_point(batch_dims):
    """Tests whether translation embeddings act on point embeddings as expected."""

    # Generate input data
    points = torch.randn(*batch_dims, 3)
    translation_vectors = torch.randn(*batch_dims, 3)

    # True translation
    translated_points_true = points + translation_vectors

    # Embed points and translations into MV
    points_embedding = embed_point(points)
    translations_embedding = embed_translation(translation_vectors)
    inv_translations_embedding = reverse(translations_embedding)

    # Compute translation in multivector space with sandwich product
    translated_points_sandwich = geometric_product(translations_embedding, points_embedding)
    translated_points_sandwich = geometric_product(
        translated_points_sandwich, inv_translations_embedding
    )
    other_components_sandwich = extract_point_embedding_reg(translated_points_sandwich)
    translated_points_sandwich = extract_point(translated_points_sandwich)

    # Verify that all translation results are consistent
    torch.testing.assert_close(translated_points_sandwich, translated_points_true, **TOLERANCES)

    # Verify that output MVs have other components at zero
    torch.testing.assert_close(
        other_components_sandwich, torch.zeros_like(other_components_sandwich), **TOLERANCES
    )


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_translation_on_plane(batch_dims):
    """Tests whether translation embeddings act on oriented plane embeddings as expected."""

    # Generate input data
    normals = torch.randn(*batch_dims, 3)
    normals /= torch.linalg.norm(normals, dim=-1).unsqueeze(-1)
    translation_vectors = torch.randn(*batch_dims, 3)
    pos = torch.zeros_like(normals)

    # Embed planes and translations into MV
    plane_embedding = embed_oriented_plane(normals, pos)
    translations_embedding = embed_translation(translation_vectors)
    inv_translations_embedding = reverse(translations_embedding)

    # Compute translation in multivector space with sandwich product
    translated_plane_sandwich = geometric_product(translations_embedding, plane_embedding)
    translated_plane_sandwich = geometric_product(
        translated_plane_sandwich, inv_translations_embedding  # ensure deprecationwarning raised
    )
    translated_normals = extract_oriented_plane(translated_plane_sandwich)

    # Verify that normals didn't change
    torch.testing.assert_close(translated_normals, normals, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_translation_on_scalar(batch_dims):
    """Tests whether translations act on scalars as expected."""

    # Scalar
    s = torch.randn(*batch_dims, 1)
    x = embed_scalar(s)

    # Translation
    t = torch.randn(*batch_dims, 3)
    u = embed_translation(t)

    # Compute rotation with sandwich product
    rx = geometric_product(geometric_product(u, x), reverse(u))
    rs = extract_scalar(rx)

    # Verify that scalar didn't change
    torch.testing.assert_close(rs, s, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_translation_on_pseudoscalar(batch_dims):
    """Tests whether translations act on pseudoscalars as expected."""

    # Scalar
    ps = torch.randn(*batch_dims, 1)
    x = embed_pseudoscalar(ps)

    # Translation
    t = torch.randn(*batch_dims, 3)
    u = embed_translation(t)

    # Compute translation with sandwich product
    rx = geometric_product(geometric_product(u, x), reverse(u))
    rps = extract_pseudoscalar(rx)

    # Verify that pseudoscalar didn't change
    torch.testing.assert_close(rps, ps, **TOLERANCES)
