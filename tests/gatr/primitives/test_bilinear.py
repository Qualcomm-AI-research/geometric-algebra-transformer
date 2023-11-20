# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
"""Unit tests of bilinear primitives."""

import pytest

from gatr.primitives.bilinear import geometric_product, outer_product
from tests.helpers import (
    BATCH_DIMS,
    TOLERANCES,
    check_consistence_with_geometric_product,
    check_consistence_with_outer_product,
    check_pin_equivariance,
)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_geometric_product_correctness(batch_dims):
    """Tests the geometric_product() primitive for correctness (that is, consistency with the
    clifford library).
    """
    check_consistence_with_geometric_product(geometric_product, batch_dims, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_outer_product_correctness(batch_dims):
    """Tests the outer_product() primitive for correctness (that is, consistency with the clifford
    library).
    """
    check_consistence_with_outer_product(outer_product, batch_dims, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_geometric_product_equivariance(batch_dims):
    """Tests the geometric_product() primitive for equivariance."""
    check_pin_equivariance(geometric_product, 2, batch_dims=batch_dims, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_outer_product_equivariance(batch_dims):
    """Tests the outer_product() primitive for equivariance."""
    check_pin_equivariance(outer_product, 2, batch_dims=batch_dims, **TOLERANCES)
