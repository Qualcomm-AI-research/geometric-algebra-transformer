# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
"""Unit tests of bilinear primitives."""

import pytest
import torch

from gatr.primitives.linear import (
    NUM_PIN_LINEAR_BASIS_ELEMENTS,
    equi_linear,
    grade_involute,
    grade_project,
    reverse,
)
from tests.helpers import (
    BATCH_DIMS,
    TOLERANCES,
    check_consistence_with_grade_involution,
    check_consistence_with_reversal,
    check_pin_equivariance,
)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_reverse_correctness(batch_dims):
    """Tests the multivector reverse for equivariance."""
    check_consistence_with_reversal(reverse, batch_dims=batch_dims, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_grade_involution_correctness(batch_dims):
    """Tests the multivector reverse for equivariance."""
    check_consistence_with_grade_involution(grade_involute, batch_dims=batch_dims, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_identity_equivariance(batch_dims):
    """Tests an identity map for equivariance (testing the test)."""
    check_pin_equivariance(lambda x: x, 1, batch_dims=batch_dims, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_grade_project_equivariance(batch_dims):
    """Tests the grade_project() primitive for equivariance."""
    check_pin_equivariance(grade_project, 1, batch_dims=batch_dims, **TOLERANCES)


@pytest.mark.parametrize(
    "input_batch_dims,coeff_batch_dims",
    [
        ((7,), (5, 7)),
        ((3, 7), (5, 7)),
        ((2, 3, 7), (5, 7)),
    ],
)
def test_linear_equivariance(input_batch_dims, coeff_batch_dims):
    """Tests the equi_linear() primitive for equivariance."""
    fn_kwargs = dict(coeffs=torch.randn(*coeff_batch_dims, NUM_PIN_LINEAR_BASIS_ELEMENTS))
    check_pin_equivariance(
        equi_linear, 1, fn_kwargs=fn_kwargs, batch_dims=input_batch_dims, **TOLERANCES
    )
