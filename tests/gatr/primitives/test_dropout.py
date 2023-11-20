# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
import pytest
import torch

from gatr.primitives.dropout import grade_dropout
from tests.helpers import BATCH_DIMS, MILD_TOLERANCES, TOLERANCES, check_pin_equivariance


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("p", [0.0, 0.2])
@pytest.mark.parametrize("training", [True, False])
def test_dropout_shape(training, p, batch_dims):
    """Tests grade_dropout() map for shape correctness."""
    x = torch.randn(*batch_dims, 16)
    y = grade_dropout(x, p=p, training=training)
    assert y.shape == x.shape


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("training", [False, True])
def test_dropout_trivial_limit(training, batch_dims):
    """Tests that grade_dropout() does nothing when p = 0."""
    x = torch.randn(*batch_dims, 16)
    y = grade_dropout(x, p=0.0, training=training)

    torch.testing.assert_close(x, y, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("p", [0.0, 0.2])
def test_dropout_expectation(p, batch_dims, num_trials=10000):
    """Tests that grade_dropout() has train-time and test-time behaviour that is identical in
    expectation."""
    x = torch.randn(*batch_dims, 16)
    y_train = grade_dropout(x.unsqueeze(0).expand(num_trials, *x.shape), p=p, training=True).mean(
        dim=0
    )
    y_test = grade_dropout(x, p=p, training=False)

    torch.testing.assert_close(y_test, x, **TOLERANCES)
    torch.testing.assert_close(
        y_train, x, **MILD_TOLERANCES
    )  # Over 10k trials we won't get perfect agreement


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("p", [0.0, 0.2])
def test_dropout_equivariance(p, batch_dims):
    """Tests the grade_dropout() primitive for equivariance (at test time)."""
    check_pin_equivariance(
        grade_dropout, 1, batch_dims=batch_dims, fn_kwargs=dict(training=False, p=p), **TOLERANCES
    )
