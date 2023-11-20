# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
import pytest
import torch

from gatr.layers.dropout import GradeDropout
from tests.helpers import MILD_TOLERANCES, TOLERANCES, check_pin_equivariance


@pytest.mark.parametrize("batch_dims", [(10,)])
@pytest.mark.parametrize("p", [0.0, 0.2])
@pytest.mark.parametrize("training", [True, False])
def test_dropout_shape(training, p, batch_dims):
    """Tests GradeDropout for shape correctness."""

    layer = GradeDropout(p=p)
    if training:
        layer.train()
    else:
        layer.eval()

    mv = torch.randn(*batch_dims, 16)
    s = torch.randn(*batch_dims)
    out_mv, out_s = layer(mv, s)

    assert out_mv.shape == mv.shape
    assert out_s.shape == s.shape


@pytest.mark.parametrize("batch_dims", [(10,)])
@pytest.mark.parametrize("training", [True, False])
def test_dropout_trivial_limit(training, batch_dims):
    """Tests that GradeDropout does nothing when p = 0."""

    layer = GradeDropout(p=0.0)
    if training:
        layer.train()
    else:
        layer.eval()

    mv = torch.randn(*batch_dims, 16)
    s = torch.randn(*batch_dims)
    out_mv, out_s = layer(mv, s)

    torch.testing.assert_close(out_mv, mv, **TOLERANCES)
    torch.testing.assert_close(out_s, s, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", [(10,)])
@pytest.mark.parametrize("p", [0.0, 0.2])
@pytest.mark.parametrize("training", [True, False])
def test_dropout_expectation(training, p, batch_dims, num_trials=10000):
    """Tests GradeDropout for correct expectation."""

    layer = GradeDropout(p=p)
    if training:
        layer.train()
    else:
        layer.eval()

    mv = torch.randn(*batch_dims, 16)
    s = torch.randn(*batch_dims)
    out_mv, out_s = layer(
        mv.unsqueeze(0).expand(num_trials, *mv.shape), s.unsqueeze(0).expand(num_trials, *s.shape)
    )
    out_mv = out_mv.mean(dim=0)
    out_s = out_s.mean(dim=0)

    torch.testing.assert_close(
        out_mv, mv, **MILD_TOLERANCES
    )  # Over 10k trials we won't get perfect agreement
    torch.testing.assert_close(out_s, s, **MILD_TOLERANCES)


@pytest.mark.parametrize("batch_dims", [(10,)])
@pytest.mark.parametrize("p", [0.0, 0.2])
def test_dropout_equivariance(p, batch_dims):
    """Tests GradeDropout for equivariance."""

    layer = GradeDropout(p=p)
    layer.eval()
    s = torch.randn(*batch_dims)
    check_pin_equivariance(layer, 1, batch_dims=batch_dims, fn_kwargs=dict(scalars=s), **TOLERANCES)
