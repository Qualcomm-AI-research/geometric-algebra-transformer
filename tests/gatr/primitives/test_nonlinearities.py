# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
"""Unit tests of nonlinearity primitives."""

import pytest
import torch

from gatr.primitives import gated_relu, gated_sigmoid
from gatr.primitives.nonlinearities import gated_gelu, gated_gelu_divide
from tests.helpers import BATCH_DIMS, TOLERANCES, check_pin_equivariance


@pytest.mark.parametrize("fn", [gated_relu, gated_gelu, gated_sigmoid, gated_gelu_divide])
@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_gated_nonlin_equivariance(fn, batch_dims):
    """Tests an identity map for equivariance (testing the test)."""
    gates = torch.randn(*batch_dims, 1)
    check_pin_equivariance(fn, 1, fn_kwargs=dict(gates=gates), batch_dims=batch_dims, **TOLERANCES)


def test_gelu_division():
    """Test that gated_gelu_divide() is equal to gelu(x)/x."""
    values = torch.randn(2, 3, 4)
    gates = torch.randn(2, 3, 4)
    y1 = gated_gelu(values, gates) / gates
    y2 = gated_gelu_divide(values, gates)
    torch.testing.assert_close(y1, y2)
