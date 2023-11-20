# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
import pytest
import torch

from gatr.layers import ApplyRotaryPositionalEncoding
from tests.helpers import BATCH_DIMS


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("additional_dims", [tuple(), (3,), (11, 13)])
@pytest.mark.parametrize("num_channels", [10])
@pytest.mark.parametrize("num_objects", [5])
def test_apply_rotary_positional_encoding(batch_dims, num_objects, additional_dims, num_channels):
    """Tests ApplyRotaryPositionalEncoding for consistency."""

    # Generate input data
    data = torch.randn(*batch_dims, num_objects, *additional_dims, num_channels)
    item_dim = len(batch_dims)

    # Layer
    layer = ApplyRotaryPositionalEncoding(num_channels, item_dim)

    # Outputs
    outputs = layer(data)

    # Outputs should have same shape as inputs, except that the number of features is increased
    assert outputs.shape == (*batch_dims, num_objects, *additional_dims, num_channels)
