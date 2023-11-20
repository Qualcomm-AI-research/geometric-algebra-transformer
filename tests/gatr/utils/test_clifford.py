# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
"""Unit tests for gatr.utils.ga_clifford."""

import pytest
import torch

from gatr.utils.clifford import mv_list_to_tensor, tensor_to_mv_list


@pytest.mark.parametrize("batch_dims", [(1,), (100,), (5, 7), tuple()])
def test_mv_list_tensor_conversion(batch_dims):
    """Tests whether the conversion between lists of clifford.Multivectors and torch.Tensors is
    cycle-consistent."""

    tensor = torch.randn(*batch_dims, 16)
    mv_list = tensor_to_mv_list(tensor)
    tensor_out = mv_list_to_tensor(mv_list, batch_shape=batch_dims)

    torch.testing.assert_close(tensor_out, tensor)
