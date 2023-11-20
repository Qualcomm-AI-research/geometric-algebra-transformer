# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
import torch

from gatr.utils.tensors import block_stack, expand_pairwise, to_nd


def test_to_nd():
    """Test to_nd."""
    assert to_nd(torch.randn(3), 3).shape == (1, 1, 3)
    assert to_nd(torch.randn(2, 3), 3).shape == (1, 2, 3)
    assert to_nd(torch.randn(2, 3, 4), 3).shape == (2, 3, 4)
    assert to_nd(torch.randn(5, 2, 3, 4), 3).shape == (10, 3, 4)
    assert to_nd(torch.randn(3), 4).shape == (1, 1, 1, 3)
    assert to_nd(torch.randn(2, 3), 4).shape == (1, 1, 2, 3)
    assert to_nd(torch.randn(2, 3, 4), 4).shape == (1, 2, 3, 4)
    assert to_nd(torch.randn(5, 2, 3, 4), 4).shape == (5, 2, 3, 4)
    assert to_nd(torch.randn(3, 5, 2, 3, 4), 4).shape == (15, 2, 3, 4)


def test_expand_pairwise():
    """Test expand pairwise."""
    x = torch.randn(2, 5)
    y = torch.randn(3, 2, 1)
    z = torch.randn(4, 1, 2, 1)
    x2, y2, z2 = expand_pairwise(x, y, z)
    assert x2.shape == (4, 3, 2, 5)
    assert y2.shape == (4, 3, 2, 5)
    assert z2.shape == (4, 3, 2, 5)


def test_block_stack():
    """Test block stacking."""
    x = torch.rand(4, 2, 3, 5)
    y = torch.rand(4, 4, 5, 5)
    z = torch.rand(4, 3, 3, 5)
    out = block_stack([x, y, z], 1, 2)
    assert list(out.shape) == [4, 9, 11, 5]
    torch.testing.assert_close(out[:, 0:2, 0:3, :], x)
    torch.testing.assert_close(out[:, 2:6, 3:8, :], y)
    torch.testing.assert_close(out[:, 6:9, 8:11, :], z)
    out[:, 0:2, 0:3, :] = 0
    out[:, 2:6, 3:8, :] = 0
    out[:, 6:9, 8:11, :] = 0
    torch.testing.assert_close(out, torch.zeros_like(out))
