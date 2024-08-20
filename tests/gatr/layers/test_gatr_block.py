# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All rights reserved.
# pylint: disable=redefined-outer-name
import pytest
import torch

from gatr.layers.attention.config import SelfAttentionConfig
from gatr.layers.gatr_block import GATrBlock
from gatr.layers.linear import EquiLinear
from gatr.layers.mlp.config import MLPConfig
from gatr.utils.tensors import construct_reference_multivector

_NUM_TOKENS = 3
_NUM_MV_CHANNELS = 4
_NUM_S_CHANNELS = 6


@pytest.fixture
def block():
    """GATrBlock fixture for testing"""
    return GATrBlock(
        _NUM_MV_CHANNELS,
        _NUM_S_CHANNELS,
        SelfAttentionConfig(),
        MLPConfig(),
        checkpoint=["mlp", "attention"],
    )


@pytest.fixture
def linear():
    """EquiLinear fixture for testing"""
    return EquiLinear(
        _NUM_MV_CHANNELS,
        _NUM_MV_CHANNELS,
        in_s_channels=_NUM_S_CHANNELS,
        out_s_channels=_NUM_S_CHANNELS,
    )


@pytest.fixture
def mv_in():
    """Multivector input fixture for testing"""
    return torch.randn(_NUM_TOKENS, _NUM_MV_CHANNELS, 16)


@pytest.fixture
def s_in():
    """Scalar input fixture for testing"""
    return torch.randn(_NUM_TOKENS, _NUM_S_CHANNELS)


@pytest.mark.parametrize("device,low_dtype", [("cpu", torch.bfloat16), ("cuda", torch.float16)])
def test_gatr_block_autocast(linear, block, mv_in, s_in, device, low_dtype):
    """Tests that AMP works correctly in GATr blocks"""

    high_dtype = torch.float32
    reference_mv = construct_reference_multivector("canonical", mv_in).to(device)
    linear.to(device)
    block.to(device)
    mv_in, s_in = mv_in.to(device), s_in.to(device)

    with torch.autocast(device, low_dtype, enabled=True):
        # Inputs
        assert mv_in.dtype == high_dtype
        assert s_in.dtype == high_dtype

        # Initial linear layer
        mv_in, s_in = linear(mv_in, scalars=s_in)
        assert mv_in.dtype == low_dtype
        assert s_in.dtype == low_dtype

        # Attention block: layer norm
        h_mv, h_s = block.norm(mv_in, scalars=s_in)
        assert h_mv.dtype == high_dtype
        # Autocast behaviour of layer_norm depends on device
        assert h_s.dtype == high_dtype if device == "cuda" else low_dtype

        # Attention block: self attention
        h_mv, h_s = block.attention(h_mv, scalars=h_s)
        assert h_mv.dtype == low_dtype
        assert h_s.dtype == low_dtype

        # Attention block: skip connection
        outputs_mv = mv_in + h_mv
        outputs_s = s_in + h_s
        assert h_mv.dtype == low_dtype
        assert h_s.dtype == low_dtype

        # MLP block: layer norm
        h_mv, h_s = block.norm(outputs_mv, scalars=outputs_s)
        assert h_s.dtype == high_dtype if device == "cuda" else low_dtype  # See above

        # MLP block: MLP
        h_mv, h_s = block.mlp(h_mv, scalars=h_s, reference_mv=reference_mv)
        assert h_mv.dtype == low_dtype
        assert h_s.dtype == low_dtype

        # MLP block: skip connection
        outputs_mv = outputs_mv + h_mv
        outputs_s = outputs_s + h_s
        assert outputs_mv.dtype == low_dtype
        assert outputs_s.dtype == low_dtype
