# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All rights reserved.
from copy import deepcopy

import pytest
import torch

from gatr.layers.linear import EquiLinear
from gatr.layers.mlp.config import MLPConfig
from gatr.layers.mlp.mlp import GeoMLP
from gatr.utils.compile_linear import (
    CompiledLinear,
    compile_equi_linear,
    compile_equi_linear_submodules,
)


@pytest.mark.parametrize("in_s_channels", [6, None])
@pytest.mark.parametrize("out_s_channels", [8, None])
@pytest.mark.parametrize("bias", [True, False])
def test_compile_linear_equivalence(in_s_channels, out_s_channels, bias):
    """
    Test equivalence of the EquiLinear and its compiled version.
    """
    in_mv_channels = 5
    out_mv_channels = 7

    equi_linear = EquiLinear(
        in_mv_channels, out_mv_channels, in_s_channels, out_s_channels, bias=bias
    )
    compiled_linear = compile_equi_linear(equi_linear)

    in_mv = torch.randn(100, in_mv_channels, 16)
    in_s = torch.randn(100, in_s_channels) if in_s_channels is not None else None

    out_mv_equi, out_s_equi = equi_linear(in_mv, in_s)
    out_mv_compiled, out_s_compiled = compiled_linear(in_mv, in_s)

    torch.testing.assert_close(out_mv_equi, out_mv_compiled)
    torch.testing.assert_close(out_s_equi, out_s_compiled)


def test_compilation_substitution():
    """Test in lienar compilation in integration.

    Verify that all EquiLinear submodules are substitued by CompiledLinear.
    Verify that the forward pass is equivalent after substitution.
    """
    mlp_org = GeoMLP(MLPConfig(mv_channels=[3, 12, 4], s_channels=[2, 12, 5]))
    mlp_compiled = deepcopy(mlp_org)
    compile_equi_linear_submodules(mlp_compiled)

    assert any(isinstance(m, EquiLinear) for m in mlp_org.modules())
    assert not any(isinstance(m, CompiledLinear) for m in mlp_org.modules())
    assert not any(isinstance(m, EquiLinear) for m in mlp_compiled.modules())
    assert any(isinstance(m, CompiledLinear) for m in mlp_compiled.modules())

    in_mv = torch.randn(2, 4, 3, 16)
    in_s = torch.randn(2, 4, 2)
    reference_mv = torch.ones(16)

    out_mv_org, out_s_org = mlp_org(in_mv, in_s, reference_mv=reference_mv)
    out_mv_compiled, out_s_compiled = mlp_compiled(in_mv, in_s, reference_mv=reference_mv)

    torch.testing.assert_close(out_mv_org, out_mv_compiled)
    torch.testing.assert_close(out_s_org, out_s_compiled)


def test_compiled_linear_backward():
    """Test we can properly run a backwards pass."""
    mlp = GeoMLP(MLPConfig(mv_channels=[3, 12, 4], s_channels=[2, 12, 5]))
    compile_equi_linear_submodules(mlp)

    for _ in range(3):
        in_mv = torch.randn(2, 4, 3, 16).requires_grad_()
        in_s = torch.randn(2, 4, 2)
        reference_mv = torch.ones(16)

        out_mv, _ = mlp(in_mv, in_s, reference_mv=reference_mv)
        out_mv.sum().backward()
