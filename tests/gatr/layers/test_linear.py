# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
import pytest
import torch

from gatr.layers.linear import EquiLinear
from tests.helpers import BATCH_DIMS, TOLERANCES, check_pin_equivariance


@pytest.mark.parametrize("batch_dims", [(100,)])
@pytest.mark.parametrize("in_mv_channels, out_mv_channels", [(200, 5), (16, 16), (5, 200)])
@pytest.mark.parametrize(
    "in_s_channels, out_s_channels", [(None, None), (None, 100), (100, None), (32, 32)]
)
@pytest.mark.parametrize("initialization", ["default", "small", "unit_scalar"])
def test_linear_layer_initialization(
    initialization,
    batch_dims,
    in_mv_channels,
    out_mv_channels,
    in_s_channels,
    out_s_channels,
    var_tolerance=10.0,
):
    """Tests the initialization of `EquiLinear`.

    The goal is that independent of the channel size, inputs with variance 1 are mapped to outputs
    with, very roughly, variance 1.
    """

    # Create layer
    try:
        layer = EquiLinear(
            in_mv_channels,
            out_mv_channels,
            in_s_channels=in_s_channels,
            out_s_channels=out_s_channels,
            initialization=initialization,
        )
    # Some initialization schemes ar enot implemented when data is all-scalar. That's fine.
    except NotImplementedError as exc:
        print(exc)
        return

    # Inputs
    inputs_mv = torch.randn(*batch_dims, in_mv_channels, 16)
    inputs_s = torch.randn(*batch_dims, in_s_channels) if in_s_channels is not None else None

    # Compute outputs
    outputs_mv, outputs_s = layer(inputs_mv, scalars=inputs_s)

    # Compute mean and variance of MV outputs
    mv_mean = outputs_mv[...].cpu().detach().to(torch.float64).mean(dim=(0, 1))
    mv_var = outputs_mv[...].cpu().detach().to(torch.float64).var(dim=(0, 1))

    print("Output multivector means and std by components:")
    for i, (mean_, var_) in enumerate(zip(mv_mean, mv_var)):
        print(f"  Component {i}: mean = {mean_:.2f}, std = {var_**0.5:.2f}")

    # Check that the mean and variance agree with expectations
    if initialization == "default":
        target_mean = torch.zeros_like(mv_mean)
        target_var = torch.ones_like(mv_var) / 3.0  # Factor 3 comes from heuristics
    elif initialization == "small":
        target_mean = torch.zeros_like(mv_mean)
        target_var = 0.01 * torch.ones_like(mv_var) / 3.0
    elif initialization == "unit_scalar":
        target_mean = torch.zeros_like(mv_mean)
        target_mean[0] = 1.0
        target_var = 0.01 * torch.ones_like(mv_var) / 3.0
    else:
        raise ValueError(initialization)

    assert torch.all(mv_mean > target_mean - 0.3)
    assert torch.all(mv_mean < target_mean + 0.3)
    assert torch.all(mv_var > target_var / var_tolerance)
    assert torch.all(mv_var < target_var * var_tolerance)

    # Same for scalar outputs
    if out_s_channels is not None:
        s_mean = outputs_s[...].cpu().detach().to(torch.float64).mean().item()
        s_var = outputs_s[...].cpu().detach().to(torch.float64).var().item()

        print(f"Output scalar: mean = {s_mean:.2f}, std = {s_var**0.5:.2f}")
        assert -0.3 < s_mean < 0.3
        if initialization in {"default", "unit_scalar"}:
            assert 1.0 / 3.0 / var_tolerance < s_var < 1.0 / 3.0 * var_tolerance
        else:
            assert 0.01 / 3.0 / var_tolerance < s_var < 0.01 / 3.0 * var_tolerance


@pytest.mark.parametrize("rescaling", [0.0, -2.0, 100.0])
@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("in_mv_channels", [9, 1])
@pytest.mark.parametrize("out_mv_channels", [7, 1])
@pytest.mark.parametrize("in_s_channels", [None, 3])
@pytest.mark.parametrize("out_s_channels", [None, 4])
def test_linear_layer_linearity(
    batch_dims, in_mv_channels, out_mv_channels, in_s_channels, out_s_channels, rescaling
):
    """Tests that the EquiLinear layer indeed describes a linear map (when the bias is deactivated).

    Checks that `f(x + rescaling * y) = f(x) + rescaling * f(y)` for random inputs `x`, `y` and
    linear layer `f(x)`.
    """
    layer = EquiLinear(
        in_mv_channels,
        out_mv_channels,
        in_s_channels=in_s_channels,
        out_s_channels=out_s_channels,
        bias=False,
    )

    # Inputs
    x_mv = torch.randn(*batch_dims, in_mv_channels, 16)
    y_mv = torch.randn(*batch_dims, in_mv_channels, 16)
    xy_mv = x_mv + rescaling * y_mv

    if in_s_channels:
        x_s = torch.randn(*batch_dims, in_s_channels)
        y_s = torch.randn(*batch_dims, in_s_channels)
        xy_s = x_s + rescaling * y_s
    else:
        x_s, y_s, xy_s = None, None, None

    # Compute outputs
    o_xy_mv, o_xy_s = layer(xy_mv, scalars=xy_s)
    o_x_mv, o_x_s = layer(x_mv, scalars=x_s)
    o_y_mv, o_y_s = layer(y_mv, scalars=y_s)

    # Check equality
    torch.testing.assert_close(o_xy_mv, o_x_mv + rescaling * o_y_mv, **TOLERANCES)

    if out_s_channels is not None:
        torch.testing.assert_close(o_xy_s, o_x_s + rescaling * o_y_s, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("in_mv_channels", [9, 1])
@pytest.mark.parametrize("out_mv_channels", [7, 1])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("in_s_channels", [None, 3])
@pytest.mark.parametrize("out_s_channels", [None, 4])
def test_linear_layer_equivariance(
    batch_dims, in_mv_channels, out_mv_channels, in_s_channels, out_s_channels, bias
):
    """Tests the equi_linear() primitive for equivariance."""
    layer = EquiLinear(
        in_mv_channels,
        out_mv_channels,
        in_s_channels=in_s_channels,
        out_s_channels=out_s_channels,
        bias=bias,
    )
    data_dims = tuple(list(batch_dims) + [in_mv_channels])
    scalars = None if in_s_channels is None else torch.randn(*batch_dims, in_s_channels)
    check_pin_equivariance(
        layer, 1, fn_kwargs=dict(scalars=scalars), batch_dims=data_dims, **TOLERANCES
    )
