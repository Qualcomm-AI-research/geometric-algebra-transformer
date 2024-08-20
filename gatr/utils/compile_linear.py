#!/usr/bin/env python3
# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All rights reserved.

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from gatr.layers.linear import EquiLinear
from gatr.primitives.linear import _compute_pin_equi_linear_basis


class CompiledLinear(nn.Module):
    """A drop-in replacement for EquiLinear for fast inference.

    Not trainable. Parameters must be set by hand.

    Parameters
    ----------
    in_mv_channels : int
        Input multivector channels
    out_mv_channels : int
        Output multivector channels
    bias : bool
        Whether a bias term is added to the scalar component of the multivector outputs
    in_s_channels : int or None
        Input scalar channels. If None, no scalars are expected nor returned.
    out_s_channels : int or None
        Output scalar channels. If None, no scalars are expected nor returned.
    """

    def __init__(
        self,
        in_mv_channels: int,
        out_mv_channels: int,
        in_s_channels: Optional[int] = None,
        out_s_channels: Optional[int] = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        in_channels = 16 * in_mv_channels + (in_s_channels or 0)
        out_channels = 16 * out_mv_channels + (out_s_channels or 0)
        self.register_buffer("weight", torch.zeros(out_channels, in_channels))
        self.register_buffer("bias", torch.zeros(out_channels) if bias else None)
        self._out_mv_channels = out_mv_channels
        self._out_s_channels = out_s_channels

    def forward(
        self, multivectors: torch.Tensor, scalars: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """Compute compiled linear map.

        Parameters
        ----------
        multivectors : torch.Tensor with shape (..., in_mv_channels, 16)
            Input multivectors
        scalars : None or torch.Tensor with shape (..., in_s_channels)
            Optional input scalars

        Returns
        -------
        outputs_mv : torch.Tensor with shape (..., out_mv_channels, 16)
            Output multivectors
        outputs_s : None or torch.Tensor with shape (..., out_s_channels)
            Output scalars, if scalars are provided. Otherwise None.
        """
        b = multivectors.shape[:-2]
        inputs = multivectors.view(*b, -1)
        if scalars is not None:
            inputs = torch.cat([inputs, scalars], -1)
        outputs = F.linear(inputs, self.weight, self.bias)  # pylint: disable=not-callable
        outputs_mv = outputs[..., : 16 * self._out_mv_channels].view(*b, -1, 16)
        if self._out_s_channels is None:
            outputs_s = None
        else:
            outputs_s = outputs[..., 16 * self._out_mv_channels :]

        return outputs_mv, outputs_s


def compile_equi_linear(equi_linear: EquiLinear) -> CompiledLinear:
    """Transform EquiLinear module to equivalent CompiledLinear module.

    Used for fast inference, not trainable!

    Parameters
    ----------
    equi_linear : EquiLinear

    Returns
    -------
    CompiledLinear
    """
    # Figure out hyperparameters of EquiLinear from parameter shapes
    out_mv_channels, in_mv_channels, _ = equi_linear.weight.shape
    in_s_channels = equi_linear.s2mvs.weight.shape[1] if equi_linear.s2mvs else None
    out_s_channels = equi_linear.mvs2s.weight.shape[0] if equi_linear.mvs2s else None
    if in_s_channels is None:
        bias = equi_linear.bias is not None
    else:
        bias = equi_linear.s2mvs.bias is not None

    compiled = CompiledLinear(
        in_mv_channels, out_mv_channels, in_s_channels, out_s_channels, bias=bias
    )
    compiled.to(device=equi_linear.weight.device, dtype=equi_linear.weight.dtype)

    # The number of channels the multivectors occupy
    offset_in = in_mv_channels * 16
    offset_out = out_mv_channels * 16

    # Materialize the equivariant maps by linearly combining the basis maps
    basis = _compute_pin_equi_linear_basis(equi_linear.weight.device, equi_linear.weight.dtype)
    mv_mv_weight = torch.einsum("y x a, a i j -> y i x j", equi_linear.weight, basis)
    compiled.weight.data[:offset_out, :offset_in] = mv_mv_weight.reshape(offset_out, offset_in)
    # We need to modify .data directly for the compiled linear maps to still be autograd-compatible
    # (in the sense of differentiating wrt their inputs, not wrt the linear parameters).

    if in_s_channels:
        compiled.weight.data[:offset_out:16, offset_in:] = equi_linear.s2mvs.weight
    if out_s_channels:
        compiled.weight.data[offset_out:, :offset_in:16] = equi_linear.mvs2s.weight
    if in_s_channels and out_s_channels:
        compiled.weight.data[offset_out:, offset_in:] = equi_linear.s2s.weight

    if bias:
        if equi_linear.bias is not None:
            compiled.bias.data[:offset_out:16] = equi_linear.bias.flatten()
        elif equi_linear.s2mvs.bias is not None:
            compiled.bias.data[:offset_out:16] = equi_linear.s2mvs.bias
        if out_s_channels:
            compiled.bias.data[offset_out:] = equi_linear.mvs2s.bias

    return compiled


def compile_equi_linear_submodules(model: nn.Module):
    """Replace all EquiLinear submodules in module with compiled version.

    Parameters
    ----------
    model : nn.Module
        Model whose EquiLinear submodules will be replaced. Argument mutated!
    """

    def walker(module):
        for name, submodule in module.named_children():
            if isinstance(submodule, EquiLinear):
                compiled_linear = compile_equi_linear(submodule)
                setattr(module, name, compiled_linear)

    model.apply(walker)
