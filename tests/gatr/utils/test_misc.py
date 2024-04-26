# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
import pytest
import torch
from torch import Tensor

from gatr.utils.misc import minimum_autocast_precision


# Choose dtypes to work on most devices -- torch.bfloat16 is not available on some GPUs
@pytest.mark.parametrize("device,amp_dtype", [("cpu", torch.bfloat16), ("cuda", torch.float16)])
def test_minimum_autocast_precision_inputs(device, amp_dtype):
    """Tests that minimum_autocast_precision() casts inputs correctly"""

    # We wrap a function that just returns the dtypes of the inputs
    @minimum_autocast_precision(torch.float32)
    def return_input_dtypes(*args, **kwargs):
        dtypes = [arg.dtype if isinstance(arg, Tensor) else None for arg in args]
        dtypes += [arg.dtype if isinstance(arg, Tensor) else None for arg in kwargs.values()]
        return dtypes

    # Inputs
    input_dtypes = [
        amp_dtype,
        torch.float32,
        torch.float64,
        torch.int8,
        torch.int32,
        torch.bool,
        None,
    ]
    inputs = [
        "banana" if dtype is None else torch.empty(3, 5, device=device, dtype=dtype)
        for dtype in input_dtypes
    ]
    expected_dtypes = [
        torch.float32,
        torch.float32,
        torch.float64,
        torch.int8,
        torch.int32,
        torch.bool,
        None,
    ]

    # Test that without autocast, nothing happens
    dtypes0 = return_input_dtypes(*inputs)
    for got, expected in zip(dtypes0, input_dtypes):
        assert got == expected

    # Test that when autocasting, inputs are correctly casted
    with torch.autocast(device, amp_dtype, enabled=True):
        dtypes1 = return_input_dtypes(*inputs)

    for got, expected in zip(dtypes1, expected_dtypes):
        assert got == expected


# Choose dtypes to work on most devices -- torch.bfloat16 is not available on some GPUs
@pytest.mark.parametrize(
    "output_mode,expected_dtype",
    [
        (None, torch.float64),
        (torch.float64, torch.float64),
        ("low", torch.bfloat16),
        ("high", torch.float64),
    ],
)
@pytest.mark.parametrize("compile_code", [False, True])
def test_minimum_autocast_precision_outputs(
    output_mode, expected_dtype, compile_code, device="cpu", amp_dtype=torch.bfloat16
):
    """Tests that minimum_autocast_precision() casts outputs correctly"""

    # We wrap a function that just returns the dtypes of the inputs
    @torch.compile(disable=not compile_code, fullgraph=True)
    @minimum_autocast_precision(torch.float32, output=output_mode)
    def sum_(*args):
        outputs = 0.0
        for arg in args:
            outputs = outputs + arg
        return outputs

    # Inputs
    input_dtypes = [torch.bfloat16, torch.float32, torch.float64]
    inputs = [torch.randn((3, 5), device=device, dtype=dtype) for dtype in input_dtypes]

    # Check output dtype
    with torch.autocast(device, amp_dtype, enabled=True):
        outputs = sum_(*inputs)
    assert outputs.dtype == expected_dtype
