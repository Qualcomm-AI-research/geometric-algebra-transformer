# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
"""Utility functions to test callables for equivariance with respect to Pin(3,0,1)."""

import torch

from gatr.utils.clifford import SlowRandomPinTransform


def get_first_output(outputs):
    """Extracts the first output of a tuple of multiple outputs.

    If only one output is present, returns that.

    This is convenient for equivariance checks: primitives usually just return one multivector
    tensor, while layers return two tensors. This function can be wrapped around a callable that's
    either a primitive or a layer and will always return the (first) multivector output.
    """
    if isinstance(outputs, tuple):
        return outputs[0]

    return outputs


def check_pin_equivariance(
    function,
    num_multivector_args=1,
    fn_kwargs=None,
    batch_dims=(1,),
    spin=False,
    rng=None,
    num_checks=3,
    **kwargs,
):
    """Checks whether a callable is equivariant with respect to the Pin(3,0,1) or Spin(3,0,1) group.

    The callable can have an arbitray number of multivector inputs.

    Parameters
    ----------
    function: Callable
        Function to be tested for equivariance. The first `num_multivector_args` positional
        arguments need to accept torch.Tensor inputs describing multivectors, and will be
        transformed as part of the equivariance test.
    num_multivector_args: int
        Number of multivector that `function` accepts.
    fn_kwargs : dict with str keys
        Keyword arguments to call `function` with.
    batch_dims : tuple of int
        Batch shape for the multivector inputs to `function`.
    spin : bool
        If True, this function tests Spin equivariance; if False, it tests Pin equivariance.
    rng : numpy.random.Generator or None
        Numpy rng to draw the inputs and transformations from.
    num_checks : int
        Number of function calls (with random inputs) to determine whether the function passes the
        equivariance test.
    kwargs
        Optional keyword arguments for the equality check. Will be passed on to np.allclose.
        This can for instance be used to specify the absolute and relative tolerance
        (by passing `atol` and `rtol` keyword arguments).
    """
    # Default arguments
    if fn_kwargs is None:
        fn_kwargs = {}

    # Propagate numpy random state to torch
    if rng is not None:
        torch.manual_seed(rng.integers(100000))

    # Loop over multiple checks
    for _ in range(num_checks):
        # Generate function inputs and Pin(3,0,1) transformations
        inputs = torch.randn(num_multivector_args, *batch_dims, 16)
        transform = SlowRandomPinTransform(rng=rng, spin=spin)

        # First function, then transformation
        outputs = get_first_output(function(*inputs, **fn_kwargs))
        transformed_outputs = transform(outputs)

        # First transformation, then function
        transformed_inputs = transform(inputs)
        outputs_of_transformed = get_first_output(function(*transformed_inputs, **fn_kwargs))

        # Check equality
        torch.testing.assert_close(transformed_outputs, outputs_of_transformed, **kwargs)


def check_pin_invariance(
    function,
    num_multivector_args=1,
    fn_kwargs=None,
    batch_dims=(1,),
    spin=False,
    rng=None,
    num_checks=3,
    **kwargs,
):
    """Checks whether a callable is invariant with respect to the Pin(3,0,1) or Spin(3,0,1) group.

    Parameters
    ----------
    function: Callable
        Function to be tested for equivariance. The first `num_multivector_args` positional
        arguments need to accept torch.Tensor inputs describing multivectors, and will be
        transformed as part of the invariance test.
    num_multivector_args: int
        Number of multivector that `function` accepts.
    fn_kwargs : dict with str keys
        Keyword arguments to call `function` with.
    batch_dims : tuple of int
        Batch shape for the multivector inputs to `function`.
    spin : bool
        If True, this function tests Spin equivariance; if False, it tests Pin equivariance.
        Since Spin is a subgroup of Pin, it is usually enough to confirm Pin equivariance.
    rng : numpy.random.Generator or None
        Numpy rng to draw the inputs and transformations from.
    num_checks : int
        Number of function calls (with random inputs) to determine whether the function passes the
        equivariance test.
    kwargs
        Optional keyword arguments for the equality check. Will be passed on to np.allclose.
        This can for instance be used to specify the absolute and relative tolerance
        (by passing `atol` and `rtol` keyword arguments).
    """
    # Default arguments
    if fn_kwargs is None:
        fn_kwargs = {}

    # Propagate numpy random state to torch
    if rng is not None:
        torch.manual_seed(rng.integers(100000))

    # Loop over multiple checks
    for _ in range(num_checks):
        # Generate function inputs
        inputs = torch.randn(num_multivector_args, *batch_dims, 16)

        # Transform inputs with Pin(3,0,1)
        transform = SlowRandomPinTransform(rng=rng, spin=spin)
        transformed_inputs = transform(inputs)

        # Evaluate function on original and transformed inputs
        outputs = get_first_output(function(*inputs, **fn_kwargs))
        outputs_of_transformed = get_first_output(function(*transformed_inputs, **fn_kwargs))

        # Check equality
        torch.testing.assert_close(outputs, outputs_of_transformed, **kwargs)
