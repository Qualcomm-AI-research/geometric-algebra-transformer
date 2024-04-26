# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
from itertools import product
from typing import Tuple

import torch

from gatr.primitives.bilinear import outer_product
from gatr.utils.einsum import gatr_cache, gatr_einsum
from gatr.utils.misc import minimum_autocast_precision

# Flag which reference join implementations we're using
_USE_EFFICIENT_JOIN = True


@gatr_cache
@torch.no_grad()
def _compute_dualization(
    device=torch.device("cpu"), dtype=torch.float32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Constructs a tensor for the dual operation.

    Parameters
    ----------
    device : torch.device
        Device
    dtype : torch.dtype
        Dtype

    Returns
    -------
    permutation : list of int
        Permutation index list to compute the dual
    factors : torch.Tensor
        Signs to multiply the dual outputs with.
    """
    permutation = [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    factors = torch.tensor(
        [1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1], device=device, dtype=dtype
    )
    return permutation, factors


@gatr_cache
@torch.no_grad()
def _compute_efficient_join(device=torch.device("cpu"), dtype=torch.float32) -> torch.Tensor:
    """Constructs a kernel for the join operation.

    The kernel is such that join(x, y)_i = einsum(kernel_ijk, x_j, x_k).

    For now, we do this in the simplest possible way: by computing the joins between two sets of
    basis vectors. (Since the join is bilinear, that should be enough.)

    Parameters
    ----------
    device : torch.device
        Device
    dtype : torch.dtype
        Dtype

    Returns
    -------
    kernel : torch.Tensor
        Joint kernel
    """

    kernel = torch.zeros((16, 16, 16), dtype=dtype, device=device)

    for i in range(16):
        for j in range(16):
            x, y = torch.zeros(16, dtype=dtype, device=device), torch.zeros(
                16, dtype=dtype, device=device
            )
            x[i] = 1.0
            y[j] = 1.0
            kernel[:, i, j] = dual(outer_product(dual(x), dual(y)))

    return kernel


@gatr_cache
@torch.no_grad()
def _compute_join_norm_idx(threshold=0.5) -> Tuple[list, list, list]:
    """Constructs everything we need to compute norm(equi_norm(x,y)) in a memory-efficient way.

    Parameters
    ----------
    threshold : float
        Threshold that determines discretization of join kernel

    Returns
    -------
    left_idx : list of list of int
        List (output components) of list (contributing terms) of indices of x
    left_idx : list of list of int
        List (output components) of list (contributing terms) of indices of y
    left_idx : list of list of float
        List (output components) of list (contributing terms) of indices of y
    """

    # Get join kernel K_ijk
    join_kernel = _compute_efficient_join()

    # Output components that contribute to the norm: all e_{i...k} without a 0 in the idx
    output_idx = [0, 2, 3, 4, 8, 9, 10, 14]

    # Identify input idx that contribute to those output idx, with the corresponding signs
    all_left_idx = []
    all_right_idx = []
    all_signs = []

    for i in output_idx:
        left_idx = []
        right_idx = []
        signs = []

        for j, k in product(range(16), repeat=2):
            if join_kernel[i, j, k] > threshold:
                left_idx.append(j)
                right_idx.append(k)
                signs.append(1.0)
            elif join_kernel[i, j, k] < -threshold:
                left_idx.append(j)
                right_idx.append(k)
                signs.append(-1.0)

        all_left_idx.append(left_idx)
        all_right_idx.append(right_idx)
        all_signs.append(signs)

    return all_left_idx, all_right_idx, all_signs


def dual(x: torch.Tensor) -> torch.Tensor:
    """Computes the dual of `inputs` (non-equivariant!).

    See Table 4 in the reference.

    References
    ----------
    Leo Dorst, "A Guided Tour to the Plane-Based Geometric Algebra PGA",
        https://geometricalgebra.org/downloads/PGA4CS.pdf

    Parameters
    ----------
    x : torch.Tensor with shape (..., 16)
        Input multivector, of which we want to compute the dual.

    Returns
    -------
    outputs : torch.Tensor with shale (..., 16)
        The dual of `inputs`, using the pseudoscalar component of `reference` as basis.
    """

    # Select factors on correct device
    perm, factors = _compute_dualization(x.device, x.dtype)

    # Compute dual
    result = factors * x[..., perm]

    return result


def equivariant_join(x: torch.Tensor, y: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Computes the equivariant join.

    ```
    equivariant_join(x, y; reference) = reference_123 * dual( dual(x) ^ dual(y) )
    ```

    This function uses either explicit_equivariant_join or efficient_equivariant_join, depending
    on whether _USE_EFFICIENT_JOIN is set.

    Parameters
    ----------
    x : torch.Tensor
        Left input multivector.
    y : torch.Tensor
        Right input multivector.
    reference : torch.Tensor
        Reference multivector to break the orientation ambiguity.

    Returns
    -------
    outputs : torch.Tensor
        Equivariant join result.
    """

    if _USE_EFFICIENT_JOIN:
        return efficient_equivariant_join(x, y, reference)

    return explicit_equivariant_join(x, y, reference)


def explicit_equivariant_join(
    x: torch.Tensor, y: torch.Tensor, reference: torch.Tensor
) -> torch.Tensor:
    """Computes the equivariant join, using the explicit, but slow, implementation.

    ```
    equivariant_join(x, y; reference) = reference_123 * dual( dual(x) ^ dual(y) )
    ```

    Parameters
    ----------
    x : torch.Tensor
        Left input multivector.
    y : torch.Tensor
        Right input multivector.
    reference : torch.Tensor
        Reference multivector to break the orientation ambiguity.

    Returns
    -------
    outputs : torch.Tensor
        Rquivariant join result.
    """
    return reference[..., [14]] * dual(outer_product(dual(x), dual(y)))


def efficient_equivariant_join(
    x: torch.Tensor, y: torch.Tensor, reference: torch.Tensor
) -> torch.Tensor:
    """Computes the equivariant join, using the efficient implementation.

    ```
    equivariant_join(x, y; reference) = reference_123 * dual( dual(x) ^ dual(y) )
    ```

    Parameters
    ----------
    x : torch.Tensor
        Left input multivector.
    y : torch.Tensor
        Right input multivector.
    reference : torch.Tensor
        Reference multivector to break the orientation ambiguity.

    Returns
    -------
    outputs : torch.Tensor
        Rquivariant join result.
    """

    kernel = _compute_efficient_join(x.device, x.dtype)
    return reference[..., [14]] * gatr_einsum("i j k , ... j, ... k -> ... i", kernel, x, y)


@minimum_autocast_precision(torch.float32)
def join_norm(
    x: torch.Tensor, y: torch.Tensor, square=False, channel_sum=False, channel_weights=None
) -> torch.Tensor:
    """Computes the norm of the join, `|join(x,y)|`, in a single operation.

    Optionally:
    - computes the squared norm instead of the norm (when `square = True`),
    - sums over channels, meaning the second-to-last dimension (when `channel_sum = True`)

    Parameters
    ----------
    x : torch.Tensor with shape (..., 16)
        Left input
    y : torch.Tensor with shape (..., 16)
        Right input
    square : bool
        If True, computes the squared norm rather than the norm
    channel_sum : bool
        If True, sums the result over channels (before taking the square root). We assume channels
        correspond to the second-to-last dimension of the input tensors.
    channel_weights : None torch.Tensor with shape (..., num_channels)
        If channel_sum is True, a non-None value of channel_weights weighs the different channels
        before summing over them. (Note that we do not perform any normalization here.)

    Returns
    -------
    norm : torch.Tensor with shape (..., 1)
        Norm of join
    """

    # Prepare computation
    output = 0.0
    all_left_idx, all_right_idx, all_signs = _compute_join_norm_idx()

    # Sum over all contributing terms
    for left_idx, right_idx, signs in zip(all_left_idx, all_right_idx, all_signs):
        # Compute contribution
        component = 0.0
        for j, k, sign in zip(left_idx, right_idx, signs):
            component = component + sign * x[..., j] * y[..., k]
        component = component**2

        # Compute channel sum if desired
        if channel_sum:
            if channel_weights is not None:
                component = component * channel_weights
            component = component.sum(dim=-1)

        output = output + component

    # Square root, unless the square norm is computed
    if not square:
        output = torch.sqrt(output)

    return output.unsqueeze(-1)
