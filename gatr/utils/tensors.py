# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
from typing import Union

import torch
from torch import Tensor


def expand_pairwise(*tensors, exclude_dims=()):
    """Expand tensors to largest, optionally excluding some axes."""
    max_dim = max(t.dim() for t in tensors)
    shapes = [(1,) * (max_dim - t.dim()) + t.shape for t in tensors]
    max_shape = [max(s[d] for s in shapes) for d in range(max_dim)]
    for d in exclude_dims:
        max_shape[d] = -1
    return tuple(t.expand(tuple(max_shape)) for t in tensors)


def to_nd(tensor, d):
    """Make tensor n-dimensional, group extra dimensions in first."""
    return tensor.view(-1, *(1,) * (max(0, d - 1 - tensor.dim())), *tensor.shape[-(d - 1) :])


def assert_equal(vals):
    """Assert all values in sequence are equal."""
    for v in vals:
        assert v == vals[0]


def block_stack(tensors, dim1, dim2):
    """Block diagonally stack tensors along dimensions dim1 and dim2."""
    assert_equal([t.dim() for t in tensors])
    shapes = [t.shape for t in tensors]
    shapes_t = list(map(list, zip(*shapes)))
    for i, ss in enumerate(shapes_t):
        if i not in (dim1, dim2):
            assert_equal(ss)

    dim2_len = sum(shapes_t[dim2])
    opts = dict(device=tensors[0].device, dtype=tensors[0].dtype)

    padded_tensors = []
    offset = 0
    for tensor in tensors:
        before_shape = list(tensor.shape)
        before_shape[dim2] = offset
        after_shape = list(tensor.shape)
        after_shape[dim2] = dim2_len - tensor.shape[dim2] - offset
        before = torch.zeros(*before_shape, **opts)
        after = torch.zeros(*after_shape, **opts)
        padded = torch.cat([before, tensor, after], dim2)
        padded_tensors.append(padded)
        offset += tensor.shape[dim2]
    return torch.cat(padded_tensors, dim1)


def construct_reference_multivector(reference: Union[Tensor, str], inputs: Tensor) -> Tensor:
    """Constructs a reference vector for the equivariant join.

    Parameters
    ----------
    reference : Tensor with shape (..., 16) or {"data", "canonical"}
        Reference multivector for the equivariant joint operation. If "data", a
        reference multivector is constructed from the mean of the input multivectors. If
        "canonical", a constant canonical reference multivector is used instead.
    inputs : Tensor with shape (..., num_items_1, num_items_2, in_mv_channels, 16)
        Input multivectors.

    Returns
    -------
    reference_mv : Tensor with shape (..., 16)
        Reference multivector for the equivariant join.

    Raises
    ------
    ValueError
        If `reference` is neither "data", "canonical", nor a Tensor.
    """

    if reference == "data":
        # When using torch-geometric-style batching, this code should be adapted to perform the
        # mean over the items in each batch, but not over the batch dimension.
        # We leave this as an exercise for the practitioner :)
        mean_dim = tuple(range(1, len(inputs.shape) - 1))
        reference_mv = torch.mean(inputs, dim=mean_dim, keepdim=True)  # (batch, 1, ..., 1, 16)
    elif reference == "canonical":
        reference_mv = torch.zeros(16, device=inputs.device, dtype=inputs.dtype)
        reference_mv[..., [14, 15]] = 1.0
    else:
        if not isinstance(reference, Tensor):
            raise ValueError(
                'Reference needs to be "data", "canonical", or torch.Tensor, but found {reference}'
            )
        reference_mv = reference

    return reference_mv
