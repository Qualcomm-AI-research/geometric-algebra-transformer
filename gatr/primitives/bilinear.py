# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
from pathlib import Path

import torch

from gatr.utils.einsum import gatr_cache, gatr_einsum

_FILENAMES = {"gp": "geometric_product.pt", "outer": "outer_product.pt"}


@gatr_cache
def _load_bilinear_basis(
    kind: str, device=torch.device("cpu"), dtype=torch.float32
) -> torch.Tensor:
    """Loads basis elements for Pin-equivariant bilinear maps between multivectors.

    Parameters
    ----------
    kind : {"gp", "outer"}
        Filename of the basis file, assumed to be found in __file__ / data
    device : torch.Device or str
        Device
    dtype : torch.Dtype
        Data type

    Returns
    -------
    basis : torch.Tensor with shape (num_basis_elements, 16, 16, 16)
        Basis elements for bilinear equivariant maps between multivectors.
    """

    # To avoid duplicate loading, base everything on float32 CPU version
    if device not in [torch.device("cpu"), "cpu"] and dtype != torch.float32:
        basis = _load_bilinear_basis(kind)
    else:
        filename = Path(__file__).parent.resolve() / "data" / _FILENAMES[kind]
        sparse_basis = torch.load(filename).to(torch.float32)
        # Convert to dense tensor
        # The reason we do that is that einsum is not defined for sparse tensors
        basis = sparse_basis.to_dense()

    return basis.to(device=device, dtype=dtype)


def geometric_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes the geometric product f(x,y) = xy.

    Parameters
    ----------
    x : torch.Tensor with shape (..., 16)
        First input multivector. Batch dimensions must be broadcastable between x and y.
    y : torch.Tensor with shape (..., 16)
        Second input multivector. Batch dimensions must be broadcastable between x and y.

    Returns
    -------
    outputs : torch.Tensor with shape (..., 16)
        Result. Batch dimensions are result of broadcasting between x, y, and coeffs.
    """

    # Select kernel on correct device
    gp = _load_bilinear_basis("gp", x.device, x.dtype)

    # Compute geometric product
    outputs = gatr_einsum("i j k, ... j, ... k -> ... i", gp, x, y)

    return outputs


def outer_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes the outer product `f(x,y) = x ^ y`.

    Parameters
    ----------
    x : torch.Tensor with shape (..., 16)
        First input multivector. Batch dimensions must be broadcastable between x and y.
    y : torch.Tensor with shape (..., 16)
        Second input multivector. Batch dimensions must be broadcastable between x and y.

    Returns
    -------
    outputs : torch.Tensor with shape (..., 16)
        Result. Batch dimensions are result of broadcasting between x, y, and coeffs.
    """

    # Select kernel on correct device
    op = _load_bilinear_basis("outer", x.device, x.dtype)

    # Compute geometric product
    outputs = gatr_einsum("i j k, ... j, ... k -> ... i", op, x, y)

    return outputs
