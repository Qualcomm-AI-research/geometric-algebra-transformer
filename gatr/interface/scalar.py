# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
import torch


def embed_scalar(scalars: torch.Tensor) -> torch.Tensor:
    """Embeds a scalar tensor into multivectors.

    Parameters
    ----------
    scalars: torch.Tensor with shape (..., 1)
        Scalar inputs.

    Returns
    -------
    multivectors: torch.Tensor with shape (..., 16)
        Multivector outputs. `multivectors[..., [0]]` is the same as `scalars`. The other components
        are zero.
    """

    non_scalar_shape = list(scalars.shape[:-1]) + [15]
    non_scalar_components = torch.zeros(
        non_scalar_shape, device=scalars.device, dtype=scalars.dtype
    )
    embedding = torch.cat((scalars, non_scalar_components), dim=-1)

    return embedding


def extract_scalar(multivectors: torch.Tensor) -> torch.Tensor:
    """Extracts scalar components from multivectors.

    Parameters
    ----------
    multivectors: torch.Tensor with shape (..., 16)
        Multivector inputs.

    Returns
    -------
    scalars: torch.Tensor with shape (..., 1)
        Scalar component of multivectors.
    """

    return multivectors[..., [0]]
