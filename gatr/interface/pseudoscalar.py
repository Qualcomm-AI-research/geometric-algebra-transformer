# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All rights reserved.
import torch


def embed_pseudoscalar(pseudoscalars: torch.Tensor) -> torch.Tensor:
    """Embeds a pseudoscalar tensor into multivectors.

    Parameters
    ----------
    pseudoscalars: torch.Tensor with shape (..., 1)
        Pseudoscalar inputs.

    Returns
    -------
    multivectors: torch.Tensor with shape (..., 16)
        Multivector outputs. `multivectors[..., [15]]` is the same as `pseudoscalars`.
        The other components are zero.
    """

    non_scalar_shape = list(pseudoscalars.shape[:-1]) + [15]
    non_scalar_components = torch.zeros(
        non_scalar_shape, device=pseudoscalars.device, dtype=pseudoscalars.dtype
    )
    embedding = torch.cat((non_scalar_components, pseudoscalars), dim=-1)

    return embedding


def extract_pseudoscalar(multivectors: torch.Tensor) -> torch.Tensor:
    """Extracts pseudoscalar components from multivectors.

    Nota bene: when the output is interpreted as a scalar,
    this function is only equivariant to SE(3), but not to mirrors.

    Parameters
    ----------
    multivectors: torch.Tensor with shape (..., 16)
        Multivector inputs.

    Returns
    -------
    pseudoscalars: torch.Tensor with shape (..., 1)
        Pseudoscalar component of multivectors.
    """

    return multivectors[..., [15]]
