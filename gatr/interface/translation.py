# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All rights reserved.
import warnings

import torch

from gatr.utils.warning import GATrDeprecationWarning


def embed_translation(translation_vector: torch.Tensor) -> torch.Tensor:
    """Embeds a 3D translation in multivectors.

    In our convention, a translation vector is embedded into a combination of the scalar and
    bivector components.

    We have (in agreement with Eq. (82) of the reference below) that
    ```
    T(t) = 1 - e_0 / 2 (t_1 e_1 + t_2 e_2 + t_3 e_3) .
    ```

    References
    ----------
    Leo Dorst, "A Guided Tour to the Plane-Based Geometric Algebra PGA",
    https://geometricalgebra.org/downloads/PGA4CS.pdf

    Parameters
    ----------
    translation_vector : torch.Tensor with shape (..., 3)
        Vectorial amount of translation.

    Returns
    -------
    multivector : torch.Tensor with shape (..., 16)
        Embedding into multivector.
    """

    # Create multivector tensor with same batch shape, same device, same dtype as input
    batch_shape = translation_vector.shape[:-1]
    multivector = torch.zeros(
        *batch_shape, 16, dtype=translation_vector.dtype, device=translation_vector.device
    )

    # Embedding into trivectors
    multivector[..., 0] = 1.0  # scalar
    multivector[..., 5:8] = (
        -0.5 * translation_vector[..., :]
    )  # Translation vector embedded in x_0i with i = 1, 2, 3

    return multivector


def extract_translation(
    multivector: torch.Tensor, divide_by_embedding_dim=False, threshold: float = 1e-3
) -> torch.Tensor:
    """DEPRECATED: Given a multivector, extract 3D translation vector from the bivector components.

    Note bene: this function is NOT equivariant, unless the input has no rotational e_{ij}.
    Hence this function is deprecated and should NOT be used, unless you know what you're doing.
    If, given a multivector, you want to extract a vector that rotates, but is invariant to
    translations, use "extract_oriented_plane".

    References
    ----------
    Leo Dorst, "A Guided Tour to the Plane-Based Geometric Algebra PGA",
    https://geometricalgebra.org/downloads/PGA4CS.pdf

    Parameters
    ----------
    multivector : torch.Tensor with shape (..., 16)
        Multivector.
    divide_by_embedding_dim : bool
        Whether to divice by the embedding dim. Proper PGA etiquette would have us do this, but it
        may not be good for NN training.
    threshold : float
        Minimum value of the additional, unphysical component. Necessary to avoid exploding values
        or NaNs when this unphysical component of the homogeneous coordinates becomes small.

    Returns
    -------
    translation : torch.Tensor with shape (..., 3)
        3D components of the translation vector.
    """
    warnings.warn(
        'The function "extract_translation" is deprecated, because it is not equivariant.',
        GATrDeprecationWarning,
        2,
    )

    translation_vector = -2.0 * multivector[..., 5:8]

    # Divide by embedding dim
    if divide_by_embedding_dim:
        embedding_dim = multivector[..., [0]]
        embedding_dim = torch.where(torch.abs(embedding_dim) > threshold, embedding_dim, threshold)
        translation_vector = translation_vector / embedding_dim

    return translation_vector
