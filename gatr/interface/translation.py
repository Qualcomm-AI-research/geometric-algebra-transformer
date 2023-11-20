# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
import torch


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
    """Given a multivector, extract a 3D translation vector from the bivector components.

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

    translation_vector = -2.0 * multivector[..., 5:8]

    # Divide by embedding dim
    if divide_by_embedding_dim:
        embedding_dim = multivector[..., [0]]
        embedding_dim = torch.where(torch.abs(embedding_dim) > threshold, embedding_dim, threshold)
        translation_vector = translation_vector / embedding_dim

    return translation_vector
