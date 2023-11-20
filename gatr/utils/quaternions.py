# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.

"""Adapted from the below.

https://github.com/pimdh/lie-vae/blob/c473d91c494b9e80f6ac2998b1b6b927d09df259/lie_vae/lie_tools.py
by Luca Falorsi, Pim de Haan, Tim Davidson at https://github.com/pimdh/lie-vae

MIT License

Copyright (c) 2020 Pim de Haan, Luca Falorsi, Tim Davidson

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch


def random_quaternion(batch_dims):
    """Returns a unit quaternion, uniformly sampled from the 3-sphere."""
    q = torch.randn(*batch_dims, 4)
    return q / torch.norm(q, dim=-1, keepdim=True)


def quaternion_to_rotation_matrix(q):
    """Normalises q and maps to group matrix.

    Assumes [x y z w] quaternion format with ij=k (Hamilton) convention.
    """
    q = q[..., [3, 0, 1, 2]]  # Convert to [w x y z] order

    q = q / q.norm(p=2, dim=-1, keepdim=True)
    r, i, j, k = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    mat = torch.stack(
        [
            r * r + i * i - j * j - k * k,
            2 * (i * j - k * r),
            2 * (i * k + j * r),
            2 * (i * j + k * r),
            r * r - i * i + j * j - k * k,
            2 * (j * k - i * r),
            2 * (i * k - j * r),
            2 * (j * k + i * r),
            r * r - i * i - j * j + k * k,
        ],
        -1,
    )

    return mat.view(*q.shape[:-1], 3, 3)
