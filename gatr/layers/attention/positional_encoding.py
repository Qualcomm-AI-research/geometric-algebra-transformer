# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.

"""Adapted from the below.

https://github.com/EleutherAI/gpt-neox/blob/737c9134bfaff7b58217d61f6619f1dcca6c484f/megatron/model/positional_embeddings.py
by EleutherAI at https://github.com/EleutherAI/gpt-neox

Copyright (c) 2021, EleutherAI

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch


class ApplyRotaryPositionalEncoding(torch.nn.Module):
    """Applies rotary position encodings (RoPE) to scalar tensors.

    References
    ----------
    Jianlin Su et al, "RoFormer: Enhanced Transformer with Rotary Position Embedding",
        arXiv:2104.09864

    Parameters
    ----------
    num_channels : int
        Number of channels (key and query size).
    item_dim : int
        Embedding dimension. Should be even.
    base : int
        Determines the frequencies.
    """

    def __init__(self, num_channels, item_dim, base=4096):
        super().__init__()

        assert (
            num_channels % 2 == 0
        ), "Number of channels needs to be even for rotary position embeddings"

        inv_freq = 1.0 / (base ** (torch.arange(0, num_channels, 2).float() / num_channels))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.device_cached = None
        self.cos_cached = None
        self.sin_cached = None
        self.item_dim = item_dim
        self.num_channels = num_channels

    def forward(self, scalars: torch.Tensor) -> torch.Tensor:
        """Computes rotary embeddings along `self.item_dim` and applies them to inputs.

        The inputs are usually scalar queries and keys.

        Assumes that the last dimension is the feature dimension (and is thus not suited
        for multivector data!).

        Parameters
        ----------
        scalars : torch.Tensor of shape (..., num_channels)
            Input data. The last dimension is assumed to be the channel / feature dimension
            (NOT the 16 dimensions of a multivector).

        Returns
        -------
        outputs : torch.Tensor of shape (..., num_channels)
            Output data. Rotary positional embeddings applied to the input tensor.
        """

        # Check inputs
        assert scalars.shape[-1] == self.num_channels

        # Compute embeddings, if not already cached
        self._compute_embeddings(scalars)

        # Apply embeddings
        outputs = scalars * self.cos_cached + self._rotate_half(scalars) * self.sin_cached

        return outputs

    def _compute_embeddings(self, inputs):
        """Computes position embeddings and stores them.

        The position embedding is computed along dimension `item_dim` of tensor `inputs`
        and is stored in `self.sin_cached` and `self.cos_cached`.

        Parameters
        ----------
        inputs : torch.Tensor
            Input data.
        """
        seq_len = inputs.shape[self.item_dim]
        if seq_len != self.seq_len_cached or inputs.device != self.device_cached:
            self.seq_len_cached = seq_len
            self.device_cached = inputs.device
            t = torch.arange(inputs.shape[self.item_dim], device=inputs.device).type_as(
                self.inv_freq
            )
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(inputs.device)

            self.cos_cached = emb.cos()
            self.sin_cached = emb.sin()

            # Insert appropriate amount of dimensions such that the embedding correctly enumerates
            # along the item dim
            item_dim = (
                self.item_dim if self.item_dim >= 0 else inputs.ndim + self.item_dim
            )  # Deal with item_dim < 0
            for _ in range(item_dim + 1, inputs.ndim - 1):
                self.cos_cached = self.cos_cached.unsqueeze(1)
                self.sin_cached = self.sin_cached.unsqueeze(1)

    @staticmethod
    def _rotate_half(inputs):
        """Utility function that "rotates" a tensor, as required for rotary embeddings."""
        x1, x2 = inputs[..., : inputs.shape[-1] // 2], inputs[..., inputs.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
