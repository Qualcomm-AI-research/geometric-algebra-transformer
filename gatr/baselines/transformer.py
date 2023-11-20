# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
"""Baseline transformer."""

from functools import partial
from typing import Optional, Tuple

import torch
from einops import rearrange
from torch import nn
from torch.utils.checkpoint import checkpoint

from gatr.layers import ApplyRotaryPositionalEncoding
from gatr.primitives.attention import scaled_dot_product_attention
from gatr.utils.tensors import to_nd


class BaselineLayerNorm(nn.Module):
    """Baseline layer norm over all dimensions except the first."""

    @staticmethod
    def forward(inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        inputs : Tensor
            Input data

        Returns
        -------
        outputs : Tensor
            Normalized inputs.
        """
        return torch.nn.functional.layer_norm(inputs, normalized_shape=inputs.shape[1:])


class MultiHeadQKVLinear(nn.Module):
    """Compute queries, keys, and values via multi-head attention.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    hidden_channels : int
        Number of hidden channels = size of query, key, and value.
    num_heads : int
        Number of attention heads.
    """

    def __init__(self, in_channels, hidden_channels, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.linear = nn.Linear(in_channels, 3 * hidden_channels * num_heads)

    def forward(self, inputs):
        """Forward pass.

        Returns
        -------
        q : Tensor
            Queries
        k : Tensor
            Keys
        v : Tensor
            Values
        """
        qkv = self.linear(inputs)  # (..., num_items, 3 * hidden_channels * num_heads)
        q, k, v = rearrange(
            qkv,
            "... items (qkv hidden_channels num_heads) -> qkv ... num_heads items hidden_channels",
            num_heads=self.num_heads,
            qkv=3,
        )
        return q, k, v


class MultiQueryQKVLinear(nn.Module):
    """Compute queries, keys, and values via multi-query attention.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    hidden_channels : int
        Number of hidden channels = size of query, key, and value.
    num_heads : int
        Number of attention heads.
    """

    def __init__(self, in_channels, hidden_channels, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.q_linear = nn.Linear(in_channels, hidden_channels * num_heads)
        self.k_linear = nn.Linear(in_channels, hidden_channels)
        self.v_linear = nn.Linear(in_channels, hidden_channels)

    def forward(self, inputs):
        """Forward pass.

        Parameters
        ----------
        inputs : Tensor
            Input data

        Returns
        -------
        q : Tensor
            Queries
        k : Tensor
            Keys
        v : Tensor
            Values
        """
        q = rearrange(
            self.q_linear(inputs),
            "... items (hidden_channels num_heads) -> ... num_heads items hidden_channels",
            num_heads=self.num_heads,
        )
        k = self.k_linear(inputs)[..., None, :, :]  # (..., head=1, item, hidden_channels)
        v = self.v_linear(inputs)[..., None, :, :]
        return q, k, v


class BaselineSelfAttention(nn.Module):
    """Baseline self-attention layer.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of input channels.
    hidden_channels : int
        Number of hidden channels = size of query, key, and value.
    num_heads : int
        Number of attention heads.
    pos_encoding : bool
        Whether to apply rotary positional embeddings along the item dimension to the scalar keys
        and queries.
    pos_enc_base : int
        Maximum frequency used in positional encodings. (The minimum frequency is always 1.)
    multi_query : bool
        Use multi-query attention instead of multi-head attention.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        num_heads: int = 8,
        pos_encoding: bool = False,
        pos_enc_base: int = 4096,
        multi_query: bool = True,
    ) -> None:
        super().__init__()

        # Store settings
        self.num_heads = num_heads
        self.hidden_channels = hidden_channels

        # Linear maps
        qkv_class = MultiQueryQKVLinear if multi_query else MultiHeadQKVLinear
        self.qkv_linear = qkv_class(in_channels, hidden_channels, num_heads)
        self.out_linear = nn.Linear(hidden_channels * num_heads, out_channels)

        # Optional positional encoding
        if pos_encoding:
            self.pos_encoding = ApplyRotaryPositionalEncoding(
                hidden_channels, item_dim=-2, base=pos_enc_base
            )
        else:
            self.pos_encoding = None

    def forward(
        self, inputs: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        inputs : Tensor
            Input data
        attention_mask : None or Tensor or xformers.ops.AttentionBias
            Optional attention mask

        Returns
        -------
        outputs : Tensor
            Outputs
        """
        q, k, v = self.qkv_linear(inputs)  # each: (..., num_heads, num_items, num_channels, 16)

        # Rotary positional encoding
        if self.pos_encoding is not None:
            q = self.pos_encoding(q)
            k = self.pos_encoding(k)

        # Attention layer
        h = self._attend(q, k, v, attention_mask)

        # Concatenate heads and transform linearly
        h = rearrange(
            h,
            "... num_heads num_items hidden_channels -> ... num_items (num_heads hidden_channels)",
        )
        outputs = self.out_linear(h)  # (..., num_items, out_channels)

        return outputs

    @staticmethod
    def _attend(q, k, v, attention_mask=None):
        """Scaled dot-product attention."""

        # Add batch dimension if needed
        bh_shape = q.shape[:-2]
        q = to_nd(q, 4)
        k = to_nd(k, 4)
        v = to_nd(v, 4)

        # SDPA
        outputs = scaled_dot_product_attention(
            q.contiguous(), k.expand_as(q).contiguous(), v.expand_as(q), attn_mask=attention_mask
        )

        # Return batch dimensions to inputs
        outputs = outputs.view(*bh_shape, *outputs.shape[-2:])

        return outputs


class BaselineTransformerBlock(nn.Module):
    """Baseline transformer block.

    Inputs are first processed by a block consisting of LayerNorm, multi-head self-attention, and
    residual connection. Then the data is processed by a block consisting of another LayerNorm, an
    item-wise two-layer MLP with GeLU activations, and another residual connection.

    Parameters
    ----------
    channels : int
        Number of input and output channels.
    num_heads : int
        Number of attention heads.
    pos_encoding : bool
        Whether to apply rotary positional embeddings along the item dimension to the scalar keys
        and queries.
    pos_encoding_base : int
        Maximum frequency used in positional encodings. (The minimum frequency is always 1.)
    increase_hidden_channels : int
        Factor by which the key, query, and value size is increased over the default value of
        hidden_channels / num_heads.
    multi_query : bool
        Use multi-query attention instead of multi-head attention.
    """

    def __init__(
        self,
        channels,
        num_heads: int = 8,
        pos_encoding: bool = False,
        pos_encoding_base: int = 4096,
        increase_hidden_channels=1,
        multi_query: bool = True,
    ) -> None:
        super().__init__()

        self.norm = BaselineLayerNorm()

        # When using positional encoding, the number of scalar hidden channels needs to be even.
        # It also should not be too small.
        hidden_channels = channels // num_heads * increase_hidden_channels
        if pos_encoding:
            hidden_channels = (hidden_channels + 1) // 2 * 2
            hidden_channels = max(hidden_channels, 16)

        self.attention = BaselineSelfAttention(
            channels,
            channels,
            hidden_channels,
            num_heads=num_heads,
            pos_encoding=pos_encoding,
            pos_enc_base=pos_encoding_base,
            multi_query=multi_query,
        )

        self.mlp = nn.Sequential(
            nn.Linear(channels, 2 * channels),
            nn.GELU(),
            nn.Linear(2 * channels, channels),
        )

    def forward(self, inputs: torch.Tensor, attention_mask=None) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        inputs : Tensor
            Input data
        attention_mask : None or Tensor or xformers.ops.AttentionBias
            Optional attention mask

        Returns
        -------
        outputs : Tensor
            Outputs
        """

        # Residual attention
        h = self.norm(inputs)
        h = self.attention(h, attention_mask)
        outputs = inputs + h

        # Residual MLP
        h = self.norm(outputs)
        h = self.mlp(h)
        outputs = outputs + h

        return outputs


class BaselineTransformer(nn.Module):
    """Baseline transformer.

    Combines num_blocks transformer blocks, each consisting of multi-head self-attention layers, an
    MLP, residual connections, and normalization layers.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    hidden_channels : int
        Number of hidden channels.
    num_blocks : int
        Number of transformer blocks.
    num_heads : int
        Number of attention heads.
    pos_encoding : bool
        Whether to apply rotary positional embeddings along the item dimension to the scalar keys
        and queries.
    pos_encoding_base : int
        Maximum frequency used in positional encodings. (The minimum frequency is always 1.)
    increase_hidden_channels : int
        Factor by which the key, query, and value size is increased over the default value of
        hidden_channels / num_heads.
    multi_query : bool
        Use multi-query attention instead of multi-head attention.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        num_blocks: int = 10,
        num_heads: int = 8,
        pos_encoding: bool = False,
        pos_encoding_base: int = 4096,
        checkpoint_blocks: bool = False,
        increase_hidden_channels=1,
        multi_query: bool = False,
    ) -> None:
        super().__init__()
        self.checkpoint_blocks = checkpoint_blocks
        self.linear_in = nn.Linear(in_channels, hidden_channels)
        self.blocks = nn.ModuleList(
            [
                BaselineTransformerBlock(
                    hidden_channels,
                    num_heads=num_heads,
                    pos_encoding=pos_encoding,
                    pos_encoding_base=pos_encoding_base,
                    increase_hidden_channels=increase_hidden_channels,
                    multi_query=multi_query,
                )
                for _ in range(num_blocks)
            ]
        )
        self.linear_out = nn.Linear(hidden_channels, out_channels)

    def forward(self, inputs: torch.Tensor, attention_mask=None) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        inputs : Tensor with shape (..., num_items, num_channels)
            Input data
        attention_mask : None or Tensor or xformers.ops.AttentionBias
            Optional attention mask

        Returns
        -------
        outputs : Tensor with shape (..., num_items, num_channels)
            Outputs
        """
        h = self.linear_in(inputs)
        for block in self.blocks:
            if self.checkpoint_blocks:
                fn = partial(block, attention_mask=attention_mask)
                h = checkpoint(fn, h)
            else:
                h = block(h, attention_mask=attention_mask)
        outputs = self.linear_out(h)
        return outputs


class BaselineAxialTransformer(nn.Module):
    """Baseline axial transformer for data with two token dimensions.

    Combines num_blocks transformer blocks, each consisting of multi-head self-attention layers, an
    MLP, residual connections, and normalization layers.

    Assumes input data with shape `(..., num_items_1, num_items_2, num_channels, [16])`.

    The first, third, fifth, ... block computes attention over the `items_2` axis. The other blocks
    compute attention over the `items_1` axis. Positional encoding can be specified separately for
    both axes.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    hidden_channels : int
        Number of hidden channels.
    num_blocks : int
        Number of transformer blocks.
    num_heads : int
        Number of attention heads.
    pos_encodings : tuple of bool
        Whether to apply rotary positional embeddings along the item dimensions to the scalar keys
        and queries.
    pos_encoding_base : int
        Maximum frequency used in positional encodings. (The minimum frequency is always 1.)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        num_blocks: int = 20,
        num_heads: int = 8,
        pos_encodings: Tuple[bool, bool] = (False, False),
        pos_encoding_base: int = 4096,
    ) -> None:
        super().__init__()
        self.linear_in = nn.Linear(in_channels, hidden_channels)
        self.blocks = nn.ModuleList(
            [
                BaselineTransformerBlock(
                    hidden_channels,
                    num_heads=num_heads,
                    pos_encoding=pos_encodings[(block + 1) % 2],
                    pos_encoding_base=pos_encoding_base,
                )
                for block in range(num_blocks)
            ]
        )
        self.linear_out = nn.Linear(hidden_channels, out_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        inputs : Tensor with shape (..., num_items1, num_items2, num_channels)
            Input data

        Returns
        -------
        outputs : Tensor with shape (..., num_items1, num_items2, num_channels)
            Outputs
        """

        rearrange_pattern = "... i j c -> ... j i c"

        h = self.linear_in(inputs)

        for i, block in enumerate(self.blocks):
            # For first, third, ... block, we want to perform attention over the first token
            # dimension. We implement this by transposing the two item dimensions.
            if i % 2 == 1:
                h = rearrange(h, rearrange_pattern)

            h = block(h)

            # Transposing back to standard axis order
            if i % 2 == 1:
                h = rearrange(h, rearrange_pattern)

        outputs = self.linear_out(h)

        return outputs
