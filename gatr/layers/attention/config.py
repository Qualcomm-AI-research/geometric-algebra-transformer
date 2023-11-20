# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass
class SelfAttentionConfig:
    """Configuration for attention.

    Parameters
    ----------
    in_mv_channels : int
        Number of input multivector channels.
    out_mv_channels : int
        Number of output multivector channels.
    num_heads : int
        Number of attention heads.
    in_s_channels : int
        Input scalar channels. If None, no scalars are expected nor returned.
    out_s_channels : int
        Output scalar channels. If None, no scalars are expected nor returned.
    additional_qk_mv_channels : int
        Whether additional multivector features for the keys and queries will be provided.
    additional_qk_s_channels : int
        Whether additional scalar features for the keys and queries will be provided.
    normalizer : str
        Normalizer function to use in sdp_dist attention
    normalizer_eps : float
        Small umerical constant for stability in the normalizer in sdp_dist attention
    multi_query: bool
        Whether to do multi-query attention
    attention_type : {"scalar", "geometric", "sdp_dist"}
        Whether the attention mechanism is based on the scalar product or also the join.
    pos_encoding : bool
        Whether to apply rotary positional embeddings along the item dimension to the scalar keys
        and queries.
    pos_enc_base : int
        Base for the frequencies in the positional encoding.
    output_init : str
        Initialization scheme for final linear layer
    increase_hidden_channels : int
        Factor by which to increase the number of hidden channels (both multivectors and scalars)
    dropout_prob : float or None
        Dropout probability
    """

    multi_query: bool = True
    in_mv_channels: Optional[int] = None
    out_mv_channels: Optional[int] = None
    in_s_channels: Optional[int] = None
    out_s_channels: Optional[int] = None
    num_heads: int = 8
    additional_qk_mv_channels: int = 0
    additional_qk_s_channels: int = 0
    normalizer_eps: Optional[float] = 1e-3
    pos_encoding: bool = False
    pos_enc_base: int = 4096
    output_init: str = "default"
    checkpoint: bool = True
    increase_hidden_channels: int = 2
    dropout_prob: Optional[float] = None

    def __post_init__(self):
        """Type checking / conversion."""
        if isinstance(self.dropout_prob, str) and self.dropout_prob.lower() in ["null", "none"]:
            self.dropout_prob = None

    @property
    def hidden_mv_channels(self) -> Optional[int]:
        """Returns the number of hidden multivector channels."""

        if self.in_mv_channels is None:
            return None

        return max(self.increase_hidden_channels * self.in_mv_channels // self.num_heads, 1)

    @property
    def hidden_s_channels(self) -> Optional[int]:
        """Returns the number of hidden scalar channels."""

        if self.in_s_channels is None:
            return None

        hidden_s_channels = max(
            self.increase_hidden_channels * self.in_s_channels // self.num_heads, 4
        )

        # When using positional encoding, the number of scalar hidden channels needs to be even.
        # It also should not be too small.
        if self.pos_encoding:
            hidden_s_channels = (hidden_s_channels + 1) // 2 * 2
            hidden_s_channels = max(hidden_s_channels, 8)

        return hidden_s_channels

    @classmethod
    def cast(cls, config: Any) -> SelfAttentionConfig:
        """Casts an object as SelfAttentionConfig."""
        if isinstance(config, SelfAttentionConfig):
            return config
        if isinstance(config, Mapping):
            return cls(**config)
        raise ValueError(f"Can not cast {config} to {cls}")
