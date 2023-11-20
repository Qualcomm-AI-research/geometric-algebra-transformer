# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Optional


@dataclass
class MLPConfig:
    """Geometric MLP configuration.

    Parameters
    ----------
    mv_channels : iterable of int
        Number of multivector channels at each layer, from input to output
    s_channels : None or iterable of int
        If not None, sets the number of scalar channels at each layer, from input to output. Length
        needs to match mv_channels
    activation : {"relu", "sigmoid", "gelu"}
        Which (gated) activation function to use
    dropout_prob : float or None
        Dropout probability
    """

    mv_channels: Optional[List[int]] = None
    s_channels: Optional[List[int]] = None
    activation: str = "gelu"
    dropout_prob: Optional[float] = None

    def __post_init__(self):
        """Type checking / conversion."""
        if isinstance(self.dropout_prob, str) and self.dropout_prob.lower() in ["null", "none"]:
            self.dropout_prob = None

    @classmethod
    def cast(cls, config: Any) -> MLPConfig:
        """Casts an object as MLPConfig."""
        if isinstance(config, MLPConfig):
            return config
        if isinstance(config, Mapping):
            return cls(**config)
        raise ValueError(f"Can not cast {config} to {cls}")
