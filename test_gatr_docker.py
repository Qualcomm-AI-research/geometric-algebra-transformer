#!/usr/bin/env python3

import os
os.environ["XFORMERS_DISABLED"] = "1"

import torch
from gatr import GATr, SelfAttentionConfig, MLPConfig
from gatr.interface import embed_point, extract_scalar

# Create a simple point cloud
points = torch.randn(2, 10, 3)  # batch_size=2, num_points=10, dim=3

# Create a GATr model
model = GATr(
    in_mv_channels=1,
    out_mv_channels=1,
    hidden_mv_channels=16,
    in_s_channels=None,
    out_s_channels=None,
    hidden_s_channels=32,
    num_blocks=2,
    attention=SelfAttentionConfig(),
    mlp=MLPConfig(),
)

# Embed points in PGA
embedded_points = embed_point(points).unsqueeze(-2)  # (2, 10, 1, 16)

# Pass through GATr
embedded_output, _ = model(embedded_points, scalars=None)

# Extract scalar output
scalar_output = extract_scalar(embedded_output)  # (2, 10, 1, 1)
final_output = torch.mean(scalar_output, dim=(-3, -2))  # (2, 1)

print("Input shape:", points.shape)
print("Output shape:", final_output.shape)
print("Output:", final_output)

print("GATr is working on macOS!")
