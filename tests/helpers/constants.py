# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
"""Settings used for multiple tests."""

# Default tolerances
TOLERANCES = dict(atol=1e-3, rtol=1e-4)
MILD_TOLERANCES = dict(atol=0.05, rtol=0.05)

# Batch dimensions that are typically checked
BATCH_DIMS = [(7, 9), tuple()]
