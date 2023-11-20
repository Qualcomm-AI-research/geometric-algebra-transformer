# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
from gatr.utils.misc import get_device

BATCHSIZE = 64
NUM_STEPS = 2000
STEPS_PER_EPOCH = 100
DATASET_SIZE = BATCHSIZE * STEPS_PER_EPOCH
NUM_EPOCHS = NUM_STEPS // STEPS_PER_EPOCH
DEVICE = get_device()
