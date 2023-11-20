# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
import os
import re

import numpy as np
import scipy
import torch
from coronary_mesh_convolution.datasets import InMemoryVesselDataset
from torch.utils.data import Dataset


class NumberedVesselDataset(InMemoryVesselDataset):
    """Artery dataset with added shape id attribute."""

    @staticmethod
    def load_process_hdf5(path):
        """Wrap super method with adding shape id."""
        _, sample = os.path.split(path)
        data = InMemoryVesselDataset.load_process_hdf5(path)
        data.shape_id = int(re.match(r"sample_(\d+)", sample).group(1))
        return data


class RotationDataset(Dataset):
    """A dataset containing random (S)O(3) elements.

    Parameters
    ----------
    num : int
        Size of the dataset (number of rotations)
    special : bool
        If True, samples from the Haar measure on SO(3); otherwise, on O(3)
    seed : int
        numpy random seed
    """

    def __init__(self, num, special, seed):
        super().__init__()
        if special:
            sampler = scipy.stats.special_ortho_group
        else:
            sampler = scipy.stats.ortho_group
        random_state = np.random.RandomState(seed=seed)
        self.data = torch.tensor(
            sampler.rvs(3, size=num, random_state=random_state), dtype=torch.float32
        )

    def __len__(self):
        """Dataset size."""
        return len(self.data)

    def __getitem__(self, idx):
        """Return an item from the dataset."""
        return self.data[idx]
