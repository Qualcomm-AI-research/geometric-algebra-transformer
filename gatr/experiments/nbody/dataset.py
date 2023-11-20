# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
import numpy as np
import torch


class NBodyDataset(torch.utils.data.Dataset):
    """N-body prediction dataset.

    Loads data generated with generate_nbody_dataset.py from disk.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to the npz file with the dataset to be loaded.
    subsample : None or float
        If not None, defines the fraction of the dataset to be used. For instance, `subsample=0.1`
        uses just 10% of the samples in the dataset.
    keep_trajectories : bool
        Whether to keep the full particle trajectories in the dataset. They are neither needed
        for training nor evaluation, but can be useful for visualization.
    """

    def __init__(self, filename, subsample=None, keep_trajectories=False):
        super().__init__()
        self.x, self.y, self.trajectories = self._load_data(
            filename, subsample, keep_trajectories=keep_trajectories
        )

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.x)

    def __getitem__(self, idx):
        """Returns the `idx`-th sample from the dataset."""
        return self.x[idx], self.y[idx]

    @staticmethod
    def _load_data(filename, subsample=None, keep_trajectories=False):
        """Loads data from file and converts to input and output tensors."""
        # Load data from file
        npz = np.load(filename, "r")
        m, x_initial, v_initial, x_final = (
            npz["m"],
            npz["x_initial"],
            npz["v_initial"],
            npz["x_final"],
        )

        # Convert to tensors
        m = torch.from_numpy(m).to(torch.float32).unsqueeze(2)
        x_initial = torch.from_numpy(x_initial).to(torch.float32)
        v_initial = torch.from_numpy(v_initial).to(torch.float32)
        x_final = torch.from_numpy(x_final).to(torch.float32)

        # Concatenate into inputs and outputs
        x = torch.cat((m, x_initial, v_initial), dim=2)  # (batchsize, num_objects, 7)
        y = x_final  # (batchsize, num_objects, 3)

        # Optionally, keep raw trajectories around (for plotting)
        if keep_trajectories:
            trajectories = npz["trajectories"]
        else:
            trajectories = None

        # Subsample
        if subsample is not None and subsample < 1.0:
            n_original = len(x)
            n_keep = int(round(subsample * n_original))
            assert 0 < n_keep <= n_original
            x = x[:n_keep]
            y = y[:n_keep]
            if trajectories is not None:
                trajectories = trajectories[:n_keep]

        return x, y, trajectories
