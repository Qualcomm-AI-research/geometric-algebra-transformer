# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
from collections.abc import Mapping
from functools import lru_cache
from itertools import product

import numpy as np
import torch


class NaNError(BaseException):
    """Exception to be raise when the training encounters a NaN in loss or model weights."""


def get_device() -> torch.device:
    """Gets CUDA if available, CPU else."""
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def frequency_check(step, every_n_steps, skip_initial=False):
    """Checks whether an action should be performed at a given step and frequency.

    Parameters
    ----------
    step : int
        Step number (one-indexed)
    every_n_steps : None or int
        Desired action frequency. None or 0 correspond to never executing the action.
    skip_initial : bool
        If True, frequency_check returns False at step 0.

    Returns
    -------
    decision : bool
        Whether the action should be executed.
    """

    if every_n_steps is None or every_n_steps == 0:
        return False

    if skip_initial and step == 0:
        return False

    return step % every_n_steps == 0


def flatten_dict(d, parent_key="", sep="."):
    """Flattens a nested dictionary with str keys."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, Mapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def sample_uniform_in_circle(n, min_radius=0.0, max_radius=1.0):
    """Samples uniformly in a 2D circle using batched rejection sampling."""

    assert 0.0 <= min_radius < max_radius, "Inconsistent inputs to sample_uniform_in_circle"

    mask = None
    samples = None

    while samples is None or np.sum(mask) > 0:
        new_samples = max_radius * np.random.uniform(low=-1, high=1, size=(n, 2))

        if samples is None:
            samples = new_samples
        else:
            samples = (1 - mask) * samples + mask * new_samples

        r2 = np.sum(samples**2, axis=-1)
        mask = np.logical_or((r2 < min_radius**2), (r2 > max_radius**2))[:, np.newaxis]

    return samples


def sample_log_uniform(min_, max_, size):
    """Samples log-uniformly from (min_, max_)."""
    if isinstance(size, tuple):
        n = int(np.product(size))
    else:
        n = size

    log_x = np.random.rand(n) * (np.log(max_) - np.log(min_)) + np.log(min_)
    x = np.exp(log_x)

    if isinstance(size, tuple):
        x = x.reshape(size)

    return x


@lru_cache()
@torch.no_grad()
def make_full_edge_index(num_nodes, batchsize=1, self_loops=False, device=torch.device("cpu")):
    """Creates a PyG-style edge index for a fully connected graph of `num_nodes` nodes."""

    # Construct fully connected edge index
    src, dst = [], []
    for i, j in product(range(num_nodes), repeat=2):
        if not self_loops and i == j:
            continue
        src.append(i)
        dst.append(j)

    edge_index_per_batch = torch.LongTensor([src, dst]).to(device)

    # Repeat for each batch element
    if batchsize > 1:
        edge_index_list = [edge_index_per_batch + k * num_nodes for k in range(batchsize)]
        edge_index = torch.cat(edge_index_list, dim=1)
    else:
        edge_index = edge_index_per_batch

    return edge_index
