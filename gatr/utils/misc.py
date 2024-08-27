# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All rights reserved.
import random
from collections.abc import Mapping
from functools import wraps
from itertools import chain, product
from typing import Any, Callable, List, Literal, Optional, Union

import numpy as np
import torch
from torch import Tensor

from gatr.utils.einsum import gatr_cache


class NaNError(BaseException):
    """Exception to be raise when the training encounters a NaN in loss or model weights."""


def get_device() -> torch.device:
    """Gets CUDA if available, CPU else."""
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def frequency_check(step, every_n_steps, skip_initial=False, include_fractional=None):
    """Checks whether an action should be performed at a given step and frequency.

    If `include_fractional` is given, the check also returns True when
    `step == round(every_n_steps * fraction)` for each `fraction` in `include_fractional`.

    Parameters
    ----------
    step : int
        Step number (one-indexed)
    every_n_steps : None or int
        Desired action frequency. None or 0 correspond to never executing the action.
    skip_initial : bool
        If True, frequency_check returns False at step 0.
    include_fractional : None or tuple of float
        If not None, the check also returns True when `step == round(every_n_steps * fraction)`
        for each `fraction` in `include_fractional`.

    Returns
    -------
    decision : bool
        Whether the action should be executed.
    """

    if every_n_steps is None or every_n_steps == 0:
        return False

    if skip_initial and step == 0:
        return False

    if include_fractional is not None:
        for fraction in include_fractional:
            if step == int(round(fraction * every_n_steps)):
                return True

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


@gatr_cache
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

    edge_index_per_batch = torch.tensor([src, dst], dtype=torch.int64, device=device)

    # Repeat for each batch element
    if batchsize > 1:
        edge_index_list = [edge_index_per_batch + k * num_nodes for k in range(batchsize)]
        edge_index = torch.cat(edge_index_list, dim=1)
    else:
        edge_index = edge_index_per_batch

    return edge_index


def get_batchsize(data):
    """Given either a tensor or a list of tensors or a dict of tensors, returns the batchsize."""

    if isinstance(data, Mapping):
        assert len(data) > 0
        data = next(iter(data.values()))
    elif isinstance(data, (tuple, list)):
        assert len(data) > 0
        data = data[0]
    elif isinstance(data, (tuple, list)):
        assert len(data) > 0
        data = data[0]

    return len(data)


@gatr_cache
def maximum_dtype(*args):
    """Return dtype with maximum precision. Cached. Compatible with compilation."""
    dtype = max(args, key=lambda dt: torch.finfo(dt).bits)
    return dtype


@gatr_cache
def minimum_dtype(*args):
    """Return dtype with maximum precision. Compatible with compilation."""
    dtype = min(args, key=lambda dt: torch.finfo(dt).bits)
    return dtype


def minimum_autocast_precision(
    min_dtype: torch.dtype = torch.float32,
    output: Optional[Union[Literal["low", "high"], torch.dtype]] = None,
    which_args: Optional[List[int]] = None,
    which_kwargs: Optional[List[str]] = None,
):
    """Decorator that ensures input tensors are autocast to a minimum precision.

    Only has an effect in autocast-enabled regions. Otherwise, does not change the function.

    Only floating-point inputs are modified. Non-tensors, integer tensors, and boolean tensors are
    untouched.

    Note: AMP is turned on and off separately for CPU and CUDA. This decorator may fail in
    the case where both devices are used, with only one of them on AMP.

    Parameters
    ----------
    min_dtype : dtype
        Minimum dtype. Default: float32.
    output: None or "low" or "high" or dtype
        Specifies which dtypes the outputs should be cast to. Only floating-point Tensor outputs
        are affected. If None, the outputs are not modified. If "low", the lowest-precision input
        dtype is used. If "high", `min_dtype` or the highest-precision input dtype is used
        (whichever is higher).
    which_args : None or list of int
        If not None, specifies which positional arguments are to be modified. If None (the default),
        all positional arguments are modified (if they are Tensors and of a floating-point dtype).
    which_kwargs : bool
        If not None, specifies which keyword arguments are to be modified. If None (the default),
        all keyword arguments are modified (if they are Tensors and of a floating-point dtype).

    Returns
    -------
    decorator : Callable
        Decorator.
    """

    def decorator(func: Callable):
        """Decorator that casts input tensors to minimum precision."""

        def _cast_in(var: Any):
            """Casts a single input to at least 32-bit precision."""
            if not isinstance(var, Tensor):
                # We don't want to modify non-Tensors
                return var
            if not var.dtype.is_floating_point:
                # Integer / boolean tensors are also not touched
                return var
            dtype = maximum_dtype(var.dtype, min_dtype)
            return var.to(dtype)

        def _cast_out(var: Any, dtype: torch.dtype):
            """Casts a single output to desired precision."""
            if not isinstance(var, Tensor):
                # We don't want to modify non-Tensors
                return var
            if not var.dtype.is_floating_point:
                # Integer / boolean tensors are also not touched
                return var
            return var.to(dtype)

        @wraps(func)
        def decorated_func(*args: Any, **kwargs: Any):
            """Decorated func."""

            # Only change dtypes in autocast-enabled regions
            if not (torch.is_autocast_enabled() or torch.is_autocast_cpu_enabled()):
                # NB: torch.is_autocast_enabled() only checks for GPU autocast
                # See https://github.com/pytorch/pytorch/issues/110966
                return func(*args, **kwargs)

            # Cast inputs to at least 32 bit
            mod_args = [
                _cast_in(arg) for i, arg in enumerate(args) if which_args is None or i in which_args
            ]
            mod_kwargs = {
                key: _cast_in(val)
                for key, val in kwargs.items()
                if which_kwargs is None or key in which_kwargs
            }

            # Call function w/o autocast enabled
            with torch.autocast(device_type="cuda", enabled=False), torch.autocast(
                device_type="cpu", enabled=False
            ):
                outputs = func(*mod_args, **mod_kwargs)

            # Cast outputs to correct dtype
            if output is None:
                return outputs

            if output in ["low", "high"]:
                in_dtypes = [
                    arg.dtype
                    for arg in chain(args, kwargs.values())
                    if isinstance(arg, Tensor) and arg.dtype.is_floating_point
                ]
                assert len(in_dtypes)
                if output == "low":
                    out_dtype = minimum_dtype(min_dtype, *in_dtypes)
                else:
                    out_dtype = maximum_dtype(*in_dtypes)
            else:
                out_dtype = output

            if isinstance(outputs, tuple):
                return (_cast_out(val, out_dtype) for val in outputs)
            else:
                return _cast_out(outputs, out_dtype)

        return decorated_func

    return decorator


DEFAULT_SEED = 1824


def seed_all(seed: int = DEFAULT_SEED) -> None:
    """Seeds all known sources of pseudo-randomness in our stack."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def models_weights_are_close(model0: torch.nn.Module, model1: torch.nn.Module):
    """Checks whether models have close weights."""
    for data0, data1 in zip(model0.state_dict().items(), model1.state_dict().items()):
        if data0[0] != data1[0]:
            raise RuntimeError("Models have differing parameter names, cannot compare.")
        if not torch.equal(data0[1], data1[1]):
            return False
    return True
