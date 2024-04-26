# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
from typing import Iterator, Tuple

import opt_einsum
import pytest
import torch
from torch.backends import opt_einsum as opt_einsum_torch

from gatr.utils import einsum
from gatr.utils.einsum import (
    _cached_einsum,
    _einsum_with_path,
    _einsum_with_path_ignored,
    _get_cached_path_for_equation_and_shapes,
    enable_cached_einsum,
)

_DIM = 5


@pytest.fixture(autouse=True, scope="module")
def disable_opt_einsum() -> Iterator[None]:
    """Disable usage of opt_einsum by torch during tests in this module."""
    opt_einsum_torch.enabled = False
    yield
    opt_einsum_torch.enabled = True


@pytest.fixture(name="einsum_eq")
def einsum_eq_fixture() -> str:
    """Provides a non-trivial einsum equation."""
    # Example equation from https://optimized-einsum.readthedocs.io/en/stable/index.html
    return "pi,qj,ijkl,rk,sl->pqrs"


@pytest.fixture(name="example_operands")
def example_operands_fixture() -> Tuple[torch.Tensor, ...]:
    """Provides tensors for a non-trivial einsum equation."""
    # Example tensors from https://optimized-einsum.readthedocs.io/en/stable/index.html
    I = torch.rand(_DIM, _DIM, _DIM, _DIM)
    C = torch.rand(_DIM, _DIM)
    return (C, C, I, C, C)


def test_opt_einsum_shape() -> None:
    """Verifies contract_path can be called with shapes.

    Note that we test upstream functionality here, which should not be necessary.
    However, with the latest release (3.3.0), this failed.

    NB: Test does not fail on every example (e.g., does not fail on `einsum_eq_fixture`).
    We use an equation from a failing test (test_pin).
    """
    einsum_eq = "i j k, ... j, ... k -> ... i"
    sizes = (_DIM, _DIM, _DIM)
    operands = tuple(torch.rand(sizes) for _ in sizes)
    op_shape = tuple(x.size() for x in operands)
    assert (
        opt_einsum.contract_path(einsum_eq, *operands, optimize="optimal")[0]
        == opt_einsum.contract_path(einsum_eq, *op_shape, optimize="optimal", shapes=True)[0]
    )


def test_e2e_cached_path(einsum_eq: str, example_operands: Tuple[torch.Tensor]) -> None:
    """Checks that torch.einsum and cached_einsum deliver the same results."""
    # pylint: disable=protected-access
    # WHEN we call cached einsum, remembering the the cache sizes
    cache_size_unused = _get_cached_path_for_equation_and_shapes.cache_info().currsize

    result_with_uncached_path = _cached_einsum(einsum_eq, *example_operands)
    cache_size_used_once = _get_cached_path_for_equation_and_shapes.cache_info().currsize

    result_with_cached_path = _cached_einsum(einsum_eq, *example_operands)
    cache_size_used_twice = _get_cached_path_for_equation_and_shapes.cache_info().currsize

    # WHEN we compute the expected result
    expected_result = torch.einsum(einsum_eq, *example_operands)

    # THEN the einsum functionality itself works
    torch.testing.assert_close(expected_result, result_with_uncached_path)
    torch.testing.assert_close(expected_result, result_with_cached_path)
    # THEN the cache was used
    assert cache_size_unused + 1 == cache_size_used_once == cache_size_used_twice


def test_toggle_einsum_caching() -> None:
    """Verifies that the global switch for gatr_einsum works as expected."""
    # pylint: disable=comparison-with-callable # Nope, we did not forget the parenthesis.
    # pylint: disable=protected-access
    assert einsum._gatr_einsum == _cached_einsum
    assert einsum._gatr_einsum_with_path == _einsum_with_path

    enable_cached_einsum(False)
    assert einsum._gatr_einsum == torch.einsum
    assert einsum._gatr_einsum_with_path == _einsum_with_path_ignored

    enable_cached_einsum(True)
    assert einsum._gatr_einsum == _cached_einsum
    assert einsum._gatr_einsum_with_path == _einsum_with_path
