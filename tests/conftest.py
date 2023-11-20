# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
import pytest
from click.testing import CliRunner


@pytest.fixture(scope="function")
def click_runner() -> CliRunner:
    """Returns a CLI runner."""
    return CliRunner()
