# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
import time
from typing import Optional, Union

import mlflow
import numpy as np

from gatr.utils.logger import logger

MAX_LOG_ATTEMPTS = 20
BASE_WAIT_TIME = 0.2  # Base wait time between attempts in seconds
WAIT_TIME_FACTOR = 1.5  # Increase of the wait time for each successive trial
WAIT_TIME_RANDOM = (
    0.5  # Upper bound on random wait time that is added to break synchronicity of multiple jobs
)

LOGGING_ENABLED = True


def log_mlflow(
    key: str, value: Union[float, str], step: Optional[int] = None, kind: str = "metric"
) -> None:
    """Logs metric or param to MLflow.

    Catches exceptions due to locked or unavailable DB and then retries.

    Parameters
    ----------
    key : str
        Key to log
    value : float or str
        Value to log
    step : None or int
        Step to log
    kind : {"metric", "param"}
        Whether to log a metric or a parameter

    Raises
    ------
    ValueError
        If `kind` is invalid
    """
    if not LOGGING_ENABLED:
        return

    for attempt in range(MAX_LOG_ATTEMPTS):
        try:
            if kind == "metric":
                mlflow.log_metric(key, value, step=step)
            elif kind == "param":
                mlflow.log_param(key, value)
            else:
                raise ValueError(
                    f"Unknown MLflow logging kind {kind}, should be 'metric' or 'param'"
                )
            return
        except Exception:  # pylint: disable=broad-except
            wait_time = (
                BASE_WAIT_TIME * WAIT_TIME_FACTOR**attempt + WAIT_TIME_RANDOM * np.random.rand()
            )
            logger.warning(
                f"Exception when logging to MLflow at step {step} (attempt {attempt + 1}). "
                f"Waiting for {wait_time:.1f}s before trying again."
            )
            time.sleep(wait_time)

    logger.error(f"Warning: Failed to write to MLflow {MAX_LOG_ATTEMPTS} times")
