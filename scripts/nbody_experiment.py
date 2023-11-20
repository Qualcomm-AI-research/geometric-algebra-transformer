#!/usr/bin/env python3
# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.

import hydra

from gatr.experiments.nbody import NBodyExperiment


@hydra.main(config_path="../config", config_name="nbody", version_base=None)
def main(cfg):
    """Entry point for n-body experiment."""
    exp = NBodyExperiment(cfg)
    exp()


if __name__ == "__main__":
    main()
