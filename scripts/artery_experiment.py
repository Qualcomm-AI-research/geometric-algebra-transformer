#!/usr/bin/env python3
# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.

import hydra

from gatr.experiments.arteries.experiment import ArteryExperiment


@hydra.main(config_path="../config", config_name="arteries", version_base=None)
def main(cfg):
    """Entry point for artery experiment."""
    exp = ArteryExperiment(cfg)
    exp()


if __name__ == "__main__":
    main()
