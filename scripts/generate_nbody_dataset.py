#!/usr/bin/env python3
# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.

from pathlib import Path

import hydra
import numpy as np

from gatr.experiments.nbody import NBodySimulator


def generate_dataset(filename, simulator, num_samples, num_planets=5, domain_shift=False):
    """Samples from n-body simulator and stores the results at `filename`."""
    assert not Path(filename).exists()
    m, x_initial, v_initial, x_final, trajectories = simulator.sample(
        num_samples, num_planets=num_planets, domain_shift=domain_shift
    )
    np.savez(
        filename,
        m=m,
        x_initial=x_initial,
        v_initial=v_initial,
        x_final=x_final,
        trajectories=trajectories,
    )


def generate_datasets(path):
    """Generates a canonical set of datasets for the n-body problem, stores them in `path`."""
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=True)

    print(f"Creating gravity dataset in {str(path)}")

    simulator = NBodySimulator()
    generate_dataset(path / "train.npz", simulator, 100000, num_planets=3, domain_shift=False)
    generate_dataset(path / "val.npz", simulator, 5000, num_planets=3, domain_shift=False)
    generate_dataset(path / "eval.npz", simulator, 5000, num_planets=3, domain_shift=False)
    generate_dataset(
        path / "e3_generalization.npz", simulator, 5000, num_planets=3, domain_shift=True
    )
    generate_dataset(
        path / "object_generalization.npz", simulator, 5000, num_planets=5, domain_shift=False
    )

    print("Done, have a nice day!")


@hydra.main(config_path="../config", config_name="nbody", version_base=None)
def main(cfg):
    """Entry point for n-body dataset generation."""
    data_dir = cfg.data.data_dir
    np.random.seed(cfg.seed)
    generate_datasets(data_dir)


if __name__ == "__main__":
    main()
