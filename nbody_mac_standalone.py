#!/usr/bin/env python3
# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.

"""
This script:
Is completely standalone and doesn't import anything from the GATr package
Implements the n-body simulator from scratch
Provides the same dataset generation functionality
Uses argparse instead of Hydra for simplicity
This should work without requiring DGL or any other problematic dependencies.
"""

from pathlib import Path
import os
import sys
import numpy as np
import argparse

# Create a standalone NBodySimulator for Mac
class SimpleNBodySimulator:
    """A simplified version of NBodySimulator for Mac compatibility."""
    
    def __init__(self, G=1.0, dt=0.01, n_steps=100):
        """Initialize the simulator."""
        self.G = G
        self.dt = dt
        self.n_steps = n_steps
    
    def _compute_acceleration(self, pos, mass):
        """Compute acceleration due to gravity."""
        n = pos.shape[0]
        acc = np.zeros_like(pos)
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    r = pos[j] - pos[i]
                    dist = np.linalg.norm(r)
                    if dist > 1e-10:  # Avoid division by zero
                        acc[i] += self.G * mass[j] * r / (dist**3)
        
        return acc
    
    def _simulate_step(self, pos, vel, mass):
        """Simulate one step using leapfrog integration."""
        acc = self._compute_acceleration(pos, mass)
        vel_half = vel + 0.5 * self.dt * acc
        pos_new = pos + self.dt * vel_half
        acc_new = self._compute_acceleration(pos_new, mass)
        vel_new = vel_half + 0.5 * self.dt * acc_new
        
        return pos_new, vel_new
    
    def _simulate_trajectory(self, pos_initial, vel_initial, mass):
        """Simulate a complete trajectory."""
        pos = pos_initial.copy()
        vel = vel_initial.copy()
        trajectory = np.zeros((self.n_steps + 1, pos.shape[0], pos.shape[1]))
        trajectory[0] = pos
        
        for i in range(self.n_steps):
            pos, vel = self._simulate_step(pos, vel, mass)
            trajectory[i + 1] = pos
        
        return trajectory, pos, vel
    
    def sample(self, num_samples, num_planets=5, domain_shift=False):
        """Sample random n-body systems and simulate them."""
        # Initialize arrays
        m = np.zeros((num_samples, num_planets))
        x_initial = np.zeros((num_samples, num_planets, 3))
        v_initial = np.zeros((num_samples, num_planets, 3))
        x_final = np.zeros((num_samples, num_planets, 3))
        trajectories = np.zeros((num_samples, self.n_steps + 1, num_planets, 3))
        
        # Set mass distribution based on domain_shift
        if domain_shift:
            mass_min, mass_max = 0.5, 2.0
        else:
            mass_min, mass_max = 0.1, 1.0
        
        # Generate samples
        for i in range(num_samples):
            # Generate random masses
            m[i] = np.random.uniform(mass_min, mass_max, num_planets)
            
            # Generate random initial positions in a cube
            x_initial[i] = np.random.uniform(-1.0, 1.0, (num_planets, 3))
            
            # Generate random initial velocities
            v_initial[i] = np.random.uniform(-0.1, 0.1, (num_planets, 3))
            
            # Simulate trajectory
            traj, final_pos, _ = self._simulate_trajectory(
                x_initial[i], v_initial[i], m[i]
            )
            
            trajectories[i] = traj
            x_final[i] = final_pos
        
        return m, x_initial, v_initial, x_final, trajectories


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

    # Use our simplified simulator
    simulator = SimpleNBodySimulator()
    
    # Generate smaller datasets for testing on Mac
    generate_dataset(path / "train.npz", simulator, 1000, num_planets=3, domain_shift=False)
    generate_dataset(path / "val.npz", simulator, 100, num_planets=3, domain_shift=False)
    generate_dataset(path / "eval.npz", simulator, 100, num_planets=3, domain_shift=False)
    generate_dataset(
        path / "e3_generalization.npz", simulator, 100, num_planets=3, domain_shift=True
    )
    generate_dataset(
        path / "object_generalization.npz", simulator, 100, num_planets=5, domain_shift=False
    )

    print("Done, have a nice day!")


def main():
    """Entry point for n-body dataset generation."""
    parser = argparse.ArgumentParser(description="Generate n-body dataset")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory for data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    np.random.seed(args.seed)
    generate_datasets(data_dir)


if __name__ == "__main__":
    main() 