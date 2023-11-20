# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
import numpy as np
from scipy.stats import special_ortho_group

from gatr.utils.misc import sample_log_uniform, sample_uniform_in_circle


class NBodySimulator:
    """Simulator for the n-body dataset.

    Each sample consists of the positions of a n particicles (1 star and n - 1 planets), both before
    and after the evolution of 100 time steps. The particle masses are also part of the sample.

    The data is generated in the following way:
    1. The masses for the star and planets are sampled.
    2. The initial planet positions are sampled around the origin in the x-y plane. The star is at
       (0, 0).
    3. The initial planet velocities are given by the velocity of a stable circular orbit, plus some
       noise.
       The star is initially at rest.
    4. The initial state is translated and rotated to an arbitrary position and orientation in
       space.
    5. The final state is computed (using Newton's equations of gravity and Newton's method).
    6. To avoid annoying outliers, samples where the initial and final position of any object
    are more than outlier_threshold are removed.

    We use units where G = 1.

    Parameters
    ----------
    time : float
        Overall time period between initial and final state.
    time_steps : int
        Number of time steps that the time window is divided into for Euler integration.
    star_mass_range : tuple of float
        Minimum and maximum mass for the star.
    planet_mass_range : tuple of float
        Minimum and maximum mass for the planets.
    radius_range : tuple of float
        Minimum and maximum initial distance of the planets from the star.
    vel_std : float
        Noise scale for the initial planet velocity.
    shift_std : float
        Noise scale for the spatial translation.
    ood_shift : float
        Noise scale for the spatial translation for the translated sample.
    outlier_threshold : float
        Threshold for the rejection of outlier samples in which initial and final positions differ
        substantially.
    """

    def __init__(
        self,
        time=0.1,
        time_steps=100,
        star_mass_range=(1.0, 10.0),
        planet_mass_range=(0.01, 0.1),
        radius_range=(0.1, 1.0),
        vel_std=0.01,
        shift_std=20.0,
        ood_shift=200.0,
        outlier_threshold=2.0,
    ):
        self.time = time
        self.delta_t = time / time_steps
        self.time_steps = time_steps
        self.star_mass_range = star_mass_range
        self.planet_mass_range = planet_mass_range
        self.radius_range = radius_range
        self.vel_std = vel_std
        self.shift_std = shift_std
        self.ood_shift = ood_shift
        self.outlier_threshold = outlier_threshold

    def sample(self, num_samples, num_planets=5, domain_shift=False):
        """Samples from gravity problem.

        Parameters
        ----------
        num_samples : int
            Number of samples.
        num_planets : int
            Number of planets in the dataset (as there is always a single star, we have
            `n = num_planets + 1`).
        domain_shift : bool
            Whether to sample with larger spatial translation to generate domain-shifted data.
        """

        # Sample a bit more initially, to make up for outlier removal later
        num_candidates = int(round(1.1 * num_samples))

        # 1. Sample planet masses
        star_mass = sample_log_uniform(*self.star_mass_range, size=(num_candidates, 1))
        planet_masses = sample_log_uniform(
            *self.planet_mass_range, size=(num_candidates, num_planets)
        )
        masses = np.concatenate((star_mass, planet_masses), axis=1)

        # 2. Sample initial positions in x-y plane around origin
        planet_pos = np.zeros((num_candidates, num_planets, 3))
        planet_pos[..., :2] = sample_uniform_in_circle(
            num_candidates * num_planets,
            min_radius=self.radius_range[0],
            max_radius=self.radius_range[1],
        ).reshape((num_candidates, num_planets, 2))
        x_initial = np.concatenate((np.zeros((num_candidates, 1, 3)), planet_pos), axis=1)

        # 3. Sample initial velocities, approximately as stable circular orbits
        planet_vel = self._sample_planet_velocities(star_mass, planet_pos)
        v_initial = np.concatenate((np.zeros((num_candidates, 1, 3)), planet_vel), axis=1)

        # 4. Translate, rotate, permute
        masses, x_initial, v_initial = self._shift_and_rotate(
            masses, x_initial, v_initial, domain_shift=domain_shift
        )

        # 5. Evolve Newtonian gravity
        x_final, trajectory = self._simulate(masses, x_initial, v_initial)

        # 6. Remove outliers
        max_distance = np.max(np.linalg.norm(x_final - x_initial, axis=-1), axis=-1)
        mask = max_distance <= self.outlier_threshold
        masses, x_initial, v_initial, x_final, trajectory = (
            masses[mask],
            x_initial[mask],
            v_initial[mask],
            x_final[mask],
            trajectory[mask],
        )

        # Cut down to requested number
        masses, x_initial, v_initial, x_final, trajectory = (
            masses[:num_samples],
            x_initial[:num_samples],
            v_initial[:num_samples],
            x_final[:num_samples],
            trajectory[:num_samples],
        )

        return masses, x_initial, v_initial, x_final, trajectory

    def _sample_planet_velocities(self, star_mass, x):
        """Samples planet velocities around those that give stable circular orbits."""

        batchsize, num_planets, _ = x.shape

        # Defines rotation plane. Clockwise or counterclockwise is random for each planet.
        orientation = np.zeros((batchsize, num_planets, 3))
        orientation[:, :, 2] = np.random.choice(
            a=[-1.0, 1.0], size=(batchsize, num_planets), p=[0.5, 0.5]
        )

        # Compute stable velocities
        star_mass = star_mass[:, :, np.newaxis]  # (batchsize, 1, 1)
        radii = np.linalg.norm(x, axis=-1)[:, :, np.newaxis]  # (batchsize, num_planets, 1)
        v_stable = np.cross(orientation, x) * star_mass**0.5 / radii**1.5

        # Add noise
        v = v_stable + np.random.normal(scale=self.vel_std, size=(batchsize, num_planets, 3))

        return v

    def _shift_and_rotate(self, m, x, v, domain_shift=False):
        """Performs random E(3) transformations and permutations on given positions / velocities."""

        batchsize, num_objects, _ = x.shape

        # Permutations over objects
        for i in range(batchsize):
            perm = np.random.permutation(num_objects)
            m[i] = m[i][perm]
            x[i] = x[i][perm, :]
            v[i] = v[i][perm, :]

        # Rotations from Haar measure
        rotations = special_ortho_group(3).rvs(size=batchsize).reshape(batchsize, 1, 3, 3)
        x = np.einsum("bnij,bnj->bni", rotations, x)
        v = np.einsum("bnij,bnj->bni", rotations, v)

        # Translations
        shifts = np.random.normal(scale=self.shift_std, size=(batchsize, 1, 3))
        x = x + shifts

        # OOD shift
        if domain_shift:
            shifts = np.array([self.ood_shift, 0, 0]).reshape((1, 1, 3))
            x = x + shifts

        return m, x, v

    def _simulate(self, m, x_initial, v_initial):
        """Evolves an initial state under Newtonian equations of motions."""

        x, v = x_initial, v_initial
        trajectory = [x_initial]

        for _ in range(self.time_steps):
            a = self._compute_accelerations(m, x)
            v = v + self.delta_t * a
            x = x + self.delta_t * v
            trajectory.append(x)

        return x, np.array(trajectory).transpose([1, 2, 0, 3])

    @staticmethod
    def _compute_accelerations(m, x):
        """Computes accelerations for a set of point masses according to Newtonian gravity."""
        batchsize, num_objects, _ = x.shape
        mm = m.reshape((batchsize, 1, num_objects, 1)) * m.reshape(
            (batchsize, num_objects, 1, 1)
        )  # (b, n, n, 1)

        distance_vectors = x.reshape((batchsize, 1, num_objects, 3)) - x.reshape(
            (batchsize, num_objects, 1, 3)
        )  # (b, n, n, 3)
        distances = np.linalg.norm(distance_vectors, axis=-1)[:, :, :, np.newaxis]  # (b, n, n, 1)
        distances[np.abs(distances) < 1e-9] = 1.0

        forces = distance_vectors * mm / distances**3  # (b, n, n, 3)
        accelerations = np.sum(forces, axis=2) / m.reshape(batchsize, num_objects, 1)  # (b, n, 3)

        return accelerations  # (b, n, 3)
