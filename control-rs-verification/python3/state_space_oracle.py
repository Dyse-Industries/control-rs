#!/usr/bin/env python3
"""State-space reference oracle generating results/state_space.scipy.h5 via SciPy."""

from pathlib import Path
import numpy as np
from scipy import signal

from h5_writer import get_results_dir, write_h5


class PendulumSim:
    def __init__(self, omega0: float = 2.0, b: float = 0.8, dt: float = 0.05):
        omega0_sq = omega0**2
        self.a_c = np.array([[0.0, 1.0], [-omega0_sq, -b]], dtype=np.float64)
        self.b_c = np.array([[0.0], [1.0]], dtype=np.float64)
        self.c_c = np.array([[1.0, 0.0]], dtype=np.float64)
        self.d_c = np.array([[0.0]], dtype=np.float64)

        self.ad, self.bd, self.cd, self.dd, _ = signal.cont2discrete(
            (self.a_c, self.b_c, self.c_c, self.d_c), dt, method="zoh"
        )

    def simulate(self, x0, n_steps=200, u_val=0.0):
        x_k = np.asarray(x0, dtype=np.float64).reshape(2, 1)
        u_k = np.array([[u_val]], dtype=np.float64)
        theta = []
        theta_dot = []
        for _ in range(n_steps):
            theta.append(float(x_k[0, 0]))
            theta_dot.append(float(x_k[1, 0]))
            x_k = self.ad @ x_k + self.bd @ u_k
        return theta, theta_dot

    def step_response(self, n_steps=100):
        x_k = np.zeros((2, 1), dtype=np.float64)
        u_k = np.array([[1.0]], dtype=np.float64)
        step_data = []
        for _ in range(n_steps):
            y_k = self.cd @ x_k + self.dd @ u_k
            step_data.append(float(y_k[0, 0]))
            x_k = self.ad @ x_k + self.bd @ u_k
        return step_data


def generate_datasets():
    sim = PendulumSim(omega0=2.0, b=0.8, dt=0.05)
    theta, theta_dot = sim.simulate(x0=[np.pi - 0.15, 0.5], n_steps=200, u_val=0.0)
    step_data = sim.step_response(100)

    datasets = {
        "phase_portrait/theta": theta,
        "phase_portrait/theta_dot": theta_dot,
        "transient/step_data": step_data,
        "discretization/a_d": sim.ad.flatten(),
        "discretization/b_d": sim.bd.flatten(),
    }

    tolerances = {
        "phase_portrait/theta": ("abs", 1e-4),
        "phase_portrait/theta_dot": ("abs", 1e-4),
        "transient/step_data": ("abs", 1e-4),
        "discretization/a_d": ("abs", 1e-4),
        "discretization/b_d": ("abs", 1e-4),
    }

    return datasets, tolerances


def main():
    datasets, tolerances = generate_datasets()
    out_file = get_results_dir() / "state_space.scipy.h5"
    write_h5(out_file, datasets, tolerances)


if __name__ == "__main__":
    main()
