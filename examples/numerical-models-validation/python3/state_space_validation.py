#!/usr/bin/env python3
"""
python3/state_space_validation.py

Executes NumPy/SciPy equivalents for state-space numerical models.
Outputs JSON results to stdout for cross-language validation with Rust.
"""

from __future__ import annotations

import json

import numpy as np
import time
from scipy import signal


class PendulumSim:
    """
    State-space pendulum simulation model.
    State vector x = [theta, theta_dot]^T
    dx/dt = A_c * x + B_c * u
    Discretized via Zero-Order Hold (ZOH).
    """

    def __init__(self, omega0: float = 2.0, b: float = 0.8, dt: float = 0.05):
        self.omega0_sq = omega0 ** 2
        self.b = b
        self.dt = dt

        # Continuous state-space matrices for pendulum dynamics
        self.a_c = np.array([[0.0, 1.0], [-self.omega0_sq, -self.b]], dtype=np.float64)
        self.b_c = np.array([[0.0], [1.0]], dtype=np.float64)
        self.c_c = np.array([[1.0, 0.0]], dtype=np.float64)
        self.d_c = np.array([[0.0]], dtype=np.float64)

        # Discretize continuous dynamics using Zero-Order Hold (ZOH)
        self.ad, self.bd, self.cd, self.dd, _ = signal.cont2discrete(
            (self.a_c, self.b_c, self.c_c, self.d_c), self.dt, method="zoh"
        )

    def simulate(
            self,
            x0: tuple[float, float] | list[float] | np.ndarray,
            n_steps: int = 200,
            u_val: float = 0.0,
    ) -> tuple[list[float], list[float]]:
        """Simulate discrete pendulum trajectory over n_steps."""
        x_k = np.asarray(x0, dtype=np.float64).reshape(2, 1)
        u_k = np.array([[u_val]], dtype=np.float64)

        theta = []
        theta_dot = []

        for _ in range(n_steps):
            theta.append(float(x_k[0, 0]))
            theta_dot.append(float(x_k[1, 0]))
            x_k = self.ad @ x_k + self.bd @ u_k

        return theta, theta_dot


def generate_state_space_correctness_data() -> dict:
    sim = PendulumSim(omega0=2.0, b=0.8, dt=0.05)
    theta, theta_dot = sim.simulate(x0=[np.pi - 0.15, 0.5], n_steps=200, u_val=0.0)

    return {
        "phase_portrait": {
            "theta": theta,
            "theta_dot": theta_dot
        }
    }


def benchmark_discretization_scaling() -> dict:
    state_sizes = [2, 4, 8, 16, 32, 64, 128]
    zoh_times = []

    for N in state_sizes:
        A = np.zeros((N, N), dtype=np.float64)
        B = np.zeros((N, 1), dtype=np.float64)
        C = np.zeros((1, N), dtype=np.float64)
        D = np.zeros((1, 1), dtype=np.float64)

        for i in range(N):
            for j in range(N):
                A[i, j] = -0.5 * (i + 1) if i == j else 0.1 / (i + j + 1)
            B[i, 0] = 1.0 / (i + 1)
            C[0, i] = 1.0 / (i + 1)

        t0 = time.perf_counter_ns()
        _ad, _bd, _, _, _ = signal.cont2discrete((A, B, C, D), 0.05, method="zoh")
        zoh_time = float(time.perf_counter_ns() - t0)

        zoh_times.append(zoh_time)

    return {
        "scaling": {
            "state_size": state_sizes,
            "zoh_time_ns": zoh_times
        }
    }


def benchmark_step_response_jitter() -> dict:
    sim = PendulumSim(omega0=2.0, b=0.8, dt=0.05)

    x_k = np.zeros((2, 1), dtype=np.float64)
    u_k = np.array([[1.0]], dtype=np.float64)

    iterations = 100
    step_times = []
    inputs = []
    step_data = []

    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        y_k = sim.cd @ x_k + sim.dd @ u_k
        x_next = sim.ad @ x_k + sim.bd @ u_k
        t_elapsed = float(time.perf_counter_ns() - t0)

        step_times.append(t_elapsed)
        inputs.append(1.0)
        step_data.append(float(y_k.item()))

        x_k = x_next

    return {
        "jitter": {
            "step_compute_times_ns": step_times,
            "input": inputs,
            "step_data": step_data,
            "step-data": step_data
        }
    }


def benchmark_controllability_observability() -> dict:
    state_sizes = [2, 4, 8, 16, 32, 64, 128]
    ctrb_times = []
    obsv_times = []

    for N in state_sizes:
        A = np.zeros((N, N), dtype=np.float64)
        B = np.zeros((N, 1), dtype=np.float64)
        C = np.zeros((1, N), dtype=np.float64)

        for i in range(N):
            for j in range(N):
                A[i, j] = -0.5 * (i + 1) if i == j else 0.1 / (i + j + 1)
            B[i, 0] = 1.0 / (i + 1)
            C[0, i] = 1.0 / (i + 1)

        # Controllability matrix
        t0 = time.perf_counter_ns()
        cols = [B]
        curr_b = B
        for _ in range(1, N):
            curr_b = A @ curr_b
            cols.append(curr_b)
        _ctrb_mat = np.hstack(cols)
        ctrb_time = float(time.perf_counter_ns() - t0)

        # Observability matrix
        t0 = time.perf_counter_ns()
        rows = [C]
        curr_c = C
        for _ in range(1, N):
            curr_c = curr_c @ A
            rows.append(curr_c)
        _obsv_mat = np.vstack(rows)
        obsv_time = float(time.perf_counter_ns() - t0)

        ctrb_times.append(ctrb_time)
        obsv_times.append(obsv_time)

    return {
        "state_size": state_sizes,
        "controllability_time_ns": ctrb_times,
        "observability_time_ns": obsv_times
    }


def run_state_space_oracle() -> dict:
    q1 = generate_state_space_correctness_data()
    q2 = benchmark_discretization_scaling()
    q3 = benchmark_step_response_jitter()
    q4 = benchmark_controllability_observability()

    return {
        "phase_portrait": q1["phase_portrait"],
        "scaling": q2["scaling"],
        "jitter": q3["jitter"],
        "control_loop": q4,
        "state_size": q4["state_size"],
        "controllability_time_ns": q4["controllability_time_ns"],
        "observability_time_ns": q4["observability_time_ns"]
    }


def run_harold_oracle() -> dict:
    import harold

    omega0, b, dt = 2.0, 0.8, 0.05
    A_c = np.array([[0., 1.], [-omega0 ** 2, -b]])
    B_c = np.array([[0.], [1.]])
    C_c = np.array([[1., 0.]])
    D_c = np.array([[0.]])

    sys_c = harold.State(A_c, B_c, C_c, D_c)
    sys_d = harold.discretize(sys_c, dt, method='zoh')

    # Phase portrait: manual state recursion with the discretized matrices
    Ad = np.array(sys_d.a)
    Bd = np.array(sys_d.b)
    Cd = np.array(sys_d.c)
    x_k = np.array([[np.pi - 0.15], [0.5]])
    u_k = np.zeros((1, 1))

    theta = []
    theta_dot = []
    for _ in range(200):
        theta.append(float(x_k[0, 0]))
        theta_dot.append(float(x_k[1, 0]))
        x_k = Ad @ x_k + Bd @ u_k

    # Step response: 100 steps with unit input
    u_step = np.ones((100, 1))
    y_step, t_step = harold.simulate_linear_system(sys_d, u_step)
    step_data = y_step.flatten().tolist()

    # Controllability and observability timing across sizes
    state_sizes = [2, 4, 8, 16, 32, 64, 128]
    ctrb_times = []
    obsv_times = []

    for N in state_sizes:
        A = np.zeros((N, N))
        B = np.zeros((N, 1))
        C = np.zeros((1, N))
        D = np.zeros((1, 1))
        for i in range(N):
            for j in range(N):
                A[i, j] = -0.5 * (i + 1) if i == j else 0.1 / (i + j + 1)
            B[i, 0] = 1.0 / (i + 1)
            C[0, i] = 1.0 / (i + 1)
        sys_n = harold.State(A, B, C, D)

        t0 = time.perf_counter_ns()
        harold.controllability_matrix(sys_n)
        ctrb_times.append(float(time.perf_counter_ns() - t0))

        t0 = time.perf_counter_ns()
        harold.observability_matrix(sys_n)
        obsv_times.append(float(time.perf_counter_ns() - t0))

    return {
        "phase_portrait": {"theta": theta, "theta_dot": theta_dot},
        "jitter": {"step_data": step_data},
        "control_loop": {
            "state_size": state_sizes,
            "controllability_time_ns": ctrb_times,
            "observability_time_ns": obsv_times,
        },
        "state_size": state_sizes,
        "controllability_time_ns": ctrb_times,
        "observability_time_ns": obsv_times,
    }


if __name__ == "__main__":
    scipy_results = run_state_space_oracle()
    harold_results = run_harold_oracle()
    print(json.dumps({"scipy": scipy_results, "harold": harold_results}))