#!/usr/bin/env python3
"""
python3/state_space_validation.py

Executes NumPy/SciPy equivalents for state-space numerical models.
Outputs JSON results to stdout for cross-language validation with Rust.
"""

from __future__ import annotations

import json
import time

import numpy as np
from scipy import signal

NUM_STEPS = 200
STIFF_STEPS = 200


def zoh_step(a_c, b_c, c_c, d_c, dt, n_steps, u_val=1.0, x0=None):
    ad, bd, cd, dd, _ = signal.cont2discrete((a_c, b_c, c_c, d_c), dt, method="zoh")
    u = np.full((n_steps, 1), u_val, dtype=np.float64)
    t = np.arange(n_steps, dtype=np.float64) * dt
    x0 = np.zeros(2, dtype=np.float64) if x0 is None else np.asarray(x0, dtype=np.float64)
    sys_d = signal.dlti(ad, bd, cd, dd, dt=dt)
    _tout, yout, xout = signal.dlsim(sys_d, u, t=t, x0=x0)
    return {
        "ad": np.asarray(ad, dtype=np.float64),
        "bd": np.asarray(bd, dtype=np.float64),
        "cd": np.asarray(cd, dtype=np.float64),
        "x1": np.asarray(xout[:n_steps, 0], dtype=np.float64),
        "x2": np.asarray(xout[:n_steps, 1], dtype=np.float64),
        "y": np.asarray(yout[:n_steps, 0], dtype=np.float64),
        "t": t,
    }


def run_state_space_oracle() -> dict:
    a_c = np.array([[0.0, 1.0], [-4.0, -0.8]], dtype=np.float64)
    b_c = np.array([[0.0], [1.0]], dtype=np.float64)
    c_c = np.array([[1.0, 0.0]], dtype=np.float64)
    d_c = np.array([[0.0]], dtype=np.float64)

    x_test = np.array([[1.0], [0.5]], dtype=np.float64)

    t0 = time.perf_counter_ns()
    x_dot = a_c @ x_test
    y_test = float((c_c @ x_test).item())
    deriv_time_ns = float(time.perf_counter_ns() - t0)

    dt = 0.05

    t0 = time.perf_counter_ns()
    ad, bd, cd, dd, _ = signal.cont2discrete((a_c, b_c, c_c, d_c), dt, method="zoh")
    zoh_time_ns = float(time.perf_counter_ns() - t0)

    t0 = time.perf_counter_ns()
    s = zoh_step(a_c, b_c, c_c, d_c, dt, NUM_STEPS, u_val=1.0, x0=[0.0, 0.0])
    step_time_ns = float(time.perf_counter_ns() - t0)

    free = zoh_step(a_c, b_c, c_c, d_c, dt, NUM_STEPS, u_val=0.0, x0=[1.0, 0.5])

    tmat = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float64)

    t0 = time.perf_counter_ns()
    t_inv = np.linalg.inv(tmat)
    a_tilde = tmat @ s["ad"] @ t_inv
    b_tilde = tmat @ s["bd"]
    c_tilde = s["cd"] @ t_inv
    similarity_time_ns = float(time.perf_counter_ns() - t0)

    # 2. Stiff plant
    a_s = np.array([[-200.0, 0.0], [0.0, -0.5]], dtype=np.float64)
    b_s = np.array([[1.0], [1.0]], dtype=np.float64)
    c_s = np.array([[1.0, 1.0]], dtype=np.float64)
    d_s = np.array([[0.0]], dtype=np.float64)

    t0 = time.perf_counter_ns()
    stiff_ad, stiff_bd, _, _, _ = signal.cont2discrete((a_s, b_s, c_s, d_s), 0.01, method="zoh")
    stiff_zoh_time_ns = float(time.perf_counter_ns() - t0)

    stiff_res = zoh_step(a_s, b_s, c_s, d_s, 0.01, STIFF_STEPS, u_val=1.0, x0=[0.0, 0.0])

    return {
        "tutorial": {
            "x_dot": x_dot.tolist(),
            "y_test": y_test,
            "ad": s["ad"].tolist(),
            "bd": s["bd"].tolist(),
            "step_x1": s["x1"].tolist(),
            "step_x2": s["x2"].tolist(),
            "step_y": s["y"].tolist(),
            "free_x1": free["x1"].tolist(),
            "free_x2": free["x2"].tolist(),
            "a_tilde": a_tilde.tolist(),
            "b_tilde": b_tilde.tolist(),
            "c_tilde": c_tilde.tolist(),
            "deriv_time_ns": deriv_time_ns,
            "zoh_time_ns": zoh_time_ns,
            "step_time_ns": step_time_ns,
            "similarity_time_ns": similarity_time_ns,
        },
        "stiff": {
            "ad": stiff_res["ad"].tolist(),
            "bd": stiff_res["bd"].tolist(),
            "y": stiff_res["y"].tolist(),
            "stiff_zoh_time_ns": stiff_zoh_time_ns,
        },
    }


if __name__ == "__main__":
    results = run_state_space_oracle()
    print(json.dumps(results))
