#!/usr/bin/env python3
"""State-space numerical prototype oracle (SciPy signal)."""

from __future__ import annotations

import numpy as np
from scipy import signal


def scenario() -> dict[str, object]:
    a_c = np.array([[0.0, 1.0], [-4.0, -0.8]], dtype=np.float64)
    b_c = np.array([[0.0], [1.0]], dtype=np.float64)
    c_c = np.array([[1.0, 0.0]], dtype=np.float64)
    d_c = np.array([[0.0]], dtype=np.float64)
    x_test = np.array([[1.0], [0.5]], dtype=np.float64)
    x_dot = a_c @ x_test
    y_test = float((c_c @ x_test).item())
    dt = 0.05
    ad, bd, cd, dd, _ = signal.cont2discrete(
        (a_c, b_c, c_c, d_c), dt, method="zoh"
    )
    n_steps = 20
    u = np.ones((n_steps, 1), dtype=np.float64)
    t = np.arange(n_steps, dtype=np.float64) * dt
    sys_d = signal.dlti(ad, bd, cd, dd, dt=dt)
    _tout, yout, xout = signal.dlsim(sys_d, u, t=t, x0=np.zeros(2))
    # dlsim returns xout with one extra terminal state.
    x1 = np.asarray(xout[:n_steps, 0], dtype=np.float64)
    x2 = np.asarray(xout[:n_steps, 1], dtype=np.float64)
    y = np.asarray(yout[:n_steps, 0], dtype=np.float64)
    tmat = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    t_inv = np.array([[1.0, -1.0], [0.0, 1.0]], dtype=np.float64)
    a_tilde = tmat @ ad @ t_inv
    b_tilde = tmat @ bd
    c_tilde = cd @ t_inv
    return {
        "a_c": a_c,
        "b_c": b_c,
        "c_c": c_c,
        "d_c": d_c,
        "x_dot": x_dot,
        "y_test": y_test,
        "dt": dt,
        "ad": np.asarray(ad, dtype=np.float64),
        "bd": np.asarray(bd, dtype=np.float64),
        "cd": np.asarray(cd, dtype=np.float64),
        "dd": np.asarray(dd, dtype=np.float64),
        "step_x1": x1,
        "step_x2": x2,
        "step_y": y,
        "a_tilde": a_tilde,
        "b_tilde": b_tilde,
        "c_tilde": c_tilde,
    }


def equiv() -> dict[str, object]:
    s = scenario()
    return {
        "X_DOT": s["x_dot"],
        "Y_TEST": np.array([s["y_test"]], dtype=np.float64),
        "AD": s["ad"],
        "BD": s["bd"],
        "STEP_X1": s["step_x1"],
        "STEP_X2": s["step_x2"],
        "STEP_Y": s["step_y"],
        "A_TILDE": s["a_tilde"],
        "B_TILDE": s["b_tilde"],
        "C_TILDE": s["c_tilde"],
    }


def _print_mat(name: str, m: np.ndarray) -> None:
    print(f"{name}:")
    for row in np.atleast_2d(m):
        print("  [" + ", ".join(f"{val:12.6f}" for val in row) + "]")


def print_transcript() -> None:
    s = scenario()
    print("=== State-Space Numerical Prototype Oracle ===")
    print("\n--- Continuous-Time System ---")
    _print_mat("A_c", s["a_c"])
    _print_mat("B_c", s["b_c"])
    _print_mat("C_c", s["c_c"])
    _print_mat("D_c", s["d_c"])
    _print_mat("x_dot at [1.0, 0.5]^T", s["x_dot"])
    print(f"y at [1.0, 0.5]^T: {s['y_test']:.6f}")
    print(f"\n--- Discrete-Time System (ZOH, Ts = {s['dt']}s) ---")
    _print_mat("A_d", s["ad"])
    _print_mat("B_d", s["bd"])
    _print_mat("C_d", s["cd"])
    _print_mat("D_d", s["dd"])
    print("\n--- 20-Step Unit Step Trajectory ---")
    print(f"{'Step':<6}{'x_1 (pos)':<16}{'x_2 (vel)':<16}{'y (output)':<16}")
    for k in range(20):
        print(
            f"{k:<6}{s['step_x1'][k]:<16.8f}"
            f"{s['step_x2'][k]:<16.8f}{s['step_y'][k]:<16.8f}"
        )
    print("\n--- Transformed System (T = [[1, 1], [0, 1]]) ---")
    _print_mat("A_tilde", s["a_tilde"])
    _print_mat("B_tilde", s["b_tilde"])
    _print_mat("C_tilde", s["c_tilde"])


if __name__ == "__main__":
    print_transcript()
