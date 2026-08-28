#!/usr/bin/env python3
"""State-space numerical oracle — writes results/state_space/python.json."""

from __future__ import annotations

import numpy as np
from scipy import signal

from vv import CRATE_ROOT, save_json, time_kernel, timing_entry

OUT_PATH = CRATE_ROOT / "results" / "state_space" / "python.json"
NUM_STEPS = 200
STIFF_STEPS = 200
ZOH_ITERS = 20
STEP_ITERS = 20


def zoh_step(a_c, b_c, c_c, d_c, dt, n_steps, u_val=1.0, x0=None):
    ad, bd, cd, dd, _ = signal.cont2discrete(
        (a_c, b_c, c_c, d_c), dt, method="zoh"
    )
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


def tutorial() -> dict:
    a_c = np.array([[0.0, 1.0], [-4.0, -0.8]], dtype=np.float64)
    b_c = np.array([[0.0], [1.0]], dtype=np.float64)
    c_c = np.array([[1.0, 0.0]], dtype=np.float64)
    d_c = np.array([[0.0]], dtype=np.float64)
    x_test = np.array([[1.0], [0.5]], dtype=np.float64)
    x_dot = a_c @ x_test
    y_test = float((c_c @ x_test).item())
    dt = 0.05
    s = zoh_step(a_c, b_c, c_c, d_c, dt, NUM_STEPS)
    free = zoh_step(a_c, b_c, c_c, d_c, dt, NUM_STEPS, u_val=0.0, x0=[1.0, 0.5])
    tmat = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    t_inv = np.array([[1.0, -1.0], [0.0, 1.0]], dtype=np.float64)
    a_tilde = tmat @ s["ad"] @ t_inv
    b_tilde = tmat @ s["bd"]
    c_tilde = s["cd"] @ t_inv
    zoh_ns = time_kernel(
        ZOH_ITERS,
        lambda: signal.cont2discrete((a_c, b_c, c_c, d_c), dt, method="zoh"),
    )
    step_ns = time_kernel(
        STEP_ITERS,
        lambda: zoh_step(a_c, b_c, c_c, d_c, dt, NUM_STEPS),
    )
    return {
        "x_dot": x_dot,
        "y_test": y_test,
        "ad": s["ad"],
        "bd": s["bd"],
        "step_x1": s["x1"],
        "step_x2": s["x2"],
        "step_y": s["y"],
        "free_x1": free["x1"],
        "free_x2": free["x2"],
        "a_tilde": a_tilde,
        "b_tilde": b_tilde,
        "c_tilde": c_tilde,
        "t": s["t"],
        "zoh_ns": zoh_ns,
        "step_ns": step_ns,
    }


def stiff() -> dict:
    a_c = np.array([[-200.0, 0.0], [0.0, -0.5]], dtype=np.float64)
    b_c = np.array([[1.0], [1.0]], dtype=np.float64)
    c_c = np.array([[1.0, 1.0]], dtype=np.float64)
    d_c = np.array([[0.0]], dtype=np.float64)
    return zoh_step(a_c, b_c, c_c, d_c, 0.01, STIFF_STEPS)


def build_artifact() -> dict:
    t = tutorial()
    s = stiff()
    return {
        "slug": "state_space",
        "source": "python",
        "values": {
            "X_DOT": t["x_dot"].tolist(),
            "Y_TEST": t["y_test"],
            "AD": t["ad"].tolist(),
            "BD": t["bd"].tolist(),
            "STEP_X1": t["step_x1"].tolist(),
            "STEP_X2": t["step_x2"].tolist(),
            "STEP_Y": t["step_y"].tolist(),
            "FREE_X1": t["free_x1"].tolist(),
            "FREE_X2": t["free_x2"].tolist(),
            "A_TILDE": t["a_tilde"].tolist(),
            "B_TILDE": t["b_tilde"].tolist(),
            "C_TILDE": t["c_tilde"].tolist(),
            "STIFF_AD": s["ad"].tolist(),
            "STIFF_BD": s["bd"].tolist(),
            "STIFF_Y": s["y"].tolist(),
        },
        "series": {
            "step_y": {"x": t["t"].tolist(), "y": t["step_y"].tolist()},
            "stiff_y": {"x": s["t"].tolist(), "y": s["y"].tolist()},
            "free_x1": {"x": t["t"].tolist(), "y": t["free_x1"].tolist()},
            "free_x2": {"x": t["t"].tolist(), "y": t["free_x2"].tolist()},
        },
        "metrics": {},
        "timings": {
            "zoh": timing_entry(ZOH_ITERS, t["zoh_ns"]),
            "step": timing_entry(STEP_ITERS, t["step_ns"]),
        },
    }


if __name__ == "__main__":
    save_json(OUT_PATH, build_artifact())
