#!/usr/bin/env python3
"""State-space numerical oracle — writes results/state_space/python.json."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import signal

CRATE_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = CRATE_ROOT / "results" / "state_space" / "python.json"


def scenario() -> dict[str, object]:
    a_c = np.array([[0.0, 1.0], [-4.0, -0.8]], dtype=np.float64)
    b_c = np.array([[0.0], [1.0]], dtype=np.float64)
    c_c = np.array([[1.0, 0.0]], dtype=np.float64)
    d_c = np.array([[0.0]], dtype=np.float64)
    x_test = np.array([[1.0], [0.5]], dtype=np.float64)
    x_dot = a_c @ x_test
    y_test = float((c_c @ x_test).item())
    dt = 0.05
    ad, bd, cd, dd, _ = signal.cont2discrete((a_c, b_c, c_c, d_c), dt, method="zoh")
    n_steps = 20
    u = np.ones((n_steps, 1), dtype=np.float64)
    t = np.arange(n_steps, dtype=np.float64) * dt
    sys_d = signal.dlti(ad, bd, cd, dd, dt=dt)
    _tout, yout, xout = signal.dlsim(sys_d, u, t=t, x0=np.zeros(2))
    x1 = np.asarray(xout[:n_steps, 0], dtype=np.float64)
    x2 = np.asarray(xout[:n_steps, 1], dtype=np.float64)
    y = np.asarray(yout[:n_steps, 0], dtype=np.float64)
    tmat = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    t_inv = np.array([[1.0, -1.0], [0.0, 1.0]], dtype=np.float64)
    a_tilde = tmat @ ad @ t_inv
    b_tilde = tmat @ bd
    c_tilde = cd @ t_inv
    return {
        "x_dot": x_dot,
        "y_test": y_test,
        "ad": np.asarray(ad, dtype=np.float64),
        "bd": bd,
        "step_x1": x1,
        "step_x2": x2,
        "step_y": y,
        "a_tilde": a_tilde,
        "b_tilde": b_tilde,
        "c_tilde": c_tilde,
        "dt": dt,
        "t": t,
    }


def build_artifact() -> dict:
    s = scenario()
    return {
        "slug": "state_space",
        "source": "python",
        "values": {
            "X_DOT": s["x_dot"].tolist(),
            "Y_TEST": s["y_test"],
            "AD": s["ad"].tolist(),
            "BD": s["bd"].tolist(),
            "STEP_X1": s["step_x1"].tolist(),
            "STEP_X2": s["step_x2"].tolist(),
            "STEP_Y": s["step_y"].tolist(),
            "A_TILDE": s["a_tilde"].tolist(),
            "B_TILDE": s["b_tilde"].tolist(),
            "C_TILDE": s["c_tilde"].tolist(),
        },
        "series": {
            "step_y": {"x": s["t"].tolist(), "y": s["step_y"].tolist()},
            "step_x1": {"x": s["t"].tolist(), "y": s["step_x1"].tolist()},
            "step_x2": {"x": s["t"].tolist(), "y": s["step_x2"].tolist()},
        },
    }


if __name__ == "__main__":
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(build_artifact(), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT_PATH}")
