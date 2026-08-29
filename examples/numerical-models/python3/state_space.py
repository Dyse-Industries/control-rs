#!/usr/bin/env python3
"""State-space numerical oracle — suite path in, result JSON on stdout."""

from __future__ import annotations

import numpy as np
from scipy import signal

from vv import case_inputs, require_int, run_cli, time_kernel, timing_entry

NUM_STEPS = 200
STIFF_STEPS = 200


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


def tutorial(suite: dict) -> dict:
    inp = case_inputs(suite, "state_space.host.tutorial_plant")
    require_int(inp, "n_steps", NUM_STEPS)
    a_c = np.array(inp["A"], dtype=np.float64)
    b_c = np.array(inp["B"], dtype=np.float64)
    c_c = np.array(inp["C"], dtype=np.float64)
    d_c = np.array(inp["D"], dtype=np.float64)
    x_test = np.array(inp["x_test"], dtype=np.float64).reshape(2, 1)
    x_dot = a_c @ x_test
    y_test = float((c_c @ x_test).item())
    dt = float(inp["Ts"])
    s = zoh_step(
        a_c, b_c, c_c, d_c, dt, NUM_STEPS,
        u_val=float(inp["step_u"]), x0=inp["step_x0"],
    )
    free = zoh_step(
        a_c, b_c, c_c, d_c, dt, NUM_STEPS,
        u_val=float(inp["free_u"]), x0=inp["free_x0"],
    )
    tmat = np.array(inp["T"], dtype=np.float64)
    t_inv = np.linalg.inv(tmat)
    a_tilde = tmat @ s["ad"] @ t_inv
    b_tilde = tmat @ s["bd"]
    c_tilde = s["cd"] @ t_inv
    zoh_iters = int(inp.get("zoh_iters", 20))
    step_iters = int(inp.get("step_iters", 20))
    zoh_ns = time_kernel(
        zoh_iters,
        lambda: signal.cont2discrete((a_c, b_c, c_c, d_c), dt, method="zoh"),
    )
    step_ns = time_kernel(
        step_iters,
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
        "zoh_iters": zoh_iters,
        "step_iters": step_iters,
    }


def stiff(suite: dict) -> dict:
    inp = case_inputs(suite, "state_space.host.stiff_zoh")
    require_int(inp, "n_steps", STIFF_STEPS)
    a_c = np.array(inp["A"], dtype=np.float64)
    b_c = np.array(inp["B"], dtype=np.float64)
    c_c = np.array(inp["C"], dtype=np.float64)
    d_c = np.array(inp["D"], dtype=np.float64)
    return zoh_step(
        a_c, b_c, c_c, d_c, float(inp["Ts"]), STIFF_STEPS,
        u_val=float(inp["u"]), x0=inp["x0"],
    )


def build_artifact(suite: dict) -> dict:
    t = tutorial(suite)
    s = stiff(suite)
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
            "zoh": timing_entry(t["zoh_iters"], t["zoh_ns"]),
            "step": timing_entry(t["step_iters"], t["step_ns"]),
        },
    }


if __name__ == "__main__":
    run_cli("state_space", build_artifact)
