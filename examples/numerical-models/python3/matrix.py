#!/usr/bin/env python3
"""Matrix numerical oracle — suite path in, result JSON on stdout."""

from __future__ import annotations

import numpy as np
from scipy import linalg

from vv import (
    case_inputs,
    require_int,
    residual_ratio,
    run_cli,
    time_kernel,
    timing_entry,
)

GEMM_N = 64
HILBERT_N = 8
SE3_N = 40


def tutorial(suite: dict) -> dict:
    arith = case_inputs(suite, "matrix.host.arithmetic")
    lu_in = case_inputs(suite, "matrix.host.lu_solve_inverse")
    m1 = np.array(arith["M1"], dtype=np.float64)
    m2 = np.array(arith["M2"], dtype=np.float64)
    a = np.array(lu_in["A"], dtype=np.float64)
    b = np.array(lu_in["b"], dtype=np.float64)
    x = linalg.solve(a, b)
    a_inv = linalg.inv(a)
    return {
        "sum": m1 + m2,
        "diff": m2 - m1,
        "prod": m1 @ m2,
        "transpose": m1.T.copy(),
        "a": a,
        "b": b,
        "x": x.reshape(3, 1),
        "a_inv": a_inv,
    }


def hilbert_case(suite: dict) -> dict:
    inp = case_inputs(suite, "matrix.host.hilbert")
    require_int(inp, "n", HILBERT_N)
    h = np.array(
        [
            [1.0 / (i + j + 1) for j in range(HILBERT_N)]
            for i in range(HILBERT_N)
        ],
        dtype=np.float64,
    )
    x_true = np.ones(HILBERT_N, dtype=np.float64)
    b = h @ x_true
    x = linalg.solve(h, b)
    h_inv = linalg.inv(h)
    kappa = float(np.linalg.cond(h, np.inf))
    return {
        "h": h,
        "b": b,
        "x": x,
        "h_inv": h_inv,
        "kappa": kappa,
        "residual_ratio": residual_ratio(h, x, b),
    }


def gemm_case(suite: dict) -> dict:
    inp = case_inputs(suite, "matrix.host.gemm")
    require_int(inp, "n", GEMM_N)
    gemm_iters = int(inp.get("iters", 200))
    i = np.arange(GEMM_N, dtype=np.float64)
    ii, jj = np.meshgrid(i, i, indexing="ij")
    ga = 0.01 * (ii + 1.0) * (jj + 3.0) / 64.0
    gb = 0.02 * (ii + 2.0) * (jj + 1.0) / 64.0
    with np.errstate(all="ignore"):
        gc = ga @ gb
        ns = time_kernel(gemm_iters, lambda: ga @ gb)
    return {
        "gemm00": float(gc[0, 0]),
        "gemm_frob": float(np.linalg.norm(gc, ord="fro")),
        "ns": ns,
        "iters": gemm_iters,
    }


def se3_chain(suite: dict) -> dict:
    inp = case_inputs(suite, "matrix.host.se3_chain")
    require_int(inp, "n", SE3_N)
    theta = float(inp["theta"])
    dx, dy, dz = (float(v) for v in inp["t"])
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    t_mat = np.array(
        [
            [c, -s, 0.0, dx],
            [s, c, 0.0, dy],
            [0.0, 0.0, 1.0, dz],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    pose = np.eye(4, dtype=np.float64)
    xyz = np.zeros((SE3_N, 3), dtype=np.float64)
    rot = np.zeros((SE3_N, 3, 3), dtype=np.float64)
    for k in range(SE3_N):
        xyz[k] = pose[:3, 3]
        rot[k] = pose[:3, :3]
        pose = t_mat @ pose
    return {"t": t_mat, "xyz": xyz, "r": rot}


def build_artifact(suite: dict) -> dict:
    t = tutorial(suite)
    h = hilbert_case(suite)
    g = gemm_case(suite)
    se3 = se3_chain(suite)
    idx = list(range(HILBERT_N))
    return {
        "slug": "matrix",
        "source": "python",
        "values": {
            "SUM": t["sum"].tolist(),
            "DIFF": t["diff"].tolist(),
            "PROD": t["prod"].tolist(),
            "TRANSPOSE": t["transpose"].tolist(),
            "X": t["x"].tolist(),
            "A_INV": t["a_inv"].tolist(),
            "HILBERT_X": h["x"].tolist(),
            "HILBERT_A_INV": h["h_inv"].tolist(),
            "GEMM00": g["gemm00"],
            "SE3_T": se3["t"].tolist(),
            "SE3_XYZ": se3["xyz"].tolist(),
            "SE3_R": se3["r"].tolist(),
        },
        "series": {
            "hilbert_x": {"x": idx, "y": h["x"].tolist()},
        },
        "metrics": {
            "residual_ratio": residual_ratio(t["a"], t["x"], t["b"]),
            "residual_ratio_hilbert": h["residual_ratio"],
            "kappa_hilbert": h["kappa"],
            "gemm_frob": g["gemm_frob"],
        },
        "timings": {
            "gemm": timing_entry(g["iters"], g["ns"]),
        },
    }


if __name__ == "__main__":
    run_cli("matrix", build_artifact)
