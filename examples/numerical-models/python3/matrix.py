#!/usr/bin/env python3
"""Matrix numerical oracle — writes results/matrix/python.json."""

from __future__ import annotations

import numpy as np
from scipy import linalg

from vv import CRATE_ROOT, residual_ratio, save_json, time_kernel, timing_entry

OUT_PATH = CRATE_ROOT / "results" / "matrix" / "python.json"
GEMM_N = 64
GEMM_ITERS = 200
HILBERT_N = 8
SE3_N = 40
SE3_THETA = 0.15
SE3_TVEC = (0.04, 0.01, 0.03)


def tutorial() -> dict:
    m1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    m2 = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
    a = np.array(
        [[3.0, 2.0, -1.0], [2.0, -2.0, 4.0], [-1.0, 0.5, -1.0]],
        dtype=np.float64,
    )
    b = np.array([1.0, -2.0, 0.0], dtype=np.float64)
    # The general, symmetric, Hermitian and positive definite solutions are obtained via calling
    # ?GESV, ?SYSV, ?HESV, and ?POSV routines of LAPACK respectively.
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


def hilbert_case() -> dict:
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


def gemm_case() -> dict:
    i = np.arange(GEMM_N, dtype=np.float64)
    ii, jj = np.meshgrid(i, i, indexing="ij")
    ga = 0.01 * (ii + 1.0) * (jj + 3.0) / 64.0
    gb = 0.02 * (ii + 2.0) * (jj + 1.0) / 64.0
    with np.errstate(all="ignore"):
        gc = ga @ gb
        ns = time_kernel(GEMM_ITERS, lambda: ga @ gb)
    return {
        "gemm00": float(gc[0, 0]),
        "gemm_frob": float(np.linalg.norm(gc, ord="fro")),
        "ns": ns,
    }


def se3_chain() -> dict:
    c = float(np.cos(SE3_THETA))
    s = float(np.sin(SE3_THETA))
    dx, dy, dz = SE3_TVEC
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


def build_artifact() -> dict:
    t = tutorial()
    h = hilbert_case()
    g = gemm_case()
    se3 = se3_chain()
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
            "gemm": timing_entry(GEMM_ITERS, g["ns"]),
        },
    }


if __name__ == "__main__":
    save_json(OUT_PATH, build_artifact())
