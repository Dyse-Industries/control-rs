#!/usr/bin/env python3
"""
python3/matrix_validation.py

Executes NumPy/SciPy equivalents of the matrix numerical models (EKF focus).
Outputs JSON results to stdout for cross-language validation with Rust.
"""

from __future__ import annotations

import json
import time

import numpy as np
from scipy.linalg import cholesky, hilbert, inv, lu_factor, lu_solve, qr, solve_triangular, svd


def generate_matrix_correctness_data() -> dict:
    n = 8
    # Initial 8x8 state covariance matrix P_0
    i_idx, j_idx = np.ogrid[:n, :n]
    diff = i_idx - j_idx
    p_0 = np.exp(-0.25 * (diff ** 2)) + 0.1 * np.eye(n)

    # Measurement matrix H, Kalman gain K, Noise R
    h = np.eye(n)
    k = 0.01 * np.ones((n, n))
    np.fill_diagonal(k, 0.4)

    r = 0.05 * np.eye(n)

    kh = k @ h
    eye = np.eye(n)
    i_minus_kh = eye - kh

    krk_t = k @ r @ k.T
    p_current = p_0.copy()

    for _ in range(100):
        p_update1 = i_minus_kh @ p_current @ i_minus_kh.T
        p_current = p_update1 + krk_t

    return {
        "covariance_heatmap": {
            "matrix": p_current.tolist()
        }
    }


def benchmark_matrix_scaling() -> dict:
    dims = [2, 4, 8, 16, 32, 64]
    iters = 1000
    means = []
    stddevs = []

    for n in dims:
        a = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    a[i, j] = 2.0 * (i + 1)
                else:
                    a[i, j] = 0.5 / (i + j + 1)

        times = []
        for _ in range(iters):
            t0 = time.perf_counter_ns()
            _inv = inv(a)
            t1 = time.perf_counter_ns()
            times.append(float(t1 - t0))

        arr = np.array(times)
        means.append(float(np.mean(arr)))
        stddevs.append(float(np.std(arr)))

    return {
        "scaling": {
            "N": dims,
            "inversion_time_ns": means,
            "inversion_stddev_ns": stddevs
        }
    }


def benchmark_ekf_update_jitter() -> dict:
    n = 32
    iters = 1000
    h_mat = hilbert(n)
    b = np.ones((n, 1))

    lu, piv = lu_factor(h_mat)
    times = []

    for _ in range(iters):
        t0 = time.perf_counter_ns()
        _x = lu_solve((lu, piv), b)
        t1 = time.perf_counter_ns()
        times.append(float(t1 - t0))

    return {
        "jitter": {
            "hilbert_solve_times_ns": times
        }
    }


def benchmark_decompositions() -> dict:
    n = 16
    iters = 1000

    # Symmetric positive-definite matrix for Cholesky
    spd = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                spd[i, j] = 10.0 + i
            else:
                spd[i, j] = 1.0 / (i + j + 2)

    b = np.ones((n, 1))

    # 1. Cholesky
    t_chol_start = time.perf_counter_ns()
    for _ in range(iters):
        c_factor = cholesky(spd, lower=True)
        y = solve_triangular(c_factor, b, lower=True)
        _x = solve_triangular(c_factor.T, y, lower=False)
    t_chol = float(time.perf_counter_ns() - t_chol_start) / iters

    # 2. LU Solve
    a_lu = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                a_lu[i, j] = 5.0 + i
            else:
                a_lu[i, j] = 0.2 * (i + j + 1)

    t_lu_start = time.perf_counter_ns()
    for _ in range(iters):
        lu, piv = lu_factor(a_lu)
        _x = lu_solve((lu, piv), b)
    t_lu = float(time.perf_counter_ns() - t_lu_start) / iters

    # 3. QR Decomposition
    a_qr = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                a_qr[i, j] = 3.0 + i
            else:
                a_qr[i, j] = 0.1 * (i + j)

    t_qr_start = time.perf_counter_ns()
    for _ in range(iters):
        _q, _r = qr(a_qr)
    t_qr = float(time.perf_counter_ns() - t_qr_start) / iters

    # 4. SVD Decomposition
    a_svd = a_lu.copy()
    t_svd_start = time.perf_counter_ns()
    for _ in range(iters):
        _u, _s, _vh = svd(a_svd)
    t_svd = float(time.perf_counter_ns() - t_svd_start) / iters

    return {
        "decomp_times_ns": {
            "cholesky": t_chol,
            "lu_solve": t_lu,
            "qr_decomp": t_qr,
            "svd": t_svd
        }
    }


def run_python_oracle() -> dict:
    q1 = generate_matrix_correctness_data()
    q2 = benchmark_matrix_scaling()
    q3 = benchmark_ekf_update_jitter()
    q4 = benchmark_decompositions()

    return {
        "covariance_heatmap": q1["covariance_heatmap"],
        "scaling": q2["scaling"],
        "jitter": q3["jitter"],
        "decomp_times_ns": q4["decomp_times_ns"]
    }


if __name__ == "__main__":
    results = run_python_oracle()
    print(json.dumps(results))