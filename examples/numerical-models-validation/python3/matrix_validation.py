#!/usr/bin/env python3
"""
python3/matrix_validation.py

Executes NumPy/SciPy equivalents of the matrix numerical models.
Outputs JSON results to stdout for cross-language validation with Rust.
"""

from __future__ import annotations

import json
import time

import numpy as np
from scipy.linalg import hilbert, inv, lu_factor, lu_solve


def compute_backward_stability(a: np.ndarray) -> tuple[np.ndarray, float, float]:
    n = a.shape[0]
    x_true = np.ones((n, 1))
    b = a @ x_true

    start_time = time.perf_counter_ns()
    lu, piv = lu_factor(a)
    x_hat = lu_solve((lu, piv), b)
    elapsed_ns = float(time.perf_counter_ns() - start_time)

    a_norm = np.linalg.norm(a, np.inf)
    x_hat_norm = np.linalg.norm(x_hat, np.inf)
    residual = (a @ x_hat) - b
    residual_norm = np.linalg.norm(residual, np.inf)

    eps = np.finfo(float).eps
    residual_ratio = residual_norm / (a_norm * x_hat_norm * eps)

    return x_hat, float(residual_ratio), elapsed_ns


def compute_matmul_chain(m_init: np.ndarray, k: np.ndarray, iterations: int) -> tuple[np.ndarray, float]:
    m_current = m_init.copy()

    start_time = time.perf_counter_ns()
    for _ in range(iterations):
        m_current = m_current @ k
    elapsed_ns = float(time.perf_counter_ns() - start_time)

    return m_current, elapsed_ns


def compute_matrix_inverse(a: np.ndarray) -> tuple[np.ndarray, float, float]:
    start_time = time.perf_counter_ns()
    a_inv = inv(a)
    elapsed_ns = float(time.perf_counter_ns() - start_time)

    ident = a @ a_inv
    identity_error = np.max(np.abs(ident - np.eye(a.shape[0])))

    return a_inv, float(identity_error), elapsed_ns


def run_python_oracle() -> dict:
    h = hilbert(8)
    h_x_hat, h_ratio, h_time_ns = compute_backward_stability(h)

    m_init = np.eye(64)
    k = np.zeros((64, 64))
    for i in range(64):
        for j in range(64):
            k[i, j] = 0.01 * (i + 1) * (j + 3) / 64.0

    gemm_final, gemm_time_ns = compute_matmul_chain(m_init, k, 200)

    a_inv_test = np.full((3, 3), 0.5)
    np.fill_diagonal(a_inv_test, 2.0)
    a_inv, identity_error, inv_time_ns = compute_matrix_inverse(a_inv_test)

    return {
        "hilbert": {
            "x_hat": h_x_hat.tolist(),
            "residual_ratio": h_ratio,
            "time_ns": h_time_ns,
        },
        "matmul_chain": {
            "final_matrix": gemm_final.tolist(),
            "final_norm": float(np.linalg.norm(gemm_final, np.inf)),
            "time_ns": gemm_time_ns,
        },
        "inversion": {
            "a_inv": a_inv.tolist(),
            "identity_error": identity_error,
            "time_ns": inv_time_ns,
        },
    }


if __name__ == "__main__":
    results = run_python_oracle()
    print(json.dumps(results))