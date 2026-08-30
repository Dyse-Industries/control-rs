#!/usr/bin/env python3
"""
python3/polynomial_validation.py

Executes NumPy equivalents for polynomial numerical models.
Outputs JSON results to stdout for cross-language validation with Rust.
"""

from __future__ import annotations

import json
import time

import numpy as np
from numpy.polynomial.polynomial import (
    polycompanion,
    polyder,
    polydiv,
    polyfromroots,
    polyint,
    polymul,
    polyval,
)

SWEEP_N = 128


def run_polynomial_oracle() -> dict:
    coeffs = np.array([2.0, -3.0, 4.0, 1.0, 0.0], dtype=np.float64)
    x_test = 2.5
    z = complex(1.0, 2.0)

    t0 = time.perf_counter_ns()
    p_real = float(polyval(x_test, coeffs))
    eval_time_ns = float(time.perf_counter_ns() - t0)

    t0 = time.perf_counter_ns()
    val_c = polyval(z, coeffs)
    complex_eval_time_ns = float(time.perf_counter_ns() - t0)
    p_c_re = float(val_c.real)
    p_c_im = float(val_c.imag)

    t0 = time.perf_counter_ns()
    deriv = polyder(coeffs)
    p_deriv = float(polyval(x_test, deriv))
    deriv_time_ns = float(time.perf_counter_ns() - t0)

    deriv5 = np.zeros(5, dtype=np.float64)
    deriv5[: deriv.size] = deriv

    t0 = time.perf_counter_ns()
    integ = polyint(coeffs, k=[5.0])
    p_integ = float(polyval(x_test, integ))
    integ_time_ns = float(time.perf_counter_ns() - t0)

    integ5 = np.zeros(5, dtype=np.float64)
    integ5[: min(5, integ.size)] = integ[:5]

    p1 = np.array([1.0, 2.0], dtype=np.float64)
    p2 = np.array([3.0, 4.0], dtype=np.float64)

    t0 = time.perf_counter_ns()
    prod = polymul(p1, p2)
    mul_time_ns = float(time.perf_counter_ns() - t0)

    t0 = time.perf_counter_ns()
    quot, rem = polydiv(prod, p1)
    div_time_ns = float(time.perf_counter_ns() - t0)

    p_monic = np.array(
        [
            1.0,
            12.0,
            66.0,
            220.0,
            495.0,
            792.0,
            924.0,
            792.0,
            495.0,
            220.0,
            66.0,
            12.0,
            1.0,
        ],
        dtype=np.float64,
    )

    t0 = time.perf_counter_ns()
    companion = np.array(polycompanion(p_monic), dtype=np.float64)
    companion_time_ns = float(time.perf_counter_ns() - t0)

    # 2. Clustered-root setup
    roots = np.arange(1, 17, dtype=np.float64)
    cluster_coeffs = polyfromroots(roots)
    sweep_x = np.linspace(0.9, 1.1, SWEEP_N, dtype=np.float64)
    cluster_y = np.asarray(polyval(sweep_x, cluster_coeffs), dtype=np.float64)

    timed_x = 1.005
    iters = 10_000
    start = time.perf_counter_ns()
    for _ in range(iters):
        polyval(timed_x, cluster_coeffs)
    elapsed_ns = float(time.perf_counter_ns() - start)

    return {
        "tutorial": {
            "p_real": p_real,
            "p_c_re": p_c_re,
            "p_c_im": p_c_im,
            "deriv": deriv5.tolist(),
            "p_deriv": p_deriv,
            "integ": integ5.tolist(),
            "p_integ": p_integ,
            "prod": prod.tolist(),
            "quot": np.asarray(quot, dtype=np.float64).tolist(),
            "rem": float(np.asarray(rem).reshape(-1)[0]),
            "companion": companion.tolist(),
            "eval_time_ns": eval_time_ns,
            "complex_eval_time_ns": complex_eval_time_ns,
            "deriv_time_ns": deriv_time_ns,
            "integ_time_ns": integ_time_ns,
            "mul_time_ns": mul_time_ns,
            "div_time_ns": div_time_ns,
            "companion_time_ns": companion_time_ns,
        },
        "clustered": {
            "coeffs": cluster_coeffs.tolist(),
            "x": sweep_x.tolist(),
            "y": cluster_y.tolist(),
            "time_ns": elapsed_ns,
        },
    }


if __name__ == "__main__":
    results = run_polynomial_oracle()
    print(json.dumps(results))
