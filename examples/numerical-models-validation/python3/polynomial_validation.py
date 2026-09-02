#!/usr/bin/env python3
"""
python3/polynomial_validation.py

Executes NumPy equivalents for polynomial numerical models across four benchmark domains:
1. Execution Time vs. Polynomial Degree (Computational Complexity: Horner vs Naive)
2. Convergence Rate of Root-Finding Algorithms (Algorithmic Efficiency)
3. Residual Error on Ill-Conditioned Polynomials (Wilkinson's Polynomial W(x))
4. Root Sensitivity and Quantization Bounds (Control System Pole Migration)
"""

from __future__ import annotations

import json
import time

import numpy as np
from numpy.polynomial.polynomial import polyder, polyfromroots, polyval


def benchmark_complexity() -> dict:
    degrees = list(range(1, 51))
    num_points = 1000
    eval_points = np.linspace(-1.0, 1.0, num_points, dtype=np.float64)

    horner_times_ns = []
    naive_times_ns = []

    for deg in degrees:
        coeffs = 1.0 / np.arange(1, deg + 2, dtype=np.float64)

        # Horner evaluation via polyval
        t0 = time.perf_counter_ns()
        horner_res = polyval(eval_points, coeffs)
        horner_elapsed = float(time.perf_counter_ns() - t0)

        # Naive direct evaluation via powers
        t1 = time.perf_counter_ns()
        naive_res = np.zeros(num_points, dtype=np.float64)
        for i in range(deg + 1):
            naive_res += coeffs[i] * (eval_points**i)
        naive_elapsed = float(time.perf_counter_ns() - t1)

        horner_times_ns.append(horner_elapsed)
        naive_times_ns.append(naive_elapsed)

    return {
        "degrees": degrees,
        "horner_time_ns": horner_times_ns,
        "naive_time_ns": naive_times_ns,
    }


def benchmark_root_convergence() -> dict:
    # Target polynomial P(x) = x^3 - 4x^2 - 11x + 30
    # Ascending order: [30.0, -11.0, -4.0, 1.0]
    p_coeffs = np.array([30.0, -11.0, -4.0, 1.0], dtype=np.float64)
    dp_coeffs = polyder(p_coeffs)

    target_root = 2.0
    distances = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    iterations_list = []

    for dist in distances:
        x = target_root + dist
        iters = 0
        max_iters = 100

        while iters < max_iters:
            fx = float(polyval(x, p_coeffs))
            if abs(fx) < 1e-6:
                break
            fpx = float(polyval(x, dp_coeffs))
            if abs(fpx) < 1e-12:
                break
            next_x = x - fx / fpx
            if abs(next_x - x) < 1e-6:
                iters += 1
                break
            x = next_x
            iters += 1

        iterations_list.append(iters)

    return {
        "target_root": target_root,
        "distances": distances,
        "iterations": iterations_list,
    }


def benchmark_wilkinson() -> dict:
    root_targets = np.arange(1, 21, dtype=np.float64)
    coeffs_f64 = polyfromroots(root_targets)
    coeffs_f32 = coeffs_f64.astype(np.float32)

    root_indices = list(range(1, 21))
    residual_f64 = []
    residual_f32 = []

    for k in root_indices:
        rk_64 = float(k)
        res_64 = float(abs(polyval(rk_64, coeffs_f64)))
        residual_f64.append(res_64)

        rk_32 = float(k)
        res_32 = float(abs(polyval(rk_32, coeffs_f32)))
        residual_f32.append(res_32)

    return {
        "root_indices": root_indices,
        "residual_f64": residual_f64,
        "residual_f32": residual_f32,
    }


def benchmark_root_sensitivity() -> dict:
    # Ground truth poles s = -1 ± 2j, -2 ± 1j
    ground_truth_roots_re = [-1.0, -1.0, -2.0, -2.0]
    ground_truth_roots_im = [2.0, -2.0, 1.0, -1.0]

    ground_truth_coeffs = np.array([25.0, 30.0, 18.0, 6.0, 1.0], dtype=np.float64)

    num_trials = 250
    np.random.seed(42)
    noise_scale = 0.015

    perturbed_re = []
    perturbed_im = []

    for _ in range(num_trials):
        p_coeffs = ground_truth_coeffs.copy()
        noise = np.random.uniform(-1.0, 1.0, size=4) * noise_scale
        p_coeffs[:4] *= 1.0 + noise

        # np.roots expects descending powers (c4*x^4 + c3*x^3 + ...)
        roots = np.roots(p_coeffs[::-1])
        for r in roots:
            perturbed_re.append(float(r.real))
            perturbed_im.append(float(r.imag))

    return {
        "ground_truth_re": ground_truth_roots_re,
        "ground_truth_im": ground_truth_roots_im,
        "perturbed_re": perturbed_re,
        "perturbed_im": perturbed_im,
    }


def run_polynomial_oracle() -> dict:
    coeffs = np.array([2.0, -3.0, 4.0, 1.0, 0.0], dtype=np.float64)
    x_test = 2.5
    z = complex(1.0, 2.0)

    p_real = float(polyval(x_test, coeffs))
    val_c = polyval(z, coeffs)
    p_c_re = float(val_c.real)
    p_c_im = float(val_c.imag)

    complexity = benchmark_complexity()
    root_convergence = benchmark_root_convergence()
    wilkinson = benchmark_wilkinson()
    root_sensitivity = benchmark_root_sensitivity()

    return {
        "complexity": complexity,
        "root_convergence": root_convergence,
        "wilkinson_residual": wilkinson,
        "root_sensitivity": root_sensitivity,
        "tutorial": {
            "p_real": p_real,
            "p_c_re": p_c_re,
            "p_c_im": p_c_im,
        },
    }


def run_flint_oracle() -> dict:
    import flint

    flint.ctx.prec = 256  # ~77 decimal digits of working precision

    # Build W(x) = product((x - k), k=1..20) using arb_poly ball arithmetic
    w = flint.arb_poly([1])
    for k in range(1, 21):
        w = w * flint.arb_poly([-k, 1])

    root_indices = list(range(1, 21))
    residual_f64_flint = []
    for k in root_indices:
        val = w(flint.arb(k))
        # Extract ball midpoint (should be exactly 0 for integer roots)
        residual_f64_flint.append(abs(float(val.mid())))

    # Tutorial polynomial: 2 + (-3)x + 4x^2 + x^3 (ascending coefficients)
    # Evaluate at x=2.5 using arb_poly
    tut_poly = flint.arb_poly([2, -3, 4, 1])
    tut_val = tut_poly(flint.arb("2.5"))
    tut_mid = float(tut_val.mid())

    return {
        "wilkinson_residual": {
            "root_indices": root_indices,
            "residual_f64_flint": residual_f64_flint,
        },
        "tutorial": {
            "p_real_flint": tut_mid,
        },
    }


if __name__ == "__main__":
    scipy_results = run_polynomial_oracle()
    flint_results = run_flint_oracle()
    print(json.dumps({"scipy": scipy_results, "flint": flint_results}))

