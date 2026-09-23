#!/usr/bin/env python3
"""Polynomial reference oracle generating target/verification/polynomial.scipy.h5 via NumPy."""

from pathlib import Path
import numpy as np

from h5_writer import get_results_dir, write_h5


def generate_datasets():
    # 1. Tutorial polynomial
    # P(x) = (x - 2)(x - 3)(x - 5) = x^3 - 10x^2 + 31x - 30
    # numpy.poly1d takes descending coefficients: [1.0, -10.0, 31.0, -30.0]
    p = np.poly1d([1.0, -10.0, 31.0, -30.0])
    p_real = float(p(2.5))
    c_val = p(1.0 + 2.0j)
    p_c_re = float(c_val.real)
    p_c_im = float(c_val.imag)

    # 2. Root convergence: Newton-Raphson on (x-2)(x+3)(x-5) = x^3 - 4x^2 - 11x + 30
    p_conv = np.poly1d([1.0, -4.0, -11.0, 30.0])
    dp_conv = p_conv.deriv()
    target_root = 2.0
    distances = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    iterations = []

    for dist in distances:
        x = target_root + dist
        iters = 0
        while iters < 100:
            iters += 1
            fx = p_conv(x)
            fpx = dp_conv(x)
            if abs(fpx) < 1e-12:
                break
            x_next = x - fx / fpx
            if abs(x_next - x) < 1e-8:
                break
            x = x_next
        iterations.append(float(iters))

    # 3. Wilkinson polynomial
    roots_20 = np.arange(1, 21, dtype=np.float64)
    poly_64 = np.poly1d(np.poly(roots_20))
    poly_32 = np.poly1d(np.poly(roots_20.astype(np.float32)))

    res_64 = [float(abs(poly_64(k))) for k in roots_20]
    res_32 = [float(abs(poly_32(k))) for k in roots_20]

    datasets = {
        "tutorial/p_real": [p_real],
        "tutorial/p_c_re": [p_c_re],
        "tutorial/p_c_im": [p_c_im],
        "root_convergence/iterations": iterations,
        "wilkinson_residual/residual_f64": res_64,
        "wilkinson_residual/residual_f32": res_32,
    }

    tolerances = {
        "tutorial/p_real": ("abs", 1e-6, {"flint": 1e-9}),
        "tutorial/p_c_re": ("abs", 1e-6, {"flint": 1e-9}),
        "tutorial/p_c_im": ("abs", 1e-6, {"flint": 1e-9}),
        "root_convergence/iterations": ("abs", 2.0),
        "wilkinson_residual/residual_f64": ("rel", 0.05),
        "wilkinson_residual/residual_f32": ("rel", 10.0),
    }

    # python-flint provides the tutorial evaluations only.
    missing_ok = {
        "root_convergence/iterations": ["flint"],
        "wilkinson_residual/residual_f64": ["flint"],
        "wilkinson_residual/residual_f32": ["flint"],
    }

    return datasets, tolerances, missing_ok


def main():
    datasets, tolerances, missing_ok = generate_datasets()
    out_file = get_results_dir() / "polynomial.scipy.h5"
    write_h5(out_file, datasets, tolerances, missing_ok)


if __name__ == "__main__":
    main()
