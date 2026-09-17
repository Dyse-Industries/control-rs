#!/usr/bin/env python3
"""
python3/polynomial_oracle.py

NumPy (and optional python-flint) oracle for the polynomial validation suite.
Mirrors `control-rs-validation/src/polynomial.rs` on the same
cancellation-dominated inputs and writes `results/polynomial.numpy.h5`, plus
`results/polynomial.flint.h5` when python-flint is installed.

The flint peer evaluates Wilkinson's polynomial in 256-bit ball arithmetic, so
its residuals are the exact answer the two double-precision implementations are
failing to reach.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from numpy.polynomial.polynomial import polyder, polydiv, polyfromroots, polymul, polyval

from h5_write import attr_specs_from_toml, present_paths, write_variant_file

CLUSTER_ORDER = 16
CLUSTER_ROOT = 1.01


def wilkinson() -> dict:
    """W(x) = prod (x - k), k = 1..20, evaluated at each of its own roots."""
    coeffs_f64 = polyfromroots(np.arange(1, 21, dtype=np.float64))
    # Expand in float32 throughout rather than rounding an f64 expansion:
    # control-rs accumulates the product in the storage type.
    coeffs_f32 = np.array([1.0], dtype=np.float32)
    for k in range(1, 21):
        coeffs_f32 = np.convolve(
            coeffs_f32, np.array([-k, 1], dtype=np.float32)
        ).astype(np.float32)

    root_indices = list(range(1, 21))
    residual_f64 = [float(abs(polyval(float(k), coeffs_f64))) for k in root_indices]
    residual_f32 = [
        float(abs(polyval(np.float32(k), coeffs_f32))) for k in root_indices
    ]

    return {
        "root_indices": root_indices,
        "residual_f64": residual_f64,
        "residual_f32": residual_f32,
        "coefficients": coeffs_f64.tolist(),
    }


def clustered() -> np.ndarray:
    """Ascending coefficients of (x - 1.01)^16."""
    return polyfromroots(np.full(CLUSTER_ORDER, CLUSTER_ROOT, dtype=np.float64))


def clustered_horner() -> dict:
    """Evaluate the clustered polynomial across [0, 2]."""
    xs = np.linspace(0.0, 2.0, 128, dtype=np.float64)
    return {"x": xs.tolist(), "values": polyval(xs, clustered()).tolist()}


def clustered_division() -> dict:
    """Divide the clustered polynomial by the factor it nearly has."""
    quot, rem = polydiv(clustered(), np.array([-CLUSTER_ROOT, 1.0]))
    return {"quot": quot.tolist(), "rem": rem.tolist()}


def scaled_product() -> dict:
    """Convolve coefficient vectors whose magnitudes span 1e16."""
    a = np.array([1e-8, 1.0, 1e8], dtype=np.float64)
    b = np.array([1e8, -1.0], dtype=np.float64)
    return {"coeffs": polymul(a, b).tolist()}


def companion() -> dict:
    """Companion matrix of (s^2 + 2s + 5)(s^2 + 4s + 5), control-rs layout."""
    coeffs = np.array([25.0, 30.0, 18.0, 6.0, 1.0], dtype=np.float64)
    deg = coeffs.size - 1
    matrix = np.zeros((deg, deg), dtype=np.float64)
    for i in range(1, deg):
        matrix[i, i - 1] = 1.0
    for i in range(deg):
        matrix[i, deg - 1] = -coeffs[i]
    return {"matrix": matrix.tolist()}


def newton_clustered() -> dict:
    """Newton iteration onto the 16-fold root, same stopping rules as Rust."""
    distances = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    max_iters = 100
    coeffs = clustered()
    derivative = polyder(coeffs)

    iterations = []
    final_x = []
    final_residual = []
    for distance in distances:
        x = CLUSTER_ROOT + distance
        iters = 0
        while iters < max_iters:
            fx = float(polyval(x, coeffs))
            fpx = float(polyval(x, derivative))
            if abs(fpx) < 1e-300:
                break
            nxt = x - fx / fpx
            iters += 1
            if abs(nxt - x) < 1e-12:
                x = nxt
                break
            x = nxt
        iterations.append(iters)
        final_x.append(x)
        final_residual.append(float(abs(polyval(x, coeffs))))

    return {
        "distances": distances,
        "iterations": iterations,
        "final_x": final_x,
        "final_residual": final_residual,
    }


def run_numpy_oracle() -> dict:
    """Assemble the NumPy payload."""
    return {
        "wilkinson": wilkinson(),
        "clustered_horner": clustered_horner(),
        "clustered_division": clustered_division(),
        "scaled_product": scaled_product(),
        "companion": companion(),
        "newton_clustered": newton_clustered(),
    }


def run_flint_oracle() -> dict | None:
    """Exact Wilkinson residuals in 256-bit ball arithmetic, or None."""
    try:
        import flint
    except ImportError:
        return None

    flint.ctx.prec = 256
    w = flint.arb_poly([1])
    for k in range(1, 21):
        w = w * flint.arb_poly([-k, 1])

    residual = [abs(float(w(flint.arb(k)).mid())) for k in range(1, 21)]
    return {"wilkinson": {"residual_f64": residual}}


GATED = [
    "wilkinson/residual_f64",
    "wilkinson/residual_f32",
    "clustered_horner/values",
    "clustered_division/quot",
    "clustered_division/rem",
    "scaled_product/coeffs",
    "companion/matrix",
]

SIGNAL_KEYS = {
    "wilkinson/residual_f64": "nmv.polynomial.wilkinson.residual_f64",
    "wilkinson/residual_f32": "nmv.polynomial.wilkinson.residual_f32",
    "clustered_horner/values": "nmv.polynomial.clustered_horner.values",
    "clustered_division/quot": "nmv.polynomial.clustered_division.quot",
    "clustered_division/rem": "nmv.polynomial.clustered_division.rem",
    "scaled_product/coeffs": "nmv.polynomial.scaled_product.coeffs",
    "companion/matrix": "nmv.polynomial.companion.matrix",
}

FLINT_KEYS = {
    "wilkinson/residual_f64": "nmv.polynomial.wilkinson.residual_f64_flint",
}


if __name__ == "__main__":
    numpy_results = run_numpy_oracle()
    table = Path(__file__).resolve().parent.parent / "tolerances/numerical_models.toml"
    flint_results = run_flint_oracle()
    specs = attr_specs_from_toml(
        table, SIGNAL_KEYS, {"flint": FLINT_KEYS} if flint_results else None
    )
    results = Path("results")
    write_variant_file(
        results / "polynomial.numpy.h5",
        numpy_results,
        gated_paths=GATED,
        meta=numpy_results,
        attr_specs=specs,
    )
    if flint_results:
        flint_paths = present_paths(flint_results, GATED)
        if flint_paths:
            write_variant_file(
                results / "polynomial.flint.h5",
                flint_results,
                gated_paths=flint_paths,
                meta=flint_results,
            )
