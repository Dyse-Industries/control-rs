#!/usr/bin/env python3
"""
python3/tensor_validation.py

Executes NumPy/SciPy equivalents for tensor and fixed-point numerical models.
Outputs JSON results to stdout for cross-language validation with Rust.
"""

from __future__ import annotations

import json
import time

import numpy as np
from scipy.interpolate import RegularGridInterpolator

CUT_N = 64
CURVED_N = 16


def _q7_raw(val: float) -> int:
    scaled = float(val) * 128.0
    if scaled >= 127.0:
        return 127
    if scaled <= -128.0:
        return -128
    if scaled >= 0.0:
        return int(scaled + 0.5)
    return int(scaled - 0.5)


def run_tensor_oracle() -> dict:
    table_affine = np.array(
        [
            [0.0, 1.0, 2.0],
            [2.0, 3.0, 4.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=np.float32,
    )
    axes_affine = (
        np.array([0.0, 1.0, 2.0], dtype=np.float32),
        np.array([0.0, 1.0, 2.0], dtype=np.float32),
    )
    interp_affine = RegularGridInterpolator(
        axes_affine, table_affine, method="linear", bounds_error=False, fill_value=None
    )
    points_affine = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [0.5, 0.5],
            [1.5, 0.5],
            [0.2, 1.8],
        ],
        dtype=np.float32,
    )

    t0 = time.perf_counter_ns()
    samples_affine = np.asarray(interp_affine(points_affine), dtype=np.float32)
    affine_interp_time_ns = float(time.perf_counter_ns() - t0)

    # 2. Curved Grid (16x16 Saddle)
    n = CURVED_N
    center = 7.5
    scale = 7.5
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    u = (ii - center) / scale
    v = (jj - center) / scale
    table_curved = np.asarray(u * u - v * v, dtype=np.float32)

    axes_curved = (
        np.arange(n, dtype=np.float32),
        np.arange(n, dtype=np.float32),
    )
    interp_curved = RegularGridInterpolator(
        axes_curved, table_curved, method="linear", bounds_error=False, fill_value=None
    )

    cut_x = np.linspace(0.0, 15.0, CUT_N, dtype=np.float32)
    pts_curved = np.stack([cut_x, np.full(CUT_N, center, dtype=np.float32)], axis=1)
    samples_curved = np.asarray(interp_curved(pts_curved), dtype=np.float32)

    interior = np.array([[7.3, 8.1]], dtype=np.float32)
    iters = 10_000
    start = time.perf_counter_ns()
    for _ in range(iters):
        interp_curved(interior)
    interp_time_ns = float(time.perf_counter_ns() - start)

    weiser_bound = 0.125 * (2.0 / scale**2 + 2.0 / scale**2)

    # 3. Quantized Fixed-Point Q7
    float_inputs = np.array(
        [
            0.7853981633974483,
            0.3333333333333333,
            0.7182818284590451,
            -0.75,
            0.0,
            0.5,
        ],
        dtype=np.float64,
    )

    t0 = time.perf_counter_ns()
    q_raw = np.array([_q7_raw(float(val)) for val in float_inputs], dtype=np.int32)
    dequant = q_raw.astype(np.float64) / 128.0
    relu_raw = np.maximum(q_raw, 0)
    relu_dequant = relu_raw.astype(np.float64) / 128.0
    quant_err = float(np.max(np.abs(float_inputs - dequant)))
    q7_time_ns = float(time.perf_counter_ns() - t0)

    return {
        "affine": {
            "samples": samples_affine.tolist(),
            "affine_interp_time_ns": affine_interp_time_ns,
        },
        "curved": {
            "table": table_curved.tolist(),
            "cut_x": cut_x.tolist(),
            "samples": samples_curved.tolist(),
            "weiser_bound": float(weiser_bound),
            "interp_time_ns": float(interp_time_ns),
        },
        "q7": {
            "q_raw": q_raw.tolist(),
            "dequant": dequant.astype(np.float32).tolist(),
            "relu_raw": relu_raw.tolist(),
            "relu_dequant": relu_dequant.astype(np.float32).tolist(),
            "quant_err": quant_err,
            "q7_time_ns": q7_time_ns,
        },
    }


if __name__ == "__main__":
    results = run_tensor_oracle()
    print(json.dumps(results))
