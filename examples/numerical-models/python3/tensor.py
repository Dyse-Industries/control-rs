#!/usr/bin/env python3
"""Tensor numerical oracle — writes results/tensor/python.json."""

from __future__ import annotations

import math

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from vv import CRATE_ROOT, save_json, time_kernel, timing_entry

OUT_PATH = CRATE_ROOT / "results" / "tensor" / "python.json"
CUT_N = 64
INTERP_ITERS = 10_000


def _q7_raw(val: float) -> int:
    scaled = float(val) * 128.0
    if scaled >= 127.0:
        return 127
    if scaled <= -128.0:
        return -128
    if scaled >= 0.0:
        return int(scaled + 0.5)
    return int(scaled - 0.5)


def affine() -> dict:
    values = np.array(
        [[0.0, 1.0, 2.0], [2.0, 3.0, 4.0], [4.0, 5.0, 6.0]],
        dtype=np.float32,
    )
    axes = (np.arange(3, dtype=np.float32), np.arange(3, dtype=np.float32))
    interp = RegularGridInterpolator(
        axes, values, method="linear", bounds_error=False, fill_value=None
    )
    points = np.array(
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
    samples = np.asarray(interp(points), dtype=np.float32)
    return {"points": points, "samples": samples}


def curved() -> dict:
    ii, jj = np.meshgrid(np.arange(16), np.arange(16), indexing="ij")
    table = np.sin(np.pi * ii / 15.0) * np.cos(np.pi * jj / 15.0)
    table = np.asarray(table, dtype=np.float32)
    axes = (
        np.arange(16, dtype=np.float32),
        np.arange(16, dtype=np.float32),
    )
    interp = RegularGridInterpolator(
        axes, table, method="linear", bounds_error=False, fill_value=None
    )
    cut_x = np.linspace(0.0, 15.0, CUT_N, dtype=np.float32)
    pts = np.stack([cut_x, cut_x], axis=1)
    samples = np.asarray(interp(pts), dtype=np.float32)
    interior = np.array([[7.3, 8.1]], dtype=np.float32)
    ns = time_kernel(INTERP_ITERS, lambda: interp(interior))
    weiser = 0.125 * 2.0 * (math.pi / 15.0) ** 2
    return {
        "table": table,
        "cut_x": cut_x,
        "samples": samples,
        "ns": ns,
        "weiser": weiser,
    }


def q7() -> dict:
    float_inputs = np.array(
        [
            math.pi / 4.0,
            1.0 / 3.0,
            math.e - 2.0,
            -0.75,
            0.0,
            0.5,
        ],
        dtype=np.float64,
    )
    q_raw = np.array([_q7_raw(float(v)) for v in float_inputs], dtype=np.int32)
    dequant = q_raw.astype(np.float64) / 128.0
    relu_raw = np.maximum(q_raw, 0)
    relu_dequant = relu_raw.astype(np.float64) / 128.0
    quant_err = float(np.max(np.abs(float_inputs - dequant)))
    return {
        "q_raw": q_raw,
        "dequant": dequant.astype(np.float32),
        "relu_raw": relu_raw,
        "relu_dequant": relu_dequant.astype(np.float32),
        "quant_err": quant_err,
    }


def build_artifact() -> dict:
    a = affine()
    c = curved()
    q = q7()
    pts = a["points"]
    return {
        "slug": "tensor",
        "source": "python",
        "values": {
            "SAMPLES": a["samples"].tolist(),
            "CURVED_SAMPLES": c["samples"].tolist(),
            "CURVED_TABLE": c["table"].tolist(),
            "CUT_X": c["cut_x"].tolist(),
            "Q_RAW": q["q_raw"].tolist(),
            "DEQUANT": q["dequant"].tolist(),
            "RELU_RAW": q["relu_raw"].tolist(),
            "RELU_DEQUANT": q["relu_dequant"].tolist(),
        },
        "series": {
            "interp": {
                "x": [float(pts[i, 0]) for i in range(len(pts))],
                "y": a["samples"].tolist(),
            },
            "curved": {
                "x": c["cut_x"].tolist(),
                "y": c["samples"].tolist(),
            },
        },
        "metrics": {
            "quant_roundtrip_max": q["quant_err"],
            "weiser_bound": c["weiser"],
        },
        "timings": {
            "interp": timing_entry(INTERP_ITERS, c["ns"]),
        },
    }


if __name__ == "__main__":
    save_json(OUT_PATH, build_artifact())
