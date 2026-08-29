#!/usr/bin/env python3
"""Tensor numerical oracle — suite path in, result JSON on stdout."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from vv import case_inputs, require_int, run_cli, time_kernel, timing_entry

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


def affine(suite: dict) -> dict:
    inp = case_inputs(suite, "tensor.host.affine_interp")
    values = np.array(inp["table"], dtype=np.float32)
    axes = tuple(np.array(ax, dtype=np.float32) for ax in inp["axes"])
    interp = RegularGridInterpolator(
        axes, values, method="linear", bounds_error=False, fill_value=None
    )
    points = np.array(inp["points"], dtype=np.float32)
    samples = np.asarray(interp(points), dtype=np.float32)
    return {"points": points, "samples": samples}


def curved(suite: dict) -> dict:
    inp = case_inputs(suite, "tensor.host.curved_grid")
    require_int(inp, "n", CURVED_N)
    require_int(inp["cut"], "n", CUT_N)
    n = CURVED_N
    center = float(inp["center"])
    scale = float(inp["scale"])
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    u = (ii - center) / scale
    v = (jj - center) / scale
    table = np.asarray(u * u - v * v, dtype=np.float32)
    axes = (
        np.arange(n, dtype=np.float32),
        np.arange(n, dtype=np.float32),
    )
    interp = RegularGridInterpolator(
        axes, table, method="linear", bounds_error=False, fill_value=None
    )
    start = float(inp["cut"]["start"])
    stop = float(inp["cut"]["stop"])
    cut_x = np.linspace(start, stop, CUT_N, dtype=np.float32)
    pts = np.stack([cut_x, np.full(CUT_N, center, dtype=np.float32)], axis=1)
    samples = np.asarray(interp(pts), dtype=np.float32)
    interior = np.array([inp["timed_point"]], dtype=np.float32)
    iters = int(inp.get("iters", 10_000))
    ns = time_kernel(iters, lambda: interp(interior))
    weiser = 0.125 * (2.0 / scale**2 + 2.0 / scale**2)
    return {
        "table": table,
        "cut_x": cut_x,
        "samples": samples,
        "ns": ns,
        "iters": iters,
        "weiser": weiser,
    }


def q7(suite: dict) -> dict:
    inp = case_inputs(suite, "tensor.host.q7_relu")
    float_inputs = np.array(inp["float_inputs"], dtype=np.float64)
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


def build_artifact(suite: dict) -> dict:
    a = affine(suite)
    c = curved(suite)
    q = q7(suite)
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
            "interp": timing_entry(c["iters"], c["ns"]),
        },
    }


if __name__ == "__main__":
    run_cli("tensor", build_artifact)
