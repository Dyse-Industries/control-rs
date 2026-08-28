#!/usr/bin/env python3
"""Tensor numerical oracle — writes results/tensor/python.json."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

CRATE_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = CRATE_ROOT / "results" / "tensor" / "python.json"


def _q7_raw(val: float) -> int:
    scaled = float(val) * 128.0
    if scaled >= 127.0:
        return 127
    if scaled <= -128.0:
        return -128
    if scaled >= 0.0:
        return int(scaled + 0.5)
    return int(scaled - 0.5)


def scenario() -> dict[str, object]:
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
    float_inputs = np.array(
        [-0.75, -0.25, 0.0, 0.25, 0.5, 0.75], dtype=np.float32
    )
    q_raw = np.array([_q7_raw(float(v)) for v in float_inputs], dtype=np.int32)
    dequant = q_raw.astype(np.float32) / 128.0
    relu_raw = np.maximum(q_raw, 0)
    relu_dequant = relu_raw.astype(np.float32) / 128.0
    return {
        "points": points,
        "samples": samples,
        "q_raw": q_raw,
        "dequant": dequant,
        "relu_raw": relu_raw,
        "relu_dequant": relu_dequant,
    }


def build_artifact() -> dict:
    s = scenario()
    pts = s["points"]
    return {
        "slug": "tensor",
        "source": "python",
        "values": {
            "SAMPLES": s["samples"].tolist(),
            "Q_RAW": s["q_raw"].tolist(),
            "DEQUANT": s["dequant"].tolist(),
            "RELU_RAW": s["relu_raw"].tolist(),
            "RELU_DEQUANT": s["relu_dequant"].tolist(),
        },
        "series": {
            "interp": {
                "x": [float(pts[i, 0]) for i in range(len(pts))],
                "y": s["samples"].tolist(),
            }
        },
    }


if __name__ == "__main__":
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(build_artifact(), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT_PATH}")
