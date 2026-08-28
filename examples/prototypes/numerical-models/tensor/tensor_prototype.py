#!/usr/bin/env python3
"""Tensor numerical prototype oracle (SciPy interpolate)."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def _q7_raw(val: float) -> int:
    """Match ``Fixed<i8, 7>::from_num`` (round half away from zero, saturate)."""
    scaled = float(val) * 128.0
    if scaled >= 127.0:
        return 127
    if scaled <= -128.0:
        return -128
    if scaled >= 0.0:
        return int(scaled + 0.5)
    return int(scaled - 0.5)


def scenario() -> dict[str, object]:
    # values[dim0, dim1] matches crate interpolate(&[coord0, coord1])
    # for ArrayTensor::from_raw column-major [[0,2,4],[1,3,5],[2,4,6]].
    values = np.array(
        [
            [0.0, 1.0, 2.0],
            [2.0, 3.0, 4.0],
            [4.0, 5.0, 6.0],
        ],
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
        "float_inputs": float_inputs,
        "q_raw": q_raw,
        "dequant": dequant,
        "relu_raw": relu_raw,
        "relu_dequant": relu_dequant,
        "values": values,
    }


def equiv() -> dict[str, object]:
    s = scenario()
    return {
        "SAMPLES": s["samples"],
        "Q_RAW": s["q_raw"].astype(np.int32),
        "DEQUANT": s["dequant"],
        "RELU_RAW": s["relu_raw"].astype(np.int32),
        "RELU_DEQUANT": s["relu_dequant"],
    }


def print_transcript() -> None:
    s = scenario()
    print("=== Tensor Numerical Prototype Oracle ===")
    print("\n--- 2D Grid Table (3x3) ---")
    # Print in the example's visual row-major layout (get([j, i])).
    visual = np.array(
        [[0.0, 2.0, 4.0], [1.0, 3.0, 5.0], [2.0, 4.0, 6.0]], dtype=np.float32
    )
    for row in visual:
        print("  " + str([float(v) for v in row]))
    print("\n--- Multilinear Continuous Interpolation ---")
    print(f"{'(x, y)':<16}{'Interpolated Value':<20}")
    for pt, val in zip(s["points"], s["samples"], strict=True):
        print(f"({pt[0]:.2f}, {pt[1]:.2f})      {val:<20.6f}")
    print("\n--- Quantized Fixed-Point Q7 Simulation ---")
    print(
        f"{'Float Input':<14}{'Q7 Raw':<10}{'Dequantized':<16}{'ReLU Output':<16}"
    )
    for f_in, raw, deq, relu_d, relu_r in zip(
        s["float_inputs"],
        s["q_raw"],
        s["dequant"],
        s["relu_dequant"],
        s["relu_raw"],
        strict=True,
    ):
        print(
            f"{f_in:<14.4f}{int(raw):<10}{deq:<16.6f}"
            f"{relu_d:<16.6f} (raw: {int(relu_r)})"
        )


if __name__ == "__main__":
    print_transcript()
