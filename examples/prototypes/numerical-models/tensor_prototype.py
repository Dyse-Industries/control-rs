#!/usr/bin/env python3
"""Tensor Numerical Prototype Oracle.

Calculates 2D multilinear grid lookup table interpolation and fixed-point Q7
quantized arithmetic with ReLU activation.
Implemented in pure Python (standard library).
"""

import math


def clamp(val, low, high):
    return max(low, min(val, high))


def multilinear_interpolate_2d(grid, x, y):
    """Interpolate in a unit-spaced 2D grid with bounds. grid: shape [R][C]."""
    r_max = len(grid) - 1
    c_max = len(grid[0]) - 1

    x_clamped = clamp(x, 0.0, float(c_max))
    y_clamped = clamp(y, 0.0, float(r_max))

    x0 = int(math.floor(x_clamped))
    y0 = int(math.floor(y_clamped))
    x1 = min(x0 + 1, c_max)
    y1 = min(y0 + 1, r_max)

    wx = x_clamped - float(x0)
    wy = y_clamped - float(y0)

    v00 = grid[y0][x0]
    v10 = grid[y0][x1]
    v01 = grid[y1][x0]
    v11 = grid[y1][x1]

    top = (1.0 - wx) * v00 + wx * v10
    bot = (1.0 - wx) * v01 + wx * v11
    return (1.0 - wy) * top + wy * bot


def main():
    print("=== Tensor Numerical Prototype Oracle ===")

    # 1. 2D Aerodynamic Calibration Grid (3x3)
    # y=0: [0.0, 2.0, 4.0]
    # y=1: [1.0, 3.0, 5.0]
    # y=2: [2.0, 4.0, 6.0]
    grid = [[0.0, 2.0, 4.0], [1.0, 3.0, 5.0], [2.0, 4.0, 6.0]]

    print("\n--- 2D Grid Table (3x3) ---")
    for row in grid:
        print("  " + str(row))

    test_points = [
        (0.0, 0.0),
        (1.0, 1.0),
        (2.0, 2.0),
        (0.5, 0.5),
        (1.5, 0.5),
        (0.2, 1.8),
    ]

    print("\n--- Multilinear Continuous Interpolation ---")
    print(f"{'(x, y)':<16}{'Interpolated Value':<20}")
    for x, y in test_points:
        val = multilinear_interpolate_2d(grid, x, y)
        print(f"({x:.2f}, {y:.2f}){'':<6}{val:<20.6f}")

    # 2. Fixed-Point Q7 Quantized Arithmetic & Activation
    # Q7 format: scaling factor = 2^7 = 128. Range [-128, 127] representing [-1.0, 0.9921875]
    print("\n--- Quantized Fixed-Point Q7 Simulation ---")
    float_inputs = [-0.75, -0.25, 0.0, 0.25, 0.5, 0.75]
    print(
        f"{'Float Input':<14}{'Q7 Raw':<10}{'Dequantized':<16}{'ReLU Output':<16}"
    )

    for f_in in float_inputs:
        q_raw = int(clamp(round(f_in * 128.0), -128, 127))
        dequant = float(q_raw) / 128.0
        q_relu_raw = max(0, q_raw)
        relu_dequant = float(q_relu_raw) / 128.0
        print(
            f"{f_in:<14.4f}{q_raw:<10}{dequant:<16.6f}{relu_dequant:<16.6f} (raw: {q_relu_raw})"
        )


if __name__ == "__main__":
    main()
