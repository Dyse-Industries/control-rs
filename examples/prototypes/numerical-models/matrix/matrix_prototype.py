#!/usr/bin/env python3
"""Matrix numerical prototype oracle (NumPy / SciPy)."""

from __future__ import annotations

import numpy as np
from scipy import linalg


def scenario() -> dict[str, np.ndarray]:
    m1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    m2 = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
    a = np.array(
        [[3.0, 2.0, -1.0], [2.0, -2.0, 4.0], [-1.0, 0.5, -1.0]],
        dtype=np.float64,
    )
    b = np.array([1.0, -2.0, 0.0], dtype=np.float64)
    x = linalg.solve(a, b)
    a_inv = linalg.inv(a)
    return {
        "m1": m1,
        "m2": m2,
        "sum": m1 + m2,
        "diff": m2 - m1,
        "prod": m1 @ m2,
        "transpose": m1.T.copy(),
        "a": a,
        "b": b.reshape(3, 1),
        "x": x.reshape(3, 1),
        "a_inv": a_inv,
        "ident": a @ a_inv,
    }


def goldens() -> dict[str, np.ndarray]:
    s = scenario()
    return {
        "SUM": s["sum"],
        "DIFF": s["diff"],
        "PROD": s["prod"],
        "TRANSPOSE": s["transpose"],
        "X": s["x"],
        "A_INV": s["a_inv"],
    }


def print_transcript() -> None:
    s = scenario()
    print("=== Matrix Numerical Prototype Oracle ===")
    print("\n--- Matrix Construction & Basic Arithmetic ---")
    for name in ("m1", "m2", "sum", "diff", "prod", "transpose"):
        label = {
            "m1": "M1",
            "m2": "M2",
            "sum": "M1 + M2",
            "diff": "M2 - M1",
            "prod": "M1 * M2",
            "transpose": "M1^T (Transpose)",
        }[name]
        print(f"{label}:")
        for row in s[name]:
            print("  [" + ", ".join(f"{val:12.6f}" for val in row) + "]")

    print("\n--- Linear System A * x = b ---")
    print("A:")
    for row in s["a"]:
        print("  [" + ", ".join(f"{val:12.6f}" for val in row) + "]")
    print("b:")
    for row in s["b"]:
        print("  [" + ", ".join(f"{val:12.6f}" for val in row) + "]")
    print("Solution x:")
    for row in s["x"]:
        print("  [" + ", ".join(f"{val:12.6f}" for val in row) + "]")
    residual = np.linalg.norm(s["a"] @ s["x"].ravel() - s["b"].ravel())
    print(f"Residual norm: {residual:.16e}")

    print("\n--- Matrix Inversion & Identity Check ---")
    print("A^-1:")
    for row in s["a_inv"]:
        print("  [" + ", ".join(f"{val:12.6f}" for val in row) + "]")
    print("A * A^-1 (Identity check):")
    for row in s["ident"]:
        print("  [" + ", ".join(f"{val:12.6f}" for val in row) + "]")


if __name__ == "__main__":
    print_transcript()
