#!/usr/bin/env python3
"""Polynomial numerical prototype oracle (NumPy)."""

from __future__ import annotations

import numpy as np
from numpy.polynomial.polynomial import (
    polycompanion,
    polyder,
    polydiv,
    polyint,
    polymul,
    polyval,
)


def scenario() -> dict[str, object]:
    coeffs = np.array([2.0, -3.0, 4.0, 1.0, 0.0], dtype=np.float64)
    x_test = 2.5
    z = 1.0 + 2.0j
    deriv = polyder(coeffs)
    deriv5 = np.zeros(5, dtype=np.float64)
    deriv5[: deriv.size] = deriv
    integ = polyint(coeffs, k=[5.0])
    integ5 = integ[:5]
    p1 = np.array([1.0, 2.0], dtype=np.float64)
    p2 = np.array([3.0, 4.0], dtype=np.float64)
    prod = polymul(p1, p2)
    quot, rem = polydiv(prod, p1)
    p_monic = np.array([-6.0, -5.0, 1.0], dtype=np.float64)
    companion = np.array(polycompanion(p_monic), dtype=np.float64)
    val_c = polyval(z, coeffs)
    return {
        "coeffs": coeffs,
        "x_test": x_test,
        "p_real": float(polyval(x_test, coeffs)),
        "p_c_re": float(val_c.real),
        "p_c_im": float(val_c.imag),
        "deriv": deriv5,
        "p_deriv": float(polyval(x_test, deriv)),
        "integ": integ5,
        "p_integ": float(polyval(x_test, integ5)),
        "prod": prod,
        "quot": np.asarray(quot, dtype=np.float64),
        "rem": np.asarray(rem, dtype=np.float64),
        "companion": companion,
        "p1": p1,
        "p_monic": p_monic,
    }


def equiv() -> dict[str, object]:
    s = scenario()
    return {
        "P_REAL": np.array([s["p_real"]], dtype=np.float64),
        "P_C_RE": np.array([s["p_c_re"]], dtype=np.float64),
        "P_C_IM": np.array([s["p_c_im"]], dtype=np.float64),
        "DERIV": s["deriv"],
        "P_DERIV": np.array([s["p_deriv"]], dtype=np.float64),
        "INTEG": s["integ"],
        "P_INTEG": np.array([s["p_integ"]], dtype=np.float64),
        "PROD": s["prod"],
        "QUOT": s["quot"],
        "REM": s["rem"],
        "COMPANION": s["companion"],
    }


def print_transcript() -> None:
    s = scenario()
    print("=== Polynomial Numerical Prototype Oracle ===")
    print("\n--- Polynomial Evaluation & Calculus ---")
    print(f"Coefficients (ascending): {list(s['coeffs'])}")
    print(f"p({s['x_test']}) = {s['p_real']:.10f}")
    print(
        f"p(1.0 + 2.0j) = {s['p_c_re']:.10f} + {s['p_c_im']:.10f}j"
    )
    print(f"p'(x) coefficients: {list(s['deriv'])}")
    print(f"p'({s['x_test']}) = {s['p_deriv']:.10f}")
    print(f"int p(x) dx (c0=5) coefficients: {list(s['integ'])}")
    print(
        f"int_0^{s['x_test']} p(t) dt + 5.0 = {s['p_integ']:.10f}"
    )
    print("\n--- Polynomial Multiplication & Division ---")
    print(f"(1 + 2x) * (3 + 4x) = {list(s['prod'])}")
    print(f"Quotient of ({list(s['prod'])}) / ({list(s['p1'])}): {list(s['quot'])}")
    print(f"Remainder: {list(s['rem'])}")
    print("\n--- Monic Companion Matrix ---")
    print(f"Monic p(x) = {list(s['p_monic'])}")
    print("Companion Matrix C:")
    for row in s["companion"]:
        print("  [" + ", ".join(f"{v:8.4f}" for v in row) + "]")


if __name__ == "__main__":
    print_transcript()
