#!/usr/bin/env python3
"""Polynomial numerical oracle — writes results/polynomial/python.json."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.polynomial.polynomial import (
    polycompanion,
    polyder,
    polydiv,
    polyint,
    polymul,
    polyval,
)

CRATE_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = CRATE_ROOT / "results" / "polynomial" / "python.json"


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
    }


def build_artifact() -> dict:
    s = scenario()
    return {
        "slug": "polynomial",
        "source": "python",
        "values": {
            "P_REAL": s["p_real"],
            "P_C_RE": s["p_c_re"],
            "P_C_IM": s["p_c_im"],
            "DERIV": s["deriv"].tolist(),
            "P_DERIV": s["p_deriv"],
            "INTEG": s["integ"].tolist(),
            "P_INTEG": s["p_integ"],
            "PROD": s["prod"].tolist(),
            "QUOT": s["quot"].tolist(),
            "REM": float(np.asarray(s["rem"]).reshape(-1)[0]),
            "COMPANION": s["companion"].tolist(),
        },
        "series": {},
    }


if __name__ == "__main__":
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(build_artifact(), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT_PATH}")
