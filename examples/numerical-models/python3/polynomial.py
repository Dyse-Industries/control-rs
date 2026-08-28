#!/usr/bin/env python3
"""Polynomial numerical oracle — writes results/polynomial/python.json."""

from __future__ import annotations

import numpy as np
from numpy.polynomial.polynomial import (
    polycompanion,
    polyder,
    polydiv,
    polyfromroots,
    polyint,
    polymul,
    polyval,
)

from vv import CRATE_ROOT, save_json, time_kernel, timing_entry

OUT_PATH = CRATE_ROOT / "results" / "polynomial" / "python.json"
SWEEP_N = 128
HORNER_ITERS = 10_000
CLUSTER_X = 1.005


def tutorial() -> dict:
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


def clustered() -> dict:
    roots = np.concatenate(
        [np.full(8, 1.0), np.full(8, 1.01)],
    ).astype(np.float64)
    coeffs = polyfromroots(roots)
    sweep_x = np.linspace(0.9, 1.1, SWEEP_N, dtype=np.float64)
    cluster_y = np.asarray(polyval(sweep_x, coeffs), dtype=np.float64)
    ns = time_kernel(HORNER_ITERS, lambda: polyval(CLUSTER_X, coeffs))
    return {
        "coeffs": np.asarray(coeffs, dtype=np.float64),
        "x": sweep_x,
        "y": cluster_y,
        "ns": ns,
    }


def build_artifact() -> dict:
    t = tutorial()
    c = clustered()
    return {
        "slug": "polynomial",
        "source": "python",
        "values": {
            "P_REAL": t["p_real"],
            "P_C_RE": t["p_c_re"],
            "P_C_IM": t["p_c_im"],
            "DERIV": t["deriv"].tolist(),
            "P_DERIV": t["p_deriv"],
            "INTEG": t["integ"].tolist(),
            "P_INTEG": t["p_integ"],
            "PROD": t["prod"].tolist(),
            "QUOT": t["quot"].tolist(),
            "REM": float(np.asarray(t["rem"]).reshape(-1)[0]),
            "COMPANION": t["companion"].tolist(),
            "CLUSTER_COEFFS": c["coeffs"].tolist(),
            "CLUSTER_X": c["x"].tolist(),
            "CLUSTER_Y": c["y"].tolist(),
        },
        "series": {
            "horner": {"x": c["x"].tolist(), "y": c["y"].tolist()},
        },
        "metrics": {},
        "timings": {
            "horner": timing_entry(HORNER_ITERS, c["ns"]),
        },
    }


if __name__ == "__main__":
    save_json(OUT_PATH, build_artifact())
