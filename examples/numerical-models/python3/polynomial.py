#!/usr/bin/env python3
"""Polynomial numerical oracle — suite path in, result JSON on stdout."""

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

from vv import case_inputs, require_int, run_cli, time_kernel, timing_entry

SWEEP_N = 128


def tutorial(suite: dict) -> dict:
    inp = case_inputs(suite, "polynomial.host.tutorial")
    coeffs = np.array(inp["coeffs"], dtype=np.float64)
    x_test = float(inp["x_real"])
    z = complex(inp["z_complex"]["re"], inp["z_complex"]["im"])
    deriv = polyder(coeffs)
    deriv5 = np.zeros(5, dtype=np.float64)
    deriv5[: deriv.size] = deriv
    integ = polyint(coeffs, k=[float(inp["integral_c0"])])
    integ5 = integ[:5]
    p1 = np.array(inp["p1"], dtype=np.float64)
    p2 = np.array(inp["p2"], dtype=np.float64)
    prod = polymul(p1, p2)
    quot, rem = polydiv(prod, p1)
    p_monic = np.array(inp["monic"], dtype=np.float64)
    require_int(inp, "monic_degree", 12)
    if p_monic.size != 13:
        raise SystemExit(f"monic length {p_monic.size} != 13")
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


def clustered(suite: dict) -> dict:
    inp = case_inputs(suite, "polynomial.host.clustered_horner")
    require_int(inp, "degree", 16)
    require_int(inp["sweep"], "n", SWEEP_N)
    start = float(inp["sweep"]["start"])
    stop = float(inp["sweep"]["stop"])
    timed_x = float(inp["timed_x"])
    iters = int(inp.get("iters", 10_000))
    roots = np.concatenate(
        [np.full(8, 1.0), np.full(8, 1.01)],
    ).astype(np.float64)
    coeffs = polyfromroots(roots)
    sweep_x = np.linspace(start, stop, SWEEP_N, dtype=np.float64)
    cluster_y = np.asarray(polyval(sweep_x, coeffs), dtype=np.float64)
    ns = time_kernel(iters, lambda: polyval(timed_x, coeffs))
    return {
        "coeffs": np.asarray(coeffs, dtype=np.float64),
        "x": sweep_x,
        "y": cluster_y,
        "ns": ns,
        "iters": iters,
    }


def build_artifact(suite: dict) -> dict:
    t = tutorial(suite)
    c = clustered(suite)
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
            "horner": timing_entry(c["iters"], c["ns"]),
        },
    }


if __name__ == "__main__":
    run_cli("polynomial", build_artifact)
