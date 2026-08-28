#!/usr/bin/env python3
"""Transfer-function numerical oracle — writes results/transfer_function/python.json."""

from __future__ import annotations

import math

import numpy as np
from numpy.polynomial.polynomial import polyfromroots, polymul
from scipy import signal

from vv import CRATE_ROOT, save_json, time_kernel, timing_entry

OUT_PATH = CRATE_ROOT / "results" / "transfer_function" / "python.json"
BODE_N = 128
BODE_ITERS = 50


def _ccf_from_scipy(num_asc: np.ndarray, den_asc: np.ndarray) -> tuple:
    order = den_asc.size - 1
    a = np.zeros((order, order), dtype=np.float64)
    for i in range(order - 1):
        a[i, i + 1] = 1.0
    a[-1, :] = -den_asc[:-1] / den_asc[-1]
    b = np.zeros((order, 1), dtype=np.float64)
    b[-1, 0] = 1.0
    num_desc = np.flip(num_asc)
    den_desc = np.flip(den_asc)
    _a_s, _b_s, _c_s, d_s = signal.tf2ss(num_desc, den_desc)
    d_val = float(np.asarray(d_s).reshape(-1)[0])
    c = np.zeros((1, order), dtype=np.float64)
    padded_num = np.zeros(order, dtype=np.float64)
    padded_num[: num_asc.size] = num_asc[:order]
    c[0, :] = padded_num / den_asc[-1] - d_val * (den_asc[:-1] / den_asc[-1])
    d = np.array([[d_val]], dtype=np.float64)
    return a, b, c, d


def build_artifact() -> dict:
    omega_c = 10.0
    w_c2 = omega_c**2
    sqrt2_wc = math.sqrt(2.0) * omega_c
    num = np.array([w_c2], dtype=np.float64)
    den = np.array([w_c2, sqrt2_wc, 1.0], dtype=np.float64)
    freqs = np.logspace(-2.0, 3.0, BODE_N, dtype=np.float64)
    _w, h = signal.freqs(np.flip(num), np.flip(den), worN=freqs)
    mag = np.abs(h)
    phase = np.angle(h)
    h1_num, h1_den = np.array([2.0]), np.array([2.0, 1.0])
    h2_num, h2_den = np.array([5.0]), np.array([5.0, 1.0])
    num_ser = polymul(h1_num, h2_num)
    den_ser = polymul(h1_den, h2_den)
    ccf_num = np.array([2.0, 3.0], dtype=np.float64)
    ccf_den = np.array([4.0, 5.0, 1.0], dtype=np.float64)
    a, b, c, d = _ccf_from_scipy(ccf_num, ccf_den)
    den_c = polymul(polyfromroots(np.full(4, -1.0)), polyfromroots(np.full(4, -1.01)))
    num_c = np.array([1.0], dtype=np.float64)
    _wc, hc = signal.freqs(np.flip(num_c), np.flip(den_c), worN=freqs)
    ns = time_kernel(
        BODE_ITERS,
        lambda: signal.freqs(np.flip(num), np.flip(den), worN=freqs),
    )
    return {
        "slug": "transfer_function",
        "source": "python",
        "values": {
            "H_RE": np.asarray(h.real, dtype=np.float64).tolist(),
            "H_IM": np.asarray(h.imag, dtype=np.float64).tolist(),
            "NUM_SER": np.asarray(num_ser, dtype=np.float64).tolist(),
            "DEN_SER": np.asarray(den_ser, dtype=np.float64).tolist(),
            "CCF_A": a.tolist(),
            "CCF_B": b.tolist(),
            "CCF_C": c.tolist(),
            "CCF_D": d.tolist(),
            "CLUSTER_H_RE": np.asarray(hc.real, dtype=np.float64).tolist(),
            "CLUSTER_H_IM": np.asarray(hc.imag, dtype=np.float64).tolist(),
            "FREQS": freqs.tolist(),
        },
        "series": {
            "bode_mag": {"x": freqs.tolist(), "y": mag.tolist()},
            "bode_phase": {"x": freqs.tolist(), "y": phase.tolist()},
            "cluster_mag": {
                "x": freqs.tolist(),
                "y": np.abs(hc).tolist(),
            },
        },
        "metrics": {},
        "timings": {
            "bode": timing_entry(BODE_ITERS, ns),
        },
    }


if __name__ == "__main__":
    save_json(OUT_PATH, build_artifact())
