#!/usr/bin/env python3
"""
python3/transfer_function_validation.py

Executes NumPy/SciPy equivalents for transfer function numerical models.
Outputs JSON results to stdout for cross-language validation with Rust.
"""

from __future__ import annotations

import json
import time

import numpy as np
from numpy.polynomial.polynomial import polyfromroots, polymul
from scipy import signal

BODE_N = 128


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


def run_transfer_function_oracle() -> dict:
    num = np.array([100.0], dtype=np.float64)
    den = np.array([100.0, 4.0, 1.0], dtype=np.float64)
    freqs = np.logspace(-2.0, 3.0, BODE_N, dtype=np.float64)
    omega_n = 10.0
    zeta = 0.2

    t0 = time.perf_counter_ns()
    _w, h = signal.freqs(np.flip(num), np.flip(den), worN=freqs)
    bode_time_ns = float(time.perf_counter_ns() - t0)
    mag = np.abs(h)
    phase = np.angle(h)

    # 2. Series Cascade
    h1_num = np.array([2.0], dtype=np.float64)
    h1_den = np.array([2.0, 1.0], dtype=np.float64)
    h2_num = np.array([5.0], dtype=np.float64)
    h2_den = np.array([5.0, 1.0], dtype=np.float64)

    t0 = time.perf_counter_ns()
    num_ser = polymul(h1_num, h2_num)
    den_ser = polymul(h1_den, h2_den)
    series_time_ns = float(time.perf_counter_ns() - t0)

    # 3. CCF Realization
    ccf_num = np.array([2.0, 3.0], dtype=np.float64)
    ccf_den = np.array([4.0, 5.0, 1.0], dtype=np.float64)

    t0 = time.perf_counter_ns()
    ccf_a, ccf_b, ccf_c, ccf_d = _ccf_from_scipy(ccf_num, ccf_den)
    ccf_time_ns = float(time.perf_counter_ns() - t0)

    # 4. Clustered Poles
    den_c = polymul(polyfromroots(np.full(4, -1.0)), polyfromroots(np.full(4, -1.01)))
    num_c = np.array([1.0], dtype=np.float64)

    t0 = time.perf_counter_ns()
    _wc, hc = signal.freqs(np.flip(num_c), np.flip(den_c), worN=freqs)
    cluster_bode_time_ns = float(time.perf_counter_ns() - t0)

    return {
        "complex_pair": {
            "h_re": np.asarray(h.real, dtype=np.float64).tolist(),
            "h_im": np.asarray(h.imag, dtype=np.float64).tolist(),
            "freqs": freqs.tolist(),
            "mag": mag.tolist(),
            "phase": phase.tolist(),
            "omega_n": omega_n,
            "zeta": zeta,
            "bode_time_ns": bode_time_ns,
        },
        "series": {
            "num_ser": np.asarray(num_ser, dtype=np.float64).tolist(),
            "den_ser": np.asarray(den_ser, dtype=np.float64).tolist(),
            "series_time_ns": series_time_ns,
        },
        "ccf": {
            "a": ccf_a.tolist(),
            "b": ccf_b.tolist(),
            "c": ccf_c.tolist(),
            "d": ccf_d.tolist(),
            "ccf_time_ns": ccf_time_ns,
        },
        "clustered": {
            "h_re": np.asarray(hc.real, dtype=np.float64).tolist(),
            "h_im": np.asarray(hc.imag, dtype=np.float64).tolist(),
            "mag": np.abs(hc).tolist(),
            "cluster_bode_time_ns": cluster_bode_time_ns,
        },
    }


if __name__ == "__main__":
    results = run_transfer_function_oracle()
    print(json.dumps(results))
