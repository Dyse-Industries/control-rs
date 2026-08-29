#!/usr/bin/env python3
"""Transfer-function numerical oracle — suite path in, result JSON on stdout."""

from __future__ import annotations

import numpy as np
from numpy.polynomial.polynomial import polyfromroots, polymul
from scipy import signal

from vv import case_inputs, require_int, run_cli, time_kernel, timing_entry

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


def build_artifact(suite: dict) -> dict:
    pair = case_inputs(suite, "transfer_function.host.complex_pair")
    require_int(pair["freqs"], "n", BODE_N)
    num = np.array(pair["num"], dtype=np.float64)
    den = np.array(pair["den"], dtype=np.float64)
    start = float(pair["freqs"]["start_log10"])
    stop = float(pair["freqs"]["stop_log10"])
    freqs = np.logspace(start, stop, BODE_N, dtype=np.float64)
    iters = int(pair.get("iters", 50))
    _w, h = signal.freqs(np.flip(num), np.flip(den), worN=freqs)
    mag = np.abs(h)
    phase = np.angle(h)
    omega_n = float(pair["omega_n"])
    zeta = float(pair["zeta"])

    ser = case_inputs(suite, "transfer_function.host.series")
    h1_num = np.array(ser["H1"]["num"], dtype=np.float64)
    h1_den = np.array(ser["H1"]["den"], dtype=np.float64)
    h2_num = np.array(ser["H2"]["num"], dtype=np.float64)
    h2_den = np.array(ser["H2"]["den"], dtype=np.float64)
    num_ser = polymul(h1_num, h2_num)
    den_ser = polymul(h1_den, h2_den)

    ccf = case_inputs(suite, "transfer_function.host.ccf")
    ccf_num = np.array(ccf["num"], dtype=np.float64)
    ccf_den = np.array(ccf["den"], dtype=np.float64)
    a, b, c, d = _ccf_from_scipy(ccf_num, ccf_den)

    cl = case_inputs(suite, "transfer_function.host.clustered_poles")
    require_int(cl["freqs"], "n", BODE_N)
    den_c = polymul(polyfromroots(np.full(4, -1.0)), polyfromroots(np.full(4, -1.01)))
    num_c = np.array(cl["num"], dtype=np.float64)
    _wc, hc = signal.freqs(np.flip(num_c), np.flip(den_c), worN=freqs)
    ns = time_kernel(
        iters,
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
            "OMEGA_N": omega_n,
            "ZETA": zeta,
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
            "bode": timing_entry(iters, ns),
        },
    }


if __name__ == "__main__":
    run_cli("transfer_function", build_artifact)
