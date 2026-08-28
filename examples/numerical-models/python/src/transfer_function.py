#!/usr/bin/env python3
"""Transfer-function numerical oracle — writes results/transfer_function/python.json."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from numpy.polynomial.polynomial import polymul
from scipy import signal

CRATE_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = CRATE_ROOT / "results" / "transfer_function" / "python.json"


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


def scenario() -> dict[str, object]:
    omega_c = 10.0
    w_c2 = omega_c**2
    sqrt2_wc = math.sqrt(2.0) * omega_c
    num = np.array([w_c2], dtype=np.float64)
    den = np.array([w_c2, sqrt2_wc, 1.0], dtype=np.float64)
    freqs = np.array([0.1, 1.0, 10.0, 100.0], dtype=np.float64)
    _w, h = signal.freqs(np.flip(num), np.flip(den), worN=freqs)
    h1_num, h1_den = np.array([2.0]), np.array([2.0, 1.0])
    h2_num, h2_den = np.array([5.0]), np.array([5.0, 1.0])
    num_ser = polymul(h1_num, h2_num)
    den_ser = polymul(h1_den, h2_den)
    ccf_num = np.array([2.0, 3.0], dtype=np.float64)
    ccf_den = np.array([4.0, 5.0, 1.0], dtype=np.float64)
    a, b, c, d = _ccf_from_scipy(ccf_num, ccf_den)
    mag = np.abs(h)
    return {
        "h_re": np.asarray(h.real, dtype=np.float64),
        "h_im": np.asarray(h.imag, dtype=np.float64),
        "num_ser": np.asarray(num_ser, dtype=np.float64),
        "den_ser": np.asarray(den_ser, dtype=np.float64),
        "ccf_a": a,
        "ccf_b": b,
        "ccf_c": c,
        "ccf_d": d,
        "freqs": freqs,
        "mag": mag,
    }


def build_artifact() -> dict:
    s = scenario()
    return {
        "slug": "transfer_function",
        "source": "python",
        "values": {
            "H_RE": s["h_re"].tolist(),
            "H_IM": s["h_im"].tolist(),
            "NUM_SER": s["num_ser"].tolist(),
            "DEN_SER": s["den_ser"].tolist(),
            "CCF_A": s["ccf_a"].tolist(),
            "CCF_B": s["ccf_b"].tolist(),
            "CCF_C": s["ccf_c"].tolist(),
            "CCF_D": s["ccf_d"].tolist(),
        },
        "series": {
            "bode_mag": {
                "x": s["freqs"].tolist(),
                "y": s["mag"].tolist(),
            }
        },
    }


if __name__ == "__main__":
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(build_artifact(), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT_PATH}")
