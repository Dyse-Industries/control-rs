#!/usr/bin/env python3
"""Transfer-function numerical prototype oracle (SciPy signal)."""

from __future__ import annotations

import math

import numpy as np
from numpy.polynomial.polynomial import polymul
from scipy import signal


def _ccf_from_scipy(num_asc: np.ndarray, den_asc: np.ndarray) -> tuple:
    """Map ``scipy.signal.tf2ss`` to crate last-row controllable companion."""
    num_desc = np.flip(num_asc)
    den_desc = np.flip(den_asc)
    a_s, b_s, c_s, d_s = signal.tf2ss(num_desc, den_desc)
    # Crate CCF: superdiagonal ones, last row -a_i (ascending, monic), B = e_n.
    order = den_asc.size - 1
    a = np.zeros((order, order), dtype=np.float64)
    for i in range(order - 1):
        a[i, i + 1] = 1.0
    a[-1, :] = -den_asc[:-1] / den_asc[-1]
    b = np.zeros((order, 1), dtype=np.float64)
    b[-1, 0] = 1.0
    d_val = float(np.asarray(d_s).reshape(-1)[0])
    # C row: numerator after extracting feedthrough, matching crate β.
    c = np.zeros((1, order), dtype=np.float64)
    padded_num = np.zeros(order, dtype=np.float64)
    padded_num[: num_asc.size] = num_asc[:order]
    c[0, :] = padded_num / den_asc[-1] - d_val * (den_asc[:-1] / den_asc[-1])
    d = np.array([[d_val]], dtype=np.float64)
    del a_s, b_s, c_s
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
    return {
        "num": num,
        "den": den,
        "freqs": freqs,
        "h_re": np.asarray(h.real, dtype=np.float64),
        "h_im": np.asarray(h.imag, dtype=np.float64),
        "num_ser": np.asarray(num_ser, dtype=np.float64),
        "den_ser": np.asarray(den_ser, dtype=np.float64),
        "ccf_a": a,
        "ccf_b": b,
        "ccf_c": c,
        "ccf_d": d,
    }


def equiv() -> dict[str, object]:
    s = scenario()
    return {
        "H_RE": s["h_re"],
        "H_IM": s["h_im"],
        "NUM_SER": s["num_ser"],
        "DEN_SER": s["den_ser"],
        "CCF_A": s["ccf_a"],
        "CCF_B": s["ccf_b"],
        "CCF_C": s["ccf_c"],
        "CCF_D": s["ccf_d"],
    }


def print_transcript() -> None:
    s = scenario()
    print("=== Transfer Function Numerical Prototype Oracle ===")
    print("\n--- Butterworth Filter (omega_c = 10.0 rad/s) ---")
    print(f"Numerator (ascending): {list(s['num'])}")
    print(f"Denominator (ascending): {list(s['den'])}")
    print("\n--- Frequency Evaluation ---")
    print(
        f"{'omega (rad/s)':<16}{'Real':<16}{'Imag':<16}"
        f"{'Mag (abs)':<16}{'Mag (dB)':<16}{'Phase (deg)':<16}"
    )
    for w, re, im in zip(s["freqs"], s["h_re"], s["h_im"], strict=True):
        mag = math.hypot(re, im)
        mag_db = 20.0 * math.log10(mag) if mag > 0 else float("-inf")
        phase = math.degrees(math.atan2(im, re))
        print(
            f"{w:<16.2f}{re:<16.8f}{im:<16.8f}"
            f"{mag:<16.8f}{mag_db:<16.8f}{phase:<16.8f}"
        )
    print("\n--- Series Cascade H1 * H2 ---")
    print("H1(s): [2.0] / [2.0, 1.0]")
    print("H2(s): [5.0] / [5.0, 1.0]")
    print(f"H_series: {list(s['num_ser'])} / {list(s['den_ser'])}")
    print("\n--- Controllable Canonical Realization ---")
    print("H(s) = (2 + 3s) / (4 + 5s + s^2)")
    for name, mat in (
        ("A", s["ccf_a"]),
        ("B", s["ccf_b"]),
        ("C", s["ccf_c"]),
        ("D", s["ccf_d"]),
    ):
        print(f"{name}:")
        for row in np.atleast_2d(mat):
            print("  " + str([float(v) for v in row]))


if __name__ == "__main__":
    print_transcript()
