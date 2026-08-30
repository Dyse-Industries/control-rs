#!/usr/bin/env python3
"""
python3/transfer_function_validation.py

Executes NumPy/SciPy equivalents for transfer function numerical models across benchmark domains:
1 & 2. Discretization Method Error (Flexible Structure Modal System with Resonant Notch Filter via synthesize_resonant_notch_system, Tustin vs ZOH Bode magnitude & phase near Nyquist)
3. Nyquist Stability Criterion & Margins (Polar curve, (-1, 0j) point, gain & phase margins)
4. Filter Topology Stability (6th-order Butterworth f32 Direct Form vs Biquad SOS)
"""

from __future__ import annotations

import json
import time

import numpy as np
from scipy import signal


def synthesize_resonant_notch_system(
    fn_hz: float, zeta_z: float, zeta_p: float, fc_hz: float
) -> tuple[np.ndarray, np.ndarray]:
    """Script-level constructor for Flexible Structure Modal System with Resonant Notch Filter."""
    wn = 2.0 * np.pi * fn_hz
    wc = 2.0 * np.pi * fc_hz

    num_notch = np.array([1.0, 2.0 * zeta_z * wn, wn * wn], dtype=np.float64)
    den_notch = np.array([1.0, 2.0 * zeta_p * wn, wn * wn], dtype=np.float64)

    num_lp = np.array([wc], dtype=np.float64)
    den_lp = np.array([1.0, wc], dtype=np.float64)

    num_s = np.polymul(num_notch, num_lp)
    den_s = np.polymul(den_notch, den_lp)

    return num_s, den_s


def benchmark_discretization_error() -> dict:
    fn_hz = 25.0
    zeta_z = 0.01
    zeta_p = 0.25
    fc_hz = 40.0

    num_s, den_s = synthesize_resonant_notch_system(fn_hz, zeta_z, zeta_p, fc_hz)

    dt = 0.005  # Fs = 200 Hz, Nyquist = 100 Hz
    sys_tustin = signal.cont2discrete((num_s, den_s), dt, method='bilinear')
    sys_zoh = signal.cont2discrete((num_s, den_s), dt, method='zoh')

    num_freqs = 1000
    freqs_hz = np.linspace(0.1, 99.5, num_freqs, dtype=np.float64)
    w_vec = 2.0 * np.pi * freqs_hz

    t0 = time.perf_counter_ns()

    _, h_cont = signal.freqs(num_s, den_s, worN=w_vec)
    cont_mag_db = (20.0 * np.log10(np.abs(h_cont))).tolist()
    cont_phase_deg = (np.angle(h_cont) * 180.0 / np.pi).tolist()

    num_t, den_t = sys_tustin[0].squeeze(), sys_tustin[1]
    _, h_tustin = signal.dfreqresp((num_t, den_t, dt), w=w_vec * dt)
    tustin_mag_db = (20.0 * np.log10(np.abs(h_tustin))).tolist()
    tustin_phase_deg = (np.angle(h_tustin) * 180.0 / np.pi).tolist()

    num_z, den_z = sys_zoh[0].squeeze(), sys_zoh[1]
    _, h_zoh = signal.dfreqresp((num_z, den_z, dt), w=w_vec * dt)
    zoh_mag_db = (20.0 * np.log10(np.abs(h_zoh))).tolist()
    zoh_phase_deg = (np.angle(h_zoh) * 180.0 / np.pi).tolist()

    bode_time_ns = float(time.perf_counter_ns() - t0)

    return {
        "freqs_hz": freqs_hz.tolist(),
        "cont_mag_db": cont_mag_db,
        "cont_phase_deg": cont_phase_deg,
        "tustin_mag_db": tustin_mag_db,
        "tustin_phase_deg": tustin_phase_deg,
        "zoh_mag_db": zoh_mag_db,
        "zoh_phase_deg": zoh_phase_deg,
        "bode_time_ns": bode_time_ns,
    }


def benchmark_nyquist_criterion() -> dict:
    num_s = np.array([50.0, 100.0], dtype=np.float64)
    den_s = np.array([1.0, 2.0, 25.0, 0.0], dtype=np.float64)

    freqs = np.logspace(-2.0, 3.0, 250, dtype=np.float64)
    _w, h = signal.freqs(num_s, den_s, worN=freqs)

    h_re = h.real.tolist()
    h_im = h.imag.tolist()

    mag = np.abs(h)
    phase = np.angle(h)

    idx_gc = np.argmin(np.abs(mag - 1.0))
    gain_crossover_w = float(freqs[idx_gc])
    phase_margin_deg = float(180.0 + phase[idx_gc] * 180.0 / np.pi)

    idx_pc = np.argmin(np.abs(np.abs(phase) - np.pi))
    phase_crossover_w = float(freqs[idx_pc])
    gain_margin_db = float(-20.0 * np.log10(mag[idx_pc]))

    return {
        "freqs": freqs.tolist(),
        "h_re": h_re,
        "h_im": h_im,
        "critical_point": [-1.0, 0.0],
        "gain_crossover_w": gain_crossover_w,
        "phase_crossover_w": phase_crossover_w,
        "phase_margin_deg": phase_margin_deg,
        "gain_margin_db": gain_margin_db,
    }


def benchmark_topology_stability() -> dict:
    cutoff_hz = 35.0
    wc = 2.0 * np.pi * cutoff_hz
    dt = 0.01

    k = np.arange(6)
    angles = np.pi * (2 * k + 7) / 12.0
    s_poles = wc * (np.cos(angles) + 1j * np.sin(angles))

    gt_z_poles = (1.0 + s_poles * dt / 2.0) / (1.0 - s_poles * dt / 2.0)
    gt_re = gt_z_poles.real.tolist()
    gt_im = gt_z_poles.imag.tolist()

    poly_f32 = np.array([1.0], dtype=np.float32)
    for pair_idx in range(3):
        p1 = gt_z_poles[2 * pair_idx]
        p2 = gt_z_poles[2 * pair_idx + 1]
        b1 = float(-(p1.real + p2.real))
        b0 = float(p1.real * p2.real + p1.imag * p1.imag)
        sec = np.array([1.0, b1, b0], dtype=np.float32)
        poly_f32 = np.polymul(poly_f32, sec)

    df_roots = np.roots(poly_f32)
    df_re = df_roots.real.astype(float).tolist()
    df_im = df_roots.imag.astype(float).tolist()

    biquad_re = []
    biquad_im = []
    for pair_idx in range(3):
        p1 = gt_z_poles[2 * pair_idx]
        p2 = gt_z_poles[2 * pair_idx + 1]
        b1 = float(-(p1.real + p2.real))
        b0 = float(p1.real * p2.real + p1.imag * p1.imag)
        roots_sec = np.roots(np.array([1.0, b1, b0], dtype=np.float32))
        for r in roots_sec:
            biquad_re.append(float(r.real))
            biquad_im.append(float(r.imag))

    return {
        "ground_truth_re": gt_re,
        "ground_truth_im": gt_im,
        "direct_form_re": df_re,
        "direct_form_im": df_im,
        "biquad_re": biquad_re,
        "biquad_im": biquad_im,
    }


def run_transfer_function_oracle() -> dict:
    discretization = benchmark_discretization_error()
    nyquist = benchmark_nyquist_criterion()
    topology = benchmark_topology_stability()

    return {
        "discretization_error": discretization,
        "nyquist_criterion": nyquist,
        "topology_stability": topology,
        "tutorial": {
            "a00": 0.0,
            "a01": 1.0,
            "a10": -4.0,
            "a11": -5.0,
        },
    }


if __name__ == "__main__":
    results = run_transfer_function_oracle()
    print(json.dumps(results))