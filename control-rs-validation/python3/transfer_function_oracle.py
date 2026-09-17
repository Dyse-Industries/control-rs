#!/usr/bin/env python3
"""
python3/transfer_function_oracle.py

Executes SciPy/harold equivalents for transfer-function numerical models.
Writes `results/transfer_function.scipy.h5` and `results/transfer_function.harold.h5`.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from numpy.polynomial.polynomial import polyfromroots
from scipy import signal

from h5_write import as_1d, write_dict_to_h5


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


def topology_stability_oracle() -> dict:
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


def generate_clustered_pole_bode() -> dict:
    roots = np.array([-1.0, -1.0, -1.0, -1.0, -1.01, -1.01, -1.01, -1.01], dtype=np.float64)
    den = polyfromroots(roots)
    num = np.array([1.0], dtype=np.float64)
    omegas = np.logspace(-1.0, 2.0, 128, dtype=np.float64)
    _, h = signal.freqs(num[::-1], den[::-1], worN=omegas)
    mag_db = (20.0 * np.log10(np.abs(h))).tolist()
    phase_deg = (np.angle(h) * 180.0 / np.pi).tolist()
    return {
        "omega": omegas.tolist(),
        "mag_db": mag_db,
        "phase_deg": phase_deg,
    }


def run_transfer_function_oracle() -> dict:
    discretization = benchmark_discretization_error()
    nyquist = benchmark_nyquist_criterion()
    topology = topology_stability_oracle()
    clustered = generate_clustered_pole_bode()

    # Dynamic Controllable Canonical Form (CCF) derivation for H(s) = (2s + 3)/(s^2 + 5s + 4)
    den_tut = np.array([1.0, 5.0, 4.0], dtype=np.float64)  # s^2 + a_1*s + a_0
    a_norm = den_tut / den_tut[0]
    a_matrix = np.array([
        [0.0, 1.0],
        [-a_norm[2], -a_norm[1]]
    ], dtype=np.float64)

    return {
        "discretization_error": discretization,
        "nyquist_criterion": nyquist,
        "topology_stability": topology,
        "clustered_pole_bode": clustered,
    }


def run_harold_oracle() -> dict | None:
    """Independent frequency response from harold, or None when unavailable."""
    try:
        import harold
    except ImportError:
        return None

    fn_hz = 25.0
    zeta_z = 0.01
    zeta_p = 0.25
    fc_hz = 40.0
    wn = 2.0 * np.pi * fn_hz
    wc = 2.0 * np.pi * fc_hz
    dt = 0.005

    num_notch = np.array([1.0, 2.0 * zeta_z * wn, wn * wn])
    den_notch = np.array([1.0, 2.0 * zeta_p * wn, wn * wn])
    num_lp = np.array([wc])
    den_lp = np.array([1.0, wc])
    num_s = np.polymul(num_notch, num_lp)
    den_s = np.polymul(den_notch, den_lp)

    # Continuous frequency response via harold (Misra-Patel Hessenberg algorithm)
    tf_cont = harold.Transfer(num_s.tolist(), den_s.tolist())
    num_freqs = 1000
    freqs_hz = np.linspace(0.1, 99.5, num_freqs)
    w_vec = 2.0 * np.pi * freqs_hz  # rad/s

    H_cont, w_out = harold.frequency_response(tf_cont, w=w_vec, w_unit='rad/s', output_unit='rad/s')
    h_cont = np.asarray(H_cont).squeeze()
    cont_mag_db = as_1d(20.0 * np.log10(np.abs(h_cont))).tolist()
    cont_phase_deg = as_1d(np.angle(h_cont) * 180.0 / np.pi).tolist()

    # Tustin discretization then freq response.
    # harold internally scales w by the system's SamplingPeriod when w_unit='rad/s',
    # so pass the raw rad/s vector here (not pre-multiplied by dt as SciPy's
    # dfreqresp requires) or the effective evaluation frequency collapses toward DC.
    sys_tustin = harold.discretize(tf_cont, dt, method='tustin')
    H_tustin, _ = harold.frequency_response(sys_tustin, w=w_vec, w_unit='rad/s', output_unit='rad/s')
    h_tustin = np.asarray(H_tustin).squeeze()
    tustin_mag_db = as_1d(20.0 * np.log10(np.abs(h_tustin))).tolist()
    tustin_phase_deg = as_1d(np.angle(h_tustin) * 180.0 / np.pi).tolist()

    # ZOH discretization then freq response
    sys_zoh = harold.discretize(tf_cont, dt, method='zoh')
    H_zoh, _ = harold.frequency_response(sys_zoh, w=w_vec, w_unit='rad/s', output_unit='rad/s')
    h_zoh = np.asarray(H_zoh).squeeze()
    zoh_mag_db = as_1d(20.0 * np.log10(np.abs(h_zoh))).tolist()
    zoh_phase_deg = as_1d(np.angle(h_zoh) * 180.0 / np.pi).tolist()

    # Nyquist: open-loop TF G(s) = (50s + 100) / (s^3 + 2s^2 + 25s)
    tf_nyq = harold.Transfer([50.0, 100.0], [1.0, 2.0, 25.0, 0.0])
    freqs_nyq = np.logspace(-2.0, 3.0, 250)
    H_nyq, _ = harold.frequency_response(tf_nyq, w=freqs_nyq, w_unit='rad/s', output_unit='rad/s')
    h_nyq = np.asarray(H_nyq).squeeze()

    mag_nyq = np.abs(h_nyq)
    phase_nyq = np.angle(h_nyq)

    idx_gc = np.argmin(np.abs(mag_nyq - 1.0))
    phase_margin_deg = float(180.0 + phase_nyq[idx_gc] * 180.0 / np.pi)
    idx_pc = np.argmin(np.abs(np.abs(phase_nyq) - np.pi))
    gain_margin_db = float(-20.0 * np.log10(mag_nyq[idx_pc]))

    return {
        "discretization_error": {
            "freqs_hz": freqs_hz.tolist(),
            "cont_mag_db": cont_mag_db,
            "cont_phase_deg": cont_phase_deg,
            "tustin_mag_db": tustin_mag_db,
            "tustin_phase_deg": tustin_phase_deg,
            "zoh_mag_db": zoh_mag_db,
            "zoh_phase_deg": zoh_phase_deg,
        },
        "nyquist_criterion": {
            "freqs": freqs_nyq.tolist(),
            "h_re": as_1d(np.real(h_nyq)).tolist(),
            "h_im": as_1d(np.imag(h_nyq)).tolist(),
            "phase_margin_deg": phase_margin_deg,
            "gain_margin_db": gain_margin_db,
        },
    }


if __name__ == "__main__":
    from pathlib import Path

    from h5_write import attr_specs_from_toml, present_paths, write_variant_file

    scipy_results = run_transfer_function_oracle()
    harold_results = run_harold_oracle()

    gated = [
        "discretization_error/cont_mag_db",
        "discretization_error/cont_phase_deg",
        "discretization_error/tustin_mag_db",
        "discretization_error/tustin_phase_deg",
        "discretization_error/zoh_mag_db",
        "discretization_error/zoh_phase_deg",
        "nyquist_criterion/h_re",
        "nyquist_criterion/h_im",
        "nyquist_criterion/phase_margin_deg",
        "nyquist_criterion/gain_margin_db",
        "clustered_pole_bode/mag_db",
        "clustered_pole_bode/phase_deg",
    ]
    signal_keys = {
        "discretization_error/cont_mag_db": "nmv.transfer_function.discretization.cont_mag_db",
        "discretization_error/cont_phase_deg": "nmv.transfer_function.discretization.cont_phase_deg",
        "discretization_error/tustin_mag_db": "nmv.transfer_function.discretization.tustin_mag_db",
        "discretization_error/tustin_phase_deg": "nmv.transfer_function.discretization.tustin_phase_deg",
        "discretization_error/zoh_mag_db": "nmv.transfer_function.discretization.zoh_mag_db",
        "discretization_error/zoh_phase_deg": "nmv.transfer_function.discretization.zoh_phase_deg",
        "nyquist_criterion/h_re": "nmv.transfer_function.nyquist.h_re",
        "nyquist_criterion/h_im": "nmv.transfer_function.nyquist.h_im",
        "nyquist_criterion/phase_margin_deg": "nmv.transfer_function.nyquist.phase_margin_deg",
        "nyquist_criterion/gain_margin_db": "nmv.transfer_function.nyquist.gain_margin_db",
        "clustered_pole_bode/mag_db": "nmv.transfer_function.clustered_pole.mag_db",
        "clustered_pole_bode/phase_deg": "nmv.transfer_function.clustered_pole.phase_deg",
    }
    harold_keys = {
        "discretization_error/cont_mag_db": "nmv.transfer_function.harold.cont_mag_db",
        "discretization_error/cont_phase_deg": "nmv.transfer_function.harold.cont_phase_deg",
        "discretization_error/tustin_mag_db": "nmv.transfer_function.harold.tustin_mag_db",
        "discretization_error/tustin_phase_deg": "nmv.transfer_function.harold.tustin_phase_deg",
        "discretization_error/zoh_mag_db": "nmv.transfer_function.harold.zoh_mag_db",
        "discretization_error/zoh_phase_deg": "nmv.transfer_function.harold.zoh_phase_deg",
        "nyquist_criterion/h_re": "nmv.transfer_function.harold.h_re",
        "nyquist_criterion/h_im": "nmv.transfer_function.harold.h_im",
        "nyquist_criterion/phase_margin_deg": "nmv.transfer_function.harold.phase_margin_deg",
        "nyquist_criterion/gain_margin_db": "nmv.transfer_function.harold.gain_margin_db",
    }
    table = Path(__file__).resolve().parent.parent / "tolerances/numerical_models.toml"
    specs = attr_specs_from_toml(
        table, signal_keys, {"harold": harold_keys} if harold_results else None
    )
    results = Path("results")
    write_variant_file(
        results / "transfer_function.scipy.h5",
        scipy_results,
        gated_paths=gated,
        meta=scipy_results,
        attr_specs=specs,
    )
    if harold_results:
        harold_paths = present_paths(harold_results, gated)
        if harold_paths:
            write_variant_file(
                results / "transfer_function.harold.h5",
                harold_results,
                gated_paths=harold_paths,
                meta=harold_results,
            )