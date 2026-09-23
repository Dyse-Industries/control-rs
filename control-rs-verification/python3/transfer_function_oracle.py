#!/usr/bin/env python3
"""Transfer function reference oracle generating target/verification/transfer_function.scipy.h5 via SciPy."""

from pathlib import Path
import numpy as np
from scipy import signal

from h5_writer import get_results_dir, write_h5


def synthesize_resonant_notch_system(fn_hz=25.0, zeta_z=0.01, zeta_p=0.25, fc_hz=40.0):
    wn = 2.0 * np.pi * fn_hz
    wc = 2.0 * np.pi * fc_hz
    num_notch = np.array([1.0, 2.0 * zeta_z * wn, wn * wn], dtype=np.float64)
    den_notch = np.array([1.0, 2.0 * zeta_p * wn, wn * wn], dtype=np.float64)
    num_lp = np.array([wc], dtype=np.float64)
    den_lp = np.array([1.0, wc], dtype=np.float64)
    num_s = np.polymul(num_notch, num_lp)
    den_s = np.polymul(den_notch, den_lp)
    return num_s, den_s


def generate_datasets():
    num_s, den_s = synthesize_resonant_notch_system()
    dt = 0.005
    sys_tustin = signal.cont2discrete((num_s, den_s), dt, method="bilinear")
    sys_zoh = signal.cont2discrete((num_s, den_s), dt, method="zoh")

    num_freqs = 200
    freqs_hz = np.linspace(0.1, 99.5, num_freqs, dtype=np.float64)
    w_vec = 2.0 * np.pi * freqs_hz

    _, h_cont = signal.freqs(num_s, den_s, worN=w_vec)
    cont_mag_db = 20.0 * np.log10(np.abs(h_cont))
    cont_phase_deg = np.angle(h_cont) * 180.0 / np.pi

    num_t, den_t = sys_tustin[0].squeeze(), sys_tustin[1]
    _, h_tustin = signal.dfreqresp((num_t, den_t, dt), w=w_vec * dt)
    tustin_mag_db = 20.0 * np.log10(np.abs(h_tustin))
    tustin_phase_deg = np.angle(h_tustin) * 180.0 / np.pi

    num_z, den_z = sys_zoh[0].squeeze(), sys_zoh[1]
    _, h_zoh = signal.dfreqresp((num_z, den_z, dt), w=w_vec * dt)
    zoh_mag_db = 20.0 * np.log10(np.abs(h_zoh))
    zoh_phase_deg = np.angle(h_zoh) * 180.0 / np.pi

    # Nyquist
    num_nyq = np.array([50.0, 100.0], dtype=np.float64)
    den_nyq = np.array([1.0, 2.0, 25.0, 0.0], dtype=np.float64)
    nyq_freqs = np.logspace(-2.0, 3.0, 200, dtype=np.float64)
    _, h_nyq = signal.freqs(num_nyq, den_nyq, worN=nyq_freqs)
    h_re = h_nyq.real
    h_im = h_nyq.imag

    mag = np.abs(h_nyq)
    phase = np.angle(h_nyq)
    idx_gc = np.argmin(np.abs(mag - 1.0))
    phase_margin_deg = float(180.0 + phase[idx_gc] * 180.0 / np.pi)

    idx_pc = np.argmin(np.abs(h_im))
    pc_mag = mag[idx_pc]
    gain_margin_db = float(-20.0 * np.log10(pc_mag))

    # Controllable canonical form of H(s) = (2s + 3) / (s^2 + 5s + 4).
    # tf2ss orders states highest-derivative first; reversing the state order
    # gives the companion form with the coefficient row last.
    a_ss, b_ss, c_ss, _ = signal.tf2ss([2.0, 3.0], [1.0, 5.0, 4.0])
    ccf_a = a_ss[::-1, ::-1].flatten()
    ccf_b = b_ss[::-1, :].flatten()
    ccf_c = c_ss[:, ::-1].flatten()

    datasets = {
        "realization/ccf_a": ccf_a,
        "realization/ccf_b": ccf_b,
        "realization/ccf_c": ccf_c,
        "bode/freqs_hz": freqs_hz,
        "bode/cont_mag_db": cont_mag_db,
        "bode/cont_phase_deg": cont_phase_deg,
        "bode/tustin_mag_db": tustin_mag_db,
        "bode/tustin_phase_deg": tustin_phase_deg,
        "bode/zoh_mag_db": zoh_mag_db,
        "bode/zoh_phase_deg": zoh_phase_deg,
        "nyquist/h_re": h_re,
        "nyquist/h_im": h_im,
        "nyquist/phase_margin_deg": [phase_margin_deg],
        "nyquist/gain_margin_db": [gain_margin_db],
    }

    tolerances = {
        "realization/ccf_a": ("abs", 1e-12),
        "realization/ccf_b": ("abs", 1e-12),
        "realization/ccf_c": ("abs", 1e-12),
        "bode/freqs_hz": ("abs", 1e-6),
        "bode/cont_mag_db": ("abs", 0.1),
        "bode/cont_phase_deg": ("abs", 0.5),
        "bode/tustin_mag_db": ("abs", 0.1),
        "bode/tustin_phase_deg": ("abs", 0.5),
        "bode/zoh_mag_db": ("abs", 0.2),
        "bode/zoh_phase_deg": ("abs", 1.0),
        "nyquist/h_re": ("abs", 0.05),
        "nyquist/h_im": ("abs", 0.05),
        "nyquist/phase_margin_deg": ("abs", 1.0),
        "nyquist/gain_margin_db": ("abs", 1.0),
    }

    return datasets, tolerances


def main():
    datasets, tolerances = generate_datasets()
    out_file = get_results_dir() / "transfer_function.scipy.h5"
    write_h5(out_file, datasets, tolerances)


if __name__ == "__main__":
    main()
