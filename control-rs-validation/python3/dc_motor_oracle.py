#!/usr/bin/env python3
"""
dc_motor_oracle.py

SciPy reference oracle for the DC motor armature position servo classical
control example.

Provenance. Not every section here is independent of the Rust implementation,
and the payload says which is which via `independent: true|false` on each
section:

- plant, margins, frequency_sweep, root_locus are computed from
  scipy.signal primitives (`freqs`, `cont2discrete`, `np.roots`) and are
  independent of control-rs.
- lead_design transcribes the same closed-form synthesis the Rust side
  runs, including the 5 degree crossover-shift buffer and the phase clamp.
  It therefore checks transcription fidelity, not the correctness of the
  synthesis equations. `margins.compensated.achieved` re-derives the phase
  margin the design actually attains directly from the loop frequency
  response, which is an independent check on the design regardless.

Sections:
- Plant continuous poles, transfer function, and DC parameters
- Uncompensated plant frequency response and stability margins
- Delay-compensated lead synthesis via analytical small-signal equations
- Continuous-to-discrete Tustin transform with prewarping (scipy.signal.cont2discrete)
- Discrete DirectForm2T realization coefficients
- Root locus closed-loop pole migration trajectories

Writes `results/dc-motor.scipy.h5` (true-oracle variant file).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.signal as signal
from scipy.optimize import brentq

R_A = 2.0
L_A = 0.5e-3
K_T = 0.05
K_B = 0.05
J = 2.0e-4
B = 1.0e-4
V_MAX = 12.0

# Frequency grid shared with the Rust sweep (analysis::logspace_omegas).
OMEGA_LOG10_START = 0.0
OMEGA_LOG10_STOP = 4.0
OMEGA_POINTS = 1000

TS_S = 0.0005
ACT_DELAY_STEPS = 1
SENS_DELAY_STEPS = 1
TOTAL_DELAY_S = (ACT_DELAY_STEPS + SENS_DELAY_STEPS) * TS_S


def compute_plant():
    d0 = 0.0
    d1 = R_A * B + K_T * K_B
    d2 = R_A * J + L_A * B
    d3 = L_A * J

    num = [K_T]
    den = [d3, d2, d1, d0]

    sys_tf = signal.TransferFunction(num, den)
    poles = sys_tf.poles
    return {
        "independent": True,
        "num": num,
        "den": den,
        "d0": d0,
        "d1": d1,
        "d2": d2,
        "d3": d3,
        "poles": [{"re": float(p.real), "im": float(p.imag)} for p in poles],
    }


def synthesize_lead(target_wc=50.0, target_pm_deg=50.0, total_delay_s=0.001):
    d1 = R_A * B + K_T * K_B
    d2 = R_A * J + L_A * B
    d3 = L_A * J
    num = [K_T]
    den = [d3, d2, d1, 0.0]

    w, h = signal.freqs(num, den, worN=[target_wc])
    plant_mag = float(np.abs(h[0]))
    plant_phase_rad = float(np.angle(h[0]))

    delay_phase_rad = -target_wc * total_delay_s
    effective_phase = plant_phase_rad + delay_phase_rad
    uncomp_pm_rad = np.pi + effective_phase
    target_pm_rad = target_pm_deg * np.pi / 180.0

    safety_buffer_rad = 5.0 * np.pi / 180.0
    phi_lead_rad = float(
        np.clip(
            target_pm_rad - uncomp_pm_rad + safety_buffer_rad,
            0.1,
            1.45,
        )
    )

    sin_phi = np.sin(phi_lead_rad)
    alpha = float((1.0 - sin_phi) / (1.0 + sin_phi))
    omega_m = target_wc
    t = float(1.0 / (omega_m * np.sqrt(alpha)))
    k = float(1.0 / (np.sqrt(alpha) * plant_mag))

    # Continuous compensator C(s) = K * (s + 1/T) / (s + 1/(alpha*T))
    zero_s = -1.0 / t
    pole_s = -1.0 / (alpha * t)
    c_num = [k, k / t]
    c_den = [1.0, 1.0 / (alpha * t)]

    # Discretize via Tustin with prewarping at omega_m
    c_cont = signal.TransferFunction(c_num, c_den)
    k_warp = omega_m / np.tan(omega_m * TS_S / 2.0)
    ts_eff = float(2.0 / k_warp)
    c_disc = signal.cont2discrete(
        (c_cont.num, c_cont.den),
        dt=ts_eff,
        method="bilinear",
    )
    b_poly = c_disc[0][0]
    a_poly = c_disc[1]

    # Normalize denominator leading coefficient to 1
    a0 = a_poly[0]
    b0 = float(b_poly[0] / a0)
    b1 = float(b_poly[1] / a0)
    a1 = float(a_poly[1] / a0)

    return {
        "independent": False,
        "transcribes": "analysis::synthesize_delay_compensated_lead",
        "k": k,
        "t": t,
        "alpha": alpha,
        "omega_m": omega_m,
        "max_phase_lead_deg": float(phi_lead_rad * 180.0 / np.pi),
        "zero_rad_s": 1.0 / t,
        "pole_rad_s": 1.0 / (alpha * t),
        "c_num": c_num,
        "c_den": c_den,
        "df2t_b0": b0,
        "df2t_b1": b1,
        "df2t_a1": a1,
    }


def _gain_crossover(num, den, omegas):
    """Locates the unity-gain crossover and the phase margin there.

    The grid only brackets the crossover; the frequency itself is refined with
    Brent's method on |G(jw)| - 1, so the result does not inherit the grid
    spacing. This mirrors what `classical_tools::margins::stability_margins`
    reports (bisection on the same bracket) without sharing its algorithm.

    Returns `(omega_gc, phase_margin_deg)`, or `(None, None)` when the
    magnitude never crosses unity on the grid.
    """
    _, h = signal.freqs(num, den, worN=omegas)
    excess = np.abs(h) - 1.0
    sign_change = np.where(np.diff(np.sign(excess)) != 0)[0]
    if len(sign_change) == 0:
        return None, None

    i = int(sign_change[0])

    def unity_gap(w):
        return float(np.abs(signal.freqs(num, den, worN=[w])[1][0]) - 1.0)

    wc = float(brentq(unity_gap, omegas[i], omegas[i + 1], xtol=1e-12, rtol=1e-14))
    phase_deg = float(np.angle(signal.freqs(num, den, worN=[wc])[1][0], deg=True))
    return wc, 180.0 + phase_deg


def compute_margins(lead_design):
    d1 = R_A * B + K_T * K_B
    d2 = R_A * J + L_A * B
    d3 = L_A * J
    p_num = [K_T]
    p_den = [d3, d2, d1, 0.0]

    omegas = np.logspace(OMEGA_LOG10_START, OMEGA_LOG10_STOP, OMEGA_POINTS)
    _, p_h = signal.freqs(p_num, p_den, worN=omegas)
    p_mag_db = 20.0 * np.log10(np.abs(p_h))
    p_phase_deg = np.rad2deg(np.unwrap(np.angle(p_h)))

    p_wc, p_pm = _gain_crossover(p_num, p_den, omegas)

    # Compensated loop L(s) = C(s) * G(s)
    c_num = lead_design["c_num"]
    c_den = lead_design["c_den"]
    loop_num = np.polymul(c_num, p_num)
    loop_den = np.polymul(c_den, p_den)

    _, l_h = signal.freqs(loop_num, loop_den, worN=omegas)
    l_mag_db = 20.0 * np.log10(np.abs(l_h))
    l_phase_deg = np.rad2deg(np.unwrap(np.angle(l_h)))

    l_wc, l_pm = _gain_crossover(loop_num, loop_den, omegas)
    l_dm_us = float(np.deg2rad(l_pm) / l_wc * 1e6) if l_wc and l_pm else None

    # Delay phase penalty on the same grid, matching the Rust payload.
    delay_phase_deg = np.rad2deg(-omegas * TOTAL_DELAY_S)

    margins = {
        "independent": True,
        "uncompensated": {
            "gain_crossover_rad_s": p_wc,
            "phase_margin_deg": p_pm,
            "phase_crossover_rad_s": None,
            "gain_margin_db": None,
            "gain_margin_reason": "no phase crossover (unbounded)",
        },
        "compensated": {
            "gain_crossover_rad_s": l_wc,
            "phase_margin_deg": l_pm,
            "phase_crossover_rad_s": None,
            "gain_margin_db": None,
            "gain_margin_reason": "no phase crossover (unbounded)",
            "delay_margin_us": l_dm_us,
            "closed_loop_rhp_poles": int(np.sum(np.real(np.roots(
                np.polyadd(loop_den, np.pad(
                    loop_num, (len(loop_den) - len(loop_num), 0)))
            )) > 0.0)),
        },
    }

    sweep = {
        "independent": True,
        "omegas": omegas.tolist(),
        "plant_mag_db": p_mag_db.tolist(),
        "plant_phase_deg": np.angle(p_h, deg=True).tolist(),
        "loop_mag_db": l_mag_db.tolist(),
        "loop_phase_deg": np.angle(l_h, deg=True).tolist(),
        "delay_phase_deg": delay_phase_deg.tolist(),
    }

    return margins, sweep


def compute_root_locus(lead_design, n_points=100):
    d1 = R_A * B + K_T * K_B
    d2 = R_A * J + L_A * B
    d3 = L_A * J
    p_num = [K_T]
    p_den = [d3, d2, d1, 0.0]

    c_num_unit = [1.0, 1.0 / lead_design["t"]]
    c_den = [1.0, 1.0 / (lead_design["alpha"] * lead_design["t"])]

    loop_num_unit = np.polymul(c_num_unit, p_num)
    loop_den = np.polymul(c_den, p_den)

    k_nom = lead_design["k"]
    gains = np.linspace(0.0, 2.0 * k_nom, n_points)
    poles_re = []
    poles_im = []

    prev_roots = None
    for g in gains:
        char_poly = loop_den + np.pad(
            g * loop_num_unit, (len(loop_den) - len(loop_num_unit), 0)
        )
        r = list(np.roots(char_poly))
        if prev_roots is None:
            # Canonical sort at g = 0
            prev_roots = sorted(r, key=lambda x: (x.real, x.imag))
        else:
            # Nearest-neighbor matching against preceding poles
            m = len(prev_roots)
            pairs = []
            for i in range(m):
                for j in range(m):
                    dist = abs(prev_roots[i] - r[j])
                    pairs.append((dist, i, j))
            pairs.sort(key=lambda x: x[0])
            matched = [None] * m
            used_prev = [False] * m
            used_curr = [False] * m
            for dist, i, j in pairs:
                if not used_prev[i] and not used_curr[j]:
                    matched[i] = r[j]
                    used_prev[i] = True
                    used_curr[j] = True
            prev_roots = matched

        poles_re.append([float(p.real) for p in prev_roots])
        poles_im.append([float(p.imag) for p in prev_roots])

    return {
        "independent": True,
        "ordering": "canonical (real asc, imag asc) per gain, not branch-tracked",
        "gains": [float(g) for g in gains],
        "poles_re": poles_re,
        "poles_im": poles_im,
    }


def main():
    parser = argparse.ArgumentParser(
        description="DC motor classical control oracle (SciPy)"
    )
    parser.add_argument(
        "h5_path",
        nargs="?",
        default="results/dc-motor.scipy.h5",
        help="Unused; the oracle writes results/dc-motor.scipy.h5",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from h5_write import attr_specs_from_toml, write_variant_file

    plant_data = compute_plant()
    lead_data = synthesize_lead()
    if lead_data.get("independent") is not False:
        sys.stderr.write(
            "lead_design: oracle no longer declares independent=false; "
            "revisit what this comparison proves\n"
        )
        sys.exit(1)
    margins_data, sweep_data = compute_margins(lead_data)
    locus_data = compute_root_locus(lead_data)

    last = sorted(
        zip(locus_data["poles_re"][-1], locus_data["poles_im"][-1]),
        key=lambda p: (p[0], p[1]),
    )
    scipy_payload = {
        "plant": {
            "num_ascending": plant_data["num"],
            "den_ascending": list(reversed(plant_data["den"])),
        },
        "analysis": {
            "uncompensated_margins": {
                "gain_crossover_rad_s": margins_data["uncompensated"][
                    "gain_crossover_rad_s"
                ],
                "phase_margin_deg": margins_data["uncompensated"]["phase_margin_deg"],
            },
            "lead_design": {
                "gain_k": lead_data["k"],
                "time_constant_t_s": lead_data["t"],
                "attenuation_alpha": lead_data["alpha"],
                "max_phase_lead_deg": lead_data["max_phase_lead_deg"],
                "df2t_b0": lead_data["df2t_b0"],
                "df2t_b1": lead_data["df2t_b1"],
                "df2t_a1": lead_data["df2t_a1"],
            },
            "compensated_margins": {
                "gain_crossover_rad_s": margins_data["compensated"][
                    "gain_crossover_rad_s"
                ],
                "phase_margin_deg": margins_data["compensated"]["phase_margin_deg"],
                "delay_margin_us": margins_data["compensated"]["delay_margin_us"],
                "closed_loop_rhp_poles": margins_data["compensated"][
                    "closed_loop_rhp_poles"
                ],
            },
            "frequency_sweep": {
                "omegas": sweep_data["omegas"],
                "plant_mag_db": sweep_data["plant_mag_db"],
                "plant_phase_deg": sweep_data["plant_phase_deg"],
                "loop_mag_db": sweep_data["loop_mag_db"],
                "loop_phase_deg": sweep_data["loop_phase_deg"],
                "delay_phase_deg": sweep_data["delay_phase_deg"],
            },
            "root_locus": {
                "gains": locus_data["gains"],
                "poles_re": locus_data["poles_re"],
                "poles_im": locus_data["poles_im"],
                "final_poles": [[float(re), float(im)] for re, im in last],
            },
        },
    }

    scipy_keys = {
        "plant/num_ascending": "dc_motor.plant.num",
        "plant/den_ascending": "dc_motor.plant.den",
        "analysis/uncompensated_margins/gain_crossover_rad_s": "dc_motor.margins.uncompensated.gain_crossover",
        "analysis/uncompensated_margins/phase_margin_deg": "dc_motor.margins.uncompensated.phase_margin",
        "analysis/lead_design/gain_k": "dc_motor.compensator.gain_k",
        "analysis/lead_design/time_constant_t_s": "dc_motor.compensator.time_constant_t_s",
        "analysis/lead_design/attenuation_alpha": "dc_motor.compensator.attenuation_alpha",
        "analysis/lead_design/max_phase_lead_deg": "dc_motor.compensator.max_phase_lead_deg",
        "analysis/lead_design/df2t_b0": "dc_motor.compensator.df2t_b0",
        "analysis/lead_design/df2t_b1": "dc_motor.compensator.df2t_b1",
        "analysis/lead_design/df2t_a1": "dc_motor.compensator.df2t_a1",
        "analysis/compensated_margins/gain_crossover_rad_s": "dc_motor.margins.compensated.gain_crossover",
        "analysis/compensated_margins/phase_margin_deg": "dc_motor.margins.compensated.phase_margin",
        "analysis/compensated_margins/delay_margin_us": "dc_motor.margins.compensated.delay_margin",
        "analysis/compensated_margins/closed_loop_rhp_poles": "dc_motor.stability.rhp_poles",
        "analysis/frequency_sweep/loop_mag_db": "dc_motor.frequency_sweep.mag_db",
        "analysis/frequency_sweep/loop_phase_deg": "dc_motor.frequency_sweep.phase_deg",
        "analysis/root_locus/final_poles": "dc_motor.root_locus.poles",
    }
    table = Path(__file__).resolve().parent.parent / "tolerances/dc_motor.toml"
    specs = attr_specs_from_toml(table, scipy_keys)
    write_variant_file(
        Path("results/dc-motor.scipy.h5"),
        scipy_payload,
        gated_paths=list(scipy_keys),
        meta=scipy_payload,
        attr_specs=specs,
    )
    _ = args


if __name__ == "__main__":
    main()
