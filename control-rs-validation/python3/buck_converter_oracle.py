#!/usr/bin/env python3
"""
python3/buck_converter_oracle.py

External Tooling Oracle for the Buck Converter Classical Control Validation:
1. Analytical & Numerical Reference via SciPy (scipy.signal):
   - Continuous plant model, natural frequency, damping ratio, poles via np.roots
   - Open-loop frequency response, gain crossover, phase margin
   - Lead compensator synthesis: K, T, alpha, omega_m, phi_m, zero, pole
   - Compensated open-loop frequency response, crossover, phase margin, delay margin
   - Closed-loop pole stability check and parameterized root locus sweep
   - Tustin bilinear discretization with prewarping at crossover (DF2T & Biquad)
2. SPICE Circuit Simulation via ngspice:
   - Small-signal AC frequency response of the physical LC power stage
   - Large-signal averaged transient step response (V_ref: 5.0 V -> 5.5 V)
   - Large-signal averaged load disturbance rejection (R_load: 1.0 Ohm -> 2.0 Ohm)
   - Cycle-by-cycle switched PWM simulation at 100 kHz measuring switching ripple
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy
from scipy import signal


def locate_ngspice() -> str | None:
    """Finds the ngspice executable path across standard system locations."""
    if "NGSPICE" in os.environ and os.path.isfile(os.environ["NGSPICE"]):
        return os.environ["NGSPICE"]

    candidates = [
        shutil.which("ngspice"),
        "/opt/homebrew/bin/ngspice",
        "/usr/local/bin/ngspice",
        "/usr/bin/ngspice",
    ]
    for c in candidates:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def locate_spice_dir() -> Path:
    """Finds the spice/ directory relative to this script or current working dir."""
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir.parent / "spice",
        Path("control-rs-validation/spice"),
        Path("spice"),
    ]
    for c in candidates:
        if c.is_dir():
            return c.resolve()
    raise FileNotFoundError("Could not locate 'spice/' directory containing buck_averaged.cir")


def compute_scipy_plant_and_margins():
    """Computes reference open-loop plant properties and frequency response using SciPy."""
    vin = 12.0
    l_h = 100e-6
    c_f = 100e-6
    r_l = 1.0
    v_out = 5.0
    d0 = v_out / vin

    wn = 1.0 / np.sqrt(l_h * c_f)
    zeta = (np.sqrt(l_h / c_f)) / (2.0 * r_l)

    num_plant = np.array([vin * wn ** 2], dtype=np.float64)
    den_plant = np.array([1.0, 2.0 * zeta * wn, wn ** 2], dtype=np.float64)

    poles = np.roots(den_plant)
    poles_list = [{"re": float(p.real), "im": float(p.imag)} for p in poles]

    # Frequency sweep matching Rust: 1000 points from 10^2 to 10^6 rad/s
    omegas = np.logspace(2.0, 6.0, 1000, dtype=np.float64)
    _, h_plant = signal.freqs(num_plant, den_plant, worN=omegas)
    mag_db = 20.0 * np.log10(np.abs(h_plant))
    phase_deg = np.angle(h_plant) * 180.0 / np.pi

    # Gain crossover (|H| = 0 dB) via zero-crossing linear interpolation
    idx_cross = np.where(np.diff(np.sign(mag_db)))[0]
    if len(idx_cross) > 0:
        i = idx_cross[0]
        frac = -mag_db[i] / (mag_db[i + 1] - mag_db[i])
        w_gc = float(omegas[i] + frac * (omegas[i + 1] - omegas[i]))
        phase_at_gc = float(phase_deg[i] + frac * (phase_deg[i + 1] - phase_deg[i]))
        phase_margin = float(180.0 + phase_at_gc)
    else:
        w_gc, phase_margin = None, None

    return {
        "v_in": vin,
        "inductance_h": l_h,
        "capacitance_f": c_f,
        "load_resistance_ohm": r_l,
        "v_out_v": v_out,
        "operating_duty_cycle": d0,
        "natural_frequency_rad_s": wn,
        "damping_ratio": zeta,
        "poles": poles_list,
        "num_plant": num_plant.tolist(),
        "den_plant": den_plant.tolist(),
        "omegas": omegas.tolist(),
        "uncomp_mag_db": mag_db.tolist(),
        "uncomp_phase_deg": phase_deg.tolist(),
        "gain_crossover_rad_s": w_gc,
        "phase_margin_deg": phase_margin,
        "phase_crossover_rad_s": None,
        "gain_margin_db": None,
    }


def synthesize_scipy_lead_compensator(plant_data: dict, target_wc=30000.0, target_pm_deg=50.0):
    """Synthesizes the rational lead compensator C(s) matching classical analytical design."""
    num_plant = np.array(plant_data["num_plant"])
    den_plant = np.array(plant_data["den_plant"])

    # 1. Frequency response at target crossover:
    _, h_at_target = signal.freqs(num_plant, den_plant, worN=[target_wc])
    g_resp = h_at_target[0]
    plant_mag = np.abs(g_resp)
    plant_phase_rad = np.angle(g_resp)

    # 2. Uncompensated phase margin at target crossover:
    uncomp_pm_rad = np.pi + plant_phase_rad
    target_pm_rad = target_pm_deg * np.pi / 180.0

    # Required phase lead with 5° margin buffer for crossover shift:
    phi_lead_rad = float(
        np.clip(target_pm_rad - uncomp_pm_rad + (5.0 * np.pi / 180.0), 0.1, 1.4)
    )

    # 3. Attenuation factor alpha
    sin_phi = np.sin(phi_lead_rad)
    alpha = float((1.0 - sin_phi) / (1.0 + sin_phi))

    # 4. Center frequency placed at target crossover
    omega_m = float(target_wc)
    t = float(1.0 / (omega_m * np.sqrt(alpha)))

    # 5. Compensator gain K for unity loop gain at omega_m
    k = float(1.0 / (np.sqrt(alpha) * plant_mag))

    zero_rad_s = 1.0 / t
    pole_rad_s = 1.0 / (alpha * t)

    # C(s) = K * (s + 1/T) / (s + 1/(alpha*T)) = (K s + K/T) / (s + 1/(alpha*T))
    # In SciPy: descending powers of s
    num_c_s = np.array([k, k * zero_rad_s], dtype=np.float64)
    den_c_s = np.array([1.0, pole_rad_s], dtype=np.float64)

    # Compensated Open-Loop L(s) = C(s) * G_vd(s)
    num_loop_s = np.convolve(num_c_s, num_plant)
    den_loop_s = np.convolve(den_c_s, den_plant)

    omegas = np.array(plant_data["omegas"])
    _, h_loop = signal.freqs(num_loop_s, den_loop_s, worN=omegas)
    comp_mag_db = 20.0 * np.log10(np.abs(h_loop))
    comp_phase_deg = np.angle(h_loop) * 180.0 / np.pi

    # Compensated gain crossover and phase margin
    idx_comp = np.where(np.diff(np.sign(comp_mag_db)))[0]
    if len(idx_comp) > 0:
        i = idx_comp[0]
        frac = -comp_mag_db[i] / (comp_mag_db[i + 1] - comp_mag_db[i])
        comp_wc = float(omegas[i] + frac * (omegas[i + 1] - omegas[i]))
        phase_at_comp_gc = float(
            comp_phase_deg[i] + frac * (comp_phase_deg[i + 1] - comp_phase_deg[i])
        )
        comp_pm_deg = float(180.0 + phase_at_comp_gc)
    else:
        comp_wc, comp_pm_deg = None, None

    delay_margin_us = (
        float((comp_pm_deg * np.pi / 180.0) / comp_wc * 1e6)
        if (comp_wc and comp_pm_deg)
        else None
    )

    # Closed-loop roots: den_loop + num_loop = 0
    den_cl = den_loop_s.copy()
    den_cl[-len(num_loop_s):] += num_loop_s
    cl_roots = np.roots(den_cl)
    cl_rhp_count = int(np.sum(cl_roots.real > 0.0))

    # Tustin Bilinear Discretization with Prewarping at omega_c = target_wc
    ts = 10.0e-6  # 10 us sampling interval (100 kHz)
    k_prewarp = target_wc / np.tan(target_wc * ts / 2.0)
    ts_eff = 2.0 / k_prewarp

    sys_d = signal.cont2discrete((num_c_s, den_c_s), dt=ts_eff, method="bilinear")
    num_z = sys_d[0].squeeze()
    den_z = sys_d[1]

    # Normalize to monic denominator: (b0 + b1 z^-1) / (1 + a1 z^-1)
    b0 = float(num_z[0] / den_z[0])
    b1 = float(num_z[1] / den_z[0])
    a1 = float(den_z[1] / den_z[0])

    return {
        "k": k,
        "t_s": t,
        "alpha": alpha,
        "omega_m_rad_s": omega_m,
        "max_phase_lead_deg": float(phi_lead_rad * 180.0 / np.pi),
        "zero_rad_s": zero_rad_s,
        "pole_rad_s": pole_rad_s,
        "comp_gain_crossover_rad_s": comp_wc,
        "comp_phase_margin_deg": comp_pm_deg,
        "delay_margin_us": delay_margin_us,
        "closed_loop_rhp_roots": cl_rhp_count,
        "comp_mag_db": comp_mag_db.tolist(),
        "comp_phase_deg": comp_phase_deg.tolist(),
        "df2t_b0": b0,
        "df2t_b1": b1,
        "df2t_a1": a1,
        "biquad": {"b0": b0, "b1": b1, "b2": 0.0, "a1": a1, "a2": 0.0},
        "num_loop_s": num_loop_s.tolist(),
        "den_loop_s": den_loop_s.tolist(),
        "num_c_s": num_c_s.tolist(),
        "den_c_s": den_c_s.tolist(),
    }


def compute_scipy_root_locus(plant_data: dict, comp_data: dict, num_points=100):
    """Sweeps loop gain 0 <= K <= 2 * K_nom and solves closed-loop roots via np.roots."""
    num_plant = np.array(plant_data["num_plant"])
    den_plant = np.array(plant_data["den_plant"])
    t = comp_data["t_s"]
    alpha = comp_data["alpha"]
    k_nom = comp_data["k"]

    # Unscaled loop L0(s) with K = 1.0:
    num_c0 = np.array([1.0, 1.0 / t])
    den_c0 = np.array([1.0, 1.0 / (alpha * t)])
    num_l0 = np.convolve(num_c0, num_plant)
    den_l0 = np.convolve(den_c0, den_plant)

    max_gain = 2.0 * k_nom
    gains = np.linspace(0.0, max_gain, num_points, dtype=np.float64)

    poles_re = []
    poles_im = []

    prev_roots = None
    for g in gains:
        p_cl = den_l0.copy()
        p_cl[-len(num_l0):] += g * num_l0
        roots = list(np.roots(p_cl))
        if prev_roots is None:
            # Canonical sort at g = 0
            prev_roots = sorted(roots, key=lambda r: (r.real, r.imag))
        else:
            # Nearest-neighbor matching against preceding poles
            m = len(prev_roots)
            pairs = []
            for i in range(m):
                for j in range(m):
                    dist = abs(prev_roots[i] - roots[j])
                    pairs.append((dist, i, j))
            pairs.sort(key=lambda x: x[0])
            matched = [None] * m
            used_prev = [False] * m
            used_curr = [False] * m
            for dist, i, j in pairs:
                if not used_prev[i] and not used_curr[j]:
                    matched[i] = roots[j]
                    used_prev[i] = True
                    used_curr[j] = True
            prev_roots = matched

        poles_re.append([float(r.real) for r in prev_roots])
        poles_im.append([float(r.imag) for r in prev_roots])

    poles_re_sorted = []
    poles_im_sorted = []
    for re_row, im_row in zip(poles_re, poles_im):
        order = sorted(range(len(re_row)), key=lambda i: (re_row[i], im_row[i]))
        poles_re_sorted.append([re_row[i] for i in order])
        poles_im_sorted.append([im_row[i] for i in order])

    return {
        "gains": gains.tolist(),
        "poles_re": poles_re,
        "poles_im": poles_im,
        "poles_re_sorted": poles_re_sorted,
        "poles_im_sorted": poles_im_sorted,
    }


def extract_step_metrics(time_arr, v_arr, t_step=0.0005, v_initial=5.0, v_target=5.5):
    """Extracts rise time, peak overshoot, settling time, steady-state error, and peak voltage."""
    actual_step = v_arr[-1] - v_initial
    final_v = float(v_arr[-1])
    sse = float(abs(final_v - v_target))
    peak_val = float(np.max(v_arr[time_arr >= t_step]))

    # Rise time 10% to 90%
    v_10 = v_initial + 0.1 * actual_step
    v_90 = v_initial + 0.9 * actual_step

    t_10, t_90 = None, None
    for t, v in zip(time_arr, v_arr):
        if t < t_step:
            continue
        if t_10 is None and v >= v_10:
            t_10 = t
        if t_90 is None and v >= v_90:
            t_90 = t
            break

    rise_time_us = float((t_90 - t_10) * 1e6) if (t_10 and t_90 and t_90 >= t_10) else 0.0

    # Peak overshoot percentage
    overshoot_pct = (
        float(((peak_val - final_v) / actual_step) * 100.0)
        if (actual_step > 0 and peak_val > final_v)
        else 0.0
    )

    # 2% settling time
    threshold_2pct = 0.02 * abs(actual_step)
    settling_time_us = 0.0
    for t, v in zip(reversed(time_arr), reversed(v_arr)):
        if t < t_step:
            break
        if abs(v - final_v) > threshold_2pct:
            settling_time_us = float(max(0.0, (t - t_step) * 1e6))
            break

    return {
        "rise_time_us": rise_time_us,
        "peak_overshoot_pct": overshoot_pct,
        "settling_time_us": settling_time_us,
        "steady_state_error_v": sse,
        "peak_voltage_v": peak_val,
        "final_voltage_v": final_v,
    }


def run_ngspice_simulations(ngspice_bin: str, spice_dir: Path):
    """Executes buck_averaged.cir and buck_switched.cir using ngspice in batch mode."""
    results_dir = spice_dir.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    averaged_netlist = spice_dir / "buck_averaged.cir"
    switched_netlist = spice_dir / "buck_switched.cir"

    # Run averaged simulation
    cmd_avg = [ngspice_bin, "-b", str(averaged_netlist)]
    proc_avg = subprocess.run(
        cmd_avg, cwd=str(spice_dir.parent), capture_output=True, text=True
    )
    if proc_avg.returncode != 0:
        raise RuntimeError(f"ngspice failed on {averaged_netlist}:\n{proc_avg.stderr}")

    # Run switched PWM simulation
    cmd_sw = [ngspice_bin, "-b", str(switched_netlist)]
    proc_sw = subprocess.run(
        cmd_sw, cwd=str(spice_dir.parent), capture_output=True, text=True
    )
    if proc_sw.returncode != 0:
        raise RuntimeError(f"ngspice failed on {switched_netlist}:\n{proc_sw.stderr}")

    # Load and parse output files
    ac_file = results_dir / "ngspice_ac_plant.txt"
    step_file = results_dir / "ngspice_step.txt"
    load_file = results_dir / "ngspice_load.txt"
    switched_file = results_dir / "ngspice_switched_step.txt"

    ac_data = np.loadtxt(ac_file)
    f_hz = ac_data[:, 0]
    ac_mag_db = ac_data[:, 1]
    ac_phase_rad = ac_data[:, 3]
    ac_phase_deg = ac_phase_rad * 180.0 / np.pi

    step_data = np.loadtxt(step_file)
    t_step = step_data[:, 0]
    vo_step = step_data[:, 1]
    d_step = step_data[:, 3]
    il_step = step_data[:, 5]

    load_data = np.loadtxt(load_file)
    t_load = load_data[:, 0]
    vo_load = load_data[:, 1]
    d_load = load_data[:, 3]
    il_load = load_data[:, 5]

    sw_data = np.loadtxt(switched_file)
    t_sw = sw_data[:, 0]
    vo_sw = sw_data[:, 1]

    # Decimate raw switched text file on disk to save space (< 300 KB vs 6.1 MB)
    if switched_file.is_file():
        np.savetxt(switched_file, sw_data[::20], fmt="%.6e")

    # Calculate metrics
    step_metrics = extract_step_metrics(t_step, vo_step, t_step=0.0005, v_initial=5.0, v_target=5.5)

    idx_1m = np.where(t_load >= 0.001)[0][0]
    load_metrics = {
        "nominal_v": float(vo_load[idx_1m]),
        "peak_v": float(np.max(vo_load[idx_1m:])),
        "restored_v": float(vo_load[-1]),
    }

    switched_ripple_mv = float((np.max(vo_sw[-200:]) - np.min(vo_sw[-200:])) * 1e3)
    switched_metrics = {
        "peak_voltage_v": float(np.max(vo_sw)),
        "final_average_v": float(np.mean(vo_sw[-200:])),
        "switching_ripple_mv_pk_pk": switched_ripple_mv,
    }

    # Decimate long transient arrays for efficient JSON transport (every 5th sample)
    stride = 5
    return {
        "ac_sweep": {
            "freq_hz": f_hz.tolist(),
            "mag_db": ac_mag_db.tolist(),
            "phase_deg": ac_phase_deg.tolist(),
        },
        "averaged_step": {
            "time_s": t_step[::stride].tolist(),
            "v_out_v": vo_step[::stride].tolist(),
            "duty_cycle": d_step[::stride].tolist(),
            "i_l_a": il_step[::stride].tolist(),
            "metrics": step_metrics,
        },
        "averaged_load": {
            "time_s": t_load[::stride].tolist(),
            "v_out_v": vo_load[::stride].tolist(),
            "duty_cycle": d_load[::stride].tolist(),
            "i_l_a": il_load[::stride].tolist(),
            "metrics": load_metrics,
        },
        "switched_step": {
            "time_s": t_sw[::stride * 4].tolist(),
            "v_out_v": vo_sw[::stride * 4].tolist(),
            "metrics": switched_metrics,
        },
    }


def run_oracle() -> dict:
    """Orchestrates both SciPy classical control calculations and ngspice circuit simulation."""
    # 1. Classical control analytical calculations via SciPy
    plant_data = compute_scipy_plant_and_margins()
    comp_data = synthesize_scipy_lead_compensator(plant_data)
    locus_data = compute_scipy_root_locus(plant_data, comp_data)

    # 2. Circuit simulation via ngspice
    ngspice_bin = locate_ngspice()
    spice_sim_data = None
    if ngspice_bin:
        try:
            spice_dir = locate_spice_dir()
            spice_sim_data = run_ngspice_simulations(ngspice_bin, spice_dir)
        except Exception as e:
            sys.stderr.write(f"Warning: ngspice execution failed: {e}\n")
    else:
        sys.stderr.write("Warning: ngspice binary not found on system PATH.\n")

    return {
        "metadata": {
            "domain": "buck_converter_classical_control",
            "scipy_version": scipy.__version__,
            "ngspice_binary": ngspice_bin,
            "has_ngspice": spice_sim_data is not None,
        },
        "plant": {
            "v_in": plant_data["v_in"],
            "inductance_h": plant_data["inductance_h"],
            "capacitance_f": plant_data["capacitance_f"],
            "load_resistance_ohm": plant_data["load_resistance_ohm"],
            "v_out_v": plant_data["v_out_v"],
            "operating_duty_cycle": plant_data["operating_duty_cycle"],
            "natural_frequency_rad_s": plant_data["natural_frequency_rad_s"],
            "damping_ratio": plant_data["damping_ratio"],
            "poles": plant_data["poles"],
        },
        "margins": {
            "uncompensated": {
                "gain_crossover_rad_s": plant_data["gain_crossover_rad_s"],
                "phase_margin_deg": plant_data["phase_margin_deg"],
                "phase_crossover_rad_s": plant_data["phase_crossover_rad_s"],
                "gain_margin_db": plant_data["gain_margin_db"],
            },
            "compensated": {
                "gain_crossover_rad_s": comp_data["comp_gain_crossover_rad_s"],
                "phase_margin_deg": comp_data["comp_phase_margin_deg"],
                "phase_crossover_rad_s": None,
                "gain_margin_db": None,
                "delay_margin_us": comp_data["delay_margin_us"],
            },
        },
        "compensator": {
            "gain_k": comp_data["k"],
            "time_constant_t_s": comp_data["t_s"],
            "attenuation_alpha": comp_data["alpha"],
            "zero_rad_s": comp_data["zero_rad_s"],
            "pole_rad_s": comp_data["pole_rad_s"],
            "center_freq_rad_s": comp_data["omega_m_rad_s"],
            "max_phase_lead_deg": comp_data["max_phase_lead_deg"],
            "closed_loop_rhp_roots": comp_data["closed_loop_rhp_roots"],
            "df2t_b0": comp_data["df2t_b0"],
            "df2t_b1": comp_data["df2t_b1"],
            "df2t_a1": comp_data["df2t_a1"],
            "biquad": comp_data["biquad"],
        },
        "frequency_sweep": {
            "omegas": plant_data["omegas"],
            "uncomp_mag_db": plant_data["uncomp_mag_db"],
            "uncomp_phase_deg": plant_data["uncomp_phase_deg"],
            "comp_mag_db": comp_data["comp_mag_db"],
            "comp_phase_deg": comp_data["comp_phase_deg"],
        },
        "root_locus": locus_data,
        "ngspice": spice_sim_data,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Buck Converter Classical Control Oracle (SciPy & ngspice)"
    )
    parser.add_argument(
        "h5_path",
        nargs="?",
        default="results/buck-converter.scipy.h5",
        help="Unused; variants are written as results/buck-converter.<variant>.h5",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from h5_write import attr_specs_from_toml, write_variant_file

    results = run_oracle()
    plant = {
        k: v
        for k, v in results["plant"].items()
        if k != "poles" and not isinstance(v, (str, bool, type(None)))
    }
    scipy_payload = {
        "plant": plant,
        "margins": {
            "uncompensated": results["margins"]["uncompensated"],
            "compensated": results["margins"]["compensated"],
        },
        "compensator": results["compensator"],
        "frequency_sweep": results["frequency_sweep"],
        "root_locus": results["root_locus"],
    }

    scipy_keys = {
        "plant/natural_frequency_rad_s": "buck.plant.natural_frequency",
        "plant/damping_ratio": "buck.plant.damping_ratio",
        "margins/uncompensated/gain_crossover_rad_s": "buck.margins.uncompensated.gain_crossover",
        "margins/uncompensated/phase_margin_deg": "buck.margins.uncompensated.phase_margin",
        "compensator/gain_k": "buck.compensator.gain_k",
        "compensator/time_constant_t_s": "buck.compensator.time_constant_t_s",
        "compensator/attenuation_alpha": "buck.compensator.attenuation_alpha",
        "compensator/zero_rad_s": "buck.compensator.zero_rad_s",
        "compensator/pole_rad_s": "buck.compensator.pole_rad_s",
        "compensator/max_phase_lead_deg": "buck.compensator.max_phase_lead_deg",
        "compensator/df2t_b0": "buck.compensator.df2t_b0",
        "compensator/df2t_b1": "buck.compensator.df2t_b1",
        "compensator/df2t_a1": "buck.compensator.df2t_a1",
        "margins/compensated/gain_crossover_rad_s": "buck.margins.compensated.gain_crossover",
        "margins/compensated/phase_margin_deg": "buck.margins.compensated.phase_margin",
        "margins/compensated/delay_margin_us": "buck.margins.compensated.delay_margin",
        "frequency_sweep/uncomp_mag_db": "buck.frequency_sweep.uncomp_mag_db",
        "frequency_sweep/comp_mag_db": "buck.frequency_sweep.comp_mag_db",
        "frequency_sweep/uncomp_phase_deg": "buck.frequency_sweep.uncomp_phase_deg",
        "frequency_sweep/comp_phase_deg": "buck.frequency_sweep.comp_phase_deg",
        "root_locus/poles_re_sorted": "buck.root_locus.poles_re",
        "root_locus/poles_im_sorted": "buck.root_locus.poles_im",
    }

    table = Path(__file__).resolve().parent.parent / "tolerances/buck_converter.toml"
    specs = attr_specs_from_toml(table, scipy_keys)
    meta = dict(scipy_payload)
    spice = results.get("ngspice")
    if spice:
        ripple = spice["switched_step"]["metrics"]["switching_ripple_mv_pk_pk"]
        meta["ngspice"] = {
            "circuit": {
                "ac_plant_dc_gain": spice["ac_sweep"]["mag_db"][0],
                "step_final_voltage": spice["averaged_step"]["metrics"]["final_voltage_v"],
                "load_restored_voltage": spice["averaged_load"]["metrics"]["restored_v"],
                "switched_ripple": ripple,
            },
            "simulation": {
                "linear_step": {
                    "metrics": {
                        "steady_state_error_v": spice["averaged_step"]["metrics"][
                            "steady_state_error_v"
                        ],
                    }
                }
            },
            "ac_sweep": spice["ac_sweep"],
            "averaged_step": spice["averaged_step"],
            "averaged_load": spice["averaged_load"],
            "switched_step": spice["switched_step"],
        }

    write_variant_file(
        Path("results/buck-converter.scipy.h5"),
        scipy_payload,
        gated_paths=list(scipy_keys),
        meta=meta,
        attr_specs=specs,
    )
    _ = args


if __name__ == "__main__":
    main()
