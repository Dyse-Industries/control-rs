//! Transfer function numerical model control-rs-verification suite.
//!
//! Evaluates continuous and discrete frequency response kernels:
//! 1. Resonant notch filter frequency response (Bode magnitude & phase)
//! 2. Tustin bilinear transform frequency warping
//! 3. Zero-Order Hold (ZOH) discretization consistency
//! 4. Nyquist stability contour and gain/phase margins
//! 5. Controllable canonical form realization

#![allow(
    missing_docs,
    clippy::arithmetic_side_effects,
    clippy::cast_precision_loss,
    clippy::indexing_slicing,
    clippy::suboptimal_flops,
    clippy::too_many_lines,
    clippy::unwrap_used
)]

use std::path::Path;

use control_rs::transfer_function::ArrayTransferFunction;

use crate::h5_writer::H5Writer;

type Tf<const N: usize, const D: usize> = ArrayTransferFunction<f64, N, D>;

fn synthesize_resonant_notch_system(
    fn_hz: f64,
    zeta_z: f64,
    zeta_p: f64,
    fc_hz: f64,
) -> Tf<3, 4> {
    let wn = 2.0 * std::f64::consts::PI * fn_hz;
    let wc = 2.0 * std::f64::consts::PI * fc_hz;

    let notch = Tf::<3, 3>::continuous(
        [wn * wn, 2.0 * zeta_z * wn, 1.0],
        [wn * wn, 2.0 * zeta_p * wn, 1.0],
    );
    let lowpass = Tf::<1, 2>::continuous([wc], [wc, 1.0]);

    notch.series::<1, 2, 3, 4>(&lowpass)
}

fn compute_bode_responses() -> (
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
) {
    let fn_hz = 25.0_f64;
    let zeta_z = 0.01_f64;
    let zeta_p = 0.25_f64;
    let fc_hz = 40.0_f64;
    let tf_cont =
        synthesize_resonant_notch_system(fn_hz, zeta_z, zeta_p, fc_hz);

    let dt = 0.005; // 200 Hz
    let tf_tustin = tf_cont.to_discrete_tustin(dt, None);
    let tf_zoh = tf_cont
        .to_discrete_zoh::<3>(dt)
        .expect("ZOH conversion failed");

    let num_freqs = 200;
    let freqs_hz: Vec<f64> = (0..num_freqs)
        .map(|i| 0.1 + 99.4 * (i as f64) / ((num_freqs - 1) as f64))
        .collect();

    let mut cont_mag_db = Vec::with_capacity(num_freqs);
    let mut cont_phase_deg = Vec::with_capacity(num_freqs);
    let mut tustin_mag_db = Vec::with_capacity(num_freqs);
    let mut tustin_phase_deg = Vec::with_capacity(num_freqs);
    let mut zoh_mag_db = Vec::with_capacity(num_freqs);
    let mut zoh_phase_deg = Vec::with_capacity(num_freqs);

    for &f_hz in &freqs_hz {
        let w = 2.0 * std::f64::consts::PI * f_hz;

        let (c_mag, c_phase) = tf_cont.bode_point(w);
        cont_mag_db.push(20.0 * c_mag.log10());
        cont_phase_deg.push(c_phase * 180.0 / std::f64::consts::PI);

        let (t_mag, t_phase) = tf_tustin.bode_point(w);
        tustin_mag_db.push(20.0 * t_mag.log10());
        tustin_phase_deg.push(t_phase * 180.0 / std::f64::consts::PI);

        let (z_mag, z_phase) = tf_zoh.bode_point(w);
        zoh_mag_db.push(20.0 * z_mag.log10());
        zoh_phase_deg.push(z_phase * 180.0 / std::f64::consts::PI);
    }

    (
        freqs_hz,
        cont_mag_db,
        cont_phase_deg,
        tustin_mag_db,
        tustin_phase_deg,
        zoh_mag_db,
        zoh_phase_deg,
    )
}

fn compute_nyquist() -> (Vec<f64>, Vec<f64>, f64, f64) {
    let tf_open = Tf::<2, 4>::continuous([100.0, 50.0], [0.0, 25.0, 2.0, 1.0]);
    let n = 200;
    let freqs: Vec<f64> = (0..n)
        .map(|i| 10.0_f64.powf(-2.0 + 5.0 * (i as f64) / ((n - 1) as f64)))
        .collect();

    let mut h_re = Vec::with_capacity(n);
    let mut h_im = Vec::with_capacity(n);
    let mut gain_crossover_w = 0.0;
    let mut phase_crossover_w = 0.0;
    let mut min_mag_diff = f64::MAX;
    let mut min_im_diff = f64::MAX;

    for &w in &freqs {
        let resp = tf_open.eval_frequency(w);
        h_re.push(resp.re);
        h_im.push(resp.im);

        let mag = (resp.re * resp.re + resp.im * resp.im).sqrt();
        if (mag - 1.0).abs() < min_mag_diff {
            min_mag_diff = (mag - 1.0).abs();
            gain_crossover_w = w;
        }

        if resp.im.abs() < min_im_diff && resp.re < 0.0 {
            min_im_diff = resp.im.abs();
            phase_crossover_w = w;
        }
    }

    let gc_resp = tf_open.eval_frequency(gain_crossover_w);
    let phase_margin_rad = std::f64::consts::PI + gc_resp.im.atan2(gc_resp.re);
    let phase_margin_deg = phase_margin_rad * 180.0 / std::f64::consts::PI;

    let pc_resp = tf_open.eval_frequency(phase_crossover_w);
    let pc_mag = (pc_resp.re * pc_resp.re + pc_resp.im * pc_resp.im).sqrt();
    let gain_margin_db = -20.0 * pc_mag.log10();

    (h_re, h_im, phase_margin_deg, gain_margin_db)
}

/// Controllable canonical form of H(s) = (2s + 3) / (s^2 + 5s + 4), row-major `(A, B, C)`.
fn compute_ccf_realization() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let tf = Tf::<2, 3>::continuous([3.0, 2.0], [4.0, 5.0, 1.0]);
    let ss = tf
        .to_controllable_canonical_form::<2>()
        .expect("CCF realization failed");
    let (a, b, c) = (ss.a(), ss.b(), ss.c());
    let mut a_flat = Vec::with_capacity(4);
    for i in 0..2 {
        for j in 0..2 {
            a_flat.push(*a.get(i, j).unwrap());
        }
    }
    let b_flat = (0..2).map(|i| *b.get(i, 0).unwrap()).collect();
    let c_flat = (0..2).map(|j| *c.get(0, j).unwrap()).collect();
    (a_flat, b_flat, c_flat)
}

/// Executes the transfer function control-rs-verification kernel and writes `target/verification/transfer_function.rust.h5`.
pub fn emit_container(output_path: &Path) -> Result<(), String> {
    let mut writer = H5Writer::new();

    let (
        freqs,
        cont_mag,
        cont_phase,
        tustin_mag,
        tustin_phase,
        zoh_mag,
        zoh_phase,
    ) = compute_bode_responses();

    writer.add_dataset("bode/freqs_hz", &freqs);
    writer.set_tolerance("bode/freqs_hz", "abs", 1e-6);

    writer.add_dataset("bode/cont_mag_db", &cont_mag);
    writer.set_tolerance("bode/cont_mag_db", "abs", 0.1);

    writer.add_dataset("bode/cont_phase_deg", &cont_phase);
    writer.set_tolerance("bode/cont_phase_deg", "abs", 0.5);

    writer.add_dataset("bode/tustin_mag_db", &tustin_mag);
    writer.set_tolerance("bode/tustin_mag_db", "abs", 0.1);

    writer.add_dataset("bode/tustin_phase_deg", &tustin_phase);
    writer.set_tolerance("bode/tustin_phase_deg", "abs", 0.5);

    writer.add_dataset("bode/zoh_mag_db", &zoh_mag);
    writer.set_tolerance("bode/zoh_mag_db", "abs", 0.2);

    writer.add_dataset("bode/zoh_phase_deg", &zoh_phase);
    writer.set_tolerance("bode/zoh_phase_deg", "abs", 1.0);

    let (h_re, h_im, pm_deg, gm_db) = compute_nyquist();
    writer.add_dataset("nyquist/h_re", &h_re);
    writer.set_tolerance("nyquist/h_re", "abs", 0.05);

    writer.add_dataset("nyquist/h_im", &h_im);
    writer.set_tolerance("nyquist/h_im", "abs", 0.05);

    writer.add_dataset("nyquist/phase_margin_deg", &[pm_deg]);
    writer.set_tolerance("nyquist/phase_margin_deg", "abs", 1.0);

    writer.add_dataset("nyquist/gain_margin_db", &[gm_db]);
    writer.set_tolerance("nyquist/gain_margin_db", "abs", 1.0);

    let (ccf_a, ccf_b, ccf_c) = compute_ccf_realization();
    for (name, data) in [
        ("realization/ccf_a", &ccf_a),
        ("realization/ccf_b", &ccf_b),
        ("realization/ccf_c", &ccf_c),
    ] {
        writer.add_dataset(name, data);
        writer.set_tolerance(name, "abs", 1e-12);
    }

    writer.write_to_file(output_path)
}
