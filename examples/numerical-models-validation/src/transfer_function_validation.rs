//! src/transfer_function_validation.rs
//!
//! Standalone validation runner for transfer function numerical models.
//! Executes four core benchmark simulations:
//! 1 & 2. Discretization Method Error (Flexible Structure Modal System with Resonant Notch Filter via synthesize_resonant_notch_system, Tustin vs ZOH Bode magnitude & phase near Nyquist)
//! 3. Nyquist Stability Criterion & Margins (Polar curve, (-1, 0j) point, gain & phase margins)
//! 4. Filter Topology Stability (6th-order Butterworth f32 Direct Form vs Biquad SOS)

use serde_json::{Value, json};
use std::fs;
use std::process::Command;
use std::time::Instant;

use control_rs::math::complex_num::Complex;
use control_rs::math::num_types::Const;
use control_rs::math::storage::ArrayStorage;
use control_rs::transfer_function::TransferFunction;

type StoreNum<const N: usize> = ArrayStorage<f64, N, 1>;
type StoreDen<const D: usize> = ArrayStorage<f64, D, 1>;
type Tf<const N: usize, const D: usize> =
    TransferFunction<f64, Const<N>, Const<D>, StoreNum<N>, StoreDen<D>>;

pub type ValidationResult = Result<(), Vec<String>>;
type ComplexRoots = Vec<Complex<f64>>;
type ValueArray<'a> = Option<&'a Vec<Value>>;

/// Script-level constructor for Proposal 1: Flexible Structure Modal System with Resonant Notch Filter.
/// G(s) = (s^2 + 2*zeta_z*wn*s + wn^2) / (s^2 + 2*zeta_p*wn*s + wn^2) * (wc / (s + wc))
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

fn logspace(start_log10: f64, stop_log10: f64, n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![10.0_f64.powf(stop_log10)];
    }
    let den = (n - 1) as f64;
    (0..n)
        .map(|i| {
            10.0_f64.powf(
                start_log10 + (stop_log10 - start_log10) * (i as f64) / den,
            )
        })
        .collect()
}

// -----------------------------------------------------------------------------
// 1 & 2. Discretization Method Error (Bode Magnitude & Phase Warping)
// -----------------------------------------------------------------------------
fn benchmark_discretization_error() -> Value {
    // Proposal 1: Flexible Structure Modal System with Resonant Notch Filter
    let fn_hz = 25.0_f64;
    let zeta_z = 0.01_f64;
    let zeta_p = 0.25_f64;
    let fc_hz = 40.0_f64;
    let tf_cont =
        synthesize_resonant_notch_system(fn_hz, zeta_z, zeta_p, fc_hz);

    let dt = 0.005; // Fs = 200 Hz, Nyquist = 100 Hz
    let tf_tustin = tf_cont.to_discrete_tustin(dt, None);
    let tf_zoh = tf_cont
        .to_discrete_zoh::<3>(dt)
        .expect("ZOH conversion failed");

    let num_freqs = 1000;
    let freqs_hz: Vec<f64> = (0..num_freqs)
        .map(|i| 0.1 + 99.4 * (i as f64) / ((num_freqs - 1) as f64))
        .collect();

    let mut cont_mag_db = Vec::with_capacity(num_freqs);
    let mut cont_phase_deg = Vec::with_capacity(num_freqs);
    let mut tustin_mag_db = Vec::with_capacity(num_freqs);
    let mut tustin_phase_deg = Vec::with_capacity(num_freqs);
    let mut zoh_mag_db = Vec::with_capacity(num_freqs);
    let mut zoh_phase_deg = Vec::with_capacity(num_freqs);

    let t0 = Instant::now();
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
    let bode_time_ns = t0.elapsed().as_nanos() as f64;

    json!({
        "freqs_hz": freqs_hz,
        "cont_mag_db": cont_mag_db,
        "cont_phase_deg": cont_phase_deg,
        "tustin_mag_db": tustin_mag_db,
        "tustin_phase_deg": tustin_phase_deg,
        "zoh_mag_db": zoh_mag_db,
        "zoh_phase_deg": zoh_phase_deg,
        "bode_time_ns": bode_time_ns,
    })
}

// -----------------------------------------------------------------------------
// 3. Nyquist Stability Criterion & Stability Margins
// -----------------------------------------------------------------------------
fn benchmark_nyquist_criterion() -> Value {
    // Open-loop transfer function H(s) = 50*(s + 2) / (s * (s^2 + 2s + 25))
    // H(s) = (100 + 50s) / (0 + 25s + 2s^2 + s^3)
    // Ascending: num = [100.0, 50.0], den = [0.0, 25.0, 2.0, 1.0]
    let tf_open = Tf::<2, 4>::continuous([100.0, 50.0], [0.0, 25.0, 2.0, 1.0]);

    let freqs = logspace(-2.0, 3.0, 250);
    let mut h_re = Vec::with_capacity(freqs.len());
    let mut h_im = Vec::with_capacity(freqs.len());

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

    json!({
        "freqs": freqs,
        "h_re": h_re,
        "h_im": h_im,
        "critical_point": [-1.0, 0.0],
        "gain_crossover_w": gain_crossover_w,
        "phase_crossover_w": phase_crossover_w,
        "phase_margin_deg": phase_margin_deg,
        "gain_margin_db": gain_margin_db,
    })
}

// -----------------------------------------------------------------------------
// 4. Filter Topology Stability (Catastrophic Cancellation in f32)
// -----------------------------------------------------------------------------
fn benchmark_topology_stability() -> Value {
    // 6th-order continuous Butterworth lowpass filter fc = 35 Hz
    let cutoff_hz = 35.0_f64;
    let wc = 2.0 * std::f64::consts::PI * cutoff_hz;
    let dt = 0.01_f64;

    // Continuous s-domain poles for 6th-order Butterworth
    let mut s_poles = Vec::with_capacity(6);
    for k in 0..6 {
        let angle = std::f64::consts::PI * (2.0 * (k as f64) + 7.0) / 12.0;
        s_poles.push(Complex::new(wc * angle.cos(), wc * angle.sin()));
    }

    // Tustin discrete mapping: z = (1 + s*dt/2) / (1 - s*dt/2)
    let mut gt_z_poles = Vec::with_capacity(6);
    for &s in &s_poles {
        let num = Complex::new(1.0, 0.0) + s * Complex::new(dt / 2.0, 0.0);
        let den = Complex::new(1.0, 0.0) - s * Complex::new(dt / 2.0, 0.0);
        gt_z_poles.push(num / den);
    }

    let gt_re: Vec<f64> = gt_z_poles.iter().map(|p| p.re).collect();
    let gt_im: Vec<f64> = gt_z_poles.iter().map(|p| p.im).collect();

    // Direct Form f32 Polynomial Expansion
    let mut df_poly_f32 = vec![1.0f32];
    for pair_idx in 0..3 {
        let p1 = gt_z_poles[2 * pair_idx];
        let p2 = gt_z_poles[2 * pair_idx + 1];
        let b1 = -(p1.re + p2.re) as f32;
        let b0 = (p1.re * p2.re + p1.im * p1.im) as f32;

        let mut next_poly = vec![0.0f32; df_poly_f32.len() + 2];
        for i in 0..df_poly_f32.len() {
            next_poly[i + 2] += df_poly_f32[i];
            next_poly[i + 1] += df_poly_f32[i] * b1;
            next_poly[i] += df_poly_f32[i] * b0;
        }
        df_poly_f32 = next_poly;
    }

    // Solve Direct Form f32 roots using Durand-Kerner
    let solve_df_roots = |c: &[f32]| -> ComplexRoots {
        let mut z = vec![
            Complex::new(0.6, 0.8),
            Complex::new(0.6, -0.8),
            Complex::new(-0.6, 0.8),
            Complex::new(-0.6, -0.8),
            Complex::new(1.1, 0.3),
            Complex::new(1.1, -0.3),
        ];

        let eval_poly = |z_val: Complex<f64>| -> Complex<f64> {
            let mut acc = Complex::new(c[6] as f64, 0.0);
            for i in (0..6).rev() {
                acc = acc * z_val + Complex::new(c[i] as f64, 0.0);
            }
            acc
        };

        for _ in 0..60 {
            let mut z_next = z.clone();
            for i in 0..6 {
                let p_val = eval_poly(z[i]);
                let mut denom = Complex::new(c[6] as f64, 0.0);
                for j in 0..6 {
                    if i != j {
                        denom = denom * (z[i] - z[j]);
                    }
                }
                if denom.re.abs() + denom.im.abs() > 1e-12 {
                    z_next[i] = z[i] - p_val / denom;
                }
            }
            z = z_next;
        }
        z
    };

    let df_roots = solve_df_roots(&df_poly_f32);
    let df_re: Vec<f64> = df_roots.iter().map(|p| p.re).collect();
    let df_im: Vec<f64> = df_roots.iter().map(|p| p.im).collect();

    // Biquad (SOS) f32 Roots
    let mut biquad_re = Vec::with_capacity(6);
    let mut biquad_im = Vec::with_capacity(6);
    for pair_idx in 0..3 {
        let p1 = gt_z_poles[2 * pair_idx];
        let p2 = gt_z_poles[2 * pair_idx + 1];
        let b1 = -(p1.re + p2.re) as f32;
        let b0 = (p1.re * p2.re + p1.im * p1.im) as f32;

        let disc = (b1 * b1 - 4.0 * b0) as f64;
        if disc < 0.0 {
            let re = (-b1 / 2.0) as f64;
            let im = (-disc).sqrt() / 2.0;
            biquad_re.push(re);
            biquad_im.push(im);
            biquad_re.push(re);
            biquad_im.push(-im);
        } else {
            let r1 = ((-b1 as f64) + disc.sqrt()) / 2.0;
            let r2 = ((-b1 as f64) - disc.sqrt()) / 2.0;
            biquad_re.push(r1);
            biquad_im.push(0.0);
            biquad_re.push(r2);
            biquad_im.push(0.0);
        }
    }

    json!({
        "ground_truth_re": gt_re,
        "ground_truth_im": gt_im,
        "direct_form_re": df_re,
        "direct_form_im": df_im,
        "biquad_re": biquad_re,
        "biquad_im": biquad_im,
    })
}

fn run_validation_default() -> Value {
    let discretization = benchmark_discretization_error();
    let nyquist = benchmark_nyquist_criterion();
    let topology = benchmark_topology_stability();

    // Legacy tutorial payload fields
    let tf_realize = Tf::<2, 3>::from_storage(
        StoreNum::from_column([2.0, 3.0]),
        StoreDen::from_column([4.0, 5.0, 1.0]),
        None,
    );
    let ss = tf_realize
        .to_controllable_canonical_form::<2>()
        .expect("CCF failed");

    json!({
        "discretization_error": discretization,
        "nyquist_criterion": nyquist,
        "topology_stability": topology,
        "tutorial": {
            "a00": ss.a().get(0, 0).copied().unwrap_or(0.0),
            "a01": ss.a().get(0, 1).copied().unwrap_or(0.0),
            "a10": ss.a().get(1, 0).copied().unwrap_or(0.0),
            "a11": ss.a().get(1, 1).copied().unwrap_or(0.0),
        }
    })
}

pub fn cross_validate(rust: &Value, python: &Value) -> ValidationResult {
    let mut errs = Vec::new();

    if python.as_object().is_none_or(|o| o.is_empty()) {
        errs.push("Python oracle returned an empty payload".to_string());
        return Err(errs);
    }

    let check_f64 = |key: &str,
                     r_opt: Option<f64>,
                     p_opt: Option<f64>,
                     tol: f64,
                     errs: &mut Vec<String>| {
        match (r_opt, p_opt) {
            (Some(r), Some(p)) => {
                if (r - p).abs() > tol {
                    errs.push(format!(
                        "{key}: rust {r} vs python {p} (tol {tol})"
                    ));
                }
            }
            _ => {
                errs.push(format!("Missing {key} in payload"));
            }
        }
    };

    check_f64(
        "tutorial a00",
        rust["tutorial"]["a00"].as_f64(),
        python["tutorial"]["a00"].as_f64(),
        1e-12,
        &mut errs,
    );
    check_f64(
        "tutorial a01",
        rust["tutorial"]["a01"].as_f64(),
        python["tutorial"]["a01"].as_f64(),
        1e-12,
        &mut errs,
    );
    check_f64(
        "tutorial a10",
        rust["tutorial"]["a10"].as_f64(),
        python["tutorial"]["a10"].as_f64(),
        1e-12,
        &mut errs,
    );
    check_f64(
        "tutorial a11",
        rust["tutorial"]["a11"].as_f64(),
        python["tutorial"]["a11"].as_f64(),
        1e-12,
        &mut errs,
    );

    let check_array = |key: &str,
                       rust_arr: ValueArray,
                       py_arr: ValueArray,
                       tol: f64,
                       errs: &mut Vec<String>| {
        match (rust_arr, py_arr) {
            (Some(r_slice), Some(p_slice)) => {
                if r_slice.len() != p_slice.len() || r_slice.is_empty() {
                    errs.push(format!(
                        "{key} length mismatch: rust {} vs python {}",
                        r_slice.len(),
                        p_slice.len()
                    ));
                } else {
                    for (i, (r, p)) in
                        r_slice.iter().zip(p_slice.iter()).enumerate()
                    {
                        let rv = r.as_f64().unwrap_or(0.0);
                        let pv = p.as_f64().unwrap_or(0.0);
                        if (rv - pv).abs() > tol {
                            errs.push(format!(
                                "{key}[{i}]: rust {rv} vs python {pv} (tol {tol})"
                            ));
                        }
                    }
                }
            }
            _ => {
                errs.push(format!("Missing {key} array in payload"));
            }
        }
    };

    // Discretization error (Bode magnitude and phase)
    check_array(
        "cont_mag_db",
        rust["discretization_error"]["cont_mag_db"].as_array(),
        python["discretization_error"]["cont_mag_db"].as_array(),
        1e-3,
        &mut errs,
    );
    check_array(
        "cont_phase_deg",
        rust["discretization_error"]["cont_phase_deg"].as_array(),
        python["discretization_error"]["cont_phase_deg"].as_array(),
        1e-2,
        &mut errs,
    );
    check_array(
        "tustin_mag_db",
        rust["discretization_error"]["tustin_mag_db"].as_array(),
        python["discretization_error"]["tustin_mag_db"].as_array(),
        1e-3,
        &mut errs,
    );
    check_array(
        "tustin_phase_deg",
        rust["discretization_error"]["tustin_phase_deg"].as_array(),
        python["discretization_error"]["tustin_phase_deg"].as_array(),
        1e-2,
        &mut errs,
    );
    check_array(
        "zoh_mag_db",
        rust["discretization_error"]["zoh_mag_db"].as_array(),
        python["discretization_error"]["zoh_mag_db"].as_array(),
        1e-3,
        &mut errs,
    );
    check_array(
        "zoh_phase_deg",
        rust["discretization_error"]["zoh_phase_deg"].as_array(),
        python["discretization_error"]["zoh_phase_deg"].as_array(),
        1e-2,
        &mut errs,
    );

    // Nyquist criterion arrays and margins
    check_array(
        "nyquist h_re",
        rust["nyquist_criterion"]["h_re"].as_array(),
        python["nyquist_criterion"]["h_re"].as_array(),
        1e-3,
        &mut errs,
    );
    check_array(
        "nyquist h_im",
        rust["nyquist_criterion"]["h_im"].as_array(),
        python["nyquist_criterion"]["h_im"].as_array(),
        1e-3,
        &mut errs,
    );
    check_f64(
        "nyquist phase_margin_deg",
        rust["nyquist_criterion"]["phase_margin_deg"].as_f64(),
        python["nyquist_criterion"]["phase_margin_deg"].as_f64(),
        0.5,
        &mut errs,
    );
    check_f64(
        "nyquist gain_margin_db",
        rust["nyquist_criterion"]["gain_margin_db"].as_f64(),
        python["nyquist_criterion"]["gain_margin_db"].as_f64(),
        0.5,
        &mut errs,
    );

    if errs.is_empty() { Ok(()) } else { Err(errs) }
}

/// Cross-validates against the harold oracle (Transfer model, Misra-Patel Hessenberg
/// frequency response, Tustin/ZOH discretization). Mirrors the SciPy check with the same
/// tolerances, since harold agrees with SciPy to ~1e-10 dB once evaluated with matching
/// frequency-unit conventions.
pub fn cross_validate_harold(rust: &Value, harold: &Value) -> ValidationResult {
    let mut errs = Vec::new();

    if harold.as_object().is_none_or(|o| o.is_empty()) {
        errs.push("Harold oracle returned an empty payload".to_string());
        return Err(errs);
    }

    let check_array = |key: &str,
                       rust_arr: ValueArray,
                       h_arr: ValueArray,
                       tol: f64,
                       errs: &mut Vec<String>| {
        match (rust_arr, h_arr) {
            (Some(r_slice), Some(h_slice)) => {
                if r_slice.len() != h_slice.len() || r_slice.is_empty() {
                    errs.push(format!(
                        "{key} length mismatch: rust {} vs harold {}",
                        r_slice.len(),
                        h_slice.len()
                    ));
                } else {
                    for (i, (r, h)) in
                        r_slice.iter().zip(h_slice.iter()).enumerate()
                    {
                        let rv = r.as_f64().unwrap_or(0.0);
                        let hv = h.as_f64().unwrap_or(0.0);
                        if (rv - hv).abs() > tol {
                            errs.push(format!(
                                "{key}[{i}]: rust {rv} vs harold {hv} (tol {tol})"
                            ));
                        }
                    }
                }
            }
            _ => errs.push(format!("Missing {key} array in payload")),
        }
    };

    check_array(
        "cont_mag_db",
        rust["discretization_error"]["cont_mag_db"].as_array(),
        harold["discretization_error"]["cont_mag_db"].as_array(),
        1e-3,
        &mut errs,
    );
    check_array(
        "cont_phase_deg",
        rust["discretization_error"]["cont_phase_deg"].as_array(),
        harold["discretization_error"]["cont_phase_deg"].as_array(),
        1e-2,
        &mut errs,
    );
    check_array(
        "tustin_mag_db",
        rust["discretization_error"]["tustin_mag_db"].as_array(),
        harold["discretization_error"]["tustin_mag_db"].as_array(),
        1e-3,
        &mut errs,
    );
    check_array(
        "tustin_phase_deg",
        rust["discretization_error"]["tustin_phase_deg"].as_array(),
        harold["discretization_error"]["tustin_phase_deg"].as_array(),
        1e-2,
        &mut errs,
    );
    check_array(
        "zoh_mag_db",
        rust["discretization_error"]["zoh_mag_db"].as_array(),
        harold["discretization_error"]["zoh_mag_db"].as_array(),
        1e-3,
        &mut errs,
    );
    check_array(
        "zoh_phase_deg",
        rust["discretization_error"]["zoh_phase_deg"].as_array(),
        harold["discretization_error"]["zoh_phase_deg"].as_array(),
        1e-2,
        &mut errs,
    );

    check_array(
        "nyquist h_re",
        rust["nyquist_criterion"]["h_re"].as_array(),
        harold["nyquist_criterion"]["h_re"].as_array(),
        1e-3,
        &mut errs,
    );
    check_array(
        "nyquist h_im",
        rust["nyquist_criterion"]["h_im"].as_array(),
        harold["nyquist_criterion"]["h_im"].as_array(),
        1e-3,
        &mut errs,
    );

    let check_f64 = |key: &str,
                     r_opt: Option<f64>,
                     h_opt: Option<f64>,
                     tol: f64,
                     errs: &mut Vec<String>| {
        match (r_opt, h_opt) {
            (Some(r), Some(h)) => {
                if (r - h).abs() > tol {
                    errs.push(format!(
                        "{key}: rust {r} vs harold {h} (tol {tol})"
                    ));
                }
            }
            _ => errs.push(format!("Missing {key} in payload")),
        }
    };

    check_f64(
        "nyquist phase_margin_deg",
        rust["nyquist_criterion"]["phase_margin_deg"].as_f64(),
        harold["nyquist_criterion"]["phase_margin_deg"].as_f64(),
        0.5,
        &mut errs,
    );
    check_f64(
        "nyquist gain_margin_db",
        rust["nyquist_criterion"]["gain_margin_db"].as_f64(),
        harold["nyquist_criterion"]["gain_margin_db"].as_f64(),
        0.5,
        &mut errs,
    );

    if errs.is_empty() { Ok(()) } else { Err(errs) }
}

pub fn run() -> Value {
    println!("Executing Rust transfer function validator...");
    let rust_results = run_validation_default();

    println!("Spawning Python oracle subprocess...");
    let py_output = Command::new("python3")
        .arg("python3/transfer_function_validation.py")
        .output()
        .expect("Failed to spawn Python process");

    if !py_output.status.success() {
        eprintln!(
            "Python oracle failed:\n{}",
            String::from_utf8_lossy(&py_output.stderr)
        );
        std::process::exit(1);
    }

    let py_payload: Value = serde_json::from_slice(&py_output.stdout)
        .expect("Failed to parse Python JSON stdout");
    let py_results = py_payload["scipy"].clone();
    let harold_results = py_payload["harold"].clone();

    if let Err(errs) = cross_validate(&rust_results, &py_results) {
        eprintln!("Transfer Function Cross-Validation Errors (scipy):");
        for e in &errs {
            eprintln!("  - {e}");
        }
        std::process::exit(1);
    }

    if let Err(errs) = cross_validate_harold(&rust_results, &harold_results) {
        eprintln!("Transfer Function Cross-Validation Errors (harold):");
        for e in &errs {
            eprintln!("  - {e}");
        }
        std::process::exit(1);
    }

    let combined_results = json!({
        "metadata": {
            "domain": "transfer_function",
            "timestamp": chrono::Utc::now().to_rfc3339()
        },
        "sources": {
            "rust": {
                "default": rust_results
            },
            "python3": {
                "scipy": py_results,
                "harold": harold_results
            }
        }
    });

    let out_dir = std::env::var("CARGO_MANIFEST_DIR")
        .map(|d| std::path::PathBuf::from(d).join("results"))
        .unwrap_or_else(|_| {
            std::path::PathBuf::from(
                "examples/numerical-models-validation/results",
            )
        });

    fs::create_dir_all(&out_dir).expect("Failed to create results directory");
    let out_path = out_dir.join("transfer_function.json");

    fs::write(
        &out_path,
        serde_json::to_string_pretty(&combined_results).unwrap(),
    )
    .expect("Failed to write results file");

    println!(
        "Success: Transfer Function cross-validation passed! Payload saved to {}",
        out_path.display()
    );

    combined_results
}

#[allow(dead_code)]
pub fn main() {
    run();
}
