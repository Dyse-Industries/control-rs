//! src/polynomial_validation.rs
//!
//! Standalone validation runner for polynomial numerical models.
//! Executes four core benchmark simulations:
//! 1. Execution Time vs. Polynomial Degree (Computational Complexity: Horner vs Naive)
//! 2. Convergence Rate of Root-Finding Algorithms (Algorithmic Efficiency)
//! 3. Residual Error on Ill-Conditioned Polynomials (Wilkinson's Polynomial W(x))
//! 4. Root Sensitivity and Quantization Bounds (Control System Pole Migration)

use serde_json::{Value, json};
use std::fs;
use std::process::Command;
use std::time::Instant;

use control_rs::math::complex_num::Complex;
use control_rs::math::num_types::Const;
use control_rs::math::storage::ArrayStorage;
use control_rs::polynomial::Polynomial;

type Store<const N: usize> = ArrayStorage<f64, N, 1>;
type Poly<const N: usize> = Polynomial<f64, Const<N>, Store<N>>;

pub type ValidationResult = Result<(), Vec<String>>;
type ComplexRoots = Vec<Complex<f64>>;

// Simple LCG PRNG for reproducible perturbation noise
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_f64(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let val = (self.state >> 33) as f64 / (1u64 << 31) as f64; // [0, 1)
        val * 2.0 - 1.0 // [-1, 1)
    }
}

// -----------------------------------------------------------------------------
// 1. Computational Complexity: Degree Sweep (1..50) Horner vs Naive
// -----------------------------------------------------------------------------
fn bench_horner_eval(deg: usize, coeffs: &[f64], eval_points: &[f64]) -> f64 {
    macro_rules! dispatch {
        ($($d:literal),*) => {
            match deg {
                $($d => {
                    const CAP: usize = $d + 1;
                    let mut buf = [0.0; CAP];
                    buf.copy_from_slice(&coeffs[..CAP]);
                    let poly = Polynomial::<f64, Const<CAP>, ArrayStorage<f64, CAP, 1>>::from_coefficients(buf);

                    let t0 = Instant::now();
                    let mut horner_sum = 0.0;
                    for &x in eval_points {
                        horner_sum += poly.evaluate(x);
                    }
                    let elapsed = t0.elapsed().as_nanos() as f64;
                    core::hint::black_box(horner_sum);
                    elapsed
                })*
                _ => unreachable!(),
            }
        };
    }

    dispatch!(
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
        39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50
    )
}

fn benchmark_complexity() -> Value {
    let degrees: Vec<usize> = (1..=50).collect();
    let num_points = 1000;
    let eval_points: Vec<f64> = (0..num_points)
        .map(|i| -1.0 + 2.0 * (i as f64) / ((num_points - 1) as f64))
        .collect();

    let mut horner_times_ns = Vec::with_capacity(degrees.len());
    let mut naive_times_ns = Vec::with_capacity(degrees.len());

    for &deg in &degrees {
        let mut coeffs_buf = vec![0.0; deg + 1];
        for (i, coeff) in coeffs_buf.iter_mut().enumerate() {
            *coeff = 1.0 / ((i + 1) as f64);
        }

        // Benchmark Horner evaluation using Poly::<deg+1>::evaluate()
        let horner_elapsed = bench_horner_eval(deg, &coeffs_buf, &eval_points);

        // Benchmark Naive evaluation over 1000 points
        let t1 = Instant::now();
        let mut naive_sum = 0.0;
        for &x in &eval_points {
            let mut acc = 0.0;
            for (i, &coeff) in coeffs_buf.iter().enumerate() {
                acc += coeff * x.powi(i as i32);
            }
            naive_sum += acc;
        }
        let naive_elapsed = t1.elapsed().as_nanos() as f64;
        core::hint::black_box(naive_sum);

        horner_times_ns.push(horner_elapsed);
        naive_times_ns.push(naive_elapsed);
    }

    json!({
        "degrees": degrees,
        "horner_time_ns": horner_times_ns,
        "naive_time_ns": naive_times_ns,
    })
}

// -----------------------------------------------------------------------------
// 2. Root-Finding Convergence Rate (Using Polynomial & Derivative)
// -----------------------------------------------------------------------------
fn benchmark_root_convergence() -> Value {
    // Target polynomial P(x) = (x - 2)(x + 3)(x - 5) = x^3 - 4x^2 - 11x + 30
    // Ascending order: [30.0, -11.0, -4.0, 1.0]
    let p = Poly::<4>::from_coefficients([30.0, -11.0, -4.0, 1.0]);
    let dp = p.derivative();

    let target_root = 2.0;
    let distances =
        vec![0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0];
    let mut iterations_list = Vec::with_capacity(distances.len());

    for &dist in &distances {
        let x0 = target_root + dist;
        let mut x = x0;
        let mut iters = 0;
        let max_iters = 100;

        while iters < max_iters {
            let fx = p.evaluate(x);
            if fx.abs() < 1e-6 {
                break;
            }
            let fpx = dp.evaluate(x);
            if fpx.abs() < 1e-12 {
                break;
            }
            let next_x = x - fx / fpx;
            if (next_x - x).abs() < 1e-6 {
                iters += 1;
                break;
            }
            x = next_x;
            iters += 1;
        }

        iterations_list.push(iters);
    }

    json!({
        "target_root": target_root,
        "distances": distances,
        "iterations": iterations_list,
    })
}

// -----------------------------------------------------------------------------
// 3. Wilkinson's Polynomial Ill-Conditioning Residual Error (Using Polynomial)
// -----------------------------------------------------------------------------
fn benchmark_wilkinson() -> Value {
    // W(x) = \prod_{i=1}^{20} (x - i)
    let mut coeffs = vec![1.0];
    for root in 1..=20 {
        let r = root as f64;
        let mut next_coeffs = vec![0.0; coeffs.len() + 1];
        for i in 0..coeffs.len() {
            next_coeffs[i + 1] += coeffs[i];
            next_coeffs[i] -= r * coeffs[i];
        }
        coeffs = next_coeffs;
    }

    let mut buf_64 = [0.0; 21];
    buf_64.copy_from_slice(&coeffs[..21]);
    let poly_64 = Poly::<21>::from_coefficients(buf_64);

    let mut buf_32 = [0.0f32; 21];
    for i in 0..21 {
        buf_32[i] = coeffs[i] as f32;
    }
    let poly_32 = Polynomial::<f32, Const<21>, ArrayStorage<f32, 21, 1>>::from_coefficients(buf_32);

    let root_indices: Vec<usize> = (1..=20).collect();
    let mut residual_f64 = Vec::with_capacity(20);
    let mut residual_f32 = Vec::with_capacity(20);

    for &k in &root_indices {
        let r_64 = k as f64;
        let val_64 = poly_64.evaluate(r_64);
        residual_f64.push(val_64.abs());

        let r_32 = k as f32;
        let val_32 = poly_32.evaluate(r_32);
        residual_f32.push(val_32.abs() as f64);
    }

    json!({
        "root_indices": root_indices,
        "residual_f64": residual_f64,
        "residual_f32": residual_f32,
    })
}

// -----------------------------------------------------------------------------
// 4. Root Sensitivity & Quantization Noise Bounds
// -----------------------------------------------------------------------------
fn benchmark_root_sensitivity() -> Value {
    // Stable system poles s = -1 ± 2j, -2 ± 1j
    // Ground truth polynomial (s^2 + 2s + 5)(s^2 + 4s + 5) = s^4 + 6s^3 + 18s^2 + 30s + 25
    // Ascending order: [25.0, 30.0, 18.0, 6.0, 1.0]
    let ground_truth_coeffs = vec![25.0, 30.0, 18.0, 6.0, 1.0];
    let ground_truth_roots_re = vec![-1.0, -1.0, -2.0, -2.0];
    let ground_truth_roots_im = vec![2.0, -2.0, 1.0, -1.0];

    // Durand-Kerner complex root finder using Poly::<5> and evaluate_complex
    let solve_quartic_roots = |poly: &Poly<5>| -> ComplexRoots {
        let mut z = vec![
            Complex::new(0.4, 0.9),
            Complex::new(0.4, -0.9),
            Complex::new(-0.4, 0.9),
            Complex::new(-0.4, -0.9),
        ];
        for _ in 0..40 {
            let mut z_next = z.clone();
            for i in 0..4 {
                // Evaluates P(z[i]) using the library's Horner complex evaluation
                let p_val = poly.evaluate_complex(z[i]);
                let mut denom = Complex::new(1.0, 0.0);
                for j in 0..4 {
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

    let num_trials = 250;
    let mut lcg = Lcg::new(42);
    let noise_scale = 0.015; // 1.5% coefficient perturbation

    let mut perturbed_re = Vec::with_capacity(num_trials * 4);
    let mut perturbed_im = Vec::with_capacity(num_trials * 4);

    for _ in 0..num_trials {
        let mut p_coeffs = [
            ground_truth_coeffs[0],
            ground_truth_coeffs[1],
            ground_truth_coeffs[2],
            ground_truth_coeffs[3],
            ground_truth_coeffs[4],
        ];
        for coeff in p_coeffs.iter_mut().take(4) {
            let noise = lcg.next_f64() * noise_scale;
            *coeff *= 1.0 + noise;
        }
        // Normalize to monic polynomial and instantiate Poly::<5>
        let c4 = p_coeffs[4];
        let poly = Poly::<5>::from_coefficients([
            p_coeffs[0] / c4,
            p_coeffs[1] / c4,
            p_coeffs[2] / c4,
            p_coeffs[3] / c4,
            1.0,
        ]);
        let roots = solve_quartic_roots(&poly);
        for r in roots {
            perturbed_re.push(r.re);
            perturbed_im.push(r.im);
        }
    }

    json!({
        "ground_truth_re": ground_truth_roots_re,
        "ground_truth_im": ground_truth_roots_im,
        "perturbed_re": perturbed_re,
        "perturbed_im": perturbed_im,
    })
}

fn run_validation_default() -> Value {
    let complexity = benchmark_complexity();
    let root_convergence = benchmark_root_convergence();
    let wilkinson = benchmark_wilkinson();
    let root_sensitivity = benchmark_root_sensitivity();

    // Legacy compatibility fields
    let p =
        Poly::<5>::from_storage(Store::from_column([2.0, -3.0, 4.0, 1.0, 0.0]));
    let val_real = p.evaluate(2.5);
    let val_complex = p.evaluate_complex(Complex::new(1.0, 2.0));
    let dp = p.derivative();
    let dp_val = dp.evaluate(2.5);

    json!({
        "complexity": complexity,
        "root_convergence": root_convergence,
        "wilkinson_residual": wilkinson,
        "root_sensitivity": root_sensitivity,
        "tutorial": {
            "p_real": val_real,
            "p_c_re": val_complex.re,
            "p_c_im": val_complex.im,
            "p_deriv": dp_val,
        }
    })
}

/// Computes Higham's backward error factor $\gamma_k = \frac{k\epsilon}{1 - k\epsilon}$.
fn gamma_bound(k: f64, eps: f64) -> f64 {
    let ke = k * eps;
    ke / (1.0 - ke)
}

/// Evaluates $\widetilde{W}(x) = \sum |a_j| x^j = \prod_{j=1}^{20} (x + j)$ for Wilkinson's polynomial.
fn wilkinson_tilde_p(x: f64) -> f64 {
    let mut prod = 1.0;
    for j in 1..=20 {
        prod *= x + (j as f64);
    }
    prod
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

    // 1. Tutorial values
    check_f64(
        "tutorial p_real",
        rust["tutorial"]["p_real"].as_f64(),
        python["tutorial"]["p_real"].as_f64(),
        1e-12,
        &mut errs,
    );
    check_f64(
        "tutorial p_c_re",
        rust["tutorial"]["p_c_re"].as_f64(),
        python["tutorial"]["p_c_re"].as_f64(),
        1e-12,
        &mut errs,
    );
    check_f64(
        "tutorial p_c_im",
        rust["tutorial"]["p_c_im"].as_f64(),
        python["tutorial"]["p_c_im"].as_f64(),
        1e-12,
        &mut errs,
    );

    // 2. Root convergence iterations cross-check
    match (
        rust["root_convergence"]["iterations"].as_array(),
        python["root_convergence"]["iterations"].as_array(),
    ) {
        (Some(r_iters), Some(p_iters)) => {
            if r_iters.len() != p_iters.len() || r_iters.is_empty() {
                errs.push(format!(
                    "root_convergence iterations length mismatch: rust {} vs python {}",
                    r_iters.len(),
                    p_iters.len()
                ));
            } else {
                for (i, (r, p)) in
                    r_iters.iter().zip(p_iters.iter()).enumerate()
                {
                    let rv = r.as_i64().unwrap_or(0);
                    let pv = p.as_i64().unwrap_or(0);
                    if (rv - pv).abs() > 1 {
                        errs.push(format!(
                            "root_convergence iterations[{i}]: rust {rv} vs python {pv}"
                        ));
                    }
                }
            }
        }
        _ => {
            errs.push(
                "Missing root_convergence.iterations array in payload"
                    .to_string(),
            );
        }
    }

    // 3. Wilkinson residual cross-check and precision loss validation
    match (
        rust["wilkinson_residual"]["residual_f64"].as_array(),
        python["wilkinson_residual"]["residual_f64"].as_array(),
        rust["wilkinson_residual"]["residual_f32"].as_array(),
        python["wilkinson_residual"]["residual_f32"].as_array(),
    ) {
        (Some(r_f64), Some(p_f64), Some(r_f32), Some(p_f32)) => {
            if r_f64.len() != 20
                || p_f64.len() != 20
                || r_f32.len() != 20
                || p_f32.len() != 20
            {
                errs.push(format!(
                    "Wilkinson residual length mismatch: rust f64 ({}), py f64 ({}), rust f32 ({}), py f32 ({}) (expected 20)",
                    r_f64.len(),
                    p_f64.len(),
                    r_f32.len(),
                    p_f32.len()
                ));
            } else {
                let gamma_40_f64 = gamma_bound(40.0, f64::EPSILON);
                let gamma_40_f32 = gamma_bound(40.0, f32::EPSILON as f64);

                for i in 0..20 {
                    let k = (i + 1) as f64;
                    let rv64 = r_f64[i].as_f64().unwrap_or(0.0);
                    let pv64 = p_f64[i].as_f64().unwrap_or(0.0);
                    let rv32 = r_f32[i].as_f64().unwrap_or(0.0);
                    let pv32 = p_f32[i].as_f64().unwrap_or(0.0);

                    let tilde_w = wilkinson_tilde_p(k);
                    let bound_f64 = gamma_40_f64 * tilde_w;
                    let bound_f32 = gamma_40_f32 * tilde_w;

                    // 1. Higham backward error bound verification
                    if rv64 > bound_f64 {
                        errs.push(format!(
                            "wilkinson residual_f64[{i}] ({rv64:.4e}) exceeds Higham backward error bound ({bound_f64:.4e})"
                        ));
                    }
                    if rv32 > bound_f32 {
                        errs.push(format!(
                            "wilkinson residual_f32[{i}] ({rv32:.4e}) exceeds Higham backward error bound ({bound_f32:.4e})"
                        ));
                    }
                    if pv32 > bound_f32 {
                        errs.push(format!(
                            "python wilkinson residual_f32[{i}] ({pv32:.4e}) exceeds Higham backward error bound ({bound_f32:.4e})"
                        ));
                    }

                    // 2. Precision loss hierarchy verification
                    if rv32 < rv64 {
                        errs.push(format!(
                            "wilkinson residual_f32[{i}] ({rv32:.4e}) < residual_f64[{i}] ({rv64:.4e}): expected precision loss in f32"
                        ));
                    }
                    if i >= 9 && rv32 < 1e6 * rv64 {
                        errs.push(format!(
                            "wilkinson residual_f32[{i}] ({rv32:.4e}) < 1e6 * residual_f64[{i}] ({rv64:.4e}): insufficient precision gap at ill-conditioned root k={}",
                            i + 1
                        ));
                    }

                    // 3. Logarithmic order-of-magnitude oracle consistency with NumPy
                    if i < 5 {
                        if rv64 > 1e7 {
                            errs.push(format!(
                                "wilkinson residual_f64[{i}] ({rv64:.4e}) exceeds well-conditioned threshold 1e7"
                            ));
                        }
                    } else {
                        let log_r = (rv64 + 1.0).log10();
                        let log_p = (pv64 + 1.0).log10();
                        let log_diff = (log_r - log_p).abs();
                        if log_diff > 2.0 {
                            errs.push(format!(
                                "wilkinson residual_f64[{i}] scale mismatch: rust log10({rv64:.4e})={log_r:.2} vs python log10({pv64:.4e})={log_p:.2} (diff {log_diff:.2} > 2.0)"
                            ));
                        }
                    }
                }
            }
        }
        _ => {
            errs.push(
                "Missing wilkinson_residual arrays in Rust or Python payload"
                    .to_string(),
            );
        }
    }

    if errs.is_empty() { Ok(()) } else { Err(errs) }
}

/// Cross-validates against the python-flint oracle (arb_poly ball arithmetic at 256-bit
/// working precision). Flint's residuals serve as a high-precision ground truth for
/// Wilkinson's polynomial: they should sit at (numerically) zero, in contrast to the large
/// f64 residuals both Rust and SciPy see from catastrophic cancellation near k=20.
pub fn cross_validate_flint(rust: &Value, flint: &Value) -> ValidationResult {
    let mut errs = Vec::new();

    if flint.as_object().is_none_or(|o| o.is_empty()) {
        errs.push("Flint oracle returned an empty payload".to_string());
        return Err(errs);
    }

    match flint["wilkinson_residual"]["residual_f64_flint"].as_array() {
        Some(residuals) => {
            if residuals.len() != 20 {
                errs.push(format!(
                    "Flint wilkinson residual_f64_flint length mismatch: expected 20, got {}",
                    residuals.len()
                ));
            } else {
                for (i, v) in residuals.iter().enumerate() {
                    let val = v.as_f64().unwrap_or(f64::NAN);
                    if !val.is_finite() || val.abs() >= 1e-6 {
                        errs.push(format!(
                            "Flint wilkinson residual_f64_flint[{i}] = {val} exceeds ground-truth tolerance 1e-6"
                        ));
                    }
                }

                // At ill-conditioned roots (k >= 10), flint's 256-bit ground truth must be
                // orders of magnitude smaller than Rust f64 residual exhibiting precision collapse.
                for i in 9..20 {
                    let rust_val = rust["wilkinson_residual"]["residual_f64"][i]
                        .as_f64()
                        .unwrap_or(0.0);
                    let flint_val = residuals[i].as_f64().unwrap_or(0.0);
                    if flint_val >= rust_val {
                        errs.push(format!(
                            "Expected flint ground-truth residual ({flint_val}) < rust f64 residual ({rust_val}) at k={}",
                            i + 1
                        ));
                    }
                }
            }
        }
        None => errs.push(
            "Missing wilkinson_residual.residual_f64_flint array in Flint payload".to_string(),
        ),
    }

    match (
        rust["tutorial"]["p_real"].as_f64(),
        flint["tutorial"]["p_real_flint"].as_f64(),
    ) {
        (Some(r), Some(f)) => {
            let diff = (r - f).abs();
            if diff > 1e-9 {
                errs.push(format!(
                    "tutorial p_real vs flint p_real_flint: rust {r} vs flint {f} (diff {diff})"
                ));
            }
        }
        _ => errs.push(
            "Missing tutorial p_real / p_real_flint in payload".to_string(),
        ),
    }

    if errs.is_empty() { Ok(()) } else { Err(errs) }
}

pub fn run() -> Value {
    println!("Executing Rust polynomial validator...");
    let rust_results = run_validation_default();

    println!("Spawning Python oracle subprocess...");
    let py_output = Command::new("python3")
        .arg("python3/polynomial_validation.py")
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
    let flint_results = py_payload["flint"].clone();

    if let Err(errs) = cross_validate(&rust_results, &py_results) {
        eprintln!("Polynomial Cross-Validation Errors (scipy):");
        for e in &errs {
            eprintln!("  - {e}");
        }
        std::process::exit(1);
    }

    if let Err(errs) = cross_validate_flint(&rust_results, &flint_results) {
        eprintln!("Polynomial Cross-Validation Errors (flint):");
        for e in &errs {
            eprintln!("  - {e}");
        }
        std::process::exit(1);
    }

    let combined_results = json!({
        "metadata": {
            "domain": "polynomial",
            "timestamp": chrono::Utc::now().to_rfc3339()
        },
        "sources": {
            "rust": {
                "default": rust_results
            },
            "python3": {
                "scipy": py_results,
                "flint": flint_results
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
    let out_path = out_dir.join("polynomial.json");

    fs::write(
        &out_path,
        serde_json::to_string_pretty(&combined_results).unwrap(),
    )
    .expect("Failed to write results file");

    println!(
        "Success: Polynomial cross-validation passed!\nResults saved to {}",
        out_path.display()
    );

    combined_results
}

#[allow(dead_code)]
pub fn main() {
    run();
}
