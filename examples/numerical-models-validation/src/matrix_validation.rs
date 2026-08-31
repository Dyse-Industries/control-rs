//! src/matrix_validation.rs
//!
//! Standalone validation runner for matrix numerical models (EKF focus).
//! Implements a 4-Quadrant validation and benchmarking suite:
//! 1. Quadrant 1 (Correctness Anchor): 8x8 EKF Covariance Update
//! 2. Quadrant 2 (Algorithmic Scaling): O(N^3) Matrix Inversion Scaling (1,000 iterations mean/stddev)
//! 3. Quadrant 3 (Determinism): 32x32 Hilbert Solve Latency Jitter (1,000 iterations)
//! 4. Quadrant 4 (Speedup Factor): Decomposition Speedup Factors vs. Python
use serde_json::{Value, json};
use std::fs;
use std::process::Command;
use std::time::Instant;

use control_rs::math::num_types::{Const, Dim};
use control_rs::math::storage::Storage;
use control_rs::matrix::{LuDecomposition, Matrix, Owned, Symmetric};

// ============================================================================
// Serialization Helpers
// ============================================================================

/// Converts any generic Matrix backend into a standard nested Rust Vec
/// for JSON serialization.
fn to_rows<S, const R: usize, const C: usize>(
    mat: &Matrix<f64, Const<R>, Const<C>, S>,
) -> Vec<Vec<f64>>
where
    Const<R>: Dim,
    Const<C>: Dim,
    S: Storage<f64, Const<R>, Const<C>>,
{
    (0..R)
        .map(|i| {
            (0..C)
                .map(|j| mat.get(i, j).copied().unwrap_or(0.0))
                .collect()
        })
        .collect()
}

// ============================================================================
// Quadrant 1: Correctness Anchor (EKF Covariance Update)
// ============================================================================

/// Generates Quadrant 1 EKF Covariance update correctness data on an 8x8 state covariance matrix.
/// P_k|k = (I - K H) * P_k|k-1 * (I - K H)^T + K * R * K^T
fn generate_matrix_correctness_data() -> Value {
    const N: usize = 8;
    // Initial 8x8 state covariance matrix P_0
    let p_0 = Owned::<f64, N, N>::from_fn(|i, j| {
        let diff = (i as f64) - (j as f64);
        (-0.25 * diff * diff).exp() + if i == j { 0.1 } else { 0.0 }
    });

    // Measurement matrix H, Kalman gain K, Noise R
    let h = Owned::<f64, N, N>::identity();
    let k = Owned::<f64, N, N>::from_fn(|i, j| if i == j { 0.4 } else { 0.01 });
    let r = Owned::<f64, N, N>::from_fn(|i, j| if i == j { 0.05 } else { 0.0 });

    let kh = &k * &h;
    let eye = Owned::<f64, N, N>::identity();
    let i_minus_kh = &eye - &kh;

    let k_t = Owned::<f64, N, N>::from_fn(|i, j| k.get(j, i).copied().unwrap());
    let kr = &k * &r;
    let krk_t = &kr * &k_t;
    let i_minus_kh_t = Owned::<f64, N, N>::from_fn(|i, j| {
        i_minus_kh.get(j, i).copied().unwrap()
    });

    let mut p_current = p_0;
    for _ in 0..100 {
        let p_temp = &i_minus_kh * &p_current;
        let p_update1 = &p_temp * &i_minus_kh_t;
        p_current = &p_update1 + &krk_t;
    }

    json!({
        "covariance_heatmap": {
            "rs_matrix": to_rows(&p_current)
        }
    })
}

// ============================================================================
// Quadrant 2: Algorithmic Scaling (O(N^3) Matrix Inversion)
// ============================================================================

macro_rules! bench_inversion_dim {
    ($N:expr, $iters:expr) => {{
        let a = Owned::<f64, $N, $N>::from_fn(|i, j| {
            if i == j {
                2.0 * ((i + 1) as f64)
            } else {
                0.5 / ((i + j + 1) as f64)
            }
        });

        let mut times = Vec::with_capacity($iters);
        for _ in 0..$iters {
            let start = Instant::now();
            let lu =
                LuDecomposition::decompose(a).expect("LU decompose failed");
            let _inv = lu.inverse().expect("LU inverse failed");
            times.push(start.elapsed().as_nanos() as f64);
        }

        let mean = times.iter().sum::<f64>() / ($iters as f64);
        let variance = times.iter().map(|t| (t - mean).powi(2)).sum::<f64>()
            / ($iters as f64);
        let stddev = variance.sqrt();
        (mean, stddev)
    }};
}

fn benchmark_matrix_scaling() -> Value {
    const ITERS: usize = 1000;
    let (m2, s2) = bench_inversion_dim!(2, ITERS);
    let (m4, s4) = bench_inversion_dim!(4, ITERS);
    let (m8, s8) = bench_inversion_dim!(8, ITERS);
    let (m16, s16) = bench_inversion_dim!(16, ITERS);
    let (m32, s32) = bench_inversion_dim!(32, ITERS);
    let (m64, s64) = bench_inversion_dim!(64, ITERS);

    json!({
        "scaling": {
            "N": [2, 4, 8, 16, 32, 64],
            "inversion_time_ns": [m2, m4, m8, m16, m32, m64],
            "inversion_stddev_ns": [s2, s4, s8, s16, s32, s64]
        }
    })
}

// ============================================================================
// Quadrant 3: Determinism (32x32 Hilbert Solve Latency Jitter)
// ============================================================================

fn benchmark_ekf_update_jitter() -> Value {
    const N: usize = 32;
    const ITERS: usize = 1000;

    let h = Owned::<f64, N, N>::from_fn(|i, j| 1.0 / ((i + j + 1) as f64));
    let b = Owned::<f64, N, 1>::from_fn(|_, _| 1.0);

    let lu =
        LuDecomposition::decompose(h).expect("Hilbert LU decomposition failed");
    let mut solve_times_ns = Vec::with_capacity(ITERS);

    for _ in 0..ITERS {
        let mut x = b;
        let start = Instant::now();
        lu.solve_mut(&mut x).expect("Hilbert solve failed");
        solve_times_ns.push(start.elapsed().as_nanos() as f64);
    }

    json!({
        "jitter": {
            "hilbert_solve_times_ns": solve_times_ns
        }
    })
}

// ============================================================================
// Quadrant 4: Decomposition Speedup Factor
// ============================================================================

fn benchmark_decompositions() -> (f64, f64, f64) {
    const N: usize = 16;
    const ITERS: usize = 1000;

    // Symmetric positive-definite matrix for Cholesky
    let spd_owned = Owned::<f64, N, N>::from_fn(|i, j| {
        if i == j {
            10.0 + (i as f64)
        } else {
            1.0 / ((i + j + 2) as f64)
        }
    });
    let spd = Symmetric::<f64, N>::from_owned(spd_owned)
        .expect("Symmetric creation failed");

    let b = Owned::<f64, N, 1>::from_fn(|_, _| 1.0);

    // 1. Cholesky
    let start_chol = Instant::now();
    for _ in 0..ITERS {
        let chol = spd.into_cholesky().expect("Cholesky failed");
        let mut x = b;
        chol.solve_mut(&mut x).expect("Cholesky solve failed");
    }
    let chol_time_ns = start_chol.elapsed().as_nanos() as f64 / (ITERS as f64);

    // 2. LU Solve
    let a_lu = Owned::<f64, N, N>::from_fn(|i, j| {
        if i == j {
            5.0 + (i as f64)
        } else {
            0.2 * ((i + j + 1) as f64)
        }
    });

    let start_lu = Instant::now();
    for _ in 0..ITERS {
        let lu = LuDecomposition::decompose(a_lu).expect("LU decompose failed");
        let mut x = b;
        lu.solve_mut(&mut x).expect("LU solve failed");
    }
    let lu_time_ns = start_lu.elapsed().as_nanos() as f64 / (ITERS as f64);

    // 3. QR Decomposition
    let a_qr = Owned::<f64, N, N>::from_fn(|i, j| {
        if i == j {
            3.0 + (i as f64)
        } else {
            0.1 * ((i + j) as f64)
        }
    });

    let start_qr = Instant::now();
    for _ in 0..ITERS {
        let mut q = Owned::<f64, N, N>::zero();
        let mut a_copy = a_qr;
        a_copy.qr_decompose_mut(&mut q);
    }
    let qr_time_ns = start_qr.elapsed().as_nanos() as f64 / (ITERS as f64);

    (chol_time_ns, lu_time_ns, qr_time_ns)
}

// ============================================================================
// Variant Runners & Orchestration
// ============================================================================

fn run_validation_default() -> (Value, f64, f64, f64) {
    let q1 = generate_matrix_correctness_data();
    let q2 = benchmark_matrix_scaling();
    let q3 = benchmark_ekf_update_jitter();
    let (chol_time, lu_time, qr_time) = benchmark_decompositions();

    let rust_payload = json!({
        "covariance_heatmap": q1["covariance_heatmap"],
        "scaling": q2["scaling"],
        "jitter": q3["jitter"],
        "decomp_times_ns": {
            "cholesky": chol_time,
            "lu_solve": lu_time,
            "qr_decomp": qr_time
        }
    });

    (rust_payload, chol_time, lu_time, qr_time)
}

pub fn cross_validate(rust: &Value, python: &Value) -> Result<(), Vec<String>> {
    let mut errs = Vec::new();

    if python.as_object().map_or(true, |o| o.is_empty()) {
        errs.push("Python oracle returned an empty payload".to_string());
        return Err(errs);
    }

    // Quadrant 1 check: covariance heatmap agreement
    match (
        rust["covariance_heatmap"]["rs_matrix"].as_array(),
        python["covariance_heatmap"]["py_matrix"].as_array(),
    ) {
        (Some(rs_cov), Some(py_cov)) => {
            if rs_cov.len() != py_cov.len() || rs_cov.is_empty() {
                errs.push(format!(
                    "covariance_heatmap row count mismatch: rust {} vs python {}",
                    rs_cov.len(),
                    py_cov.len()
                ));
            } else {
                for (i, (r_row, p_row)) in rs_cov.iter().zip(py_cov.iter()).enumerate()
                {
                    match (r_row.as_array(), p_row.as_array()) {
                        (Some(r_cols), Some(p_cols)) => {
                            if r_cols.len() != p_cols.len() || r_cols.is_empty() {
                                errs.push(format!(
                                    "covariance_heatmap[{i}] col count mismatch: rust {} vs python {}",
                                    r_cols.len(),
                                    p_cols.len()
                                ));
                            } else {
                                for (j, (rv, pv)) in
                                    r_cols.iter().zip(p_cols.iter()).enumerate()
                                {
                                    let r_val = rv.as_f64().unwrap_or(0.0);
                                    let p_val = pv.as_f64().unwrap_or(0.0);
                                    let diff = (r_val - p_val).abs();
                                    if diff > 1e-4 {
                                        errs.push(format!("covariance_heatmap[{i}][{j}]: rust {r_val} vs python {pv} (diff {diff})"));
                                    }
                                }
                            }
                        }
                        _ => {
                            errs.push(format!(
                                "Missing covariance_heatmap[{i}] row array in payload"
                            ));
                        }
                    }
                }
            }
        }
        _ => {
            errs.push("Missing covariance_heatmap arrays in payload".to_string());
        }
    }

    // Quadrant 2 check: scaling data presence
    if rust["scaling"]["inversion_time_ns"]
        .as_array()
        .map_or(0, |a| a.len())
        != 6
    {
        errs.push(
            "Rust scaling inversion_time_ns does not have 6 entries"
                .to_string(),
        );
    }
    if python["scaling"]["inversion_time_ns"]
        .as_array()
        .map_or(0, |a| a.len())
        != 6
    {
        errs.push(
            "Python scaling inversion_time_ns does not have 6 entries"
                .to_string(),
        );
    }

    // Quadrant 3 check: jitter data presence (1000 samples)
    if rust["jitter"]["hilbert_solve_times_ns"]
        .as_array()
        .map_or(0, |a| a.len())
        != 1000
    {
        errs.push(
            "Rust jitter hilbert_solve_times_ns does not have 1000 entries"
                .to_string(),
        );
    }
    if python["jitter"]["hilbert_solve_times_ns"]
        .as_array()
        .map_or(0, |a| a.len())
        != 1000
    {
        errs.push(
            "Python jitter hilbert_solve_times_ns does not have 1000 entries"
                .to_string(),
        );
    }

    if errs.is_empty() { Ok(()) } else { Err(errs) }
}

pub fn run() -> Value {
    println!("Executing Rust matrix validator...");
    let (mut rust_results, chol_rs, lu_rs, qr_rs) = run_validation_default();

    println!("Spawning Python oracle subprocess...");
    let script_paths = [
        "python3/matrix_validation.py",
        "examples/numerical-models-validation/python3/matrix_validation.py",
    ];
    let script_path = script_paths
        .iter()
        .find(|p| std::path::Path::new(p).exists())
        .expect("Could not find python3/matrix_validation.py");

    let py_output = Command::new("python3")
        .arg(script_path)
        .output()
        .expect("Failed to spawn Python process");

    if !py_output.status.success() {
        eprintln!(
            "Python oracle failed:\n{}",
            String::from_utf8_lossy(&py_output.stderr)
        );
        std::process::exit(1);
    }

    let py_results: Value = serde_json::from_slice(&py_output.stdout)
        .expect("Failed to parse Python JSON stdout");

    if let Err(errs) = cross_validate(&rust_results, &py_results) {
        eprintln!("Matrix Cross-Validation Errors:");
        for e in &errs {
            eprintln!("  - {e}");
        }
        std::process::exit(1);
    }

    // Compute Quadrant 4 speedup ratios: T_python / T_rust
    let chol_py = py_results["decomp_times_ns"]["cholesky"]
        .as_f64()
        .unwrap_or(0.0);
    let lu_py = py_results["decomp_times_ns"]["lu_solve"]
        .as_f64()
        .unwrap_or(0.0);
    let qr_py = py_results["decomp_times_ns"]["qr_decomp"]
        .as_f64()
        .unwrap_or(0.0);

    let speedup_chol = if chol_rs > 0.0 {
        chol_py / chol_rs
    } else {
        0.0
    };
    let speedup_lu = if lu_rs > 0.0 { lu_py / lu_rs } else { 0.0 };
    let speedup_qr = if qr_rs > 0.0 { qr_py / qr_rs } else { 0.0 };

    let speedup_json = json!({
        "cholesky": (speedup_chol * 10.0).round() / 10.0,
        "lu_solve": (speedup_lu * 10.0).round() / 10.0,
        "qr_decomp": (speedup_qr * 10.0).round() / 10.0,
        "svd": Value::Null
    });

    rust_results["speedup"] = speedup_json.clone();

    let mut py_combined = py_results.clone();
    py_combined["speedup"] = speedup_json;

    let combined_results = json!({
        "rust": rust_results,
        "python3": py_combined
    });

    let out_dir = std::env::var("CARGO_MANIFEST_DIR")
        .map(|d| std::path::PathBuf::from(d).join("results"))
        .unwrap_or_else(|_| {
            std::path::PathBuf::from(
                "examples/numerical-models-validation/results",
            )
        });

    fs::create_dir_all(&out_dir).expect("Failed to create results directory");
    let out_path = out_dir.join("matrix.json");

    fs::write(
        &out_path,
        serde_json::to_string_pretty(&combined_results).unwrap(),
    )
    .expect("Failed to write results file");

    println!(
        "Success: Matrix cross-validation passed! Payload saved to {}",
        out_path.display()
    );

    combined_results
}

#[allow(dead_code)]
pub fn main() {
    run();
}
