//! src/matrix_validation.rs
//!
//! Standalone validation runner for matrix numerical models.
//! Reads a minimal suite JSON, computes outputs via generic test suites,
//! and emits the resulting matrices to stdout for cross-language comparison.
use serde_json::{json, Value};
use std::fs;
use std::process::Command;
use std::time::Instant;

use control_rs::math::num_types::{Const, Dim};
use control_rs::math::storage::{DenseStorage, Storage};
use control_rs::math::subprograms::DefaultBlas;
use control_rs::matrix::{LuDecomposition, Matrix, Owned};

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
// Generic Validation Suites
// ============================================================================

/// Computes a linear solve on a matrix `a` using manufactured solution `x=1`
/// and returns the solved vector `x_hat`, residual ratio, and execution time in nanoseconds.
pub fn compute_backward_stability<S, const N: usize>(
    a: &Matrix<f64, Const<N>, Const<N>, S>,
) -> (Owned<f64, N, 1>, f64, f64)
where
    Const<N>: Dim,
    S: DenseStorage<f64, R = Const<N>, C = Const<N>>,
{
    let x_true = Owned::<f64, N, 1>::from_fn(|_, _| 1.0);
    let b = a * &x_true;
    let a_owned = Owned::<f64, N, N>::from_fn(|i, j| a.get(i, j).copied().unwrap());

    let start_time = Instant::now();
    let lu = LuDecomposition::decompose(a_owned).expect("LU decompose failed");
    let mut x_hat = b;
    lu.solve_mut_with::<DefaultBlas, 1>(&mut x_hat)
        .expect("LU solve failed");
    let elapsed_ns = start_time.elapsed().as_nanos() as f64;

    let a_norm = a.inf_norm();
    let mut x_hat_norm = 0.0_f64;
    for i in 0..N {
        if let Some(&val) = x_hat.get(i, 0) {
            if val.abs() > x_hat_norm {
                x_hat_norm = val.abs();
            }
        }
    }

    let a_x_hat = a * &x_hat;
    let residual = &a_x_hat - &b;
    let mut residual_norm = 0.0_f64;
    for i in 0..N {
        if let Some(&val) = residual.get(i, 0) {
            if val.abs() > residual_norm {
                residual_norm = val.abs();
            }
        }
    }

    let residual_ratio = residual_norm / (a_norm * x_hat_norm * f64::EPSILON);
    (x_hat, residual_ratio, elapsed_ns)
}

/// Executes a matrix multiplication chain M_next = M_current * K
/// and returns the resulting accumulated matrix and its execution time in nanoseconds.
pub fn compute_matmul_chain<S1, S2, const N: usize>(
    m_init: &Matrix<f64, Const<N>, Const<N>, S1>,
    k: &Matrix<f64, Const<N>, Const<N>, S2>,
    iterations: usize,
) -> (Owned<f64, N, N>, f64)
where
    Const<N>: Dim,
    S1: DenseStorage<f64, R = Const<N>, C = Const<N>>,
    S2: DenseStorage<f64, R = Const<N>, C = Const<N>>,
{
    let mut m_current = Owned::<f64, N, N>::from_fn(|i, j| m_init.get(i, j).copied().unwrap());

    let start_time = Instant::now();
    for _ in 0..iterations {
        m_current = &m_current * k;
    }
    let elapsed_ns = start_time.elapsed().as_nanos() as f64;

    (m_current, elapsed_ns)
}

/// Computes the inverse of matrix `a` via LU decomposition.
/// Returns the inverted matrix, maximum error, and execution time in nanoseconds.
pub fn compute_matrix_inverse<S, const N: usize>(
    a: &Matrix<f64, Const<N>, Const<N>, S>,
) -> (Owned<f64, N, N>, f64, f64)
where
    Const<N>: Dim,
    S: DenseStorage<f64, R = Const<N>, C = Const<N>>,
{
    let a_owned = Owned::<f64, N, N>::from_fn(|i, j| a.get(i, j).copied().unwrap());

    let start_time = Instant::now();
    let lu = LuDecomposition::decompose_with::<DefaultBlas>(a_owned).expect("LU decompose failed");
    let a_inv = lu
        .inverse_with::<DefaultBlas>()
        .expect("Matrix inversion failed");
    let elapsed_ns = start_time.elapsed().as_nanos() as f64;

    let ident = a * &a_inv;
    let mut max_err = 0.0_f64;
    for i in 0..N {
        for j in 0..N {
            let expected = if i == j { 1.0 } else { 0.0 };
            let got = ident.get(i, j).copied().unwrap_or(0.0);
            let err = (got - expected).abs();
            if err > max_err {
                max_err = err;
            }
        }
    }

    (a_inv, max_err, elapsed_ns)
}

// ============================================================================
// Variant Runners & Orchestration
// ============================================================================

fn run_validation_default() -> Value {
    const HILBERT_N: usize = 8;
    const GEMM_N: usize = 64;

    let h = Owned::<f64, HILBERT_N, HILBERT_N>::from_fn(|i, j| 1.0 / ((i + j + 1) as f64));
    let (h_x_hat, h_ratio, h_time_ns) = compute_backward_stability(&h);

    let m_init = Owned::<f64, GEMM_N, GEMM_N>::identity();
    let k = Owned::<f64, GEMM_N, GEMM_N>::from_fn(|i, j| {
        0.01 * ((i + 1) as f64) * ((j + 3) as f64) / 64.0
    });
    let (gemm_final, gemm_time_ns) = compute_matmul_chain(&m_init, &k, 200);

    let a_inv_test = Owned::<f64, 3, 3>::from_fn(|i, j| if i == j { 2.0 } else { 0.5 });
    let (a_inv, identity_error, inv_time_ns) = compute_matrix_inverse(&a_inv_test);

    json!({
        "hilbert": {
            "x_hat": to_rows(&h_x_hat),
            "residual_ratio": h_ratio,
            "time_ns": h_time_ns
        },
        "matmul_chain": {
            "final_matrix": to_rows(&gemm_final),
            "final_norm": gemm_final.inf_norm(),
            "time_ns": gemm_time_ns
        },
        "inversion": {
            "a_inv": to_rows(&a_inv),
            "identity_error": identity_error,
            "time_ns": inv_time_ns
        }
    })
}

pub fn cross_validate(rust: &Value, python: &Value) -> Result<(), Vec<String>> {
    let mut errs = Vec::new();

    if let (Some(rs_x), Some(py_x)) = (
        rust["hilbert"]["x_hat"].as_array(),
        python["hilbert"]["x_hat"].as_array(),
    ) {
        for (i, (r, p)) in rs_x.iter().zip(py_x.iter()).enumerate() {
            let rv = r[0].as_f64().unwrap_or(0.0);
            let pv = p[0].as_f64().unwrap_or(0.0);
            if (rv - pv).abs() > 1e-6 {
                errs.push(format!("Hilbert x_hat[{i}]: rust {rv} vs python {pv}"));
            }
        }
    }

    if let Some(rr) = rust["hilbert"]["residual_ratio"].as_f64() {
        if rr >= 20.0 {
            errs.push(format!("Rust Hilbert residual ratio {rr} >= 20.0"));
        }
    }

    if let (Some(r_norm), Some(p_norm)) = (
        rust["matmul_chain"]["final_norm"].as_f64(),
        python["matmul_chain"]["final_norm"].as_f64(),
    ) {
        let rel_err = (r_norm - p_norm).abs() / p_norm.max(1e-12);
        if rel_err > 1e-6 {
            errs.push(format!("Matmul chain final_norm rel_err {rel_err} > 1e-6"));
        }
    }

    if let (Some(rs_inv), Some(py_inv)) = (
        rust["inversion"]["a_inv"].as_array(),
        python["inversion"]["a_inv"].as_array(),
    ) {
        for (i, (r_row, p_row)) in rs_inv.iter().zip(py_inv.iter()).enumerate() {
            if let (Some(r_cols), Some(p_cols)) = (r_row.as_array(), p_row.as_array()) {
                for (j, (rv, pv)) in r_cols.iter().zip(p_cols.iter()).enumerate() {
                    let r_val = rv.as_f64().unwrap_or(0.0);
                    let p_val = pv.as_f64().unwrap_or(0.0);
                    if (r_val - p_val).abs() > 1e-12 {
                        errs.push(format!("a_inv[{i}][{j}]: rust {r_val} vs python {p_val}"));
                    }
                }
            }
        }
    }

    if errs.is_empty() { Ok(()) } else { Err(errs) }
}

pub fn run() -> Value {
    println!("Executing Rust matrix validator...");
    let rust_results = run_validation_default();

    println!("Spawning Python oracle subprocess...");
    let py_output = Command::new("python3")
        .arg("python3/matrix_validation.py")
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

    let combined_results = json!({
        "rust": rust_results,
        "python3": py_results
    });

    fs::create_dir_all("results").expect("Failed to create results directory");
    let out_path = "results/matrix.json";

    fs::write(
        out_path,
        serde_json::to_string_pretty(&combined_results).unwrap(),
    )
    .expect("Failed to write results file");

    println!(
        "Success: Matrix cross-validation passed! Payload saved to {}",
        out_path
    );

    combined_results
}

#[allow(dead_code)]
pub fn main() {
    run();
}