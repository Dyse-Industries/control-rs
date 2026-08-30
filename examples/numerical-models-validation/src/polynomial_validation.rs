//! src/polynomial_validation.rs
//!
//! Standalone validation runner for polynomial numerical models.
//! Computes outputs natively in Rust, spawns Python oracle subprocess,
//! and emits the combined results payload to results/polynomial.json.

use serde_json::{json, Value};
use std::fs;
use std::process::Command;
use std::time::Instant;

use control_rs::math::complex_num::Complex;
use control_rs::math::dsp::DefaultDsp;
use control_rs::math::num_types::{Const, Dim};
use control_rs::math::storage::{ArrayStorage, Storage};
use control_rs::matrix::Matrix;
use control_rs::polynomial::Polynomial;

type Store<const N: usize> = ArrayStorage<f64, N, 1>;
type Dsp = DefaultDsp;
type Poly<const N: usize> = Polynomial<f64, Const<N>, Store<N>>;

const SWEEP_N: usize = 128;

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

fn first_n<const N: usize>(p: &Poly<N>, n: usize) -> Vec<f64>
where
    Const<N>: Dim,
{
    (0..n).map(|i| p.get(i).copied().unwrap_or(0.0)).collect()
}

fn run_validation_default() -> Value {
    let p = Poly::<5>::from_storage(Store::from_column([2.0, -3.0, 4.0, 1.0, 0.0]));
    let x_test = 2.5;

    let t_eval = Instant::now();
    let val_real = p.evaluate(x_test);
    let eval_time_ns = t_eval.elapsed().as_nanos() as f64;

    let s_test = Complex::new(1.0, 2.0);
    let t_c_eval = Instant::now();
    let val_complex = p.evaluate_complex(s_test);
    let complex_eval_time_ns = t_c_eval.elapsed().as_nanos() as f64;

    let t_deriv = Instant::now();
    let dp = p.derivative();
    let dp_val = dp.evaluate(x_test);
    let deriv_time_ns = t_deriv.elapsed().as_nanos() as f64;

    let c0 = 5.0;
    let t_integ = Instant::now();
    let integ = p.integral(c0);
    let integ_val = integ.evaluate(x_test);
    let integ_time_ns = t_integ.elapsed().as_nanos() as f64;

    let p1 = Poly::<2>::from_storage(Store::from_column([1.0, 2.0]));
    let p2 = Poly::<2>::from_storage(Store::from_column([3.0, 4.0]));

    let t_mul = Instant::now();
    let prod = p1.mul_poly_with::<Dsp, 2, 3>(&p2);
    let mul_time_ns = t_mul.elapsed().as_nanos() as f64;

    let t_div = Instant::now();
    let (quot, rem) = prod.div_rem::<2, 2, 1>(&p1).expect("div_rem failed");
    let div_time_ns = t_div.elapsed().as_nanos() as f64;

    let p_monic = Poly::<13>::from_storage(Store::from_column([
        1.0, 12.0, 66.0, 220.0, 495.0, 792.0, 924.0, 792.0, 495.0, 220.0, 66.0, 12.0, 1.0,
    ]));
    let t_comp = Instant::now();
    let comp = p_monic.companion_matrix::<12>().expect("companion matrix failed");
    let companion_time_ns = t_comp.elapsed().as_nanos() as f64;

    let cluster = Poly::<17>::from_storage(Store::from_column([
        20922789888000.0,
        -70734282393600.0,
        102992244837120.0,
        -87077748875904.0,
        48366009233424.0,
        -18861567058880.0,
        5374523477960.0,
        -1146901283528.0,
        185953177553.0,
        -23057159840.0,
        2185031420.0,
        -156952432.0,
        8394022.0,
        -323680.0,
        8500.0,
        -136.0,
        1.0,
    ]));

    let sweep_start = 0.9_f64;
    let sweep_stop = 1.1_f64;
    let sweep_x: Vec<f64> = (0..SWEEP_N)
        .map(|i| sweep_start + (sweep_stop - sweep_start) * (i as f64) / ((SWEEP_N - 1) as f64))
        .collect();
    let cluster_y: Vec<f64> = sweep_x.iter().map(|&x| cluster.evaluate(x)).collect();

    let timed_x = 1.005;
    let iters = 10_000;
    let start_time = Instant::now();
    for _ in 0..iters {
        let _ = cluster.evaluate(core::hint::black_box(timed_x));
    }
    let elapsed_ns = start_time.elapsed().as_nanos() as f64;

    json!({
        "tutorial": {
            "p_real": val_real,
            "p_c_re": val_complex.re,
            "p_c_im": val_complex.im,
            "deriv": first_n(&dp, 5),
            "p_deriv": dp_val,
            "integ": first_n(&integ, 5),
            "p_integ": integ_val,
            "prod": first_n(&prod, 3),
            "quot": first_n(&quot, 2),
            "rem": rem.get(0).copied().unwrap_or(0.0),
            "companion": to_rows(&comp),
            "eval_time_ns": eval_time_ns,
            "complex_eval_time_ns": complex_eval_time_ns,
            "deriv_time_ns": deriv_time_ns,
            "integ_time_ns": integ_time_ns,
            "mul_time_ns": mul_time_ns,
            "div_time_ns": div_time_ns,
            "companion_time_ns": companion_time_ns,
        },
        "clustered": {
            "coeffs": first_n(&cluster, 17),
            "x": sweep_x,
            "y": cluster_y,
            "time_ns": elapsed_ns,
        }
    })
}

pub fn cross_validate(rust: &Value, python: &Value) -> Result<(), Vec<String>> {
    let mut errs = Vec::new();

    let check_f64 = |key: &str, r: f64, p: f64, tol: f64, errs: &mut Vec<String>| {
        if (r - p).abs() > tol {
            errs.push(format!("{key}: rust {r} vs python {p} (tol {tol})"));
        }
    };

    if let (Some(r), Some(p)) = (
        rust["tutorial"]["p_real"].as_f64(),
        python["tutorial"]["p_real"].as_f64(),
    ) {
        check_f64("p_real", r, p, 1e-12, &mut errs);
    }

    if let (Some(r), Some(p)) = (
        rust["tutorial"]["p_c_re"].as_f64(),
        python["tutorial"]["p_c_re"].as_f64(),
    ) {
        check_f64("p_c_re", r, p, 1e-12, &mut errs);
    }

    if let (Some(r), Some(p)) = (
        rust["tutorial"]["p_c_im"].as_f64(),
        python["tutorial"]["p_c_im"].as_f64(),
    ) {
        check_f64("p_c_im", r, p, 1e-12, &mut errs);
    }

    if let (Some(r), Some(p)) = (
        rust["tutorial"]["p_deriv"].as_f64(),
        python["tutorial"]["p_deriv"].as_f64(),
    ) {
        check_f64("p_deriv", r, p, 1e-12, &mut errs);
    }

    if let (Some(r), Some(p)) = (
        rust["tutorial"]["p_integ"].as_f64(),
        python["tutorial"]["p_integ"].as_f64(),
    ) {
        check_f64("p_integ", r, p, 1e-12, &mut errs);
    }

    if let (Some(r_y), Some(p_y)) = (
        rust["clustered"]["y"].as_array(),
        python["clustered"]["y"].as_array(),
    ) {
        for (i, (r, p)) in r_y.iter().zip(p_y.iter()).enumerate() {
            let rv = r.as_f64().unwrap_or(0.0);
            let pv = p.as_f64().unwrap_or(0.0);
            let rel_err = (rv - pv).abs() / pv.abs().max(1.0);
            if rel_err > 1e-6 {
                errs.push(format!("clustered y[{i}]: rust {rv} vs python {pv} (rel_err {rel_err})"));
            }
        }
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

    let py_results: Value = serde_json::from_slice(&py_output.stdout)
        .expect("Failed to parse Python JSON stdout");

    if let Err(errs) = cross_validate(&rust_results, &py_results) {
        eprintln!("Polynomial Cross-Validation Errors:");
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
    let out_path = "results/polynomial.json";

    fs::write(
        out_path,
        serde_json::to_string_pretty(&combined_results).unwrap(),
    )
    .expect("Failed to write results file");

    println!(
        "Success: Polynomial cross-validation passed! Payload saved to {}",
        out_path
    );

    combined_results
}

#[allow(dead_code)]
pub fn main() {
    run();
}
