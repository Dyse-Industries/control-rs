//! src/transfer_function_validation.rs
//!
//! Standalone validation runner for transfer function numerical models.
//! Computes outputs natively in Rust, spawns Python oracle subprocess,
//! and emits the combined results payload to results/transfer_function.json.

use serde_json::{json, Value};
use std::fs;
use std::process::Command;
use std::time::Instant;

use control_rs::math::dsp::DefaultDsp;
use control_rs::math::num_types::{Const, Dim};
use control_rs::math::storage::{ArrayStorage, Storage};
use control_rs::math::subprograms::DefaultBlas;
use control_rs::matrix::Matrix;
use control_rs::polynomial::Polynomial;
use control_rs::transfer_function::TransferFunction;

type StoreNum<const N: usize> = ArrayStorage<f64, N, 1>;
type StoreDen<const D: usize> = ArrayStorage<f64, D, 1>;
type Dsp = DefaultDsp;
type Blas = DefaultBlas;
type Tf<const N: usize, const D: usize> =
    TransferFunction<f64, Const<N>, Const<D>, StoreNum<N>, StoreDen<D>>;

const BODE_N: usize = 128;

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

fn logspace(start_log10: f64, stop_log10: f64, n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![10.0_f64.powf(stop_log10)];
    }
    let den = (n - 1) as f64;
    (0..n)
        .map(|i| 10.0_f64.powf(start_log10 + (stop_log10 - start_log10) * (i as f64) / den))
        .collect()
}

fn binomial(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    let mut v = 1.0_f64;
    for i in 0..k {
        v = v * ((n - i) as f64) / ((i + 1) as f64);
    }
    v
}

fn poly_s_plus_a_4(a: f64) -> [f64; 5] {
    let n = 4_usize;
    let mut c = [0.0_f64; 5];
    for k in 0..=n {
        c[k] = binomial(n, k) * a.powi((n - k) as i32);
    }
    c
}

fn run_validation_default() -> Value {
    let omega_n = 10.0;
    let zeta = 0.2;
    let tf = Tf::<1, 3>::from_storage(
        StoreNum::from_column([100.0]),
        StoreDen::from_column([100.0, 4.0, 1.0]),
        None,
    );

    let freqs = logspace(-2.0, 3.0, BODE_N);
    let mut h_re = vec![0.0_f64; BODE_N];
    let mut h_im = vec![0.0_f64; BODE_N];
    let mut mag = vec![0.0_f64; BODE_N];
    let mut phase = vec![0.0_f64; BODE_N];

    let t_bode = Instant::now();
    for (idx, &w) in freqs.iter().enumerate() {
        let resp = tf.eval_frequency(w);
        let (mag_pt, phase_rad) = tf.bode_point(w);
        h_re[idx] = resp.re;
        h_im[idx] = resp.im;
        mag[idx] = mag_pt;
        phase[idx] = phase_rad;
    }
    let bode_time_ns = t_bode.elapsed().as_nanos() as f64;

    let h1 = Tf::<1, 2>::from_storage(
        StoreNum::from_column([2.0]),
        StoreDen::from_column([2.0, 1.0]),
        None,
    );
    let h2 = Tf::<1, 2>::from_storage(
        StoreNum::from_column([5.0]),
        StoreDen::from_column([5.0, 1.0]),
        None,
    );

    let t_series = Instant::now();
    let h_series = h1.series_with::<Dsp, 1, 2, 1, 3>(&h2);
    let series_time_ns = t_series.elapsed().as_nanos() as f64;

    let tf_realize = Tf::<2, 3>::from_storage(
        StoreNum::from_column([2.0, 3.0]),
        StoreDen::from_column([4.0, 5.0, 1.0]),
        None,
    );

    let t_ccf = Instant::now();
    let ss = tf_realize
        .to_controllable_canonical_form_with::<Blas, 2>()
        .expect("CCF failed");
    let ccf_time_ns = t_ccf.elapsed().as_nanos() as f64;

    let d1 = Polynomial::<f64, Const<5>, StoreDen<5>>::from_storage(
        StoreDen::from_column(poly_s_plus_a_4(1.0)),
    );
    let d2 = Polynomial::<f64, Const<5>, StoreDen<5>>::from_storage(
        StoreDen::from_column(poly_s_plus_a_4(1.01)),
    );
    let den_c = d1.mul_poly_with::<Dsp, 5, 9>(&d2);
    let mut den_arr = [0.0_f64; 9];
    for i in 0..9 {
        den_arr[i] = den_c.get(i).copied().unwrap_or(0.0);
    }
    let tf_c = Tf::<1, 9>::from_storage(
        StoreNum::from_column([1.0]),
        StoreDen::from_column(den_arr),
        None,
    );
    let mut c_re = vec![0.0_f64; BODE_N];
    let mut c_im = vec![0.0_f64; BODE_N];
    let mut c_mag = vec![0.0_f64; BODE_N];

    let t_c_bode = Instant::now();
    for (idx, &w) in freqs.iter().enumerate() {
        let resp = tf_c.eval_frequency(w);
        c_re[idx] = resp.re;
        c_im[idx] = resp.im;
        c_mag[idx] = (resp.re * resp.re + resp.im * resp.im).sqrt();
    }
    let cluster_bode_time_ns = t_c_bode.elapsed().as_nanos() as f64;

    json!({
        "complex_pair": {
            "h_re": h_re,
            "h_im": h_im,
            "freqs": freqs,
            "mag": mag,
            "phase": phase,
            "omega_n": omega_n,
            "zeta": zeta,
            "bode_time_ns": bode_time_ns,
        },
        "series": {
            "num_ser": [h_series.num_slice()[0]],
            "den_ser": [h_series.den_slice()[0], h_series.den_slice()[1], h_series.den_slice()[2]],
            "series_time_ns": series_time_ns,
        },
        "ccf": {
            "a": to_rows(&ss.a()),
            "b": to_rows(&ss.b()),
            "c": to_rows(&ss.c()),
            "d": to_rows(&ss.d()),
            "ccf_time_ns": ccf_time_ns,
        },
        "clustered": {
            "h_re": c_re,
            "h_im": c_im,
            "mag": c_mag,
            "cluster_bode_time_ns": cluster_bode_time_ns,
        }
    })
}

pub fn cross_validate(rust: &Value, python: &Value) -> Result<(), Vec<String>> {
    let mut errs = Vec::new();

    if let (Some(r_re), Some(p_re)) = (
        rust["complex_pair"]["h_re"].as_array(),
        python["complex_pair"]["h_re"].as_array(),
    ) {
        for (i, (r, p)) in r_re.iter().zip(p_re.iter()).enumerate() {
            let rv = r.as_f64().unwrap_or(0.0);
            let pv = p.as_f64().unwrap_or(0.0);
            if (rv - pv).abs() > 1e-6 {
                errs.push(format!("complex_pair h_re[{i}]: rust {rv} vs python {pv}"));
            }
        }
    }

    if let (Some(r_num), Some(p_num)) = (
        rust["series"]["num_ser"].as_array(),
        python["series"]["num_ser"].as_array(),
    ) {
        for (i, (r, p)) in r_num.iter().zip(p_num.iter()).enumerate() {
            let rv = r.as_f64().unwrap_or(0.0);
            let pv = p.as_f64().unwrap_or(0.0);
            if (rv - pv).abs() > 1e-12 {
                errs.push(format!("series num_ser[{i}]: rust {rv} vs python {pv}"));
            }
        }
    }

    if let (Some(r_den), Some(p_den)) = (
        rust["series"]["den_ser"].as_array(),
        python["series"]["den_ser"].as_array(),
    ) {
        for (i, (r, p)) in r_den.iter().zip(p_den.iter()).enumerate() {
            let rv = r.as_f64().unwrap_or(0.0);
            let pv = p.as_f64().unwrap_or(0.0);
            if (rv - pv).abs() > 1e-12 {
                errs.push(format!("series den_ser[{i}]: rust {rv} vs python {pv}"));
            }
        }
    }

    if errs.is_empty() { Ok(()) } else { Err(errs) }
}

pub fn run() -> Value {
    println!("Executing Rust transfer-function validator...");
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

    let py_results: Value = serde_json::from_slice(&py_output.stdout)
        .expect("Failed to parse Python JSON stdout");

    if let Err(errs) = cross_validate(&rust_results, &py_results) {
        eprintln!("Transfer Function Cross-Validation Errors:");
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
    let out_path = "results/transfer_function.json";

    fs::write(
        out_path,
        serde_json::to_string_pretty(&combined_results).unwrap(),
    )
    .expect("Failed to write results file");

    println!(
        "Success: Transfer Function cross-validation passed! Payload saved to {}",
        out_path
    );

    combined_results
}

#[allow(dead_code)]
pub fn main() {
    run();
}
