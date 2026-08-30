//! src/state_space_validation.rs
//!
//! Standalone validation runner for state-space numerical models.
//! Computes outputs natively in Rust, spawns Python oracle subprocess,
//! and emits the combined results payload to results/state_space.json.

use serde_json::{json, Value};
use std::fs;
use std::process::Command;
use std::time::Instant;

use control_rs::math::num_types::{Const, Dim};
use control_rs::math::storage::{ArrayStorage, Storage};
use control_rs::matrix::Matrix;
use control_rs::state_space::ArrayStateSpace;

type Store<const R: usize, const C: usize> = ArrayStorage<f64, R, C>;
type Mat<const R: usize, const C: usize> = Matrix<f64, Const<R>, Const<C>, Store<R, C>>;

const NUM_STEPS: usize = 200;
const STIFF_STEPS: usize = 200;

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

fn run_traj(
    sys_d: &ArrayStateSpace<f64, 2, 1, 1>,
    n: usize,
    x0: [f64; 2],
    u: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut x_k = Mat::<2, 1>::from_storage(Store::from_array([[x0[0], x0[1]]]));
    let u_k = Mat::<1, 1>::from_fn(|_, _| u);
    let mut traj_x1 = vec![0.0_f64; n];
    let mut traj_x2 = vec![0.0_f64; n];
    let mut traj_y = vec![0.0_f64; n];
    for k in 0..n {
        let pos = x_k.get(0, 0).copied().unwrap_or(0.0);
        let vel = x_k.get(1, 0).copied().unwrap_or(0.0);
        let (x_next, y_k) = sys_d.step(&x_k, &u_k);
        let y_out = y_k.get(0, 0).copied().unwrap_or(0.0);
        traj_x1[k] = pos;
        traj_x2[k] = vel;
        traj_y[k] = y_out;
        x_k = x_next;
    }
    (traj_x1, traj_x2, traj_y)
}

fn run_validation_default() -> Value {
    let a_c = Mat::<2, 2>::from_storage(Store::from_array([[0.0, -4.0], [1.0, -0.8]]));
    let b_c = Mat::<2, 1>::from_storage(Store::from_array([[0.0, 1.0]]));
    let c_c = Mat::<1, 2>::from_storage(Store::from_array([[1.0], [0.0]]));
    let d_c = Mat::<1, 1>::from_storage(Store::from_array([[0.0]]));

    let sys_c = ArrayStateSpace::continuous(a_c, b_c, c_c, d_c);

    let x_test = Mat::<2, 1>::from_storage(Store::from_array([[1.0, 0.5]]));
    let u_test = Mat::<1, 1>::from_fn(|_, _| 0.0);

    let t_deriv = Instant::now();
    let (x_dot, y_test) = sys_c.derivative(&x_test, &u_test);
    let deriv_time_ns = t_deriv.elapsed().as_nanos() as f64;
    let y_val = y_test.get(0, 0).copied().unwrap_or(0.0);

    let dt = 0.05;
    let t_zoh = Instant::now();
    let sys_d = sys_c.to_discrete_zoh(dt);
    let zoh_time_ns = t_zoh.elapsed().as_nanos() as f64;

    let t_step = Instant::now();
    let (step_x1, step_x2, step_y) = run_traj(&sys_d, NUM_STEPS, [0.0, 0.0], 1.0);
    let step_time_ns = t_step.elapsed().as_nanos() as f64;

    let (free_x1, free_x2, _) = run_traj(&sys_d, NUM_STEPS, [1.0, 0.5], 0.0);

    let t = Mat::<2, 2>::from_storage(Store::from_array([[1.0, 0.0], [1.0, 1.0]]));
    let t_sim = Instant::now();
    let sys_transformed = sys_d.similarity_transform(&t).expect("similarity transform");
    let similarity_time_ns = t_sim.elapsed().as_nanos() as f64;

    // 2. Stiff Plant Setup
    let a_s = Mat::<2, 2>::from_storage(Store::from_array([[-200.0, 0.0], [0.0, -0.5]]));
    let b_s = Mat::<2, 1>::from_storage(Store::from_array([[1.0, 1.0]]));
    let c_s = Mat::<1, 2>::from_storage(Store::from_array([[1.0], [1.0]]));
    let d_s = Mat::<1, 1>::from_storage(Store::from_array([[0.0]]));
    let sys_s = ArrayStateSpace::continuous(a_s, b_s, c_s, d_s);
    let dt_s = 0.01;

    let t_stiff_zoh = Instant::now();
    let sys_sd = sys_s.to_discrete_zoh(dt_s);
    let stiff_zoh_time_ns = t_stiff_zoh.elapsed().as_nanos() as f64;

    let (_, _, stiff_y) = run_traj(&sys_sd, STIFF_STEPS, [0.0, 0.0], 1.0);

    json!({
        "tutorial": {
            "x_dot": to_rows(&x_dot),
            "y_test": y_val,
            "ad": to_rows(&sys_d.a()),
            "bd": to_rows(&sys_d.b()),
            "step_x1": step_x1,
            "step_x2": step_x2,
            "step_y": step_y,
            "free_x1": free_x1,
            "free_x2": free_x2,
            "a_tilde": to_rows(&sys_transformed.a()),
            "b_tilde": to_rows(&sys_transformed.b()),
            "c_tilde": to_rows(&sys_transformed.c()),
            "deriv_time_ns": deriv_time_ns,
            "zoh_time_ns": zoh_time_ns,
            "step_time_ns": step_time_ns,
            "similarity_time_ns": similarity_time_ns,
        },
        "stiff": {
            "ad": to_rows(&sys_sd.a()),
            "bd": to_rows(&sys_sd.b()),
            "y": stiff_y,
            "stiff_zoh_time_ns": stiff_zoh_time_ns,
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
        rust["tutorial"]["y_test"].as_f64(),
        python["tutorial"]["y_test"].as_f64(),
    ) {
        check_f64("y_test", r, p, 1e-12, &mut errs);
    }

    if let (Some(r_y), Some(p_y)) = (
        rust["tutorial"]["step_y"].as_array(),
        python["tutorial"]["step_y"].as_array(),
    ) {
        for (i, (r, p)) in r_y.iter().zip(p_y.iter()).enumerate() {
            let rv = r.as_f64().unwrap_or(0.0);
            let pv = p.as_f64().unwrap_or(0.0);
            if (rv - pv).abs() > 1e-6 {
                errs.push(format!("step_y[{i}]: rust {rv} vs python {pv}"));
            }
        }
    }

    if let (Some(r_ad), Some(p_ad)) = (
        rust["tutorial"]["ad"].as_array(),
        python["tutorial"]["ad"].as_array(),
    ) {
        for (i, (r_row, p_row)) in r_ad.iter().zip(p_ad.iter()).enumerate() {
            if let (Some(r_cols), Some(p_cols)) = (r_row.as_array(), p_row.as_array()) {
                for (j, (rv, pv)) in r_cols.iter().zip(p_cols.iter()).enumerate() {
                    let r_val = rv.as_f64().unwrap_or(0.0);
                    let p_val = pv.as_f64().unwrap_or(0.0);
                    if (r_val - p_val).abs() > 1e-12 {
                        errs.push(format!("ad[{i}][{j}]: rust {r_val} vs python {p_val}"));
                    }
                }
            }
        }
    }

    if errs.is_empty() { Ok(()) } else { Err(errs) }
}

pub fn run() -> Value {
    println!("Executing Rust state-space validator...");
    let rust_results = run_validation_default();

    println!("Spawning Python oracle subprocess...");
    let py_output = Command::new("python3")
        .arg("python3/state_space_validation.py")
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
        eprintln!("State-Space Cross-Validation Errors:");
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
    let out_path = "results/state_space.json";

    fs::write(
        out_path,
        serde_json::to_string_pretty(&combined_results).unwrap(),
    )
    .expect("Failed to write results file");

    println!(
        "Success: State-Space cross-validation passed! Payload saved to {}",
        out_path
    );

    combined_results
}

#[allow(dead_code)]
pub fn main() {
    run();
}
