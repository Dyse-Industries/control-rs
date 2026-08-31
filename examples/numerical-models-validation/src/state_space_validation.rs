//! src/state_space_validation.rs
//!
//! Standalone validation runner for state-space numerical models.
//! Computes outputs natively in Rust, spawns Python oracle subprocess,
//! and emits the combined results payload to results/state_space.json.

use control_rs::math::num_types::Const;
use control_rs::math::storage::ArrayStorage;
use control_rs::matrix::Matrix;
use control_rs::state_space::ArrayStateSpace;
use serde_json::{Value, json};
use std::fs;
use std::process::Command;
use std::time::Instant;

type Store<const R: usize, const C: usize> = ArrayStorage<f64, R, C>;
type Mat<const R: usize, const C: usize> =
    Matrix<f64, Const<R>, Const<C>, Store<R, C>>;

/// Explicit Pendulum State-Space Simulation Model.
pub struct PendulumSim {
    pub sys_d: ArrayStateSpace<f64, 2, 1, 1>,
}

impl PendulumSim {
    pub fn new(omega0: f64, b: f64, dt: f64) -> Self {
        let omega0_sq = omega0 * omega0;
        let a_c = Mat::<2, 2>::from_storage(Store::from_array([
            [0.0, -omega0_sq],
            [1.0, -b],
        ]));
        let b_c = Mat::<2, 1>::from_storage(Store::from_array([[0.0, 1.0]]));
        let c_c = Mat::<1, 2>::from_storage(Store::from_array([[1.0], [0.0]]));
        let d_c = Mat::<1, 1>::from_storage(Store::from_array([[0.0]]));

        let sys_c = ArrayStateSpace::continuous(a_c, b_c, c_c, d_c);
        let sys_d = sys_c.to_discrete_zoh(dt);
        Self { sys_d }
    }

    pub fn simulate(
        &self,
        x0: [f64; 2],
        n_steps: usize,
        u_val: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut x_k =
            Mat::<2, 1>::from_storage(Store::from_array([[x0[0], x0[1]]]));
        let u_k = Mat::<1, 1>::from_fn(|_, _| u_val);

        let mut theta = Vec::with_capacity(n_steps);
        let mut theta_dot = Vec::with_capacity(n_steps);

        for _ in 0..n_steps {
            let th = x_k.get(0, 0).copied().unwrap_or(0.0);
            let th_dot = x_k.get(1, 0).copied().unwrap_or(0.0);
            theta.push(th);
            theta_dot.push(th_dot);
            let (x_next, _) = self.sys_d.step(&x_k, &u_k);
            x_k = x_next;
        }

        (theta, theta_dot)
    }
}

fn generate_state_space_correctness_data() -> Value {
    let sim = PendulumSim::new(2.0, 0.8, 0.05);
    let (theta, theta_dot) =
        sim.simulate([std::f64::consts::PI - 0.15, 0.5], 200, 0.0);

    json!({
        "phase_portrait": {
            "theta": theta,
            "theta_dot": theta_dot
        }
    })
}

macro_rules! bench_dim {
    ($N:expr) => {{
        let mut a = Mat::<$N, $N>::zero();
        let mut b = Mat::<$N, 1>::zero();
        let mut c = Mat::<1, $N>::zero();
        let d = Mat::<1, 1>::zero();

        for i in 0..$N {
            for j in 0..$N {
                let val = if i == j {
                    -0.5 * ((i + 1) as f64)
                } else {
                    0.1 / ((i + j + 1) as f64)
                };
                if let Some(target) = a.get_mut(i, j) {
                    *target = val;
                }
            }
            if let Some(target) = b.get_mut(i, 0) {
                *target = 1.0 / ((i + 1) as f64);
            }
            if let Some(target) = c.get_mut(0, i) {
                *target = 1.0 / ((i + 1) as f64);
            }
        }

        let sys_c = ArrayStateSpace::continuous(a, b, c, d);

        let t_zoh = Instant::now();
        let _sys_d = sys_c.to_discrete_zoh(0.05);
        let zoh_time_ns = t_zoh.elapsed().as_nanos() as f64;

        let t_ctrb = Instant::now();
        let _ctrb = sys_c.controllability_matrix::<$N>();
        let ctrb_time_ns = t_ctrb.elapsed().as_nanos() as f64;

        let t_obsv = Instant::now();
        let _obsv = sys_c.observability_matrix::<$N>();
        let obsv_time_ns = t_obsv.elapsed().as_nanos() as f64;

        (zoh_time_ns, ctrb_time_ns, obsv_time_ns)
    }};
}

fn benchmark_discretization_scaling() -> (Value, Vec<f64>, Vec<f64>) {
    let (zoh2, ctrb2, obsv2) = bench_dim!(2);
    let (zoh4, ctrb4, obsv4) = bench_dim!(4);
    let (zoh8, ctrb8, obsv8) = bench_dim!(8);
    let (zoh16, ctrb16, obsv16) = bench_dim!(16);
    let (zoh32, ctrb32, obsv32) = bench_dim!(32);
    let (zoh64, ctrb64, obsv64) = bench_dim!(64);
    let (zoh128, ctrb128, obsv128) = bench_dim!(128);

    let state_size = vec![2, 4, 8, 16, 32, 64, 128];
    let zoh_time_ns = vec![zoh2, zoh4, zoh8, zoh16, zoh32, zoh64, zoh128];
    let ctrb_time_ns =
        vec![ctrb2, ctrb4, ctrb8, ctrb16, ctrb32, ctrb64, ctrb128];
    let obsv_time_ns =
        vec![obsv2, obsv4, obsv8, obsv16, obsv32, obsv64, obsv128];

    let scaling_json = json!({
        "scaling": {
            "state_size": state_size,
            "zoh_time_ns": zoh_time_ns
        }
    });

    (scaling_json, ctrb_time_ns, obsv_time_ns)
}

fn benchmark_step_response_jitter() -> Value {
    let sim = PendulumSim::new(2.0, 0.8, 0.05);

    let mut x_k = Mat::<2, 1>::zero();
    let u_k = Mat::<1, 1>::from_fn(|_, _| 1.0);

    let iterations = 100;
    let mut step_compute_times_ns = Vec::with_capacity(iterations);
    let mut input = Vec::with_capacity(iterations);
    let mut step_data = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        let t_start = Instant::now();
        let (x_next, y_k) = sim.sys_d.step(&x_k, &u_k);
        let elapsed = t_start.elapsed().as_nanos() as f64;

        step_compute_times_ns.push(elapsed);
        input.push(1.0);
        let y_val = y_k.get(0, 0).copied().unwrap_or(0.0);
        step_data.push(y_val);

        x_k = x_next;
    }

    json!({
        "jitter": {
            "step_compute_times_ns": step_compute_times_ns,
            "input": input,
            "step_data": step_data.clone(),
            "step-data": step_data
        }
    })
}

fn track_control_loop_allocations(
    ctrb_time_ns: Vec<f64>,
    obsv_time_ns: Vec<f64>,
) -> Value {
    json!({
        "state_size": vec![2, 4, 8, 16, 32, 64, 128],
        "controllability_time_ns": ctrb_time_ns,
        "observability_time_ns": obsv_time_ns
    })
}

fn run_validation_default() -> Value {
    let q1 = generate_state_space_correctness_data();
    let (q2, ctrb_times, obsv_times) = benchmark_discretization_scaling();
    let q3 = benchmark_step_response_jitter();
    let q4 =
        track_control_loop_allocations(ctrb_times.clone(), obsv_times.clone());

    json!({
        "phase_portrait": q1["phase_portrait"],
        "scaling": q2["scaling"],
        "jitter": q3["jitter"],
        "control_loop": q4,
        "state_size": vec![2, 4, 8, 16, 32, 64, 128],
        "controllability_time_ns": ctrb_times,
        "observability_time_ns": obsv_times
    })
}

pub fn cross_validate(rust: &Value, python: &Value) -> Result<(), Vec<String>> {
    let mut errs = Vec::new();

    if python.as_object().map_or(true, |o| o.is_empty()) {
        errs.push("Python oracle returned an empty payload".to_string());
        return Err(errs);
    }

    match (
        rust["phase_portrait"]["theta"].as_array(),
        python["phase_portrait"]["theta"].as_array(),
    ) {
        (Some(r_th), Some(p_th)) => {
            if r_th.len() != p_th.len() || r_th.is_empty() {
                errs.push(format!(
                    "phase_portrait.theta length mismatch: rust {} vs python {}",
                    r_th.len(),
                    p_th.len()
                ));
            } else {
                for (i, (r, p)) in r_th.iter().zip(p_th.iter()).enumerate() {
                    let rv = r.as_f64().unwrap_or(0.0);
                    let pv = p.as_f64().unwrap_or(0.0);
                    if (rv - pv).abs() > 1e-6 {
                        errs.push(format!(
                            "phase_portrait.theta[{i}]: rust {rv} vs python {pv}"
                        ));
                    }
                }
            }
        }
        _ => {
            errs.push("Missing phase_portrait.theta in payload".to_string());
        }
    }

    match (
        rust["phase_portrait"]["theta_dot"].as_array(),
        python["phase_portrait"]["theta_dot"].as_array(),
    ) {
        (Some(r_thd), Some(p_thd)) => {
            if r_thd.len() != p_thd.len() || r_thd.is_empty() {
                errs.push(format!(
                    "phase_portrait.theta_dot length mismatch: rust {} vs python {}",
                    r_thd.len(),
                    p_thd.len()
                ));
            } else {
                for (i, (r, p)) in r_thd.iter().zip(p_thd.iter()).enumerate() {
                    let rv = r.as_f64().unwrap_or(0.0);
                    let pv = p.as_f64().unwrap_or(0.0);
                    if (rv - pv).abs() > 1e-6 {
                        errs.push(format!(
                            "phase_portrait.theta_dot[{i}]: rust {rv} vs python {pv}"
                        ));
                    }
                }
            }
        }
        _ => {
            errs.push("Missing phase_portrait.theta_dot in payload".to_string());
        }
    }

    match (
        rust["jitter"]["step_data"].as_array(),
        python["jitter"]["step_data"].as_array(),
    ) {
        (Some(r_sd), Some(p_sd)) => {
            if r_sd.len() != p_sd.len() || r_sd.is_empty() {
                errs.push(format!(
                    "jitter.step_data length mismatch: rust {} vs python {}",
                    r_sd.len(),
                    p_sd.len()
                ));
            } else {
                for (i, (r, p)) in r_sd.iter().zip(p_sd.iter()).enumerate() {
                    let rv = r.as_f64().unwrap_or(0.0);
                    let pv = p.as_f64().unwrap_or(0.0);
                    if (rv - pv).abs() > 1e-6 {
                        errs.push(format!(
                            "jitter.step_data[{i}]: rust {rv} vs python {pv}"
                        ));
                    }
                }
            }
        }
        _ => {
            errs.push("Missing jitter.step_data in payload".to_string());
        }
    }

    if errs.is_empty() { Ok(()) } else { Err(errs) }
}

/// Cross-validates against the harold oracle (State model + ZOH discretization via
/// harold.discretize, phase portrait via manual recursion on harold's discretized
/// matrices, step response via harold.simulate_linear_system).
pub fn cross_validate_harold(rust: &Value, harold: &Value) -> Result<(), Vec<String>> {
    let mut errs = Vec::new();

    if harold.as_object().map_or(true, |o| o.is_empty()) {
        errs.push("Harold oracle returned an empty payload".to_string());
        return Err(errs);
    }

    let check_series = |key: &str,
                        rust_arr: Option<&Vec<Value>>,
                        harold_arr: Option<&Vec<Value>>,
                        errs: &mut Vec<String>| {
        match (rust_arr, harold_arr) {
            (Some(r_arr), Some(h_arr)) => {
                if r_arr.len() != h_arr.len() || r_arr.is_empty() {
                    errs.push(format!(
                        "{key} length mismatch: rust {} vs harold {}",
                        r_arr.len(),
                        h_arr.len()
                    ));
                } else {
                    for (i, (r, h)) in r_arr.iter().zip(h_arr.iter()).enumerate() {
                        let rv = r.as_f64().unwrap_or(0.0);
                        let hv = h.as_f64().unwrap_or(0.0);
                        if (rv - hv).abs() > 1e-6 {
                            errs.push(format!("{key}[{i}]: rust {rv} vs harold {hv}"));
                        }
                    }
                }
            }
            _ => errs.push(format!("Missing {key} in payload")),
        }
    };

    check_series(
        "phase_portrait.theta",
        rust["phase_portrait"]["theta"].as_array(),
        harold["phase_portrait"]["theta"].as_array(),
        &mut errs,
    );
    check_series(
        "phase_portrait.theta_dot",
        rust["phase_portrait"]["theta_dot"].as_array(),
        harold["phase_portrait"]["theta_dot"].as_array(),
        &mut errs,
    );
    check_series(
        "jitter.step_data",
        rust["jitter"]["step_data"].as_array(),
        harold["jitter"]["step_data"].as_array(),
        &mut errs,
    );

    for key in ["state_size", "controllability_time_ns", "observability_time_ns"] {
        if harold[key].as_array().map_or(0, |a| a.len()) != 7 {
            errs.push(format!("Harold {key} does not have 7 entries"));
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

    let py_payload: Value = serde_json::from_slice(&py_output.stdout)
        .expect("Failed to parse Python JSON stdout");
    let py_results = py_payload["scipy"].clone();
    let harold_results = py_payload["harold"].clone();

    if let Err(errs) = cross_validate(&rust_results, &py_results) {
        eprintln!("State-Space Cross-Validation Errors (scipy):");
        for e in &errs {
            eprintln!("  - {e}");
        }
        std::process::exit(1);
    }

    if let Err(errs) = cross_validate_harold(&rust_results, &harold_results) {
        eprintln!("State-Space Cross-Validation Errors (harold):");
        for e in &errs {
            eprintln!("  - {e}");
        }
        std::process::exit(1);
    }

    let combined_results = json!({
        "metadata": {
            "domain": "state_space",
            "timestamp": "2026-08-30T22:30:28-06:00"
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
    let out_path = out_dir.join("state_space.json");

    fs::write(
        &out_path,
        serde_json::to_string_pretty(&combined_results).unwrap(),
    )
    .expect("Failed to write results file");

    println!(
        "Success: State-Space cross-validation passed! Payload saved to {}",
        out_path.display()
    );

    combined_results
}

#[allow(dead_code)]
pub fn main() {
    run();
}
