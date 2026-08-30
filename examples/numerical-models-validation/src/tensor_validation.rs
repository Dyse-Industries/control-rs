//! src/tensor_validation.rs
//!
//! Standalone validation runner for tensor and fixed-point numerical models.
//! Computes outputs natively in Rust, spawns Python oracle subprocess,
//! and emits the combined results payload to results/tensor.json.

use serde_json::{json, Value};
use std::fs;
use std::process::Command;
use std::time::Instant;

use control_rs::math::fixed_num::Quantized;
use control_rs::math::storage::ArrayStorage;
use control_rs::tensor::{Shape2D, Tensor};

type Store = ArrayStorage<f32, 3, 3>;
type Grid = Tensor<f32, Shape2D<3, 3>, Store>;
type CurvedStore = ArrayStorage<f32, 16, 16>;
type Curved = Tensor<f32, Shape2D<16, 16>, CurvedStore>;
type Q7 = Quantized<i8, 7>;

const CUT_N: usize = 64;

fn run_validation_default() -> Value {
    let grid_store = ArrayStorage::from_array([
        [0.0_f32, 2.0_f32, 4.0_f32],
        [1.0_f32, 3.0_f32, 5.0_f32],
        [2.0_f32, 4.0_f32, 6.0_f32],
    ]);
    let grid = Grid::from_storage(grid_store);

    let test_points: [[f32; 2]; 6] = [
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [0.5, 0.5],
        [1.5, 0.5],
        [0.2, 1.8],
    ];
    let mut samples_affine = [0.0_f32; 6];

    let t_affine = Instant::now();
    for (idx, pt) in test_points.iter().enumerate() {
        samples_affine[idx] = grid.interpolate(pt);
    }
    let affine_interp_time_ns = t_affine.elapsed().as_nanos() as f64;

    let center = 7.5_f64;
    let scale = 7.5_f64;
    let curved = Curved::from_fn(|idx| {
        let i = idx[0] as f64;
        let j = idx[1] as f64;
        let u = (i - center) / scale;
        let v = (j - center) / scale;
        (u * u - v * v) as f32
    });

    let cut_start = 0.0_f32;
    let cut_stop = 15.0_f32;
    let mut cut_x = vec![0.0_f32; CUT_N];
    let mut curved_samples = vec![0.0_f32; CUT_N];
    for k in 0..CUT_N {
        let t = cut_start + (cut_stop - cut_start) * (k as f32) / ((CUT_N - 1) as f32);
        cut_x[k] = t;
        curved_samples[k] = curved.interpolate(&[t, center as f32]);
    }

    let mut curved_table = Vec::with_capacity(16);
    for i in 0..16 {
        let mut row = Vec::with_capacity(16);
        for j in 0..16 {
            row.push(curved.get(&[i, j]).copied().unwrap_or(0.0));
        }
        curved_table.push(row);
    }
    let weiser_bound = 0.125 * (2.0 / scale.powi(2) + 2.0 / scale.powi(2));

    let interior = [7.3_f32, 8.1_f32];
    let interp_iters = 10_000;
    let start_interp = Instant::now();
    for _ in 0..interp_iters {
        let _ = core::hint::black_box(curved.interpolate(&interior));
    }
    let interp_time_ns = start_interp.elapsed().as_nanos() as f64;

    let float_inputs = [
        0.7853981633974483,
        0.3333333333333333,
        0.7182818284590451,
        -0.75,
        0.0,
        0.5,
    ];
    let mut q_raw = [0i32; 6];
    let mut dequant = [0.0f32; 6];
    let mut relu_raw = [0i32; 6];
    let mut relu_dequant = [0.0f32; 6];
    let mut quant_err = 0.0_f64;

    let t_q7 = Instant::now();
    for (idx, &f_in) in float_inputs.iter().enumerate() {
        let q = Q7::quantize(f_in);
        q_raw[idx] = i32::from(q.raw());
        dequant[idx] = q.dequantize() as f32;
        let relu_raw_i8 = q.raw().max(0);
        relu_raw[idx] = i32::from(relu_raw_i8);
        relu_dequant[idx] = Q7::from_raw(relu_raw_i8).dequantize() as f32;
        quant_err = quant_err.max((f_in - q.dequantize()).abs());
    }
    let q7_time_ns = t_q7.elapsed().as_nanos() as f64;

    json!({
        "affine": {
            "samples": samples_affine,
            "affine_interp_time_ns": affine_interp_time_ns,
        },
        "curved": {
            "table": curved_table,
            "cut_x": cut_x,
            "samples": curved_samples,
            "weiser_bound": weiser_bound,
            "interp_time_ns": interp_time_ns,
        },
        "q7": {
            "q_raw": q_raw,
            "dequant": dequant,
            "relu_raw": relu_raw,
            "relu_dequant": relu_dequant,
            "quant_err": quant_err,
            "q7_time_ns": q7_time_ns,
        }
    })
}

pub fn cross_validate(rust: &Value, python: &Value) -> Result<(), Vec<String>> {
    let mut errs = Vec::new();

    if let (Some(r_samples), Some(p_samples)) = (
        rust["affine"]["samples"].as_array(),
        python["affine"]["samples"].as_array(),
    ) {
        for (i, (r, p)) in r_samples.iter().zip(p_samples.iter()).enumerate() {
            let rv = r.as_f64().unwrap_or(0.0);
            let pv = p.as_f64().unwrap_or(0.0);
            if (rv - pv).abs() > 1e-6 {
                errs.push(format!("affine sample[{i}]: rust {rv} vs python {pv}"));
            }
        }
    }

    if let (Some(r_samples), Some(p_samples)) = (
        rust["curved"]["samples"].as_array(),
        python["curved"]["samples"].as_array(),
    ) {
        for (i, (r, p)) in r_samples.iter().zip(p_samples.iter()).enumerate() {
            let rv = r.as_f64().unwrap_or(0.0);
            let pv = p.as_f64().unwrap_or(0.0);
            if (rv - pv).abs() > 1e-4 {
                errs.push(format!("curved sample[{i}]: rust {rv} vs python {pv}"));
            }
        }
    }

    if let (Some(r_raw), Some(p_raw)) = (
        rust["q7"]["q_raw"].as_array(),
        python["q7"]["q_raw"].as_array(),
    ) {
        for (i, (r, p)) in r_raw.iter().zip(p_raw.iter()).enumerate() {
            let rv = r.as_i64().unwrap_or(0);
            let pv = p.as_i64().unwrap_or(0);
            if rv != pv {
                errs.push(format!("q7 q_raw[{i}]: rust {rv} vs python {pv}"));
            }
        }
    }

    if errs.is_empty() { Ok(()) } else { Err(errs) }
}

pub fn run() -> Value {
    println!("Executing Rust tensor validator...");
    let rust_results = run_validation_default();

    println!("Spawning Python oracle subprocess...");
    let py_output = Command::new("python3")
        .arg("python3/tensor_validation.py")
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
        eprintln!("Tensor Cross-Validation Errors:");
        for e in &errs {
            eprintln!("  - {e}");
        }
        std::process::exit(1);
    }

    let combined_results = json!({
        "rust": rust_results,
        "python3": py_results
    });

    let out_dir = std::env::var("CARGO_MANIFEST_DIR")
        .map(|d| std::path::PathBuf::from(d).join("results"))
        .unwrap_or_else(|_| std::path::PathBuf::from("examples/numerical-models-validation/results"));

    fs::create_dir_all(&out_dir).expect("Failed to create results directory");
    let out_path = out_dir.join("tensor.json");

    fs::write(
        &out_path,
        serde_json::to_string_pretty(&combined_results).unwrap(),
    )
    .expect("Failed to write results file");

    println!(
        "Success: Tensor cross-validation passed! Payload saved to {}",
        out_path.display()
    );

    combined_results
}

#[allow(dead_code)]
pub fn main() {
    run();
}
