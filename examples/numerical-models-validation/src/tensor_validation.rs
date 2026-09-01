//! src/tensor_validation.rs
//!
//! Standalone validation runner for tensor and fixed-point numerical models.
//! Executes four core benchmark panels:
//! Panel 1: Multilinear Interpolation Manifold (3D Saddle Point z = x^2 - y^2)
//! Panel 2: Tensor Contraction Relative Error (ArrayTensor::contract_into matrix multiplication)
//! Panel 3: Quantized Precision Boundaries (Quantized<i8, 7> edge-case scaling & saturation)
//! Panel 4: Bare-Metal Timing Profile (Zero-copy stack vs dynamic heap allocation baselines)

use serde_json::{Value, json};
use std::fs;
use std::process::Command;
use std::time::Instant;

use control_rs::math::fixed_num::Quantized;
use control_rs::tensor::ArrayTensor;

type Tensor16x16 = ArrayTensor<f32, 16, 16>;
type Q7 = Quantized<i8, 7>;
pub type ValidationResult = Result<(), Vec<String>>;
type ContractionBenchResult = (Value, Tensor16x16, Tensor16x16, Tensor16x16);

const MESH_N: usize = 40;

// -----------------------------------------------------------------------------
// Panel 1: Multilinear Interpolation Manifold (3D Saddle Point z = x^2 - y^2)
// -----------------------------------------------------------------------------
fn benchmark_interpolation_manifold() -> (Value, Tensor16x16) {
    let center = 7.5_f64;
    let scale = 3.75_f64; // maps index [0, 15] to [-2.0, 2.0]

    // Construct 16x16 grid sampling z = x^2 - y^2 over [-2.0, 2.0]
    let grid = Tensor16x16::from_fn(|idx| {
        let x = (idx[0] as f64 - center) / scale;
        let y = (idx[1] as f64 - center) / scale;
        (x * x - y * y) as f32
    });

    let grid_table = (0..16)
        .map(|i| {
            (0..16)
                .map(|j| grid.get(&[i, j]).copied().unwrap_or(0.0))
                .collect::<Vec<f32>>()
        })
        .collect::<Vec<_>>();

    // Dense 40x40 fractional evaluation mesh over [0.0, 15.0]
    let mut interp_mesh = vec![vec![0.0_f32; MESH_N]; MESH_N];
    let mut exact_mesh = vec![vec![0.0_f32; MESH_N]; MESH_N];
    let mut mesh_u = vec![0.0_f32; MESH_N];
    let mut mesh_v = vec![0.0_f32; MESH_N];

    for i in 0..MESH_N {
        let u = 15.0_f64 * (i as f64) / ((MESH_N - 1) as f64);
        mesh_u[i] = u as f32;
        let x = (u - center) / scale;

        for j in 0..MESH_N {
            let v = 15.0_f64 * (j as f64) / ((MESH_N - 1) as f64);
            if i == 0 {
                mesh_v[j] = v as f32;
            }
            let y = (v - center) / scale;

            interp_mesh[i][j] = grid.interpolate(&[u as f32, v as f32]);
            exact_mesh[i][j] = (x * x - y * y) as f32;
        }
    }

    let payload = json!({
        "grid_table": grid_table,
        "mesh_u": mesh_u,
        "mesh_v": mesh_v,
        "interp_mesh": interp_mesh,
        "exact_mesh": exact_mesh,
    });

    (payload, grid)
}

// -----------------------------------------------------------------------------
// Panel 2: Tensor Contraction Relative Error (ArrayTensor::contract_into)
// -----------------------------------------------------------------------------
fn benchmark_tensor_contraction() -> ContractionBenchResult {
    let tensor_a = Tensor16x16::from_fn(|idx| {
        let i = idx[0] as f32;
        let j = idx[1] as f32;
        (i * 0.5 + j * 0.3).sin() * 10.0
    });

    let tensor_b = Tensor16x16::from_fn(|idx| {
        let i = idx[0] as f32;
        let j = idx[1] as f32;
        (i * 0.3 - j * 0.4).cos() * 5.0
    });

    let mut tensor_c = Tensor16x16::zero();
    tensor_a.contract_into(&tensor_b, &mut tensor_c);

    let mut mat_a = vec![vec![0.0_f32; 16]; 16];
    let mut mat_b = vec![vec![0.0_f32; 16]; 16];
    let mut mat_c = vec![vec![0.0_f32; 16]; 16];

    for i in 0..16 {
        for j in 0..16 {
            mat_a[i][j] = tensor_a.get(&[i, j]).copied().unwrap_or(0.0);
            mat_b[i][j] = tensor_b.get(&[i, j]).copied().unwrap_or(0.0);
            mat_c[i][j] = tensor_c.get(&[i, j]).copied().unwrap_or(0.0);
        }
    }

    let payload = json!({
        "mat_a": mat_a,
        "mat_b": mat_b,
        "mat_c": mat_c,
    });

    (payload, tensor_a, tensor_b, tensor_c)
}

// -----------------------------------------------------------------------------
// Panel 3: Quantized Precision Boundaries (Quantized<i8, 7>)
// -----------------------------------------------------------------------------
fn benchmark_quantized_boundaries() -> Value {
    use control_rs::tensor::{Activation, TableActivation};

    let float_inputs = [
        -1.5_f32, -1.0, -0.75, -0.5, -0.125, -0.0078125, 0.0, 0.0078125, 0.125,
        0.5, 0.75, 0.9921875, 1.0, 1.5,
    ];

    let n = float_inputs.len();
    let mut q_raw = vec![0i32; n];
    let mut dequant = vec![0.0_f32; n];
    let mut quant_err = vec![0.0_f32; n];

    for (idx, &f_in) in float_inputs.iter().enumerate() {
        let q = Q7::quantize(f_in as f64);
        q_raw[idx] = i32::from(q.raw());
        let dq = q.dequantize() as f32;
        dequant[idx] = dq;
        quant_err[idx] = (f_in - dq).abs();
    }

    // Sweep for TableActivation Tanh validation (61 breakpoints, 121 evaluation points)
    let mut breakpoints = [0.0f32; 61];
    let mut values = [0.0f32; 61];
    for i in 0..61 {
        let x = -3.0f32 + (i as f32) * 0.1f32;
        breakpoints[i] = x;
        values[i] = x.tanh();
    }
    let tanh_lut = TableActivation {
        breakpoints,
        values,
    };

    let mut act_inputs = vec![0.0f32; 121];
    let mut act_outputs = vec![0.0f32; 121];
    let mut act_outputs_q_raw = vec![0i32; 121];

    for i in 0..121 {
        let x = -3.0f32 + (i as f32) * 0.05f32;
        act_inputs[i] = x;
        let y = tanh_lut.apply(x);
        act_outputs[i] = y;

        // Quantize back to Q7 to compare against TFLite's int8 outputs
        let q_y = Q7::quantize(y as f64);
        act_outputs_q_raw[i] = i32::from(q_y.raw());
    }

    json!({
        "float_inputs": float_inputs,
        "q_raw": q_raw,
        "dequant": dequant,
        "quant_err": quant_err,
        "act_inputs": act_inputs,
        "act_outputs": act_outputs,
        "act_outputs_q_raw": act_outputs_q_raw,
    })
}

// -----------------------------------------------------------------------------
// Panel 4: Bare-Metal Timing Profile (Zero-Copy Stack vs Baselines)
// -----------------------------------------------------------------------------
fn benchmark_contract_n<const N: usize>(iters: usize) -> f64
where
    control_rs::math::num_types::Const<N>: control_rs::math::num_types::Dim,
{
    let a = ArrayTensor::<f32, N, N>::from_fn(|idx| {
        (idx[0] as f32 * 0.5 + idx[1] as f32 * 0.3).sin() * 10.0
    });
    let b = ArrayTensor::<f32, N, N>::from_fn(|idx| {
        (idx[0] as f32 * 0.3 - idx[1] as f32 * 0.4).cos() * 5.0
    });
    let mut c = ArrayTensor::<f32, N, N>::zero();

    let start = Instant::now();
    for _ in 0..iters {
        a.contract_into(&b, &mut c);
    }
    (start.elapsed().as_nanos() as f64) / (iters as f64)
}

fn benchmark_timing_profile(grid: &Tensor16x16) -> Value {
    let interp_iters = 100_000;
    let point = [7.3_f32, 8.1_f32];
    let t_interp = Instant::now();
    for _ in 0..interp_iters {
        let _ = core::hint::black_box(grid.interpolate(&point));
    }
    let interp_time_ns =
        (t_interp.elapsed().as_nanos() as f64) / (interp_iters as f64);

    let quant_iters = 100_000;
    let t_quant = Instant::now();
    for _ in 0..quant_iters {
        let _ =
            core::hint::black_box(Q7::quantize(core::f64::consts::FRAC_PI_4));
    }
    let quant_time_ns =
        (t_quant.elapsed().as_nanos() as f64) / (quant_iters as f64);

    let sizes = [4, 8, 16, 32, 64];
    let t_4 = benchmark_contract_n::<4>(100_000);
    let t_8 = benchmark_contract_n::<8>(50_000);
    let t_16 = benchmark_contract_n::<16>(20_000);
    let t_32 = benchmark_contract_n::<32>(5_000);
    let t_64 = benchmark_contract_n::<64>(1_000);

    let contract_times_ns = [t_4, t_8, t_16, t_32, t_64];

    json!({
        "interp_time_ns": interp_time_ns,
        "quant_time_ns": quant_time_ns,
        "sizes": sizes,
        "contract_times_ns": contract_times_ns,
    })
}

fn run_validation_default() -> Value {
    let (manifold, grid) = benchmark_interpolation_manifold();
    let (contraction, _tensor_a, _tensor_b, _tensor_c) =
        benchmark_tensor_contraction();
    let boundaries = benchmark_quantized_boundaries();
    let timing = benchmark_timing_profile(&grid);

    json!({
        "manifold": manifold,
        "contraction": contraction,
        "boundaries": boundaries,
        "timing": timing,
    })
}

pub fn cross_validate(rust: &Value, python: &Value) -> ValidationResult {
    let mut errs = Vec::new();

    if python.as_object().is_none_or(|o| o.is_empty()) {
        errs.push("Python oracle returned an empty payload".to_string());
        return Err(errs);
    }

    // 1. Manifold interpolation sample agreement
    match (
        rust["manifold"]["interp_mesh"].as_array(),
        python["manifold"]["interp_mesh"].as_array(),
    ) {
        (Some(r_mesh), Some(p_mesh)) => {
            if r_mesh.len() != p_mesh.len() || r_mesh.len() != MESH_N {
                errs.push(format!(
                    "manifold interp_mesh row count mismatch: rust {} vs python {} (expected {MESH_N})",
                    r_mesh.len(),
                    p_mesh.len()
                ));
            } else {
                for (i, (r_row, p_row)) in
                    r_mesh.iter().zip(p_mesh.iter()).enumerate()
                {
                    match (r_row.as_array(), p_row.as_array()) {
                        (Some(r_vals), Some(p_vals)) => {
                            if r_vals.len() != p_vals.len()
                                || r_vals.len() != MESH_N
                            {
                                errs.push(format!(
                                    "manifold interp_mesh[{i}] col count mismatch: rust {} vs python {} (expected {MESH_N})",
                                    r_vals.len(),
                                    p_vals.len()
                                ));
                            } else {
                                for (j, (rv, pv)) in
                                    r_vals.iter().zip(p_vals.iter()).enumerate()
                                {
                                    let r_num = rv.as_f64().unwrap_or(0.0);
                                    let p_num = pv.as_f64().unwrap_or(0.0);
                                    if (r_num - p_num).abs() > 1e-4 {
                                        errs.push(format!("manifold interp[{i}][{j}]: rust {r_num} vs python {p_num}"));
                                    }
                                }
                            }
                        }
                        _ => {
                            errs.push(format!(
                                "Missing manifold interp_mesh[{i}] row array in payload"
                            ));
                        }
                    }
                }
            }
        }
        _ => {
            errs.push(
                "Missing manifold.interp_mesh array in payload".to_string(),
            );
        }
    }

    // 2. Contraction relative error agreement
    match (
        rust["contraction"]["mat_c"].as_array(),
        python["contraction"]["mat_c"].as_array(),
    ) {
        (Some(r_mat), Some(p_mat)) => {
            if r_mat.len() != p_mat.len() || r_mat.len() != 16 {
                errs.push(format!(
                    "contraction mat_c row count mismatch: rust {} vs python {} (expected 16)",
                    r_mat.len(),
                    p_mat.len()
                ));
            } else {
                for (i, (r_row, p_row)) in
                    r_mat.iter().zip(p_mat.iter()).enumerate()
                {
                    match (r_row.as_array(), p_row.as_array()) {
                        (Some(r_vals), Some(p_vals)) => {
                            if r_vals.len() != p_vals.len()
                                || r_vals.len() != 16
                            {
                                errs.push(format!(
                                    "contraction mat_c[{i}] col count mismatch: rust {} vs python {} (expected 16)",
                                    r_vals.len(),
                                    p_vals.len()
                                ));
                            } else {
                                for (j, (rv, pv)) in
                                    r_vals.iter().zip(p_vals.iter()).enumerate()
                                {
                                    let r_num = rv.as_f64().unwrap_or(0.0);
                                    let p_num = pv.as_f64().unwrap_or(0.0);
                                    let rel_err = (r_num - p_num).abs()
                                        / (p_num.abs() + 1e-12);
                                    if rel_err > 1e-4 {
                                        errs.push(format!("contraction mat_c[{i}][{j}]: rust {r_num} vs python {p_num} (rel {rel_err})"));
                                    }
                                }
                            }
                        }
                        _ => {
                            errs.push(format!(
                                "Missing contraction mat_c[{i}] row array in payload"
                            ));
                        }
                    }
                }
            }
        }
        _ => {
            errs.push("Missing contraction.mat_c array in payload".to_string());
        }
    }

    // 3. Q7 Raw fixed-point byte exactness
    match (
        rust["boundaries"]["q_raw"].as_array(),
        python["boundaries"]["q_raw"].as_array(),
    ) {
        (Some(r_raw), Some(p_raw)) => {
            if r_raw.len() != p_raw.len() || r_raw.is_empty() {
                errs.push(format!(
                    "boundaries q_raw length mismatch: rust {} vs python {}",
                    r_raw.len(),
                    p_raw.len()
                ));
            } else {
                for (i, (r, p)) in r_raw.iter().zip(p_raw.iter()).enumerate() {
                    let rv = r.as_i64().unwrap_or(0);
                    let pv = p.as_i64().unwrap_or(0);
                    if rv != pv {
                        errs.push(format!(
                            "q7 q_raw[{i}]: rust {rv} vs python {pv}"
                        ));
                    }
                }
            }
        }
        _ => {
            errs.push("Missing boundaries.q_raw array in payload".to_string());
        }
    }

    // 4. TableActivation float closeness to SciPy exact Tanh
    match (
        rust["boundaries"]["act_outputs"].as_array(),
        python["boundaries"]["act_exact"].as_array(),
    ) {
        (Some(r_act), Some(p_act)) => {
            if r_act.len() != p_act.len() || r_act.is_empty() {
                errs.push(format!(
                    "boundaries act_outputs length mismatch: rust {} vs python {}",
                    r_act.len(),
                    p_act.len()
                ));
            } else {
                for (i, (r, p)) in r_act.iter().zip(p_act.iter()).enumerate() {
                    let rv = r.as_f64().unwrap_or(0.0);
                    let pv = p.as_f64().unwrap_or(0.0);
                    let diff = (rv - pv).abs();
                    if diff > 1e-3 {
                        errs.push(format!("TableActivation vs SciPy index {i}: rust {rv} vs python {pv} (diff {diff} exceeds 1e-3)"));
                    }
                }
            }
        }
        _ => {
            errs.push(
                "Missing act_outputs or act_exact in payload".to_string(),
            );
        }
    }

    // 5. TableActivation float closeness to TFLite interpreter (in dequantized float space)
    match (
        rust["boundaries"]["act_outputs"].as_array(),
        python["boundaries"]["tflite_dequant"].as_array(),
    ) {
        (Some(r_act), Some(t_dequant)) => {
            if r_act.len() != t_dequant.len() || r_act.is_empty() {
                errs.push(format!(
                    "boundaries act_outputs vs tflite_dequant length mismatch: rust {} vs python {}",
                    r_act.len(),
                    t_dequant.len()
                ));
            } else {
                for (i, (r, t)) in
                    r_act.iter().zip(t_dequant.iter()).enumerate()
                {
                    let rv = r.as_f64().unwrap_or(0.0);
                    let tv = t.as_f64().unwrap_or(0.0);
                    let diff = (rv - tv).abs();
                    // int8 quantization resolution is ~0.008-0.02, plus linear LUT approximation error
                    if diff > 0.05 {
                        errs.push(format!(
                            "TableActivation vs TFLite index {i}: rust {rv} vs tflite dequant {tv} (diff {diff} exceeds 0.05)"
                        ));
                    }
                }
            }
        }
        _ => {
            errs.push(
                "Missing act_outputs or tflite_dequant in payload".to_string(),
            );
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

    let py_payload: Value = serde_json::from_slice(&py_output.stdout)
        .expect("Failed to parse Python JSON stdout");
    let py_results = if py_payload.get("scipy").is_some() {
        py_payload["scipy"].clone()
    } else {
        py_payload.clone()
    };

    if let Err(errs) = cross_validate(&rust_results, &py_results) {
        eprintln!("Tensor Cross-Validation Errors:");
        for e in &errs {
            eprintln!("  - {e}");
        }
        std::process::exit(1);
    }

    let combined_results = json!({
        "metadata": {
            "domain": "tensor",
            "timestamp": chrono::Utc::now().to_rfc3339()
        },
        "sources": {
            "rust": {
                "default": rust_results
            },
            "python3": {
                "scipy": py_results
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
    let out_path = out_dir.join("tensor.json");

    fs::write(
        &out_path,
        serde_json::to_string_pretty(&combined_results).unwrap(),
    )
    .expect("Failed to write results file");

    println!(
        "Success: Tensor cross-validation passed!\nResults saved to {}",
        out_path.display()
    );

    combined_results
}

#[allow(dead_code)]
pub fn main() {
    run();
}
