//! Tensor demo. Copy this file and point `Store` at your backend.
//!
//! `Store::from_array` uses **column-major** literals for the 2-D grid. The
//! `from_raw` layout matches `ArrayTensor::from_raw` in the Python oracle.

use crate::suite::{
    case_inputs, emit_stdout, json_f64, json_f64_vec, json_rows, json_usize,
    require_usize,
};
use crate::{ABS_F32, native_artifact, time_kernel, timing_entry};
use control_rs::math::fixed_num::Quantized;
use control_rs::math::storage::ArrayStorage;
use control_rs::tensor::{Shape2D, Tensor};
use serde_json::{Value, json};

/// Swap this for a custom flat buffer.
type Store = ArrayStorage<f32, 3, 3>;
type Grid = Tensor<f32, Shape2D<3, 3>, Store>;
type CurvedStore = ArrayStorage<f32, 16, 16>;
type Curved = Tensor<f32, Shape2D<16, 16>, CurvedStore>;
type Q7 = Quantized<i8, 7>;

const CUT_N: usize = 64;
const CURVED_N: usize = 16;
const QUANT_LSB: f64 = 1.0 / 256.0;

fn f32_rows_storage<const R: usize, const C: usize>(
    v: &Value,
) -> ArrayStorage<f32, R, C> {
    let rows = json_rows(v);
    if rows.len() != R {
        eprintln!("table has {} rows, expected {R}", rows.len());
        std::process::exit(1);
    }
    let mut cols = [[0.0_f32; R]; C];
    for (i, row) in rows.iter().enumerate() {
        if row.len() != C {
            eprintln!("row {i} has {} cols, expected {C}", row.len());
            std::process::exit(1);
        }
        for (j, &val) in row.iter().enumerate() {
            cols[j][i] = val as f32;
        }
    }
    ArrayStorage::from_array(cols)
}

pub fn run(suite: &Value) {
    eprintln!("=== Tensor Numerical Model Example ===");

    let affine = case_inputs(suite, "tensor.host.affine_interp");
    let grid = Grid::from_storage(storage_from_rows_f32(&affine["table"]));

    eprintln!("\n--- 2D Grid Table (3x3) ---");
    for i in 0..3 {
        eprint!("  [");
        for j in 0..3 {
            if j > 0 {
                eprint!(", ");
            }
            eprint!("{:6.1}", grid.get(&[i, j]).copied().unwrap_or(0.0));
        }
        eprintln!("]");
    }

    let points = json_rows(&affine["points"]);
    if points.len() != 6 {
        eprintln!("expected 6 interpolation points, got {}", points.len());
        std::process::exit(1);
    }
    let mut test_points = [[0.0f32; 2]; 6];
    for (i, pt) in points.iter().enumerate() {
        test_points[i] = [pt[0] as f32, pt[1] as f32];
    }

    eprintln!("\n--- Multilinear Continuous Interpolation ---");
    eprintln!("{:<16}{:<20}", "(x, y)", "Interpolated Value");
    let mut samples = [0.0f32; 6];
    for (idx, pt) in test_points.iter().enumerate() {
        let val = grid.interpolate(pt);
        samples[idx] = val;
        eprintln!("({:.2}, {:.2})      {:<20.6}", pt[0], pt[1], val);
    }

    for i in 0..3 {
        for j in 0..3 {
            let node = [i as f32, j as f32];
            let interp = grid.interpolate(&node);
            let table = grid.get(&[i, j]).copied().unwrap();
            assert!(
                (interp - table).abs() <= ABS_F32,
                "grid node ({i},{j}): {interp} vs {table}"
            );
        }
    }

    let curved_in = case_inputs(suite, "tensor.host.curved_grid");
    require_usize(curved_in, "n", CURVED_N);
    require_usize(&curved_in["cut"], "n", CUT_N);
    eprintln!("\n--- 16x16 saddle u^2 - v^2 ---");
    let center = json_f64(&curved_in["center"]);
    let scale = json_f64(&curved_in["scale"]);
    let curved = Curved::from_fn(|idx| {
        let i = idx[0] as f64;
        let j = idx[1] as f64;
        let u = (i - center) / scale;
        let v = (j - center) / scale;
        (u * u - v * v) as f32
    });
    for &i in &[0usize, 15] {
        for &j in &[0usize, 15] {
            let node = [i as f32, j as f32];
            let interp = curved.interpolate(&node);
            let table = curved.get(&[i, j]).copied().unwrap();
            assert!(
                (interp - table).abs() <= ABS_F32,
                "curved node ({i},{j}): {interp} vs {table}"
            );
        }
    }
    let cut_start = json_f64(&curved_in["cut"]["start"]) as f32;
    let cut_stop = json_f64(&curved_in["cut"]["stop"]) as f32;
    let mut cut_x = vec![0.0_f32; CUT_N];
    let mut curved_samples = vec![0.0_f32; CUT_N];
    for k in 0..CUT_N {
        let t = cut_start
            + (cut_stop - cut_start) * (k as f32) / ((CUT_N - 1) as f32);
        cut_x[k] = t;
        curved_samples[k] = curved.interpolate(&[t, center as f32]);
    }
    let mut curved_table = Vec::with_capacity(16);
    for i in 0..16 {
        let mut row = Vec::with_capacity(16);
        for j in 0..16 {
            row.push(curved.get(&[i, j]).copied().unwrap());
        }
        curved_table.push(row);
    }
    let weiser = 0.125 * (2.0 / scale.powi(2) + 2.0 / scale.powi(2));
    eprintln!("Weiser bound (analytic): {weiser:.6e}");

    let timed = json_f64_vec(&curved_in["timed_point"]);
    let interior = [timed[0] as f32, timed[1] as f32];
    let interp_iters = json_usize(&curved_in["iters"]).unwrap_or(10_000) as u32;
    let interp_ns = time_kernel(interp_iters, || {
        curved.interpolate(&core::hint::black_box(interior))
    });
    eprintln!("interp min ns ({interp_iters} iters): {interp_ns}");

    let q7 = case_inputs(suite, "tensor.host.q7_relu");
    eprintln!("\n--- Quantized Fixed-Point Q7 ---");
    let float_inputs = json_f64_vec(&q7["float_inputs"]);
    if float_inputs.len() != 6 {
        eprintln!("expected 6 Q7 inputs, got {}", float_inputs.len());
        std::process::exit(1);
    }
    eprintln!(
        "{:<14}{:<10}{:<16}{:<16}",
        "Float Input", "Q7 Raw", "Dequantized", "ReLU Output"
    );

    let mut q_raw = [0i32; 6];
    let mut dequant = [0.0f32; 6];
    let mut relu_raw = [0i32; 6];
    let mut relu_dequant = [0.0f32; 6];
    let mut quant_err = 0.0_f64;
    for (idx, &f_in) in float_inputs.iter().enumerate() {
        let q = Q7::quantize(f_in);
        q_raw[idx] = i32::from(q.raw());
        dequant[idx] = q.dequantize() as f32;
        let relu_raw_i8 = q.raw().max(0);
        relu_raw[idx] = i32::from(relu_raw_i8);
        relu_dequant[idx] = Q7::from_raw(relu_raw_i8).dequantize() as f32;
        quant_err = quant_err.max((f_in - q.dequantize()).abs());
        eprintln!(
            "{:<14.4}{:<10}{:<16.6}{:<16.6} (raw: {})",
            f_in, q_raw[idx], dequant[idx], relu_dequant[idx], relu_raw[idx]
        );
        assert!(relu_raw[idx] >= 0, "ReLU raw is negative");
    }
    assert!(
        quant_err <= QUANT_LSB,
        "Q7 round-trip {quant_err} exceeds {QUANT_LSB}"
    );

    let values = json!({
        "SAMPLES": samples,
        "CURVED_SAMPLES": curved_samples,
        "CURVED_TABLE": curved_table,
        "CUT_X": cut_x,
        "Q_RAW": q_raw,
        "DEQUANT": dequant,
        "RELU_RAW": relu_raw,
        "RELU_DEQUANT": relu_dequant,
    });
    let series = json!({
        "interp": {
            "x": test_points.iter().map(|p| f64::from(p[0])).collect::<Vec<_>>(),
            "y": samples
        },
        "curved": { "x": cut_x, "y": curved_samples },
    });
    let metrics = json!({
        "quant_roundtrip_max": quant_err,
        "weiser_bound": weiser,
    });
    let timings = json!({
        "interp": timing_entry(interp_iters, interp_ns),
    });
    emit_stdout(&native_artifact("tensor", values, series, metrics, timings));
}

fn storage_from_rows_f32(v: &Value) -> ArrayStorage<f32, 3, 3> {
    f32_rows_storage::<3, 3>(v)
}
