//! Tensor demo. Copy this file and point `Store` at your backend.
//!
//! `Store::from_array` uses **column-major** literals for the 2-D grid. The
//! `from_raw` layout matches `ArrayTensor::from_raw` in the Python oracle.

use crate::{ABS_F32, native_artifact, save, time_kernel, timing_entry};
use control_rs::math::fixed_num::Quantized;
use control_rs::math::storage::ArrayStorage;
use control_rs::tensor::{Shape2D, Tensor};
use serde_json::json;

/// Swap this for a custom flat buffer.
type Store = ArrayStorage<f32, 3, 3>;
type Grid = Tensor<f32, Shape2D<3, 3>, Store>;
type CurvedStore = ArrayStorage<f32, 16, 16>;
type Curved = Tensor<f32, Shape2D<16, 16>, CurvedStore>;
type Q7 = Quantized<i8, 7>;

const CUT_N: usize = 64;
const INTERP_ITERS: u32 = 10_000;
const QUANT_LSB: f64 = 1.0 / 256.0;

pub fn main() {
    println!("=== Tensor Numerical Model Example ===");

    let grid = Grid::from_storage(Store::from_array([
        [0.0, 2.0, 4.0],
        [1.0, 3.0, 5.0],
        [2.0, 4.0, 6.0],
    ]));

    println!("\n--- 2D Grid Table (3x3) ---");
    for i in 0..3 {
        print!("  [");
        for j in 0..3 {
            if j > 0 {
                print!(", ");
            }
            print!("{:6.1}", grid.get(&[i, j]).copied().unwrap_or(0.0));
        }
        println!("]");
    }

    let test_points = [
        [0.0f32, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [0.5, 0.5],
        [1.5, 0.5],
        [0.2, 1.8],
    ];

    println!("\n--- Multilinear Continuous Interpolation ---");
    println!("{:<16}{:<20}", "(x, y)", "Interpolated Value");
    let mut samples = [0.0f32; 6];
    for (idx, pt) in test_points.iter().enumerate() {
        let val = grid.interpolate(pt);
        samples[idx] = val;
        println!("({:.2}, {:.2})      {:<20.6}", pt[0], pt[1], val);
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

    println!("\n--- 16x16 curved grid sin(πi/15) cos(πj/15) ---");
    // f64 π then store f32, matching `python3/tensor.py`.
    let pi = core::f64::consts::PI;
    let curved = Curved::from_fn(|idx| {
        let i = idx[0] as f64;
        let j = idx[1] as f64;
        ((pi * i / 15.0).sin() * (pi * j / 15.0).cos()) as f32
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
    let mut cut_x = vec![0.0_f32; CUT_N];
    let mut curved_samples = vec![0.0_f32; CUT_N];
    for k in 0..CUT_N {
        let t = 15.0 * (k as f32) / ((CUT_N - 1) as f32);
        cut_x[k] = t;
        curved_samples[k] = curved.interpolate(&[t, t]);
    }
    let mut curved_table = Vec::with_capacity(16);
    for i in 0..16 {
        let mut row = Vec::with_capacity(16);
        for j in 0..16 {
            row.push(curved.get(&[i, j]).copied().unwrap());
        }
        curved_table.push(row);
    }
    let weiser = 0.125 * 2.0 * (core::f64::consts::PI / 15.0).powi(2);
    println!("Weiser bound (analytic): {weiser:.6e}");

    let interior = [7.3_f32, 8.1];
    let interp_ns = time_kernel(INTERP_ITERS, || {
        curved.interpolate(&core::hint::black_box(interior))
    });
    println!("interp min ns ({INTERP_ITERS} iters): {interp_ns}");

    println!("\n--- Quantized Fixed-Point Q7 ---");
    let float_inputs = [
        core::f64::consts::FRAC_PI_4,
        1.0 / 3.0,
        core::f64::consts::E - 2.0,
        -0.75,
        0.0,
        0.5,
    ];
    println!(
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
        println!(
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
        "interp": { "x": [0.0, 1.0, 2.0, 0.5, 1.5, 0.2], "y": samples },
        "curved": { "x": cut_x, "y": curved_samples },
    });
    let metrics = json!({
        "quant_roundtrip_max": quant_err,
        "weiser_bound": weiser,
    });
    let timings = json!({
        "interp": timing_entry(INTERP_ITERS, interp_ns),
    });
    save(
        "results/tensor/native.json",
        &native_artifact("tensor", values, series, metrics, timings),
    );
}
