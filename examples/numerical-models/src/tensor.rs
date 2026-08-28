//! Tensor demo. Copy this file and point `Store` at your backend.
//!
//! `Store::from_array` uses **column-major** literals for the 2-D grid. The
//! `from_raw` layout matches `ArrayTensor::from_raw` in the Python oracle.

use crate::{ABS_F32, native_artifact, save};
use control_rs::math::fixed_num::Quantized;
use control_rs::math::storage::ArrayStorage;
use control_rs::tensor::{Shape2D, Tensor};
use serde_json::json;

/// Swap this for a custom flat buffer.
type Store = ArrayStorage<f32, 3, 3>;
type Grid = Tensor<f32, Shape2D<3, 3>, Store>;
type Q7 = Quantized<i8, 7>;

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

    println!("\n--- Quantized Fixed-Point Q7 Simulation ---");
    let float_inputs = [-0.75f32, -0.25, 0.0, 0.25, 0.5, 0.75];
    println!(
        "{:<14}{:<10}{:<16}{:<16}",
        "Float Input", "Q7 Raw", "Dequantized", "ReLU Output"
    );

    let mut q_raw = [0i32; 6];
    let mut dequant = [0.0f32; 6];
    let mut relu_raw = [0i32; 6];
    let mut relu_dequant = [0.0f32; 6];
    for (idx, &f_in) in float_inputs.iter().enumerate() {
        let q = Q7::quantize(f64::from(f_in));
        q_raw[idx] = i32::from(q.raw());
        dequant[idx] = q.dequantize() as f32;
        let relu_raw_i8 = q.raw().max(0);
        relu_raw[idx] = i32::from(relu_raw_i8);
        relu_dequant[idx] = Q7::from_raw(relu_raw_i8).dequantize() as f32;
        println!(
            "{:<14.4}{:<10}{:<16.6}{:<16.6} (raw: {})",
            f_in, q_raw[idx], dequant[idx], relu_dequant[idx], relu_raw[idx]
        );
        assert!(relu_raw[idx] >= 0, "ReLU raw is negative");
    }

    let query_x0: [f32; 6] = test_points.map(|p| p[0]);
    let values = json!({
        "SAMPLES": samples,
        "Q_RAW": q_raw,
        "DEQUANT": dequant,
        "RELU_RAW": relu_raw,
        "RELU_DEQUANT": relu_dequant,
        "query_x0": query_x0,
    });
    let series = json!({
        "interp": { "x": query_x0, "y": samples },
    });
    save(
        "results/tensor/native.json",
        &native_artifact("tensor", values, series),
    );
}
