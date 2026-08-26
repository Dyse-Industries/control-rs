//! Tensor Numerical Model Example
//!
//! Demonstrates 2D multilinear grid lookup table interpolation and
//! fixed-point Q7 quantized scalar arithmetic with `ReLU` activation.

#![allow(
    clippy::print_stdout,
    clippy::uninlined_format_args,
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::many_single_char_names,
    clippy::items_after_statements,
    clippy::type_complexity,
    clippy::doc_markdown
)]

use control_rs::tensor::{Activation, ArrayTensor, Quantized, Relu};

type Q7 = Quantized<i8, 7>;

fn main() {
    println!("=== Tensor Numerical Model Example ===");

    // 1. 2D Aerodynamic Calibration Grid Table (3x3)
    // Row 0: [0.0, 2.0, 4.0]
    // Row 1: [1.0, 3.0, 5.0]
    // Row 2: [2.0, 4.0, 6.0]
    let grid = ArrayTensor::<f32, 3, 3>::from_raw([
        [0.0, 2.0, 4.0],
        [1.0, 3.0, 5.0],
        [2.0, 4.0, 6.0],
    ]);

    println!("\n--- 2D Grid Table (3x3) ---");
    for i in 0..3 {
        print!("  [");
        for j in 0..3 {
            if j > 0 {
                print!(", ");
            }
            print!("{:6.1}", grid.get(&[j, i]).copied().unwrap_or(0.0));
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
    for pt in test_points {
        let val = grid.interpolate(&pt);
        println!("({:.2}, {:.2})      {:<20.6}", pt[0], pt[1], val);
    }

    // 2. Fixed-Point Q7 Quantized Arithmetic & ReLU Activation
    // Q7 format: scaling factor = 2^7 = 128
    let relu = Relu;

    println!("\n--- Quantized Fixed-Point Q7 Simulation ---");
    let float_inputs = [-0.75f32, -0.25, 0.0, 0.25, 0.5, 0.75];
    println!(
        "{:<14}{:<10}{:<16}{:<16}",
        "Float Input", "Q7 Raw", "Dequantized", "ReLU Output"
    );

    for f_in in float_inputs {
        let q = Q7::quantize(f_in);
        let dequant = q.dequantize();
        let q_relu = relu.apply(f_in);
        let q_raw = q.raw();
        let relu_raw = Q7::quantize(q_relu).raw();

        println!(
            "{:<14.4}{:<10}{:<16.6}{:<16.6} (raw: {})",
            f_in, q_raw, dequant, q_relu, relu_raw
        );
    }
}
