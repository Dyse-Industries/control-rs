//! Tensor numerical model control-rs-verification suite.
//!
//! Evaluates multi-dimensional tensor kernels:
//! 1. Multilinear interpolation manifold (3D saddle point $z = x^2 - y^2$)
//! 2. High-order tensor contraction ($A \times B \to C$)
//! 3. Fixed-point quantization precision boundaries (`Quantized<i8, 7>`)
//! 4. `TableActivation` tanh sweep

#![allow(
    missing_docs,
    clippy::arithmetic_side_effects,
    clippy::cast_precision_loss,
    clippy::indexing_slicing,
    clippy::suboptimal_flops,
    clippy::too_many_lines,
    clippy::unwrap_used
)]

use std::path::Path;

use control_rs::tensor::{Activation, ArrayTensor, Quantized, TableActivation};

use crate::h5_writer::H5Writer;

type Tensor16x16 = ArrayTensor<f32, 16, 16>;

fn compute_interpolation_mesh() -> Vec<f64> {
    let center = 7.5_f64;
    let scale = 3.75_f64;

    let grid = Tensor16x16::from_fn(|idx| {
        let x = (idx[0] as f64 - center) / scale;
        let y = (idx[1] as f64 - center) / scale;
        (x * x - y * y) as f32
    });

    const EVAL_N: usize = 20;
    let mut out = Vec::with_capacity(EVAL_N * EVAL_N);
    for i in 0..EVAL_N {
        for j in 0..EVAL_N {
            let u = 15.0_f32 * (i as f32) / ((EVAL_N - 1) as f32);
            let v = 15.0_f32 * (j as f32) / ((EVAL_N - 1) as f32);
            let val = grid.interpolate(&[u, v]);
            out.push(val as f64);
        }
    }
    out
}

fn compute_tensor_contraction() -> Vec<f64> {
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

    let rows = tensor_c.to_row_arrays();
    let mut out = Vec::with_capacity(256);
    for row in &rows {
        for &val in row {
            out.push(val as f64);
        }
    }
    out
}

fn compute_quantized_boundaries() -> Vec<f64> {
    type Q7 = Quantized<i8, 7>;
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

    let float_inputs = [
        -1.5_f32, -1.0, -0.75, -0.5, -0.125, -0.0078125, 0.0, 0.0078125, 0.125,
        0.5, 0.75, 0.9921875, 1.0, 1.5,
    ];

    let mut outputs = Vec::with_capacity(float_inputs.len());
    for &x in &float_inputs {
        let y = tanh_lut.apply(x);
        let q_y = Q7::quantize(f64::from(y));
        outputs.push(q_y.dequantize());
    }

    outputs
}

/// Raw `Quantized<i8, 7>` representation of the boundary inputs.
fn compute_q7_raw() -> Vec<f64> {
    type Q7 = Quantized<i8, 7>;
    let float_inputs = [
        -1.5_f32, -1.0, -0.75, -0.5, -0.125, -0.0078125, 0.0, 0.0078125, 0.125,
        0.5, 0.75, 0.9921875, 1.0, 1.5,
    ];
    float_inputs
        .iter()
        .map(|&x| f64::from(Q7::quantize(f64::from(x)).raw()))
        .collect()
}

/// `TableActivation` tanh (61 breakpoints on [-3, 3]) evaluated at 121 points.
fn compute_activation_sweep() -> Vec<f64> {
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
    (0..121)
        .map(|i| f64::from(tanh_lut.apply(-3.0f32 + (i as f32) * 0.05f32)))
        .collect()
}

/// Executes the tensor control-rs-verification kernel and writes `target/verification/tensor.rust.h5`.
pub fn emit_container(output_path: &Path) -> Result<(), String> {
    let mut writer = H5Writer::new();

    let interp = compute_interpolation_mesh();
    writer.add_dataset("manifold/interp_mesh", &interp);
    writer.set_tolerance("manifold/interp_mesh", "abs", 0.05);

    let mat_c = compute_tensor_contraction();
    writer.add_dataset("contraction/mat_c", &mat_c);
    writer.set_tolerance("contraction/mat_c", "abs", 2e-4);

    let act = compute_quantized_boundaries();
    writer.add_dataset("boundaries/act_outputs", &act);
    writer.set_tolerance("boundaries/act_outputs", "abs", 0.02);

    writer.add_dataset("boundaries/q_raw", &compute_q7_raw());
    writer.set_tolerance("boundaries/q_raw", "abs", 0.0);

    writer.add_dataset("activation/act_outputs", &compute_activation_sweep());
    writer.set_tolerance("activation/act_outputs", "abs", 1e-3);

    writer.write_to_file(output_path)
}
