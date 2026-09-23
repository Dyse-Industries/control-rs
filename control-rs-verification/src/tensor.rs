//! Tensor numerical model control-rs-verification suite.
//!
//! Evaluates multi-dimensional tensor kernels:
//! 1. Multilinear interpolation manifold (3D saddle point $z = x^2 - y^2$)
//! 2. High-order tensor contraction ($A \times B \to C$)
//! 3. Fixed-point quantization precision boundaries (`Quantized<i8, 7>`)
//! 4. `TableActivation` tanh sweep

#![allow(missing_docs)]

use std::path::Path;

use control_rs::tensor::{Activation, ArrayTensor, Quantized, TableActivation};

use crate::h5_writer::H5Writer;
use crate::numeric::{KernelResult, index_f32};

/// Interpolation queries per grid axis.
const EVAL_N: usize = 20;

/// Inputs straddling the `Quantized<i8, 7>` range and resolution limits.
const Q7_BOUNDARY_INPUTS: [f32; 14] = [
    -1.5,
    -1.0,
    -0.75,
    -0.5,
    -0.125,
    -0.007_812_5,
    0.0,
    0.007_812_5,
    0.125,
    0.5,
    0.75,
    0.992_187_5,
    1.0,
    1.5,
];

type Q7 = Quantized<i8, 7>;
type Tensor16x16 = ArrayTensor<f32, 16, 16>;

/// Row and column index of a rank-2 tensor element.
const fn index2(idx: &[usize]) -> (usize, usize) {
    match *idx {
        [i, j, ..] => (i, j),
        [i] => (i, 0),
        [] => (0, 0),
    }
}

/// 61-point tanh lookup table on `[-3, 3]` with 0.1 spacing.
fn tanh_table() -> TableActivation<f32, 61> {
    let mut breakpoints = [0.0f32; 61];
    let mut values = [0.0f32; 61];
    for (i, (bp, v)) in breakpoints.iter_mut().zip(&mut values).enumerate() {
        let x = index_f32(i).mul_add(0.1, -3.0);
        *bp = x;
        *v = x.tanh();
    }
    TableActivation {
        breakpoints,
        values,
    }
}

fn compute_interpolation_mesh() -> Vec<f64> {
    let center = 7.5_f32;
    let scale = 3.75_f32;

    // Evaluated in `f32`, the grid's storage precision.
    let grid = Tensor16x16::from_fn(|idx| {
        let (i, j) = index2(idx);
        let x = (index_f32(i) - center) / scale;
        let y = (index_f32(j) - center) / scale;
        x.mul_add(x, -(y * y))
    });

    let last = index_f32(EVAL_N.saturating_sub(1));
    let mut out = Vec::with_capacity(EVAL_N.saturating_mul(EVAL_N));
    for i in 0..EVAL_N {
        for j in 0..EVAL_N {
            let u = 15.0_f32 * index_f32(i) / last;
            let v = 15.0_f32 * index_f32(j) / last;
            out.push(f64::from(grid.interpolate(&[u, v])));
        }
    }
    out
}

fn compute_tensor_contraction() -> Vec<f64> {
    let tensor_a = Tensor16x16::from_fn(|idx| {
        let (i, j) = index2(idx);
        index_f32(i).mul_add(0.5, index_f32(j) * 0.3).sin() * 10.0
    });

    let tensor_b = Tensor16x16::from_fn(|idx| {
        let (i, j) = index2(idx);
        index_f32(i).mul_add(0.3, -(index_f32(j) * 0.4)).cos() * 5.0
    });

    let mut tensor_c = Tensor16x16::zero();
    tensor_a.contract_into(&tensor_b, &mut tensor_c);

    tensor_c
        .to_row_arrays()
        .iter()
        .flatten()
        .map(|&val| f64::from(val))
        .collect()
}

fn compute_quantized_boundaries() -> Vec<f64> {
    let tanh_lut = tanh_table();
    Q7_BOUNDARY_INPUTS
        .iter()
        .map(|&x| Q7::quantize(f64::from(tanh_lut.apply(x))).dequantize())
        .collect()
}

/// Raw `Quantized<i8, 7>` representation of the boundary inputs.
fn compute_q7_raw() -> Vec<f64> {
    Q7_BOUNDARY_INPUTS
        .iter()
        .map(|&x| f64::from(Q7::quantize(f64::from(x)).raw()))
        .collect()
}

/// `TableActivation` tanh (61 breakpoints on [-3, 3]) evaluated at 121 points.
fn compute_activation_sweep() -> Vec<f64> {
    let tanh_lut = tanh_table();
    (0..121)
        .map(|i| f64::from(tanh_lut.apply(index_f32(i).mul_add(0.05, -3.0))))
        .collect()
}

/// Executes the tensor control-rs-verification kernel and writes `target/verification/tensor.rust.h5`.
///
/// # Errors
///
/// Returns an error if the container cannot be written.
pub fn emit_container(output_path: &Path) -> KernelResult<()> {
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
