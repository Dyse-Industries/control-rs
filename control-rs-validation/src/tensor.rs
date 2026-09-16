//! Tensor suite: interpolation error, contraction drift and quantization.
//!
//! `f32` throughout, because that is where the error is visible: a bilinear
//! interpolation manifold sampled off-grid, a 16x16 contraction whose relative
//! error accumulates over 16 products per entry, and `Quantized<i8, 7>` at its
//! saturation and rounding boundaries. `examples/tensor.rs` shows the same API
//! on a 2x2 table where none of this matters.
//!
//! Timing lives in `benches/numerical_models.rs`, not here.

use serde_json::{Value, json};

use control_rs::math::fixed_num::Quantized;
use control_rs::tensor::ArrayTensor;

type Tensor16x16 = ArrayTensor<f32, 16, 16>;
type Q7 = Quantized<i8, 7>;
type ContractionBenchResult = (Value, Tensor16x16, Tensor16x16, Tensor16x16);

const MESH_N: usize = 40;

// -----------------------------------------------------------------------------
// Panel 1: Multilinear Interpolation Manifold (3D Saddle Point z = x^2 - y^2)
// -----------------------------------------------------------------------------
fn interpolation_manifold() -> (Value, Tensor16x16) {
    let center = 7.5_f64;
    let scale = 3.75_f64; // maps index [0, 15] to [-2.0, 2.0]

    // Construct 16x16 grid sampling z = x^2 - y^2 over [-2.0, 2.0]
    let grid = Tensor16x16::from_fn(|idx| {
        let x = (idx[0] as f64 - center) / scale;
        let y = (idx[1] as f64 - center) / scale;
        (x * x - y * y) as f32
    });

    let grid_table = grid.to_rows();

    // Dense 40x40 fractional evaluation mesh over [0.0, 15.0]
    let mut mesh_u = [0.0_f32; MESH_N];
    let mut mesh_v = [0.0_f32; MESH_N];

    let interp_mesh = ArrayTensor::<f32, MESH_N, MESH_N>::from_fn(|idx| {
        let i = idx[0];
        let j = idx[1];
        let u = 15.0_f64 * (i as f64) / ((MESH_N - 1) as f64);
        let v = 15.0_f64 * (j as f64) / ((MESH_N - 1) as f64);
        grid.interpolate(&[u as f32, v as f32])
    });

    let exact_mesh = ArrayTensor::<f32, MESH_N, MESH_N>::from_fn(|idx| {
        let i = idx[0];
        let j = idx[1];
        let u = 15.0_f64 * (i as f64) / ((MESH_N - 1) as f64);
        let v = 15.0_f64 * (j as f64) / ((MESH_N - 1) as f64);
        let x = (u - center) / scale;
        let y = (v - center) / scale;
        (x * x - y * y) as f32
    });

    for i in 0..MESH_N {
        let u = 15.0_f64 * (i as f64) / ((MESH_N - 1) as f64);
        mesh_u[i] = u as f32;
        let v = 15.0_f64 * (i as f64) / ((MESH_N - 1) as f64);
        mesh_v[i] = v as f32;
    }

    let interp_arr = interp_mesh.to_rows();
    let exact_arr = exact_mesh.to_rows();
    let interp_slices =
        core::array::from_fn::<_, MESH_N, _>(|i| &interp_arr[i][..]);
    let exact_slices =
        core::array::from_fn::<_, MESH_N, _>(|i| &exact_arr[i][..]);

    let payload = json!({
        "grid_table": grid_table,
        "mesh_u": &mesh_u[..],
        "mesh_v": &mesh_v[..],
        "interp_mesh": &interp_slices[..],
        "exact_mesh": &exact_slices[..],
    });

    (payload, grid)
}

// -----------------------------------------------------------------------------
// Panel 2: Tensor Contraction Relative Error (ArrayTensor::contract_into)
// -----------------------------------------------------------------------------
fn tensor_contraction() -> ContractionBenchResult {
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

    let mat_a = tensor_a.to_rows();
    let mat_b = tensor_b.to_rows();
    let mat_c = tensor_c.to_rows();

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
fn quantized_boundaries() -> Value {
    use control_rs::tensor::{Activation, TableActivation};

    let float_inputs = [
        -1.5_f32, -1.0, -0.75, -0.5, -0.125, -0.0078125, 0.0, 0.0078125, 0.125,
        0.5, 0.75, 0.9921875, 1.0, 1.5,
    ];

    let mut q_raw = [0i32; 14];
    let mut dequant = [0.0_f32; 14];
    let mut quant_err = [0.0_f32; 14];

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

    let mut act_inputs = [0.0f32; 121];
    let mut act_outputs = [0.0f32; 121];
    let mut act_outputs_q_raw = [0i32; 121];

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
        "float_inputs": &float_inputs[..],
        "q_raw": &q_raw[..],
        "dequant": &dequant[..],
        "quant_err": &quant_err[..],
        "act_inputs": &act_inputs[..],
        "act_outputs": &act_outputs[..],
        "act_outputs_q_raw": &act_outputs_q_raw[..],
    })
}

/// Assemble the full tensor-suite payload.
#[must_use]
pub fn payload() -> Value {
    let (manifold, _grid) = interpolation_manifold();
    let (contraction, _a, _b, _c) = tensor_contraction();

    json!({
        "manifold": manifold,
        "contraction": contraction,
        "boundaries": quantized_boundaries(),
    })
}

/// Datasets gated by the comparator; every other key is context only.
pub const GATED_PATHS: &[&str] = &[
    "manifold/interp_mesh",
    "contraction/mat_c",
    "boundaries/q_raw",
    "boundaries/act_outputs",
];

/// Emit `results/tensor.rust.h5`.
pub fn run() {
    println!("tensor: interpolation, contraction and quantization error");
    crate::write_rust_container("tensor", &payload(), GATED_PATHS);
}
