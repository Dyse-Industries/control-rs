//! Grid interpolation and optional Q7 quantization.
//!
//! Run with `cargo run --example tensor`.

use control_rs::tensor::{ArrayTensor, Quantized};

/// Signed 8-bit fixed point with 7 fractional bits.
type Q7 = Quantized<i8, 7>;

/// Look up a 2×2 table and quantize a scalar into Q7.
fn main() {
    let table = ArrayTensor::<f32, 2, 2>::from_cols([[1.0, 2.0], [3.0, 4.0]]);
    let value = table.interpolate(&[0.5, 0.5]);
    println!("2x2 table from_cols([[1, 2], [3, 4]])");
    println!("interpolate([0.5, 0.5]) = {value:.4}  (expected 2.5)");

    let q = Q7::quantize(0.75);
    println!(
        "Q7::quantize(0.75) raw = {}, dequant = {:.6}",
        q.raw(),
        q.dequantize()
    );
}
