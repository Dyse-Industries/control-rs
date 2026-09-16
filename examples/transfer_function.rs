//! Bode point and controllable canonical realization.
//!
//! Run with `cargo run --example transfer_function`.
#![allow(clippy::expect_used)]

use control_rs::transfer_function::ArrayTransferFunction;

/// Evaluate H(s) = 4 / (s² + 2s + 4) at ω = 2 rad/s.
fn main() {
    let h =
        ArrayTransferFunction::<f64, 1, 3>::continuous([4.0], [4.0, 2.0, 1.0]);
    let (mag, phase) = h.bode_point(2.0);
    let ss = h
        .to_controllable_canonical_form::<2>()
        .expect("controllable canonical form");

    println!("H(s) = 4 / (s^2 + 2s + 4)");
    println!("num (ascending) = {:?}", h.num_slice());
    println!("den (ascending) = {:?}", h.den_slice());
    println!("|H(j2)| = {mag:.6}");
    println!("∠H(j2) = {phase:.6} rad ({:.2} deg)", phase.to_degrees());
    println!("CCF A (rows) = {:?}", ss.a().to_rows());
    println!("CCF B (rows) = {:?}", ss.b().to_rows());
    println!("CCF C (rows) = {:?}", ss.c().to_rows());
    println!("CCF D (rows) = {:?}", ss.d().to_rows());
}
