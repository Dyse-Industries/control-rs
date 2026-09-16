//! Continuous plant, ZOH discretization, and one discrete step.
//!
//! Run with `cargo run --example state_space`.
#![allow(clippy::expect_used)]

use control_rs::matrix::Owned;
use control_rs::state_space::ArrayStateSpace;

/// Discretize a 2-state SISO oscillator and step it from rest.
fn main() {
    let sys_c = ArrayStateSpace::<f64, 2, 1, 1>::continuous(
        [[0.0, 1.0], [-4.0, -0.8]],
        [[0.0], [1.0]],
        [[1.0, 0.0]],
        [[0.0]],
    );
    let sys_d = sys_c.to_discrete_zoh(0.05);
    let (x_next, y) = sys_d.step(
        &Owned::<f64, 2, 1>::zero(),
        &Owned::<f64, 1, 1>::scalar(1.0),
    );

    println!("Continuous A (rows) = {:?}", sys_c.a().to_rows());
    println!("ZOH Ts = 0.05 s");
    println!("Discrete A (rows) = {:?}", sys_d.a().to_rows());
    println!(
        "x_next = [{:.6}, {:.6}]",
        *x_next.get(0, 0).expect("x_next row 0"),
        *x_next.get(1, 0).expect("x_next row 1"),
    );
    println!("y = {:.6}", *y.get(0, 0).expect("y row 0"));
}
