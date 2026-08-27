//! State-Space Numerical Model Example
//!
//! Demonstrates continuous-time dynamical system modeling, Zero-Order Hold (ZOH)
//! discretization via matrix exponential, discrete simulation, and similarity transformations.

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

use control_rs::math::num_types::{Const, Dim};
use control_rs::matrix::Owned;
use control_rs::state_space::ArrayStateSpace;

fn print_matrix<const R: usize, const C: usize>(
    name: &str,
    m: &Owned<f64, R, C>,
) where
    Const<R>: Dim,
    Const<C>: Dim,
{
    println!("{name}:");
    for i in 0..R {
        print!("  [");
        for j in 0..C {
            if j > 0 {
                print!(", ");
            }
            print!("{:12.6}", m.get(i, j).copied().unwrap_or(0.0));
        }
        println!("]");
    }
}

fn main() {
    println!("=== State-Space Numerical Model Example ===");

    // 1. 2nd-Order Continuous Spring-Mass-Damper System
    // \ddot{x} + 0.8 \dot{x} + 4 x = u
    // State: x_1 = pos, x_2 = vel
    // Column-major array layout: [[col0], [col1]]
    let a_c: Owned<f64, 2, 2> = Owned::from_array([[0.0, -4.0], [1.0, -0.8]]);
    let b_c: Owned<f64, 2, 1> = Owned::from_array([[0.0, 1.0]]);
    let c_c: Owned<f64, 1, 2> = Owned::from_array([[1.0], [0.0]]);
    let d_c: Owned<f64, 1, 1> = Owned::zero();

    let sys_c = ArrayStateSpace::continuous(a_c, b_c, c_c, d_c);

    println!("\n--- Continuous-Time System ---");
    print_matrix("A_c", &sys_c.a());
    print_matrix("B_c", &sys_c.b());
    print_matrix("C_c", &sys_c.c());
    print_matrix("D_c", &sys_c.d());

    let x_test: Owned<f64, 2, 1> = Owned::from_array([[1.0, 0.5]]);
    let u_test: Owned<f64, 1, 1> = Owned::zero();
    let (x_dot, y_test) = sys_c.derivative(&x_test, &u_test);

    print_matrix("x_dot at [1.0, 0.5]^T", &x_dot);
    println!(
        "y at [1.0, 0.5]^T: {:.6}",
        y_test.get(0, 0).copied().unwrap_or(0.0)
    );

    // 2. Exact ZOH Discretization for Ts = 0.05s
    let dt = 0.05;
    let sys_d = sys_c.to_discrete_zoh(dt);

    println!("\n--- Discrete-Time System (ZOH, Ts = {dt}s) ---");
    print_matrix("A_d", &sys_d.a());
    print_matrix("B_d", &sys_d.b());
    print_matrix("C_d", &sys_d.c());
    print_matrix("D_d", &sys_d.d());

    // 3. 20-step Discrete Unit Step Simulation (u[k] = 1.0, x[0] = 0)
    let num_steps = 20;
    let mut x_k: Owned<f64, 2, 1> = Owned::zero();
    let u_step: Owned<f64, 1, 1> = Owned::from_fn(|_, _| 1.0);

    println!("\n--- 20-Step Unit Step Trajectory ---");
    println!(
        "{:<6}{:<16}{:<16}{:<16}",
        "Step", "x_1 (pos)", "x_2 (vel)", "y (output)"
    );
    for k in 0..num_steps {
        let pos = x_k.get(0, 0).copied().unwrap_or(0.0);
        let vel = x_k.get(1, 0).copied().unwrap_or(0.0);
        let (x_next, y_k) = sys_d.step(&x_k, &u_step);
        let y_val = y_k.get(0, 0).copied().unwrap_or(0.0);
        println!("{k:<6}{pos:<16.8}{vel:<16.8}{y_val:<16.8}");
        x_k = x_next;
    }

    // 4. Similarity Coordinate Transformation: T = [[1, 1], [0, 1]]
    // Column-major: col0 = [1.0, 0.0], col1 = [1.0, 1.0]
    let t: Owned<f64, 2, 2> = Owned::from_array([[1.0, 0.0], [1.0, 1.0]]);
    if let Ok(sys_transformed) = sys_d.similarity_transform(&t) {
        println!("\n--- Transformed System (T = [[1, 1], [0, 1]]) ---");
        print_matrix("A_tilde", &sys_transformed.a());
        print_matrix("B_tilde", &sys_transformed.b());
        print_matrix("C_tilde", &sys_transformed.c());
    }
}
