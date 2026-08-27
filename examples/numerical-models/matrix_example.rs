//! Matrix Numerical Model Example
//!
//! Demonstrates matrix construction, arithmetic operations, transposition,
//! LU decomposition, linear system solving, and matrix inversion with identity verification.

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
use control_rs::matrix::{LuDecomposition, Owned};

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
    println!("=== Matrix Numerical Model Example ===");

    // 1. Matrix Construction and Arithmetic
    let m1: Owned<f64, 2, 2> = Owned::from_array([[1.0, 3.0], [2.0, 4.0]]); // col0=[1, 3], col1=[2, 4]
    let m2: Owned<f64, 2, 2> = Owned::from_array([[5.0, 7.0], [6.0, 8.0]]); // col0=[5, 7], col1=[6, 8]

    println!("\n--- Matrix Construction & Basic Arithmetic ---");
    print_matrix("M1", &m1);
    print_matrix("M2", &m2);

    let sum: Owned<f64, 2, 2> = &m1 + &m2;
    let diff: Owned<f64, 2, 2> = &m2 - &m1;
    let prod: Owned<f64, 2, 2> = &m1 * &m2;
    let trans: Owned<f64, 2, 2> = m1.transpose();

    print_matrix("M1 + M2", &sum);
    print_matrix("M2 - M1", &diff);
    print_matrix("M1 * M2", &prod);
    print_matrix("M1^T (Transpose)", &trans);

    // 2. Linear System Solve: A * x = b
    // Column-major array format: [[col0], [col1], [col2]]
    let a: Owned<f64, 3, 3> = Owned::from_array([
        [3.0, 2.0, -1.0],  // col 0
        [2.0, -2.0, 0.5],  // col 1
        [-1.0, 4.0, -1.0], // col 2
    ]);
    let b: Owned<f64, 3, 1> = Owned::from_array([[1.0, -2.0, 0.0]]);

    println!("\n--- Linear System A * x = b ---");
    print_matrix("A", &a);
    print_matrix("b", &b);

    if let Ok(lu) = LuDecomposition::decompose(a) {
        let mut x = b;
        lu.solve_mut(&mut x).expect("LU solve");
        print_matrix("Solution x", &x);

        // 3. Matrix Inversion & Identity Verification
        if let Ok(a_inv) = lu.inverse() {
            println!("\n--- Matrix Inversion & Identity Check ---");
            print_matrix("A^-1", &a_inv);

            let ident_check: Owned<f64, 3, 3> = &a * &a_inv;
            print_matrix("A * A^-1 (Identity check)", &ident_check);
        }
    }
}
