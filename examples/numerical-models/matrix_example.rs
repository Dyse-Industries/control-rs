//! Matrix Numerical Model Example
//!
//! Demonstrates matrix construction, multiplication, LU decomposition,
//! linear system solving, matrix inversion, and Kalman filter covariance update.

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

    // 1. Linear System Solve: A * x = b
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
        lu.solve_mut(&mut x);
        print_matrix("Solution x", &x);

        // Matrix Inversion
        if let Ok(a_inv) = lu.inverse() {
            print_matrix("A^-1", &a_inv);

            let ident_check: Owned<f64, 3, 3> = &a * &a_inv;
            print_matrix("A * A^-1 (Identity check)", &ident_check);
        }
    }

    // 2. Discrete Kalman Filter Measurement Covariance Update:
    // P_post = (I - K * H) * P_prior
    let p_prior: Owned<f64, 2, 2> = Owned::from_array([[2.0, 0.5], [0.5, 1.0]]);
    let h: Owned<f64, 1, 2> = Owned::from_array([[1.0], [0.0]]);
    let k: Owned<f64, 2, 1> = Owned::from_array([[0.6, 0.2]]);
    let identity: Owned<f64, 2, 2> = Owned::identity();

    println!("\n--- Discrete Kalman Filter Covariance Update ---");
    print_matrix("P_prior", &p_prior);
    print_matrix("H", &h);
    print_matrix("K", &k);

    let kh: Owned<f64, 2, 2> = &k * &h;
    let i_minus_kh: Owned<f64, 2, 2> = &identity - &kh;
    let p_post: Owned<f64, 2, 2> = &i_minus_kh * &p_prior;

    print_matrix("I - K*H", &i_minus_kh);
    print_matrix("P_post", &p_post);
}
