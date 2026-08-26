//! Polynomial Numerical Model Example
//!
//! Demonstrates polynomial construction, real and complex Horner evaluation,
//! differentiation, integration, Euclidean division, and Frobenius companion matrix formulation.

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

use control_rs::math::complex_num::Complex;
use control_rs::polynomial::ArrayPolynomial;

fn main() {
    println!("=== Polynomial Numerical Model Example ===");

    // 1. Real Polynomial: p(x) = 2 - 3x + 4x^2 + x^3
    let p = ArrayPolynomial::<f64, 5>::from_coefficients([
        2.0, -3.0, 4.0, 1.0, 0.0,
    ]);
    println!("\n--- Polynomial Evaluation & Calculus ---");
    println!("Coefficients (ascending): {:?}", p.as_slice());

    let x_test = 2.5;
    let val_real = p.evaluate(x_test);
    println!("p({x_test}) = {val_real:.10}");

    let s_test = Complex::new(1.0, 2.0);
    let val_complex = p.evaluate_complex(s_test);
    println!(
        "p({:.1} + {:.1}j) = {:.10} + {:.10}j",
        s_test.re, s_test.im, val_complex.re, val_complex.im
    );

    let dp = p.derivative();
    println!("p'(x) coefficients: {:?}", dp.as_slice());
    println!("p'({x_test}) = {:.10}", dp.evaluate(x_test));

    let integ = p.integral(5.0);
    println!("int p(x) dx (c0=5) coefficients: {:?}", integ.as_slice());
    println!(
        "int_0^{x_test} p(t) dt + 5.0 = {:.10}",
        integ.evaluate(x_test)
    );

    // 2. Polynomial Multiplication & Division
    let p1 = ArrayPolynomial::<f64, 2>::from_coefficients([1.0, 2.0]); // (1 + 2x)
    let p2 = ArrayPolynomial::<f64, 2>::from_coefficients([3.0, 4.0]); // (3 + 4x)
    let prod = p1.mul_poly::<2, 3>(&p2);
    println!("\n--- Polynomial Multiplication & Division ---");
    println!("(1 + 2x) * (3 + 4x) = {:?}", prod.as_slice());

    if let Ok((quot, rem)) = prod.div_rem::<2, 2, 1>(&p1) {
        println!(
            "Quotient of ({:?}) / ({:?}): {:?}",
            prod.as_slice(),
            p1.as_slice(),
            quot.as_slice()
        );
        println!("Remainder: {:?}", rem.as_slice());
    }

    // 3. Monic Polynomial & Companion Matrix
    // Monic p(x) = -6 - 5x + x^2  (roots: 6 and -1)
    let p_monic =
        ArrayPolynomial::<f64, 3>::from_coefficients([-6.0, -5.0, 1.0]);
    println!("\n--- Monic Companion Matrix ---");
    println!("Monic p(x) coefficients: {:?}", p_monic.as_slice());
    assert!(p_monic.is_monic());

    if let Ok(comp) = p_monic.companion_matrix::<2>() {
        println!("Companion Matrix C:");
        for i in 0..2 {
            print!("  [");
            for j in 0..2 {
                if j > 0 {
                    print!(", ");
                }
                print!("{:8.4}", comp.get(i, j).copied().unwrap_or(0.0));
            }
            println!("]");
        }
    }
}
