//! Polynomial evaluation and convolution product.
//!
//! Run with `cargo run --example polynomial`.

use control_rs::polynomial::ArrayPolynomial;

/// Evaluate 1 + 2x + 3x² and multiply by 4 + 5x.
fn main() {
    let p = ArrayPolynomial::<f64, 3>::from_coefficients([1.0, 2.0, 3.0]);
    let q = ArrayPolynomial::<f64, 2>::from_coefficients([4.0, 5.0]);
    let y = p.evaluate(2.0);
    let product = p.mul_poly::<2, 4>(&q);

    println!("p(x) = 1 + 2x + 3x^2,  coeffs = {:?}", p.to_coefficients());
    println!("p(2) = {y:.4}  (expected 17)");
    println!("q(x) = 4 + 5x,  coeffs = {:?}", q.to_coefficients());
    println!(
        "p*q coeffs (capacity N+M-1 = 4) = {:?}",
        product.to_coefficients()
    );
}
