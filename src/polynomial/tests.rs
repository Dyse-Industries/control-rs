#![allow(clippy::float_cmp)]

use crate::polynomial::{Constant, Line, Polynomial, StaticPolynomial};

#[test]
fn test_constant_polynomial() {
    let zero = Constant::new(0.0);
    assert_eq!(zero.degree(), None); // The zero polynomial has undefined/no degree
    assert_eq!(zero.leading_coefficient(), None);

    let one = Constant::new(1.0);
    assert_eq!(one.degree(), Some(0));
    assert_eq!(one.leading_coefficient(), Some(&1.0));
    assert_eq!(one.evaluate(5.0), 1.0);
    assert_eq!(one.coefficients(), &[1.0]);
}

#[test]
fn test_line_polynomial() {
    // 2x + 3
    let line = Line::new(2.0, 3.0);
    assert_eq!(line.degree(), Some(1));
    assert_eq!(line.leading_coefficient(), Some(&2.0));
    assert_eq!(line.evaluate(4.0), 11.0); // 2 * 4 + 3 = 11
    assert_eq!(line.coefficients(), &[3.0, 2.0]);
}

#[test]
fn test_static_polynomial_creation() {
    // 3x^2 + 2x + 1
    let p = StaticPolynomial::from_coefficients([1.0, 2.0, 3.0]);
    assert_eq!(p.degree(), Some(2));
    assert_eq!(p.leading_coefficient(), Some(&3.0));
    assert_eq!(p.evaluate(2.0), 17.0); // 3 * 4 + 2 * 2 + 1 = 17

    let p_desc = StaticPolynomial::from_descending([3.0, 2.0, 1.0]);
    assert_eq!(p_desc.coefficients(), &[1.0, 2.0, 3.0]);
}

#[test]
fn test_polynomial_arithmetic() {
    let p1 = StaticPolynomial::from_coefficients([1.0, 2.0, 3.0]);
    let p2 = StaticPolynomial::from_coefficients([4.0, 5.0, 6.0]);

    // Addition
    let p_sum = p1 + p2;
    assert_eq!(p_sum.coefficients(), &[5.0, 7.0, 9.0]);

    // Subtraction
    let p_diff = p1 - p2;
    assert_eq!(p_diff.coefficients(), &[-3.0, -3.0, -3.0]);

    // AddAssign & SubAssign
    let mut p_mut = p1;
    p_mut += p2;
    assert_eq!(p_mut.coefficients(), &[5.0, 7.0, 9.0]);
    p_mut -= p2;
    assert_eq!(p_mut.coefficients(), &[1.0, 2.0, 3.0]);
}

#[test]
fn test_polynomial_multiplication() {
    // p1 = 2x + 1 (length 2)
    let p1 = StaticPolynomial::from_coefficients([1.0, 2.0]);
    // p2 = 3x + 4 (length 2)
    let p2 = StaticPolynomial::from_coefficients([4.0, 3.0]);

    // (2x + 1)(3x + 4) = 6x^2 + 9x + 4 (length 2 + 2 - 1 = 3)
    let p_prod: StaticPolynomial<f64, 3> = p1.mul_poly(&p2);
    assert_eq!(p_prod.coefficients(), &[4.0, 11.0, 6.0]);
}

#[test]
fn test_polynomial_derivative() {
    // p = 3x^2 + 2x + 1
    let p = StaticPolynomial::from_coefficients([1.0, 2.0, 3.0]);
    // dp = 6x + 2
    let dp = p.derivative();
    assert_eq!(dp.coefficients(), &[2.0, 6.0, 0.0]);

    // Constant derivative
    let c = Constant::new(5.0);
    let dc = c.derivative();
    assert_eq!(dc.coefficients(), &[0.0]); // Derivative of constant is 0 (length 1)
}
