//! Polynomial HIL and unit test suite.
#![allow(
    clippy::arithmetic_side_effects,
    clippy::float_cmp,
    clippy::doc_markdown
)]

#[cfg_attr(not(test), control_rs_macros::hil_suite)]
/// Polynomial representation tests.
pub mod test_polynomial {
    use crate::polynomial::{Constant, DensePolynomial, Line, Polynomial};

    #[cfg_attr(test, test)]
    /// Verifies DensePolynomial creation, coefficients, and degree.
    fn test_dense_polynomial_degree_and_coefficients() {
        let p = DensePolynomial::new([1.0, 2.0, 3.0]);
        assert_eq!(p.degree(), Some(2));
        assert_eq!(p.coefficients(), &[1.0, 2.0, 3.0]);
        assert_eq!(p.leading_coefficient(), Some(&3.0));
    }

    #[cfg_attr(test, test)]
    /// Verifies polynomial evaluation using Horner's method.
    fn test_dense_polynomial_evaluation() {
        // p(x) = 3x^2 + 2x + 1
        let p = DensePolynomial::new([1.0, 2.0, 3.0]);
        // p(2) = 3*4 + 2*2 + 1 = 17.0
        assert_eq!(p.evaluate(2.0), 17.0);
    }

    #[cfg_attr(test, test)]
    /// Verifies Constant polynomial behavior.
    fn test_constant_polynomial() {
        let p = Constant::new(5.0);
        assert_eq!(p.degree(), Some(0));
        assert_eq!(p.evaluate(10.0), 5.0);
        assert_eq!(p.leading_coefficient(), Some(&5.0));
    }

    #[cfg_attr(test, test)]
    /// Verifies Line polynomial behavior.
    fn test_line_polynomial() {
        // y = 2x + 3
        let p = Line::new(2.0, 3.0);
        assert_eq!(p.degree(), Some(1));
        assert_eq!(p.evaluate(4.0), 11.0);
        assert_eq!(p.leading_coefficient(), Some(&2.0));
    }
}
