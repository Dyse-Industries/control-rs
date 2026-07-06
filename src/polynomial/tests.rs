#![allow(clippy::float_cmp)]
#![allow(dead_code)]
#![allow(clippy::arithmetic_side_effects)]
#![allow(clippy::unwrap_used)]
#![allow(clippy::indexing_slicing)]
#![allow(clippy::panic)]
#![allow(clippy::missing_const_for_fn)]

#[cfg_attr(all(not(test), not(feature = "std")), control_rs_macros::hil_suite)]
/// HIL and unit test suite for polynomials.
pub mod polynomial_tests {
    use crate::polynomial::{Constant, Line, Polynomial};

    #[cfg_attr(test, test)]
    fn test_constant_polynomial() {
        let zero = Constant::new(0.0);
        assert_eq!(zero.degree(), None);
        assert_eq!(zero.leading_coefficient(), None);

        let one = Constant::new(1.0);
        assert_eq!(one.degree(), Some(0));
        assert_eq!(one.leading_coefficient(), Some(&1.0));
        assert_eq!(one.evaluate(5.0), 1.0);
        assert_eq!(one.coefficients(), &[1.0]);
    }

    #[cfg_attr(test, test)]
    fn test_line_polynomial() {
        let line = Line::new(2.0, 3.0);
        assert_eq!(line.degree(), Some(1));
        assert_eq!(line.leading_coefficient(), Some(&2.0));
        assert_eq!(line.evaluate(4.0), 11.0);
        assert_eq!(line.coefficients(), &[3.0, 2.0]);
    }

    #[cfg_attr(test, test)]
    fn test_static_polynomial_creation() {
        let p = Polynomial::from_coefficients([1.0, 2.0, 3.0]);
        assert_eq!(p.degree(), Some(2));
        assert_eq!(p.leading_coefficient(), Some(&3.0));
        assert_eq!(p.evaluate(2.0), 17.0);

        let p_desc = Polynomial::from_descending([3.0, 2.0, 1.0]);
        assert_eq!(p_desc.coefficients(), &[1.0, 2.0, 3.0]);
    }

    #[cfg_attr(test, test)]
    fn test_polynomial_arithmetic() {
        let p1 = Polynomial::from_coefficients([1.0, 2.0, 3.0]);
        let p2 = Polynomial::from_coefficients([4.0, 5.0, 6.0]);

        let p_sum = p1 + p2;
        assert_eq!(p_sum.coefficients(), &[5.0, 7.0, 9.0]);

        let p_diff = p1 - p2;
        assert_eq!(p_diff.coefficients(), &[-3.0, -3.0, -3.0]);

        let mut p_mut = p1;
        p_mut += p2;
        assert_eq!(p_mut.coefficients(), &[5.0, 7.0, 9.0]);
        p_mut -= p2;
        assert_eq!(p_mut.coefficients(), &[1.0, 2.0, 3.0]);
    }

    #[cfg_attr(test, test)]
    fn test_polynomial_multiplication() {
        let p1 = Polynomial::from_coefficients([1.0, 2.0]);
        let p2 = Polynomial::from_coefficients([4.0, 3.0]);

        let p_prod: Polynomial<f64, 3> = p1.mul_poly(&p2);
        assert_eq!(p_prod.coefficients(), &[4.0, 11.0, 6.0]);
    }

    #[cfg_attr(test, test)]
    fn test_polynomial_derivative() {
        let p = Polynomial::from_coefficients([1.0, 2.0, 3.0]);
        let dp = p.derivative();
        assert_eq!(dp.coefficients(), &[2.0, 6.0, 0.0]);

        let c = Constant::new(5.0);
        let dc = c.derivative();
        assert_eq!(dc.coefficients(), &[0.0]);
    }

    #[cfg_attr(test, test)]
    fn test_polynomial_scaling() {
        let mut p = Polynomial::from_coefficients([1.0, 2.0, 3.0]);
        p *= 2.0;
        assert_eq!(p.coefficients(), &[2.0, 4.0, 6.0]);
        p /= 2.0;
        assert_eq!(p.coefficients(), &[1.0, 2.0, 3.0]);
    }

    #[cfg_attr(test, test)]
    fn test_polynomial_division() {
        // (3x^2 + 5x + 6) / (x + 2)
        // Expected Quotient: 3x - 1 (coefficients [-1.0, 3.0])
        // Expected Remainder: 8 (coefficients [8.0])
        let p1 = Polynomial::from_coefficients([6.0, 5.0, 3.0]);
        let p2 = Polynomial::from_coefficients([2.0, 1.0]);

        let (q, r) = p1.div_rem::<2, 2, 1>(&p2);
        assert_eq!(q.coefficients(), &[-1.0, 3.0]);
        assert_eq!(r.coefficients(), &[8.0]);
    }
}
