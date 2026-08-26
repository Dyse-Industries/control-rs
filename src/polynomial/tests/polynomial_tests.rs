//! # Polynomial Unit and Property Tests
#![allow(
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::similar_names,
    clippy::unwrap_used,
    clippy::items_after_statements,
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::approx_constant
)]

#[cfg_attr(not(test), control_rs_macros::ets_suite)]
pub mod polynomial_test_suite {
    use crate::assert_almost_eq;
    use crate::math::complex_num::Complex;
    use crate::polynomial::ArrayPolynomial;

    #[cfg_attr(test, test)]
    fn test_polynomial_evaluation() {
        // p(x) = 1 + 2x + 3x^2
        let p = ArrayPolynomial::<f64, 3>::from_coefficients([1.0, 2.0, 3.0]);
        assert_eq!(p.capacity(), 3);
        assert_eq!(p.degree(), Some(2));
        assert_eq!(p.leading_coefficient(), Some(&3.0));
        assert!(!p.is_monic());

        // At x = 0: p(0) = 1
        assert_almost_eq!(p.evaluate(0.0), 1.0, 1e-12);
        // At x = 2: p(2) = 1 + 4 + 12 = 17
        assert_almost_eq!(p.evaluate(2.0), 17.0, 1e-12);
        // At x = -1: p(-1) = 1 - 2 + 3 = 2
        assert_almost_eq!(p.evaluate(-1.0), 2.0, 1e-12);
    }

    #[cfg_attr(test, test)]
    fn test_polynomial_complex_evaluation() {
        // p(x) = 1 + x^2 (roots at +/- j)
        let p = ArrayPolynomial::<f64, 3>::from_coefficients([1.0, 0.0, 1.0]);
        let j = Complex::new(0.0, 1.0);
        let res = p.evaluate_complex(j);
        assert_almost_eq!(res.re, 0.0, 1e-12);
        assert_almost_eq!(res.im, 0.0, 1e-12);
    }

    #[cfg_attr(test, test)]
    fn test_polynomial_arithmetic() {
        // p1(x) = 1 + 2x + 3x^2
        let p1 = ArrayPolynomial::<f64, 3>::from_coefficients([1.0, 2.0, 3.0]);
        // p2(x) = 4 + 5x + 6x^2
        let p2 = ArrayPolynomial::<f64, 3>::from_coefficients([4.0, 5.0, 6.0]);

        let sum = &p1 + &p2;
        assert_eq!(sum.as_slice(), &[5.0, 7.0, 9.0]);

        let diff = &p2 - &p1;
        assert_eq!(diff.as_slice(), &[3.0, 3.0, 3.0]);

        let neg = -&p1;
        assert_eq!(neg.as_slice(), &[-1.0, -2.0, -3.0]);
    }

    #[cfg_attr(test, test)]
    fn test_polynomial_derivative() {
        // p(x) = 5 + 3x + 4x^2 + 2x^3 -> p'(x) = 3 + 8x + 6x^2
        let p =
            ArrayPolynomial::<f64, 4>::from_coefficients([5.0, 3.0, 4.0, 2.0]);
        let dp = p.derivative();
        assert_eq!(dp.get(0), Some(&3.0));
        assert_eq!(dp.get(1), Some(&8.0));
        assert_eq!(dp.get(2), Some(&6.0));
    }

    #[cfg_attr(test, test)]
    fn test_polynomial_integral() {
        // p(x) = 3 + 8x + 6x^2 -> \int p(x) dx with c0 = 5 -> 5 + 3x + 4x^2 + 2x^3
        let p =
            ArrayPolynomial::<f64, 4>::from_coefficients([3.0, 8.0, 6.0, 0.0]);
        let integ = p.integral(5.0);
        assert_almost_eq!(integ.get(0).copied().unwrap(), 5.0, 1e-12);
        assert_almost_eq!(integ.get(1).copied().unwrap(), 3.0, 1e-12);
        assert_almost_eq!(integ.get(2).copied().unwrap(), 4.0, 1e-12);
        assert_almost_eq!(integ.get(3).copied().unwrap(), 2.0, 1e-12);
    }

    #[cfg_attr(test, test)]
    fn test_polynomial_multiplication() {
        // (1 + 2x) * (3 + 4x) = 3 + 10x + 8x^2
        let p1 = ArrayPolynomial::<f64, 2>::from_coefficients([1.0, 2.0]);
        let p2 = ArrayPolynomial::<f64, 2>::from_coefficients([3.0, 4.0]);
        let prod = p1.mul_poly::<2, 3>(&p2);
        assert_eq!(prod.as_slice(), &[3.0, 10.0, 8.0]);
    }

    #[cfg_attr(test, test)]
    fn test_polynomial_div_rem() {
        // (3 + 10x + 8x^2) / (1 + 2x) = (3 + 4x) rem 0
        let num =
            ArrayPolynomial::<f64, 3>::from_coefficients([3.0, 10.0, 8.0]);
        let den = ArrayPolynomial::<f64, 2>::from_coefficients([1.0, 2.0]);
        let (quot, rem) = num.div_rem::<2, 2, 1>(&den).unwrap();
        assert_almost_eq!(quot.get(0).copied().unwrap(), 3.0, 1e-12);
        assert_almost_eq!(quot.get(1).copied().unwrap(), 4.0, 1e-12);
        assert_almost_eq!(rem.get(0).copied().unwrap(), 0.0, 1e-12);
    }

    #[cfg_attr(test, test)]
    fn test_companion_matrix() {
        // Monic polynomial: p(x) = -6 - 5x + x^2  (roots: 6 and -1)
        let p = ArrayPolynomial::<f64, 3>::from_coefficients([-6.0, -5.0, 1.0]);
        assert!(p.is_monic());
        let comp = p.companion_matrix::<2>().unwrap();
        // C = [[0, 6], [1, 5]]
        assert_eq!(comp.get(0, 0), Some(&0.0));
        assert_eq!(comp.get(0, 1), Some(&6.0));
        assert_eq!(comp.get(1, 0), Some(&1.0));
        assert_eq!(comp.get(1, 1), Some(&5.0));
    }
}
