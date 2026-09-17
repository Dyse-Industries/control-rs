//! # Routh-Hurwitz Stability Unit and Verification Tests
#![allow(clippy::unwrap_used)]

#[cfg_attr(not(test), control_rs_macros::ets_suite)]
pub mod routh_test_suite {
    use crate::classical_tools::routh::{RouthError, stability};
    use crate::math::num_types::{Const, Dim};
    use crate::polynomial::ArrayPolynomial;

    const EPSILON: f64 = 1e-9;

    /// Independent oracle: counts roots with a real part clearly to the
    /// right of the imaginary axis by actually solving `poly`, without
    /// sharing any code with [`stability`]'s Routh-array recurrence. A
    /// `1e-6` guard band absorbs the root solver's own floating-point noise
    /// for roots that sit exactly on the axis (e.g. `+/- j`, which the
    /// solver returns with a real part around `1e-16`, not exactly `0.0`).
    ///
    /// The iterate is taken whether or not the solver met its step bound: a
    /// repeated root converges only linearly and stalls well above that
    /// bound while sitting far inside the same guard band, so requiring
    /// convergence here would cost the oracle every multiple-root case.
    fn oracle_rhp_count<const N: usize>(poly: &ArrayPolynomial<f64, N>) -> usize
    where
        Const<N>: Dim,
    {
        poly.roots_best_effort()
            .unwrap()
            .0
            .iter()
            .take(N.saturating_sub(1))
            .filter(|root| root.re > 1e-6)
            .count()
    }

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: classical-tools#FR-1
    /// Method: Requirements-based test
    fn test_stable_second_order() {
        // (s + 1)(s + 2) = s^2 + 3s + 2.
        let poly =
            ArrayPolynomial::<f64, 3>::from_coefficients([2.0, 3.0, 1.0]);
        assert_eq!(stability(&poly, EPSILON), Ok(oracle_rhp_count(&poly)));
        assert_eq!(stability(&poly, EPSILON), Ok(0));
    }

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: classical-tools#FR-1
    /// Method: Requirements-based test
    fn test_single_rhp_root() {
        // (s - 1)(s + 2) = s^2 + s - 2: one RHP root at s = 1.
        let poly =
            ArrayPolynomial::<f64, 3>::from_coefficients([-2.0, 1.0, 1.0]);
        assert_eq!(stability(&poly, EPSILON), Ok(oracle_rhp_count(&poly)));
        assert_eq!(stability(&poly, EPSILON), Ok(1));
    }

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: classical-tools#FR-1
    /// Method: Requirements-based test
    fn test_stable_triple_root() {
        // (s + 1)^3 = s^3 + 3s^2 + 3s + 1.
        let poly =
            ArrayPolynomial::<f64, 4>::from_coefficients([1.0, 3.0, 3.0, 1.0]);
        assert_eq!(stability(&poly, EPSILON), Ok(oracle_rhp_count(&poly)));
        assert_eq!(stability(&poly, EPSILON), Ok(0));
    }

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: classical-tools#FR-2
    /// Method: Requirements-based test
    fn test_row_of_zeros_degenerate_case() {
        // (s^2 + 1)(s + 1)^2 = s^4 + 2s^3 + 2s^2 + 2s + 1: a pair of purely
        // imaginary roots (s = +/- j) forces a Routh row of zeros. RHP count
        // is still 0 (the jw-axis pair is not right-half-plane).
        let poly = ArrayPolynomial::<f64, 5>::from_coefficients([
            1.0, 2.0, 2.0, 2.0, 1.0,
        ]);
        assert_eq!(stability(&poly, EPSILON), Ok(oracle_rhp_count(&poly)));
        assert_eq!(stability(&poly, EPSILON), Ok(0));
    }

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: classical-tools#FR-2
    /// Method: Requirements-based test
    fn test_first_column_zero_degenerate_case() {
        // s^4 + s^3 + 2s^2 + 2s + 3: row 2 of the Routh array has a vanishing
        // first column with a nonzero remainder, forcing the epsilon
        // substitution rather than the row-of-zeros path.
        let poly = ArrayPolynomial::<f64, 5>::from_coefficients([
            3.0, 2.0, 2.0, 1.0, 1.0,
        ]);
        assert_eq!(stability(&poly, EPSILON), Ok(oracle_rhp_count(&poly)));
    }

    #[cfg_attr(test, test)]
    /// Regression: the epsilon substituted for a vanishing first-column
    /// divisor must be the value the first column carries, not the zero it
    /// replaced. `count_sign_changes` skips exact zeros, so recording the
    /// pre-substitution zero loses both sign changes whenever the entries
    /// bracketing it share a sign, undercounting the RHP roots by two.
    ///
    /// # Verification
    /// Trace: classical-tools#FR-14
    /// Method: Requirements-based test
    fn test_first_column_zero_between_negative_entries() {
        // s^4 - s^3 - 1. Roots: 1.380278, 0.219447 +/- 0.914474j and
        // -0.819173, so three lie in the open RHP and none on the axis.
        // The Routh first column is [1, -1, 0, -1e9, -1]; the recorded 0 is
        // the epsilon-substituted divisor and carries a positive sign.
        let poly = ArrayPolynomial::<f64, 5>::from_coefficients([
            -1.0, 0.0, 0.0, -1.0, 1.0,
        ]);
        assert_eq!(stability(&poly, EPSILON), Ok(oracle_rhp_count(&poly)));
        assert_eq!(stability(&poly, EPSILON), Ok(3));
    }

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: classical-tools#FR-1
    /// Method: Requirements-based test
    fn test_zero_leading_coefficient_errors() {
        let poly =
            ArrayPolynomial::<f64, 3>::from_coefficients([1.0, 2.0, 0.0]);
        assert_eq!(
            stability(&poly, EPSILON),
            Err(RouthError::ZeroLeadingCoefficient)
        );
    }

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: classical-tools#FR-1
    /// Method: Requirements-based test
    fn test_degree_zero_and_empty_are_trivially_stable() {
        let constant = ArrayPolynomial::<f64, 1>::from_coefficients([5.0]);
        assert_eq!(stability(&constant, EPSILON), Ok(0));

        let empty = ArrayPolynomial::<f64, 0>::from_coefficients([]);
        assert_eq!(stability(&empty, EPSILON), Ok(0));
    }
}
