//! # Lead, Lag, and Lead-Lag Compensator Unit and Verification Tests
#![allow(clippy::arithmetic_side_effects)]

#[cfg_attr(not(test), control_rs_macros::ets_suite)]
pub mod compensators_test_suite {
    use crate::classical_tools::compensators::{
        CompensatorError, lag, lead, lead_lag,
    };
    // Needed on no_std/libm targets, where `f64` has no inherent `mul_add`;
    // std's inherent method wins on host builds, making this unused there.
    #[allow(unused_imports)]
    use crate::math::num_traits::Float;

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: classical-tools#FR-8
    /// Method: Requirements-based test
    fn test_lead_places_pole_farther_than_zero() {
        // K = 1, T = 1, alpha = 0.1: zero at s = -1/T = -1, pole at
        // s = -1/(alpha*T) = -10. A lead network places its pole farther
        // from the origin than its zero.
        let tf = lead::<f64>(1.0, 1.0, 0.1).unwrap();
        let zero = tf.zeros().unwrap()[0];
        let pole = tf.poles().unwrap()[0];
        assert!((zero.re - (-1.0)).abs() < 1e-9, "zero = {zero:?}");
        assert!((pole.re - (-10.0)).abs() < 1e-9, "pole = {pole:?}");
        assert!(pole.re.abs() > zero.re.abs());
    }

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: classical-tools#FR-8
    /// Method: Requirements-based test
    fn test_lag_places_zero_farther_than_pole() {
        // K = 1, T = 1, alpha = 10: zero at s = -1, pole at
        // s = -1/(alpha*T) = -0.1. A lag network places its zero farther
        // from the origin than its pole.
        let tf = lag::<f64>(1.0, 1.0, 10.0).unwrap();
        let zero = tf.zeros().unwrap()[0];
        let pole = tf.poles().unwrap()[0];
        assert!((zero.re - (-1.0)).abs() < 1e-9, "zero = {zero:?}");
        assert!((pole.re - (-0.1)).abs() < 1e-9, "pole = {pole:?}");
        assert!(zero.re.abs() > pole.re.abs());
    }

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: classical-tools#FR-8
    /// Method: Requirements-based test
    fn test_dc_gain_matches_k_times_alpha() {
        // C(0) = K * (1/T) / (1/(alpha*T)) = K * alpha, independent of the
        // internal ascending-coefficient representation.
        let k = 3.0_f64;
        let alpha = 0.2_f64;
        let tf = lead(k, 2.0, alpha).unwrap();
        let dc = tf.eval_frequency(0.0);
        assert!(k.mul_add(-alpha, dc.re).abs() < 1e-9, "dc = {dc:?}");
        assert!(dc.im.abs() < 1e-12);
    }

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: classical-tools#FR-8
    /// Method: Requirements-based test
    fn test_invalid_alpha_errors() {
        assert_eq!(
            lead::<f64>(1.0, 1.0, 1.0),
            Err(CompensatorError::InvalidAlpha)
        );
        assert_eq!(
            lead::<f64>(1.0, 1.0, 2.0),
            Err(CompensatorError::InvalidAlpha)
        );
        assert_eq!(
            lag::<f64>(1.0, 1.0, 1.0),
            Err(CompensatorError::InvalidAlpha)
        );
        assert_eq!(
            lag::<f64>(1.0, 1.0, 0.5),
            Err(CompensatorError::InvalidAlpha)
        );
    }

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: classical-tools#FR-8
    /// Method: Requirements-based test
    fn test_lead_lag_response_matches_product_of_stages() {
        // C_total(jw) must equal C_lead(jw) * C_lag(jw): an algebraic
        // invariant of series (cascade) connection that does not depend on
        // `lead_lag`'s internal polynomial-convolution implementation.
        let lead_stage = lead(2.0, 0.5, 0.2).unwrap();
        let lag_stage = lag(1.5, 4.0, 8.0).unwrap();
        let combined = lead_lag(2.0, 0.5, 0.2, 1.5, 4.0, 8.0).unwrap();

        for &omega in &[0.1_f64, 0.5, 1.0, 3.0, 10.0] {
            let expected = lead_stage.eval_frequency(omega)
                * lag_stage.eval_frequency(omega);
            let actual = combined.eval_frequency(omega);
            assert!(
                (actual - expected).magnitude() < 1e-9,
                "omega = {omega}: actual = {actual:?}, expected = {expected:?}"
            );
        }
    }

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: classical-tools#FR-8
    /// Method: Requirements-based test
    fn test_lead_lag_propagates_invalid_alpha() {
        assert_eq!(
            lead_lag::<f64>(1.0, 1.0, 1.0, 1.0, 1.0, 2.0),
            Err(CompensatorError::InvalidAlpha)
        );
        assert_eq!(
            lead_lag::<f64>(1.0, 1.0, 0.5, 1.0, 1.0, 1.0),
            Err(CompensatorError::InvalidAlpha)
        );
    }

    #[cfg_attr(test, test)]
    /// Lead requires `0 < alpha < 1`. `alpha <= 0` yields Inf/RHP poles.
    ///
    /// # Verification
    /// Trace: classical-tools#FR-11
    /// Method: Requirements-based test
    fn test_lead_rejects_non_positive_alpha() {
        assert_eq!(
            lead::<f64>(1.0, 1.0, 0.0),
            Err(CompensatorError::InvalidAlpha)
        );
        assert_eq!(
            lead::<f64>(1.0, 1.0, -0.5),
            Err(CompensatorError::InvalidAlpha)
        );
    }

    #[cfg_attr(test, test)]
    /// A non-positive time constant is refused so `1/T` is finite.
    ///
    /// # Verification
    /// Trace: classical-tools#FR-11
    /// Method: Requirements-based test
    fn test_lead_and_lag_reject_non_positive_time_constant() {
        assert!(lead::<f64>(1.0, 0.0, 0.5).is_err());
        assert!(lead::<f64>(1.0, -1.0, 0.5).is_err());
        assert!(lag::<f64>(1.0, 0.0, 2.0).is_err());
        assert!(lag::<f64>(1.0, -1.0, 2.0).is_err());
    }
}
