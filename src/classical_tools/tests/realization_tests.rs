//! # Firmware Realization Unit and Verification Tests
#![allow(clippy::indexing_slicing)]

#[cfg_attr(not(test), control_rs_macros::ets_suite)]
pub mod firmware_test_suite {
    use crate::assert_almost_eq;
    use crate::classical_tools::realization::{
        Biquad, BiquadCascade, DirectForm2T,
    };
    #[allow(unused_imports)]
    use crate::math::num_traits::Float;

    /// Independent Direct Form I reference oracle for a second-order section,
    /// used to cross-validate [`Biquad`] and [`DirectForm2T`] without sharing
    /// their DF2T recurrence.
    fn direct_form1_reference(
        coeffs: (f64, f64, f64, f64, f64),
        inputs: &[f64],
    ) -> [f64; 8] {
        let (b0, b1, b2, a1, a2) = coeffs;
        let (mut x1, mut x2, mut y1, mut y2) = (0.0_f64, 0.0, 0.0, 0.0);
        let mut out = [0.0_f64; 8];
        for (k, &u) in inputs.iter().enumerate() {
            let y = a2.mul_add(
                -y2,
                a1.mul_add(-y1, b2.mul_add(x2, b0.mul_add(u, b1 * x1))),
            );
            out[k] = y;
            x2 = x1;
            x1 = u;
            y2 = y1;
            y1 = y;
        }
        out
    }

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: classical-tools#FR-6
    /// Method: Requirements-based test
    fn test_biquad_identity_pass_through() {
        let mut section = Biquad::new(1.0_f64, 0.0, 0.0, 0.0, 0.0);
        assert_almost_eq!(section.update(3.0), 3.0, 1e-15);
        assert_almost_eq!(section.update(-2.0), -2.0, 1e-15);
        assert_almost_eq!(section.update(0.0), 0.0, 1e-15);
    }

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: classical-tools#FR-6
    /// Method: Requirements-based test
    fn test_biquad_matches_direct_form1_reference() {
        // H(z) = 0.25 / (1 - z^-1 + 0.25 z^-2): double pole at z = 0.5.
        let coeffs = (0.25_f64, 0.0, 0.0, -1.0, 0.25);
        let inputs = [1.0_f64, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let expected = direct_form1_reference(coeffs, &inputs);

        let mut section =
            Biquad::new(coeffs.0, coeffs.1, coeffs.2, coeffs.3, coeffs.4);
        for (k, &u) in inputs.iter().enumerate() {
            assert_almost_eq!(section.update(u), expected[k], 1e-12);
        }
    }

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: classical-tools#FR-6
    /// Method: Requirements-based test
    fn test_biquad_step_response_reaches_dc_gain() {
        // Same double-pole-at-0.5 section as above: DC gain
        // H(1) = b0 / (1 + a1 + a2) = 0.25 / 0.25 = 1.0.
        let mut section = Biquad::new(0.25_f64, 0.0, 0.0, -1.0, 0.25);
        let mut y = 0.0;
        for _ in 0..200 {
            y = section.update(1.0);
        }
        assert_almost_eq!(y, 1.0, 1e-9);
    }

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: classical-tools#FR-6
    /// Method: Requirements-based test
    fn test_biquad_reset_clears_state() {
        let mut section = Biquad::new(0.25_f64, 0.1, 0.05, -1.0, 0.25);
        let first = section.update(1.0);
        section.update(1.0);
        section.update(1.0);
        section.reset();
        let after_reset = section.update(1.0);
        assert_almost_eq!(after_reset, first, 1e-15);
    }

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: classical-tools#FR-5
    /// Method: Requirements-based test
    fn test_direct_form2t_order2_matches_biquad() {
        let coeffs = (0.5_f64, -0.2, 0.1, -0.6, 0.3);
        let inputs = [1.0_f64, 0.5, -0.5, 1.0, 0.0, -1.0, 1.0, 1.0];

        let mut biquad =
            Biquad::new(coeffs.0, coeffs.1, coeffs.2, coeffs.3, coeffs.4);
        let mut df2t = DirectForm2T::new(
            coeffs.0,
            [coeffs.1, coeffs.2],
            [coeffs.3, coeffs.4],
        );

        for &u in &inputs {
            assert_almost_eq!(biquad.update(u), df2t.update(u), 1e-12);
        }
    }

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: classical-tools#FR-5
    /// Method: Requirements-based test
    fn test_direct_form2t_matches_direct_form1_reference() {
        let coeffs = (0.5_f64, -0.2, 0.1, -0.6, 0.3);
        let inputs = [1.0_f64, 0.5, -0.5, 1.0, 0.0, -1.0, 1.0, 1.0];
        let expected = direct_form1_reference(coeffs, &inputs);

        let mut df2t = DirectForm2T::new(
            coeffs.0,
            [coeffs.1, coeffs.2],
            [coeffs.3, coeffs.4],
        );
        for (k, &u) in inputs.iter().enumerate() {
            assert_almost_eq!(df2t.update(u), expected[k], 1e-12);
        }
    }

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: classical-tools#FR-5
    /// Method: Requirements-based test
    fn test_direct_form2t_order_zero_is_pure_gain() {
        let mut filter = DirectForm2T::<f64, 0>::new(2.5, [], []);
        assert_almost_eq!(filter.update(4.0), 10.0, 1e-15);
        assert_almost_eq!(filter.update(-1.0), -2.5, 1e-15);
    }

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: classical-tools#FR-5
    /// Method: Requirements-based test
    fn test_direct_form2t_reset_clears_state() {
        let mut filter = DirectForm2T::new(0.5_f64, [-0.2, 0.1], [-0.6, 0.3]);
        let first = filter.update(1.0);
        filter.update(0.5);
        filter.update(-0.5);
        filter.reset();
        let after_reset = filter.update(1.0);
        assert_almost_eq!(after_reset, first, 1e-15);
    }

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: classical-tools#FR-6
    /// Method: Requirements-based test
    fn test_biquad_cascade_matches_sequential_sections() {
        let mut cascade = BiquadCascade::new([
            Biquad::new(0.25_f64, 0.0, 0.0, -1.0, 0.25),
            Biquad::new(0.5, -0.2, 0.1, -0.6, 0.3),
        ]);
        let mut section_a = Biquad::new(0.25_f64, 0.0, 0.0, -1.0, 0.25);
        let mut section_b = Biquad::new(0.5, -0.2, 0.1, -0.6, 0.3);

        let inputs = [1.0_f64, 0.5, -0.5, 1.0, 0.0, -1.0];
        for &u in &inputs {
            let expected = section_b.update(section_a.update(u));
            assert_almost_eq!(cascade.update(u), expected, 1e-12);
        }
    }

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: classical-tools#FR-6
    /// Method: Requirements-based test
    fn test_biquad_cascade_reset_clears_every_section() {
        let mut cascade = BiquadCascade::new([
            Biquad::new(0.25_f64, 0.1, 0.05, -1.0, 0.25),
            Biquad::new(0.5, -0.2, 0.1, -0.6, 0.3),
        ]);
        let first = cascade.update(1.0);
        cascade.update(1.0);
        cascade.reset();
        let after_reset = cascade.update(1.0);
        assert_almost_eq!(after_reset, first, 1e-15);
    }
}
