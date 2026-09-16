//! # Frequency-Domain Stability Margin Unit and Verification Tests

#[cfg_attr(not(test), control_rs_macros::ets_suite)]
pub mod margins_test_suite {
    use crate::classical_tools::margins::stability_margins;
    // Needed on no_std/libm targets, where `f64` has no inherent `pow`,
    // `sqrt`, or `mul_add`; std's inherent methods win on host builds,
    // making these unused there.
    #[allow(unused_imports)]
    use crate::math::num_traits::{Exponential, Float, Radical};
    use crate::transfer_function::ArrayTransferFunction;

    /// `G(s) = 4 / (s + 1)^3`, i.e. `num = [4]`, `den = [1, 3, 3, 1]`
    /// ascending. Analytically: `|G(jw)|^2 = 16 / (1 + w^2)^3` crosses unity
    /// at `w_gc = sqrt(16^(1/3) - 1)`, and `arg G(jw) = -3 atan(w)` crosses
    /// `-180 deg` at `w_pc = tan(60 deg) = sqrt(3)` (independent of the
    /// numerator gain).
    const fn _plant() -> ArrayTransferFunction<f64, 1, 4> {
        ArrayTransferFunction::<f64, 1, 4>::continuous(
            [4.0],
            [1.0, 3.0, 3.0, 1.0],
        )
    }

    fn _sweep_omegas() -> [f64; 2000] {
        core::array::from_fn(|i| (i as f64).mul_add(5.0 / 2000.0, 0.001))
    }

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: classical-tools#FR-4, classical-tools#NFR-3
    /// Method: Requirements-based test
    fn test_gain_crossover_satisfies_unity_magnitude() {
        let tf = _plant();
        let omegas = _sweep_omegas();
        let margins = stability_margins(&tf, &omegas);

        let wgc = margins.gain_crossover_freq.unwrap();
        let mag = tf.eval_frequency(wgc).magnitude();
        assert!((mag - 1.0).abs() < 1e-3, "|G(j*{wgc})| = {mag}, expected 1");

        let expected_wgc = (16f64.pow(1.0 / 3.0) - 1.0).sqrt();
        assert!(
            (wgc - expected_wgc).abs() < 1e-2,
            "w_gc = {wgc}, expected ~{expected_wgc}"
        );
    }

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: classical-tools#FR-4, classical-tools#NFR-3
    /// Method: Requirements-based test
    fn test_phase_crossover_satisfies_negative_real_axis() {
        let tf = _plant();
        let omegas = _sweep_omegas();
        let margins = stability_margins(&tf, &omegas);

        let wpc = margins.phase_crossover_freq.unwrap();
        let resp = tf.eval_frequency(wpc);
        assert!(resp.im.abs() < 1e-3, "Im(G(j*{wpc})) = {}", resp.im);
        assert!(resp.re < 0.0, "Re(G(j*{wpc})) = {} (expected < 0)", resp.re);

        let expected_wpc = 3f64.sqrt();
        assert!(
            (wpc - expected_wpc).abs() < 1e-2,
            "w_pc = {wpc}, expected ~{expected_wpc}"
        );
    }

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: classical-tools#FR-4, classical-tools#NFR-3
    /// Method: Requirements-based test
    fn test_gain_and_phase_margins_are_positive_for_stable_plant() {
        let tf = _plant();
        let omegas = _sweep_omegas();
        let margins = stability_margins(&tf, &omegas);

        // Gain crossover occurs before phase crossover in frequency, so the
        // system has positive gain and phase margin at this gain (K = 4).
        assert!(
            margins.gain_crossover_freq.unwrap()
                < margins.phase_crossover_freq.unwrap()
        );
        assert!(margins.phase_margin.unwrap() > 0.0);
        assert!(margins.gain_margin.unwrap() > 1.0);
        assert!(margins.delay_margin.unwrap() > 0.0);
    }

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: classical-tools#FR-4, classical-tools#NFR-3
    /// Method: Requirements-based test
    fn test_no_crossing_within_range_yields_none() {
        let tf = _plant();
        // Well below either crossover: magnitude near 8 (DC gain) throughout,
        // phase near 0.
        let omegas: [f64; 50] =
            core::array::from_fn(|i| (i as f64).mul_add(0.001, 0.001));
        let margins = stability_margins(&tf, &omegas);
        assert_eq!(margins.gain_crossover_freq, None);
        assert_eq!(margins.phase_crossover_freq, None);
        assert_eq!(margins.gain_margin, None);
        assert_eq!(margins.phase_margin, None);
        assert_eq!(margins.delay_margin, None);
    }

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: classical-tools#FR-4, classical-tools#NFR-3
    /// Method: Requirements-based test
    fn test_degenerate_sweep_yields_none() {
        let tf = _plant();
        let margins = stability_margins(&tf, &[1.0]);
        assert_eq!(margins.gain_crossover_freq, None);
        assert_eq!(margins.phase_crossover_freq, None);
    }
}
