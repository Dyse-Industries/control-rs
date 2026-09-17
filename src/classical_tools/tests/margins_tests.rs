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
        let wgc = margins.gain_crossover_freq.unwrap();
        let expected_wgc = (16f64.pow(1.0 / 3.0) - 1.0).sqrt();
        assert!(
            ((wgc - expected_wgc) / expected_wgc).abs() <= 1e-4,
            "w_gc = {wgc}, expected ~{expected_wgc} (NFR-3 relative 1e-4)"
        );

        let expected_pm =
            (-3.0f64).mul_add(expected_wgc.atan(), core::f64::consts::PI);
        let pm = margins.phase_margin.unwrap();
        let pm_err_deg =
            (pm - expected_pm).abs() * (180.0 / core::f64::consts::PI);
        assert!(
            pm_err_deg <= 0.1,
            "phi_m = {pm} rad, expected ~{expected_pm} ({pm_err_deg} deg error)"
        );

        let gm = margins.gain_margin.unwrap();
        assert!(((gm - 2.0) / 2.0).abs() <= 1e-4, "K_g = {gm}, expected 2");
        assert!(margins.delay_margin.unwrap() > 0.0);
    }

    /// `L(s) = 10 / (s (s + 1) (s + 2))`, i.e. `num = [10]`,
    /// `den = [0, 2, 3, 1]` ascending. Analytically: `Im L(jw) = 0` at
    /// `w_pc = sqrt(2)` where `|L| = 10/6`, so `K_g = 0.6 < 1` and the closed
    /// loop is unstable; `|L(jw)| = 1` at `w_gc = 1.802203`, past the
    /// `-180 deg` crossing, where `arg L = +167.003 deg` on the principal
    /// branch and the phase margin is `-12.997 deg = -0.226844 rad`.
    const fn _unstable_loop() -> ArrayTransferFunction<f64, 1, 4> {
        ArrayTransferFunction::<f64, 1, 4>::continuous(
            [10.0],
            [0.0, 2.0, 3.0, 1.0],
        )
    }

    #[cfg_attr(test, test)]
    /// Regression: `pi + arg()` is not a phase margin. `arg()` is principal
    /// on `(-pi, pi]`, so a gain crossover past the `-180 deg` crossing
    /// yields `2 pi - |phi_m|` unless the sum is wrapped back onto
    /// `(-pi, pi]`. Unwrapped, a loop with `K_g < 1` reports a phase margin
    /// near `+360 deg` and a positive delay margin.
    ///
    /// # Verification
    /// Trace: classical-tools#FR-4, classical-tools#NFR-3
    /// Method: Requirements-based test
    fn test_unstable_loop_reports_negative_phase_and_delay_margin() {
        let tf = _unstable_loop();
        let omegas = _sweep_omegas();
        let margins = stability_margins(&tf, &omegas);

        // Gain crossover lies past phase crossover: the loop is unstable.
        let wpc = margins.phase_crossover_freq.unwrap();
        let wgc = margins.gain_crossover_freq.unwrap();
        assert!(wgc > wpc, "w_gc = {wgc}, w_pc = {wpc}");
        let gain_margin = margins.gain_margin.unwrap();
        assert!((gain_margin - 0.6).abs() < 1e-3, "K_g = {gain_margin}");

        let phase_margin = margins.phase_margin.unwrap();
        assert!(
            phase_margin < 0.0,
            "phi_m = {phase_margin} rad, expected negative (K_g < 1)"
        );
        assert!(
            (phase_margin - -0.226_844).abs() < 1e-3,
            "phi_m = {phase_margin} rad, expected ~-0.226844"
        );

        let delay_margin = margins.delay_margin.unwrap();
        assert!(
            (delay_margin - -0.125_870).abs() < 1e-3,
            "tau_m = {delay_margin} s, expected ~-0.125870"
        );
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
    /// A response that vanishes over the sweep has no reciprocal, so the
    /// gain margin is absent rather than infinite or NaN.
    ///
    /// # Verification
    /// Trace: classical-tools#FR-4
    /// Method: Requirements-based test
    fn test_vanishing_response_leaves_gain_margin_absent() {
        let tf = ArrayTransferFunction::<f64, 1, 4>::continuous(
            [0.0],
            [1.0, 3.0, 3.0, 1.0],
        );
        let omegas = _sweep_omegas();
        let margins = stability_margins(&tf, &omegas);
        assert_eq!(margins.gain_margin, None);
        assert_eq!(margins.gain_crossover_freq, None);
        assert_eq!(margins.delay_margin, None);
        for value in [
            margins.gain_margin,
            margins.phase_margin,
            margins.delay_margin,
        ]
        .into_iter()
        .flatten()
        {
            assert!(value.is_finite(), "margin {value} must be finite");
        }
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

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: classical-tools#FR-4
    /// Method: Requirements-based test
    fn test_gain_margin_absent_when_magnitude_vanishes() {
        let tf =
            ArrayTransferFunction::<f64, 1, 2>::continuous([0.0], [1.0, 1.0]);
        let omegas = _sweep_omegas();
        let margins = stability_margins(&tf, &omegas);
        assert_eq!(margins.gain_margin, None);
    }
}
