//! # Transfer Function Unit and Verification Tests
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
pub mod transfer_function_test_suite {
    use crate::assert_almost_eq;
    use crate::transfer_function::ArrayTransferFunction;

    #[cfg_attr(test, test)]
    fn test_frequency_response_continuous() {
        // 1st-order low-pass filter: H(s) = 1 / (1 + s)
        let tf =
            ArrayTransferFunction::<f64, 1, 2>::continuous([1.0], [1.0, 1.0]);

        // DC: H(0) = 1.0 + 0j
        let h0 = tf.eval_frequency(0.0);
        assert_almost_eq!(h0.re, 1.0, 1e-12);
        assert_almost_eq!(h0.im, 0.0, 1e-12);

        // Corner frequency: omega = 1.0 -> H(j) = 1 / (1 + j) = 0.5 - 0.5j
        let h1 = tf.eval_frequency(1.0);
        assert_almost_eq!(h1.re, 0.5, 1e-12);
        assert_almost_eq!(h1.im, -0.5, 1e-12);

        let (mag, phase) = tf.bode_point(1.0);
        assert_almost_eq!(mag, core::f64::consts::FRAC_1_SQRT_2, 1e-6); // 1 / sqrt(2)
        assert_almost_eq!(phase, -core::f64::consts::FRAC_PI_4, 1e-6); // -45 deg
    }

    #[cfg_attr(test, test)]
    fn test_transfer_function_series() {
        // H1(s) = 1 / (1 + s), H2(s) = 2 / (2 + s)
        let h1 =
            ArrayTransferFunction::<f64, 1, 2>::continuous([1.0], [1.0, 1.0]);
        let h2 =
            ArrayTransferFunction::<f64, 1, 2>::continuous([2.0], [2.0, 1.0]);

        // H_series = 2 / (2 + 3s + s^2)
        let h_ser = h1.series::<1, 2, 1, 3>(&h2);
        assert_eq!(h_ser.num_slice(), &[2.0]);
        assert_eq!(h_ser.den_slice(), &[2.0, 3.0, 1.0]);
    }

    #[cfg_attr(test, test)]
    fn test_controllable_canonical_form() {
        // H(s) = (2 + 3s) / (4 + 5s + s^2)  [monic denominator]
        let tf = ArrayTransferFunction::<f64, 2, 3>::continuous(
            [2.0, 3.0],
            [4.0, 5.0, 1.0],
        );
        let ss = tf.to_controllable_canonical_form::<2>().unwrap();

        // Controllable companion (design §4.10):
        // A = [[0, 1], [-4, -5]], B = [[0], [1]], C = [[2, 3]], D = [[0]]
        assert_almost_eq!(ss.a().get(0, 0).copied().unwrap(), 0.0, 1e-12);
        assert_almost_eq!(ss.a().get(0, 1).copied().unwrap(), 1.0, 1e-12);
        assert_almost_eq!(ss.a().get(1, 0).copied().unwrap(), -4.0, 1e-12);
        assert_almost_eq!(ss.a().get(1, 1).copied().unwrap(), -5.0, 1e-12);
        assert_almost_eq!(ss.b().get(0, 0).copied().unwrap(), 0.0, 1e-12);
        assert_almost_eq!(ss.b().get(1, 0).copied().unwrap(), 1.0, 1e-12);
        assert_almost_eq!(ss.c().get(0, 0).copied().unwrap(), 2.0, 1e-12);
        assert_almost_eq!(ss.c().get(0, 1).copied().unwrap(), 3.0, 1e-12);
        assert_almost_eq!(ss.d().get(0, 0).copied().unwrap(), 0.0, 1e-12);

        // Realization identity: C(sI-A)^{-1}B + D must match H(s) at s = jω.
        let h_tf = tf.eval_frequency(1.0);
        // Manual 2x2 solve of (sI-A)x = B at s = j, then y = Cx.
        // sI-A = [[j, -1], [4, j+5]]; det = -1 + 5j + 4 = 3 + 5j
        // (sI-A)^{-1}B = (1/det)[1; j] => y = (2 + 3j)/det
        let det_re = 3.0;
        let det_im = 5.0;
        let det_abs2 = det_re * det_re + det_im * det_im;
        let num_re = 2.0;
        let num_im = 3.0;
        let h_ss_re = (num_re * det_re + num_im * det_im) / det_abs2;
        let h_ss_im = (num_im * det_re - num_re * det_im) / det_abs2;
        assert_almost_eq!(h_tf.re, h_ss_re, 1e-12);
        assert_almost_eq!(h_tf.im, h_ss_im, 1e-12);
    }

    #[cfg_attr(test, test)]
    fn test_controllable_canonical_form_with_feedthrough() {
        // Proper but not strictly: H(s) = (2 + 3s + s^2) / (4 + 5s + s^2)
        // => d = 1, β = (2-4, 3-5) = (-2, -2)
        let tf = ArrayTransferFunction::<f64, 3, 3>::continuous(
            [2.0, 3.0, 1.0],
            [4.0, 5.0, 1.0],
        );
        let ss = tf.to_controllable_canonical_form::<2>().unwrap();
        assert_almost_eq!(ss.d().get(0, 0).copied().unwrap(), 1.0, 1e-12);
        assert_almost_eq!(ss.c().get(0, 0).copied().unwrap(), -2.0, 1e-12);
        assert_almost_eq!(ss.c().get(0, 1).copied().unwrap(), -2.0, 1e-12);
    }

    #[cfg_attr(test, test)]
    fn test_evaluate_complex_empty_numerator() {
        // N = 0 is a valid Dim; Horner must not underflow usize.
        let tf = ArrayTransferFunction::<f64, 0, 2>::continuous([], [1.0, 1.0]);
        let h = tf
            .evaluate_complex(crate::math::complex_num::Complex::new(0.0, 1.0));
        assert_almost_eq!(h.re, 0.0, 1e-12);
        assert_almost_eq!(h.im, 0.0, 1e-12);
    }
}
