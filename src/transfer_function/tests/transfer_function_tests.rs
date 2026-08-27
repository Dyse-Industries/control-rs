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
    use crate::transfer_function::{
        ArrayTransferFunction, TransferFunctionError,
    };

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

        // Golden from examples/prototypes/numerical-models/transfer_function_prototype.py
        let p1 =
            ArrayTransferFunction::<f64, 1, 2>::continuous([2.0], [2.0, 1.0]);
        let p2 =
            ArrayTransferFunction::<f64, 1, 2>::continuous([5.0], [5.0, 1.0]);
        let p_ser = p1.series::<1, 2, 1, 3>(&p2);
        assert_eq!(p_ser.num_slice(), &[10.0]);
        assert_eq!(p_ser.den_slice(), &[10.0, 7.0, 1.0]);
    }

    #[cfg_attr(test, test)]
    fn test_parallel_and_feedback() {
        let h1 =
            ArrayTransferFunction::<f64, 1, 2>::continuous([1.0], [1.0, 1.0]);
        let h2 =
            ArrayTransferFunction::<f64, 1, 2>::continuous([1.0], [1.0, 1.0]);
        // H+H = 2/(1+s) = 2(1+s) / (1+s)^2 = (2+2s)/(1+2s+s^2)
        let h_par = h1.parallel::<1, 2, 3, 3>(&h2);
        assert_almost_eq!(h_par.num_slice()[0], 2.0, 1e-12);
        assert_almost_eq!(h_par.num_slice()[1], 2.0, 1e-12);
        assert_almost_eq!(h_par.den_slice()[0], 1.0, 1e-12);
        assert_almost_eq!(h_par.den_slice()[2], 1.0, 1e-12);

        // H/(1+H^2): num = 1+s, den = 2+2s+s^2
        let h_fb = h1.feedback::<1, 2, 2, 3>(&h2);
        assert_almost_eq!(h_fb.num_slice()[0], 1.0, 1e-12);
        assert_almost_eq!(h_fb.num_slice()[1], 1.0, 1e-12);
        assert_almost_eq!(h_fb.den_slice()[0], 2.0, 1e-12);
        assert_almost_eq!(h_fb.den_slice()[1], 2.0, 1e-12);
        assert_almost_eq!(h_fb.den_slice()[2], 1.0, 1e-12);
    }

    #[cfg_attr(test, test)]
    #[allow(clippy::too_many_lines)]
    fn test_try_from_coefficients() {
        assert_eq!(
            ArrayTransferFunction::<f64, 1, 2>::try_continuous(
                [1.0],
                [1.0, 0.0]
            ),
            Err(TransferFunctionError::ZeroLeadingDenominatorCoefficient)
        );
        assert_eq!(
            ArrayTransferFunction::<f64, 3, 2>::try_continuous(
                [1.0, 0.0, 1.0],
                [1.0, 1.0]
            ),
            Err(TransferFunctionError::ImproperSystem)
        );
        let ok = ArrayTransferFunction::<f64, 1, 2>::try_discrete(
            [1.0],
            [1.0, 1.0],
            0.1,
        )
        .unwrap();
        assert!(ok.is_discrete());
        assert_eq!(ok.sample_time(), Some(0.1));
        assert!(!ok.is_continuous());
        use core::fmt::Write;
        struct StackBuf([u8; 192], usize);
        impl Write for StackBuf {
            fn write_str(&mut self, s: &str) -> core::fmt::Result {
                let rest = self.0.len().saturating_sub(self.1);
                let n = rest.min(s.len());
                self.0[self.1..self.1 + n].copy_from_slice(&s.as_bytes()[..n]);
                self.1 += n;
                Ok(())
            }
        }
        let mut zbuf = StackBuf([0u8; 192], 0);
        write!(
            &mut zbuf,
            "{}",
            TransferFunctionError::ZeroLeadingDenominatorCoefficient
        )
        .unwrap();
        assert!(
            core::str::from_utf8(&zbuf.0[..zbuf.1])
                .unwrap()
                .contains("denominator")
        );
        let mut ibuf = StackBuf([0u8; 192], 0);
        write!(&mut ibuf, "{}", TransferFunctionError::ImproperSystem).unwrap();
        assert!(
            core::str::from_utf8(&ibuf.0[..ibuf.1])
                .unwrap()
                .contains("improper")
        );
        let from_st = ArrayTransferFunction::<f64, 1, 2>::from_storage(
            crate::math::storage::ArrayStorage::from_column([1.0]),
            crate::math::storage::ArrayStorage::from_column([1.0, 1.0]),
            None,
        );
        assert!(from_st.is_continuous());
        let disc_ss = ok.to_observable_canonical_form::<1>().unwrap();
        assert!(disc_ss.is_discrete());
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

        let ocf = tf.to_observable_canonical_form::<2>().unwrap();
        assert_almost_eq!(ocf.a().get(1, 0).copied().unwrap(), 1.0, 1e-12);
        assert_almost_eq!(ocf.b().get(0, 0).copied().unwrap(), 2.0, 1e-12);
        assert_almost_eq!(ocf.c().get(0, 1).copied().unwrap(), 1.0, 1e-12);

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
    fn test_ccf_proper_feedthrough() {
        // H(s) = (1 + 2s) / (1 + s)  → d = 2, β0 = 1 - 2*1 = -1
        let tf = ArrayTransferFunction::<f64, 2, 2>::continuous(
            [1.0, 2.0],
            [1.0, 1.0],
        );
        let ss = tf.to_controllable_canonical_form::<1>().unwrap();
        assert_almost_eq!(ss.a().get(0, 0).copied().unwrap(), -1.0, 1e-12);
        assert_almost_eq!(ss.d().get(0, 0).copied().unwrap(), 2.0, 1e-12);
        assert_almost_eq!(ss.c().get(0, 0).copied().unwrap(), -1.0, 1e-12);
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

    #[cfg_attr(test, test)]
    fn test_tustin_and_zoh() {
        let tf =
            ArrayTransferFunction::<f64, 1, 2>::continuous([1.0], [1.0, 1.0]);
        let dt = 0.1;
        let z_tustin = tf.to_discrete_tustin(dt, None);
        assert!(z_tustin.is_discrete());
        let z_zoh = tf.to_discrete_zoh::<1>(dt).unwrap();
        // Integrator-like lowpass ZOH: pole e^{-dt}
        assert_almost_eq!(
            z_zoh.den_slice()[0] / z_zoh.den_slice()[1],
            -0.904_837_418_035_959_5, // -exp(-dt)
            1e-8
        );
        let _ = z_tustin;
    }

    #[cfg_attr(test, test)]
    fn test_controllable_canonical_form_realization_invariant_order2() {
        use crate::math::complex_num::Complex;

        // 2nd-order test case: H(s) = (2 + 3s) / (4 + 5s + s^2)
        let tf2 = ArrayTransferFunction::<f64, 2, 3>::continuous(
            [2.0, 3.0],
            [4.0, 5.0, 1.0],
        );
        let ss2 = tf2.to_controllable_canonical_form::<2>().unwrap();

        let omegas = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0];
        for &w in &omegas {
            let s = Complex::new(0.0, w);
            let h_tf = tf2.eval_frequency(w);

            // Compute H_ss(s) = C * (sI - A)^(-1) * B + D for 2x2 system
            let a00 = *ss2.a().get(0, 0).unwrap();
            let a01 = *ss2.a().get(0, 1).unwrap();
            let a10 = *ss2.a().get(1, 0).unwrap();
            let a11 = *ss2.a().get(1, 1).unwrap();

            let b0 = *ss2.b().get(0, 0).unwrap();
            let b1 = *ss2.b().get(1, 0).unwrap();

            let c0 = *ss2.c().get(0, 0).unwrap();
            let c1 = *ss2.c().get(0, 1).unwrap();

            let d0 = *ss2.d().get(0, 0).unwrap();

            // sI - A = [[s - a00, -a01], [-a10, s - a11]]
            let m00 = s - Complex::from_real(a00);
            let m01 = Complex::from_real(-a01);
            let m10 = Complex::from_real(-a10);
            let m11 = s - Complex::from_real(a11);

            let det = m00 * m11 - m01 * m10;
            let inv00 = m11 / det;
            let inv01 = -m01 / det;
            let inv10 = -m10 / det;
            let inv11 = m00 / det;

            let x0 =
                inv00 * Complex::from_real(b0) + inv01 * Complex::from_real(b1);
            let x1 =
                inv10 * Complex::from_real(b0) + inv11 * Complex::from_real(b1);

            let h_ss = Complex::from_real(c0) * x0
                + Complex::from_real(c1) * x1
                + Complex::from_real(d0);

            assert_almost_eq!(h_ss.re, h_tf.re, 1e-12);
            assert_almost_eq!(h_ss.im, h_tf.im, 1e-12);
        }
    }

    #[cfg_attr(test, test)]
    fn test_controllable_canonical_form_realization_invariant_order3() {
        use crate::math::complex_num::Complex;

        // 3rd-order test case: H(s) = (1 + 2s + 3s^2) / (6 + 11s + 6s^2 + s^3)
        let tf3 = ArrayTransferFunction::<f64, 3, 4>::continuous(
            [1.0, 2.0, 3.0],
            [6.0, 11.0, 6.0, 1.0],
        );
        let ss3 = tf3.to_controllable_canonical_form::<3>().unwrap();

        assert_almost_eq!(ss3.a().get(0, 1).copied().unwrap(), 1.0, 1e-12);
        assert_almost_eq!(ss3.a().get(1, 2).copied().unwrap(), 1.0, 1e-12);
        assert_almost_eq!(ss3.a().get(2, 0).copied().unwrap(), -6.0, 1e-12);
        assert_almost_eq!(ss3.a().get(2, 1).copied().unwrap(), -11.0, 1e-12);
        assert_almost_eq!(ss3.a().get(2, 2).copied().unwrap(), -6.0, 1e-12);
        assert_almost_eq!(ss3.b().get(2, 0).copied().unwrap(), 1.0, 1e-12);
        assert_almost_eq!(ss3.c().get(0, 0).copied().unwrap(), 1.0, 1e-12);
        assert_almost_eq!(ss3.c().get(0, 1).copied().unwrap(), 2.0, 1e-12);
        assert_almost_eq!(ss3.c().get(0, 2).copied().unwrap(), 3.0, 1e-12);

        let omegas = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0];
        for &w in &omegas {
            let s = Complex::new(0.0, w);
            let h_tf = tf3.eval_frequency(w);

            let a0 = -(*ss3.a().get(2, 0).unwrap());
            let a1 = -(*ss3.a().get(2, 1).unwrap());
            let a2 = -(*ss3.a().get(2, 2).unwrap());

            let den_s = s * s * s
                + Complex::from_real(a2) * s * s
                + Complex::from_real(a1) * s
                + Complex::from_real(a0);
            let x0 = Complex::from_real(1.0) / den_s;
            let x1 = s * x0;
            let x2 = s * s * x0;

            let c0 = *ss3.c().get(0, 0).unwrap();
            let c1 = *ss3.c().get(0, 1).unwrap();
            let c2 = *ss3.c().get(0, 2).unwrap();

            let h_ss = Complex::from_real(c0) * x0
                + Complex::from_real(c1) * x1
                + Complex::from_real(c2) * x2;

            assert_almost_eq!(h_ss.re, h_tf.re, 1e-12);
            assert_almost_eq!(h_ss.im, h_tf.im, 1e-12);
        }
    }

    #[cfg_attr(test, test)]
    fn test_transfer_function_view() {
        let mut tf =
            ArrayTransferFunction::<f64, 1, 2>::continuous([1.0], [1.0, 1.0]);
        let view = tf.view();
        let h0 = view.eval_frequency(0.0);
        assert_almost_eq!(h0.re, 1.0, 1e-12);
        assert_almost_eq!(h0.im, 0.0, 1e-12);
        {
            let vm = tf.view_mut();
            let h1 = vm.eval_frequency(1.0);
            assert_almost_eq!(h1.re, 0.5, 1e-12);
            assert_almost_eq!(h1.im, -0.5, 1e-12);
        }
    }
}
