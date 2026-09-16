//! # Polynomial Unit Tests
// Expected values here are hand-derived reference expressions; the `mul_add`
// form clippy suggests obscures the algebra being asserted and is not a
// performance concern in a test oracle.
#![allow(clippy::suboptimal_flops)]

#[cfg_attr(not(test), control_rs_macros::ets_suite)]
pub mod polynomial_test_suite {
    use crate::assert_almost_eq;
    use crate::math::complex_num::Complex;
    #[allow(unused_imports)]
    use crate::math::num_traits::{Float, Radical};
    use crate::matrix::Owned;
    use crate::polynomial::{
        ArrayPolynomial, DivisionError, QuadraticRootError,
    };
    use core::convert::TryFrom;

    /// A flat array converts to coefficients in ascending power order,
    /// matching `from_coefficients`.
    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: polynomial-design#FR-1
    /// Method: Requirements-based test
    fn test_polynomial_array_conversion() {
        let converted: ArrayPolynomial<f64, 3> = [1.0, 2.0, 3.0].into();
        assert_eq!(
            converted,
            ArrayPolynomial::<f64, 3>::from_coefficients([1.0, 2.0, 3.0])
        );
        // 1 + 2x + 3x^2 at x = 2 is 17.
        assert_almost_eq!(converted.evaluate(2.0), 17.0, 1e-12);
    }

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: polynomial-design#FR-2
    /// Method: Requirements-based test
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

    /// Higham $\gamma_k = k\varepsilon / (1 - k\varepsilon)$.
    fn _gamma(k: f64) -> f64 {
        let ke = k * f64::EPSILON;
        if ke >= 1.0 {
            f64::INFINITY
        } else {
            ke / (1.0 - ke)
        }
    }

    #[cfg_attr(test, test)]
    /// Horner backward error $\lvert p(x)-\hat p(x)\rvert \le \gamma_{2n}\tilde p(\lvert x\rvert)$
    /// (`polynomial-design.md` §6.3).
    /// # Verification
    /// Trace: polynomial-design#FR-2
    /// Method: Requirements-based test
    fn test_horner_backward_error() {
        // p(x) = 1 + x + x^2 + x^3; n = 3; closed form (1 - x^4)/(1 - x).
        let p =
            ArrayPolynomial::<f64, 4>::from_coefficients([1.0, 1.0, 1.0, 1.0]);
        let x = 1.0_f64 / 3.0;
        let exact = (x * x * x).mul_add(-x, 1.0) / (1.0 - x);
        let hat = p.evaluate(x);
        let tilde = [1.0, 1.0, 1.0, 1.0]
            .iter()
            .rev()
            .fold(0.0_f64, |acc, &c| acc.mul_add(x.abs(), c));
        let bound = _gamma(6.0) * tilde;
        let err = (exact - hat).abs();
        assert!(
            err <= bound.max(f64::EPSILON),
            "Horner |p-hat|={err} exceeds gamma_6 tilde_p={bound}"
        );
    }

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: polynomial-design#FR-2
    /// Method: Requirements-based test
    fn test_polynomial_complex_evaluation() {
        // p(x) = 1 + x^2 (roots at +/- j)
        let p = ArrayPolynomial::<f64, 3>::from_coefficients([1.0, 0.0, 1.0]);
        let j = Complex::new(0.0, 1.0);
        let res = p.evaluate_complex(j);
        assert_almost_eq!(res.re, 0.0, 1e-12);
        assert_almost_eq!(res.im, 0.0, 1e-12);
    }

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: polynomial-design#FR-1
    /// Method: Requirements-based test
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
    /// # Verification
    /// Trace: polynomial-design#FR-2
    /// Method: Requirements-based test
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
    /// # Verification
    /// Trace: polynomial-design#FR-2
    /// Method: Requirements-based test
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
    /// # Verification
    /// Trace: polynomial-design#FR-3
    /// Method: Requirements-based test
    fn test_polynomial_multiplication() {
        // (1 + 2x) * (3 + 4x) = 3 + 10x + 8x^2
        let p1 = ArrayPolynomial::<f64, 2>::from_coefficients([1.0, 2.0]);
        let p2 = ArrayPolynomial::<f64, 2>::from_coefficients([3.0, 4.0]);
        let prod = p1.mul_poly::<2, 3>(&p2);
        assert_eq!(prod.as_slice(), &[3.0, 10.0, 8.0]);
        let conv = p1.mul_with_conv::<2, 3>(&p2);
        assert_eq!(conv.as_slice(), prod.as_slice());
    }

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: polynomial-design#FR-4
    /// Method: Requirements-based test
    fn test_polynomial_div_rem() {
        // (3 + 10x + 8x^2) / (1 + 2x) = (3 + 4x) rem 0
        let num =
            ArrayPolynomial::<f64, 3>::from_coefficients([3.0, 10.0, 8.0]);
        let den = ArrayPolynomial::<f64, 2>::from_coefficients([1.0, 2.0]);
        let (quot, rem) = num.div_rem::<2, 2, 1>(&den).unwrap();
        assert_almost_eq!(quot.get(0).copied().unwrap(), 3.0, 1e-12);
        assert_almost_eq!(quot.get(1).copied().unwrap(), 4.0, 1e-12);
        assert_almost_eq!(rem.get(0).copied().unwrap(), 0.0, 1e-12);

        let zero_den = ArrayPolynomial::<f64, 2>::zero();
        assert_eq!(
            num.div_rem::<2, 2, 1>(&zero_den),
            Err(DivisionError::ZeroLeadingCoefficient)
        );
        // Capacities agree, actual degrees do not: deg(divisor) > deg(dividend)
        // is a value-dependent condition and stays a runtime error.
        let low = ArrayPolynomial::<f64, 3>::from_coefficients([3.0, 0.0, 0.0]);
        let high =
            ArrayPolynomial::<f64, 3>::from_coefficients([1.0, 0.0, 1.0]);
        assert_eq!(
            low.div_rem::<3, 1, 2>(&high),
            Err(DivisionError::DegreeMismatch)
        );
    }

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: polynomial-design#FR-5
    /// Method: Requirements-based test
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
        let via_try = Owned::<f64, 2, 2>::try_from(&p).unwrap();
        assert_eq!(via_try.get(0, 1), Some(&6.0));
        assert_eq!(via_try.get(1, 1), Some(&5.0));

        // Charpoly of C is p: eigenvalues are the known roots 6 and -1.
        let a00 = *comp.get(0, 0).unwrap();
        let a01 = *comp.get(0, 1).unwrap();
        let a10 = *comp.get(1, 0).unwrap();
        let a11 = *comp.get(1, 1).unwrap();
        let tr = a00 + a11;
        let det = a00 * a11 - a01 * a10;
        let disc = Radical::sqrt(tr * tr - 4.0 * det);
        let mut lam = [f64::midpoint(tr, disc), f64::midpoint(tr, -disc)];
        if lam[0] < lam[1] {
            lam.swap(0, 1);
        }
        let bound = 20.0 * f64::EPSILON * (1.0 + 6.0);
        assert!(
            (lam[0] - 6.0).abs() <= bound,
            "companion eigenvalue {} vs root 6",
            lam[0]
        );
        assert!(
            (lam[1] + 1.0).abs() <= bound,
            "companion eigenvalue {} vs root -1",
            lam[1]
        );
    }

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: polynomial-design#FR-6, polynomial-design#FR-7
    /// Method: Requirements-based test
    fn test_cubic_quintic_bilinear() {
        let cubic = ArrayPolynomial::<f64, 4>::cubic(0.0, 1.0, 0.0, 0.0);
        assert_almost_eq!(cubic.evaluate(0.0), 0.0, 1e-12);
        assert_almost_eq!(cubic.evaluate(1.0), 1.0, 1e-12);
        assert_almost_eq!(cubic.evaluate(0.5), 0.5, 1e-12);
        let d_cubic = cubic.derivative();
        assert_almost_eq!(d_cubic.evaluate(0.0), 0.0, 1e-12);
        assert_almost_eq!(d_cubic.evaluate(1.0), 0.0, 1e-12);

        let quintic =
            ArrayPolynomial::<f64, 6>::quintic(0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
        assert_almost_eq!(quintic.evaluate(0.0), 0.0, 1e-12);
        assert_almost_eq!(quintic.evaluate(1.0), 1.0, 1e-12);
        let first_deriv = quintic.derivative();
        let second_deriv = first_deriv.derivative();
        assert_almost_eq!(first_deriv.evaluate(0.0), 0.0, 1e-12);
        assert_almost_eq!(first_deriv.evaluate(1.0), 0.0, 1e-12);
        assert_almost_eq!(second_deriv.evaluate(0.0), 0.0, 1e-12);
        assert_almost_eq!(second_deriv.evaluate(1.0), 0.0, 1e-12);

        // p(s) = 1 + s, Ts = 2 → k = 1; clear (z+1): 2z
        let p = ArrayPolynomial::<f64, 2>::from_coefficients([1.0, 1.0]);
        let z = p.compose_bilinear(2.0);
        assert_almost_eq!(z.get(0).copied().unwrap(), 0.0, 1e-12);
        assert_almost_eq!(z.get(1).copied().unwrap(), 2.0, 1e-12);
    }

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: polynomial-design#FR-1
    /// Method: Requirements-based test
    fn test_polynomial_constructors_and_storage() {
        let c = ArrayPolynomial::<f64, 1>::constant(4.0);
        assert_almost_eq!(c.evaluate(99.0), 4.0, 1e-12);
        let line = ArrayPolynomial::<f64, 2>::line(1.0, 2.0);
        assert_almost_eq!(line.evaluate(3.0), 7.0, 1e-12);
        let from_fn = ArrayPolynomial::<f64, 3>::from_fn(|i| i as f64);
        assert_almost_eq!(from_fn.get(2).copied().unwrap(), 2.0);
        let mut p =
            ArrayPolynomial::<f64, 3>::from_coefficients([1.0, 2.0, 3.0]);
        assert_eq!(p.storage().as_slice().len(), 3);
        *p.get_mut(0).unwrap() = 9.0;
        let storage = p.into_storage();
        assert_almost_eq!(storage.as_slice()[0], 9.0);
        assert_eq!(
            ArrayPolynomial::<f64, 2>::from_coefficients([1.0, 0.0])
                .div_rem::<1, 2, 0>(
                    &ArrayPolynomial::<f64, 1>::from_coefficients([0.0])
                )
                .err(),
            Some(DivisionError::ZeroLeadingCoefficient)
        );
    }

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: polynomial-design#FR-1
    /// Method: Requirements-based test
    fn test_polynomial_view() {
        let mut p =
            ArrayPolynomial::<f64, 3>::from_coefficients([1.0, 2.0, 3.0]);
        let view = p.view();
        assert_eq!(view.get(0), Some(&1.0));
        assert_eq!(view.get(2), Some(&3.0));
        assert_almost_eq!(view.evaluate(2.0), 17.0, 1e-12);
        {
            let mut vm = p.view_mut();
            if let Some(c0) = vm.get_mut(0) {
                *c0 = 4.0;
            }
        }
        assert_almost_eq!(p.evaluate(2.0), 20.0, 1e-12);
    }

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: polynomial-design#FR-8, polynomial-design#FR-10
    /// Method: Requirements-based test
    fn test_quadratic_roots() {
        let order_pair = |a: f64, b: f64| -> (f64, f64) {
            if a <= b { (a, b) } else { (b, a) }
        };

        // Distinct real roots: (x - 2)(x - 3) = 6 - 5x + x^2
        let p_real =
            ArrayPolynomial::<f64, 3>::from_coefficients([6.0, -5.0, 1.0]);
        let roots = p_real.roots_quadratic().unwrap();
        let (r_min, r_max) = order_pair(roots[0].re, roots[1].re);
        assert_almost_eq!(r_min, 2.0, 1e-12);
        assert_almost_eq!(r_max, 3.0, 1e-12);
        assert_almost_eq!(roots[0].im, 0.0, 1e-12);
        assert_almost_eq!(roots[1].im, 0.0, 1e-12);

        // Repeated real roots: (x - 4)^2 = 16 - 8x + x^2
        let p_rep =
            ArrayPolynomial::<f64, 3>::from_coefficients([16.0, -8.0, 1.0]);
        let roots_rep = p_rep.roots_quadratic().unwrap();
        assert_almost_eq!(roots_rep[0].re, 4.0, 1e-12);
        assert_almost_eq!(roots_rep[1].re, 4.0, 1e-12);
        assert_almost_eq!(roots_rep[0].im, 0.0, 1e-12);
        assert_almost_eq!(roots_rep[1].im, 0.0, 1e-12);

        // Complex conjugate roots: s^2 + 2s + 5 = 0 -> s = -1 ± 2j
        let p_c = ArrayPolynomial::<f64, 3>::from_coefficients([5.0, 2.0, 1.0]);
        let roots_c = p_c.roots_quadratic().unwrap();
        assert_almost_eq!(roots_c[0].re, -1.0, 1e-12);
        assert_almost_eq!(roots_c[1].re, -1.0, 1e-12);
        let (im_min, im_max) = order_pair(roots_c[0].im, roots_c[1].im);
        assert_almost_eq!(im_min, -2.0, 1e-12);
        assert_almost_eq!(im_max, 2.0, 1e-12);

        // Negative leading coefficient: -2x^2 + 6x - 4 = 0 -> roots 1, 2
        let p_neg =
            ArrayPolynomial::<f64, 3>::from_coefficients([-4.0, 6.0, -2.0]);
        let roots_neg = p_neg.roots_quadratic().unwrap();
        let (neg_min, neg_max) = order_pair(roots_neg[0].re, roots_neg[1].re);
        assert_almost_eq!(neg_min, 1.0, 1e-12);
        assert_almost_eq!(neg_max, 2.0, 1e-12);

        // Degenerate zero leading coefficient: c2 = 0
        let p_degen =
            ArrayPolynomial::<f64, 3>::from_coefficients([1.0, 2.0, 0.0]);
        assert_eq!(
            p_degen.roots_quadratic().err(),
            Some(QuadraticRootError::ZeroLeadingCoefficient)
        );
    }

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: polynomial-design#FR-8, polynomial-design#FR-10
    /// Method: Requirements-based test
    fn test_quadratic_cancellation() {
        let order_pair = |a: f64, b: f64| -> (f64, f64) {
            if a <= b { (a, b) } else { (b, a) }
        };

        // (x - 1e7)(x - 1e-7) = x^2 - (1e7 + 1e-7)x + 1
        let r1_exact = 1e7_f64;
        let r2_exact = 1e-7_f64;
        let p = ArrayPolynomial::<f64, 3>::from_coefficients([
            1.0,
            -(r1_exact + r2_exact),
            1.0,
        ]);
        let roots = p.roots_quadratic().unwrap();
        let (r_min, r_max) = order_pair(roots[0].re, roots[1].re);

        let rel_err1 = (r_min - r2_exact).abs() / r2_exact;
        let rel_err2 = (r_max - r1_exact).abs() / r1_exact;

        assert!(
            rel_err1 < 1e-12,
            "Small root relative error {rel_err1} exceeds threshold (subtractive cancellation occurred)"
        );
        assert!(
            rel_err2 < 1e-12,
            "Large root relative error {rel_err2} exceeds threshold"
        );
    }

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: polynomial-design#FR-8, polynomial-design#FR-10
    /// Method: Requirements-based test
    fn test_generic_roots() {
        use crate::polynomial::RootError;

        // Degree 1 (Linear, N=2): 6 + 2x = 0 -> root -3
        let p_lin = ArrayPolynomial::<f64, 2>::from_coefficients([6.0, 2.0]);
        let root_lin = p_lin.line_intercept().unwrap();
        assert_almost_eq!(root_lin.re, -3.0, 1e-12);
        assert_almost_eq!(root_lin.im, 0.0, 1e-12);
        let roots_gen1 = p_lin.roots().unwrap();
        assert_eq!(roots_gen1.len(), 2);
        assert_almost_eq!(roots_gen1[0].re, -3.0, 1e-12);
        assert_almost_eq!(roots_gen1[1].re, 0.0, 1e-12);

        // Degree 2 (Quadratic, N=3): 6 - 5x + x^2 = 0 -> roots 2, 3
        let p_quad =
            ArrayPolynomial::<f64, 3>::from_coefficients([6.0, -5.0, 1.0]);
        let roots_gen2 = p_quad.roots().unwrap();
        assert_eq!(roots_gen2.len(), 3);
        let mut re_parts = [roots_gen2[0].re, roots_gen2[1].re];
        if re_parts[0] > re_parts[1] {
            re_parts.swap(0, 1);
        }
        assert_almost_eq!(re_parts[0], 2.0, 1e-12);
        assert_almost_eq!(re_parts[1], 3.0, 1e-12);
        assert_almost_eq!(roots_gen2[2].re, 0.0, 1e-12);

        // Degree 3 (Cubic, N=4): (x-1)(x-2)(x-3) = -6 + 11x - 6x^2 + x^3 = 0
        let p_cubic = ArrayPolynomial::<f64, 4>::from_coefficients([
            -6.0, 11.0, -6.0, 1.0,
        ]);
        let roots_cubic = p_cubic.roots().unwrap();
        assert_eq!(roots_cubic.len(), 4);
        let mut re_cubic =
            [roots_cubic[0].re, roots_cubic[1].re, roots_cubic[2].re];
        // Sort roots
        for i in 0..3 {
            for j in (i + 1)..3 {
                if re_cubic[i] > re_cubic[j] {
                    re_cubic.swap(i, j);
                }
            }
        }
        assert_almost_eq!(re_cubic[0], 1.0, 1e-8);
        assert_almost_eq!(re_cubic[1], 2.0, 1e-8);
        assert_almost_eq!(re_cubic[2], 3.0, 1e-8);
        assert_almost_eq!(roots_cubic[3].re, 0.0, 1e-12);

        // Degree 4 (Quartic, N=5): (s^2 + 2s + 5)(s^2 + 4s + 5) = 25 + 30s + 18s^2 + 6s^3 + s^4
        let p_quartic = ArrayPolynomial::<f64, 5>::from_coefficients([
            25.0, 30.0, 18.0, 6.0, 1.0,
        ]);
        let roots_quartic = p_quartic.roots().unwrap();
        assert_eq!(roots_quartic.len(), 5);
        for r in &roots_quartic[0..4] {
            let p_eval = p_quartic.evaluate_complex(*r);
            assert!(
                p_eval.magnitude() < 1e-8,
                "Quartic root residual exceeds bound"
            );
        }

        // Zero leading coefficient error
        let p_degen =
            ArrayPolynomial::<f64, 3>::from_coefficients([1.0, 2.0, 0.0]);
        assert_eq!(
            p_degen.roots().err(),
            Some(RootError::ZeroLeadingCoefficient)
        );
    }

    /// `from_roots` expands $\prod_{k=0}^{N-2}(x - r_k)$ into ascending
    /// coefficients, using the first `N-1` entries and ignoring the last.
    ///
    /// Trace: polynomial-design#FR-2
    /// Method: Requirements-based test
    #[cfg_attr(test, test)]
    fn test_from_roots_expands_exactly() {
        // (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6
        let p = ArrayPolynomial::<f64, 4>::from_roots([1.0, 2.0, 3.0, 0.0]);
        assert_almost_eq!(p.to_coefficients()[0], -6.0, 1e-12);
        assert_almost_eq!(p.to_coefficients()[1], 11.0, 1e-12);
        assert_almost_eq!(p.to_coefficients()[2], -6.0, 1e-12);
        assert_almost_eq!(p.to_coefficients()[3], 1.0, 1e-12);

        // A repeated root must not collapse: (x-2)^2 = x^2 - 4x + 4.
        let sq = ArrayPolynomial::<f64, 3>::from_roots([2.0, 2.0, 0.0]);
        assert_almost_eq!(sq.to_coefficients()[0], 4.0, 1e-12);
        assert_almost_eq!(sq.to_coefficients()[1], -4.0, 1e-12);
        assert_almost_eq!(sq.to_coefficients()[2], 1.0, 1e-12);

        // Negative roots flip the sign pattern: (x+1)(x+3) = x^2 + 4x + 3.
        let neg = ArrayPolynomial::<f64, 3>::from_roots([-1.0, -3.0, 0.0]);
        assert_almost_eq!(neg.to_coefficients()[0], 3.0, 1e-12);
        assert_almost_eq!(neg.to_coefficients()[1], 4.0, 1e-12);
        assert_almost_eq!(neg.to_coefficients()[2], 1.0, 1e-12);

        // Degree 0 leaves the monic constant.
        let c = ArrayPolynomial::<f64, 1>::from_roots([7.0]);
        assert_almost_eq!(c.to_coefficients()[0], 1.0, 1e-12);
    }

    /// Round trip: every root fed to `from_roots` evaluates to zero, and the
    /// result is monic. This catches a coefficient permutation that a single
    /// hard-coded expansion would not.
    ///
    /// Trace: polynomial-design#FR-2
    /// Method: Requirements-based test
    #[cfg_attr(test, test)]
    fn test_from_roots_round_trip_evaluates_to_zero() {
        let roots = [0.5, -2.0, 4.0, 1.5, 0.0];
        let p = ArrayPolynomial::<f64, 5>::from_roots(roots);
        for r in &roots[..4] {
            assert_almost_eq!(p.evaluate(*r), 0.0, 1e-12);
        }
        // Monic in the top coefficient.
        assert_almost_eq!(p.to_coefficients()[4], 1.0, 1e-12);
        // A non-root does not evaluate to zero.
        assert!(p.evaluate(3.0).abs() > 1.0);
    }

    /// `cubic` is the Hermite basis on $t \in [0,1]$: it reproduces both
    /// endpoint positions and both endpoint slopes.
    ///
    /// Trace: polynomial-design#FR-6
    /// Method: Requirements-based test
    #[cfg_attr(test, test)]
    fn test_cubic_hermite_coefficients_and_endpoints() {
        // p0=0, p1=1, v0=0, v1=0 gives the smoothstep 3t^2 - 2t^3.
        let s = ArrayPolynomial::<f64, 4>::cubic(0.0, 1.0, 0.0, 0.0);
        assert_almost_eq!(s.to_coefficients()[0], 0.0, 1e-12);
        assert_almost_eq!(s.to_coefficients()[1], 0.0, 1e-12);
        assert_almost_eq!(s.to_coefficients()[2], 3.0, 1e-12);
        assert_almost_eq!(s.to_coefficients()[3], -2.0, 1e-12);

        // General segment: endpoints and slopes are reproduced exactly.
        let (p0, p1, v0, v1) = (1.0, -2.0, 0.5, 3.0);
        let c = ArrayPolynomial::<f64, 4>::cubic(p0, p1, v0, v1);
        assert_almost_eq!(c.evaluate(0.0), p0, 1e-12);
        assert_almost_eq!(c.evaluate(1.0), p1, 1e-12);
        let d = c.derivative();
        assert_almost_eq!(d.evaluate(0.0), v0, 1e-12);
        assert_almost_eq!(d.evaluate(1.0), v1, 1e-12);

        // A straight line is degree 1: the cubic terms vanish.
        let line = ArrayPolynomial::<f64, 4>::cubic(0.0, 1.0, 1.0, 1.0);
        assert_almost_eq!(line.to_coefficients()[0], 0.0, 1e-12);
        assert_almost_eq!(line.to_coefficients()[1], 1.0, 1e-12);
        assert_almost_eq!(line.to_coefficients()[2], 0.0, 1e-12);
        assert_almost_eq!(line.to_coefficients()[3], 0.0, 1e-12);
    }

    /// `quintic` reproduces position, velocity and acceleration at both ends.
    /// Each boundary condition pins a different coefficient group, so a wrong
    /// constant in any of `c2..c5` breaks at least one of them.
    ///
    /// Trace: polynomial-design#FR-6
    /// Method: Requirements-based test
    #[cfg_attr(test, test)]
    fn test_quintic_hermite_boundary_conditions() {
        // Rest-to-rest unit step: 10t^3 - 15t^4 + 6t^5.
        let s =
            ArrayPolynomial::<f64, 6>::quintic(0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
        assert_almost_eq!(s.to_coefficients()[0], 0.0, 1e-12);
        assert_almost_eq!(s.to_coefficients()[1], 0.0, 1e-12);
        assert_almost_eq!(s.to_coefficients()[2], 0.0, 1e-12);
        assert_almost_eq!(s.to_coefficients()[3], 10.0, 1e-12);
        assert_almost_eq!(s.to_coefficients()[4], -15.0, 1e-12);
        assert_almost_eq!(s.to_coefficients()[5], 6.0, 1e-12);

        let (p0, p1, v0, v1, a0, a1) = (2.0, -1.0, 0.25, -0.75, 1.5, -0.5);
        let q = ArrayPolynomial::<f64, 6>::quintic(p0, p1, v0, v1, a0, a1);
        let d1 = q.derivative();
        let d2 = d1.derivative();

        assert_almost_eq!(q.evaluate(0.0), p0, 1e-12);
        assert_almost_eq!(q.evaluate(1.0), p1, 1e-12);
        assert_almost_eq!(d1.evaluate(0.0), v0, 1e-12);
        assert_almost_eq!(d1.evaluate(1.0), v1, 1e-12);
        assert_almost_eq!(d2.evaluate(0.0), a0, 1e-12);
        assert_almost_eq!(d2.evaluate(1.0), a1, 1e-12);

        // c2 is exactly a0/2 and c1 is exactly v0, independent of the rest.
        let coeffs = q.to_coefficients();
        assert_almost_eq!(coeffs[0], p0, 1e-12);
        assert_almost_eq!(coeffs[1], v0, 1e-12);
        assert_almost_eq!(coeffs[2], a0 / 2.0, 1e-12);

        // A constant segment stays constant.
        let flat =
            ArrayPolynomial::<f64, 6>::quintic(3.0, 3.0, 0.0, 0.0, 0.0, 0.0);
        assert_almost_eq!(flat.to_coefficients()[0], 3.0, 1e-12);
        assert_almost_eq!(flat.to_coefficients()[1], 0.0, 1e-12);
        assert_almost_eq!(flat.to_coefficients()[2], 0.0, 1e-12);
        assert_almost_eq!(flat.to_coefficients()[3], 0.0, 1e-12);
        assert_almost_eq!(flat.to_coefficients()[4], 0.0, 1e-12);
        assert_almost_eq!(flat.to_coefficients()[5], 0.0, 1e-12);
    }

    /// `div_rem` reproduces long division: quotient and remainder satisfy
    /// $a = q b + r$ with $\deg r < \deg b$.
    ///
    /// Trace: polynomial-design#FR-4
    /// Method: Requirements-based test
    #[cfg_attr(test, test)]
    fn test_div_rem_known_divisions() {
        // (x^3 - 1) / (x - 1) = x^2 + x + 1, remainder 0.
        let a =
            ArrayPolynomial::<f64, 4>::from_coefficients([-1.0, 0.0, 0.0, 1.0]);
        let b = ArrayPolynomial::<f64, 2>::from_coefficients([-1.0, 1.0]);
        let (q, r) = a.div_rem::<2, 3, 1>(&b).unwrap();
        assert_almost_eq!(q.to_coefficients()[0], 1.0, 1e-12);
        assert_almost_eq!(q.to_coefficients()[1], 1.0, 1e-12);
        assert_almost_eq!(q.to_coefficients()[2], 1.0, 1e-12);
        assert_almost_eq!(r.to_coefficients()[0], 0.0, 1e-12);

        // (x^2 + 1) / (x - 1) = x + 1, remainder 2.
        let c = ArrayPolynomial::<f64, 3>::from_coefficients([1.0, 0.0, 1.0]);
        let (q2, r2) = c.div_rem::<2, 2, 1>(&b).unwrap();
        assert_almost_eq!(q2.to_coefficients()[0], 1.0, 1e-12);
        assert_almost_eq!(q2.to_coefficients()[1], 1.0, 1e-12);
        assert_almost_eq!(r2.to_coefficients()[0], 2.0, 1e-12);

        // A non-monic divisor scales the quotient: (2x^2) / (2x) = x.
        let d = ArrayPolynomial::<f64, 3>::from_coefficients([0.0, 0.0, 2.0]);
        let e = ArrayPolynomial::<f64, 2>::from_coefficients([0.0, 2.0]);
        let (q3, r3) = d.div_rem::<2, 2, 1>(&e).unwrap();
        assert_almost_eq!(q3.to_coefficients()[0], 0.0, 1e-12);
        assert_almost_eq!(q3.to_coefficients()[1], 1.0, 1e-12);
        assert_almost_eq!(r3.to_coefficients()[0], 0.0, 1e-12);
    }

    /// Division rejects a zero-leading divisor and a divisor of higher degree
    /// than the dividend.
    ///
    /// Trace: polynomial-design#FR-4
    /// Method: Requirements-based test
    #[cfg_attr(test, test)]
    fn test_div_rem_rejects_degenerate_divisors() {
        let a = ArrayPolynomial::<f64, 3>::from_coefficients([1.0, 2.0, 3.0]);

        let zero = ArrayPolynomial::<f64, 2>::from_coefficients([0.0, 0.0]);
        assert_eq!(
            a.div_rem::<2, 2, 1>(&zero).err(),
            Some(DivisionError::ZeroLeadingCoefficient)
        );

        // A divisor of higher degree than the dividend is rejected: here the
        // dividend is degree 1 and the divisor degree 2.
        let low = ArrayPolynomial::<f64, 3>::from_coefficients([1.0, 1.0, 0.0]);
        let high =
            ArrayPolynomial::<f64, 3>::from_coefficients([1.0, 1.0, 1.0]);
        assert_eq!(
            low.div_rem::<3, 1, 2>(&high).err(),
            Some(DivisionError::DegreeMismatch)
        );

        // Dividing by a degree-0 constant is legal and returns the scaled
        // dividend with no remainder.
        let one = ArrayPolynomial::<f64, 2>::from_coefficients([2.0, 0.0]);
        let (q, r) = a.div_rem::<2, 2, 1>(&one).unwrap();
        assert_almost_eq!(q.to_coefficients()[0], 0.5, 1e-12);
        assert_almost_eq!(q.to_coefficients()[1], 1.0, 1e-12);
        assert_almost_eq!(r.to_coefficients()[0], 0.0, 1e-12);
    }

    /// The companion matrix carries subdiagonal ones and the negated
    /// coefficients in its last column, and is rejected for a non-monic
    /// polynomial.
    ///
    /// Trace: polynomial-design#FR-5
    /// Method: Requirements-based test
    #[cfg_attr(test, test)]
    fn test_companion_matrix_structure() {
        // x^3 - 6x^2 + 11x - 6, ascending [-6, 11, -6, 1].
        let p = ArrayPolynomial::<f64, 4>::from_coefficients([
            -6.0, 11.0, -6.0, 1.0,
        ]);
        let c = p.companion_matrix::<3>().unwrap();

        // Subdiagonal ones.
        assert_eq!(c.get(1, 0), Some(&1.0));
        assert_eq!(c.get(2, 1), Some(&1.0));
        // Everything else outside the last column is zero.
        assert_eq!(c.get(0, 0), Some(&0.0));
        assert_eq!(c.get(0, 1), Some(&0.0));
        assert_eq!(c.get(1, 1), Some(&0.0));
        assert_eq!(c.get(2, 0), Some(&0.0));
        // Last column is -c_i.
        assert_eq!(c.get(0, 2), Some(&6.0));
        assert_eq!(c.get(1, 2), Some(&-11.0));
        assert_eq!(c.get(2, 2), Some(&6.0));

        let non_monic =
            ArrayPolynomial::<f64, 3>::from_coefficients([1.0, 2.0, 3.0]);
        assert!(non_monic.companion_matrix::<2>().is_err());
        assert!(Owned::<f64, 2, 2>::try_from(&non_monic).is_err());
    }

    /// Eigenvalue correspondence: each root $\lambda$ of the polynomial
    /// satisfies $C v = \lambda v$ for the companion matrix $C$, with the
    /// Horner tail $v = [\lambda^2 - 6\lambda + 11,\; \lambda - 6,\; 1]$.
    ///
    /// This is the property the companion form exists for, and no structural
    /// assertion implies it.
    ///
    /// Trace: polynomial-design#FR-5
    /// Method: Requirements-based test
    #[cfg_attr(test, test)]
    fn test_companion_matrix_eigenvalue_correspondence() {
        let p = ArrayPolynomial::<f64, 4>::from_roots([1.0, 2.0, 3.0, 0.0]);
        let c = p.companion_matrix::<3>().unwrap();

        for lambda in [1.0_f64, 2.0, 3.0] {
            let v = [lambda * lambda - 6.0 * lambda + 11.0, lambda - 6.0, 1.0];
            for i in 0..3 {
                let mut row = 0.0;
                for (j, vj) in v.iter().enumerate() {
                    row += c.get(i, j).copied().unwrap_or(0.0) * vj;
                }
                assert_almost_eq!(row, lambda * v[i], 1e-12);
            }
        }
    }

    /// `aberth_initial_seeds` places `deg` points on the circle of radius
    /// $1 + \max_i |c_i / c_n|$, evenly spaced with a $\pi/4n$ offset so no
    /// seed starts on the real axis.
    ///
    /// Trace: polynomial-design#FR-3
    /// Method: Requirements-based test
    #[cfg_attr(test, test)]
    fn test_aberth_initial_seeds_lie_on_the_cauchy_circle() {
        // x^3 - 6x^2 + 11x - 6: max |c_i| = 11, so radius = 12.
        let p = ArrayPolynomial::<f64, 4>::from_coefficients([
            -6.0, 11.0, -6.0, 1.0,
        ]);
        let seeds = p.aberth_initial_seeds(3, 1.0);

        for seed in seeds.iter().take(3) {
            let modulus = seed.re.hypot(seed.im);
            assert_almost_eq!(modulus, 12.0, 1e-12);
            // The pi/4n offset keeps every seed off the real axis.
            assert!(seed.im.abs() > 1e-9);
        }

        // Seeds are distinct, which the Aberth update divides by.
        for i in 0..3 {
            for j in (i + 1)..3 {
                let dr = seeds[i].re - seeds[j].re;
                let di = seeds[i].im - seeds[j].im;
                assert!(dr * dr + di * di > 1e-6);
            }
        }

        // Entries past `deg` are left at the origin.
        assert_almost_eq!(seeds[3].re, 0.0, 1e-12);
        assert_almost_eq!(seeds[3].im, 0.0, 1e-12);

        // The radius tracks the coefficients: scaling the leading term down
        // by 2 doubles every ratio and so grows the circle.
        let q = ArrayPolynomial::<f64, 3>::from_coefficients([4.0, 0.0, 1.0]);
        let qs = q.aberth_initial_seeds(2, 1.0);
        let qm = qs[0].re.hypot(qs[0].im);
        assert_almost_eq!(qm, 5.0, 1e-12);
    }

    /// `aberth_solver` converges to the exact roots of a split polynomial and
    /// to the conjugate pair of an irreducible one.
    ///
    /// Trace: polynomial-design#FR-3
    /// Method: Requirements-based test
    #[cfg_attr(test, test)]
    fn test_aberth_solver_finds_exact_roots() {
        let p = ArrayPolynomial::<f64, 4>::from_roots([1.0, 2.0, 3.0, 0.0]);
        let roots = p.aberth_roots().unwrap();
        let mut re = [roots[0].re, roots[1].re, roots[2].re];
        re.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        assert_almost_eq!(re[0], 1.0, 1e-9);
        assert_almost_eq!(re[1], 2.0, 1e-9);
        assert_almost_eq!(re[2], 3.0, 1e-9);
        for root in roots.iter().take(3) {
            assert!(root.im.abs() < 1e-9);
        }

        // x^2 + 1 has roots +/- i.
        let c = ArrayPolynomial::<f64, 3>::from_coefficients([1.0, 0.0, 1.0]);
        let cr = c.aberth_roots().unwrap();
        for root in cr.iter().take(2) {
            assert_almost_eq!(root.re, 0.0, 1e-8);
            assert_almost_eq!(root.im.abs(), 1.0, 1e-8);
        }
        assert!(cr[0].im * cr[1].im < 0.0);

        // A repeated root still converges, at the reduced rate the method
        // gives for multiplicity 2.
        let d = ArrayPolynomial::<f64, 3>::from_roots([2.0, 2.0, 0.0]);
        let dr = d.aberth_roots().unwrap();
        assert_almost_eq!(dr[0].re, 2.0, 1e-4);
        assert_almost_eq!(dr[1].re, 2.0, 1e-4);

        // Every returned root is a root: residual at the solution is zero.
        for root in roots.iter().take(3) {
            let val = p.evaluate_complex(*root);
            assert!(val.re.abs() < 1e-8 && val.im.abs() < 1e-8);
        }
    }

    /// The Aberth iteration is capped at 80 sweeps, so a polynomial it cannot
    /// resolve must still return rather than spin.
    ///
    /// A high-multiplicity root converges only linearly and does not reach the
    /// tolerance inside the budget, which is exactly the case the cap exists
    /// for. The contract is termination with finite output, not accuracy.
    ///
    /// Trace: polynomial-design#NFR-1
    /// Method: Requirements-based test
    #[cfg_attr(test, test)]
    fn test_aberth_terminates_on_a_non_converging_polynomial() {
        // (x - 1)^5: multiplicity 5, the slowest case for the method.
        let p = ArrayPolynomial::<f64, 6>::from_roots([1.0; 6]);
        let roots = p.aberth_roots().unwrap();

        for root in roots.iter().take(5) {
            assert!(
                root.re.is_finite() && root.im.is_finite(),
                "the iteration budget must return finite values"
            );
            // Even unconverged, the seeds have moved toward the true root.
            assert!(root.re.abs() < 10.0);
        }

        // A polynomial with a zero leading coefficient is rejected before the
        // loop is entered at all.
        let degenerate =
            ArrayPolynomial::<f64, 3>::from_coefficients([1.0, 2.0, 0.0]);
        assert!(degenerate.aberth_roots().is_err());
    }
}
