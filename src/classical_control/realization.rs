//! Firmware realizations: direct forms, second-order sections, cascade,
//! factorization, quantization and section stability (FR-8 to FR-10,
//! FR-17 to FR-20).
//!
//! Every realization uses the denominator convention
//! `H(z) = (b_0 + b_1 z^-1 + ... + b_n z^-n) / (1 + a_1 z^-1 + ... + a_n z^-n)`.
//! Sections store `-a_1, -a_2` so each output is one `MulAcc` chain of
//! multiply accumulates that narrows once (C-8). `Df1` keeps two past
//! inputs and two past outputs in `T`; `Df2t` and `DirectForm2T` keep their
//! states in `T::Acc`, so only the output narrows. `update`, `reset` and
//! `set_coefficients` are infallible and run a fixed operation count set by
//! the order or the section count (NFR-4).

use super::{ClassicalError, Succ, TypeNum};
use crate::math::complex_num::Complex;
use crate::math::fixed_num::{Fixed, FixedRepr};
use crate::math::num_traits::{Float, MulAcc, Scalar};
use crate::math::num_types::{Const, Dim, DimAdd, DimMax};
use crate::math::ops::{
    SaturatingAdd, SaturatingDiv, SaturatingMul, SaturatingNeg, SaturatingSub,
};
use crate::polynomial::ArrayPolynomial;
use crate::transfer_function::ArrayTransferFunction;

#[cfg(kani)]
mod proofs {
    use super::*;

    #[kani::proof]
    pub fn prove_df1_fixed_update_total() {
        type Q = Fixed<i16, 13>;
        let raw: [i16; 10] = kani::any();
        let [b0, b1, b2, a1, a2, u1, u2, y1, y2, u] = raw.map(Q::from_bits);
        let mut s = Df1::from(SectionCoefficients { b0, b1, b2, a1, a2 });
        s.u1 = u1;
        s.u2 = u2;
        s.y1 = y1;
        s.y2 = y2;
        let y = s.update(u);
        assert!(y.to_bits() >= i16::MIN);
    }
}

mod sealed {
    /// Seals [`super::Section`] to the crate's section structures.
    pub trait Sealed {}
}

#[cfg(any(test, feature = "ets"))]
#[cfg_attr(not(test), control_rs_macros::ets_suite)]
/// Unit and ETS test suite for the firmware realizations.
pub mod tests {
    use super::*;
    use crate::math::num_traits::Trig;
    use crate::tensor::ArrayTensor;

    /// Samples per output sequence.
    const K: usize = if cfg!(miri) { 64 } else { 256 };
    /// Cross-check tolerance `<case>/output`, relative to the peak output.
    const OUTPUT_REL: f64 = 1e-12;
    /// Cross-check tolerance `<case>/sections`, relative to the peak
    /// response.
    const SECTIONS_REL: f64 = 1e-9;
    /// Frequency count and spacing of the response cross-check sweep.
    const SWEEP: usize = if cfg!(miri) { 16 } else { 64 };
    const SWEEP_STEP: f64 = if cfg!(miri) { 0.196 } else { 0.049 };
    /// Unit roundoff of `f64`.
    const U: f64 = f64::EPSILON / 2.0;

    type Q13 = Fixed<i16, 13>;
    type Q29 = Fixed<i32, 29>;
    type Coeffs = SectionCoefficients<f64>;
    type ErrorBound = Result<(f64, f64), ClassicalError>;
    type Pairs = [(f64, f64)];
    type TfPair = ([f64; 7], [f64; 7]);

    /// Next value of a 64-bit xorshift generator in `[-1, 1)`.
    fn uniform(state: &mut u64) -> f64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        let top = u32::try_from(*state >> 32).unwrap_or(0);
        f64::from(top) / 2_147_483_648.0 - 1.0
    }

    /// Deterministic excitation: a step plus two sinusoids, scaled.
    fn input(scale: f64) -> [f64; K] {
        core::array::from_fn(|k| {
            let t = f64::from(u16::try_from(k).unwrap_or(0));
            scale * (0.5 + 0.3 * Trig::sin(0.2 * t) + 0.2 * Trig::cos(1.3 * t))
        })
    }

    /// Direct difference equation
    /// `y_k = Σ b_i u_(k-i) - Σ a_i y_(k-i)` with `a_0 = 1`
    /// over `i >= 0` and `i >= 1` (independent reference).
    fn reference(num: &[f64], den: &[f64], seq: &[f64; K]) -> [f64; K] {
        let mut out = [0.0; K];
        for step in 0..K {
            let tap = |coef: &[f64], sig: &[f64]| -> f64 {
                coef.iter()
                    .enumerate()
                    .filter_map(|(lag, c)| {
                        step.checked_sub(lag)
                            .and_then(|idx| sig.get(idx))
                            .map(|v| c * v)
                    })
                    .sum()
            };
            let value = tap(num, seq) - tap(den, &out);
            if let Some(slot) = out.get_mut(step) {
                *slot = value;
            }
        }
        out
    }

    /// Maximum absolute difference relative to the peak of `want`.
    fn rel_err(got: &[f64], want: &[f64]) -> f64 {
        let peak = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let err = got
            .iter()
            .zip(want)
            .fold(0.0f64, |m, (g, w)| m.max((g - w).abs()));
        err / peak
    }

    /// Ascending coefficients of `prod (z - r)` over real roots and
    /// conjugate pairs `rho e^+/- jtheta`, padded to 9.
    fn poly(real: &[f64], pairs: &Pairs) -> [f64; 9] {
        let mut c = [0.0; 9];
        if let Some(c0) = c.first_mut() {
            *c0 = 1.0;
        }
        let factors =
            real.iter().map(|r| [-r, 1.0, 0.0]).chain(pairs.iter().map(
                |(rho, th)| [rho * rho, -2.0 * rho * Trig::cos(*th), 1.0],
            ));
        for f in factors {
            c = core::array::from_fn(|k| {
                f.iter()
                    .enumerate()
                    .filter_map(|(j, fj)| {
                        k.checked_sub(j).and_then(|i| c.get(i)).map(|v| fj * v)
                    })
                    .sum()
            });
        }
        c
    }

    /// Ascending `[f64; M]` prefix of a padded coefficient array.
    fn take<const M: usize>(c: &[f64; 9]) -> [f64; M] {
        core::array::from_fn(|i| c.get(i).copied().unwrap_or(0.0))
    }

    /// Descending (`z^-1`) coefficients of an ascending array, normalized
    /// by the leading denominator coefficient `lead`.
    fn desc<const M: usize>(c: &[f64; M], lead: f64) -> [f64; M] {
        let mut out = *c;
        out.reverse();
        out.map(|v| v / lead)
    }

    /// Runs a `DirectForm2T` built from the source TF against the direct
    /// recurrence of the same TF.
    fn check_df2t<const ORDER: usize, const D: usize>(
        num: &[f64; 9],
        den: &[f64; 9],
    ) -> Result<(), ClassicalError>
    where
        Const<ORDER>: Dim,
        TypeNum<ORDER>: DimAdd<Const<1>>,
        Const<D>: Dim<TypeNum = Succ<ORDER>>,
        TypeNum<D>: DimMax<TypeNum<D>, Output = TypeNum<D>>,
    {
        let (n, d) = (take::<D>(num), take::<D>(den));
        let tf = ArrayTransferFunction::<f64, D, D>::discrete(n, d, 0.01);
        let mut f = DirectForm2T::<f64, ORDER>::from_transfer_function(&tf)?;
        let u = input(1.0);
        let got = u.map(|x| f.update(x));
        let lead = d.last().copied().unwrap_or(1.0);
        let want = reference(&desc(&n, lead), &desc(&d, lead), &u);
        assert!(rel_err(&got, &want) <= OUTPUT_REL);
        Ok(())
    }

    #[cfg_attr(test, test)]
    /// `DirectForm2T` outputs for orders 1 to 4 meet the output tolerance
    /// against the direct recurrence (FR-8).
    fn df2t_matches_reference() {
        check_df2t::<1, 2>(
            &poly(&[-0.5], &[]),
            &poly(&[0.8], &[]).map(|v| 2.0 * v),
        )
        .unwrap();
        check_df2t::<2, 3>(
            &poly(&[], &[(1.0, 1.0)]),
            &poly(&[], &[(0.95, 0.3)]),
        )
        .unwrap();
        check_df2t::<3, 4>(
            &poly(&[-1.0, 0.5, 0.1], &[]),
            &poly(&[0.7, 0.2, -0.5], &[]),
        )
        .unwrap();
        check_df2t::<4, 5>(
            &poly(&[-1.0, 0.2], &[(1.0, 2.0)]),
            &poly(&[0.7, -0.4], &[(0.9, 0.6)]),
        )
        .unwrap();
    }

    #[cfg_attr(test, test)]
    /// After `reset` the output sequence equals that of a new realization
    /// exactly (FR-8).
    fn df2t_reset() {
        let tf = ArrayTransferFunction::<f64, 3, 3>::discrete(
            [0.2, 0.3, 0.1],
            [0.5, -1.2, 1.0],
            0.1,
        );
        let mut used =
            DirectForm2T::<f64, 2>::from_transfer_function(&tf).unwrap();
        let mut fresh = used;
        let u = input(1.0);
        for &x in &u {
            used.update(x);
        }
        used.reset();
        for &x in &u {
            assert_eq!(used.update(x).to_bits(), fresh.update(x).to_bits());
        }
    }

    #[cfg_attr(test, test)]
    /// A continuous source returns `NotDiscrete` (FR-8, FR-10).
    fn df2t_from_continuous() {
        let tf = ArrayTransferFunction::<f64, 2, 2>::continuous(
            [1.0, 1.0],
            [2.0, 1.0],
        );
        assert!(matches!(
            DirectForm2T::<f64, 1>::from_transfer_function(&tf),
            Err(ClassicalError::NotDiscrete)
        ));
        assert!(matches!(
            to_sections::<f64, 2, 2, 1>(&tf),
            Err(ClassicalError::NotDiscrete)
        ));
    }

    #[cfg_attr(test, test)]
    /// A zero leading denominator coefficient is rejected (FR-8).
    fn df2t_zero_leading_denominator() {
        let tf = ArrayTransferFunction::<f64, 1, 3>::discrete(
            [1.0],
            [1.0, 2.0, 0.0],
            0.1,
        );
        assert!(matches!(
            DirectForm2T::<f64, 2>::from_transfer_function(&tf),
            Err(ClassicalError::ZeroLeadingCoefficient)
        ));
    }

    #[cfg_attr(test, test)]
    /// `DirectForm2T<Fixed<i32, 29>, 2>` driven at full scale saturates
    /// without panic or wrap (FR-8).
    fn df2t_fixed_total() {
        let one = Q29::from_num(1.0);
        let a = [Q29::from_num(-0.5), Q29::from_num(0.25)];
        let mut f = DirectForm2T::<Q29, 2>::new(one, [one, one], a);
        let mut last = Q29::ZERO;
        for _ in 0..64 {
            last = f.update(Q29::MAX);
            assert!(last >= Q29::ZERO);
        }
        assert_eq!(last, Q29::MAX);
        for _ in 0..64 {
            last = f.update(Q29::MIN);
        }
        assert_eq!(last, Q29::MIN);
    }

    /// Section with poles `rho e^+/- jtheta`, zeros on the unit circle
    /// at `+/-phi` and gain `g`.
    fn biquad(rho: f64, theta: f64, phi: f64, g: f64) -> Coeffs {
        SectionCoefficients {
            b0: g,
            b1: -2.0 * g * Trig::cos(phi),
            b2: g,
            a1: -2.0 * rho * Trig::cos(theta),
            a2: rho * rho,
        }
    }

    /// Three-section test cascade with per-section gain `g`.
    fn cascade_case(g: f64) -> [Coeffs; 3] {
        [
            biquad(0.9, 0.4, 1.5, g),
            biquad(0.8, 1.1, 2.2, g),
            biquad(0.6, 2.0, 2.8, g),
        ]
    }

    /// Product of descending polynomials `p` (length 7) and `q`.
    fn conv(p: &[f64; 7], q: [f64; 3]) -> [f64; 7] {
        core::array::from_fn(|k| {
            q.iter()
                .enumerate()
                .filter_map(|(j, qj)| {
                    k.checked_sub(j).and_then(|i| p.get(i)).map(|v| qj * v)
                })
                .sum()
        })
    }

    /// Full transfer function of a cascade as descending `(b, a)` in `z^-1`.
    fn cascade_tf(c: &[Coeffs]) -> TfPair {
        let unit = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        c.iter().fold((unit, unit), |(b, a), s| {
            (conv(&b, [s.b0, s.b1, s.b2]), conv(&a, [1.0, s.a1, s.a2]))
        })
    }

    #[cfg_attr(test, test)]
    /// `BiquadCascade` outputs of `Df1` and `Df2t` sections meet the output
    /// tolerance against the direct recurrence (FR-9).
    fn cascade_matches_reference() {
        let c = cascade_case(0.2);
        let (b, a) = cascade_tf(&c);
        let u = input(1.0);
        let want = reference(&b, &a, &u);
        let mut df1 = BiquadCascade::<Df1<f64>, 3>::from_coefficients(&c);
        let mut df2t = BiquadCascade::<Df2t<f64>, 3>::from_coefficients(&c);
        assert!(rel_err(&u.map(|x| df1.update(x)), &want) <= OUTPUT_REL);
        assert!(rel_err(&u.map(|x| df2t.update(x)), &want) <= OUTPUT_REL);
    }

    #[cfg_attr(test, test)]
    /// After `reset` every section state is zero (FR-9).
    fn cascade_reset() {
        let c = cascade_case(0.2);
        let mut df1 = BiquadCascade::<Df1<f64>, 3>::from_coefficients(&c);
        let mut df2t = BiquadCascade::<Df2t<f64>, 3>::from_coefficients(&c);
        for x in input(1.0) {
            df1.update(x);
            df2t.update(x);
        }
        df1.reset();
        df2t.reset();
        let zero = |v: &f64| v.to_bits() == 0;
        assert!(
            df1.sections()
                .iter()
                .all(|s| [s.u1, s.u2, s.y1, s.y2].iter().all(zero))
        );
        assert!(df2t.sections().iter().all(|s| zero(&s.d1) && zero(&s.d2)));
    }

    /// Frequency response of sections at `q = e^-jw`.
    fn sections_response(c: &[Coeffs], w: f64) -> Complex<f64> {
        let q = Complex::new(Trig::cos(w), -Trig::sin(w));
        let eval = |c0: f64, c1: f64, c2: f64| {
            let lin = Complex::new(c1, 0.0).saturating_mul(&q);
            let quad =
                Complex::new(c2, 0.0).saturating_mul(&q.saturating_mul(&q));
            Complex::new(c0, 0.0)
                .saturating_add(&lin)
                .saturating_add(&quad)
        };
        c.iter().fold(Complex::new(1.0, 0.0), |acc, s| {
            let ratio =
                eval(s.b0, s.b1, s.b2).saturating_div(&eval(1.0, s.a1, s.a2));
            acc.saturating_mul(&ratio)
        })
    }

    /// Factors a discrete TF and compares the cascade response to the
    /// source at 64 frequencies, relative to the peak source response.
    fn check_sections<const D: usize, const L: usize>(
        num: &[f64; 9],
        den: &[f64; 9],
    ) -> SectionsResult<f64, L>
    where
        Const<D>: Dim,
        TypeNum<D>: DimMax<TypeNum<D>, Output = TypeNum<D>>,
    {
        let tf = ArrayTransferFunction::<f64, D, D>::discrete(
            take(num),
            take(den),
            1.0,
        );
        let c = to_sections::<f64, D, D, L>(&tf)?;
        let w: [f64; SWEEP] = core::array::from_fn(|k| {
            SWEEP_STEP * f64::from(u8::try_from(k).unwrap_or(0)) + 0.01
        });
        let peak = w
            .iter()
            .fold(0.0f64, |m, wk| m.max(tf.eval_frequency(*wk).magnitude()));
        for wk in w {
            let want = tf.eval_frequency(wk);
            let got = sections_response(&c, wk);
            let err =
                Complex::new(got.re - want.re, got.im - want.im).magnitude();
            assert!(err <= SECTIONS_REL * peak, "D={D} w={wk} err={err}");
        }
        Ok(c)
    }

    #[cfg_attr(test, test)]
    /// Factored cascades reproduce the source response for even and odd
    /// orders up to 8, with zeros at infinity and at the origin (FR-10).
    fn sections_reproduce_source() {
        let p = [(0.95, 0.3), (0.8, 1.0), (0.6, 2.0), (0.5, 2.7)];
        let z = [(1.0, 0.9), (1.0, 1.7), (0.9, 2.5), (1.2, 0.2)];
        let pairs = |n: usize| p.get(..n).unwrap_or(&[]);
        let zpairs = |n: usize| z.get(..n).unwrap_or(&[]);
        check_sections::<2, 1>(&poly(&[-1.0], &[]), &poly(&[0.7], &[]))
            .unwrap();
        check_sections::<3, 1>(&poly(&[], zpairs(1)), &poly(&[], pairs(1)))
            .unwrap();
        check_sections::<4, 2>(&poly(&[], zpairs(1)), &poly(&[0.7], pairs(1)))
            .unwrap();
        check_sections::<5, 2>(
            &poly(&[0.0, -1.0], zpairs(1)),
            &poly(&[], pairs(2)),
        )
        .unwrap();
        check_sections::<6, 3>(
            &poly(&[-1.0], zpairs(2)),
            &poly(&[-0.3], pairs(2)),
        )
        .unwrap();
        check_sections::<7, 3>(&poly(&[0.5], zpairs(2)), &poly(&[], pairs(3)))
            .unwrap();
        check_sections::<8, 4>(
            &poly(&[-1.0, 0.4], zpairs(2)),
            &poly(&[0.2], pairs(3)),
        )
        .unwrap();
        check_sections::<9, 4>(&poly(&[], zpairs(4)), &poly(&[], pairs(4)))
            .unwrap();
    }

    #[cfg_attr(test, test)]
    /// Sections pair each pole with its nearest zero, poles taken in order
    /// of distance to the unit circle and placed from the last section
    /// backward; an odd order yields one section with `b2 = a2 = 0`
    /// (FR-10).
    fn sections_pairing_order() {
        let num = poly(&[-1.0], &[(1.0, 0.6), (1.0, 1.6)]);
        let den = poly(&[0.3], &[(0.9, 0.5), (0.5, 1.5)]);
        let c = check_sections::<6, 3>(&num, &den).unwrap();
        let [first, mid, last] = c;
        let near = |got: f64, want: f64| (got - want).abs() <= 1e-9;
        assert!(near(last.a1, -1.8 * Trig::cos(0.5f64)) && near(last.a2, 0.81));
        assert!(
            near(last.b1 / last.b0, -2.0 * Trig::cos(0.6f64))
                && near(last.b2 / last.b0, 1.0)
        );
        assert!(
            near(mid.a1, -2.0 * 0.5 * Trig::cos(1.5f64)) && near(mid.a2, 0.25)
        );
        assert!(near(mid.b1 / mid.b0, -2.0 * Trig::cos(1.6f64)));
        assert!(
            near(first.a1, -0.3)
                && first.a2.to_bits() == 0
                && first.b2.to_bits() == 0
        );
        assert!(near(first.b1 / first.b0, 1.0));
        let odd = c
            .iter()
            .filter(|s| s.a2.to_bits() == 0 && s.b2.to_bits() == 0)
            .count();
        assert_eq!(odd, 1);
        assert!(matches!(
            to_sections::<f64, 6, 6, 2>(&ArrayTransferFunction::discrete(
                take(&num),
                take(&den),
                1.0
            )),
            Err(ClassicalError::SectionCount)
        ));
    }

    /// Peak of `sum_j |h_j|` over the impulse response of a `Df1` cascade.
    fn l1_gain(c: &[Coeffs; 3]) -> f64 {
        let mut s = BiquadCascade::<Df1<f64>, 3>::from_coefficients(c);
        (0..(if cfg!(miri) { 1024u16 } else { 4096 }))
            .map(|k| s.update(if k == 0 { 1.0 } else { 0.0 }).abs())
            .sum()
    }

    #[cfg_attr(test, test)]
    /// For `f64` sections, `Df1` and `Df2t` agree within
    /// `10 L u sum |h|` of the peak output (FR-17).
    fn df1_matches_df2t() {
        let c = cascade_case(0.2);
        let u = input(1.0);
        let mut df1 = BiquadCascade::<Df1<f64>, 3>::from_coefficients(&c);
        let mut df2t = BiquadCascade::<Df2t<f64>, 3>::from_coefficients(&c);
        let y2 = u.map(|x| df2t.update(x));
        let y1 = u.map(|x| df1.update(x));
        assert!(rel_err(&y1, &y2) <= 10.0 * 3.0 * U * l1_gain(&c));
    }

    /// Noise gain `0.5 Σ_s ||g_s||_1` in units of `Delta`: the
    /// rounding at section `s` passes its own poles and every later
    /// section.
    fn noise_bound(c: &[Coeffs; 3]) -> f64 {
        let half: f64 = (0..3usize)
            .map(|s| {
                let mut chain = *c;
                for (i, sec) in chain.iter_mut().enumerate() {
                    if i < s {
                        *sec = SectionCoefficients {
                            b0: 1.0,
                            b1: 0.0,
                            b2: 0.0,
                            a1: 0.0,
                            a2: 0.0,
                        };
                    } else if i == s {
                        sec.b0 = 1.0;
                        sec.b1 = 0.0;
                        sec.b2 = 0.0;
                    }
                }
                l1_gain(&chain)
            })
            .sum();
        0.5 * half
    }

    /// `Df1<Fixed>` cascade output against the `f64` realization of the
    /// same quantized sections, in units of `delta`.
    fn fixed_error<R: FixedRepr, const S: usize>(scale: f64) -> ErrorBound
    where
        Const<S>: Dim + DimMax<R::BitsDim, Output = R::BitsDim>,
        Fixed<R, S>: Scalar + SaturatingNeg + MulAcc + PartialOrd,
    {
        let q = quantize::<R, S, 3>(&cascade_case(0.2))?;
        let c64 = q.map(|s| SectionCoefficients {
            b0: s.b0.to_num(),
            b1: s.b1.to_num(),
            b2: s.b2.to_num(),
            a1: s.a1.to_num(),
            a2: s.a2.to_num(),
        });
        let mut fixed =
            BiquadCascade::<Df1<Fixed<R, S>>, 3>::from_coefficients(&q);
        let mut float = BiquadCascade::<Df1<f64>, 3>::from_coefficients(&c64);
        let delta = Fixed::<R, S>::DELTA.to_num();
        let err = input(scale).iter().fold(0.0f64, |m, &x| {
            let xq = Fixed::<R, S>::from_num(x);
            let y = fixed.update(xq).to_num();
            m.max((y - float.update(xq.to_num())).abs())
        });
        Ok((err / delta, noise_bound(&c64)))
    }

    #[cfg_attr(test, test)]
    /// `Df1<Fixed<i16, 13>>` and `Df1<Fixed<i32, 29>>` outputs stay within
    /// the noise-gain bound of the `f64` realization (FR-17).
    fn df1_fixed_matches_reference() {
        let (e13, b13) = fixed_error::<i16, 13>(0.25).unwrap();
        assert!(e13 <= b13, "Q13 {e13} > {b13}");
        let (e29, b29) = fixed_error::<i32, 29>(0.25).unwrap();
        assert!(e29 <= b29, "Q29 {e29} > {b29}");
    }

    /// Rounds `x / 2^13` with ties to even.
    const fn round13(x: i64) -> i64 {
        let quot = x.div_euclid(8192);
        let twice = x.rem_euclid(8192).saturating_mul(2);
        if twice > 8192 || (twice == 8192 && quot.rem_euclid(2) == 1) {
            quot.saturating_add(1)
        } else {
            quot
        }
    }

    #[cfg_attr(test, test)]
    /// Without saturation each `Df1<Fixed<i16, 13>>` output equals the
    /// exact sum of its five products rounded once with ties to even
    /// (FR-17).
    fn df1_fixed_single_rounding() {
        let c = quantize::<i16, 13, 1>(&[biquad(0.9, 0.4, 1.5, 0.2)]).unwrap();
        let [c] = c;
        let mut s = Df1::from(c);
        let mut state = 0x2545_F491_4F6C_DD1D_u64;
        for _ in 0..(if cfg!(miri) { 200 } else { 2_000 }) {
            let u = Q13::from_num(0.5 * uniform(&mut state));
            let raw = [s.b0, s.b1, s.b2, s.na1, s.na2]
                .map(|v| i64::from(v.to_bits()));
            let sig =
                [u, s.u1, s.u2, s.y1, s.y2].map(|v| i64::from(v.to_bits()));
            let exact: i64 = raw.iter().zip(&sig).map(|(a, b)| a * b).sum();
            let want = Ord::clamp(
                round13(exact),
                i64::from(i16::MIN),
                i64::from(i16::MAX),
            );
            assert_eq!(i64::from(s.update(u).to_bits()), want);
        }
    }

    #[cfg_attr(test, test)]
    /// `set_coefficients` keeps the state, and the next output equals that
    /// of a section built with the new coefficients and the same state
    /// (FR-18).
    fn set_coefficients_keeps_state() {
        let [old, new, _] = cascade_case(0.3);
        let mut a = Df1::from(old);
        let mut b = Df2t::from(old);
        for x in input(1.0).iter().take(10) {
            a.update(*x);
            b.update(*x);
        }
        let (sa, sb) = (a, b);
        a.set_coefficients(&new);
        b.set_coefficients(&new);
        assert_eq!(
            [a.u1, a.u2, a.y1, a.y2].map(f64::to_bits),
            [sa.u1, sa.u2, sa.y1, sa.y2].map(f64::to_bits)
        );
        assert_eq!(
            [b.d1, b.d2].map(f64::to_bits),
            [sb.d1, sb.d2].map(f64::to_bits)
        );
        let mut ra = Df1 {
            u1: sa.u1,
            u2: sa.u2,
            y1: sa.y1,
            y2: sa.y2,
            ..Df1::from(new)
        };
        let mut rb = Df2t::from(new);
        rb.d1 = sb.d1;
        rb.d2 = sb.d2;
        assert_eq!(a.update(0.7).to_bits(), ra.update(0.7).to_bits());
        assert_eq!(b.update(0.7).to_bits(), rb.update(0.7).to_bits());
        let mut f = DirectForm2T::<f64, 2>::new(1.0, [0.5, 0.25], [-0.5, 0.1]);
        f.update(1.0);
        let state = f.state;
        f.set_coefficients(0.3, [0.2, 0.1], [0.4, 0.2]);
        assert_eq!(state.map(f64::to_bits), f.state.map(f64::to_bits));
        let mut cascade =
            BiquadCascade::<Df1<f64>, 3>::from_coefficients(&cascade_case(0.2));
        cascade.update(1.0);
        let before = *cascade.sections();
        cascade.set_coefficients(&cascade_case(0.4));
        for (x, y) in before.iter().zip(cascade.sections()) {
            assert_eq!(
                [x.u1, x.y1].map(f64::to_bits),
                [y.u1, y.y1].map(f64::to_bits)
            );
        }
    }

    #[cfg_attr(test, test)]
    /// Quantized coefficients lie within `delta / 2` of their source and an
    /// out-of-range coefficient names its section (FR-19).
    fn quantize_round_and_range() {
        let src = cascade_case(0.2);
        let q = quantize::<i16, 13, 3>(&src).unwrap();
        let delta = Q13::DELTA.to_num();
        for (s, d) in src.iter().zip(&q) {
            let pairs = [
                (s.b0, d.b0),
                (s.b1, d.b1),
                (s.b2, d.b2),
                (s.a1, d.a1),
                (s.a2, d.a2),
            ];
            assert!(
                pairs
                    .iter()
                    .all(|(x, y)| (x - y.to_num()).abs() <= delta / 2.0)
            );
        }
        let mut bad = src;
        if let Some(s) = bad.get_mut(1) {
            s.b1 = 4.5;
        }
        assert_eq!(
            quantize::<i16, 13, 3>(&bad),
            Err(ClassicalError::CoefficientRange { section: 1 })
        );
    }

    #[cfg_attr(test, test)]
    /// `is_stable` equals `max |pole| < 1` for pole pairs at radius 0.5,
    /// 0.99, 1.0 and 1.01 and for first-order sections (FR-20).
    fn stability_triangle() {
        for rho in [0.5, 0.99, 1.0, 1.01] {
            for k in 0..=32u8 {
                let theta = core::f64::consts::PI * f64::from(k) / 32.0;
                let c = biquad(rho, theta, 1.0, 1.0);
                assert_eq!(is_stable(&c), rho < 1.0, "rho={rho} theta={theta}");
            }
            let first = SectionCoefficients {
                b0: 1.0,
                b1: 0.0,
                b2: 0.0,
                a1: -rho,
                a2: 0.0,
            };
            assert_eq!(is_stable(&first), rho < 1.0);
            assert_eq!(
                all_stable(&[first, biquad(0.5, 1.0, 1.0, 1.0)]),
                rho < 1.0
            );
        }
        let q = Q13::from_num(0.75);
        let fixed = SectionCoefficients {
            b0: q,
            b1: q,
            b2: q,
            a1: Q13::from_num(-1.5),
            a2: q,
        };
        assert!(is_stable(&fixed));
    }

    #[cfg_attr(test, test)]
    /// Multilinear interpolation of a `Tensor` grid of stable sections is
    /// stable at 1,000 random query points (FR-20).
    fn interpolated_schedule_stable() {
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let mut grid = [[(0.0f64, 0.0f64); 5]; 4];
        for (r, row) in grid.iter_mut().enumerate() {
            for (c, cell) in row.iter_mut().enumerate() {
                let rho = 0.97 * (0.5 * uniform(&mut state) + 0.5);
                let theta =
                    core::f64::consts::PI * (0.5 * uniform(&mut state) + 0.5);
                let s = if (r + c) % 3 == 0 {
                    let mut sec = biquad(rho, theta, 1.0, 1.0);
                    sec.a1 = -(rho + 0.3 * rho);
                    sec.a2 = 0.3 * rho * rho;
                    sec
                } else {
                    biquad(rho, theta, 1.0, 1.0)
                };
                assert!(is_stable(&s));
                *cell = (s.a1, s.a2);
            }
        }
        let a1 = ArrayTensor::<f64, 4, 5>::from_fn(|i| {
            grid.get(i.first().copied().unwrap_or(0))
                .and_then(|row| row.get(i.get(1).copied().unwrap_or(0)))
                .map_or(0.0, |v| v.0)
        });
        let a2 = ArrayTensor::<f64, 4, 5>::from_fn(|i| {
            grid.get(i.first().copied().unwrap_or(0))
                .and_then(|row| row.get(i.get(1).copied().unwrap_or(0)))
                .map_or(0.0, |v| v.1)
        });
        for _ in 0..(if cfg!(miri) { 100 } else { 1_000 }) {
            let x = [
                1.5 * (uniform(&mut state) + 1.0),
                2.0 * (uniform(&mut state) + 1.0),
            ];
            let s = SectionCoefficients {
                b0: 1.0,
                b1: 0.0,
                b2: 0.0,
                a1: a1.interpolate(&x),
                a2: a2.interpolate(&x),
            };
            assert!(is_stable(&s));
        }
    }

    /// One section with `b0` set and the other coefficients zero.
    const fn with_b0(b0: f64) -> [Coeffs; 1] {
        [Coeffs {
            b0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
        }]
    }

    #[cfg_attr(test, test)]
    /// `quantize` accepts exactly the values that round into the
    /// representable range: `[MIN - delta/2, MAX + delta/2)` (FR-19).
    fn quantize_range_edges() {
        let delta: f64 = Q13::DELTA.to_num();
        let (min, max) = (-4.0, 4.0 - delta);
        assert!(quantize::<i16, 13, 1>(&with_b0(min)).is_ok());
        assert!(quantize::<i16, 13, 1>(&with_b0(max)).is_ok());
        let bad = Err(ClassicalError::CoefficientRange { section: 0 });
        assert_eq!(quantize::<i16, 13, 1>(&with_b0(min - delta)), bad);
        assert_eq!(quantize::<i16, 13, 1>(&with_b0(max + delta / 2.0)), bad);
    }

    #[cfg_attr(test, test)]
    /// The imaginary tolerance of `is_real` scales with `max(1, |z|)`.
    fn real_tolerance_scales_with_magnitude() {
        assert!(is_real(Complex::new(1000.0, 1e-6)));
        assert!(!is_real(Complex::new(1000.0, 1e-3)));
        assert!(!is_real(Complex::new(0.5, 1e-6)));
        assert!(is_real(Complex::new(0.5, 1e-9)));
    }

    #[cfg_attr(test, test)]
    /// Equal scores select the earliest remaining pole and zero.
    fn ties_select_earliest() {
        let (a, b) = (Complex::new(0.5, 0.5), Complex::new(0.5, -0.5));
        let mut poles = [Some(a), Some(b)];
        let got = take_pole(&mut poles, |_| true, |_| 1.0);
        assert_eq!(got, Some(a));
        assert_eq!(poles, [None, Some(b)]);
        let mut zeros = [Zero::At(a), Zero::At(b)];
        let target = Complex::new(0.5, 0.0);
        let got = take_zero(&mut zeros, |_| true, target);
        assert_eq!(got, Zero::At(a));
        assert_eq!(zeros, [Zero::Used, Zero::At(b)]);
    }

    #[cfg_attr(test, test)]
    /// A lone real pole is paired with a single real zero even when more
    /// real zeros remain (FR-10).
    fn lone_real_pole_takes_one_zero() {
        let c = check_sections::<4, 2>(
            &poly(&[-0.5, 0.3, 0.6], &[]),
            &poly(&[0.97], &[(0.8, 1.0)]),
        )
        .unwrap();
        let first_order = c
            .iter()
            .filter(|s| s.a2.to_bits() == 0 && s.b2.to_bits() == 0)
            .count();
        assert_eq!(first_order, 1);
    }

    #[cfg_attr(test, test)]
    /// A complex pole takes a complex zero pair when only one real zero
    /// remains, leaving the real zero for the real pole (FR-10).
    fn complex_pole_preserves_last_real_zero() {
        check_sections::<4, 2>(
            &poly(&[0.9], &[(1.0, 2.5)]),
            &poly(&[0.7], &[(0.95, 0.3)]),
        )
        .unwrap();
    }

    #[cfg_attr(test, test)]
    /// A real pole with other real poles remaining takes a complex zero
    /// pair when only one real zero is left, so the cascade matches the
    /// source (FR-10).
    fn real_pole_preserves_complex_zero_pair() {
        // num = (z - 0.85)(z^2 - z + 0.5), den = (z - 0.9)(z - 0.5)(z - 0.2).
        // Without the complex-zero preference the first real pole claimed
        // 0.85 and left z2 unused, dropping 0.5 +/- 0.5j.
        check_sections::<4, 2>(
            &poly(&[0.85], &[(0.5f64.sqrt(), core::f64::consts::FRAC_PI_4)]),
            &poly(&[0.9, 0.5, 0.2], &[]),
        )
        .unwrap();
    }

    #[cfg_attr(test, test)]
    /// A constant discrete TF keeps its gain in a single section (FR-10).
    fn constant_tf_section_holds_gain() {
        let tf =
            ArrayTransferFunction::<f64, 1, 1>::discrete([3.0], [2.0], 0.1);
        assert!(matches!(
            to_sections::<f64, 1, 1, 0>(&tf),
            Err(ClassicalError::SectionCount)
        ));
        let c = to_sections::<f64, 1, 1, 1>(&tf).unwrap();
        assert_eq!(c.len(), 1);
        assert!((c[0].b0 - 1.5).abs() < 1e-15);
        assert_eq!(c[0].b1.to_bits(), 0);
        assert_eq!(c[0].b2.to_bits(), 0);
        assert_eq!(c[0].a1.to_bits(), 0);
        assert_eq!(c[0].a2.to_bits(), 0);
        let mut cascade = BiquadCascade::<Df1<f64>, 1>::from_coefficients(&c);
        assert!((cascade.update(1.0) - 1.5).abs() < 1e-15);
    }
}

/// `L` section coefficient sets, first section first.
pub type SectionArray<T, const L: usize> = [SectionCoefficients<T>; L];

/// A slice of section coefficient sets.
pub type SectionSlice<T> = [SectionCoefficients<T>];

/// Quantized sections or the first section out of range.
pub type QuantizeResult<Repr, const SHIFT: usize, const L: usize> =
    Result<SectionArray<Fixed<Repr, SHIFT>, L>, ClassicalError>;

/// Factored sections or the reason factorization failed.
pub type SectionsResult<T, const L: usize> =
    Result<SectionArray<T, L>, ClassicalError>;

/// Accumulator states of a realization.
type AccState<T, const N: usize> = [<T as MulAcc>::Acc; N];

/// Numerator and denominator arrays.
type ArrayPair<T, const D: usize> = ([T; D], [T; D]);

/// Root buffer.
type Roots<T, const D: usize> = [Complex<T>; D];

/// Remaining poles during pairing.
type PoleSlots<T, const D: usize> = [Option<Complex<T>>; D];

/// Remaining zeros during pairing.
type ZeroSlots<T, const D: usize> = [Zero<T>; D];

/// Zeros of a numerator or the reason they cannot be computed.
type ZerosResult<T, const D: usize> = Result<ZeroSlots<T, D>, ClassicalError>;

/// A pole taken from the remaining set.
type MaybePole<T> = Option<Complex<T>>;

/// First-order factor `[c_0, c_1]` in `q = z^-1`.
type Factor<T> = [Complex<T>; 2];

/// The two zeros of a section.
type ZeroPair<T> = [Zero<T>; 2];

/// The two poles of a section; `None` for a first-order section.
type PolePair<T> = [Option<Complex<T>>; 2];

/// Zeros and poles of one section.
type Pairing<T> = (ZeroPair<T>, PolePair<T>);

/// A zero slot during pairing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Zero<T> {
    /// A finite zero at the given location.
    At(Complex<T>),
    /// A zero at infinity: a pure delay factor `z^-1`.
    Infinite,
    /// Consumed or absent.
    Used,
}

/// Coefficients of one first- or second-order section in the
/// `1 + a_1 z^-1 + a_2 z^-2` denominator convention (FR-18).
///
/// A first-order section has `b_2 = a_2 = 0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SectionCoefficients<T> {
    /// `b_0`.
    pub b0: T,
    /// `b_1`.
    pub b1: T,
    /// `b_2`.
    pub b2: T,
    /// `a_1`.
    pub a1: T,
    /// `a_2`.
    pub a2: T,
}

/// Direct Form I second-order section (FR-17).
///
/// Holds `(b_0, b_1, b_2, -a_1, -a_2)` and
/// `(u_k-1, u_k-2, y_k-1, y_k-2)`; each output is one five-term
/// `MulAcc` chain narrowed once. DF1 keeps its state in signal units and
/// overflows internally only if the output does, which makes it the
/// structure for fixed point and for scheduled coefficients.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Df1<T> {
    b0: T,
    b1: T,
    b2: T,
    na1: T,
    na2: T,
    u1: T,
    u2: T,
    y1: T,
    y2: T,
}

/// Transposed Direct Form II second-order section.
///
/// Holds `(b_0, b_1, b_2, -a_1, -a_2)` and two states in `T::Acc`, the
/// wide dynamic range TDF-II states need; half the state of [`Df1`].
#[derive(Clone, Copy)]
pub struct Df2t<T: MulAcc> {
    b0: T,
    b1: T,
    b2: T,
    na1: T,
    na2: T,
    d1: T::Acc,
    d2: T::Acc,
}

/// Transposed Direct Form II realization of order `ORDER` (FR-8).
#[derive(Clone, Copy)]
pub struct DirectForm2T<T: MulAcc, const ORDER: usize> {
    b0: T,
    b: [T; ORDER],
    na: [T; ORDER],
    state: AccState<T, ORDER>,
}

/// Second-order sections of one structure run in series (FR-9).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BiquadCascade<S, const L: usize> {
    sections: [S; L],
}

/// Run-time surface of a section: one sample per `update`, `reset` and
/// coefficient replacement that keeps the state (FR-18).
///
/// Sealed: implemented by [`Df1`] and [`Df2t`].
pub trait Section<T>: sealed::Sealed {
    /// Processes one input sample and returns the output.
    fn update(&mut self, u: T) -> T;
    /// Sets every state to zero.
    fn reset(&mut self);
    /// Replaces the coefficients without clearing the state.
    fn set_coefficients(&mut self, c: &SectionCoefficients<T>);
}

impl<T> sealed::Sealed for Df1<T> {}

impl<T: MulAcc> sealed::Sealed for Df2t<T> {}

impl<T: Scalar + SaturatingNeg + MulAcc> From<SectionCoefficients<T>>
    for Df1<T>
{
    fn from(c: SectionCoefficients<T>) -> Self {
        Self {
            b0: c.b0,
            b1: c.b1,
            b2: c.b2,
            na1: c.a1.saturating_neg(),
            na2: c.a2.saturating_neg(),
            u1: T::ZERO,
            u2: T::ZERO,
            y1: T::ZERO,
            y2: T::ZERO,
        }
    }
}

impl<T: Scalar + SaturatingNeg + MulAcc> From<SectionCoefficients<T>>
    for Df2t<T>
{
    fn from(c: SectionCoefficients<T>) -> Self {
        let zero = T::ZERO.to_acc();
        Self {
            b0: c.b0,
            b1: c.b1,
            b2: c.b2,
            na1: c.a1.saturating_neg(),
            na2: c.a2.saturating_neg(),
            d1: zero,
            d2: zero,
        }
    }
}

impl<T: Scalar + SaturatingNeg + MulAcc> Section<T> for Df1<T> {
    #[inline]
    fn update(&mut self, u: T) -> T {
        let acc = T::mac(T::ZERO.to_acc(), self.b0, u);
        let acc = T::mac(acc, self.b1, self.u1);
        let acc = T::mac(acc, self.b2, self.u2);
        let acc = T::mac(acc, self.na1, self.y1);
        let y = T::from_acc(T::mac(acc, self.na2, self.y2));
        self.u2 = self.u1;
        self.u1 = u;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }

    fn reset(&mut self) {
        self.u1 = T::ZERO;
        self.u2 = T::ZERO;
        self.y1 = T::ZERO;
        self.y2 = T::ZERO;
    }

    fn set_coefficients(&mut self, c: &SectionCoefficients<T>) {
        self.b0 = c.b0;
        self.b1 = c.b1;
        self.b2 = c.b2;
        self.na1 = c.a1.saturating_neg();
        self.na2 = c.a2.saturating_neg();
    }
}

impl<T: Scalar + SaturatingNeg + MulAcc> Section<T> for Df2t<T> {
    #[inline]
    fn update(&mut self, u: T) -> T {
        let y = T::from_acc(T::mac(self.d1, self.b0, u));
        self.d1 = T::mac(T::mac(self.d2, self.b1, u), self.na1, y);
        self.d2 = T::mac(T::mac(T::ZERO.to_acc(), self.b2, u), self.na2, y);
        y
    }

    fn reset(&mut self) {
        self.d1 = T::ZERO.to_acc();
        self.d2 = T::ZERO.to_acc();
    }

    fn set_coefficients(&mut self, c: &SectionCoefficients<T>) {
        self.b0 = c.b0;
        self.b1 = c.b1;
        self.b2 = c.b2;
        self.na1 = c.a1.saturating_neg();
        self.na2 = c.a2.saturating_neg();
    }
}

impl<T: Scalar + SaturatingNeg + MulAcc, const ORDER: usize>
    DirectForm2T<T, ORDER>
{
    /// Builds a realization from `b_0`, `(b_1, ..., b_n)` and
    /// `(a_1, ..., a_n)` with zero state.
    #[must_use]
    pub fn new(b0: T, b: [T; ORDER], a: [T; ORDER]) -> Self {
        Self {
            b0,
            b,
            na: a.map(|v| v.saturating_neg()),
            state: [T::ZERO.to_acc(); ORDER],
        }
    }

    /// Processes one input sample and returns the output.
    #[inline]
    pub fn update(&mut self, u: T) -> T {
        let zero = T::ZERO.to_acc();
        let first = self.state.first().copied().unwrap_or(zero);
        let y = T::from_acc(T::mac(first, self.b0, u));
        for i in 0..ORDER {
            let next =
                self.state.get(i.saturating_add(1)).copied().unwrap_or(zero);
            if let (Some(d), Some(&bi), Some(&ai)) =
                (self.state.get_mut(i), self.b.get(i), self.na.get(i))
            {
                *d = T::mac(T::mac(next, bi, u), ai, y);
            }
        }
        y
    }

    /// Sets every state to zero.
    pub fn reset(&mut self) {
        self.state = [T::ZERO.to_acc(); ORDER];
    }

    /// Replaces the coefficients without clearing the state (FR-18).
    pub fn set_coefficients(&mut self, b0: T, b: [T; ORDER], a: [T; ORDER]) {
        self.b0 = b0;
        self.b = b;
        self.na = a.map(|v| v.saturating_neg());
    }
}

impl<T: Float + Copy + MulAcc, const ORDER: usize> DirectForm2T<T, ORDER> {
    /// Builds a realization from a discrete `TransferFunction` divided
    /// through by its leading denominator coefficient (FR-8).
    ///
    /// `ORDER` is the denominator degree `D - 1` and `N <= D`.
    ///
    /// # Errors
    /// - [`ClassicalError::NotDiscrete`] when `tf` is continuous.
    /// - [`ClassicalError::ZeroLeadingCoefficient`] when the leading
    ///   denominator coefficient has absolute value at most `T::epsilon()`.
    pub fn from_transfer_function<const N: usize, const D: usize>(
        tf: &ArrayTransferFunction<T, N, D>,
    ) -> Result<Self, ClassicalError>
    where
        Const<N>: Dim,
        Const<ORDER>: Dim,
        TypeNum<ORDER>: DimAdd<Const<1>>,
        Const<D>: Dim<TypeNum = Succ<ORDER>>,
        TypeNum<N>: DimMax<TypeNum<D>, Output = TypeNum<D>>,
    {
        if tf.is_continuous() {
            return Err(ClassicalError::NotDiscrete);
        }
        let (num, den) = descending::<T, N, D>(tf);
        let lead = den.first().copied().unwrap_or(T::ONE);
        match lead.abs().partial_cmp(&T::epsilon()) {
            Some(core::cmp::Ordering::Greater) => {}
            _ => return Err(ClassicalError::ZeroLeadingCoefficient),
        }
        let scale = |v: &T| v.saturating_div(&lead);
        let mut b = [T::ZERO; ORDER];
        let mut a = [T::ZERO; ORDER];
        b.iter_mut()
            .zip(num.iter().skip(1))
            .for_each(|(o, v)| *o = scale(v));
        a.iter_mut()
            .zip(den.iter().skip(1))
            .for_each(|(o, v)| *o = scale(v));
        let b0 = num.first().map_or(T::ZERO, scale);
        Ok(Self::new(b0, b, a))
    }
}

impl<S, const L: usize> BiquadCascade<S, L> {
    /// Builds a cascade from its sections, first section first.
    #[must_use]
    pub const fn new(sections: [S; L]) -> Self {
        Self { sections }
    }

    /// Builds a cascade with zero state from section coefficients.
    #[must_use]
    pub fn from_coefficients<T: Copy>(c: &SectionArray<T, L>) -> Self
    where
        S: From<SectionCoefficients<T>>,
    {
        Self {
            sections: c.map(S::from),
        }
    }

    /// The sections in execution order.
    #[must_use]
    pub const fn sections(&self) -> &[S; L] {
        &self.sections
    }
}

impl<S, const L: usize> BiquadCascade<S, L> {
    /// Processes one input sample through every section in order.
    #[inline]
    pub fn update<T>(&mut self, u: T) -> T
    where
        S: Section<T>,
    {
        self.sections.iter_mut().fold(u, |x, s| s.update(x))
    }

    /// Replaces every section's coefficients without clearing state
    /// (FR-18).
    pub fn set_coefficients<T>(&mut self, c: &SectionArray<T, L>)
    where
        S: Section<T>,
    {
        self.sections
            .iter_mut()
            .zip(c)
            .for_each(|(s, ci)| s.set_coefficients(ci));
    }
}

impl<T: Scalar + SaturatingNeg + MulAcc, const L: usize>
    BiquadCascade<Df1<T>, L>
{
    /// Sets every section state to zero.
    pub fn reset(&mut self) {
        self.sections.iter_mut().for_each(Section::<T>::reset);
    }
}

impl<T: Scalar + SaturatingNeg + MulAcc, const L: usize>
    BiquadCascade<Df2t<T>, L>
{
    /// Sets every section state to zero.
    pub fn reset(&mut self) {
        self.sections.iter_mut().for_each(Section::<T>::reset);
    }
}

/// Reports whether a first- or second-order section has its poles strictly
/// inside the unit circle: `|a_2| < 1` and `|a_1| < 1 + a_2` (FR-20).
///
/// Exact for `Fixed`; no root finding.
#[must_use]
pub fn is_stable<T: Scalar + SaturatingNeg + PartialOrd + Copy>(
    c: &SectionCoefficients<T>,
) -> bool {
    let abs = |v: T| if v < T::ZERO { v.saturating_neg() } else { v };
    abs(c.a2) < T::ONE && abs(c.a1) < T::ONE.saturating_add(&c.a2)
}

/// Reports whether every section in `sections` passes [`is_stable`]
/// (FR-20).
#[must_use]
pub fn all_stable<T: Scalar + SaturatingNeg + PartialOrd + Copy>(
    sections: &SectionSlice<T>,
) -> bool {
    sections.iter().all(is_stable)
}

/// Quantizes floating-point section coefficients to `Fixed<Repr, SHIFT>`
/// with round-to-nearest (FR-19).
///
/// # Errors
/// [`ClassicalError::CoefficientRange`] naming the first section with a
/// coefficient that does not round into the representable range; no
/// coefficient is saturated.
pub fn quantize<Repr: FixedRepr, const SHIFT: usize, const L: usize>(
    sections: &SectionArray<f64, L>,
) -> QuantizeResult<Repr, SHIFT, L>
where
    Const<SHIFT>: Dim + DimMax<Repr::BitsDim, Output = Repr::BitsDim>,
    Fixed<Repr, SHIFT>: Scalar,
{
    let half = Fixed::<Repr, SHIFT>::DELTA.to_num() / 2.0;
    let lo = Fixed::<Repr, SHIFT>::MIN.to_num() - half;
    let hi = Fixed::<Repr, SHIFT>::MAX.to_num() + half;
    let mut out = [SectionCoefficients {
        b0: Fixed::ZERO,
        b1: Fixed::ZERO,
        b2: Fixed::ZERO,
        a1: Fixed::ZERO,
        a2: Fixed::ZERO,
    }; L];
    for ((dst, src), section) in out.iter_mut().zip(sections).zip(0usize..) {
        let all = [src.b0, src.b1, src.b2, src.a1, src.a2];
        if !all.iter().all(|v| *v >= lo && *v < hi) {
            return Err(ClassicalError::CoefficientRange { section });
        }
        let [b0, b1, b2, a1, a2] = all.map(Fixed::from_num);
        *dst = SectionCoefficients { b0, b1, b2, a1, a2 };
    }
    Ok(out)
}

/// Factors a discrete `TransferFunction` into `L` second-order sections
/// (FR-10).
///
/// Poles and zeros are paired as `SciPy` `zpk2sos` pairs them with
/// `pairing = "nearest"`: sections are formed starting with the pole
/// closest to the unit circle, each paired with its nearest remaining zero
/// and a conjugate or next real pole, and are placed from the last section
/// backward. For odd order the last remaining real pole forms the one
/// first-order section (`b_2 = a_2 = 0`) with its nearest real zero. The
/// overall gain goes into the first section. Zeros at infinity (numerator
/// degree below denominator degree) become pure delays in their section.
/// `L` must be `max(1, ceil((D - 1) / 2))` so a constant (`D = 1`) still
/// has one section to hold the overall gain.
///
/// # Errors
/// - [`ClassicalError::NotDiscrete`]: `tf` is continuous.
/// - [`ClassicalError::SectionCount`]: `L` does not match the order.
/// - [`ClassicalError::Root`]: poles or zeros cannot be computed, or the
///   numerator is zero.
pub fn to_sections<
    T: Float + Copy,
    const N: usize,
    const D: usize,
    const L: usize,
>(
    tf: &ArrayTransferFunction<T, N, D>,
) -> SectionsResult<T, L>
where
    Const<N>: Dim,
    Const<D>: Dim,
    TypeNum<N>: DimMax<TypeNum<D>, Output = TypeNum<D>>,
{
    if tf.is_continuous() {
        return Err(ClassicalError::NotDiscrete);
    }
    let order = D.saturating_sub(1);
    // Order 0 has no poles to pair. `L = 1` holds a gain-only section;
    // `L = 0` would return an empty product and drop the overall gain.
    let needed = if order == 0 { 1 } else { order.div_ceil(2) };
    if needed != L {
        return Err(ClassicalError::SectionCount);
    }
    let mut num = [T::ZERO; D];
    num.iter_mut()
        .zip(tf.num_slice())
        .for_each(|(o, &v)| *o = v);
    let mut poles = [None; D];
    let all_poles = tf.poles()?;
    for (dst, p) in poles.iter_mut().zip(all_poles).take(order) {
        *dst = Some(p);
    }
    let mut zeros = finite_zeros(&num, order)?;
    let lead_num = num
        .iter()
        .rev()
        .find(|v| **v != T::ZERO)
        .copied()
        .unwrap_or(T::ZERO);
    let lead_den = tf.den_slice().last().copied().unwrap_or(T::ONE);
    let mut out = pair_sections::<T, D, L>(&mut poles, &mut zeros);
    if let Some(first) = out.first_mut() {
        let g = lead_num.saturating_div(&lead_den);
        first.b0 = first.b0.saturating_mul(&g);
        first.b1 = first.b1.saturating_mul(&g);
        first.b2 = first.b2.saturating_mul(&g);
    }
    Ok(out)
}

/// Numerator and denominator in descending powers of `z`, padded to `D`.
fn descending<T: Float + Copy, const N: usize, const D: usize>(
    tf: &ArrayTransferFunction<T, N, D>,
) -> ArrayPair<T, D>
where
    Const<N>: Dim,
    Const<D>: Dim,
{
    let mut num = [T::ZERO; D];
    let mut den = [T::ZERO; D];
    num.iter_mut()
        .rev()
        .zip(tf.num_slice())
        .for_each(|(o, &v)| *o = v);
    den.iter_mut()
        .rev()
        .zip(tf.den_slice())
        .for_each(|(o, &v)| *o = v);
    (num, den)
}

/// Zeros of the ascending numerator `num` of a degree-`order` system:
/// exact zeros at the origin, finite zeros from the reversed polynomial and
/// zeros at infinity for each missing leading coefficient.
fn finite_zeros<T: Float + Copy, const D: usize>(
    num: &[T; D],
    order: usize,
) -> ZerosResult<T, D>
where
    Const<D>: Dim,
{
    let top = num.iter().rposition(|v| *v != T::ZERO);
    let low = num.iter().position(|v| *v != T::ZERO);
    let (Some(top), Some(low)) = (top, low) else {
        return Err(ClassicalError::Root(
            crate::polynomial::RootError::ZeroLeadingCoefficient,
        ));
    };
    // R(w) = Σ_j num[j + low] w^(order - j): nonzero leading coefficient.
    let mut rev = [T::ZERO; D];
    for (j, &c) in num.iter().skip(low).enumerate() {
        if let Some(slot) = rev.get_mut(order.saturating_sub(j)) {
            *slot = c;
        }
    }
    let mut w = ArrayPolynomial::<T, D>::from_coefficients(rev).roots()?;
    let finite = top.saturating_sub(low);
    sort_by_magnitude_desc(&mut w, order);
    let mut out = [Zero::Used; D];
    let one = Complex::new(T::ONE, T::ZERO);
    let kinds = w
        .iter()
        .take(finite)
        .map(|wi| Zero::At(one.saturating_div(wi)))
        .chain((0..low).map(|_| Zero::At(Complex::new(T::ZERO, T::ZERO))))
        .chain((0..order.saturating_sub(top)).map(|_| Zero::Infinite));
    out.iter_mut().zip(kinds).for_each(|(o, k)| *o = k);
    Ok(out)
}

/// Sorts the first `n` entries by descending magnitude (insertion sort,
/// initialization only).
fn sort_by_magnitude_desc<T: Float + Copy, const D: usize>(
    roots: &mut Roots<T, D>,
    count: usize,
) {
    let head = roots.get_mut(..count).unwrap_or(&mut []);
    for start in 1..head.len() {
        let mut pos = start;
        while pos > 0 {
            let prev = pos.saturating_sub(1);
            let swap = matches!(
                (head.get(prev), head.get(pos)),
                (Some(lhs), Some(rhs)) if rhs.magnitude() > lhs.magnitude()
            );
            if !swap {
                break;
            }
            head.swap(prev, pos);
            pos = prev;
        }
    }
}

/// Whether `z` is real within `sqrt(eps) max(1, |z|)`.
fn is_real<T: Float + Copy>(z: Complex<T>) -> bool {
    let mag = z.magnitude();
    let scale = if mag > T::ONE { mag } else { T::ONE };
    z.im.abs() <= T::epsilon().sqrt().saturating_mul(&scale)
}

/// Whether a zero slot is real (finite real or at infinity).
fn zero_is_real<T: Float + Copy>(z: Zero<T>) -> bool {
    match z {
        Zero::At(c) => is_real(c),
        Zero::Infinite => true,
        Zero::Used => false,
    }
}

/// Removes and returns the remaining pole minimizing `score` among those
/// passing `keep`.
fn take_pole<T: Float + Copy, const D: usize>(
    poles: &mut PoleSlots<T, D>,
    keep: impl Fn(Complex<T>) -> bool,
    score: impl Fn(Complex<T>) -> T,
) -> MaybePole<T> {
    let mut best = None;
    for (i, p) in poles.iter().enumerate() {
        if let Some(p) = p.filter(|p| keep(*p)) {
            let sc = score(p);
            if best.is_none_or(|(_, b)| sc < b) {
                best = Some((i, sc));
            }
        }
    }
    best.and_then(|(i, _)| poles.get_mut(i))
        .and_then(Option::take)
}

/// Removes and returns the remaining zero nearest `target` among those
/// passing `keep`; zeros at infinity rank after every finite zero.
fn take_zero<T: Float + Copy, const D: usize>(
    zeros: &mut ZeroSlots<T, D>,
    keep: impl Fn(Zero<T>) -> bool,
    target: Complex<T>,
) -> Zero<T> {
    let mut best = None;
    for (i, z) in zeros.iter().enumerate() {
        let sc = match *z {
            Zero::Used => continue,
            _ if !keep(*z) => continue,
            Zero::At(c) => Some(c.saturating_sub(&target).magnitude()),
            Zero::Infinite => None,
        };
        let better = match (best, sc) {
            (None, _) | (Some((_, None)), Some(_)) => true,
            (Some((_, Some(b))), Some(v)) => v < b,
            _ => false,
        };
        if better {
            best = Some((i, sc));
        }
    }
    best.and_then(|(i, _)| zeros.get_mut(i))
        .map_or(Zero::Used, |z| core::mem::replace(z, Zero::Used))
}

/// Factor `(1 - z q)` of a finite zero, `q` of a zero at infinity and `1`
/// of an absent zero, with `q = z^-1`.
fn zero_factor<T: Float + Copy>(z: Zero<T>) -> Factor<T> {
    let zero = Complex::new(T::ZERO, T::ZERO);
    let one = Complex::new(T::ONE, T::ZERO);
    match z {
        Zero::At(c) => [one, zero.saturating_sub(&c)],
        Zero::Infinite => [zero, one],
        Zero::Used => [one, zero],
    }
}

/// Real coefficients of the product of two first-order factors.
fn product<T: Float + Copy>(f: Factor<T>, g: Factor<T>) -> [T; 3] {
    let [f0, f1] = f;
    let [g0, g1] = g;
    [
        f0.saturating_mul(&g0).re,
        f0.saturating_mul(&g1)
            .saturating_add(&f1.saturating_mul(&g0))
            .re,
        f1.saturating_mul(&g1).re,
    ]
}

/// Section with zeros `z` and poles `p` (an absent pole contributes 1).
fn section<T: Float + Copy>(
    z: ZeroPair<T>,
    p: PolePair<T>,
) -> SectionCoefficients<T> {
    let [z1, z2] = z;
    let [b0, b1, b2] = product(zero_factor(z1), zero_factor(z2));
    let first_order = p[1].is_none();
    let [p1, p2] = p.map(|q| zero_factor(q.map_or(Zero::Used, Zero::At)));
    let [_, a1, a2] = product(p1, p2);
    let exact = |v: T, absent: bool| if absent { T::ZERO } else { v };
    SectionCoefficients {
        b0,
        b1,
        b2: exact(b2, matches!(z2, Zero::Used)),
        a1,
        a2: exact(a2, first_order),
    }
}

/// Distance of `p` from the unit circle.
fn circle_distance<T: Float + Copy>(p: Complex<T>) -> T {
    p.magnitude().saturating_sub(&T::ONE).abs()
}

/// Pairs the remaining poles and zeros into `L` sections, filling from the
/// last section backward.
fn pair_sections<T: Float + Copy, const D: usize, const L: usize>(
    poles: &mut PoleSlots<T, D>,
    zeros: &mut ZeroSlots<T, D>,
) -> SectionArray<T, L> {
    let identity = SectionCoefficients {
        b0: T::ONE,
        b1: T::ZERO,
        b2: T::ZERO,
        a1: T::ZERO,
        a2: T::ZERO,
    };
    let mut out = [identity; L];
    for dst in out.iter_mut().rev() {
        let Some(p1) = take_pole(poles, |_| true, circle_distance) else {
            break;
        };
        let (z, p) = pair_one(poles, zeros, p1);
        *dst = section(z, p);
    }
    out
}

/// Chooses the partner pole and both zeros of a section led by `p1`.
fn pair_one<T: Float + Copy, const D: usize>(
    poles: &mut PoleSlots<T, D>,
    zeros: &mut ZeroSlots<T, D>,
    p1: Complex<T>,
) -> Pairing<T> {
    let real_poles_left = poles.iter().flatten().any(|p| is_real(*p));
    if is_real(p1) && !real_poles_left {
        let z1 = take_zero(zeros, zero_is_real, p1);
        return ([z1, Zero::Used], [Some(p1), None]);
    }
    // With exactly one real zero left (finite or at infinity), prefer a
    // complex zero so a later real pole can claim that real zero. Applies
    // for both complex and real `p1`: a real `p1` with other real poles
    // remaining would otherwise take the last real zero and leave `z2`
    // empty, silently dropping the complex zero pair.
    let real_zeros = zeros.iter().filter(|z| zero_is_real(**z)).count();
    let z1 = if real_zeros == 1 {
        take_zero(zeros, |z| matches!(z, Zero::At(c) if !is_real(c)), p1)
    } else {
        take_zero(zeros, |_| true, p1)
    };
    let z1_real = zero_is_real(z1);
    if !is_real(p1) {
        let p2 = take_pole(
            poles,
            |_| true,
            |q| q.saturating_sub(&p1.conj()).magnitude(),
        );
        let z2 = match z1 {
            Zero::At(c) if !z1_real => take_zero(zeros, |_| true, c.conj()),
            _ => take_zero(zeros, zero_is_real, p1),
        };
        return ([z1, z2], [Some(p1), p2]);
    }
    if let (Zero::At(c), false) = (z1, z1_real) {
        let z2 = take_zero(zeros, |_| true, c.conj());
        let p2 =
            take_pole(poles, is_real, |q| q.saturating_sub(&c).magnitude());
        return ([z1, z2], [Some(p1), p2]);
    }
    let p2 = take_pole(poles, is_real, circle_distance);
    let z2 = take_zero(zeros, zero_is_real, p2.unwrap_or(p1));
    ([z1, z2], [Some(p1), p2])
}
