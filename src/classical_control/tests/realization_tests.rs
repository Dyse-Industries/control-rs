//! Unit tests for `realization`.

use crate::classical_control::realization::Zero;

use super::*;
use crate::classical_control::realization::*;

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
    let factors = real.iter().map(|r| [-r, 1.0, 0.0]).chain(
        pairs
            .iter()
            .map(|(rho, th)| [rho * rho, -2.0 * rho * Trig::cos(*th), 1.0]),
    );
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

#[test]
/// `DirectForm2T` outputs for orders 1 to 4 meet the output tolerance
/// against the direct recurrence (FR-8).
fn df2t_matches_reference() {
    check_df2t::<1, 2>(
        &poly(&[-0.5], &[]),
        &poly(&[0.8], &[]).map(|v| 2.0 * v),
    )
    .unwrap();
    check_df2t::<2, 3>(&poly(&[], &[(1.0, 1.0)]), &poly(&[], &[(0.95, 0.3)]))
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

#[test]
/// After `reset` the output sequence equals that of a new realization
/// exactly (FR-8).
fn df2t_reset() {
    let tf = ArrayTransferFunction::<f64, 3, 3>::discrete(
        [0.2, 0.3, 0.1],
        [0.5, -1.2, 1.0],
        0.1,
    );
    let mut used = DirectForm2T::<f64, 2>::from_transfer_function(&tf).unwrap();
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

#[test]
/// A continuous source returns `NotDiscrete` (FR-8, FR-10).
fn df2t_from_continuous() {
    let tf =
        ArrayTransferFunction::<f64, 2, 2>::continuous([1.0, 1.0], [2.0, 1.0]);
    assert!(matches!(
        DirectForm2T::<f64, 1>::from_transfer_function(&tf),
        Err(ClassicalError::NotDiscrete)
    ));
    assert!(matches!(
        to_sections::<f64, 2, 2, 1>(&tf),
        Err(ClassicalError::NotDiscrete)
    ));
}

#[test]
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

#[test]
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

#[test]
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

#[test]
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
    assert!(df2t.sections().iter().all(|s| s.0.state.iter().all(zero)));
}

/// Frequency response of sections at `q = e^-jw`.
fn sections_response(c: &[Coeffs], w: f64) -> Complex<f64> {
    let q = Complex::new(Trig::cos(w), -Trig::sin(w));
    let eval = |c0: f64, c1: f64, c2: f64| {
        let lin = Complex::new(c1, 0.0).saturating_mul(&q);
        let quad = Complex::new(c2, 0.0).saturating_mul(&q.saturating_mul(&q));
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
    let tf =
        ArrayTransferFunction::<f64, D, D>::discrete(take(num), take(den), 1.0);
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
        let err = Complex::new(got.re - want.re, got.im - want.im).magnitude();
        assert!(err <= SECTIONS_REL * peak, "D={D} w={wk} err={err}");
    }
    Ok(c)
}

#[test]
/// Factored cascades reproduce the source response for even and odd
/// orders up to 8, with zeros at infinity and at the origin (FR-10).
fn sections_reproduce_source() {
    let p = [(0.95, 0.3), (0.8, 1.0), (0.6, 2.0), (0.5, 2.7)];
    let z = [(1.0, 0.9), (1.0, 1.7), (0.9, 2.5), (1.2, 0.2)];
    let pairs = |n: usize| p.get(..n).unwrap_or(&[]);
    let zpairs = |n: usize| z.get(..n).unwrap_or(&[]);
    check_sections::<2, 1>(&poly(&[-1.0], &[]), &poly(&[0.7], &[])).unwrap();
    check_sections::<3, 1>(&poly(&[], zpairs(1)), &poly(&[], pairs(1)))
        .unwrap();
    check_sections::<4, 2>(&poly(&[], zpairs(1)), &poly(&[0.7], pairs(1)))
        .unwrap();
    check_sections::<5, 2>(
        &poly(&[0.0, -1.0], zpairs(1)),
        &poly(&[], pairs(2)),
    )
    .unwrap();
    check_sections::<6, 3>(&poly(&[-1.0], zpairs(2)), &poly(&[-0.3], pairs(2)))
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

#[test]
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
    assert!(near(mid.a1, -2.0 * 0.5 * Trig::cos(1.5f64)) && near(mid.a2, 0.25));
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

#[test]
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
    Fixed<R, S>: Scalar + Signed + MulAcc + PartialOrd,
{
    let q = quantize::<R, S, 3>(&cascade_case(0.2))?;
    let c64 = q.map(|s| SectionCoefficients {
        b0: s.b0.to_num(),
        b1: s.b1.to_num(),
        b2: s.b2.to_num(),
        a1: s.a1.to_num(),
        a2: s.a2.to_num(),
    });
    let mut fixed = BiquadCascade::<Df1<Fixed<R, S>>, 3>::from_coefficients(&q);
    let mut float = BiquadCascade::<Df1<f64>, 3>::from_coefficients(&c64);
    let delta = Fixed::<R, S>::DELTA.to_num();
    let err = input(scale).iter().fold(0.0f64, |m, &x| {
        let xq = Fixed::<R, S>::from_num(x);
        let y = fixed.update(xq).to_num();
        m.max((y - float.update(xq.to_num())).abs())
    });
    Ok((err / delta, noise_bound(&c64)))
}

#[test]
/// `Df1<Fixed<i16, 13>>` and `Df1<Fixed<i32, 29>>` outputs stay within
/// the noise-gain bound of the `f64` realization (FR-17).
fn df1_fixed_matches_reference() {
    let (e13, b13) = fixed_error::<i16, 13>(0.25).unwrap();
    assert!(e13 <= b13, "Q13 {e13} > {b13}");
    let (e29, b29) = fixed_error::<i32, 29>(0.25).unwrap();
    assert!(e29 <= b29, "Q29 {e29} > {b29}");
}

#[test]
/// Signed fixed-point keeps IIR feedback: `From` stores `-a_i`, so a
/// step into `{b = 0.25, 0.5, 0.25; a = 0.5, 0.25}` matches `f64`
/// within one LSB (C-8). Unsigned `Fixed` is rejected by the `Signed`
/// bound; its `saturating_neg` is zero and would clear the taps.
fn df1_signed_fixed_keeps_feedback() {
    type Q = Fixed<i16, 8>;
    let c = SectionCoefficients {
        b0: Q::from_num(0.25),
        b1: Q::from_num(0.5),
        b2: Q::from_num(0.25),
        a1: Q::from_num(0.5),
        a2: Q::from_num(0.25),
    };
    let mut fixed = Df1::from(c);
    let mut float = Df1::from(SectionCoefficients {
        b0: 0.25,
        b1: 0.5,
        b2: 0.25,
        a1: 0.5,
        a2: 0.25,
    });
    let delta: f64 = Q::DELTA.to_num();
    for _ in 0..8 {
        let y = fixed.update(Q::from_num(1.0)).to_num();
        let e = float.update(1.0);
        assert!((y - e).abs() <= delta, "got={y} exp={e}");
    }
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

#[test]
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
        let raw =
            [s.b0, s.b1, s.b2, s.na1, s.na2].map(|v| i64::from(v.to_bits()));
        let sig = [u, s.u1, s.u2, s.y1, s.y2].map(|v| i64::from(v.to_bits()));
        let exact: i64 = raw.iter().zip(&sig).map(|(a, b)| a * b).sum();
        let want = Ord::clamp(
            round13(exact),
            i64::from(i16::MIN),
            i64::from(i16::MAX),
        );
        assert_eq!(i64::from(s.update(u).to_bits()), want);
    }
}

#[test]
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
    assert_eq!(b.0.state.map(f64::to_bits), sb.0.state.map(f64::to_bits));
    let mut ra = Df1 {
        u1: sa.u1,
        u2: sa.u2,
        y1: sa.y1,
        y2: sa.y2,
        ..Df1::from(new)
    };
    let mut rb = Df2t::from(new);
    rb.0.state = sb.0.state;
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

#[test]
/// Realizations compare equal exactly when every coefficient and state
/// matches.
fn realizations_compare_by_value() {
    let base = DirectForm2T::<f64, 2>::new(1.0, [0.5, 0.25], [-0.5, 0.1]);
    assert_eq!(base, base);
    let mut moved = base;
    moved.update(1.0);
    assert_ne!(base, moved);
    let others = [
        DirectForm2T::<f64, 2>::new(2.0, [0.5, 0.25], [-0.5, 0.1]),
        DirectForm2T::<f64, 2>::new(1.0, [0.6, 0.25], [-0.5, 0.1]),
        DirectForm2T::<f64, 2>::new(1.0, [0.5, 0.25], [-0.4, 0.1]),
    ];
    for other in others {
        assert_ne!(base, other);
    }
    let c = cascade_case(0.3)[0];
    assert_eq!(Df2t::from(c), Df2t::from(c));
    let mut running = Df2t::from(c);
    running.update(1.0);
    assert_ne!(Df2t::from(c), running);
}

#[test]
/// `section_count` is `max(1, ceil((D - 1) / 2))` (FR-10).
fn section_count_values() {
    assert_eq!(
        [0, 1, 2, 3, 4, 5, 6, 9].map(section_count),
        [1, 1, 1, 1, 2, 2, 3, 4]
    );
}

#[test]
/// A running section reports the coefficients it was built or updated
/// with, so a gain schedule can check it with `is_stable` (FR-18,
/// FR-20).
fn coefficients_round_trip() {
    let [old, new, _] = cascade_case(0.3);
    let mut a = Df1::from(old);
    let mut b = Df2t::from(old);
    assert_eq!(a.coefficients(), old);
    assert_eq!(b.coefficients(), old);
    a.set_coefficients(&new);
    b.set_coefficients(&new);
    assert_eq!(a.coefficients(), new);
    assert_eq!(b.coefficients(), new);
    assert!(is_stable(&a.coefficients()));
}

#[test]
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

#[test]
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
        assert_eq!(all_stable(&[first, biquad(0.5, 1.0, 1.0, 1.0)]), rho < 1.0);
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

#[test]
/// Roots of equal magnitude keep their order: the sort is stable.
fn sort_keeps_equal_magnitudes_in_order() {
    let mut roots = [
        Complex::new(0.5, 0.0),
        Complex::new(0.0, 2.0),
        Complex::new(0.0, -2.0),
        Complex::new(2.0, 0.0),
    ];
    sort_by_magnitude_desc(&mut roots, 4);
    assert_eq!(
        roots,
        [
            Complex::new(0.0, 2.0),
            Complex::new(0.0, -2.0),
            Complex::new(2.0, 0.0),
            Complex::new(0.5, 0.0),
        ]
    );
}

#[test]
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

#[test]
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

#[test]
/// The nearest `f64` to `Fixed::MAX` and `Fixed::MIN` quantize for every
/// `Repr`, including `i64` where `MAX` has no exact `f64` (FR-19).
fn quantize_accepts_representable_extremes() {
    fn extremes<R: FixedRepr, const S: usize>() -> [f64; 2]
    where
        Const<S>: Dim + DimMax<R::BitsDim, Output = R::BitsDim>,
        Fixed<R, S>: Scalar,
    {
        [Fixed::<R, S>::MIN.to_num(), Fixed::<R, S>::MAX.to_num()]
    }
    for v in extremes::<i64, 30>() {
        assert!(quantize::<i64, 30, 1>(&with_b0(v)).is_ok(), "{v}");
    }
    for v in extremes::<i64, 2>() {
        assert!(quantize::<i64, 2, 1>(&with_b0(v)).is_ok(), "{v}");
    }
    for v in extremes::<i32, 29>() {
        assert!(quantize::<i32, 29, 1>(&with_b0(v)).is_ok(), "{v}");
    }
    let bad = Err(ClassicalError::CoefficientRange { section: 0 });
    // The neighbours of the `i64` images lie beyond the range.
    let [min, max] = extremes::<i64, 30>();
    let above = f64::from_bits(max.to_bits() + 1);
    let below = f64::from_bits(min.to_bits() + 1);
    assert_eq!(quantize::<i64, 30, 1>(&with_b0(above)), bad);
    assert_eq!(quantize::<i64, 30, 1>(&with_b0(below)), bad);
    assert_eq!(quantize::<i64, 30, 1>(&with_b0(f64::NAN)), bad);
    assert_eq!(quantize::<i64, 30, 1>(&with_b0(f64::INFINITY)), bad);
    assert_eq!(quantize::<i64, 30, 1>(&with_b0(1e10)), bad);
    assert_eq!(quantize::<i64, 30, 1>(&with_b0(-1e10)), bad);
}

#[test]
/// The imaginary tolerance of `is_real` scales with `max(1, |z|)`.
fn real_tolerance_scales_with_magnitude() {
    assert!(is_real(Complex::new(1000.0, 1e-6)));
    assert!(!is_real(Complex::new(1000.0, 1e-3)));
    assert!(!is_real(Complex::new(0.5, 1e-6)));
    assert!(is_real(Complex::new(0.5, 1e-9)));
}

#[test]
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

#[test]
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

#[test]
/// A complex pole takes a complex zero pair when only one real zero
/// remains, leaving the real zero for the real pole (FR-10).
fn complex_pole_preserves_last_real_zero() {
    check_sections::<4, 2>(
        &poly(&[0.9], &[(1.0, 2.5)]),
        &poly(&[0.7], &[(0.95, 0.3)]),
    )
    .unwrap();
}

#[test]
/// A real pole with other real poles remaining takes a complex zero
/// pair when only one real zero is left, so the cascade matches the
/// source (FR-10).
fn real_pole_preserves_complex_zero_pair() {
    // num = (z - 0.85)(z^2 - z + 0.5), den = (z - 0.9)(z - 0.5)(z - 0.2).
    // Without the complex-zero preference the first real pole claimed
    // 0.85 and left z2 unused, dropping 0.5 +/- 0.5j.
    check_sections::<4, 2>(
        &poly(
            &[0.85],
            &[(
                core::f64::consts::FRAC_1_SQRT_2,
                core::f64::consts::FRAC_PI_4,
            )],
        ),
        &poly(&[0.9, 0.5, 0.2], &[]),
    )
    .unwrap();
}

#[test]
/// A constant discrete TF keeps its gain in a single section (FR-10).
fn constant_tf_section_holds_gain() {
    let tf = ArrayTransferFunction::<f64, 1, 1>::discrete([3.0], [2.0], 0.1);
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
