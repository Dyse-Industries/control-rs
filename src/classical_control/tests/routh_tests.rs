//! Unit tests for `routh`.

use super::*;
use crate::classical_control::routh::*;

/// Ascending coefficients with the right-half-plane root count from
/// `numpy.roots` (no roots on the imaginary axis, none symmetric).
const EPSILON_ROW_CASES: [EpsilonCase; 6] = [
    ([-3.0, -3.0, -3.0, -3.0, 0.0, 1.0], 3),
    ([-3.0, -3.0, -3.0, -2.0, 0.0, 2.0], 3),
    ([-3.0, -3.0, -3.0, -1.0, 0.0, -2.0], 2),
    ([-3.0, -3.0, -3.0, -1.0, 0.0, -1.0], 2),
    ([-3.0, -3.0, -3.0, 1.0, 0.0, 1.0], 1),
    ([-3.0, -3.0, -3.0, 2.0, 0.0, 3.0], 1),
];

const UNIT: [f64; TERMS] = [1.0, 0.0, 0.0, 0.0];

/// Ascending coefficients and the right-half-plane root count.
type EpsilonCase = ([f64; 6], usize);

#[test]
fn routh_matches_known_roots() {
    let base = [-1.0, -2.0, -3.0, -4.0, -5.0];
    for rhp in 0..=4 {
        let mut roots = [0.0; 6];
        for ((dst, &r), k) in roots.iter_mut().zip(&base).zip(0..) {
            *dst = if k < rhp { -r } else { r };
        }
        let poly = ArrayPolynomial::<f64, 6>::from_roots(roots);
        let got = routh_count(&poly, 1e-12).unwrap();
        assert_eq!(got.rhp, rhp);
        assert!(!got.imaginary_axis);
    }
}

#[test]
fn routh_zero_first_column() {
    let poly =
        ArrayPolynomial::<f64, 5>::from_coefficients([3.0, 2.0, 2.0, 1.0, 1.0]);
    let got = routh_count(&poly, 1e-12).unwrap();
    assert_eq!(got.rhp, 2);
}

#[test]
fn routh_zero_row() {
    let poly =
        ArrayPolynomial::<f64, 4>::from_coefficients([1.0, 1.0, 1.0, 1.0]);
    let got = routh_count(&poly, 1e-12).unwrap();
    assert_eq!(got.rhp, 0);
    assert!(got.imaginary_axis);
}

#[test]
fn routh_zero_leading_coefficient() {
    let poly = ArrayPolynomial::<f64, 3>::from_coefficients([1.0, 1.0, 0.0]);
    assert_eq!(
        routh_count(&poly, 1e-12),
        Err(ClassicalError::ZeroLeadingCoefficient)
    );
}

#[test]
fn negative_leading_coefficient() {
    // -(s + 1)(s + 2): no right-half-plane roots.
    let poly = ArrayPolynomial::<f64, 3>::from_coefficients([-2.0, -3.0, -1.0]);
    let got = routh_count(&poly, 1e-12).unwrap();
    assert_eq!(got.rhp, 0);
}

fn term(c: [f64; TERMS], power: i32) -> Term<f64> {
    term_until(c, power, EXACT)
}

fn term_until(c: [f64; TERMS], power: i32, until: i32) -> Term<f64> {
    Term { c, power, until }
}

#[test]
fn term_mul_propagates_validity() {
    let tol = 1e-12;
    let p = term_until(UNIT, 1, 10).mul(term(UNIT, 2), tol).unwrap();
    assert_eq!(p.until, 12);
    let p = term(UNIT, 1).mul(term_until(UNIT, 2, 10), tol).unwrap();
    assert_eq!(p.until, 11);
    let p = term(UNIT, 1).mul(term(UNIT, 2), tol).unwrap();
    assert_eq!(p.until, EXACT);
}

#[test]
fn term_minus_tracks_dropped_terms() {
    let tol = 1e-12;
    // Terms shifted past the stored width are lost.
    let lost = term(UNIT, 0)
        .minus(term([0.0, 0.0, 0.0, 1.0], 5), tol)
        .unwrap();
    assert_eq!(lost.until, 4);
    // Shifted terms that land in the upper half of the buffer are lost.
    let lost = term(UNIT, 0).minus(term([1.0; TERMS], 2), tol).unwrap();
    assert_eq!(lost.until, 4);
    let kept = term(UNIT, 0).minus(term(UNIT, 3), tol).unwrap();
    assert_eq!(kept.until, EXACT);
}

#[test]
fn term_div_propagates_validity() {
    let tol = 1e-12;
    let q = term_until(UNIT, 0, 10).div(term(UNIT, 2), tol).unwrap();
    assert_eq!((q.power, q.until), (-2, 8));
    let q = term(UNIT, 0).div(term(UNIT, 2), tol).unwrap();
    assert_eq!((q.power, q.until), (-2, EXACT));
    let q = term(UNIT, 0).div(term_until(UNIT, 1, 10), tol).unwrap();
    assert_eq!((q.power, q.until), (-1, 8));
}

#[test]
fn term_constant_chops_at_tolerance() {
    assert!(Term::constant(1e-3, 1e-3).is_zero());
    assert!(!Term::constant(2e-3, 1e-3).is_zero());
    assert!(Term::<f64>::zero().is_zero());
    assert!(!term(UNIT, 0).is_zero());
}

#[test]
fn routh_root_at_origin_sets_imaginary_axis() {
    // s (s + 1)(s + 2)
    let poly =
        ArrayPolynomial::<f64, 4>::from_coefficients([0.0, 2.0, 3.0, 1.0]);
    let got = routh_count(&poly, 1e-12).unwrap();
    assert_eq!((got.rhp, got.imaginary_axis), (0, true));
}

#[test]
fn symmetric_roots_detection() {
    // 1 + 0 s + s^2 and 1 + s + s^2 + s^3 have roots +-i.
    assert!(symmetric_roots_possible::<f64, 4>(&[1.0, 0.0, 1.0]));
    assert!(symmetric_roots_possible::<f64, 4>(&[1.0, 1.0, 1.0, 1.0]));
    // Degree below two never pairs, even with a zero constant.
    assert!(!symmetric_roots_possible::<f64, 4>(&[0.0, 1.0]));
    assert!(!symmetric_roots_possible::<f64, 4>(&[1.0, 1.0]));
    assert!(!symmetric_roots_possible::<f64, 4>(&[1.0]));
    assert!(!symmetric_roots_possible::<f64, 4>(&[2.0, 3.0, 1.0]));
}

#[test]
fn run_epsilon_only_when_allowed() {
    let c = [3.0, 2.0, 2.0, 1.0, 1.0];
    assert_eq!(run::<f64, 5>(&c, 1e-12, false), Err(Fail::Singular));
    let got = run::<f64, 5>(&c, 1e-12, true).unwrap();
    assert_eq!(got.rhp, 2);
}

#[test]
fn run_zero_leading_is_not_negative() {
    let got = run::<f64, 3>(&[1.0, 1.0, 0.0], 1e-12, false).unwrap();
    assert_eq!(got.rhp, 0);
}

#[test]
fn term_minus_keeps_dominant_order_and_lower_terms() {
    let tol = 1e-12;
    // 2 epsilon - 3 epsilon^2: the lower power dominates and keeps both terms.
    let a = term([2.0, 0.0, 0.0, 0.0], 1);
    let b = term([3.0, 0.0, 0.0, 0.0], 2);
    let d = a.minus(b, tol).unwrap();
    assert_eq!((d.c, d.power), ([2.0, -3.0, 0.0, 0.0], 1));
    // 0 - x = -x and x - 0 = x.
    let z = Term::zero();
    assert_eq!(
        z.minus(a, tol).unwrap().head().to_bits(),
        (-2.0f64).to_bits()
    );
    assert_eq!(a.minus(z, tol).unwrap().head().to_bits(), 2.0f64.to_bits());
}

#[test]
fn term_cancellation_promotes_next_order() {
    // (3/epsilon - 3) - (3/epsilon - 3) is exactly zero. With different lower
    // terms the leading terms cancel and the remainder, not zero, becomes
    // the head: a leading-term-only entry would have dropped it.
    let tol = 1e-12;
    let a = term([3.0, -3.0, 0.0, 0.0], -1);
    assert!(a.minus(a, tol).unwrap().is_zero());
    let b = term([3.0, -2.0, 0.0, 0.0], -1);
    let d = a.minus(b, tol).unwrap();
    assert_eq!((d.head(), d.power), (-1.0, 0));
}

#[test]
fn term_mul_by_zero() {
    let tol = 1e-12;
    let x = term([2.0, 1.0, 0.0, 0.0], 3);
    assert!(Term::zero().mul(x, tol).unwrap().is_zero());
    assert!(x.mul(Term::zero(), tol).unwrap().is_zero());
    let p = x.mul(term([4.0, 0.0, 0.0, 0.0], -1), tol).unwrap();
    assert_eq!((p.c, p.power), ([8.0, 4.0, 0.0, 0.0], 2));
}

#[test]
fn term_div_series() {
    // (1) / (1 + epsilon) = 1 - epsilon + epsilon^2 - epsilon^3 + ...
    let q = term([1.0, 0.0, 0.0, 0.0], 0)
        .div(term([1.0, 1.0, 0.0, 0.0], 0), 1e-12)
        .unwrap();
    assert_eq!(
        q.c.map(f64::to_bits),
        [1.0, -1.0, 1.0, -1.0].map(f64::to_bits)
    );
    assert!(q.until != EXACT);
}

#[test]
fn epsilon_row_cancellation_keeps_coefficients() {
    // FR-2: a zero first-column entry becomes epsilon; when the leading
    // terms of a later entry cancel, the next-order term must survive.
    // Dropping it reports a false imaginary axis and counts too few roots.
    for (c, rhp) in EPSILON_ROW_CASES {
        let poly = ArrayPolynomial::<f64, 6>::from_coefficients(c);
        let got = routh_count(&poly, 1e-12).unwrap();
        assert_eq!(got.rhp, rhp, "{c:?}");
        assert!(!got.imaginary_axis, "{c:?}");
    }
}

#[test]
fn routh_count_is_invariant_to_positive_scaling() {
    for (c, rhp) in EPSILON_ROW_CASES {
        for k in [0.5, 4.0] {
            let scaled = c.map(|v| v * k);
            let poly = ArrayPolynomial::<f64, 6>::from_coefficients(scaled);
            let got = routh_count(&poly, 1e-12).unwrap();
            assert_eq!((got.rhp, got.imaginary_axis), (rhp, false));
        }
    }
}

/// Deterministic xorshift for the sweep below.
fn next(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

#[test]
fn routh_agrees_with_roots_on_integer_polynomials() {
    // Oracle: the right-half-plane count of `Polynomial::roots` for
    // integer coefficients in [-3, 3] whose roots are off the axis,
    // unpaired and separated. The test is silent on how the array was
    // completed; it may report `RouthIndeterminate` but never a wrong
    // count (FR-2).
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let (mut checked, mut indeterminate) = (0u32, 0u32);
    for _ in 0..if cfg!(miri) { 40 } else { 4000 } {
        let c: [f64; 7] = core::array::from_fn(|_| {
            f64::from(i8::try_from(next(&mut state) % 7).unwrap() - 3)
        });
        if c[6] == 0.0 {
            continue;
        }
        let poly = ArrayPolynomial::<f64, 7>::from_coefficients(c);
        let Ok(roots) = poly.roots() else { continue };
        let roots = roots.get(..6).unwrap();
        let off_axis = roots.iter().all(|r| r.re.abs() > 1e-6);
        let unpaired = roots.iter().enumerate().all(|(i, a)| {
            roots.get(..i).unwrap().iter().all(|b| {
                (a.re + b.re).hypot(a.im + b.im) > 1e-3
                    && (a.re - b.re).hypot(a.im - b.im) > 1e-2
            })
        });
        if !(off_axis && unpaired) {
            continue;
        }
        let want = roots.iter().filter(|r| r.re > 0.0).count();
        checked += 1;
        match routh_count(&poly, 1e-12) {
            Ok(got) => assert_eq!(
                (got.rhp, got.imaginary_axis),
                (want, false),
                "{c:?}"
            ),
            Err(ClassicalError::RouthIndeterminate) => indeterminate += 1,
            Err(e) => panic!("{c:?}: {e}"),
        }
    }
    assert!(checked > 100 || cfg!(miri));
    assert!(
        indeterminate * 50 <= checked.max(1),
        "{indeterminate}/{checked}"
    );
}
