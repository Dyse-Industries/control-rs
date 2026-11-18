//! Unit tests for `margins`.

use super::*;
use crate::classical_control::margins::*;

const U: f64 = f64::EPSILON / 2.0;

#[test]
fn is_finite_requires_both_parts() {
    let nan = f64::NAN;
    let inf = f64::INFINITY;
    assert!(is_finite(Complex::new(1.0, 2.0)));
    assert!(!is_finite(Complex::new(nan, 2.0)));
    assert!(!is_finite(Complex::new(1.0, nan)));
    assert!(!is_finite(Complex::new(nan, nan)));
    assert!(!is_finite(Complex::new(inf, 0.0)));
    assert!(!is_finite(Complex::new(0.0, -inf)));
    assert!(!is_finite(Complex::new(inf, inf)));
}

#[test]
fn overflowed_samples_do_not_seed_stability_margin() {
    // FR-5: IEEE overflow of L must be skipped like NaN. A constant
    // `1e300 / 1e-150` evaluates to Inf; a prior `is_finite` that only
    // rejected NaN reported `Some(+Inf)`.
    let sys = ArrayTransferFunction::<f64, 1, 1>::continuous([1e300], [1e-150]);
    assert!(!is_finite(sys.eval_frequency(1.0)));
    let m = margin(&sys, &[1.0, 2.0]);
    assert_eq!(m.stability_margin, None);
    assert_eq!(m.phase_crossing, None);
    assert_eq!(m.gain_crossing, None);
}

fn plant() -> ArrayTransferFunction<f64, 1, 4> {
    ArrayTransferFunction::continuous([4.0], [1.0, 3.0, 3.0, 1.0])
}

fn grid() -> [f64; 61] {
    core::array::from_fn(|k| 0.05 * f64::from(u8::try_from(k).unwrap()) + 0.01)
}

#[test]
fn third_order_margins() {
    let m = stability_margins::<_, _, _, _, 2>(&plant(), &grid());
    let pc = m.phase_crossings[0].unwrap();
    let gc = m.gain_crossings[0].unwrap();
    let steps = i32::try_from(BISECTION_STEPS).unwrap();
    let bound = |w: f64| 0.05 / (2f64.powi(steps) * w) + 10.0 * U;
    let w_phase = 3f64.sqrt();
    let w_gain = (16f64.cbrt() - 1.0).sqrt();
    assert!(((pc.omega - w_phase) / w_phase).abs() <= bound(w_phase));
    assert!(((pc.gain_margin - 2.0) / 2.0).abs() <= bound(w_phase));
    assert!(((gc.omega - w_gain) / w_gain).abs() <= bound(w_gain));
    // The phase margin moves by |d phi / d w| = 3 / (1 + w^2) per unit of
    // crossover error, and 180 deg + arg L cancels: a phase rounding of
    // 10u * 180 deg is 180 / pm times larger relative to pm.
    let pm = 180.0 - 3.0 * w_gain.atan().to_degrees();
    let slope = (3.0 / w_gain.mul_add(w_gain, 1.0)).to_degrees();
    let pm_bound = 0.05 / 2f64.powi(steps) * slope / pm + 10.0 * U * 180.0 / pm;
    let e = ((gc.phase_margin_deg - pm) / pm).abs();
    assert!(e <= pm_bound, "phase margin error {e} > {pm_bound}");
    assert!(m.phase_crossings[1].is_none());
    let sm = m.stability_margin.unwrap();
    assert!(sm > 0.0 && sm < 1.0);
}

#[test]
fn margin_picks_dominant_crossings() {
    let w = grid();
    let all = stability_margins::<_, _, _, _, 2>(&plant(), &w);
    let m = margin(&plant(), &w);
    assert_eq!(m.phase_crossing, all.phase_crossings[0]);
    assert_eq!(m.gain_crossing, all.gain_crossings[0]);
    assert_eq!(m.stability_margin, all.stability_margin);
    // 20 log_10 2 for the 4 / (s + 1)^3 loop.
    let db = m.phase_crossing.unwrap().gain_margin_db();
    assert!((db - 20.0 * 2f64.log10()).abs() <= 1e-9);
}

#[test]
fn dominant_phase_crossing_is_closest_to_unity_gain() {
    let at = |gain_margin: f64| PhaseCrossing {
        omega: 1.0,
        gain_margin,
    };
    assert!(phase_dominates(&at(1.25), &at(2.0)));
    assert!(!phase_dominates(&at(2.0), &at(1.25)));
    // Below 0 dB the margin is a gain reduction, ranked the same way.
    assert!(phase_dominates(&at(0.8), &at(0.5)));
    assert!(!phase_dominates(&at(0.5), &at(0.8)));
    // A tie keeps the earlier crossing.
    assert!(!phase_dominates(&at(2.0), &at(2.0)));
}

#[test]
fn dominant_gain_crossing_has_smallest_phase_margin() {
    let at = |phase_margin_deg: f64| GainCrossing {
        omega: 1.0,
        phase_margin_deg,
        delay_margin: 0.0,
    };
    assert!(gain_dominates(&at(20.0), &at(45.0)));
    assert!(!gain_dominates(&at(45.0), &at(20.0)));
    // A negative margin ranks by its magnitude.
    assert!(gain_dominates(&at(-10.0), &at(30.0)));
    assert!(!gain_dominates(&at(30.0), &at(-10.0)));
    assert!(!gain_dominates(&at(-30.0), &at(30.0)));
    assert!(!gain_dominates(&at(30.0), &at(30.0)));
}

#[test]
fn sample_on_imaginary_axis_with_positive_far_end() {
    // Re L = w^4 - 10 w^2 + 9 and Im L = w (4 - w^2). L(j1) = 3j has
    // Re L = 0 exactly, L(j3.5) has Re L > 0, and Im L changes sign
    // between them at w = 2, where L = -15.
    let sys = ArrayTransferFunction::<f64, 5, 5>::continuous(
        [9.0, 4.0, 10.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
    );
    for w in [[1.0, 3.5], [3.5, 1.0]] {
        let m = stability_margins::<_, _, _, _, 1>(&sys, &w);
        let pc = m.phase_crossings[0].unwrap();
        assert!((pc.omega - 2.0).abs() <= 1e-9);
        assert!((pc.gain_margin - 1.0 / 15.0).abs() <= 1e-9);
    }
}

#[test]
fn coarse_sweep_straddling_the_imaginary_axis() {
    // 4 / (s + 1)^3 sampled at 0.3 (Re L > 0) and 3.0 (Re L < 0): Im L
    // changes sign across the pair, and the crossover at sqrt(3) lies
    // between them.
    let m = stability_margins::<_, _, _, _, 1>(&plant(), &[0.3, 3.0]);
    let pc = m.phase_crossings[0].unwrap();
    assert!((pc.omega - 3f64.sqrt()).abs() <= 1e-9);
    assert!((pc.gain_margin - 2.0).abs() <= 1e-9);
}

#[test]
fn empty_and_non_finite_sweeps_have_no_stability_margin() {
    let sys = ArrayTransferFunction::<f64, 1, 2>::continuous([1.0], [0.0, 1.0]);
    let m = margin(&sys, &[0.0]);
    assert_eq!(m.stability_margin, None);
    assert_eq!(m.phase_crossing, None);
    assert_eq!(m.gain_crossing, None);
}

/// `L(s) = (c0 + c1 s + c2 s^2 + c3 s^3) / 1`.
fn cubic(c: [f64; 4]) -> ArrayTransferFunction<f64, 4, 4> {
    ArrayTransferFunction::continuous(c, [1.0, 0.0, 0.0, 0.0])
}

#[test]
fn sample_on_real_axis_is_not_negative_half() {
    // Im L(1) = 0 exactly: the bracket lies on the side where Im L < 0.
    let sys = cubic([-2.0, 1.0, 0.0, 1.0]);
    let m = stability_margins::<_, _, _, _, 1>(&sys, &[0.5, 1.0, 1.5]);
    let pc = m.phase_crossings[0].unwrap();
    assert!(pc.omega > 1.0);
}

#[test]
fn sample_on_imaginary_axis_still_brackets() {
    // Re L(1) = 0 exactly at one end of the pair and Im L changes sign.
    // Re L = 1 - w^2 and Im L = 1.44 w - w^3, so L(j1.2) = -0.44 is a
    // phase crossover with gain margin 1 / 0.44.
    let sys = cubic([1.0, 1.44, 1.0, 1.0]);
    for w in [[1.0, 1.5], [1.5, 1.0]] {
        let m = stability_margins::<_, _, _, _, 1>(&sys, &w);
        let pc = m.phase_crossings[0].unwrap();
        assert!((pc.omega - 1.2).abs() <= 1e-9);
        assert!((pc.gain_margin - 1.0 / 0.44).abs() <= 1e-9);
    }
}

#[test]
fn unit_gain_sample_is_not_above() {
    // |1 / (j w)| = 1 exactly at w = 1.
    let sys = ArrayTransferFunction::<f64, 1, 2>::continuous([1.0], [0.0, 1.0]);
    let m = stability_margins::<_, _, _, _, 1>(&sys, &[0.5, 1.0, 2.0]);
    let gc = m.gain_crossings[0].unwrap();
    assert!(gc.omega < 1.0);
}

#[test]
fn absent_crossover() {
    let sys = ArrayTransferFunction::<f64, 1, 2>::continuous([2.0], [1.0, 1.0]);
    let m = stability_margins::<_, _, _, _, 2>(&sys, &grid());
    assert!(m.phase_crossings.iter().all(Option::is_none));
    assert!(m.gain_crossings[0].is_some());
}

#[test]
fn margins_cross_check() {
    // python-control 0.10.2 `stability_margins` on these loops; the
    // independent oracle is the closed form of each crossover.
    let sys =
        ArrayTransferFunction::<f64, 1, 3>::continuous([10.0], [0.0, 1.0, 1.0]);
    let w: [f64; 201] = core::array::from_fn(|k| {
        0.05 * f64::from(u8::try_from(k).unwrap()) + 0.01
    });
    let m = stability_margins::<_, _, _, _, 2>(&sys, &w);
    let gc = m.gain_crossings[0].unwrap();
    // |L| = 1 at w^4 + w^2 - 100 = 0.
    let w_gain = f64::midpoint(-1.0, 401f64.sqrt()).sqrt();
    assert!(((gc.omega - w_gain) / w_gain).abs() <= 1e-9);
    let pm = 90.0 - w_gain.atan().to_degrees();
    assert!(((gc.phase_margin_deg - pm) / pm).abs() <= 1e-9);
    assert!(m.phase_crossings[0].is_none());
}

#[test]
fn delay_margin_closed_form() {
    let m = stability_margins::<_, _, _, _, 2>(&plant(), &grid());
    let gc = m.gain_crossings[0].unwrap();
    let w_gain = (16f64.cbrt() - 1.0).sqrt();
    let pm = core::f64::consts::PI - 3.0 * w_gain.atan();
    let dm = pm / w_gain;
    assert!(((gc.delay_margin - dm) / dm).abs() <= 1e-9);

    let ts = 0.05;
    let dsys = plant().to_discrete_tustin(ts, None);
    let dm_d = stability_margins::<_, _, _, _, 2>(&dsys, &grid());
    let dgc = dm_d.gain_crossings[0].unwrap();
    let expected = dgc.phase_margin_deg.to_radians() / dgc.omega / ts;
    assert!(((dgc.delay_margin - expected) / expected).abs() <= 1e-12);
}

#[test]
fn positive_real_im_crossing_is_not_phase_crossover() {
    // Endpoints have `Re L < 0` and opposite `Im`. The refined Im-zero
    // root is `L(j1) = 0.5`, on the positive real axis.
    let sys = ArrayTransferFunction::<f64, 5, 5>::continuous(
        [-1.0, 1.0, -2.5, 1.0, -1.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
    );
    let m = stability_margins::<_, _, _, _, 1>(&sys, &[0.5, 1.5]);
    assert!(m.phase_crossings[0].is_none());
}

#[test]
fn integrator_at_zero_frequency_skips_nan() {
    // L = 1 / (s (s + 1)): L(j0) is NaN. A linear grid that starts at
    // zero must still recover the finite gain crossover.
    let sys =
        ArrayTransferFunction::<f64, 1, 3>::continuous([1.0], [0.0, 1.0, 1.0]);
    let w: [f64; 201] =
        core::array::from_fn(|k| 0.1 * f64::from(u8::try_from(k).unwrap()));
    let m = stability_margins::<_, _, _, _, 1>(&sys, &w);
    let gc = m.gain_crossings[0].unwrap();
    // |L| = 1 at w^4 + w^2 - 1 = 0 => w = sqrt((-1 + sqrt(5)) / 2).
    let w_gain = f64::midpoint(-1.0, 5f64.sqrt()).sqrt();
    assert!(((gc.omega - w_gain) / w_gain).abs() <= 1e-9);
    let pm = 90.0 - w_gain.atan().to_degrees();
    assert!(((gc.phase_margin_deg - pm) / pm).abs() <= 1e-9);
    assert!(m.stability_margin.is_some_and(|s| s.is_finite() && s > 0.0));
}

/// Brute-force `min_k |1 + L(w_k)|` over the finite samples.
fn brute_stability_margin<const M: usize>(
    sys: &ArrayTransferFunction<f64, 4, 4>,
    w: &[f64; M],
) -> f64 {
    w.iter()
        .map(|&x| {
            let l = sys.eval_frequency(x);
            Complex::new(l.re + 1.0, l.im).magnitude()
        })
        .fold(f64::INFINITY, f64::min)
}

#[test]
fn stability_margin_is_minimum_over_samples() {
    // FR-5: the margin is the minimum, not the first or last sample.
    // For `4/(s+1)^3` the minimum lies strictly inside the sweep.
    let sys = plant().to_discrete_tustin(0.05, None);
    let w = grid();
    let m = stability_margins::<_, _, _, _, 1>(&sys, &w);
    let want = brute_stability_margin(&sys, &w);
    let got = m.stability_margin.unwrap();
    assert_eq!(got.to_bits(), want.to_bits());
    let last = sys.eval_frequency(w[60]);
    let last = Complex::new(last.re + 1.0, last.im).magnitude();
    assert!(got < last);
    // A single finite sample seeds the margin.
    let one = stability_margins::<_, _, _, _, 1>(&plant(), &[1.0]);
    let l = plant().eval_frequency(1.0);
    let d = Complex::new(l.re + 1.0, l.im).magnitude();
    assert_eq!(one.stability_margin.unwrap().to_bits(), d.to_bits());
}

#[test]
fn reported_crossovers_satisfy_their_definition_near_poles() {
    // FR-5: for poles on the axis placed at, near and away from the
    // bracket midpoints, every reported crossover is a finite point with
    // |L| = 1 (gain) or Re L < 0 and Im L = 0 (phase).
    for k in 0..=20u8 {
        let p = 0.7 + 0.04 * f64::from(k);
        let sys = ArrayTransferFunction::<f64, 1, 4>::continuous(
            [2.0],
            [p * p, p * p, 1.0, 1.0],
        );
        for sweep in [[0.5, 1.5], [0.25, 1.75], [0.6, 1.4]] {
            let m = stability_margins::<_, _, _, _, 2>(&sys, &sweep);
            for gc in m.gain_crossings.iter().flatten() {
                let l = sys.eval_frequency(gc.omega);
                assert!(l.magnitude().is_finite(), "p = {p}");
                assert!((l.magnitude() - 1.0).abs() <= 1e-6, "p = {p}");
            }
            for pc in m.phase_crossings.iter().flatten() {
                let l = sys.eval_frequency(pc.omega);
                assert!(l.re < 0.0 && l.im.abs() <= 1e-6 * l.re.abs());
            }
        }
    }
}

#[test]
fn bisection_does_not_converge_onto_a_pole() {
    // FR-5: L = 2 / ((s^2 + 1)(s + 1)) has poles at +/- j. The bracket
    // [0.5, 1.5] has |L| > 1 at the left end and < 1 at the right, and
    // its first midpoint is the pole, where L is NaN. A reported
    // crossover must satisfy | |L| - 1 | within the tolerance and be finite.
    let sys = ArrayTransferFunction::<f64, 1, 4>::continuous(
        [2.0],
        [1.0, 1.0, 1.0, 1.0],
    );
    let m = stability_margins::<_, _, _, _, 2>(&sys, &[0.5, 1.5]);
    let gc = m.gain_crossings[0].expect("crossover inside the sweep");
    let l = sys.eval_frequency(gc.omega);
    assert!(l.magnitude().is_finite());
    assert!(
        (l.magnitude() - 1.0).abs() <= 1e-6,
        "|L| = {}",
        l.magnitude()
    );
    assert!((gc.omega - 1.0).abs() > 1e-3, "converged onto the pole");
    let pm = 180.0 + l.arg().to_degrees();
    let wrapped = (pm + 180.0).rem_euclid(360.0) - 180.0;
    assert!((gc.phase_margin_deg - wrapped).abs() <= 1e-6);
    assert!(gc.delay_margin.is_finite());
}
