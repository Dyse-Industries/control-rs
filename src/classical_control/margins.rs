//! Gain, phase, stability and delay margins (FR-5, FR-16).
//!
//! Crossings are bracketed by sign changes between consecutive sweep
//! samples: the gain crossover by `|L| - 1` and the phase crossover by
//! `Im L` while `Re L < 0`, which avoids the
//! `+/- 180 deg` branch cut. Each bracket is refined by `B` bisection
//! steps on `L`, so the crossover error is the bracket width divided by
//! `2^B`. Only crossings inside the sweep are found.

use super::response::degrees;
use crate::math::complex_num::Complex;
use crate::math::num_traits::Float;
use crate::math::num_types::{Const, Dim};
use crate::math::ops::SaturatingAdd;
use crate::transfer_function::ArrayTransferFunction;

/// A phase crossover: `Im L = 0` with
/// `Re L < 0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhaseCrossing<T> {
    /// Crossover frequency `w_pc`.
    pub omega: T,
    /// Gain margin `1 / |L(w_pc)|` (absolute, not dB).
    pub gain_margin: T,
}

/// A gain crossover: `|L| = 1` (FR-5, FR-16).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GainCrossing<T> {
    /// Crossover frequency `w_gc`.
    pub omega: T,
    /// Phase margin `180 deg + arg L(w_gc)` in
    /// `(-180 deg, 180 deg]`.
    pub phase_margin_deg: T,
    /// Delay margin `phi_m / w_gc` with `phi_m` in radians: time
    /// units for continuous systems, samples for discrete systems.
    pub delay_margin: T,
}

/// Up to `C` phase crossovers; `None` where absent.
pub type PhaseCrossings<T, const C: usize> = [Option<PhaseCrossing<T>>; C];

/// Up to `C` gain crossovers; `None` where absent.
pub type GainCrossings<T, const C: usize> = [Option<GainCrossing<T>>; C];

/// A sweep sample `(w, L(jw))`.
type Sample<T> = (T, Complex<T>);

/// Optional slots for crossings.
type Slots<V, const C: usize> = [Option<V>; C];

/// A refined phase crossover, if bracketed.
type MaybePhase<T> = Option<PhaseCrossing<T>>;

/// A refined gain crossover, if bracketed.
type MaybeGain<T> = Option<GainCrossing<T>>;

/// Every crossing found in a sweep, in frequency order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Margins<T, const C: usize> {
    /// Up to `C` phase crossovers; `None` where absent.
    pub phase_crossings: PhaseCrossings<T, C>,
    /// Up to `C` gain crossovers; `None` where absent.
    pub gain_crossings: GainCrossings<T, C>,
    /// `min_k |1 + L(w_k)|` over the sweep samples; zero for an
    /// empty sweep.
    pub stability_margin: T,
}

/// Computes gain, phase, stability and delay margins over an ascending
/// frequency sweep (FR-5, FR-16).
///
/// `C` bounds the crossings reported per kind; later crossings are
/// dropped. `B` is the number of bisection steps per crossing. The cost is
/// `M` evaluations of `L` plus `B` per crossing (NFR-3). Non-finite
/// samples (for example an integrator at `w = 0`) are skipped and break
/// brackets so they neither seed `stability_margin` nor invent crossings.
///
/// # Example
/// ```
/// use control_rs::classical_control::stability_margins;
/// use control_rs::transfer_function::ArrayTransferFunction;
///
/// let l = ArrayTransferFunction::<f64, 1, 4>::continuous([4.0], [1.0, 3.0, 3.0, 1.0]);
/// let w: [f64; 50] = core::array::from_fn(|k| 0.1 * (k as f64) + 0.05);
/// let m = stability_margins::<_, 1, 4, 50, 1, 40>(&l, &w);
/// let gm = m.phase_crossings[0].map(|p| p.gain_margin);
/// assert!(gm.is_some_and(|g| (g - 2.0).abs() < 1e-9));
/// ```
#[must_use]
pub fn stability_margins<
    T: Float + Copy,
    const N: usize,
    const D: usize,
    const M: usize,
    const C: usize,
    const B: usize,
>(
    sys: &ArrayTransferFunction<T, N, D>,
    omegas: &[T; M],
) -> Margins<T, C>
where
    Const<N>: Dim,
    Const<D>: Dim,
{
    let one = Complex::new(T::ONE, T::ZERO);
    let mut out = Margins {
        phase_crossings: [None; C],
        gain_crossings: [None; C],
        stability_margin: T::ZERO,
    };
    let (mut n_phase, mut n_gain) = (0usize, 0usize);
    let mut prev = None;
    let mut seen_finite = false;
    for &w in omegas {
        let l = sys.eval_frequency(w);
        // Non-finite `L` (for example an integrator at `w = 0`) must not seed
        // `stability_margin` or invent a gain-crossing bracket: NaN never
        // compares less than a prior distance, and `above(NaN)` is false.
        if !is_finite(l) {
            prev = None;
            continue;
        }
        let dist = l.saturating_add(&one).magnitude();
        if !seen_finite || dist < out.stability_margin {
            out.stability_margin = dist;
        }
        seen_finite = true;
        if let Some((wa, la)) = prev {
            if let Some(pc) =
                phase_crossing::<T, N, D, B>(sys, (wa, la), (w, l))
            {
                store(&mut out.phase_crossings, &mut n_phase, pc);
            }
            if let Some(gc) = gain_crossing::<T, N, D, B>(sys, (wa, la), (w, l))
            {
                store(&mut out.gain_crossings, &mut n_gain, gc);
            }
        }
        prev = Some((w, l));
    }
    out
}

/// Writes `value` into the next free slot, if any.
fn store<V, const C: usize>(
    slots: &mut Slots<V, C>,
    next: &mut usize,
    value: V,
) {
    if let Some(slot) = slots.get_mut(*next) {
        *slot = Some(value);
        *next = next.saturating_add(1);
    }
}

/// Refines a phase crossover bracketed by `a` and `b`.
fn phase_crossing<
    T: Float + Copy,
    const N: usize,
    const D: usize,
    const B: usize,
>(
    sys: &ArrayTransferFunction<T, N, D>,
    a: Sample<T>,
    b: Sample<T>,
) -> MaybePhase<T>
where
    Const<N>: Dim,
    Const<D>: Dim,
{
    let im_sign = |z: Complex<T>| z.im < T::ZERO;
    let brackets =
        im_sign(a.1) != im_sign(b.1) && a.1.re < T::ZERO && b.1.re < T::ZERO;
    if !brackets {
        return None;
    }
    let omega = bisect::<T, N, D, B>(sys, a.0, b.0, im_sign)?;
    let l = sys.eval_frequency(omega);
    // Both endpoints can lie in `Re L < 0` while the refined Im-zero root
    // sits in `Re L >= 0`. FR-5 requires `Re L < 0` at that root.
    if l.re.partial_cmp(&T::ZERO) != Some(core::cmp::Ordering::Less) {
        return None;
    }
    Some(PhaseCrossing {
        omega,
        gain_margin: T::ONE.saturating_div(&l.magnitude()),
    })
}

/// Refines a gain crossover bracketed by `a` and `b`.
fn gain_crossing<
    T: Float + Copy,
    const N: usize,
    const D: usize,
    const B: usize,
>(
    sys: &ArrayTransferFunction<T, N, D>,
    a: Sample<T>,
    b: Sample<T>,
) -> MaybeGain<T>
where
    Const<N>: Dim,
    Const<D>: Dim,
{
    let above = |z: Complex<T>| z.magnitude() > T::ONE;
    if above(a.1) == above(b.1) {
        return None;
    }
    let omega = bisect::<T, N, D, B>(sys, a.0, b.0, above)?;
    let l = sys.eval_frequency(omega);
    // A refined crossover must still be a finite point of `L`.
    if !is_finite(l) {
        return None;
    }
    let neg = Complex::new(
        T::ZERO.saturating_sub(&l.re),
        T::ZERO.saturating_sub(&l.im),
    );
    let pm_rad = neg.arg();
    let seconds = pm_rad.saturating_div(&omega);
    let delay_margin = sys
        .sample_time()
        .map_or(seconds, |ts| seconds.saturating_div(&ts));
    Some(GainCrossing {
        omega,
        phase_margin_deg: degrees(pm_rad),
        delay_margin,
    })
}

/// `B` bisection steps on `[lo, hi]` for a change of `side(L(w))`; returns
/// the final midpoint, or `None` when a pole of `L` cannot be stepped around.
///
/// A midpoint where `L` is not finite (a pole of `L` on the imaginary axis)
/// carries no side. It is moved a third of the way toward `hi`; if `L` is
/// still not finite there, the bracket is abandoned. Reading a non-finite
/// `L` as one side lets the interval shrink onto the pole (FR-5).
fn bisect<T, const N: usize, const D: usize, const B: usize>(
    sys: &ArrayTransferFunction<T, N, D>,
    lo: T,
    hi: T,
    side: impl Fn(Complex<T>) -> bool,
) -> Option<T>
where
    T: Float + Copy,
    Const<N>: Dim,
    Const<D>: Dim,
{
    let two = T::ONE.saturating_add(&T::ONE);
    let three = two.saturating_add(&T::ONE);
    let probe = |w: T| {
        let l = sys.eval_frequency(w);
        is_finite(l).then(|| side(l))
    };
    let lo_side = probe(lo)?;
    let (mut lo, mut hi) = (lo, hi);
    for _ in 0..B {
        let mid = lo.saturating_add(&hi).saturating_div(&two);
        let (w, s) = if let Some(s) = probe(mid) {
            (mid, s)
        } else {
            let w = mid.saturating_add(
                &hi.saturating_sub(&mid).saturating_div(&three),
            );
            (w, probe(w)?)
        };
        if s == lo_side {
            lo = w;
        } else {
            hi = w;
        }
    }
    Some(lo.saturating_add(&hi).saturating_div(&two))
}

/// Whether both parts of `z` are finite (not NaN; infinities compare equal
/// to themselves and are caught by the magnitude tests).
fn is_finite<T: Float + Copy>(z: Complex<T>) -> bool {
    z.re.partial_cmp(&z.re).is_some() && z.im.partial_cmp(&z.im).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    const U: f64 = f64::EPSILON / 2.0;

    #[test]
    fn is_finite_requires_both_parts() {
        let nan = f64::NAN;
        assert!(is_finite(Complex::new(1.0, 2.0)));
        assert!(!is_finite(Complex::new(nan, 2.0)));
        assert!(!is_finite(Complex::new(1.0, nan)));
        assert!(!is_finite(Complex::new(nan, nan)));
    }

    fn plant() -> ArrayTransferFunction<f64, 1, 4> {
        ArrayTransferFunction::continuous([4.0], [1.0, 3.0, 3.0, 1.0])
    }

    fn grid() -> [f64; 61] {
        core::array::from_fn(|k| {
            0.05 * f64::from(u8::try_from(k).unwrap()) + 0.01
        })
    }

    #[test]
    fn third_order_margins() {
        let m = stability_margins::<_, 1, 4, 61, 2, 60>(&plant(), &grid());
        let pc = m.phase_crossings[0].unwrap();
        let gc = m.gain_crossings[0].unwrap();
        let bound = |w: f64| 0.05 / (2f64.powi(60) * w) + 10.0 * U;
        let w_phase = 3f64.sqrt();
        let w_gain = (16f64.cbrt() - 1.0).sqrt();
        assert!(((pc.omega - w_phase) / w_phase).abs() <= bound(w_phase));
        assert!(((pc.gain_margin - 2.0) / 2.0).abs() <= bound(w_phase));
        assert!(((gc.omega - w_gain) / w_gain).abs() <= bound(w_gain));
        // 180 deg + arg L cancels: a phase rounding of 10u * 180 deg is
        // 180 / pm times larger relative to pm.
        let pm = 180.0 - 3.0 * w_gain.atan().to_degrees();
        let pm_bound = bound(w_gain) + 10.0 * U * 180.0 / pm;
        let e = ((gc.phase_margin_deg - pm) / pm).abs();
        assert!(e <= pm_bound, "phase margin error {e} > {pm_bound}");
        assert!(m.phase_crossings[1].is_none());
        assert!(m.stability_margin > 0.0 && m.stability_margin < 1.0);
    }

    /// `L(s) = (c0 + c1 s + c2 s^2 + c3 s^3) / 1`.
    fn cubic(c: [f64; 4]) -> ArrayTransferFunction<f64, 4, 4> {
        ArrayTransferFunction::continuous(c, [1.0, 0.0, 0.0, 0.0])
    }

    #[test]
    fn sample_on_real_axis_is_not_negative_half() {
        // Im L(1) = 0 exactly: the bracket lies on the side where Im L < 0.
        let sys = cubic([-2.0, 1.0, 0.0, 1.0]);
        let m = stability_margins::<_, 4, 4, 3, 1, 40>(&sys, &[0.5, 1.0, 1.5]);
        let pc = m.phase_crossings[0].unwrap();
        assert!(pc.omega > 1.0);
    }

    #[test]
    fn sample_on_imaginary_axis_is_not_bracketed() {
        // Re L(1) = 0 exactly at one end of the pair, Im L changes sign.
        let sys = cubic([1.0, 1.44, 1.0, 1.0]);
        let fwd = stability_margins::<_, 4, 4, 2, 1, 40>(&sys, &[1.0, 1.5]);
        assert!(fwd.phase_crossings[0].is_none());
        let rev = stability_margins::<_, 4, 4, 2, 1, 40>(&sys, &[1.5, 1.0]);
        assert!(rev.phase_crossings[0].is_none());
    }

    #[test]
    fn unit_gain_sample_is_not_above() {
        // |1 / (j w)| = 1 exactly at w = 1.
        let sys =
            ArrayTransferFunction::<f64, 1, 2>::continuous([1.0], [0.0, 1.0]);
        let m = stability_margins::<_, 1, 2, 3, 1, 40>(&sys, &[0.5, 1.0, 2.0]);
        let gc = m.gain_crossings[0].unwrap();
        assert!(gc.omega < 1.0);
    }

    #[test]
    fn absent_crossover() {
        let sys =
            ArrayTransferFunction::<f64, 1, 2>::continuous([2.0], [1.0, 1.0]);
        let m = stability_margins::<_, 1, 2, 61, 2, 40>(&sys, &grid());
        assert!(m.phase_crossings.iter().all(Option::is_none));
        assert!(m.gain_crossings[0].is_some());
    }

    #[test]
    fn margins_cross_check() {
        // python-control 0.10.2 `stability_margins` on these loops; the
        // independent oracle is the closed form of each crossover.
        let sys = ArrayTransferFunction::<f64, 1, 3>::continuous(
            [10.0],
            [0.0, 1.0, 1.0],
        );
        let w: [f64; 201] = core::array::from_fn(|k| {
            0.05 * f64::from(u8::try_from(k).unwrap()) + 0.01
        });
        let m = stability_margins::<_, 1, 3, 201, 2, 40>(&sys, &w);
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
        let m = stability_margins::<_, 1, 4, 61, 2, 40>(&plant(), &grid());
        let gc = m.gain_crossings[0].unwrap();
        let w_gain = (16f64.cbrt() - 1.0).sqrt();
        let pm = core::f64::consts::PI - 3.0 * w_gain.atan();
        let dm = pm / w_gain;
        assert!(((gc.delay_margin - dm) / dm).abs() <= 1e-9);

        let ts = 0.05;
        let dsys = plant().to_discrete_tustin(ts, None);
        let dm_d = stability_margins::<_, 4, 4, 61, 2, 40>(&dsys, &grid());
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
        let m = stability_margins::<_, 5, 5, 2, 1, 40>(&sys, &[0.5, 1.5]);
        assert!(m.phase_crossings[0].is_none());
    }

    #[test]
    fn integrator_at_zero_frequency_skips_nan() {
        // L = 1 / (s (s + 1)): L(j0) is NaN. A linear grid that starts at
        // zero must still recover the finite gain crossover.
        let sys = ArrayTransferFunction::<f64, 1, 3>::continuous(
            [1.0],
            [0.0, 1.0, 1.0],
        );
        let w: [f64; 201] =
            core::array::from_fn(|k| 0.1 * f64::from(u8::try_from(k).unwrap()));
        let m = stability_margins::<_, 1, 3, 201, 1, 40>(&sys, &w);
        let gc = m.gain_crossings[0].unwrap();
        // |L| = 1 at w^4 + w^2 - 1 = 0 => w = sqrt((-1 + sqrt(5)) / 2).
        let w_gain = f64::midpoint(-1.0, 5f64.sqrt()).sqrt();
        assert!(((gc.omega - w_gain) / w_gain).abs() <= 1e-9);
        let pm = 90.0 - w_gain.atan().to_degrees();
        assert!(((gc.phase_margin_deg - pm) / pm).abs() <= 1e-9);
        assert!(m.stability_margin.is_finite() && m.stability_margin > 0.0);
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
        let m = stability_margins::<_, 4, 4, 61, 1, 40>(&sys, &w);
        let want = brute_stability_margin(&sys, &w);
        assert_eq!(m.stability_margin.to_bits(), want.to_bits());
        let last = sys.eval_frequency(w[60]);
        let last = Complex::new(last.re + 1.0, last.im).magnitude();
        assert!(m.stability_margin < last);
        // A single finite sample seeds the margin.
        let one = stability_margins::<_, 1, 4, 1, 1, 40>(&plant(), &[1.0]);
        let l = plant().eval_frequency(1.0);
        let d = Complex::new(l.re + 1.0, l.im).magnitude();
        assert_eq!(one.stability_margin.to_bits(), d.to_bits());
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
                let m = stability_margins::<_, 1, 4, 2, 2, 40>(&sys, &sweep);
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
        let m = stability_margins::<_, 1, 4, 2, 2, 40>(&sys, &[0.5, 1.5]);
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
}
