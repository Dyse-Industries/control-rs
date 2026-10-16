//! Gain, phase, stability and delay margins (FR-5, FR-16).
//!
//! Crossings are bracketed by sign changes between consecutive sweep
//! samples: the gain crossover by `|L| - 1` and the phase crossover by
//! `Im L` while `Re L < 0`, which avoids the
//! `+/- 180 deg` branch cut. Each bracket is refined by
//! [`BISECTION_STEPS`] bisection steps on `L`, so the crossover error is the
//! bracket width divided by `2^BISECTION_STEPS`. Only crossings inside the
//! sweep are found, and they are reported in sweep order, which is frequency
//! order for an ascending sweep.
//!
//! [`margin`] returns the one gain margin and one phase margin a loop-shaping
//! user reads. [`stability_margins`] returns every crossing.

use super::response::{degrees, magnitude_db};
use crate::math::complex_num::Complex;
use crate::math::num_traits::Float;
use crate::math::num_types::{Const, Dim};
use crate::math::ops::SaturatingAdd;
use crate::transfer_function::ArrayTransferFunction;

/// Bisection steps per crossing (NFR-3). The bracket shrinks by `2^40`, so
/// a crossover is known to within the sweep step times `2^-40`.
pub const BISECTION_STEPS: usize = 40;

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
    /// units for continuous systems, samples for discrete systems. It is
    /// negative exactly when the phase margin is negative.
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

/// The dominant gain and phase margin of a loop, as MATLAB `margin` and
/// python-control `margin` report them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Margin<T> {
    /// The phase crossover whose gain margin is closest to 0 dB, or `None`
    /// when the sweep holds no phase crossover.
    pub phase_crossing: MaybePhase<T>,
    /// The gain crossover with the smallest `|phase margin|`, or `None`
    /// when the sweep holds no gain crossover.
    pub gain_crossing: MaybeGain<T>,
    /// `min_k |1 + L(w_k)|` over the finite sweep samples, or `None` when no
    /// sample is finite.
    pub stability_margin: Option<T>,
}

/// Every crossing found in a sweep, in sweep order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Margins<T, const C: usize> {
    /// Up to `C` phase crossovers; `None` where absent.
    pub phase_crossings: PhaseCrossings<T, C>,
    /// Up to `C` gain crossovers; `None` where absent.
    pub gain_crossings: GainCrossings<T, C>,
    /// `min_k |1 + L(w_k)|` over the finite sweep samples, or `None` when no
    /// sample is finite.
    pub stability_margin: Option<T>,
}

impl<T: Float + Copy> PhaseCrossing<T> {
    /// Gain margin in dB, `20 log_10` of [`gain_margin`](Self::gain_margin).
    #[must_use]
    pub fn gain_margin_db(&self) -> T {
        magnitude_db(Complex::new(self.gain_margin, T::ZERO))
    }
}

/// Computes the dominant gain and phase margin over a frequency sweep
/// (FR-5, FR-16).
///
/// This is the one-call form of [`stability_margins`]: among the crossings
/// in the sweep it keeps the phase crossover whose gain margin is closest to
/// 0 dB and the gain crossover with the smallest `|phase margin|`, using
/// constant stack. The sweep must be ascending for the crossings to be in
/// frequency order, and it must be fine enough to resolve each crossover:
/// a bracket holding two crossings of one kind is not detected. Non-finite
/// samples (for example an integrator at `w = 0`) are skipped.
///
/// # Example
/// ```
/// use control_rs::classical_control::margin;
/// use control_rs::transfer_function::ArrayTransferFunction;
///
/// let l = ArrayTransferFunction::<f64, 1, 4>::continuous([4.0], [1.0, 3.0, 3.0, 1.0]);
/// let w: [f64; 50] = core::array::from_fn(|k| 0.1 * (k as f64) + 0.05);
/// let m = margin(&l, &w);
/// let gm = m.phase_crossing.map(|p| p.gain_margin);
/// assert!(gm.is_some_and(|g| (g - 2.0).abs() < 1e-9));
/// ```
#[must_use]
pub fn margin<T: Float + Copy, const N: usize, const D: usize, const M: usize>(
    sys: &ArrayTransferFunction<T, N, D>,
    omegas: &[T; M],
) -> Margin<T>
where
    Const<N>: Dim,
    Const<D>: Dim,
{
    let mut best_phase: MaybePhase<T> = None;
    let mut best_gain: MaybeGain<T> = None;
    let stability_margin = sweep(
        sys,
        omegas,
        &mut |pc| {
            if best_phase.is_none_or(|b| phase_dominates(&pc, &b)) {
                best_phase = Some(pc);
            }
        },
        &mut |gc| {
            if best_gain.is_none_or(|b| gain_dominates(&gc, &b)) {
                best_gain = Some(gc);
            }
        },
    );
    Margin {
        phase_crossing: best_phase,
        gain_crossing: best_gain,
        stability_margin,
    }
}

/// Whether `new` replaces `best` as the dominant phase crossing: a gain
/// margin closer to 0 dB wins, and a tie keeps the earlier crossing.
fn phase_dominates<T: Float + Copy>(
    new: &PhaseCrossing<T>,
    best: &PhaseCrossing<T>,
) -> bool {
    new.gain_margin.log10().abs() < best.gain_margin.log10().abs()
}

/// Whether `new` replaces `best` as the dominant gain crossing: a smaller
/// `|phase margin|` wins, and a tie keeps the earlier crossing.
fn gain_dominates<T: Float + Copy>(
    new: &GainCrossing<T>,
    best: &GainCrossing<T>,
) -> bool {
    new.phase_margin_deg.abs() < best.phase_margin_deg.abs()
}

/// Computes every gain and phase crossover, the stability margin and the
/// delay margins over a frequency sweep (FR-5, FR-16).
///
/// `C` bounds the crossings reported per kind; later crossings are
/// dropped. The cost is `M` evaluations of `L` plus [`BISECTION_STEPS`] per
/// crossing (NFR-3). Use [`margin`] for the dominant pair. Non-finite
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
/// let m = stability_margins::<_, _, _, _, 1>(&l, &w);
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
>(
    sys: &ArrayTransferFunction<T, N, D>,
    omegas: &[T; M],
) -> Margins<T, C>
where
    Const<N>: Dim,
    Const<D>: Dim,
{
    let mut phase_crossings = [None; C];
    let mut gain_crossings = [None; C];
    let (mut n_phase, mut n_gain) = (0usize, 0usize);
    let stability_margin = sweep(
        sys,
        omegas,
        &mut |pc| store(&mut phase_crossings, &mut n_phase, pc),
        &mut |gc| store(&mut gain_crossings, &mut n_gain, gc),
    );
    Margins {
        phase_crossings,
        gain_crossings,
        stability_margin,
    }
}

/// One pass over the sweep: reports each refined crossing to the sinks and
/// returns the minimum `|1 + L|` over the finite samples.
fn sweep<T: Float + Copy, const N: usize, const D: usize, const M: usize>(
    sys: &ArrayTransferFunction<T, N, D>,
    omegas: &[T; M],
    on_phase: &mut impl FnMut(PhaseCrossing<T>),
    on_gain: &mut impl FnMut(GainCrossing<T>),
) -> Option<T>
where
    Const<N>: Dim,
    Const<D>: Dim,
{
    let one = Complex::new(T::ONE, T::ZERO);
    let mut stability_margin: Option<T> = None;
    let mut prev = None;
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
        if stability_margin.is_none_or(|m| dist < m) {
            stability_margin = Some(dist);
        }
        if let Some((wa, la)) = prev {
            if let Some(pc) = phase_crossing(sys, (wa, la), (w, l)) {
                on_phase(pc);
            }
            if let Some(gc) = gain_crossing(sys, (wa, la), (w, l)) {
                on_gain(gc);
            }
        }
        prev = Some((w, l));
    }
    stability_margin
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
fn phase_crossing<T: Float + Copy, const N: usize, const D: usize>(
    sys: &ArrayTransferFunction<T, N, D>,
    a: Sample<T>,
    b: Sample<T>,
) -> MaybePhase<T>
where
    Const<N>: Dim,
    Const<D>: Dim,
{
    let im_sign = |z: Complex<T>| z.im < T::ZERO;
    // A sign change of `Im L` with at least one end in `Re L <= 0` may hold a
    // phase crossover even when the other end is in `Re L > 0`, as in a
    // coarse sweep. Only a bracket with both ends in `Re L > 0` is skipped
    // here; the refined root is checked below.
    let brackets = im_sign(a.1) != im_sign(b.1)
        && (a.1.re <= T::ZERO || b.1.re <= T::ZERO);
    if !brackets {
        return None;
    }
    let omega = bisect(sys, a.0, b.0, im_sign)?;
    let l = sys.eval_frequency(omega);
    // The refined Im-zero root can sit in `Re L >= 0`, on the positive real
    // axis. FR-5 requires `Re L < 0` at the root.
    if l.re.partial_cmp(&T::ZERO) != Some(core::cmp::Ordering::Less) {
        return None;
    }
    // A sign change of `Im L` across a pole also converges to a point where
    // `L` is large and not real. A crossover needs `L` real at the root.
    let tol = T::epsilon().sqrt().saturating_mul(&l.magnitude());
    if l.im.abs().partial_cmp(&tol) == Some(core::cmp::Ordering::Greater) {
        return None;
    }
    Some(PhaseCrossing {
        omega,
        gain_margin: T::ONE.saturating_div(&l.magnitude()),
    })
}

/// Refines a gain crossover bracketed by `a` and `b`.
fn gain_crossing<T: Float + Copy, const N: usize, const D: usize>(
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
    let omega = bisect(sys, a.0, b.0, above)?;
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

/// [`BISECTION_STEPS`] bisection steps on `[lo, hi]` for a change of
/// `side(L(w))`; returns the final midpoint, or `None` when a pole of `L`
/// cannot be stepped around.
///
/// A midpoint where `L` is not finite (a pole of `L` on the imaginary axis)
/// carries no side. It is moved a third of the way toward `hi`; if `L` is
/// still not finite there, the bracket is abandoned. Reading a non-finite
/// `L` as one side lets the interval shrink onto the pole (FR-5).
fn bisect<T, const N: usize, const D: usize>(
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
    for _ in 0..BISECTION_STEPS {
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
        let pm_bound =
            0.05 / 2f64.powi(steps) * slope / pm + 10.0 * U * 180.0 / pm;
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
        let sys =
            ArrayTransferFunction::<f64, 1, 2>::continuous([1.0], [0.0, 1.0]);
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
        let sys =
            ArrayTransferFunction::<f64, 1, 2>::continuous([1.0], [0.0, 1.0]);
        let m = stability_margins::<_, _, _, _, 1>(&sys, &[0.5, 1.0, 2.0]);
        let gc = m.gain_crossings[0].unwrap();
        assert!(gc.omega < 1.0);
    }

    #[test]
    fn absent_crossover() {
        let sys =
            ArrayTransferFunction::<f64, 1, 2>::continuous([2.0], [1.0, 1.0]);
        let m = stability_margins::<_, _, _, _, 2>(&sys, &grid());
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
        let sys = ArrayTransferFunction::<f64, 1, 3>::continuous(
            [1.0],
            [0.0, 1.0, 1.0],
        );
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
}
