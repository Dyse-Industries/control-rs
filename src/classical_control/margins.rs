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
pub(crate) fn phase_dominates<T: Float + Copy>(
    new: &PhaseCrossing<T>,
    best: &PhaseCrossing<T>,
) -> bool {
    new.gain_margin.log10().abs() < best.gain_margin.log10().abs()
}

/// Whether `new` replaces `best` as the dominant gain crossing: a smaller
/// `|phase margin|` wins, and a tie keeps the earlier crossing.
pub(crate) fn gain_dominates<T: Float + Copy>(
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

/// Whether both parts of `z` are finite (not NaN and not infinite).
///
/// `x - x` is zero only for a finite `x`; NaN and infinity give NaN, so the
/// equality fails. `partial_cmp` alone is not enough: infinity compares equal
/// to itself.
pub(crate) fn is_finite<T: Float + Copy>(z: Complex<T>) -> bool {
    let finite = |x: T| x.saturating_sub(&x) == T::ZERO;
    finite(z.re) && finite(z.im)
}
