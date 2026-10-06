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
/// `M` evaluations of `L` plus `B` per crossing (NFR-3).
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
    for &w in omegas {
        let l = sys.eval_frequency(w);
        let dist = l.saturating_add(&one).magnitude();
        if prev.is_none() || dist < out.stability_margin {
            out.stability_margin = dist;
        }
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
    let omega = bisect::<T, N, D, B>(sys, a.0, b.0, im_sign);
    let l = sys.eval_frequency(omega);
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
    let omega = bisect::<T, N, D, B>(sys, a.0, b.0, above);
    let l = sys.eval_frequency(omega);
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
/// the final midpoint.
fn bisect<T, const N: usize, const D: usize, const B: usize>(
    sys: &ArrayTransferFunction<T, N, D>,
    lo: T,
    hi: T,
    side: impl Fn(Complex<T>) -> bool,
) -> T
where
    T: Float + Copy,
    Const<N>: Dim,
    Const<D>: Dim,
{
    let two = T::ONE.saturating_add(&T::ONE);
    let lo_side = side(sys.eval_frequency(lo));
    let (mut lo, mut hi) = (lo, hi);
    for _ in 0..B {
        let mid = lo.saturating_add(&hi).saturating_div(&two);
        if side(sys.eval_frequency(mid)) == lo_side {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo.saturating_add(&hi).saturating_div(&two)
}

#[cfg(test)]
mod tests {
    use super::*;

    const U: f64 = f64::EPSILON / 2.0;

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
}
