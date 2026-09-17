//! # Frequency-Domain Stability Margin Extraction
//!
//! Sweeps a transfer function's frequency response and extracts gain
//! margin, phase margin, and their crossover frequencies (python-control,
//! 2024): gain margin $K_g$ and phase crossover frequency $\omega_{pc}$ from
//! where $\arg G(j\omega)$ first reaches $-180°$, phase margin $\Phi_m$ and
//! gain crossover frequency $\omega_{gc}$ from where $|G(j\omega)|$ first
//! reaches unity gain (0 dB). Crossings are refined by ten-step interval
//! bisection (Davidson, 2020) once bracketed by the coarse sweep.
//!
//! The $-180°$ phase crossing is located as a sign change in
//! $\mathrm{Im}(G(j\omega))$ while $\mathrm{Re}(G(j\omega)) < 0$, rather than
//! by unwrapping `atan2`-based phase, sidestepping its branch-cut
//! discontinuity at $\pm 180°$. Phase margin still needs a continuous phase
//! value at $\omega_{gc}$, so the sweep accumulates an unwrapped phase
//! trajectory (adjacent principal arguments adjusted by $\pm 2\pi$) and
//! evaluates $\Phi_m = \pi + \phi_{\mathrm{unwrapped}}(\omega_{gc})$.
#![allow(
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::doc_markdown,
    clippy::type_complexity
)]

use crate::math::complex_num::Complex;
use crate::math::num_traits::Float;
use crate::math::num_types::{Const, Dim};
use crate::transfer_function::ArrayTransferFunction;

/// Stability margins extracted from a swept frequency response.
///
/// A field is `None` when the corresponding crossing was not found within
/// the supplied frequency sweep.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Margins<T> {
    /// Gain margin $K_g = 1 / |G(j\omega_{pc})|$.
    pub gain_margin: Option<T>,
    /// Phase crossover frequency $\omega_{pc}$, where $\arg G(j\omega)$
    /// first reaches $-180°$.
    pub phase_crossover_freq: Option<T>,
    /// Phase margin $\Phi_m = \pi + \phi_{\mathrm{unwrapped}}(\omega_{gc})$,
    /// in radians. Negative when the unwrapped phase at gain crossover is
    /// past $-180°$.
    pub phase_margin: Option<T>,
    /// Gain crossover frequency $\omega_{gc}$, where $|G(j\omega)|$ first
    /// reaches unity.
    pub gain_crossover_freq: Option<T>,
    /// Delay margin $\tau_m = \Phi_m / \omega_{gc}$.
    pub delay_margin: Option<T>,
}

/// Sweeps `tf`'s frequency response over `omegas` (at least two points, in
/// increasing order) and extracts its stability margins.
///
/// Phase margin uses the sweep's continuously unwrapped phase at the gain
/// crossover, not the principal `atan2` argument alone. Plants whose phase
/// lags past $-180°$ before $|G|=1$ therefore report a negative
/// $\Phi_m$ instead of a wrapped positive value near $+2\pi + \Phi_m$.
///
/// # Generic Arguments
/// - `T`: Floating-point scalar type.
/// - `N`: Numerator coefficient capacity (ascending powers).
/// - `D`: Denominator coefficient capacity (ascending powers).
///
/// # Arguments
/// - `tf`: Transfer function to sweep.
/// - `omegas`: Frequency samples $\omega$ in ascending order (rad/s). Fewer
///   than two samples yields an empty [`Margins`] result.
///
/// # Returns
/// [`Margins`] with each field set when the corresponding crossing exists
/// inside `omegas`.
///
/// # Panics
/// Never panics.
///
/// # Examples
/// ```
/// use control_rs::classical_tools::margins::stability_margins;
/// use control_rs::transfer_function::ArrayTransferFunction;
///
/// // G(s) = 4 / (s+1)^3: positive margins at this gain.
/// let tf = ArrayTransferFunction::<f64, 1, 4>::continuous(
///     [4.0],
///     [1.0, 3.0, 3.0, 1.0],
/// );
/// let omegas: [f64; 500] =
///     core::array::from_fn(|i| 0.001 + (i as f64) * (5.0 / 500.0));
/// let margins = stability_margins(&tf, &omegas);
/// assert!(margins.phase_margin.unwrap() > 0.0);
/// assert!(margins.gain_margin.unwrap() > 1.0);
/// ```
#[must_use]
pub fn stability_margins<T: Float + Copy, const N: usize, const D: usize>(
    tf: &ArrayTransferFunction<T, N, D>,
    omegas: &[T],
) -> Margins<T>
where
    Const<N>: Dim,
    Const<D>: Dim,
{
    let mut margins = Margins {
        gain_margin: None,
        phase_crossover_freq: None,
        phase_margin: None,
        gain_crossover_freq: None,
        delay_margin: None,
    };
    if omegas.len() < 2 {
        return margins;
    }

    let mut prev_omega = omegas[0];
    let mut prev = tf.eval_frequency(prev_omega);
    let mut prev_phase = prev.arg();

    for &omega in &omegas[1..] {
        let cur = tf.eval_frequency(omega);
        let cur_phase = unwrap_phase_step(prev_phase, cur.arg());

        if margins.gain_crossover_freq.is_none()
            && crosses(prev.magnitude() - T::ONE, cur.magnitude() - T::ONE)
        {
            let (wc, _resp, phase_at_wc) =
                bisect_gain_crossover(tf, prev_omega, omega, prev_phase);
            margins.gain_crossover_freq = Some(wc);
            margins.phase_margin = Some(T::PI + phase_at_wc);
        }

        if margins.phase_crossover_freq.is_none()
            && prev.re < T::ZERO
            && cur.re < T::ZERO
            && crosses(prev.im, cur.im)
        {
            let (wc, resp) = bisect_phase_crossover(tf, prev_omega, omega);
            margins.phase_crossover_freq = Some(wc);
            let mag = resp.magnitude();
            margins.gain_margin = if mag > T::ZERO {
                Some(T::ONE / mag)
            } else {
                None
            };
        }

        prev_omega = omega;
        prev = cur;
        prev_phase = cur_phase;
    }

    margins.delay_margin =
        match (margins.phase_margin, margins.gain_crossover_freq) {
            (Some(phase_margin), Some(wc)) if wc != T::ZERO => {
                Some(phase_margin / wc)
            }
            _ => None,
        };
    margins
}

/// True if `prev` and `cur` have strictly opposite signs, or `cur` lands
/// exactly on zero coming from a nonzero `prev`.
fn crosses<T: Float + Copy>(prev: T, cur: T) -> bool {
    (prev > T::ZERO && cur <= T::ZERO) || (prev < T::ZERO && cur >= T::ZERO)
}

/// Advances continuous phase by attaching `cur_principal` (∈ (−π, π]) to
/// `prev_unwrapped` with the unique $\pm 2\pi$ offset that keeps the step
/// inside (−π, π].
fn unwrap_phase_step<T: Float + Copy>(
    prev_unwrapped: T,
    cur_principal: T,
) -> T {
    let two_pi = T::PI + T::PI;
    let mut delta = cur_principal - prev_unwrapped;
    while delta > T::PI {
        delta = delta - two_pi;
    }
    while delta < -T::PI {
        delta = delta + two_pi;
    }
    prev_unwrapped + delta
}

/// Ten-step bisection locating where $|G(j\omega)|$ crosses unity gain
/// within $(\omega_{lo}, \omega_{hi})$, returning the unwrapped phase at the
/// refined frequency by continuing unwrap from `phase_lo` at `w_lo`.
fn bisect_gain_crossover<T: Float + Copy, const N: usize, const D: usize>(
    tf: &ArrayTransferFunction<T, N, D>,
    mut w_lo: T,
    mut w_hi: T,
    mut phase_lo: T,
) -> (T, Complex<T>, T)
where
    Const<N>: Dim,
    Const<D>: Dim,
{
    let mut value_lo = tf.eval_frequency(w_lo).magnitude() - T::ONE;
    let two = T::ONE + T::ONE;
    let mut mid = w_hi;
    let mut mid_resp = tf.eval_frequency(w_hi);
    let mut phase_mid = unwrap_phase_step(phase_lo, mid_resp.arg());
    for _ in 0..10 {
        mid = (w_lo + w_hi) / two;
        mid_resp = tf.eval_frequency(mid);
        phase_mid = unwrap_phase_step(phase_lo, mid_resp.arg());
        let value_mid = mid_resp.magnitude() - T::ONE;
        if crosses(value_lo, value_mid) {
            w_hi = mid;
        } else {
            w_lo = mid;
            value_lo = value_mid;
            phase_lo = phase_mid;
        }
    }
    (mid, mid_resp, phase_mid)
}

/// Ten-step bisection locating where $\mathrm{Im}(G(j\omega))$ crosses zero
/// within $(\omega_{lo}, \omega_{hi})$.
fn bisect_phase_crossover<T: Float + Copy, const N: usize, const D: usize>(
    tf: &ArrayTransferFunction<T, N, D>,
    mut w_lo: T,
    mut w_hi: T,
) -> (T, Complex<T>)
where
    Const<N>: Dim,
    Const<D>: Dim,
{
    let mut value_lo = tf.eval_frequency(w_lo).im;
    let two = T::ONE + T::ONE;
    let mut mid = w_hi;
    let mut mid_resp = tf.eval_frequency(w_hi);
    for _ in 0..10 {
        mid = (w_lo + w_hi) / two;
        mid_resp = tf.eval_frequency(mid);
        let value_mid = mid_resp.im;
        if crosses(value_lo, value_mid) {
            w_hi = mid;
        } else {
            w_lo = mid;
            value_lo = value_mid;
        }
    }
    (mid, mid_resp)
}
