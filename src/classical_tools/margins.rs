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
//! discontinuity at $\pm 180°$.
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
    /// Phase margin $\Phi_m = \pi + \arg G(j\omega_{gc})$, in radians.
    pub phase_margin: Option<T>,
    /// Gain crossover frequency $\omega_{gc}$, where $|G(j\omega)|$ first
    /// reaches unity.
    pub gain_crossover_freq: Option<T>,
    /// Delay margin $\tau_m = \Phi_m / \omega_{gc}$.
    pub delay_margin: Option<T>,
}

/// Sweeps `tf`'s frequency response over `omegas` (at least two points, in
/// increasing order) and extracts its stability margins.
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

    for &omega in &omegas[1..] {
        let cur = tf.eval_frequency(omega);

        if margins.gain_crossover_freq.is_none()
            && crosses(prev.magnitude() - T::ONE, cur.magnitude() - T::ONE)
        {
            let (wc, resp) = bisect_gain_crossover(tf, prev_omega, omega);
            margins.gain_crossover_freq = Some(wc);
            margins.phase_margin = Some(wrap_to_pi(T::PI + resp.arg()));
        }

        if margins.phase_crossover_freq.is_none()
            && prev.re < T::ZERO
            && cur.re < T::ZERO
            && crosses(prev.im, cur.im)
        {
            let (wc, resp) = bisect_phase_crossover(tf, prev_omega, omega);
            margins.phase_crossover_freq = Some(wc);
            let mag = resp.magnitude();
            if mag > T::ZERO {
                margins.gain_margin = Some(T::ONE / mag);
            }
        }

        prev_omega = omega;
        prev = cur;
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

/// Wraps `x` onto $(-\pi, \pi]$ so $\pi + \mathrm{atan2}(\cdot)$ is a phase
/// margin rather than a principal-sum near $2\pi$.
fn wrap_to_pi<T: Float + Copy>(mut x: T) -> T {
    let two_pi = T::PI + T::PI;
    while x > T::PI {
        x = x - two_pi;
    }
    while x <= -T::PI {
        x = x + two_pi;
    }
    x
}

/// Ten-step bisection locating where $|G(j\omega)|$ crosses unity gain
/// within $(\omega_{lo}, \omega_{hi})$.
fn bisect_gain_crossover<T: Float + Copy, const N: usize, const D: usize>(
    tf: &ArrayTransferFunction<T, N, D>,
    mut w_lo: T,
    mut w_hi: T,
) -> (T, Complex<T>)
where
    Const<N>: Dim,
    Const<D>: Dim,
{
    let mut value_lo = tf.eval_frequency(w_lo).magnitude() - T::ONE;
    let two = T::ONE + T::ONE;
    let mut mid = w_hi;
    let mut mid_resp = tf.eval_frequency(w_hi);
    for _ in 0..10 {
        mid = (w_lo + w_hi) / two;
        mid_resp = tf.eval_frequency(mid);
        let value_mid = mid_resp.magnitude() - T::ONE;
        if crosses(value_lo, value_mid) {
            w_hi = mid;
        } else {
            w_lo = mid;
            value_lo = value_mid;
        }
    }
    (mid, mid_resp)
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
