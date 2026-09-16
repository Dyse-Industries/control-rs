//! # Firmware Realization Topologies
//!
//! Discrete-time filter execution structures that translate rational
//! $H(z)$ transfer functions into numerically conditioned difference
//! equations for real-time firmware execution:
//!
//! - [`Biquad`]: a single second-order section (Transposed Direct Form II).
//! - [`BiquadCascade`]: a series of [`Biquad`] sections realizing a
//!   higher-order filter as cascaded Second-Order Sections (SOS), which
//!   minimizes coefficient quantization sensitivity and roundoff noise
//!   relative to a single high-order direct-form realization.
//! - [`DirectForm2T`]: a canonical, minimal-delay Transposed Direct Form II
//!   realization of an arbitrary-order discrete transfer function.
//!
//! All three structs execute in `#![no_std]`, allocation-free, constant time
//! per `update` call, and use [`Float::mul_add`] for each multiply-accumulate
//! term so the compiler can lower it to a fused multiply-add instruction on
//! targets that provide one.
#![allow(
    clippy::arbitrary_source_item_ordering,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::too_many_arguments,
    clippy::doc_markdown,
    clippy::too_long_first_doc_paragraph,
    clippy::type_complexity
)]

use crate::math::num_traits::Float;

/// A single second-order section (biquad) realized in Transposed Direct Form
/// II (DF2T):
/// $$H(z) = \frac{b_0 + b_1 z^{-1} + b_2 z^{-2}}{1 + a_1 z^{-1} + a_2 z^{-2}}$$
///
/// DF2T maintains only two state delays ($d_1, d_2$), which eliminates
/// internal summing-junction overflow relative to Direct Form I and
/// minimizes coefficient sensitivity in floating-point arithmetic.
///
/// # Example
/// ```rust
/// use control_rs::classical_tools::realization::Biquad;
///
/// // H(z) = 1 (identity section: pass-through).
/// let mut section = Biquad::new(1.0_f64, 0.0, 0.0, 0.0, 0.0);
/// assert_eq!(section.update(3.0), 3.0);
/// assert_eq!(section.update(-2.0), -2.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Biquad<T> {
    /// Numerator coefficient $b_0$.
    pub b0: T,
    /// Numerator coefficient $b_1$.
    pub b1: T,
    /// Numerator coefficient $b_2$.
    pub b2: T,
    /// Denominator coefficient $a_1$ (the denominator is normalized so that
    /// $a_0 = 1$).
    pub a1: T,
    /// Denominator coefficient $a_2$.
    pub a2: T,
    /// First state delay $d_1$.
    d1: T,
    /// Second state delay $d_2$.
    d2: T,
}

impl<T: Float + Copy> Biquad<T> {
    /// Constructs a biquad section from its transfer function coefficients,
    /// with both state delays initialized to zero.
    #[must_use]
    pub const fn new(b0: T, b1: T, b2: T, a1: T, a2: T) -> Self {
        Self {
            b0,
            b1,
            b2,
            a1,
            a2,
            d1: T::ZERO,
            d2: T::ZERO,
        }
    }

    /// Advances the section by one sample, returning $y\[k\]$ for input $u\[k\]$.
    #[inline]
    pub fn update(&mut self, input: T) -> T {
        let output = self.b0.mul_add(input, self.d1);
        self.d1 = self.b1.mul_add(input, self.d2) - self.a1 * output;
        self.d2 = self.b2 * input - self.a2 * output;
        output
    }

    /// Zeros both state delays, discarding filter history.
    pub const fn reset(&mut self) {
        self.d1 = T::ZERO;
        self.d2 = T::ZERO;
    }
}

/// A cascade of [`Biquad`] second-order sections realizing a higher-order
/// discrete transfer function:
/// $$H(z) = \prod_{k=1}^{L} \frac{b_{0,k} + b_{1,k} z^{-1} + b_{2,k} z^{-2}}{1 + a_{1,k} z^{-1} + a_{2,k} z^{-2}}$$
///
/// Factoring a high-order filter into $L = \lceil n/2 \rceil$ cascaded
/// second-order sections (SOS) avoids the ill-conditioned root sensitivity of
/// a single high-degree difference equation, reducing roundoff noise by
/// orders of magnitude.
///
/// # Example
/// ```rust
/// use control_rs::classical_tools::realization::{Biquad, BiquadCascade};
///
/// // Two cascaded identity sections.
/// let mut cascade = BiquadCascade::new([
///     Biquad::new(1.0_f64, 0.0, 0.0, 0.0, 0.0),
///     Biquad::new(1.0, 0.0, 0.0, 0.0, 0.0),
/// ]);
/// assert_eq!(cascade.update(2.0), 2.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BiquadCascade<T, const NUM_SECTIONS: usize> {
    /// The cascaded second-order sections, applied in array order.
    pub sections: [Biquad<T>; NUM_SECTIONS],
}

impl<T, const NUM_SECTIONS: usize> BiquadCascade<T, NUM_SECTIONS> {
    /// Constructs a cascade from an ordered array of sections.
    #[must_use]
    pub const fn new(sections: [Biquad<T>; NUM_SECTIONS]) -> Self {
        Self { sections }
    }
}

impl<T: Float + Copy, const NUM_SECTIONS: usize>
    BiquadCascade<T, NUM_SECTIONS>
{
    /// Advances every section by one sample in cascade order, returning the
    /// final section's output $y\[k\]$.
    #[inline]
    pub fn update(&mut self, mut signal: T) -> T {
        for section in &mut self.sections {
            signal = section.update(signal);
        }
        signal
    }

    /// Zeros the state delays of every section, discarding filter history.
    pub fn reset(&mut self) {
        for section in &mut self.sections {
            section.reset();
        }
    }
}

/// A canonical, minimal-delay Transposed Direct Form II (DF2T) realization of
/// an arbitrary-order discrete transfer function:
/// $$H(z) = \frac{b_0 + b_1 z^{-1} + \dots + b_{\text{ORDER}} z^{-\text{ORDER}}}{1 + a_1 z^{-1} + \dots + a_{\text{ORDER}} z^{-\text{ORDER}}}$$
///
/// Maintains a state delay vector $d_1, \dots, d_{\text{ORDER}}$:
/// $$y\[k\] = b_0 u\[k\] + d_1\[k-1\]$$
/// $$d_i\[k\] = b_i u\[k\] - a_i y\[k\] + d_{i+1}\[k-1\]$$
///
/// [`Biquad`] is the specialized `ORDER = 2` case. Prefer [`BiquadCascade`]
/// for filters of degree $n \ge 2$: factoring into cascaded second-order
/// sections is far less sensitive to coefficient quantization than a single
/// high-order direct form.
///
/// # Example
/// ```rust
/// use control_rs::classical_tools::realization::DirectForm2T;
///
/// // H(z) = 1 (identity: pass-through).
/// let mut filter = DirectForm2T::new(1.0_f64, [0.0, 0.0], [0.0, 0.0]);
/// assert_eq!(filter.update(5.0), 5.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirectForm2T<T, const ORDER: usize> {
    /// Numerator leading coefficient $b_0$.
    pub b0: T,
    /// Remaining numerator coefficients $b_1, \dots, b_{\text{ORDER}}$.
    pub b: [T; ORDER],
    /// Denominator coefficients $a_1, \dots, a_{\text{ORDER}}$ (the
    /// denominator is normalized so that $a_0 = 1$).
    pub a: [T; ORDER],
    /// State delays $d_1, \dots, d_{\text{ORDER}}$.
    d: [T; ORDER],
}

impl<T: Float + Copy, const ORDER: usize> DirectForm2T<T, ORDER> {
    /// Constructs a filter from its transfer function coefficients, with
    /// every state delay initialized to zero.
    #[must_use]
    pub fn new(b0: T, b: [T; ORDER], a: [T; ORDER]) -> Self {
        Self {
            b0,
            b,
            a,
            d: core::array::from_fn(|_| T::ZERO),
        }
    }

    /// Advances the filter by one sample, returning $y\[k\]$ for input $u\[k\]$.
    ///
    /// `ORDER == 0` degenerates to the pure gain $y\[k\] = b_0 u\[k\]$.
    #[inline]
    pub fn update(&mut self, input: T) -> T {
        let output = if ORDER == 0 {
            self.b0 * input
        } else {
            self.b0.mul_add(input, self.d[0])
        };
        for i in 0..ORDER {
            let next_d = if i + 1 < ORDER {
                self.d[i + 1]
            } else {
                T::ZERO
            };
            self.d[i] = self.b[i].mul_add(input, next_d) - self.a[i] * output;
        }
        output
    }

    /// Zeros every state delay, discarding filter history.
    pub fn reset(&mut self) {
        self.d = core::array::from_fn(|_| T::ZERO);
    }
}
