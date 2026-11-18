//! Frequency response for Bode, Nyquist and Nichols data (FR-3).
//!
//! Continuous systems are evaluated at `s = jw` and discrete systems at
//! `z = e^jw T_s`. Bode data is (magnitude, phase) against
//! `w`, Nyquist data is the complex value and Nichols data is magnitude
//! in dB against phase. No routine samples frequencies itself.

use crate::math::complex_num::Complex;
use crate::math::num_traits::Float;
use crate::math::num_types::{Const, Dim};
use crate::transfer_function::ArrayTransferFunction;

/// One frequency-response sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResponsePoint<T> {
    /// Angular frequency `w` (rad per time unit).
    pub omega: T,
    /// `G(jw)` or `G(e^jw T_s)`.
    pub value: Complex<T>,
    /// `20 log_10 |G|`.
    pub magnitude_db: T,
    /// Phase in degrees, unwrapped along the sweep.
    pub phase_deg: T,
}

/// Frequency-response samples, one per caller frequency.
pub type Response<T, const M: usize> = [ResponsePoint<T>; M];

/// Running phase unwrap in degrees with a fixed operation count per sample.
pub(crate) struct PhaseUnwrap<T> {
    prev: Option<T>,
    offset: T,
}

impl<T: Float + Copy> PhaseUnwrap<T> {
    pub(crate) const fn new() -> Self {
        Self {
            prev: None,
            offset: T::ZERO,
        }
    }

    /// Returns the unwrapped value of the principal phase `p` in degrees.
    pub(crate) fn next(&mut self, p: T) -> T {
        let half = T::from_usize(180);
        let full = T::from_usize(360);
        if let Some(prev) = self.prev {
            let step = p.saturating_sub(&prev);
            if step > half {
                self.offset = self.offset.saturating_sub(&full);
            } else if step < half.saturating_neg() {
                self.offset = self.offset.saturating_add(&full);
            }
        }
        self.prev = Some(p);
        p.saturating_add(&self.offset)
    }
}

/// Evaluates the frequency response at every caller frequency (FR-3).
///
/// Phase is unwrapped along `omegas` so consecutive samples differ by less
/// than `180 deg`; the first sample keeps its principal value. The cost
/// is one numerator and one denominator Horner evaluation per frequency
/// (NFR-3).
///
/// # Example
/// ```
/// use control_rs::classical_control::frequency_response;
/// use control_rs::transfer_function::ArrayTransferFunction;
///
/// let g = ArrayTransferFunction::<f64, 1, 2>::continuous([1.0], [1.0, 1.0]);
/// let [p] = frequency_response(&g, &[1.0]);
/// assert!((p.phase_deg + 45.0).abs() < 1e-12);
/// ```
#[must_use]
pub fn frequency_response<
    T: Float + Copy,
    const N: usize,
    const D: usize,
    const M: usize,
>(
    sys: &ArrayTransferFunction<T, N, D>,
    omegas: &[T; M],
) -> Response<T, M>
where
    Const<N>: Dim,
    Const<D>: Dim,
{
    let mut out = [ResponsePoint {
        omega: T::ZERO,
        value: Complex::new(T::ZERO, T::ZERO),
        magnitude_db: T::ZERO,
        phase_deg: T::ZERO,
    }; M];
    let mut unwrap = PhaseUnwrap::new();
    for (dst, &omega) in out.iter_mut().zip(omegas) {
        let value = sys.eval_frequency(omega);
        *dst = ResponsePoint {
            omega,
            value,
            magnitude_db: magnitude_db(value),
            phase_deg: unwrap.next(degrees(value.arg())),
        };
    }
    out
}

/// Radians to degrees.
pub(crate) fn degrees<T: Float + Copy>(rad: T) -> T {
    rad.saturating_mul(&T::from_usize(180))
        .saturating_div(&T::PI)
}

/// `20 log_10 |z|`.
pub(crate) fn magnitude_db<T: Float + Copy>(z: Complex<T>) -> T {
    T::from_usize(20).saturating_mul(&z.magnitude().log10())
}
