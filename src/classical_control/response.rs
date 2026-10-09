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
struct PhaseUnwrap<T> {
    prev: Option<T>,
    offset: T,
}

impl<T: Float + Copy> PhaseUnwrap<T> {
    const fn new() -> Self {
        Self {
            prev: None,
            offset: T::ZERO,
        }
    }

    /// Returns the unwrapped value of the principal phase `p` in degrees.
    fn next(&mut self, p: T) -> T {
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

#[cfg(test)]
mod tests {
    use super::*;

    const U: f64 = f64::EPSILON / 2.0;

    fn grid() -> [f64; 41] {
        core::array::from_fn(|k| {
            10f64.powf(-2.0 + 0.1 * f64::from(u8::try_from(k).unwrap()))
        })
    }

    #[test]
    fn first_second_order_closed_form() {
        let w = grid();
        let first =
            ArrayTransferFunction::<f64, 1, 2>::continuous([2.0], [2.0, 1.0]);
        for p in frequency_response(&first, &w) {
            let mag = 2.0 / (p.omega * p.omega + 4.0).sqrt();
            let phase = -(p.omega / 2.0).atan().to_degrees();
            assert!(
                (10f64.powf(p.magnitude_db / 20.0) - mag).abs()
                    <= 10.0 * U * mag * 4.0
            );
            assert!((p.phase_deg - phase).abs() <= 1e-9);
        }
        let (wn, zeta) = (3.0, 0.2);
        let second = ArrayTransferFunction::<f64, 1, 3>::continuous(
            [wn * wn],
            [wn * wn, 2.0 * zeta * wn, 1.0],
        );
        for p in frequency_response(&second, &w) {
            let re = wn * wn - p.omega * p.omega;
            let im = 2.0 * zeta * wn * p.omega;
            let mag = wn * wn / re.hypot(im);
            let phase = -im.atan2(re).to_degrees();
            assert!(
                (10f64.powf(p.magnitude_db / 20.0) - mag).abs()
                    <= 10.0 * U * mag * 4.0
            );
            assert!((p.phase_deg - phase).abs() <= 1e-9);
        }
    }

    #[test]
    fn discrete_unit_circle() {
        let ts = 0.1;
        let sys = ArrayTransferFunction::<f64, 1, 2>::discrete(
            [0.5],
            [-0.5, 1.0],
            ts,
        );
        for p in frequency_response(&sys, &grid()) {
            let theta = p.omega * ts;
            let z = Complex::new(theta.cos(), theta.sin());
            let expected =
                Complex::new(0.5, 0.0) / (z - Complex::new(0.5, 0.0));
            let err = (p.value - expected).magnitude();
            assert!(err <= 10.0 * U * expected.magnitude() * 4.0);
        }
    }

    #[test]
    fn phase_unwrapped() {
        let sys = ArrayTransferFunction::<f64, 1, 5>::continuous(
            [1.0],
            [1.0, 4.0, 6.0, 4.0, 1.0],
        );
        let pts = frequency_response(&sys, &grid());
        for pair in pts.windows(2) {
            if let [a, b] = pair {
                assert!((b.phase_deg - a.phase_deg).abs() < 180.0);
            }
        }
        let last = pts.last().unwrap().phase_deg;
        assert!(last < -300.0);
    }

    #[test]
    fn unwrap_threshold() {
        let mut up = PhaseUnwrap::<f64>::new();
        up.next(0.0);
        assert!((up.next(180.0) - 180.0).abs() < 1e-12);
        let mut down = PhaseUnwrap::<f64>::new();
        down.next(0.0);
        assert!((down.next(-180.0) + 180.0).abs() < 1e-12);
        let mut wrap = PhaseUnwrap::<f64>::new();
        wrap.next(100.0);
        assert!((wrap.next(-100.0) - 260.0).abs() < 1e-12);
    }
}
