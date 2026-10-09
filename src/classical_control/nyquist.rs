//! Nyquist encirclement count (FR-4).
//!
//! The contour runs up the imaginary axis from `-jw_max` to
//! `jw_max` and closes through the semicircle at infinity, where a
//! proper `L` is constant, so the closing segment joins the last sample to
//! the first. Open-loop poles within `indent_radius` of the imaginary axis
//! are passed on the right by a semicircular indentation of that radius.
//! Encirclements of `-1` are counted as signed crossings of the negative
//! real axis by `1 + L`, so the result is an exact integer for a given
//! contour.

use super::ClassicalError;
use crate::math::complex_num::Complex;
use crate::math::num_traits::Float;
use crate::math::num_types::{Const, Dim};
use crate::math::ops::SaturatingAdd;
use crate::transfer_function::ArrayTransferFunction;

/// Result of a Nyquist encirclement count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NyquistCount {
    /// Net clockwise encirclements `N` of `-1`; closed-loop right-half-plane
    /// poles number `N + P`.
    pub encirclements: i32,
    /// Open-loop poles `P` to the right of the indented contour.
    pub open_loop_rhp: usize,
}

/// Nyquist data buffer: `L(s_k)` at each of `M` contour samples.
pub type Contour<T, const M: usize> = [Complex<T>; M];

/// A slice of complex values.
type ComplexSlice<T> = [Complex<T>];

/// Counts the net encirclements of `-1` by `L` on the Nyquist contour
/// (FR-4).
///
/// `contour` fixes the sample count `M >= 3` and receives `L(s_k)`, the
/// Nyquist data. Samples lie at `w_k = w_max u_k^3` with `u_k`
/// uniform in `[-1, 1]`, dense near `w = 0`; an odd `M` places one
/// sample at `w = 0`. Near an open-loop pole `p` with
/// `|Re p| <= r`, a sample `s = jw` moves to
/// `jw + sqrt(r^2 - (w - Im p)^2)`. The routine
/// reports the count and `P`; it does not infer closed-loop stability.
///
/// # Errors
/// - [`ClassicalError::NotContinuous`]: `sys` is discrete.
/// - [`ClassicalError::InvalidParameter`]: `w_max <= 0`, `r <= 0`
///   or `M < 3`.
/// - [`ClassicalError::Improper`]: `deg(num) > deg(den)`, or the
///   denominator is zero. The contour closes last-to-first, which counts
///   the arc at infinity only for a proper `L`.
/// - [`ClassicalError::Root`]: the open-loop poles cannot be computed.
/// - [`ClassicalError::ContourThroughCriticalPoint`]: a sample or a segment
///   between consecutive samples (including the last-to-first close)
///   satisfies `|1 + L| <= sqrt(eps) (1 + |L|)` at the nearest point on that
///   piece, or `L` is non-finite (an open-loop pole lies on the indented
///   contour).
pub fn nyquist_encirclements<
    T: Float + Copy,
    const N: usize,
    const D: usize,
    const M: usize,
>(
    sys: &ArrayTransferFunction<T, N, D>,
    omega_max: T,
    indent_radius: T,
    contour: &mut Contour<T, M>,
) -> Result<NyquistCount, ClassicalError>
where
    Const<N>: Dim,
    Const<D>: Dim,
{
    if sys.is_discrete() {
        return Err(ClassicalError::NotContinuous);
    }
    if !(omega_max > T::ZERO && indent_radius > T::ZERO) || M < 3 {
        return Err(ClassicalError::InvalidParameter);
    }
    // Closing last-to-first models a proper `L` (constant as `|s| → ∞`).
    // An improper open loop adds encirclements on that arc outside this
    // contour.
    let num_deg = sys.num_slice().iter().rposition(|v| *v != T::ZERO);
    let den_deg = sys.den_slice().iter().rposition(|v| *v != T::ZERO);
    match (num_deg, den_deg) {
        (_, None) => return Err(ClassicalError::Improper),
        (Some(n), Some(d)) if n > d => return Err(ClassicalError::Improper),
        _ => {}
    }
    let poles = sys.poles()?;
    let order = D.saturating_sub(1);
    let open_loop_rhp = poles
        .iter()
        .take(order)
        .filter(|p| p.re > indent_radius)
        .count();
    let one = Complex::new(T::ONE, T::ZERO);
    let tol = T::epsilon().sqrt();
    let last = T::from_usize(M.saturating_sub(1));
    for (dst, k) in contour.iter_mut().zip(0usize..) {
        let u = T::from_usize(k.saturating_mul(2))
            .saturating_div(&last)
            .saturating_sub(&T::ONE);
        let omega =
            omega_max.saturating_mul(&u.saturating_mul(&u).saturating_mul(&u));
        let s = indent(omega, poles.get(..order).unwrap_or(&[]), indent_radius);
        let l = sys.evaluate_complex(s);
        // `partial_cmp` rejects non-finite `L` (NaN from a pole on the
        // indent) as well as samples within the critical-point tolerance.
        let dist = l.saturating_add(&one).magnitude();
        let bound = tol.saturating_mul(&T::ONE.saturating_add(&l.magnitude()));
        match dist.partial_cmp(&bound) {
            Some(core::cmp::Ordering::Greater) => {}
            _ => return Err(ClassicalError::ContourThroughCriticalPoint),
        }
        *dst = l;
    }
    // Sample checks miss a plot that passes through `-1` between samples
    // (for example `L = 1/s^2` on the negative real axis). Reject those
    // segments with the same relative tolerance.
    let first = contour.first().copied();
    let closing = contour.last().copied().zip(first);
    let hits = contour
        .windows(2)
        .filter_map(|w| match w {
            [a, b] => Some((*a, *b)),
            _ => None,
        })
        .chain(closing)
        .any(|(a, b)| segment_near_critical(a, b, tol));
    if hits {
        return Err(ClassicalError::ContourThroughCriticalPoint);
    }
    let ccw = winding(contour);
    Ok(NyquistCount {
        encirclements: ccw.saturating_neg(),
        open_loop_rhp,
    })
}

/// Contour point at frequency `omega`, displaced right around near-axis
/// poles.
fn indent<T: Float + Copy>(
    omega: T,
    poles: &ComplexSlice<T>,
    r: T,
) -> Complex<T> {
    let r2 = r.saturating_mul(&r);
    let shift = poles
        .iter()
        .filter(|p| p.re.abs() <= r)
        .map(|p| {
            let d = omega.saturating_sub(&p.im);
            r2.saturating_sub(&d.saturating_mul(&d))
        })
        .filter(|g| *g > T::ZERO)
        .fold(T::ZERO, |acc, g| {
            let h = g.sqrt();
            if h > acc { h } else { acc }
        });
    Complex::new(shift, omega)
}

/// Net counterclockwise winding of `1 + L` about the origin over the closed
/// sample sequence, from signed crossings of the negative real axis.
fn winding<T: Float + Copy>(l: &ComplexSlice<T>) -> i32 {
    let first = l.first().copied();
    let closing = l.last().copied().zip(first);
    l.windows(2)
        .filter_map(|w| match w {
            [a, b] => Some((*a, *b)),
            _ => None,
        })
        .chain(closing)
        .map(|(a, b)| {
            crossing(
                a.re.saturating_add(&T::ONE),
                a.im,
                b.re.saturating_add(&T::ONE),
                b.im,
            )
        })
        .fold(0i32, i32::saturating_add)
}

/// `+1` for a counterclockwise crossing of the negative real axis by the
/// segment `(ar, ai) -> (br, bi)`, `-1` for clockwise, else `0`.
fn crossing<T: Float + Copy>(ar: T, ai: T, br: T, bi: T) -> i32 {
    let a_upper = ai >= T::ZERO;
    if a_upper == (bi >= T::ZERO) {
        return 0;
    }
    let t = ai.saturating_div(&ai.saturating_sub(&bi));
    let re = ar.saturating_add(&br.saturating_sub(&ar).saturating_mul(&t));
    match (re < T::ZERO, a_upper) {
        (true, true) => 1,
        (true, false) => -1,
        _ => 0,
    }
}

/// Whether the chord `a -> b` of `L` comes within the critical-point
/// tolerance of `-1` (equivalently, whether `1 + L` comes that near the
/// origin).
fn segment_near_critical<T: Float + Copy>(
    a: Complex<T>,
    b: Complex<T>,
    tol: T,
) -> bool {
    let ar = a.re.saturating_add(&T::ONE);
    let ai = a.im;
    let br = b.re.saturating_add(&T::ONE);
    let bi = b.im;
    let ab_r = br.saturating_sub(&ar);
    let ab_i = bi.saturating_sub(&ai);
    let ab2 = ab_r
        .saturating_mul(&ab_r)
        .saturating_add(&ab_i.saturating_mul(&ab_i));
    let scale_a = T::ONE.saturating_add(&a.magnitude());
    let scale_b = T::ONE.saturating_add(&b.magnitude());
    let scale = if scale_b > scale_a { scale_b } else { scale_a };
    let bound = tol.saturating_mul(&scale);
    // Degenerate chord: distance is the endpoint magnitude (already checked
    // at samples; keep the test so a zero-length NaN chord still rejects).
    let (pr, pi) = if ab2 <= T::ZERO {
        (ar, ai)
    } else {
        // `t = clamp(dot(-a, ab) / |ab|^2, 0, 1)` in `1 + L` coordinates.
        let t_num = ar
            .saturating_neg()
            .saturating_mul(&ab_r)
            .saturating_add(&ai.saturating_neg().saturating_mul(&ab_i));
        let t = t_num.saturating_div(&ab2);
        let t = if t < T::ZERO {
            T::ZERO
        } else if t > T::ONE {
            T::ONE
        } else {
            t
        };
        (
            ar.saturating_add(&ab_r.saturating_mul(&t)),
            ai.saturating_add(&ab_i.saturating_mul(&t)),
        )
    };
    let dist = Complex::new(pr, pi).magnitude();
    !matches!(dist.partial_cmp(&bound), Some(core::cmp::Ordering::Greater))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Contour samples; fewer under Miri, where the interpreter dominates.
    const SAMPLES: usize = if cfg!(miri) { 201 } else { 1001 };

    fn count<const N: usize, const D: usize>(
        sys: &ArrayTransferFunction<f64, N, D>,
    ) -> Result<NyquistCount, ClassicalError>
    where
        Const<N>: Dim,
        Const<D>: Dim,
    {
        let mut contour = [Complex::new(0.0, 0.0); SAMPLES];
        nyquist_encirclements(sys, 100.0, 1e-3, &mut contour)
    }

    #[test]
    fn encirclements_known_cases() {
        // Closed loop (s + 1)^3 + 2: stable.
        let stable = ArrayTransferFunction::<f64, 1, 4>::continuous(
            [2.0],
            [1.0, 3.0, 3.0, 1.0],
        );
        let got = count(&stable).unwrap();
        assert_eq!((got.encirclements, got.open_loop_rhp), (0, 0));
        // Closed loop (s + 1)^3 + 10: two right-half-plane poles.
        let unstable = ArrayTransferFunction::<f64, 1, 4>::continuous(
            [10.0],
            [1.0, 3.0, 3.0, 1.0],
        );
        let got = count(&unstable).unwrap();
        assert_eq!((got.encirclements, got.open_loop_rhp), (2, 0));
        // L = 2 / (s - 1): closed loop s + 1 is stable, one CCW encirclement.
        let open_unstable =
            ArrayTransferFunction::<f64, 1, 2>::continuous([2.0], [-1.0, 1.0]);
        let got = count(&open_unstable).unwrap();
        assert_eq!((got.encirclements, got.open_loop_rhp), (-1, 1));
        // Integrating plant L = 1 / (s (s + 1)): closed loop stable.
        let integrating = ArrayTransferFunction::<f64, 1, 3>::continuous(
            [1.0],
            [0.0, 1.0, 1.0],
        );
        let got = count(&integrating).unwrap();
        assert_eq!((got.encirclements, got.open_loop_rhp), (0, 0));
        let discrete = ArrayTransferFunction::<f64, 1, 2>::discrete(
            [1.0],
            [0.5, 1.0],
            0.1,
        );
        assert_eq!(count(&discrete), Err(ClassicalError::NotContinuous));
    }

    #[test]
    fn improper_open_loop_is_rejected() {
        // L = s^3: 1 + s^3 has two right-half-plane roots, and a jω-only
        // contour would report (N, P) = (0, 0).
        let improper = ArrayTransferFunction::<f64, 4, 1>::continuous(
            [0.0, 0.0, 0.0, 1.0],
            [1.0],
        );
        assert_eq!(count(&improper), Err(ClassicalError::Improper));
        let proper = ArrayTransferFunction::<f64, 2, 2>::continuous(
            [1.0, 2.0],
            [3.0, 4.0],
        );
        assert!(count(&proper).is_ok());
    }

    #[test]
    fn contour_through_critical_point() {
        // L(j0) = -1 exactly at the contour's middle sample.
        let sys =
            ArrayTransferFunction::<f64, 1, 2>::continuous([-1.0], [1.0, 1.0]);
        assert_eq!(
            count(&sys),
            Err(ClassicalError::ContourThroughCriticalPoint)
        );
    }

    #[test]
    fn parameter_bounds() {
        let sys = ArrayTransferFunction::<f64, 1, 3>::continuous(
            [2.0],
            [1.0, 3.0, 3.0],
        );
        let bad = Err(ClassicalError::InvalidParameter);
        let mut c5 = [Complex::new(0.0, 0.0); 5];
        assert_eq!(nyquist_encirclements(&sys, 0.0, 1e-3, &mut c5), bad);
        assert_eq!(nyquist_encirclements(&sys, 10.0, 0.0, &mut c5), bad);
        assert_eq!(nyquist_encirclements(&sys, -1.0, 1e-3, &mut c5), bad);
        let mut c2 = [Complex::new(0.0, 0.0); 2];
        assert_eq!(nyquist_encirclements(&sys, 10.0, 1e-3, &mut c2), bad);
        let mut c3 = [Complex::new(0.0, 0.0); 3];
        assert!(nyquist_encirclements(&sys, 10.0, 1e-3, &mut c3).is_ok());
    }

    #[test]
    fn pole_on_indent_radius_is_contour_error() {
        // Pole at `s = r` lies on the rightmost indent sample `s = r`, so
        // `L` is NaN and the contour is rejected (FR-4).
        let r = 1e-3;
        let sys =
            ArrayTransferFunction::<f64, 1, 2>::continuous([1.0], [-r, 1.0]);
        assert_eq!(
            count(&sys),
            Err(ClassicalError::ContourThroughCriticalPoint)
        );
    }

    #[test]
    fn contour_indents_around_axis_pole() {
        let r = 1e-3;
        let sys = ArrayTransferFunction::<f64, 1, 3>::continuous(
            [1.0],
            [0.0, 1.0, 1.0],
        );
        let mut contour = [Complex::new(0.0, 0.0); 3];
        nyquist_encirclements(&sys, 100.0, r, &mut contour).unwrap();
        // Middle sample w = 0 moves to s = r: L = 1 / (r (r + 1)).
        let expected = 1.0 / (r * (r + 1.0));
        assert!(((contour[1].re - expected) / expected).abs() < 1e-9);
        assert!(contour[1].im.abs() < 1e-9 * expected);
    }

    #[test]
    fn contour_through_minus_one_between_samples() {
        // L = 1/s^2 travels the negative real axis through -1 between the
        // ω samples; sample-only checks would accept a wrong N = 1.
        let sys = ArrayTransferFunction::<f64, 1, 3>::continuous(
            [1.0],
            [0.0, 0.0, 1.0],
        );
        let mut contour = [Complex::new(0.0, 0.0); SAMPLES];
        assert_eq!(
            nyquist_encirclements(&sys, 100.0, 1e-3, &mut contour),
            Err(ClassicalError::ContourThroughCriticalPoint)
        );
    }

    #[test]
    fn crossing_through_critical_point_is_not_counted() {
        // The segment passes exactly through the origin of `1 + L`.
        assert_eq!(crossing(-1.0, 1.0, 1.0, -1.0), 0);
        assert_eq!(crossing(-2.0, 1.0, 0.0, -1.0), 1);
        assert_eq!(crossing(-2.0, -1.0, 0.0, 1.0), -1);
    }
}
