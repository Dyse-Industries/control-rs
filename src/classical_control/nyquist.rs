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
use crate::transfer_function::ArrayTransferFunction;

/// Intervals per indentation arc. Odd, so no arc point sits at the apex
/// `Im p`, where a pole with `Re p = r` lies on the arc.
const ARC: usize = 15;

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

/// An optional contour point.
type Maybe<T> = Option<Complex<T>>;

/// A slice of complex values.
pub(crate) type ComplexSlice<T> = [Complex<T>];

/// Streaming winding counter for `1 + L`. Rejects any point or chord that
/// passes within the critical-point tolerance of `-1`.
pub(crate) struct Walker<T> {
    tol: T,
    first: Maybe<T>,
    prev: Maybe<T>,
    winding: i32,
}

impl<T: Float + Copy> Walker<T> {
    pub(crate) const fn new(tol: T) -> Self {
        Self {
            tol,
            first: None,
            prev: None,
            winding: 0,
        }
    }

    pub(crate) fn push(&mut self, l: Complex<T>) -> Result<(), ClassicalError> {
        let shifted = Complex::new(l.re.saturating_add(&T::ONE), l.im);
        match self.prev {
            None => {
                self.first = Some(l);
                clear_of_origin(shifted, shifted, l.magnitude(), self.tol)?;
            }
            Some(a) => self.segment(a, l)?,
        }
        self.prev = Some(l);
        Ok(())
    }

    fn segment(
        &mut self,
        a: Complex<T>,
        b: Complex<T>,
    ) -> Result<(), ClassicalError> {
        let sa = Complex::new(a.re.saturating_add(&T::ONE), a.im);
        let sb = Complex::new(b.re.saturating_add(&T::ONE), b.im);
        let (ma, mb) = (a.magnitude(), b.magnitude());
        let scale = match ma.partial_cmp(&mb) {
            Some(core::cmp::Ordering::Greater) => ma,
            _ => mb,
        };
        clear_of_origin(sa, sb, scale, self.tol)?;
        self.winding = self
            .winding
            .saturating_add(crossing(sa.re, sa.im, sb.re, sb.im));
        Ok(())
    }

    /// Closes last-to-first and returns the counterclockwise winding.
    fn finish(mut self) -> Result<i32, ClassicalError> {
        if let Some((a, b)) = self.prev.zip(self.first) {
            self.segment(a, b)?;
        }
        Ok(self.winding)
    }
}

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
/// - [`ClassicalError::ContourThroughCriticalPoint`]: a sample satisfies
///   `|1 + L| <= sqrt(eps) (1 + |L|)`, or `L` is non-finite (an open-loop
///   pole lies on the indented contour).
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
    let tol = T::epsilon().sqrt();
    let last = T::from_usize(M.saturating_sub(1));
    let axis_poles = poles.get(..order).unwrap_or(&[]);
    let mut walker = Walker::new(tol);
    let mut prev_omega: Option<T> = None;
    for (dst, k) in contour.iter_mut().zip(0usize..) {
        let u = T::from_usize(k.saturating_mul(2))
            .saturating_div(&last)
            .saturating_sub(&T::ONE);
        let omega =
            omega_max.saturating_mul(&u.saturating_mul(&u).saturating_mul(&u));
        // Arc points between the previous sample and this one: the
        // indentation belongs to the contour, so it is traced whatever the
        // grid spacing. They feed the winding count only.
        if let Some(lo) = prev_omega {
            let mut cur = lo;
            // Each pole contributes `ARC + 1` arc points, which bounds the
            // number of arc points in a gap.
            for _ in axis_poles.iter().flat_map(|_| 0..=ARC) {
                let Some(w) =
                    next_arc_point(axis_poles, indent_radius, cur, omega)
                else {
                    break;
                };
                walker.push(sys.evaluate_complex(indent(
                    w,
                    axis_poles,
                    indent_radius,
                )))?;
                cur = w;
            }
        }
        let l = sys.evaluate_complex(indent(omega, axis_poles, indent_radius));
        walker.push(l)?;
        *dst = l;
        prev_omega = Some(omega);
    }
    let ccw = walker.finish()?;
    Ok(NyquistCount {
        encirclements: ccw.saturating_neg(),
        open_loop_rhp,
    })
}

/// Smallest arc frequency strictly inside `(lo, hi)`, over all near-axis
/// poles. Arc points cluster at the arc ends, where the contour turns
/// fastest: `x = (3v - v^3) / 2` for `v` uniform in `[-1, 1]`.
pub(crate) fn next_arc_point<T: Float + Copy>(
    poles: &ComplexSlice<T>,
    r: T,
    lo: T,
    hi: T,
) -> Option<T> {
    let two = T::ONE.saturating_add(&T::ONE);
    let three = two.saturating_add(&T::ONE);
    let steps = T::from_usize(ARC);
    poles
        .iter()
        .filter(|p| p.re.abs() <= r)
        .flat_map(|p| {
            (0..=ARC).map(move |j| {
                let v = T::from_usize(j)
                    .saturating_mul(&two)
                    .saturating_div(&steps)
                    .saturating_sub(&T::ONE);
                let cube = v.saturating_mul(&v).saturating_mul(&v);
                let x = three
                    .saturating_mul(&v)
                    .saturating_sub(&cube)
                    .saturating_div(&two);
                p.im.saturating_add(&r.saturating_mul(&x))
            })
        })
        .filter(|w| *w > lo && *w < hi)
        .fold(None, |best: Option<T>, w| match best {
            Some(b) if b <= w => Some(b),
            _ => Some(w),
        })
}

/// `Ok` when the segment `a -> b` (in `1 + L` coordinates) stays farther
/// from the origin than `tol (1 + max(|La|, |Lb|))`; non-finite data is
/// rejected too.
fn clear_of_origin<T: Float + Copy>(
    a: Complex<T>,
    b: Complex<T>,
    scale: T,
    tol: T,
) -> Result<(), ClassicalError> {
    let d =
        Complex::new(b.re.saturating_sub(&a.re), b.im.saturating_sub(&a.im));
    let len2 =
        d.re.saturating_mul(&d.re)
            .saturating_add(&d.im.saturating_mul(&d.im));
    let along =
        a.re.saturating_mul(&d.re)
            .saturating_add(&a.im.saturating_mul(&d.im));
    // Parameter of the closest point, clamped to the segment.
    let t = if len2 > T::ZERO {
        let t = T::ZERO.saturating_sub(&along).saturating_div(&len2);
        if t < T::ZERO {
            T::ZERO
        } else if t > T::ONE {
            T::ONE
        } else {
            t
        }
    } else {
        T::ZERO
    };
    let dist = Complex::new(
        a.re.saturating_add(&d.re.saturating_mul(&t)),
        a.im.saturating_add(&d.im.saturating_mul(&t)),
    )
    .magnitude();
    let bound = tol.saturating_mul(&T::ONE.saturating_add(&scale));
    match dist.partial_cmp(&bound) {
        Some(core::cmp::Ordering::Greater) => Ok(()),
        _ => Err(ClassicalError::ContourThroughCriticalPoint),
    }
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

/// `+1` for a counterclockwise crossing of the negative real axis by the
/// segment `(ar, ai) -> (br, bi)`, `-1` for clockwise, else `0`.
pub(crate) fn crossing<T: Float + Copy>(ar: T, ai: T, br: T, bi: T) -> i32 {
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
