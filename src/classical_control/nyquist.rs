//! Nyquist encirclement count (FR-4).
//!
//! The contour runs up the imaginary axis from `-jw_max` to
//! `jw_max` and closes through the semicircle at infinity, where a
//! proper `L` is constant, so the closing segment joins the last sample to
//! the first. On-axis open-loop poles are passed on the right by a semicircular
//! indentation of radius `indent_radius`.
//! Encirclements of `-1` are counted as signed crossings of the negative
//! real axis by `1 + L`, so the result is an exact integer for a given
//! contour.

use super::ClassicalError;
use crate::math::complex_num::Complex;
use crate::math::num_traits::Float;
use crate::math::num_types::{Const, Dim};
use crate::math::ops::SaturatingAdd;
use crate::transfer_function::ArrayTransferFunction;

/// Samples placed inside each indent band `(Im p - r, Im p + r)`.
///
/// A single tip sample is not enough for the discrete winding of
/// `1 + L` to see the indent image; seven in-band abscissae match the
/// density that already yields a correct count when `r` is large enough
/// for the cubic grid to enter the band unaided.
const INDENT_SAMPLES: usize = 7;

/// Result of a Nyquist encirclement count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NyquistCount {
    /// Net clockwise encirclements `N` of `-1`; closed-loop right-half-plane
    /// poles number `N + P`.
    pub encirclements: i32,
    /// Open-loop poles `P` with positive real part.
    pub open_loop_rhp: usize,
}

/// Nyquist data buffer: `L(s_k)` at each of `M` contour samples.
pub type Contour<T, const M: usize> = [Complex<T>; M];

/// A slice of complex values.
type ComplexSlice<T> = [Complex<T>];

/// Distinct indent abscissae collected from the open-loop poles.
struct IndentSites<T, const M: usize> {
    /// Imaginary parts that need an indent fan.
    ims: [T; M],
    /// Number of valid entries in `ims`.
    count: usize,
}

/// Frequency grid with a reservation mask for indent sample placement.
struct OmegaSlots<'a, T> {
    /// Contour abscissae.
    omegas: &'a mut [T],
    /// Whether the matching abscissa is already reserved.
    reserved: &'a mut [bool],
}

/// Parameters for one indent fan on the frequency grid.
struct FanSpec<T> {
    /// Indent radius.
    radius: T,
    /// Sweep limit; fan targets must stay inside `±omega_max`.
    omega_max: T,
    /// Number of abscissae to place in the open indent band.
    count: usize,
}

/// Collected indent sites or the reason they cannot be formed.
type IndentSitesResult<T, const M: usize> =
    Result<IndentSites<T, M>, ClassicalError>;

impl<T: Float + Copy> OmegaSlots<'_, T> {
    /// Reserves the nearest free sample at `target`.
    fn reserve_nearest(&mut self, target: T) -> Result<(), ClassicalError> {
        let mut best_i = None;
        let mut best_d = T::ZERO;
        for (i, (w, taken)) in
            self.omegas.iter().zip(self.reserved.iter()).enumerate()
        {
            if *taken {
                continue;
            }
            let d = w.saturating_sub(&target).abs();
            let better = match best_i {
                None => true,
                Some(_) => matches!(
                    d.partial_cmp(&best_d),
                    Some(core::cmp::Ordering::Less)
                ),
            };
            if better {
                best_i = Some(i);
                best_d = d;
            }
        }
        let Some(i) = best_i else {
            return Err(ClassicalError::InvalidParameter);
        };
        match (self.omegas.get_mut(i), self.reserved.get_mut(i)) {
            (Some(w), Some(flag)) => {
                *w = target;
                *flag = true;
                Ok(())
            }
            _ => Err(ClassicalError::InvalidParameter),
        }
    }

    /// Places a uniform fan of abscissae in `(im - radius, im + radius)`.
    fn place_fan(
        &mut self,
        im: T,
        spec: &FanSpec<T>,
    ) -> Result<(), ClassicalError> {
        let denom = T::from_usize(spec.count.saturating_add(1));
        for j in 1..=spec.count {
            let frac = T::from_usize(j)
                .saturating_mul(&T::from_usize(2))
                .saturating_div(&denom)
                .saturating_sub(&T::ONE);
            let target = im.saturating_add(&spec.radius.saturating_mul(&frac));
            if target.abs() > spec.omega_max {
                return Err(ClassicalError::InvalidParameter);
            }
            self.reserve_nearest(target)?;
        }
        Ok(())
    }
}

/// Counts the net encirclements of `-1` by `L` on the Nyquist contour
/// (FR-4).
///
/// `contour` fixes the sample count `M >= 3` and receives `L(s_k)`, the
/// Nyquist data. Samples lie at `w_k = w_max u_k^3` with `u_k`
/// uniform in `[-1, 1]`, dense near `w = 0`; an odd `M` places one
/// sample at `w = 0`. Open-loop poles with `Re p` above the on-axis
/// tolerance `min(r, sqrt(eps))` and positive count in `P`. Poles inside
/// that tolerance are indented: for each distinct `Im p`, seven unreserved
/// samples move into `(Im p - r, Im p + r)` so the semicircle is realized
/// even when the cubic grid misses that band. Lightly unstable poles with
/// `sqrt(eps) < Re p <= r` stay in `P` on the plain `jw` axis, because
/// indenting them would drop them from `P` without a matching change in
/// `N`. Near an indented pole, a sample `s = jw` moves to
/// `jw + sqrt(r^2 - (w - Im p)^2)`. The routine reports the count and
/// `P`; it does not infer closed-loop stability.
///
/// # Errors
/// - [`ClassicalError::NotContinuous`]: `sys` is discrete.
/// - [`ClassicalError::InvalidParameter`]: `w_max <= 0`, `r <= 0`,
///   `M < 3`, an indented open-loop pole lies outside
///   `[-w_max, w_max]`, or `M` cannot reserve an indent fan within `r`
///   of every indented pole.
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
    let ol_poles = poles.get(..order).unwrap_or(&[]);
    // Poles inside the on-axis tolerance are indented and omitted from P.
    // Lightly unstable poles beyond that tolerance remain in P.
    let open_loop_rhp = ol_poles
        .iter()
        .filter(|p| p.re > T::ZERO && !needs_indent(p, indent_radius))
        .count();
    let mut omegas = omega_grid::<T, M>(omega_max);
    ensure_indent_samples(&mut omegas, ol_poles, indent_radius, omega_max)?;
    let one = Complex::new(T::ONE, T::ZERO);
    let tol = T::epsilon().sqrt();
    for (dst, omega) in contour.iter_mut().zip(omegas.iter().copied()) {
        let s = indent(omega, ol_poles, indent_radius);
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
    let ccw = winding(contour);
    Ok(NyquistCount {
        encirclements: ccw.saturating_neg(),
        open_loop_rhp,
    })
}

/// Cubic frequency grid on `[-omega_max, omega_max]`.
fn omega_grid<T: Float + Copy, const M: usize>(omega_max: T) -> [T; M] {
    let last = T::from_usize(M.saturating_sub(1));
    let mut omegas = [T::ZERO; M];
    for (dst, k) in omegas.iter_mut().zip(0usize..) {
        let u = T::from_usize(k.saturating_mul(2))
            .saturating_div(&last)
            .saturating_sub(&T::ONE);
        *dst =
            omega_max.saturating_mul(&u.saturating_mul(&u).saturating_mul(&u));
    }
    omegas
}

/// Distinct imaginary parts of poles that need an indent, in contour range.
fn indent_sites<T: Float + Copy, const M: usize>(
    poles: &ComplexSlice<T>,
    r: T,
    omega_max: T,
) -> IndentSitesResult<T, M> {
    let mut sites = IndentSites {
        ims: [T::ZERO; M],
        count: 0,
    };
    for p in poles.iter().filter(|p| needs_indent(p, r)) {
        let im = p.im;
        if im.abs() > omega_max || sites.count >= M {
            return Err(ClassicalError::InvalidParameter);
        }
        if sites.ims.iter().take(sites.count).any(|v| *v == im) {
            continue;
        }
        if let Some(slot) = sites.ims.get_mut(sites.count) {
            *slot = im;
            sites.count = sites.count.saturating_add(1);
        }
    }
    Ok(sites)
}

/// Reserves a fan of samples across each indented pole and fails when
/// the indented contour cannot be realized on this grid.
fn ensure_indent_samples<T: Float + Copy, const M: usize>(
    omegas: &mut [T; M],
    poles: &ComplexSlice<T>,
    r: T,
    omega_max: T,
) -> Result<(), ClassicalError> {
    let sites = indent_sites::<T, M>(poles, r, omega_max)?;
    if sites.count == 0 {
        return Ok(());
    }
    let per_indent = match M.checked_div(sites.count) {
        Some(share) if share >= 1 => INDENT_SAMPLES.min(share),
        _ => 1,
    };
    if per_indent < 3
        && sites.ims.iter().take(sites.count).any(|im| *im != T::ZERO)
    {
        return Err(ClassicalError::InvalidParameter);
    }
    let mut reserved = [false; M];
    let mut slots = OmegaSlots {
        omegas: omegas.as_mut_slice(),
        reserved: reserved.as_mut_slice(),
    };
    let fan = FanSpec {
        radius: r,
        omega_max,
        count: per_indent,
    };
    for im in sites.ims.iter().take(sites.count) {
        slots.place_fan(*im, &fan)?;
    }
    omegas.sort_unstable_by(|a, b| {
        a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal)
    });
    for im in sites.ims.iter().take(sites.count) {
        let in_band = omegas
            .iter()
            .filter(|w| w.saturating_sub(im).abs() < r)
            .count();
        if in_band < per_indent {
            return Err(ClassicalError::InvalidParameter);
        }
    }
    Ok(())
}

/// Whether `p` receives a right-hand indentation of radius `r`.
///
/// The on-axis band is `min(r, sqrt(eps))`, so root-finder noise of a few
/// ulps still indents a true `jw` pole, while a lightly unstable pole with
/// `Re` well above that floor stays in `P` on the plain `jw` axis.
fn needs_indent<T: Float + Copy>(p: &Complex<T>, r: T) -> bool {
    let axis_tol = T::epsilon().sqrt();
    let tol = if r < axis_tol { r } else { axis_tol };
    p.re.abs() <= tol
}

/// Contour point at frequency `omega`, displaced right around indented
/// poles.
fn indent<T: Float + Copy>(
    omega: T,
    poles: &ComplexSlice<T>,
    r: T,
) -> Complex<T> {
    let r2 = r.saturating_mul(&r);
    let shift = poles
        .iter()
        .filter(|p| needs_indent(p, r))
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
    fn light_rhp_pole_counts_in_open_loop_rhp() {
        // Pole at `s = r > 0` is open-loop RHP, not an indent target.
        let r = 1e-3;
        let sys =
            ArrayTransferFunction::<f64, 1, 2>::continuous([1.0], [-r, 1.0]);
        let got = count(&sys).unwrap();
        assert_eq!(got.open_loop_rhp, 1);
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
    fn needs_indent_accepts_root_noise() {
        let r = 1e-3;
        let p = Complex::new(2.35e-9, 1.0);
        assert!(needs_indent(&p, r), "tol={}", f64::EPSILON.sqrt());
        assert!(!needs_indent(&Complex::new(5e-4, 50.0), r));
    }

    #[test]
    fn crossing_through_critical_point_is_not_counted() {
        // The segment passes exactly through the origin of `1 + L`.
        assert_eq!(crossing(-1.0, 1.0, 1.0, -1.0), 0);
        assert_eq!(crossing(-2.0, 1.0, 0.0, -1.0), 1);
        assert_eq!(crossing(-2.0, -1.0, 0.0, 1.0), -1);
    }

    #[test]
    fn indent_covers_axis_poles_missed_by_cubic_grid() {
        // L = 1/(s^2 + 1)^2: double poles at ±j. The cubic grid with
        // M = SAMPLES and w_max = 100 has min |w - 1| ≈ 7.8e-3, so a
        // radius of 1e-3 never indented before sample reservation.
        let sys = ArrayTransferFunction::<f64, 1, 5>::continuous(
            [1.0],
            [1.0, 0.0, 2.0, 0.0, 1.0],
        );
        let got = count(&sys).unwrap();
        assert_eq!((got.encirclements, got.open_loop_rhp), (2, 0));
    }

    #[test]
    fn light_rhp_poles_count_in_p() {
        // Poles at 5e-4 ± j50 are open-loop RHP. Indenting them would drop
        // them from P without adding encirclements, so Z = N + P would read
        // stable while the closed loop has two RHP poles.
        let a = 5e-4;
        let b = 50.0;
        let sys = ArrayTransferFunction::<f64, 1, 3>::continuous(
            [1.0],
            [a * a + b * b, -2.0 * a, 1.0],
        );
        let got = count(&sys).unwrap();
        assert_eq!((got.encirclements, got.open_loop_rhp), (0, 2));
    }

    #[test]
    fn near_axis_pole_beyond_omega_max_is_rejected() {
        let sys = ArrayTransferFunction::<f64, 1, 3>::continuous(
            [1.0],
            [1.0, 0.0, 1.0],
        );
        let mut contour = [Complex::new(0.0, 0.0); 11];
        assert_eq!(
            nyquist_encirclements(&sys, 0.5, 1e-3, &mut contour),
            Err(ClassicalError::InvalidParameter)
        );
    }
}
