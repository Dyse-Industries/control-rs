//! Routh-Hurwitz right-half-plane root count (FR-2).
//!
//! The Routh array is built two rows at a time from the coefficient slice.
//! A row of zeros is replaced by the coefficients of the derivative of the
//! auxiliary polynomial and sets the imaginary-axis flag. A zero first
//! entry with a nonzero remainder is resolved in three stages, each tried
//! only when the previous one meets a zero first entry:
//!
//! 1. the array of the polynomial itself, which needs no `epsilon`;
//! 2. the array of the reversed polynomial `s^n p(1/s)`, whose roots are
//!    the reciprocals of the roots of `p` and so keep both the sign of the
//!    real part and the imaginary-axis roots;
//! 3. the array with the zero entry replaced by `epsilon > 0`, every entry
//!    carried as a series `c_0 epsilon^k + c_1 epsilon^(k+1) + ...` of
//!    `TERMS` terms, so the sign of an entry is its sign as
//!    `epsilon -> 0^+` and no numeric `epsilon` is used.
//!
//! Stage 3 is exact for the perturbed polynomial `p + epsilon q`, whose roots
//! approach those of `p` as `epsilon -> 0^+`. That is the count of `p` unless `p`
//! has roots on the imaginary axis, which the perturbation can push to
//! either side. A Hurwitz determinant test therefore gates stage 3: when
//! `p` may have a pair of roots `z, -z`, or a series entry loses every
//! retained term to a truncation, the result is
//! [`ClassicalError::RouthIndeterminate`] and never a guess.

use super::ClassicalError;
use crate::math::num_traits::Float;
use crate::math::num_types::{Const, Dim};
use crate::polynomial::ArrayPolynomial;

/// Marks a series that is exact: no term beyond the stored ones is lost.
const EXACT: i32 = i32::MAX;

/// Terms kept per entry in the `epsilon` series.
const TERMS: usize = 4;

/// `TERMS` as an exponent.
const TERMS_I32: i32 = 4;

/// Result of the Routh-Hurwitz test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RouthCount {
    /// Number of roots in the open right half plane.
    pub rhp: usize,
    /// Whether a row of zeros indicated roots symmetric about the origin,
    /// including roots on the imaginary axis.
    pub imaginary_axis: bool,
}

/// A Routh row of series entries.
type Row<T, const N: usize> = [Term<T>; N];

/// Result of building one Routh row.
type RowResult<T, const N: usize> = Result<Row<T, N>, Fail>;

/// Square matrix.
type Matrix<T, const N: usize> = [[T; N]; N];

/// Why an array could not be completed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Fail {
    /// A zero first entry that needs `epsilon`.
    Singular,
    /// An ambiguous cancellation or an exhausted series.
    Indeterminate,
}

/// Entry `sum_i c[i] epsilon^(power + i)`, known for exponents below `until`.
#[derive(Debug, Clone, Copy)]
struct Term<T> {
    c: [T; TERMS],
    power: i32,
    until: i32,
}

impl<T: Float + Copy> Term<T> {
    const fn zero() -> Self {
        Self {
            c: [T::ZERO; TERMS],
            power: 0,
            until: EXACT,
        }
    }

    fn constant(v: T, tol: T) -> Self {
        let mut t = Self::zero();
        if let Some(slot) = t.c.first_mut().filter(|_| v.abs() > tol) {
            *slot = v;
        }
        t
    }

    /// The infinitesimal `epsilon`.
    const fn epsilon() -> Self {
        let mut t = Self::zero();
        if let Some(slot) = t.c.first_mut() {
            *slot = T::ONE;
        }
        t.power = 1;
        t
    }

    fn is_zero(&self) -> bool {
        self.c.iter().all(|v| *v == T::ZERO)
    }

    /// Leading coefficient.
    fn head(&self) -> T {
        self.c.first().copied().unwrap_or(T::ZERO)
    }

    fn neg(self) -> Self {
        Self {
            c: self.c.map(|v| T::ZERO.saturating_sub(&v)),
            ..self
        }
    }

    /// Chops coefficients within `tol` of zero. An all-zero result is exact
    /// zero only if no term was lost. A leading cancellation promotes the
    /// next-order term to the head, so no stored coefficient is dropped.
    fn settle(mut self, tol: T) -> Result<Self, Fail> {
        self.c = self.c.map(|v| if v.abs() <= tol { T::ZERO } else { v });
        if self.is_zero() {
            return if self.until == EXACT {
                Ok(Self::zero())
            } else {
                Err(Fail::Indeterminate)
            };
        }
        for _ in 0..TERMS {
            if self.head() != T::ZERO {
                break;
            }
            self.c.rotate_left(1);
            self.power = self.power.saturating_add(1);
        }
        Ok(self)
    }

    /// Product of two entries.
    fn mul(self, b: Self, tol: T) -> Result<Self, Fail> {
        if self.is_zero() || b.is_zero() {
            return Ok(Self::zero());
        }
        let mut full = [T::ZERO; 7];
        for (i, x) in self.c.iter().enumerate() {
            for (j, y) in b.c.iter().enumerate() {
                if let Some(slot) = full.get_mut(i.saturating_add(j)) {
                    *slot = slot.saturating_add(&x.saturating_mul(y));
                }
            }
        }
        let power = self.power.saturating_add(b.power);
        let mut until = EXACT;
        if self.until != EXACT {
            until = until.min(b.power.saturating_add(self.until));
        }
        if b.until != EXACT {
            until = until.min(self.power.saturating_add(b.until));
        }
        let dropped = full.iter().skip(TERMS).any(|v| *v != T::ZERO);
        if dropped {
            until = until.min(power.saturating_add(TERMS_I32));
        }
        let mut c = [T::ZERO; TERMS];
        c.iter_mut().zip(full).for_each(|(d, v)| *d = v);
        Self { c, power, until }.settle(tol)
    }

    /// `self - rhs`, aligned on the lower exponent.
    fn minus(self, rhs: Self, tol: T) -> Result<Self, Fail> {
        if self.is_zero() {
            return Ok(rhs.neg());
        }
        if rhs.is_zero() {
            return Ok(self);
        }
        let power = self.power.min(rhs.power);
        let mut full = [T::ZERO; 8];
        let mut dropped = false;
        for (t, sign) in
            [(self, T::ONE), (rhs, T::ZERO.saturating_sub(&T::ONE))]
        {
            let shift = t.power.saturating_sub(power);
            for (i, v) in t.c.iter().enumerate() {
                let at = usize::try_from(shift)
                    .unwrap_or(usize::MAX)
                    .saturating_add(i);
                match full.get_mut(at) {
                    Some(slot) => {
                        *slot = slot.saturating_add(&sign.saturating_mul(v));
                    }
                    None => dropped = true,
                }
            }
        }
        dropped |= full.iter().skip(TERMS).any(|v| *v != T::ZERO);
        let mut until = self.until.min(rhs.until);
        if dropped {
            until = until.min(power.saturating_add(TERMS_I32));
        }
        let mut c = [T::ZERO; TERMS];
        c.iter_mut().zip(full).for_each(|(d, v)| *d = v);
        Self { c, power, until }.settle(tol)
    }

    /// Series quotient; `d` has a nonzero leading coefficient.
    fn div(self, d: Self, tol: T) -> Result<Self, Fail> {
        if self.is_zero() {
            return Ok(self);
        }
        let mut q = [T::ZERO; TERMS];
        for k in 0..TERMS {
            let mut acc = self.c.get(k).copied().unwrap_or(T::ZERO);
            for j in 1..=k {
                let dj = d.c.get(j).copied().unwrap_or(T::ZERO);
                let qk = q.get(k.saturating_sub(j)).copied().unwrap_or(T::ZERO);
                acc = acc.saturating_sub(&dj.saturating_mul(&qk));
            }
            if let Some(slot) = q.get_mut(k) {
                *slot = acc.saturating_div(&d.head());
            }
        }
        let power = self.power.saturating_sub(d.power);
        let mut until = EXACT;
        if self.until != EXACT {
            until = until.min(self.until.saturating_sub(d.power));
        }
        if d.until != EXACT {
            let valid = d.until.saturating_sub(d.power);
            until = until.min(power.saturating_add(valid));
        }
        let monomial = d.c.iter().skip(1).all(|v| *v == T::ZERO);
        if !monomial {
            until = until.min(power.saturating_add(TERMS_I32));
        }
        Self { c: q, power, until }.settle(tol)
    }
}

/// Counts the open right-half-plane roots of a real polynomial without
/// computing them (FR-2).
///
/// `poly` holds ascending coefficients of degree `N - 1`. A computed or given
/// coefficient with magnitude at most `tol` is treated as zero. The array
/// has a fixed operation count of order `N^2` (NFR-3) and uses two stack
/// rows. Zero low-order coefficients are roots at the origin: they are
/// removed and set the imaginary-axis flag.
///
/// # Generic Arguments
/// * `T` - scalar type, a [`Float`].
/// * `N` - coefficient count of the polynomial.
///
/// # Arguments
/// * `poly` - real polynomial, ascending coefficients.
/// * `tol` - magnitude at or below which a coefficient is zero.
///
/// # Returns
/// * `Ok(count)` - [`RouthCount`] with the open right-half-plane root count and the imaginary-axis flag.
/// * `Err` - see [Errors](#errors).
///
/// # Errors
/// - [`ClassicalError::ZeroLeadingCoefficient`] when the leading coefficient
///   is within `tol` of zero.
/// - [`ClassicalError::RouthIndeterminate`] when the array needs `epsilon` and
///   a cancellation cannot be resolved (see the module documentation).
///
/// # Example
/// ```
/// use control_rs::classical_control::routh_count;
/// use control_rs::polynomial::ArrayPolynomial;
///
/// // s^3 + s^2 + s + 1 = (s + 1)(s^2 + 1)
/// let p = ArrayPolynomial::<f64, 4>::from_coefficients([1.0, 1.0, 1.0, 1.0]);
/// let r = routh_count(&p, 1e-12)?;
/// assert_eq!(r.rhp, 0);
/// assert!(r.imaginary_axis);
/// # Ok::<(), control_rs::classical_control::ClassicalError>(())
/// ```
pub fn routh_count<T: Float + Copy, const N: usize>(
    poly: &ArrayPolynomial<T, N>,
    tol: T,
) -> Result<RouthCount, ClassicalError>
where
    Const<N>: Dim,
{
    let coeffs = poly.as_slice();
    let degree = N.saturating_sub(1);
    let leading = coeffs.get(degree).copied().unwrap_or(T::ZERO);
    if leading.abs() <= tol {
        return Err(ClassicalError::ZeroLeadingCoefficient);
    }
    let low = coeffs
        .iter()
        .take(degree)
        .take_while(|c| c.abs() <= tol)
        .count();
    let body = coeffs.get(low..).unwrap_or(&[]);
    let origin = low > 0;
    let finish = |c: RouthCount| RouthCount {
        rhp: c.rhp,
        imaginary_axis: c.imaginary_axis || origin,
    };
    match run::<T, N>(body, tol, false) {
        Ok(c) => return Ok(finish(c)),
        Err(Fail::Indeterminate) => {
            return Err(ClassicalError::RouthIndeterminate);
        }
        Err(Fail::Singular) => {}
    }
    let mut reversed = [T::ZERO; N];
    reversed
        .iter_mut()
        .zip(body.iter().rev())
        .for_each(|(d, v)| *d = *v);
    let reversed = reversed.get(..body.len()).unwrap_or(&[]);
    match run::<T, N>(reversed, tol, false) {
        Ok(c) => return Ok(finish(c)),
        Err(Fail::Indeterminate) => {
            return Err(ClassicalError::RouthIndeterminate);
        }
        Err(Fail::Singular) => {}
    }
    if symmetric_roots_possible::<T, N>(body) {
        return Err(ClassicalError::RouthIndeterminate);
    }
    run::<T, N>(body, tol, true)
        .map(finish)
        .map_err(|_| ClassicalError::RouthIndeterminate)
}

/// Whether the polynomial may have a pair of roots `z, -z` (including a pair
/// on the imaginary axis): its Hurwitz determinant `Delta_n` vanishes. The
/// determinant is compared with the Hadamard bound, the product of the row
/// norms, so the test is scale invariant, against `sqrt(epsilon)`.
fn symmetric_roots_possible<T: Float + Copy, const N: usize>(
    coeffs: &[T],
) -> bool {
    let n = coeffs.len().saturating_sub(1);
    if n < 2 {
        return false;
    }
    // H[i][j] = a_(2j - i), a_k = coeffs[n - k], for i, j = 1..=n.
    let mut h = [[T::ZERO; N]; N];
    for (i, row) in h.iter_mut().enumerate().take(n) {
        for (j, slot) in row.iter_mut().enumerate().take(n) {
            let k = (2usize.saturating_mul(j.saturating_add(1)))
                .checked_sub(i.saturating_add(1));
            let at = k.and_then(|k| n.checked_sub(k));
            *slot = at
                .filter(|_| k.is_some_and(|k| k <= n))
                .and_then(|a| coeffs.get(a).copied())
                .unwrap_or(T::ZERO);
        }
    }
    let mut bound = T::ONE;
    for row in h.iter().take(n) {
        let norm = row
            .iter()
            .take(n)
            .fold(T::ZERO, |acc, v| acc.saturating_add(&v.saturating_mul(v)))
            .sqrt();
        bound = bound.saturating_mul(&norm);
    }
    let det = eliminate(&mut h, n);
    det.abs()
        .partial_cmp(&T::epsilon().sqrt().saturating_mul(&bound))
        != Some(core::cmp::Ordering::Greater)
}

/// Determinant of the leading `n x n` block by Gaussian elimination with
/// partial pivoting; the block is destroyed.
fn eliminate<T: Float + Copy, const N: usize>(
    m: &mut Matrix<T, N>,
    n: usize,
) -> T {
    let mut det = T::ONE;
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|a, b| {
                let va = m
                    .get(*a)
                    .and_then(|r| r.get(col))
                    .map_or(T::ZERO, |v| v.abs());
                let vb = m
                    .get(*b)
                    .and_then(|r| r.get(col))
                    .map_or(T::ZERO, |v| v.abs());
                va.partial_cmp(&vb).unwrap_or(core::cmp::Ordering::Equal)
            })
            .unwrap_or(col);
        if pivot != col {
            m.swap(pivot, col);
            det = T::ZERO.saturating_sub(&det);
        }
        let head = m
            .get(col)
            .and_then(|r| r.get(col))
            .copied()
            .unwrap_or(T::ZERO);
        if head == T::ZERO {
            return T::ZERO;
        }
        det = det.saturating_mul(&head);
        let pivot_row = m.get(col).copied().unwrap_or([T::ZERO; N]);
        for r in col.saturating_add(1)..n {
            if let Some(row) = m.get_mut(r) {
                let f = row
                    .get(col)
                    .copied()
                    .unwrap_or(T::ZERO)
                    .saturating_div(&head);
                for (dst, src) in row.iter_mut().zip(pivot_row.iter()).skip(col)
                {
                    *dst = dst.saturating_sub(&f.saturating_mul(src));
                }
            }
        }
    }
    det
}

/// One pass over the array of `coeffs` (ascending, nonzero leading and
/// constant coefficients). With `eps_allowed` false a zero first entry
/// returns [`Fail::Singular`].
fn run<T: Float + Copy, const N: usize>(
    coeffs: &[T],
    tol: T,
    eps_allowed: bool,
) -> Result<RouthCount, Fail> {
    let degree = coeffs.len().saturating_sub(1);
    let leading = coeffs.get(degree).copied().unwrap_or(T::ZERO);
    let mut out = RouthCount {
        rhp: 0,
        imaginary_axis: false,
    };
    let mut upper = first_row::<T, N>(coeffs, degree, tol);
    let mut lower = first_row::<T, N>(coeffs, degree.saturating_sub(1), tol);
    let mut prev_negative = leading < T::ZERO;
    for order in (0..degree).rev() {
        if lower.iter().all(Term::is_zero) {
            out.imaginary_axis = true;
            lower = aux_derivative(&upper, order.saturating_add(1), tol)?;
        } else if lower.first().is_some_and(Term::is_zero) {
            if !eps_allowed {
                return Err(Fail::Singular);
            }
            if let Some(head) = lower.first_mut() {
                *head = Term::epsilon();
            }
        }
        let negative = lower.first().is_some_and(|t| {
            t.head().partial_cmp(&T::ZERO) == Some(core::cmp::Ordering::Less)
        });
        if negative != prev_negative {
            out.rhp = out.rhp.saturating_add(1);
        }
        prev_negative = negative;
        let next = next_row(&upper, &lower, tol)?;
        upper = lower;
        lower = next;
    }
    Ok(out)
}

/// Row of coefficients of `s^top, s^(top-2), ...`.
fn first_row<T: Float + Copy, const N: usize>(
    coeffs: &[T],
    top: usize,
    tol: T,
) -> Row<T, N> {
    let mut row = [Term::zero(); N];
    let picks = coeffs.iter().take(top.saturating_add(1)).rev().step_by(2);
    for (dst, &c) in row.iter_mut().zip(picks) {
        *dst = Term::constant(c, tol);
    }
    row
}

/// Coefficients of the derivative of the auxiliary polynomial of degree
/// `order` held in `row` (powers `order, order - 2, ...`).
fn aux_derivative<T: Float + Copy, const N: usize>(
    row: &Row<T, N>,
    order: usize,
    tol: T,
) -> RowResult<T, N> {
    let mut out = [Term::zero(); N];
    for ((dst, src), j) in out.iter_mut().zip(row).zip(0usize..) {
        let power = order.saturating_sub(j.saturating_mul(2));
        let factor = Term::constant(T::from_usize(power), tol);
        *dst = src.mul(factor, tol)?;
    }
    Ok(out)
}

/// Next Routh row: `(l0 * u[j+1] - u0 * l[j+1]) / l0`.
fn next_row<T: Float + Copy, const N: usize>(
    upper: &Row<T, N>,
    lower: &Row<T, N>,
    tol: T,
) -> RowResult<T, N> {
    let mut out = [Term::zero(); N];
    let u0 = upper.first().copied().unwrap_or(Term::zero());
    let l0 = lower.first().copied().unwrap_or(Term::zero());
    if l0.is_zero() {
        return Ok(out);
    }
    let shifted = upper.iter().skip(1).zip(lower.iter().skip(1));
    for (dst, (&u, &l)) in out.iter_mut().zip(shifted) {
        *dst = l0.mul(u, tol)?.minus(u0.mul(l, tol)?, tol)?.div(l0, tol)?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ascending coefficients with the right-half-plane root count from
    /// `numpy.roots` (no roots on the imaginary axis, none symmetric).
    const EPSILON_ROW_CASES: [EpsilonCase; 6] = [
        ([-3.0, -3.0, -3.0, -3.0, 0.0, 1.0], 3),
        ([-3.0, -3.0, -3.0, -2.0, 0.0, 2.0], 3),
        ([-3.0, -3.0, -3.0, -1.0, 0.0, -2.0], 2),
        ([-3.0, -3.0, -3.0, -1.0, 0.0, -1.0], 2),
        ([-3.0, -3.0, -3.0, 1.0, 0.0, 1.0], 1),
        ([-3.0, -3.0, -3.0, 2.0, 0.0, 3.0], 1),
    ];

    const UNIT: [f64; TERMS] = [1.0, 0.0, 0.0, 0.0];

    /// Ascending coefficients and the right-half-plane root count.
    type EpsilonCase = ([f64; 6], usize);

    #[test]
    fn routh_matches_known_roots() {
        let base = [-1.0, -2.0, -3.0, -4.0, -5.0];
        for rhp in 0..=4 {
            let mut roots = [0.0; 6];
            for ((dst, &r), k) in roots.iter_mut().zip(&base).zip(0..) {
                *dst = if k < rhp { -r } else { r };
            }
            let poly = ArrayPolynomial::<f64, 6>::from_roots(roots);
            let got = routh_count(&poly, 1e-12).unwrap();
            assert_eq!(got.rhp, rhp);
            assert!(!got.imaginary_axis);
        }
    }

    #[test]
    fn routh_zero_first_column() {
        let poly = ArrayPolynomial::<f64, 5>::from_coefficients([
            3.0, 2.0, 2.0, 1.0, 1.0,
        ]);
        let got = routh_count(&poly, 1e-12).unwrap();
        assert_eq!(got.rhp, 2);
    }

    #[test]
    fn routh_zero_row() {
        let poly =
            ArrayPolynomial::<f64, 4>::from_coefficients([1.0, 1.0, 1.0, 1.0]);
        let got = routh_count(&poly, 1e-12).unwrap();
        assert_eq!(got.rhp, 0);
        assert!(got.imaginary_axis);
    }

    #[test]
    fn routh_zero_leading_coefficient() {
        let poly =
            ArrayPolynomial::<f64, 3>::from_coefficients([1.0, 1.0, 0.0]);
        assert_eq!(
            routh_count(&poly, 1e-12),
            Err(ClassicalError::ZeroLeadingCoefficient)
        );
    }

    #[test]
    fn negative_leading_coefficient() {
        // -(s + 1)(s + 2): no right-half-plane roots.
        let poly =
            ArrayPolynomial::<f64, 3>::from_coefficients([-2.0, -3.0, -1.0]);
        let got = routh_count(&poly, 1e-12).unwrap();
        assert_eq!(got.rhp, 0);
    }

    fn term(c: [f64; TERMS], power: i32) -> Term<f64> {
        term_until(c, power, EXACT)
    }

    fn term_until(c: [f64; TERMS], power: i32, until: i32) -> Term<f64> {
        Term { c, power, until }
    }

    #[test]
    fn term_mul_propagates_validity() {
        let tol = 1e-12;
        let p = term_until(UNIT, 1, 10).mul(term(UNIT, 2), tol).unwrap();
        assert_eq!(p.until, 12);
        let p = term(UNIT, 1).mul(term_until(UNIT, 2, 10), tol).unwrap();
        assert_eq!(p.until, 11);
        let p = term(UNIT, 1).mul(term(UNIT, 2), tol).unwrap();
        assert_eq!(p.until, EXACT);
    }

    #[test]
    fn term_minus_tracks_dropped_terms() {
        let tol = 1e-12;
        // Terms shifted past the stored width are lost.
        let lost = term(UNIT, 0)
            .minus(term([0.0, 0.0, 0.0, 1.0], 5), tol)
            .unwrap();
        assert_eq!(lost.until, 4);
        // Shifted terms that land in the upper half of the buffer are lost.
        let lost = term(UNIT, 0).minus(term([1.0; TERMS], 2), tol).unwrap();
        assert_eq!(lost.until, 4);
        let kept = term(UNIT, 0).minus(term(UNIT, 3), tol).unwrap();
        assert_eq!(kept.until, EXACT);
    }

    #[test]
    fn term_div_propagates_validity() {
        let tol = 1e-12;
        let q = term_until(UNIT, 0, 10).div(term(UNIT, 2), tol).unwrap();
        assert_eq!((q.power, q.until), (-2, 8));
        let q = term(UNIT, 0).div(term(UNIT, 2), tol).unwrap();
        assert_eq!((q.power, q.until), (-2, EXACT));
        let q = term(UNIT, 0).div(term_until(UNIT, 1, 10), tol).unwrap();
        assert_eq!((q.power, q.until), (-1, 8));
    }

    #[test]
    fn term_constant_chops_at_tolerance() {
        assert!(Term::constant(1e-3, 1e-3).is_zero());
        assert!(!Term::constant(2e-3, 1e-3).is_zero());
        assert!(Term::<f64>::zero().is_zero());
        assert!(!term(UNIT, 0).is_zero());
    }

    #[test]
    fn routh_root_at_origin_sets_imaginary_axis() {
        // s (s + 1)(s + 2)
        let poly =
            ArrayPolynomial::<f64, 4>::from_coefficients([0.0, 2.0, 3.0, 1.0]);
        let got = routh_count(&poly, 1e-12).unwrap();
        assert_eq!((got.rhp, got.imaginary_axis), (0, true));
    }

    #[test]
    fn symmetric_roots_detection() {
        // 1 + 0 s + s^2 and 1 + s + s^2 + s^3 have roots +-i.
        assert!(symmetric_roots_possible::<f64, 4>(&[1.0, 0.0, 1.0]));
        assert!(symmetric_roots_possible::<f64, 4>(&[1.0, 1.0, 1.0, 1.0]));
        // Degree below two never pairs, even with a zero constant.
        assert!(!symmetric_roots_possible::<f64, 4>(&[0.0, 1.0]));
        assert!(!symmetric_roots_possible::<f64, 4>(&[1.0, 1.0]));
        assert!(!symmetric_roots_possible::<f64, 4>(&[1.0]));
        assert!(!symmetric_roots_possible::<f64, 4>(&[2.0, 3.0, 1.0]));
    }

    #[test]
    fn run_epsilon_only_when_allowed() {
        let c = [3.0, 2.0, 2.0, 1.0, 1.0];
        assert_eq!(run::<f64, 5>(&c, 1e-12, false), Err(Fail::Singular));
        let got = run::<f64, 5>(&c, 1e-12, true).unwrap();
        assert_eq!(got.rhp, 2);
    }

    #[test]
    fn run_zero_leading_is_not_negative() {
        let got = run::<f64, 3>(&[1.0, 1.0, 0.0], 1e-12, false).unwrap();
        assert_eq!(got.rhp, 0);
    }

    #[test]
    fn term_minus_keeps_dominant_order_and_lower_terms() {
        let tol = 1e-12;
        // 2 epsilon - 3 epsilon^2: the lower power dominates and keeps both terms.
        let a = term([2.0, 0.0, 0.0, 0.0], 1);
        let b = term([3.0, 0.0, 0.0, 0.0], 2);
        let d = a.minus(b, tol).unwrap();
        assert_eq!((d.c, d.power), ([2.0, -3.0, 0.0, 0.0], 1));
        // 0 - x = -x and x - 0 = x.
        let z = Term::zero();
        assert_eq!(
            z.minus(a, tol).unwrap().head().to_bits(),
            (-2.0f64).to_bits()
        );
        assert_eq!(a.minus(z, tol).unwrap().head().to_bits(), 2.0f64.to_bits());
    }

    #[test]
    fn term_cancellation_promotes_next_order() {
        // (3/epsilon - 3) - (3/epsilon - 3) is exactly zero. With different lower
        // terms the leading terms cancel and the remainder, not zero, becomes
        // the head: a leading-term-only entry would have dropped it.
        let tol = 1e-12;
        let a = term([3.0, -3.0, 0.0, 0.0], -1);
        assert!(a.minus(a, tol).unwrap().is_zero());
        let b = term([3.0, -2.0, 0.0, 0.0], -1);
        let d = a.minus(b, tol).unwrap();
        assert_eq!((d.head(), d.power), (-1.0, 0));
    }

    #[test]
    fn term_mul_by_zero() {
        let tol = 1e-12;
        let x = term([2.0, 1.0, 0.0, 0.0], 3);
        assert!(Term::zero().mul(x, tol).unwrap().is_zero());
        assert!(x.mul(Term::zero(), tol).unwrap().is_zero());
        let p = x.mul(term([4.0, 0.0, 0.0, 0.0], -1), tol).unwrap();
        assert_eq!((p.c, p.power), ([8.0, 4.0, 0.0, 0.0], 2));
    }

    #[test]
    fn term_div_series() {
        // (1) / (1 + epsilon) = 1 - epsilon + epsilon^2 - epsilon^3 + ...
        let q = term([1.0, 0.0, 0.0, 0.0], 0)
            .div(term([1.0, 1.0, 0.0, 0.0], 0), 1e-12)
            .unwrap();
        assert_eq!(
            q.c.map(f64::to_bits),
            [1.0, -1.0, 1.0, -1.0].map(f64::to_bits)
        );
        assert!(q.until != EXACT);
    }

    #[test]
    fn epsilon_row_cancellation_keeps_coefficients() {
        // FR-2: a zero first-column entry becomes epsilon; when the leading
        // terms of a later entry cancel, the next-order term must survive.
        // Dropping it reports a false imaginary axis and counts too few roots.
        for (c, rhp) in EPSILON_ROW_CASES {
            let poly = ArrayPolynomial::<f64, 6>::from_coefficients(c);
            let got = routh_count(&poly, 1e-12).unwrap();
            assert_eq!(got.rhp, rhp, "{c:?}");
            assert!(!got.imaginary_axis, "{c:?}");
        }
    }

    #[test]
    fn routh_count_is_invariant_to_positive_scaling() {
        for (c, rhp) in EPSILON_ROW_CASES {
            for k in [0.5, 4.0] {
                let scaled = c.map(|v| v * k);
                let poly = ArrayPolynomial::<f64, 6>::from_coefficients(scaled);
                let got = routh_count(&poly, 1e-12).unwrap();
                assert_eq!((got.rhp, got.imaginary_axis), (rhp, false));
            }
        }
    }

    /// Deterministic xorshift for the sweep below.
    fn next(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    #[test]
    fn routh_agrees_with_roots_on_integer_polynomials() {
        // Oracle: the right-half-plane count of `Polynomial::roots` for
        // integer coefficients in [-3, 3] whose roots are off the axis,
        // unpaired and separated. The test is silent on how the array was
        // completed; it may report `RouthIndeterminate` but never a wrong
        // count (FR-2).
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let (mut checked, mut indeterminate) = (0u32, 0u32);
        for _ in 0..if cfg!(miri) { 40 } else { 4000 } {
            let c: [f64; 7] = core::array::from_fn(|_| {
                f64::from(i8::try_from(next(&mut state) % 7).unwrap() - 3)
            });
            if c[6] == 0.0 {
                continue;
            }
            let poly = ArrayPolynomial::<f64, 7>::from_coefficients(c);
            let Ok(roots) = poly.roots() else { continue };
            let roots = roots.get(..6).unwrap();
            let off_axis = roots.iter().all(|r| r.re.abs() > 1e-6);
            let unpaired = roots.iter().enumerate().all(|(i, a)| {
                roots.get(..i).unwrap().iter().all(|b| {
                    (a.re + b.re).hypot(a.im + b.im) > 1e-3
                        && (a.re - b.re).hypot(a.im - b.im) > 1e-2
                })
            });
            if !(off_axis && unpaired) {
                continue;
            }
            let want = roots.iter().filter(|r| r.re > 0.0).count();
            checked += 1;
            match routh_count(&poly, 1e-12) {
                Ok(got) => assert_eq!(
                    (got.rhp, got.imaginary_axis),
                    (want, false),
                    "{c:?}"
                ),
                Err(ClassicalError::RouthIndeterminate) => indeterminate += 1,
                Err(e) => panic!("{c:?}: {e}"),
            }
        }
        assert!(checked > 100 || cfg!(miri));
        assert!(
            indeterminate * 50 <= checked.max(1),
            "{indeterminate}/{checked}"
        );
    }
}
