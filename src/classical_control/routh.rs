//! Routh-Hurwitz right-half-plane root count (FR-2).
//!
//! The Routh array is built two rows at a time from the coefficient slice.
//! A zero first-column entry with a nonzero remainder is replaced by
//! `eps > 0` and the array is carried as leading terms
//! `c eps^k`, so the sign of each entry is its sign as
//! `eps -> 0^+`; no numeric `eps` is used. A row of zeros is
//! replaced by the coefficients of the derivative of the auxiliary
//! polynomial and sets the imaginary-axis flag.

use super::ClassicalError;
use crate::math::num_traits::Float;
use crate::math::num_types::{Const, Dim};
use crate::polynomial::ArrayPolynomial;

/// Result of the Routh-Hurwitz test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RouthCount {
    /// Number of roots in the open right half plane.
    pub rhp: usize,
    /// Whether a row of zeros indicated roots symmetric about the origin,
    /// including roots on the imaginary axis.
    pub imaginary_axis: bool,
}

/// A Routh row of leading terms.
type Row<T, const N: usize> = [Term<T>; N];

/// Leading term `coeff * eps^power` of a Routh entry as `eps -> 0+`.
#[derive(Debug, Clone, Copy)]
struct Term<T> {
    coeff: T,
    power: i32,
}

impl<T: Float + Copy> Term<T> {
    const fn zero() -> Self {
        Self {
            coeff: T::ZERO,
            power: 0,
        }
    }

    fn is_zero(self) -> bool {
        self.coeff == T::ZERO
    }

    /// `self - rhs`, keeping the dominant term as `eps -> 0+`. Equal-order
    /// terms whose difference is within `tol` cancel to zero.
    fn minus(self, rhs: Self, tol: T) -> Self {
        if self.is_zero() {
            return rhs.neg();
        }
        if rhs.is_zero() || self.power < rhs.power {
            return self;
        }
        if rhs.power < self.power {
            return rhs.neg();
        }
        Self::chop(self.coeff.saturating_sub(&rhs.coeff), self.power, tol)
    }

    fn chop(coeff: T, power: i32, tol: T) -> Self {
        if coeff.abs() <= tol {
            Self::zero()
        } else {
            Self { coeff, power }
        }
    }

    fn div(self, d: Self) -> Self {
        if self.is_zero() {
            return self;
        }
        Self {
            coeff: self.coeff.saturating_div(&d.coeff),
            power: self.power.saturating_sub(d.power),
        }
    }

    fn mul(self, b: Self) -> Self {
        if self.is_zero() || b.is_zero() {
            return Self::zero();
        }
        Self {
            coeff: self.coeff.saturating_mul(&b.coeff),
            power: self.power.saturating_add(b.power),
        }
    }

    fn neg(self) -> Self {
        Self {
            coeff: T::ZERO.saturating_sub(&self.coeff),
            power: self.power,
        }
    }
}

/// Counts the open right-half-plane roots of a real polynomial without
/// computing them (FR-2).
///
/// `poly` holds ascending coefficients of degree `N - 1`. A computed or given
/// coefficient with magnitude at most `tol` is treated as zero. The array
/// has a fixed operation count of order `N^2` (NFR-3) and uses two stack
/// rows.
///
/// # Errors
/// [`ClassicalError::ZeroLeadingCoefficient`] when the leading coefficient
/// is within `tol` of zero.
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
    let mut upper = first_row::<T, N>(coeffs, degree, tol);
    let mut lower = first_row::<T, N>(coeffs, degree.saturating_sub(1), tol);
    let mut prev_negative = leading < T::ZERO;
    let mut out = RouthCount {
        rhp: 0,
        imaginary_axis: false,
    };
    for order in (0..degree).rev() {
        if lower.iter().all(|t| t.is_zero()) {
            out.imaginary_axis = true;
            lower = aux_derivative(&upper, order.saturating_add(1));
        } else if let Some(head) = lower.first_mut().filter(|t| t.is_zero()) {
            *head = Term {
                coeff: T::ONE,
                power: 1,
            };
        }
        let negative = lower.first().is_some_and(|t| t.coeff < T::ZERO);
        if negative != prev_negative {
            out.rhp = out.rhp.saturating_add(1);
        }
        prev_negative = negative;
        let next = next_row(&upper, &lower, tol);
        upper = lower;
        lower = next;
    }
    Ok(out)
}

/// Row of coefficients of `s^highest, s^(highest-2), ...`.
fn first_row<T: Float + Copy, const N: usize>(
    coeffs: &[T],
    highest: usize,
    tol: T,
) -> Row<T, N> {
    let mut row = [Term::zero(); N];
    let picks = coeffs
        .iter()
        .take(highest.saturating_add(1))
        .rev()
        .step_by(2);
    for (dst, &c) in row.iter_mut().zip(picks) {
        *dst = Term::chop(c, 0, tol);
    }
    row
}

/// Coefficients of the derivative of the auxiliary polynomial of degree
/// `order` held in `row` (powers `order, order - 2, ...`).
fn aux_derivative<T: Float + Copy, const N: usize>(
    row: &Row<T, N>,
    order: usize,
) -> Row<T, N> {
    let mut out = [Term::zero(); N];
    for ((dst, src), j) in out.iter_mut().zip(row).zip(0usize..) {
        let power = order.saturating_sub(j.saturating_mul(2));
        let factor = Term {
            coeff: T::from_usize(power),
            power: 0,
        };
        *dst = src.mul(factor);
    }
    out
}

/// Next Routh row: `(l0 * u[j+1] - u0 * l[j+1]) / l0`.
fn next_row<T: Float + Copy, const N: usize>(
    upper: &Row<T, N>,
    lower: &Row<T, N>,
    tol: T,
) -> Row<T, N> {
    let mut out = [Term::zero(); N];
    let u0 = upper.first().copied().unwrap_or(Term::zero());
    let l0 = lower.first().copied().unwrap_or(Term::zero());
    if l0.is_zero() {
        return out;
    }
    let shifted = upper.iter().skip(1).zip(lower.iter().skip(1));
    for (dst, (&u, &l)) in out.iter_mut().zip(shifted) {
        *dst = l0.mul(u).minus(u0.mul(l), tol).div(l0);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn term_minus_dominant_order() {
        let term = |coeff, power| Term { coeff, power };
        let tol = 1e-12;
        let a = term(2.0, 1).minus(Term::zero(), tol);
        assert_eq!((a.coeff, a.power), (2.0, 1));
        let b = term(2.0, 1).minus(term(3.0, 2), tol);
        assert_eq!((b.coeff, b.power), (2.0, 1));
        let c = term(2.0, 2).minus(term(3.0, 1), tol);
        assert_eq!((c.coeff, c.power), (-3.0, 1));
    }

    #[test]
    fn term_mul_by_zero() {
        let zero_times = Term::zero().mul(Term {
            coeff: 2.0,
            power: 3,
        });
        let times_zero = Term {
            coeff: 2.0,
            power: 3,
        }
        .mul(Term::zero());
        assert_eq!((zero_times.coeff, zero_times.power), (0.0, 0));
        assert_eq!((times_zero.coeff, times_zero.power), (0.0, 0));
    }
}
