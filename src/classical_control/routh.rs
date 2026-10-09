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

/// A Routh row of ε-polynomial entries.
type Row<T, const N: usize> = [Term<T>; N];

/// Up to two lowest-order monomials of a Routh entry as `eps -> 0+`.
///
/// A single leading monomial is not enough: when two equal-order leading
/// terms cancel, the sign of the entry is the next power of `eps`, which a
/// one-term representation has already discarded.
#[derive(Debug, Clone, Copy)]
struct Term<T> {
    /// Lowest-order coefficient; zero means the whole entry is zero.
    coeff: T,
    /// Power of `eps` for [`Self::coeff`].
    power: i32,
    /// Next-order coefficient, or zero when unused.
    next_coeff: T,
    /// Power of `eps` for [`Self::next_coeff`]; ignored when
    /// `next_coeff == 0`.
    next_power: i32,
}

/// One monomial `coeff * eps^power` while combining Routh entries.
#[derive(Debug, Clone, Copy)]
struct Mono<T> {
    coeff: T,
    power: i32,
}

/// Fixed workspace of at most four monomials while combining two two-term
/// Routh entries.
type MonoScratch<T> = [Mono<T>; 4];

impl<T: Float + Copy> Mono<T> {
    const fn zero() -> Self {
        Self {
            coeff: T::ZERO,
            power: 0,
        }
    }
}

impl<T: Float + Copy> Term<T> {
    const fn zero() -> Self {
        Self {
            coeff: T::ZERO,
            power: 0,
            next_coeff: T::ZERO,
            next_power: 0,
        }
    }

    fn is_zero(self) -> bool {
        self.coeff == T::ZERO
    }

    /// One monomial `coeff * eps^power`, or zero when `|coeff| <= tol`.
    fn chop(coeff: T, power: i32, tol: T) -> Self {
        if coeff.abs() <= tol {
            Self::zero()
        } else {
            Self {
                coeff,
                power,
                next_coeff: T::ZERO,
                next_power: 0,
            }
        }
    }

    /// Builds a term from up to four raw monomials, keeping the two lowest
    /// distinct powers after like terms combine and `|c| <= tol` drops.
    fn from_monomials(parts: MonoScratch<T>, tol: T) -> Self {
        let mut merged: MonoScratch<T> = [Mono::zero(); 4];
        let mut n = 0usize;
        for part in parts {
            if part.coeff.abs() <= tol {
                continue;
            }
            let existing = merged.get(..n).and_then(|slot| {
                slot.iter().position(|m| m.power == part.power)
            });
            if let Some(i) = existing {
                let Some(cur) = merged.get(i).copied() else {
                    continue;
                };
                let sum = cur.coeff.saturating_add(&part.coeff);
                if sum.abs() <= tol {
                    // Remove cancelled slot by swapping with the last.
                    n = n.saturating_sub(1);
                    if i < n {
                        let Some(tail) = merged.get(n).copied() else {
                            continue;
                        };
                        if let Some(dst) = merged.get_mut(i) {
                            *dst = tail;
                        }
                    }
                } else if let Some(dst) = merged.get_mut(i) {
                    dst.coeff = sum;
                }
            } else if let Some(dst) = merged.get_mut(n) {
                *dst = part;
                n = n.saturating_add(1);
            }
        }
        // Sort ascending by power (selection sort; n <= 4).
        for i in 0..n {
            let mut best = i;
            for j in i.saturating_add(1)..n {
                let pj = merged.get(j).map(|m| m.power);
                let pb = merged.get(best).map(|m| m.power);
                if matches!((pj, pb), (Some(a), Some(b)) if a < b) {
                    best = j;
                }
            }
            if best != i {
                merged.swap(i, best);
            }
        }
        let first = merged.first().copied().unwrap_or(Mono::zero());
        let second = merged.get(1).copied().unwrap_or(Mono::zero());
        match n {
            0 => Self::zero(),
            1 => Self::chop(first.coeff, first.power, T::ZERO),
            _ => Self {
                coeff: first.coeff,
                power: first.power,
                next_coeff: second.coeff,
                next_power: second.power,
            },
        }
    }

    const fn low(self) -> Mono<T> {
        Mono {
            coeff: self.coeff,
            power: self.power,
        }
    }

    const fn high(self) -> Mono<T> {
        Mono {
            coeff: self.next_coeff,
            power: self.next_power,
        }
    }

    /// `self - rhs`, retaining the two lowest powers of `eps`.
    fn minus(self, rhs: Self, tol: T) -> Self {
        let neg = rhs.neg();
        Self::from_monomials(
            [self.low(), self.high(), neg.low(), neg.high()],
            tol,
        )
    }

    fn div(self, d: Self) -> Self {
        if self.is_zero() || d.is_zero() {
            return Self::zero();
        }
        // Divide by the dominant monomial of `d` (Routh pivots are leading
        // terms). Higher-order content in `d` is neglected, matching the
        // classical ε → 0+ pivot.
        let scale = d.coeff;
        let dp = d.power;
        Self {
            coeff: self.coeff.saturating_div(&scale),
            power: self.power.saturating_sub(dp),
            next_coeff: if self.next_coeff == T::ZERO {
                T::ZERO
            } else {
                self.next_coeff.saturating_div(&scale)
            },
            next_power: self.next_power.saturating_sub(dp),
        }
    }

    fn mul(self, b: Self) -> Self {
        if self.is_zero() || b.is_zero() {
            return Self::zero();
        }
        let prod = |x: Mono<T>, y: Mono<T>| -> Mono<T> {
            if x.coeff == T::ZERO || y.coeff == T::ZERO {
                Mono::zero()
            } else {
                Mono {
                    coeff: x.coeff.saturating_mul(&y.coeff),
                    power: x.power.saturating_add(y.power),
                }
            }
        };
        // `tol = 0`: products of already-chopped monomials stay exact here.
        Self::from_monomials(
            [
                prod(self.low(), b.low()),
                prod(self.low(), b.high()),
                prod(self.high(), b.low()),
                prod(self.high(), b.high()),
            ],
            T::ZERO,
        )
    }

    fn neg(self) -> Self {
        Self {
            coeff: T::ZERO.saturating_sub(&self.coeff),
            power: self.power,
            next_coeff: T::ZERO.saturating_sub(&self.next_coeff),
            next_power: self.next_power,
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
                next_coeff: T::ZERO,
                next_power: 0,
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

/// Row of coefficients of `s^start, s^(start-2), ...`.
fn first_row<T: Float + Copy, const N: usize>(
    coeffs: &[T],
    start: usize,
    tol: T,
) -> Row<T, N> {
    let mut row = [Term::zero(); N];
    let picks = coeffs.iter().take(start.saturating_add(1)).rev().step_by(2);
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
            next_coeff: T::ZERO,
            next_power: 0,
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
        let term = |coeff, power| Term {
            coeff,
            power,
            next_coeff: 0.0,
            next_power: 0,
        };
        let tol = 1e-12;
        let a = term(2.0, 1).minus(Term::zero(), tol);
        assert_eq!((a.coeff, a.power), (2.0, 1));
        let b = term(2.0, 1).minus(term(3.0, 2), tol);
        assert_eq!(
            (b.coeff, b.power, b.next_coeff, b.next_power),
            (2.0, 1, -3.0, 2)
        );
        let c = term(2.0, 2).minus(term(3.0, 1), tol);
        assert_eq!(
            (c.coeff, c.power, c.next_coeff, c.next_power),
            (-3.0, 1, 2.0, 2)
        );
    }

    #[test]
    fn routh_eps_cancellation_keeps_next_order() {
        // Degree-7 poly with a zero `s^6` coefficient: the ε first-column
        // replacement cancels at leading order on a later row; a one-term
        // representation falsely reports a zero row (imaginary-axis) and
        // under-counts RHP roots.
        let roots = [1.0, 2.0, -3.0, 4.0, -5.0, -6.0, 7.0, 0.0];
        let poly = ArrayPolynomial::<f64, 8>::from_roots(roots);
        let got = routh_count(&poly, 1e-12).unwrap();
        assert_eq!(got.rhp, 4);
        assert!(!got.imaginary_axis);
    }

    #[test]
    fn term_mul_by_zero() {
        let mono = |coeff, power| Term {
            coeff,
            power,
            next_coeff: 0.0,
            next_power: 0,
        };
        let zero_times = Term::zero().mul(mono(2.0, 3));
        let times_zero = mono(2.0, 3).mul(Term::zero());
        assert_eq!((zero_times.coeff, zero_times.power), (0.0, 0));
        assert_eq!((times_zero.coeff, times_zero.power), (0.0, 0));
    }
}
