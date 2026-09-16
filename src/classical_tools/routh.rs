//! # Routh-Hurwitz Stability Engine
//!
//! Determines the number of open right-half-plane (RHP) roots of a
//! characteristic polynomial directly from its coefficients, without
//! computing the roots themselves (Shamshiri, 2009).
//!
//! Given $P(s) = a_n s^n + a_{n-1} s^{n-1} + \dots + a_0$, the Routh array
//! $R \in \mathbb{R}^{(n+1) \times m}$ ($m = \lceil (n+1)/2 \rceil$) is built
//! two rows at a time:
//! - Row 0: $[a_n, a_{n-2}, a_{n-4}, \dots]$.
//! - Row 1: $[a_{n-1}, a_{n-3}, a_{n-5}, \dots]$.
//! - Row $i \ge 2$, column $j$:
//!   $R_{i,j} = \dfrac{R_{i-1,0} R_{i-2,j+1} - R_{i-2,0} R_{i-1,j+1}}{R_{i-1,0}}$.
//!
//! The number of open RHP roots equals the number of sign changes down the
//! first column $R_{:,0}$ (Shamshiri, 2009). Two degeneracies are resolved
//! (Davidson, 2020; Sagharchi, 2016):
//! - **First-column zero**: the divisor $R_{i-1,0}$ is replaced by a small
//!   `epsilon` to avoid division by zero.
//! - **Row of zeros**: row $i$ vanishing entirely indicates root symmetry
//!   about the origin. The row above it, $R_{i-1,\bullet}$, is reinterpreted
//!   as the auxiliary polynomial
//!   $A(s) = \sum_k R_{i-1,k} s^{n-(i-1)-2k}$, and row $i$ is replaced by the
//!   coefficients of $\frac{d}{ds} A(s)$.
#![allow(
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::needless_range_loop,
    clippy::doc_markdown,
    clippy::type_complexity
)]

use crate::math::num_traits::Float;
use crate::math::num_types::{Const, Dim};
use crate::polynomial::ArrayPolynomial;
use core::fmt;

/// Errors from [`stability`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouthError {
    /// The leading coefficient ($a_n$) is zero: not a valid degree-$n$
    /// characteristic polynomial.
    ZeroLeadingCoefficient,
    /// A row-of-zeros substitute derived from the auxiliary polynomial's
    /// derivative was itself entirely zero; the degeneracy could not be
    /// resolved.
    UnresolvableDegeneracy,
}

impl fmt::Display for RouthError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroLeadingCoefficient => {
                write!(f, "leading coefficient must be non-zero")
            }
            Self::UnresolvableDegeneracy => {
                write!(f, "row-of-zeros substitute was itself all-zero")
            }
        }
    }
}

/// Counts the open right-half-plane roots of `poly`'s characteristic
/// polynomial via the Routh-Hurwitz array.
///
/// `epsilon` is the small positive substitute used for a vanishing
/// first-column divisor, and the tolerance below which a value is treated as
/// zero when detecting a row of zeros (a value of $10^{-12}$ is standard for
/// `f64`; scale accordingly for `f32`).
///
/// # Errors
/// - [`RouthError::ZeroLeadingCoefficient`] if `poly`'s leading coefficient
///   is zero.
/// - [`RouthError::UnresolvableDegeneracy`] if a row-of-zeros substitute
///   derived from the auxiliary polynomial's derivative is itself entirely
///   zero.
pub fn stability<T: Float + Copy, const N: usize>(
    poly: &ArrayPolynomial<T, N>,
    epsilon: T,
) -> Result<usize, RouthError>
where
    Const<N>: Dim,
{
    if N == 0 {
        return Ok(0);
    }
    let descending = descending_coefficients(poly);
    if descending[0] == T::ZERO {
        return Err(RouthError::ZeroLeadingCoefficient);
    }

    let m = N.div_ceil(2);
    let (mut prev2, mut prev1) = initial_rows::<T, N>(&descending, m);

    let mut first_col = [T::ZERO; N];
    first_col[0] = prev2[0];
    if N > 1 {
        first_col[1] = prev1[0];
    }

    for i in 2..N {
        if (0..m).all(|j| prev1[j].abs() <= epsilon) {
            // `prev1` (row i-1) vanished entirely. Rebuild it from the
            // auxiliary polynomial formed by `prev2` (row i-2), which sits
            // at power `p = (N - 1) - (i - 2) = N + 1 - i`.
            prev1 =
                resolve_row_of_zeros::<T, N>(&prev2, N + 1 - i, m, epsilon)?;
        }
        if prev1[0].abs() <= epsilon {
            // First-column zero: substitute a small epsilon so the divisor
            // below never vanishes.
            prev1[0] = epsilon;
        }

        let row = next_row::<T, N>(&prev1, &prev2, m);
        first_col[i] = row[0];
        prev2 = prev1;
        prev1 = row;
    }

    Ok(count_sign_changes(&first_col))
}

/// Extracts `poly`'s coefficients in descending power order
/// ($a_n, a_{n-1}, \dots, a_0$), matching the classical Routh-array
/// presentation (the crate otherwise stores polynomials in ascending order).
fn descending_coefficients<T: Float + Copy, const N: usize>(
    poly: &ArrayPolynomial<T, N>,
) -> [T; N]
where
    Const<N>: Dim,
{
    let ascending = poly.to_coefficients();
    let mut descending = [T::ZERO; N];
    for k in 0..N {
        descending[k] = ascending[N - 1 - k];
    }
    descending
}

/// Builds the Routh array's first two rows directly from the descending
/// coefficients: row 0 takes the even-offset terms, row 1 the odd-offset
/// terms.
fn initial_rows<T: Float + Copy, const N: usize>(
    descending: &[T; N],
    m: usize,
) -> ([T; N], [T; N]) {
    let coeff_at = |i: usize| if i < N { descending[i] } else { T::ZERO };
    let mut row0 = [T::ZERO; N];
    let mut row1 = [T::ZERO; N];
    for j in 0..m {
        row0[j] = coeff_at(2 * j);
        row1[j] = coeff_at(2 * j + 1);
    }
    (row0, row1)
}

/// Rebuilds a vanished row from the derivative of the auxiliary polynomial
/// formed by the row above it (`above`, at power `power`):
/// $A(s) = \sum_k \text{above}_k s^{\text{power} - 2k}$, replaced by
/// $\frac{d}{ds} A(s)$.
fn resolve_row_of_zeros<T: Float + Copy, const N: usize>(
    above: &[T; N],
    power: usize,
    m: usize,
    epsilon: T,
) -> Result<[T; N], RouthError> {
    let mut derivative = [T::ZERO; N];
    for j in 0..m {
        let exponent = T::from_usize(power) - T::from_usize(2 * j);
        derivative[j] = exponent * above[j];
    }
    if (0..m).all(|j| derivative[j].abs() <= epsilon) {
        return Err(RouthError::UnresolvableDegeneracy);
    }
    Ok(derivative)
}

/// Computes the next Routh row from the two rows above it:
/// $R_{i,j} = (R_{i-1,0} R_{i-2,j+1} - R_{i-2,0} R_{i-1,j+1}) / R_{i-1,0}$.
fn next_row<T: Float + Copy, const N: usize>(
    prev1: &[T; N],
    prev2: &[T; N],
    m: usize,
) -> [T; N] {
    let mut row = [T::ZERO; N];
    for j in 0..m {
        let next2 = if j + 1 < N { prev2[j + 1] } else { T::ZERO };
        let next1 = if j + 1 < N { prev1[j + 1] } else { T::ZERO };
        row[j] = (prev1[0] * next2 - prev2[0] * next1) / prev1[0];
    }
    row
}

/// Counts sign changes down the first column, skipping over exact zeros
/// (which carry no sign of their own) rather than treating them as a change.
fn count_sign_changes<T: Float + Copy, const N: usize>(
    first_col: &[T; N],
) -> usize {
    let mut sign_changes = 0_usize;
    let mut last_sign = 0_i32;
    for &value in first_col {
        let sign = if value > T::ZERO {
            1
        } else if value < T::ZERO {
            -1
        } else {
            0
        };
        if sign != 0 {
            if last_sign != 0 && sign != last_sign {
                sign_changes += 1;
            }
            last_sign = sign;
        }
    }
    sign_changes
}
