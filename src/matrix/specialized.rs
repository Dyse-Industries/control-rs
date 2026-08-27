//! Structural specializations.
//!
//! New-type wrappers around [`Owned`] enforcing a mathematical invariant, so
//! callers who hold one of these types get a statically-known guarantee
//! (upper/lower triangular, symmetric) instead of re-checking it. A full
//! square matrix is wrapped (not a packed triangular layout) — this trades
//! memory space for cache-friendly, slice-compatible storage
//! (`matrix-design.md` §4.10).
//!
//! `Const<D>: Dim` is restated on every impl/struct below — see
//! `decomposition.rs`'s module doc for why.
#![allow(
    clippy::arbitrary_source_item_ordering,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    // `l_ii`/`l_ij`/`u_ii`/`u_ij` below are standard linear-algebra index
    // notation, not accidentally-similar English words.
    clippy::similar_names
)]

use super::Owned;
use crate::math::LinAlgResult;
use crate::math::num_traits::Float;
use crate::math::num_types::{Const, Dim};
use crate::math::storage::{Diag, Trans, UpLo};
use crate::math::subprograms::{DefaultBlas, level2::Trsv};

/// A square matrix known to have all entries below the diagonal within
/// tolerance of zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UpperTriangular<T, const D: usize>(Owned<T, D, D>)
where
    Const<D>: Dim;

/// A square matrix known to have all entries above the diagonal within
/// tolerance of zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LowerTriangular<T, const D: usize>(Owned<T, D, D>)
where
    Const<D>: Dim;

/// A square matrix known to satisfy `m[i][j] == m[j][i]` within tolerance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Symmetric<T, const D: usize>(Owned<T, D, D>)
where
    Const<D>: Dim;

impl<T: Float + Copy, const D: usize> UpperTriangular<T, D>
where
    Const<D>: Dim,
{
    /// Wraps `m` if every entry below the diagonal is within `T::epsilon()`
    /// of zero, otherwise returns `None`.
    #[must_use]
    pub fn from_owned(m: Owned<T, D, D>) -> Option<Self> {
        for i in 0..D {
            for j in 0..i {
                let below_diagonal = m.get(i, j).copied().unwrap_or(T::ZERO);
                if below_diagonal.abs() >= T::epsilon() {
                    return None;
                }
            }
        }
        Some(Self(m))
    }

    /// Borrows the wrapped matrix.
    #[must_use]
    pub const fn as_matrix(&self) -> &Owned<T, D, D> {
        &self.0
    }

    /// Unwraps the underlying matrix.
    #[must_use]
    pub const fn into_inner(self) -> Owned<T, D, D> {
        self.0
    }

    /// Wraps `m` without validating the upper-triangular invariant.
    /// Crate-internal: only for decomposition code that has just produced a
    /// provably upper-triangular result (e.g. `QrDecomposition`'s `R`
    /// factor), where re-validating via `from_owned`'s `T::epsilon()` scan
    /// would be redundant.
    pub(super) const fn from_owned_unchecked(m: Owned<T, D, D>) -> Self {
        Self(m)
    }
}

impl<T: Float + Copy, const D: usize> LowerTriangular<T, D>
where
    Const<D>: Dim,
{
    /// Wraps `m` if every entry above the diagonal is within `T::epsilon()`
    /// of zero, otherwise returns `None`.
    #[must_use]
    pub fn from_owned(m: Owned<T, D, D>) -> Option<Self> {
        for i in 0..D {
            for j in (i + 1)..D {
                let above_diagonal = m.get(i, j).copied().unwrap_or(T::ZERO);
                if above_diagonal.abs() >= T::epsilon() {
                    return None;
                }
            }
        }
        Some(Self(m))
    }

    /// Borrows the wrapped matrix.
    #[must_use]
    pub const fn as_matrix(&self) -> &Owned<T, D, D> {
        &self.0
    }

    /// Unwraps the underlying matrix.
    #[must_use]
    pub const fn into_inner(self) -> Owned<T, D, D> {
        self.0
    }

    /// Wraps `m` without validating the lower-triangular invariant.
    /// Crate-internal: only for decomposition code that has just produced a
    /// provably lower-triangular result (e.g. `CholeskyDecomposition`'s `L`
    /// factor), where re-validating via `from_owned`'s `T::epsilon()` scan
    /// would be redundant.
    pub(super) const fn from_owned_unchecked(m: Owned<T, D, D>) -> Self {
        Self(m)
    }
}

impl<T: Float + Copy, const D: usize> Symmetric<T, D>
where
    Const<D>: Dim,
{
    /// Wraps `m` if `m[i][j]` and `m[j][i]` agree within `T::epsilon()` for
    /// every `i < j`, otherwise returns `None`.
    #[must_use]
    pub fn from_owned(m: Owned<T, D, D>) -> Option<Self> {
        for i in 0..D {
            for j in (i + 1)..D {
                let upper = m.get(i, j).copied().unwrap_or(T::ZERO);
                let lower = m.get(j, i).copied().unwrap_or(T::ZERO);
                if (upper - lower).abs() >= T::epsilon() {
                    return None;
                }
            }
        }
        Some(Self(m))
    }

    /// Borrows the wrapped matrix.
    #[must_use]
    pub const fn as_matrix(&self) -> &Owned<T, D, D> {
        &self.0
    }

    /// Mutably borrows the wrapped matrix. Crate-internal: mutation could
    /// break the symmetry invariant, so this is only exposed to `matrix`'s
    /// own decomposition code, which restores a well-defined state
    /// (`LdltDecomposition`) before returning control.
    pub(super) const fn as_matrix_mut(&mut self) -> &mut Owned<T, D, D> {
        &mut self.0
    }

    /// Unwraps the underlying matrix.
    #[must_use]
    pub const fn into_inner(self) -> Owned<T, D, D> {
        self.0
    }
}

/// Solves `L * x = b` for a lower triangular `L`, via forward substitution.
///
/// # Errors
/// Returns [`LinAlgError::SingularMatrix`] if any diagonal entry of `L` is
/// within `T::epsilon()` of zero.
#[allow(clippy::type_complexity)]
pub fn solve_lower_triangular<T: Float + Copy, const D: usize>(
    l: &LowerTriangular<T, D>,
    b: &Owned<T, D, 1>,
) -> LinAlgResult<Owned<T, D, 1>>
where
    Const<D>: Dim,
{
    let mut x = *b;
    DefaultBlas::trsv(
        UpLo::Lower,
        Trans::NoTrans,
        Diag::NonUnit,
        &l.as_matrix().storage,
        &mut x.storage,
    )?;
    Ok(x)
}

/// Solves `U * x = b` for an upper triangular `U`, via back substitution.
///
/// # Errors
/// Returns [`LinAlgError::SingularMatrix`] if any diagonal entry of `U` is
/// within `T::epsilon()` of zero.
#[allow(clippy::type_complexity)]
pub fn solve_upper_triangular<T: Float + Copy, const D: usize>(
    u: &UpperTriangular<T, D>,
    b: &Owned<T, D, 1>,
) -> LinAlgResult<Owned<T, D, 1>>
where
    Const<D>: Dim,
{
    let mut x = *b;
    DefaultBlas::trsv(
        UpLo::Upper,
        Trans::NoTrans,
        Diag::NonUnit,
        &u.as_matrix().storage,
        &mut x.storage,
    )?;
    Ok(x)
}
