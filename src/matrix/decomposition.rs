//! In-place matrix factorizations.
//!
//! Decomposition objects encapsulate a matrix's factors alongside
//! statically bounded auxiliary state (pivot indices), with no heap
//! allocation — `matrix-design.md` §4.7. Explicit decomposition objects
//! (rather than a convenience `invert() -> Matrix`) avoid hiding an
//! `O(N^3)` operation's stack allocation and let repeated solves reuse one
//! factorization (`matrix-design.md` §5.1).
//!
//! Concrete `impl<T, const D: usize>` blocks (not `Dim`-generic) throughout:
//! `matrix-design.md` §4.7 itself flags the `Dim`-vs-`const usize` bridge
//! for decomposition objects as illustrative/unresolved, and staying
//! concrete lets pivot/solve scratch use plain `[T; D]`/`[usize; D]` stack
//! arrays. `Const<D>: Dim` (and `Const<COLS>: Dim` for `solve_mut`) is
//! restated on every impl/struct below because `Dim` is implemented per
//! literal (`Const<1>` .. `Const<1024>` plus extra powers of two through
//! `Const<16384>`, `num_types` generated impls), not via a blanket
//! `impl<const N: usize> Dim for Const<N>` — the
//! same pattern `storage.rs`'s own `ArrayStorage` impls already use.
#![allow(
    clippy::arbitrary_source_item_ordering,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    // `l_ik`/`l_jk`/`u_ii`/`d_j` etc. below are standard linear-algebra
    // index notation, not accidentally-similar English words.
    clippy::similar_names,
    // Every loop below indexes both a local scratch array (`y`, `col`) and
    // `self.data.storage` by the same loop variable in the same body, so
    // no single iterator covers both accesses.
    clippy::needless_range_loop,
    clippy::type_complexity,
    clippy::doc_markdown,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::option_if_let_else,
    clippy::must_use_candidate
)]

use super::{LowerTriangular, Owned, Symmetric, UpperTriangular};
use crate::math::num_traits::{Float, Radical};
use crate::math::num_types::{Const, Dim};
use crate::math::storage::{
    DenseStorage, DenseStorageMut, Diag, Side, Trans, UpLo,
};
use crate::math::subprograms::{
    DefaultBlas,
    lapack::{Geqrf, Getrf, Getrs, Ormqr, Potrf, Potrs},
    level3::{Gemm, Trsm},
};
use crate::math::{LinAlgError, LinAlgResult};

/// `LU` factorization with partial pivoting (`P * A = L * U`).
///
/// `L` (unit lower triangular, implicit unit diagonal) and `U` (upper
/// triangular) are packed into a single `D x D` buffer: `L`'s strict lower
/// triangle overwrites `A`'s, `U` occupies the diagonal and upper triangle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LuDecomposition<T, const D: usize>
where
    Const<D>: Dim,
{
    data: Owned<T, D, D>,
    pivots: [usize; D],
    row_exchanges: usize,
}

/// `LDL^T` factorization for symmetric matrices (`A = L * D * L^T`), with no
/// square roots — chosen as the default symmetric solver over Cholesky for
/// that reason (`matrix-design.md` §5.6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LdltDecomposition<T, const D: usize>
where
    Const<D>: Dim,
{
    data: Owned<T, D, D>,
}

/// Cholesky factorization for symmetric *positive-definite* matrices
/// (`A = L * L^T`).
///
/// Half the arithmetic of `LU` and no pivoting, at the cost of requiring
/// square roots per diagonal entry and being restricted to positive-definite
/// input (`matrix-design.md` §5.5) — `LDL^T` is the default symmetric solver
/// for that reason; this is available when a caller already knows `A` is
/// positive definite.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CholeskyDecomposition<T, const D: usize>
where
    Const<D>: Dim,
{
    l: LowerTriangular<T, D>,
}

/// `QR` factorization via Householder reflections (`A = Q * R`), restricted
/// to square matrices.
///
/// Numerically stable even for rank-deficient or ill-conditioned `A` —
/// unlike `LU`/`LDL^T`, decomposition itself never fails
/// (`matrix-design.md` §5.5); only `solve_mut` can fail, if `R` turns out
/// singular.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QrDecomposition<T, const D: usize>
where
    Const<D>: Dim,
{
    q: Owned<T, D, D>,
    r: UpperTriangular<T, D>,
}

impl<T: Float + Copy, const D: usize> Owned<T, D, D>
where
    Const<D>: Dim,
{
    /// Performs `LU` decomposition with partial pivoting purely in-place.
    ///
    /// Overwrites `self` with the packed `L`/`U` factors and populates the
    /// caller-provided pivot scratch array with the resulting row
    /// permutation (`pivots[i]` is the original row now at position `i`).
    ///
    /// # Errors
    /// Returns [`LinAlgError::WorkspaceTooSmall`] if `pivots.len() != D`.
    /// Returns [`LinAlgError::SingularMatrix`] if a column's largest
    /// available pivot magnitude falls within `T::epsilon()` of zero.
    pub fn lu_decompose_mut(
        &mut self,
        pivots: &mut [usize],
    ) -> LinAlgResult<usize> {
        if pivots.len() != D {
            return Err(LinAlgError::WorkspaceTooSmall);
        }
        DefaultBlas::getrf(&mut self.storage, pivots)?;
        let mut row_exchanges = 0usize;
        for (k, &p) in pivots.iter().enumerate().take(D) {
            if p != k {
                row_exchanges += 1;
            }
        }
        Ok(row_exchanges)
    }

    /// Consumes the matrix to construct a stack-allocated [`LuDecomposition`].
    ///
    /// # Errors
    /// See [`Owned::lu_decompose_mut`].
    #[allow(clippy::type_complexity)]
    pub fn into_lu(mut self) -> LinAlgResult<LuDecomposition<T, D>> {
        let mut pivots = [0usize; D];
        let row_exchanges = self.lu_decompose_mut(&mut pivots)?;
        Ok(LuDecomposition {
            data: self,
            pivots,
            row_exchanges,
        })
    }

    /// Inverts a square matrix purely in-place.
    ///
    /// # Errors
    /// Returns [`LinAlgError::SingularMatrix`] if the matrix is singular.
    pub fn invert_mut(&mut self, pivots: &mut [usize]) -> LinAlgResult<()> {
        let mut factored = *self;
        let row_exchanges = factored.lu_decompose_mut(pivots)?;
        let mut local_pivots = [0usize; D];
        local_pivots.copy_from_slice(pivots);
        let lu = LuDecomposition {
            data: factored,
            pivots: local_pivots,
            row_exchanges,
        };
        let mut inv = Self::identity();
        lu.solve_mut::<D>(&mut inv)?;
        *self = inv;
        Ok(())
    }

    /// Computes the matrix inverse into a destination buffer.
    ///
    /// # Errors
    /// Returns [`LinAlgError::SingularMatrix`] if the matrix is singular.
    pub fn invert_into(
        &self,
        dest: &mut Self,
        pivots: &mut [usize],
    ) -> LinAlgResult<()> {
        let mut copy = *self;
        copy.invert_mut(pivots)?;
        *dest = copy;
        Ok(())
    }
}

impl<T: Float + Copy, const D: usize> LuDecomposition<T, D>
where
    Const<D>: Dim,
{
    /// Factorizes matrix `a` into its LU decomposition.
    pub fn decompose(a: Owned<T, D, D>) -> LinAlgResult<Self> {
        a.into_lu()
    }

    /// Computes the matrix inverse.
    ///
    /// # Errors
    /// Propagates solve failures from [`LuDecomposition::solve_mut`].
    pub fn inverse(&self) -> LinAlgResult<Owned<T, D, D>> {
        let mut inv = Owned::<T, D, D>::identity();
        self.solve_mut::<D>(&mut inv)?;
        Ok(inv)
    }

    /// Computes `det(A)` in `O(D)` time from the diagonal of `U`, applying
    /// the sign flip implied by the number of row exchanges.
    #[must_use]
    pub fn determinant(&self) -> T {
        let mut det = T::ONE;
        for i in 0..D {
            // Safety: `i < D`.
            det = det * unsafe { *self.data.storage.get_unchecked(i, i) };
        }
        if self.row_exchanges % 2 == 1 {
            det = T::ZERO - det;
        }
        det
    }

    /// Solves `A * x = b` in-place.
    ///
    /// # Errors
    /// Propagates [`LinAlgError`] from the underlying `getrs` routine
    /// (e.g. [`LinAlgError::WorkspaceTooSmall`] if the stored pivot vector is
    /// shorter than `D`).
    pub fn solve_mut<const COLS: usize>(
        &self,
        b: &mut Owned<T, D, COLS>,
    ) -> LinAlgResult<()>
    where
        Const<COLS>: Dim,
    {
        DefaultBlas::getrs(
            Trans::NoTrans,
            &self.data.storage,
            &self.pivots,
            &mut b.storage,
        )
    }
}

impl<T: Float + Copy, const D: usize> Symmetric<T, D>
where
    Const<D>: Dim,
{
    /// Performs `LDL^T` decomposition purely in-place (no pivoting — adequate
    /// for well-conditioned symmetric matrices; block pivoting for
    /// indefinite matrices is out of scope, `matrix-design.md` §5.6).
    ///
    /// # Errors
    /// Returns [`LinAlgError::SingularMatrix`] if a diagonal pivot falls
    /// within `T::epsilon()` of zero.
    pub fn ldlt_decompose_mut(&mut self) -> LinAlgResult<()> {
        let data = self.as_matrix_mut();
        for j in 0..D {
            // Safety: `j < D`.
            let mut d_j = unsafe { *data.storage.get_unchecked(j, j) };
            for k in 0..j {
                // Safety: `j < D`, `k < D`.
                unsafe {
                    let l_jk = *data.storage.get_unchecked(j, k);
                    let d_k = *data.storage.get_unchecked(k, k);
                    d_j = d_j - (l_jk * l_jk * d_k);
                }
            }
            if d_j.abs() < T::epsilon() {
                return Err(LinAlgError::SingularMatrix);
            }
            // Safety: `j < D`.
            unsafe {
                *data.storage.get_unchecked_mut(j, j) = d_j;
            }

            for i in (j + 1)..D {
                // Safety: `i < D`, `j < D`.
                let mut num = unsafe { *data.storage.get_unchecked(i, j) };
                for k in 0..j {
                    // Safety: `i < D`, `j < D`, `k < D`.
                    unsafe {
                        let l_ik = *data.storage.get_unchecked(i, k);
                        let l_jk = *data.storage.get_unchecked(j, k);
                        let d_k = *data.storage.get_unchecked(k, k);
                        num = num - (l_ik * l_jk * d_k);
                    }
                }
                // Safety: `i < D`, `j < D`.
                unsafe {
                    *data.storage.get_unchecked_mut(i, j) = num / d_j;
                }
            }
        }
        Ok(())
    }

    /// Consumes the matrix to construct a stack-allocated
    /// [`LdltDecomposition`].
    ///
    /// # Errors
    /// See [`Symmetric::ldlt_decompose_mut`].
    #[allow(clippy::type_complexity)]
    pub fn into_ldlt(mut self) -> LinAlgResult<LdltDecomposition<T, D>> {
        self.ldlt_decompose_mut()?;
        Ok(LdltDecomposition {
            data: self.into_inner(),
        })
    }
}

impl<T: Float + Copy, const D: usize> Symmetric<T, D>
where
    Const<D>: Dim,
{
    /// Performs Cholesky (`A = L * L^T`) decomposition purely in-place.
    ///
    /// # Errors
    /// Returns [`LinAlgError::NotPositiveDefinite`] if a diagonal pivot is not
    /// positive, i.e. `self` is not positive definite.
    pub fn cholesky_decompose_mut(&mut self) -> LinAlgResult<()>
    where
        T::Real: Radical,
    {
        let data = self.as_matrix_mut();
        DefaultBlas::potrf(UpLo::Lower, &mut data.storage)?;
        for i in 0..D {
            for j in (i + 1)..D {
                unsafe {
                    *data.storage.get_unchecked_mut(i, j) = T::ZERO;
                }
            }
        }
        Ok(())
    }

    /// Consumes the matrix to construct a stack-allocated [`CholeskyDecomposition`].
    ///
    /// # Errors
    /// See [`Symmetric::cholesky_decompose_mut`].
    #[allow(clippy::type_complexity)]
    pub fn into_cholesky(mut self) -> LinAlgResult<CholeskyDecomposition<T, D>>
    where
        T::Real: Radical,
    {
        self.cholesky_decompose_mut()?;
        Ok(CholeskyDecomposition {
            l: LowerTriangular::from_owned_unchecked(self.into_inner()),
        })
    }
}

impl<T: Float + Copy, const D: usize> CholeskyDecomposition<T, D>
where
    Const<D>: Dim,
{
    /// Solves `A * x = b` in-place.
    pub fn solve_mut<const COLS: usize>(
        &self,
        b: &mut Owned<T, D, COLS>,
    ) -> LinAlgResult<()>
    where
        Const<COLS>: Dim,
    {
        DefaultBlas::potrs(
            UpLo::Lower,
            &self.l.as_matrix().storage,
            &mut b.storage,
        )
    }
}

impl<T: Float + Copy, const D: usize> Owned<T, D, D>
where
    Const<D>: Dim,
{
    /// Performs `QR` decomposition (`A = Q * R`) via Householder
    /// reflections.
    pub fn qr_decompose_mut(&mut self, q: &mut Self)
    where
        T::Real: Radical,
    {
        *q = Self::identity();
        let mut tau = [T::ZERO; D];
        let mut work = [T::ZERO; D];
        let _ = DefaultBlas::geqrf(&mut self.storage, &mut tau, &mut work);
        let _ = DefaultBlas::ormqr(
            Side::Left,
            Trans::NoTrans,
            &self.storage,
            &tau,
            &mut q.storage,
            &mut work,
        );

        // Force exact zeros below the diagonal: reflections leave only
        // numerical noise there, and `UpperTriangular` callers must be able
        // to trust the packed zero region.
        for i in 0..D {
            for j in 0..i {
                unsafe {
                    *self.storage.get_unchecked_mut(i, j) = T::ZERO;
                }
            }
        }
    }

    /// Consumes the matrix to construct a stack-allocated
    /// [`QrDecomposition`].
    #[allow(clippy::type_complexity)]
    pub fn into_qr(mut self) -> QrDecomposition<T, D>
    where
        T::Real: Radical,
    {
        let mut q = Self::zero();
        self.qr_decompose_mut(&mut q);
        QrDecomposition {
            q,
            r: UpperTriangular::from_owned_unchecked(self),
        }
    }
}

impl<T: Float + Copy, const D: usize> QrDecomposition<T, D>
where
    Const<D>: Dim,
{
    /// Solves `A * x = b` in-place.
    ///
    /// # Errors
    /// Returns [`LinAlgError::SingularMatrix`] if any diagonal entry of `R`
    /// is within `T::epsilon()` of zero (`A` is rank-deficient).
    pub fn solve_mut<const COLS: usize>(
        &self,
        b: &mut Owned<T, D, COLS>,
    ) -> LinAlgResult<()>
    where
        Const<COLS>: Dim,
    {
        let r = self.r.as_matrix();
        let tol = T::epsilon() * T::from_usize(128);
        for i in 0..D {
            let r_ii = unsafe { *r.storage.get_unchecked(i, i) };
            if r_ii.abs() <= tol {
                return Err(LinAlgError::SingularMatrix);
            }
        }

        let mut y = Owned::<T, D, COLS>::zero();
        DefaultBlas::gemm(
            Trans::Trans,
            Trans::NoTrans,
            T::ONE,
            &self.q.storage,
            &b.storage,
            T::ZERO,
            &mut y.storage,
        );

        DefaultBlas::trsm(
            Side::Left,
            UpLo::Upper,
            Trans::NoTrans,
            Diag::NonUnit,
            T::ONE,
            &self.r.as_matrix().storage,
            &mut y.storage,
        )?;

        *b = y;
        Ok(())
    }
}

impl<T: Float + Copy, const D: usize> LdltDecomposition<T, D>
where
    Const<D>: Dim,
{
    /// Computes `det(A)` in `O(D)` time as the product of `D`'s diagonal.
    #[must_use]
    pub fn determinant(&self) -> T {
        let mut det = T::ONE;
        for i in 0..D {
            // Safety: `i < D`.
            det = det * unsafe { *self.data.storage.get_unchecked(i, i) };
        }
        det
    }

    /// Solves `A * x = b` in-place.
    pub fn solve_mut<const COLS: usize>(
        &self,
        b: &mut Owned<T, D, COLS>,
    ) -> LinAlgResult<()>
    where
        Const<COLS>: Dim,
    {
        DefaultBlas::trsm(
            Side::Left,
            UpLo::Lower,
            Trans::NoTrans,
            Diag::Unit,
            T::ONE,
            &self.data.storage,
            &mut b.storage,
        )?;

        for i in 0..D {
            let d_i = unsafe { *self.data.storage.get_unchecked(i, i) };
            if d_i.abs() < T::epsilon() {
                return Err(LinAlgError::SingularMatrix);
            }
            for c in 0..COLS {
                unsafe {
                    let z = *b.storage.get_unchecked(i, c);
                    *b.storage.get_unchecked_mut(i, c) = z / d_i;
                }
            }
        }

        DefaultBlas::trsm(
            Side::Left,
            UpLo::Lower,
            Trans::Trans,
            Diag::Unit,
            T::ONE,
            &self.data.storage,
            &mut b.storage,
        )
    }
}

/// Rectangular Householder $QR$ ($R \ge C$): $A = Q R$ with $Q \in \mathbb{R}^{R \times R}$
/// and upper-trapezoidal $R \in \mathbb{R}^{R \times C}$.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QrDecompositionRect<T, const R: usize, const C: usize>
where
    Const<R>: Dim,
    Const<C>: Dim,
{
    /// Orthogonal factor.
    pub q: Owned<T, R, R>,
    /// Upper-trapezoidal factor.
    pub r: Owned<T, R, C>,
}

impl<T: Float + Copy, const R: usize, const C: usize> Owned<T, R, C>
where
    Const<R>: Dim,
    Const<C>: Dim,
{
    /// Rectangular $QR$ via Householder reflections. Requires $R \ge C$.
    ///
    /// # Errors
    /// Returns [`LinAlgError::WorkspaceTooSmall`] if $R < C$.
    pub fn into_qr_rect(mut self) -> LinAlgResult<QrDecompositionRect<T, R, C>>
    where
        T::Real: Radical,
    {
        if R < C {
            return Err(LinAlgError::WorkspaceTooSmall);
        }
        let mut tau = [T::ZERO; C];
        let mut work = [T::ZERO; R];
        DefaultBlas::geqrf(&mut self.storage, &mut tau, &mut work)?;
        let mut q = Owned::<T, R, R>::identity();
        DefaultBlas::ormqr(
            Side::Left,
            Trans::NoTrans,
            &self.storage,
            &tau,
            &mut q.storage,
            &mut work,
        )?;
        for i in 0..R {
            for j in 0..C {
                if i > j {
                    if let Some(elem) = self.get_mut(i, j) {
                        *elem = T::ZERO;
                    }
                }
            }
        }
        Ok(QrDecompositionRect { q, r: self })
    }
}
