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
    ArrayStorage, DenseStorage, DenseStorageMut, Diag, Side, Trans, UpLo,
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
    /// Performs `LU` decomposition with partial pivoting purely in-place using a specific BLAS engine.
    ///
    /// Overwrites `self` with the packed `L`/`U` factors and populates the
    /// caller-provided pivot scratch array with the resulting row
    /// permutation (`pivots[i]` is the original row now at position `i`).
    ///
    /// # Errors
    /// Returns [`LinAlgError::WorkspaceTooSmall`] if `pivots.len() != D`.
    /// Returns [`LinAlgError::SingularMatrix`] if a column's largest
    /// available pivot magnitude falls within `T::epsilon()` of zero.
    pub fn lu_decompose_mut_with<B: Getrf<T, ArrayStorage<T, D, D>>>(
        &mut self,
        pivots: &mut [usize],
    ) -> LinAlgResult<usize> {
        if pivots.len() != D {
            return Err(LinAlgError::WorkspaceTooSmall);
        }
        B::getrf(&mut self.storage, pivots)?;
        let mut row_exchanges = 0usize;
        for (k, &p) in pivots.iter().enumerate().take(D) {
            if p != k {
                row_exchanges += 1;
            }
        }
        Ok(row_exchanges)
    }

    /// Performs `LU` decomposition with partial pivoting purely in-place using the default BLAS engine.
    pub fn lu_decompose_mut(
        &mut self,
        pivots: &mut [usize],
    ) -> LinAlgResult<usize> {
        self.lu_decompose_mut_with::<DefaultBlas>(pivots)
    }

    /// Consumes the matrix to construct a stack-allocated [`LuDecomposition`] using a specific BLAS engine.
    ///
    /// # Errors
    /// See [`Owned::lu_decompose_mut`].
    #[allow(clippy::type_complexity)]
    pub fn into_lu_with<B: Getrf<T, ArrayStorage<T, D, D>>>(
        mut self,
    ) -> LinAlgResult<LuDecomposition<T, D>> {
        let mut pivots = [0usize; D];
        let row_exchanges = self.lu_decompose_mut_with::<B>(&mut pivots)?;
        Ok(LuDecomposition {
            data: self,
            pivots,
            row_exchanges,
        })
    }

    /// Consumes the matrix to construct a stack-allocated [`LuDecomposition`] using the default BLAS engine.
    ///
    /// # Errors
    /// See [`Owned::lu_decompose_mut`].
    #[allow(clippy::type_complexity)]
    pub fn into_lu(self) -> LinAlgResult<LuDecomposition<T, D>> {
        self.into_lu_with::<DefaultBlas>()
    }

    /// Inverts a square matrix purely in-place using a specific BLAS engine.
    ///
    /// # Errors
    /// Returns [`LinAlgError::SingularMatrix`] if the matrix is singular.
    pub fn invert_mut_with<
        B: Getrf<T, ArrayStorage<T, D, D>>
            + Getrs<T, ArrayStorage<T, D, D>, ArrayStorage<T, D, D>>,
    >(
        &mut self,
        pivots: &mut [usize],
    ) -> LinAlgResult<()> {
        let mut factored = *self;
        let row_exchanges = factored.lu_decompose_mut_with::<B>(pivots)?;
        let mut local_pivots = [0usize; D];
        local_pivots.copy_from_slice(pivots);
        let lu = LuDecomposition {
            data: factored,
            pivots: local_pivots,
            row_exchanges,
        };
        let mut inv = Self::identity();
        lu.solve_mut_with::<B, D>(&mut inv)?;
        *self = inv;
        Ok(())
    }

    /// Inverts a square matrix purely in-place using the default BLAS engine.
    pub fn invert_mut(&mut self, pivots: &mut [usize]) -> LinAlgResult<()> {
        self.invert_mut_with::<DefaultBlas>(pivots)
    }

    /// Computes the matrix inverse into a destination buffer using a specific BLAS engine.
    ///
    /// # Errors
    /// Returns [`LinAlgError::SingularMatrix`] if the matrix is singular.
    pub fn invert_into_with<
        B: Getrf<T, ArrayStorage<T, D, D>>
            + Getrs<T, ArrayStorage<T, D, D>, ArrayStorage<T, D, D>>,
    >(
        &self,
        dest: &mut Self,
        pivots: &mut [usize],
    ) -> LinAlgResult<()> {
        let mut copy = *self;
        copy.invert_mut_with::<B>(pivots)?;
        *dest = copy;
        Ok(())
    }

    /// Computes the matrix inverse into a destination buffer using the default BLAS engine.
    pub fn invert_into(
        &self,
        dest: &mut Self,
        pivots: &mut [usize],
    ) -> LinAlgResult<()> {
        self.invert_into_with::<DefaultBlas>(dest, pivots)
    }
}

impl<T: Float + Copy, const D: usize> LuDecomposition<T, D>
where
    Const<D>: Dim,
{
    /// Factorizes matrix `a` into its LU decomposition using a specific BLAS engine.
    pub fn decompose_with<B: Getrf<T, ArrayStorage<T, D, D>>>(
        a: Owned<T, D, D>,
    ) -> LinAlgResult<Self> {
        a.into_lu_with::<B>()
    }

    /// Factorizes matrix `a` into its LU decomposition using the default BLAS engine.
    pub fn decompose(a: Owned<T, D, D>) -> LinAlgResult<Self> {
        a.into_lu()
    }

    /// Computes the matrix inverse using a specific BLAS engine.
    ///
    /// # Errors
    /// Propagates solve failures from [`LuDecomposition::solve_mut_with`].
    pub fn inverse_with<
        B: Getrs<T, ArrayStorage<T, D, D>, ArrayStorage<T, D, D>>,
    >(
        &self,
    ) -> LinAlgResult<Owned<T, D, D>> {
        let mut inv = Owned::<T, D, D>::identity();
        self.solve_mut_with::<B, D>(&mut inv)?;
        Ok(inv)
    }

    /// Computes the matrix inverse using the default BLAS engine.
    ///
    /// # Errors
    /// See [`LuDecomposition::inverse_with`].
    pub fn inverse(&self) -> LinAlgResult<Owned<T, D, D>> {
        self.inverse_with::<DefaultBlas>()
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

    /// Solves `A * x = b` in-place using a specific BLAS engine.
    ///
    /// # Errors
    /// Propagates [`LinAlgError`] from the underlying `getrs` routine
    /// (e.g. [`LinAlgError::WorkspaceTooSmall`] if the stored pivot vector is
    /// shorter than `D`).
    pub fn solve_mut_with<
        B: Getrs<T, ArrayStorage<T, D, D>, ArrayStorage<T, D, COLS>>,
        const COLS: usize,
    >(
        &self,
        b: &mut Owned<T, D, COLS>,
    ) -> LinAlgResult<()>
    where
        Const<COLS>: Dim,
    {
        B::getrs(
            Trans::NoTrans,
            &self.data.storage,
            &self.pivots,
            &mut b.storage,
        )
    }

    /// Solves `A * x = b` in-place using the default BLAS engine.
    ///
    /// # Errors
    /// See [`LuDecomposition::solve_mut_with`].
    pub fn solve_mut<const COLS: usize>(
        &self,
        b: &mut Owned<T, D, COLS>,
    ) -> LinAlgResult<()>
    where
        Const<COLS>: Dim,
    {
        self.solve_mut_with::<DefaultBlas, COLS>(b)
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
    /// Performs Cholesky (`A = L * L^T`) decomposition purely in-place using a specific BLAS engine.
    ///
    /// # Errors
    /// Returns [`LinAlgError::NotPositiveDefinite`] if a diagonal pivot is not
    /// positive, i.e. `self` is not positive definite.
    pub fn cholesky_decompose_mut_with<B: Potrf<T, ArrayStorage<T, D, D>>>(
        &mut self,
    ) -> LinAlgResult<()>
    where
        T::Real: Radical,
    {
        let data = self.as_matrix_mut();
        B::potrf(UpLo::Lower, &mut data.storage)?;
        for i in 0..D {
            for j in (i + 1)..D {
                unsafe {
                    *data.storage.get_unchecked_mut(i, j) = T::ZERO;
                }
            }
        }
        Ok(())
    }

    /// Performs Cholesky (`A = L * L^T`) decomposition purely in-place using the default BLAS engine.
    pub fn cholesky_decompose_mut(&mut self) -> LinAlgResult<()>
    where
        T::Real: Radical,
    {
        self.cholesky_decompose_mut_with::<DefaultBlas>()
    }

    /// Consumes the matrix to construct a stack-allocated [`CholeskyDecomposition`] using a specific BLAS engine.
    ///
    /// # Errors
    /// See [`Symmetric::cholesky_decompose_mut_with`].
    #[allow(clippy::type_complexity)]
    pub fn into_cholesky_with<B: Potrf<T, ArrayStorage<T, D, D>>>(
        mut self,
    ) -> LinAlgResult<CholeskyDecomposition<T, D>>
    where
        T::Real: Radical,
    {
        self.cholesky_decompose_mut_with::<B>()?;
        Ok(CholeskyDecomposition {
            l: LowerTriangular::from_owned_unchecked(self.into_inner()),
        })
    }

    /// Consumes the matrix to construct a stack-allocated [`CholeskyDecomposition`] using the default BLAS engine.
    ///
    /// # Errors
    /// See [`Symmetric::cholesky_decompose_mut`].
    #[allow(clippy::type_complexity)]
    pub fn into_cholesky(self) -> LinAlgResult<CholeskyDecomposition<T, D>>
    where
        T::Real: Radical,
    {
        self.into_cholesky_with::<DefaultBlas>()
    }
}

impl<T: Float + Copy, const D: usize> CholeskyDecomposition<T, D>
where
    Const<D>: Dim,
{
    /// Solves `A * x = b` in-place using a specific BLAS engine.
    pub fn solve_mut_with<
        B: Potrs<T, ArrayStorage<T, D, D>, ArrayStorage<T, D, COLS>>,
        const COLS: usize,
    >(
        &self,
        b: &mut Owned<T, D, COLS>,
    ) where
        Const<COLS>: Dim,
    {
        let _ =
            B::potrs(UpLo::Lower, &self.l.as_matrix().storage, &mut b.storage);
    }

    /// Solves `A * x = b` in-place using the default BLAS engine.
    pub fn solve_mut<const COLS: usize>(&self, b: &mut Owned<T, D, COLS>)
    where
        Const<COLS>: Dim,
    {
        self.solve_mut_with::<DefaultBlas, COLS>(b);
    }
}

impl<T: Float + Copy, const D: usize> Owned<T, D, D>
where
    Const<D>: Dim,
{
    /// Performs `QR` decomposition (`A = Q * R`) via Householder
    /// reflections using a specific BLAS engine.
    pub fn qr_decompose_mut_with<
        B: Geqrf<T, ArrayStorage<T, D, D>>
            + Ormqr<T, ArrayStorage<T, D, D>, ArrayStorage<T, D, D>>,
    >(
        &mut self,
        q: &mut Self,
    ) where
        T::Real: Radical,
    {
        *q = Self::identity();
        let mut tau = [T::ZERO; D];
        let mut work = [T::ZERO; D];
        let _ = B::geqrf(&mut self.storage, &mut tau, &mut work);
        let _ = B::ormqr(
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

    /// Performs `QR` decomposition (`A = Q * R`) via Householder
    /// reflections using the default BLAS engine.
    pub fn qr_decompose_mut(&mut self, q: &mut Self)
    where
        T::Real: Radical,
    {
        self.qr_decompose_mut_with::<DefaultBlas>(q);
    }

    /// Consumes the matrix to construct a stack-allocated
    /// [`QrDecomposition`] using a specific BLAS engine.
    #[allow(clippy::type_complexity)]
    pub fn into_qr_with<
        B: Geqrf<T, ArrayStorage<T, D, D>>
            + Ormqr<T, ArrayStorage<T, D, D>, ArrayStorage<T, D, D>>,
    >(
        mut self,
    ) -> QrDecomposition<T, D>
    where
        T::Real: Radical,
    {
        let mut q = Self::zero();
        self.qr_decompose_mut_with::<B>(&mut q);
        QrDecomposition {
            q,
            r: UpperTriangular::from_owned_unchecked(self),
        }
    }

    /// Consumes the matrix to construct a stack-allocated
    /// [`QrDecomposition`] using the default BLAS engine.
    #[allow(clippy::type_complexity)]
    pub fn into_qr(self) -> QrDecomposition<T, D>
    where
        T::Real: Radical,
    {
        self.into_qr_with::<DefaultBlas>()
    }
}

impl<T: Float + Copy, const D: usize> QrDecomposition<T, D>
where
    Const<D>: Dim,
{
    /// Solves `A * x = b` in-place using a specific BLAS engine.
    ///
    /// # Errors
    /// Returns [`LinAlgError::SingularMatrix`] if any diagonal entry of `R`
    /// is within `T::epsilon()` of zero (`A` is rank-deficient).
    pub fn solve_mut_with<B, const COLS: usize>(
        &self,
        b: &mut Owned<T, D, COLS>,
    ) -> LinAlgResult<()>
    where
        B: Gemm<
                T,
                ArrayStorage<T, D, D>,
                ArrayStorage<T, D, COLS>,
                ArrayStorage<T, D, COLS>,
            > + Trsm<T, ArrayStorage<T, D, D>, ArrayStorage<T, D, COLS>>,
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
        B::gemm(
            Trans::Trans,
            Trans::NoTrans,
            T::ONE,
            &self.q.storage,
            &b.storage,
            T::ZERO,
            &mut y.storage,
        );

        B::trsm(
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

    /// Solves `A * x = b` in-place using the default BLAS engine.
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
        self.solve_mut_with::<DefaultBlas, COLS>(b)
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

    /// Solves `A * x = b` in-place using a specific BLAS engine.
    pub fn solve_mut_with<
        B: Trsm<T, ArrayStorage<T, D, D>, ArrayStorage<T, D, COLS>>,
        const COLS: usize,
    >(
        &self,
        b: &mut Owned<T, D, COLS>,
    ) where
        Const<COLS>: Dim,
    {
        let _ = B::trsm(
            Side::Left,
            UpLo::Lower,
            Trans::NoTrans,
            Diag::Unit,
            T::ONE,
            &self.data.storage,
            &mut b.storage,
        );

        for i in 0..D {
            let d_i = unsafe { *self.data.storage.get_unchecked(i, i) };
            if d_i.abs() >= T::epsilon() {
                for c in 0..COLS {
                    unsafe {
                        let z = *b.storage.get_unchecked(i, c);
                        *b.storage.get_unchecked_mut(i, c) = z / d_i;
                    }
                }
            }
        }

        let _ = B::trsm(
            Side::Left,
            UpLo::Lower,
            Trans::Trans,
            Diag::Unit,
            T::ONE,
            &self.data.storage,
            &mut b.storage,
        );
    }

    /// Solves `A * x = b` in-place using the default BLAS engine.
    pub fn solve_mut<const COLS: usize>(&self, b: &mut Owned<T, D, COLS>)
    where
        Const<COLS>: Dim,
    {
        self.solve_mut_with::<DefaultBlas, COLS>(b);
    }
}
