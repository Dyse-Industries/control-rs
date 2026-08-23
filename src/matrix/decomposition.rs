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
    clippy::needless_range_loop
)]

use super::{LowerTriangular, Owned, Symmetric, UpperTriangular};
use crate::math::num_traits::Float;
use crate::math::num_types::{Const, Dim};
use crate::math::storage::{DenseStorage, DenseStorageMut};
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
    /// Returns [`LinAlgError::SingularMatrix`] if a column's largest
    /// available pivot magnitude falls within `T::epsilon()` of zero.
    ///
    /// # Panics
    /// Panics if `pivots.len() != D`.
    pub fn lu_decompose_mut(
        &mut self,
        pivots: &mut [usize],
    ) -> LinAlgResult<usize> {
        assert_eq!(pivots.len(), D, "Pivot scratch buffer size mismatch");
        for (idx, p) in pivots.iter_mut().enumerate() {
            *p = idx;
        }

        let mut row_exchanges = 0usize;
        for k in 0..D {
            let mut pivot_row = k;
            // Safety: `k < D`.
            let mut max_val = unsafe { self.storage.get_unchecked(k, k) }.abs();
            for i in (k + 1)..D {
                // Safety: `i < D`, `k < D`.
                let val = unsafe { self.storage.get_unchecked(i, k) }.abs();
                if val > max_val {
                    max_val = val;
                    pivot_row = i;
                }
            }
            if max_val < T::epsilon() {
                return Err(LinAlgError::SingularMatrix);
            }

            if pivot_row != k {
                for col in 0..D {
                    // Safety: `k < D`, `pivot_row < D`, `col < D`.
                    unsafe {
                        let a = *self.storage.get_unchecked(k, col);
                        let b = *self.storage.get_unchecked(pivot_row, col);
                        *self.storage.get_unchecked_mut(k, col) = b;
                        *self.storage.get_unchecked_mut(pivot_row, col) = a;
                    }
                }
                pivots.swap(k, pivot_row);
                row_exchanges += 1;
            }

            for i in (k + 1)..D {
                // Safety: `i < D`, `k < D`.
                let factor = unsafe {
                    *self.storage.get_unchecked(i, k)
                        / *self.storage.get_unchecked(k, k)
                };
                // Safety: `i < D`, `k < D`.
                unsafe {
                    *self.storage.get_unchecked_mut(i, k) = factor;
                }
                for j in (k + 1)..D {
                    // Safety: `i < D`, `j < D`, `k < D`.
                    unsafe {
                        let ukj = *self.storage.get_unchecked(k, j);
                        let aij = *self.storage.get_unchecked(i, j);
                        *self.storage.get_unchecked_mut(i, j) =
                            aij - (factor * ukj);
                    }
                }
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

    /// Inverts a square matrix purely in-place using caller-provided pivot
    /// scratch space.
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
        lu.solve_mut(&mut inv);
        *self = inv;
        Ok(())
    }

    /// Computes the matrix inverse into a caller-provided destination
    /// buffer, without mutating `self`.
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

    /// Solves `A * x = b` in-place by mutating each column of `b` into the
    /// corresponding solution column, via row permutation, forward
    /// substitution (`L`) and back substitution (`U`).
    pub fn solve_mut<const COLS: usize>(&self, b: &mut Owned<T, D, COLS>)
    where
        Const<COLS>: Dim,
    {
        for c in 0..COLS {
            let mut col = [T::ZERO; D];
            for i in 0..D {
                // Safety: `i < D`, `c < COLS`.
                col[i] = unsafe { *b.storage.get_unchecked(i, c) };
            }

            let mut y = [T::ZERO; D];
            for i in 0..D {
                y[i] = col[self.pivots[i]];
            }

            // Forward substitution: L * z = y (L unit lower triangular).
            for i in 0..D {
                let mut sum = y[i];
                for k in 0..i {
                    // Safety: `i < D`, `k < D`.
                    let l_ik =
                        unsafe { *self.data.storage.get_unchecked(i, k) };
                    sum = sum - (l_ik * y[k]);
                }
                y[i] = sum;
            }

            // Back substitution: U * x = z.
            for step in 0..D {
                let i = D - 1 - step;
                let mut sum = y[i];
                for k in (i + 1)..D {
                    // Safety: `i < D`, `k < D`.
                    let u_ik =
                        unsafe { *self.data.storage.get_unchecked(i, k) };
                    sum = sum - (u_ik * y[k]);
                }
                // Safety: `i < D`.
                let u_ii = unsafe { *self.data.storage.get_unchecked(i, i) };
                y[i] = sum / u_ii;
            }

            for i in 0..D {
                // Safety: `i < D`, `c < COLS`.
                unsafe {
                    *b.storage.get_unchecked_mut(i, c) = y[i];
                }
            }
        }
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
    /// Performs Cholesky (`A = L * L^T`) decomposition purely in-place on
    /// the strict lower triangle and diagonal (no pivoting — restricted to
    /// positive-definite matrices). Like `ldlt_decompose_mut`, the strict
    /// upper triangle is left untouched (stale): every consumer of the
    /// result only reads `i >= j` entries.
    ///
    /// # Errors
    /// Returns [`LinAlgError::SingularMatrix`] if a diagonal pivot is not
    /// positive (within `T::epsilon()`), i.e. `self` is not positive
    /// definite.
    pub fn cholesky_decompose_mut(&mut self) -> LinAlgResult<()> {
        let data = self.as_matrix_mut();
        for j in 0..D {
            // Safety: `j < D`.
            let mut sum = unsafe { *data.storage.get_unchecked(j, j) };
            for k in 0..j {
                // Safety: `j < D`, `k < D`.
                let l_jk = unsafe { *data.storage.get_unchecked(j, k) };
                sum = sum - (l_jk * l_jk);
            }
            if sum <= T::epsilon() {
                return Err(LinAlgError::SingularMatrix);
            }
            let l_jj = sum.sqrt();
            // Safety: `j < D`.
            unsafe {
                *data.storage.get_unchecked_mut(j, j) = l_jj;
            }

            for i in (j + 1)..D {
                // Safety: `i < D`, `j < D`.
                let mut num = unsafe { *data.storage.get_unchecked(i, j) };
                for k in 0..j {
                    // Safety: `i < D`, `j < D`, `k < D`.
                    unsafe {
                        let l_ik = *data.storage.get_unchecked(i, k);
                        let l_jk = *data.storage.get_unchecked(j, k);
                        num = num - (l_ik * l_jk);
                    }
                }
                // Safety: `i < D`, `j < D`.
                unsafe {
                    *data.storage.get_unchecked_mut(i, j) = num / l_jj;
                }
            }
        }
        Ok(())
    }

    /// Consumes the matrix to construct a stack-allocated
    /// [`CholeskyDecomposition`].
    ///
    /// # Errors
    /// See [`Symmetric::cholesky_decompose_mut`].
    #[allow(clippy::type_complexity)]
    pub fn into_cholesky(
        mut self,
    ) -> LinAlgResult<CholeskyDecomposition<T, D>> {
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
    /// Solves `A * x = b` in-place: forward substitution (`L`), back
    /// substitution (`L^T`).
    pub fn solve_mut<const COLS: usize>(&self, b: &mut Owned<T, D, COLS>)
    where
        Const<COLS>: Dim,
    {
        let l = self.l.as_matrix();
        for c in 0..COLS {
            let mut y = [T::ZERO; D];
            for i in 0..D {
                // Safety: `i < D`, `c < COLS`.
                y[i] = unsafe { *b.storage.get_unchecked(i, c) };
            }

            // Forward substitution: L * z = y.
            for i in 0..D {
                let mut sum = y[i];
                for k in 0..i {
                    // Safety: `i < D`, `k < D`.
                    let l_ik = unsafe { *l.storage.get_unchecked(i, k) };
                    sum = sum - (l_ik * y[k]);
                }
                // Safety: `i < D`.
                let l_ii = unsafe { *l.storage.get_unchecked(i, i) };
                y[i] = sum / l_ii;
            }

            // Back substitution: L^T * x = z.
            for step in 0..D {
                let i = D - 1 - step;
                let mut sum = y[i];
                for k in (i + 1)..D {
                    // Safety: `k < D`, `i < D` — reads `L[k][i]`, which is
                    // `L^T[i][k]`.
                    let l_ki = unsafe { *l.storage.get_unchecked(k, i) };
                    sum = sum - (l_ki * y[k]);
                }
                // Safety: `i < D`.
                let l_ii = unsafe { *l.storage.get_unchecked(i, i) };
                y[i] = sum / l_ii;
            }

            for i in 0..D {
                // Safety: `i < D`, `c < COLS`.
                unsafe {
                    *b.storage.get_unchecked_mut(i, c) = y[i];
                }
            }
        }
    }
}

/// Left-multiplies `m` by the Householder reflector
/// `H = I - 2*v*v^T / norm_v_sq` (rows/columns `k..D`, matching `v`'s
/// nonzero range): `m := H * m`. Shared by `qr_decompose_mut`'s
/// `R := H * R` step.
fn reflect_rows<T: Float + Copy, const D: usize>(
    m: &mut Owned<T, D, D>,
    v: &[T; D],
    norm_v_sq: T,
    k: usize,
) where
    Const<D>: Dim,
{
    for j in k..D {
        let mut dot = T::ZERO;
        for i in k..D {
            // Safety: `i < D`, `j < D`.
            let m_ij = unsafe { *m.storage.get_unchecked(i, j) };
            dot = dot + (v[i] * m_ij);
        }
        let factor = (dot + dot) / norm_v_sq;
        for i in k..D {
            // Safety: `i < D`, `j < D`.
            unsafe {
                let m_ij = *m.storage.get_unchecked(i, j);
                *m.storage.get_unchecked_mut(i, j) = m_ij - (factor * v[i]);
            }
        }
    }
}

/// Right-multiplies `m` by the Householder reflector
/// `H = I - 2*v*v^T / norm_v_sq` (columns `k..D`, all rows): `m := m * H`.
/// Shared by `qr_decompose_mut`'s `Q := Q * H` step.
fn reflect_cols<T: Float + Copy, const D: usize>(
    m: &mut Owned<T, D, D>,
    v: &[T; D],
    norm_v_sq: T,
    k: usize,
) where
    Const<D>: Dim,
{
    for i in 0..D {
        let mut dot = T::ZERO;
        for j in k..D {
            // Safety: `i < D`, `j < D`.
            let m_ij = unsafe { *m.storage.get_unchecked(i, j) };
            dot = dot + (m_ij * v[j]);
        }
        let factor = (dot + dot) / norm_v_sq;
        for j in k..D {
            // Safety: `i < D`, `j < D`.
            unsafe {
                let m_ij = *m.storage.get_unchecked(i, j);
                *m.storage.get_unchecked_mut(i, j) = m_ij - (factor * v[j]);
            }
        }
    }
}

impl<T: Float + Copy, const D: usize> Owned<T, D, D>
where
    Const<D>: Dim,
{
    /// Performs `QR` decomposition (`A = Q * R`) via Householder
    /// reflections, purely in-place: overwrites `self` with the
    /// upper-triangular `R` factor and writes the orthogonal `Q` factor
    /// into the caller-provided `q` destination buffer (`q`'s initial
    /// contents are discarded).
    ///
    /// Unlike `lu_decompose_mut`/`ldlt_decompose_mut`, this never fails:
    /// Householder `QR` is defined (and numerically stable) even for
    /// rank-deficient or ill-conditioned `A` (`matrix-design.md` §5.5) — a
    /// near-zero pivot simply leaves the corresponding column of `R`
    /// already triangular rather than requiring a reflection.
    pub fn qr_decompose_mut(&mut self, q: &mut Self) {
        *q = Self::identity();
        for k in 0..D.saturating_sub(1) {
            // Norm of R's k-th column, rows k..D.
            let mut norm_x = T::ZERO;
            for i in k..D {
                // Safety: `i < D`, `k < D`.
                let r_ik = unsafe { *self.storage.get_unchecked(i, k) };
                norm_x = norm_x + (r_ik * r_ik);
            }
            let norm_x = norm_x.sqrt();
            if norm_x <= T::epsilon() {
                continue; // Column already zero below the diagonal.
            }

            // Safety: `k < D`.
            let r_kk = unsafe { *self.storage.get_unchecked(k, k) };
            let alpha = if r_kk.is_sign_negative() {
                norm_x
            } else {
                T::ZERO - norm_x
            };

            let mut v = [T::ZERO; D];
            for i in k..D {
                // Safety: `i < D`, `k < D`.
                v[i] = unsafe { *self.storage.get_unchecked(i, k) };
            }
            v[k] = v[k] - alpha;

            let mut norm_v_sq = T::ZERO;
            for vi in v.iter().skip(k) {
                norm_v_sq = norm_v_sq + (*vi * *vi);
            }
            if norm_v_sq <= T::epsilon() {
                continue; // Reflector would be the identity.
            }

            reflect_rows(self, &v, norm_v_sq, k); // R := H * R
            reflect_cols(q, &v, norm_v_sq, k); // Q := Q * H
        }

        // Force exact zeros below the diagonal: reflections leave only
        // numerical noise there, and `UpperTriangular` callers must be able
        // to trust the packed zero region.
        for i in 0..D {
            for j in 0..i {
                // Safety: `i < D`, `j < D`.
                unsafe {
                    *self.storage.get_unchecked_mut(i, j) = T::ZERO;
                }
            }
        }
    }

    /// Consumes the matrix to construct a stack-allocated
    /// [`QrDecomposition`].
    #[allow(clippy::type_complexity)]
    pub fn into_qr(mut self) -> QrDecomposition<T, D> {
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
    /// Solves `A * x = b` in-place: `y = Q^T * b`, then back substitution
    /// `R * x = y`.
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
        for i in 0..D {
            // Safety: `i < D`.
            let r_ii = unsafe { *r.storage.get_unchecked(i, i) };
            if r_ii.abs() < T::epsilon() {
                return Err(LinAlgError::SingularMatrix);
            }
        }

        for c in 0..COLS {
            // y = Q^T * b (column c).
            let mut y = [T::ZERO; D];
            for i in 0..D {
                let mut sum = T::ZERO;
                for k in 0..D {
                    // Safety: `k < D`, `i < D` — reads `Q[k][i]`, which is
                    // `Q^T[i][k]`.
                    let q_ki = unsafe { *self.q.storage.get_unchecked(k, i) };
                    // Safety: `k < D`, `c < COLS`.
                    let b_kc = unsafe { *b.storage.get_unchecked(k, c) };
                    sum = sum + (q_ki * b_kc);
                }
                y[i] = sum;
            }

            // Back substitution: R * x = y.
            for step in 0..D {
                let i = D - 1 - step;
                let mut sum = y[i];
                for k in (i + 1)..D {
                    // Safety: `i < D`, `k < D`.
                    let r_ik = unsafe { *r.storage.get_unchecked(i, k) };
                    sum = sum - (r_ik * y[k]);
                }
                // Safety: `i < D`.
                let r_ii = unsafe { *r.storage.get_unchecked(i, i) };
                y[i] = sum / r_ii;
            }

            for i in 0..D {
                // Safety: `i < D`, `c < COLS`.
                unsafe {
                    *b.storage.get_unchecked_mut(i, c) = y[i];
                }
            }
        }
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

    /// Solves `A * x = b` in-place: forward substitution (`L`), diagonal
    /// scaling (`D`), back substitution (`L^T`).
    pub fn solve_mut<const COLS: usize>(&self, b: &mut Owned<T, D, COLS>)
    where
        Const<COLS>: Dim,
    {
        for c in 0..COLS {
            let mut y = [T::ZERO; D];
            for i in 0..D {
                // Safety: `i < D`, `c < COLS`.
                y[i] = unsafe { *b.storage.get_unchecked(i, c) };
            }

            for i in 0..D {
                let mut sum = y[i];
                for k in 0..i {
                    // Safety: `i < D`, `k < D`.
                    let l_ik =
                        unsafe { *self.data.storage.get_unchecked(i, k) };
                    sum = sum - (l_ik * y[k]);
                }
                y[i] = sum;
            }

            for i in 0..D {
                // Safety: `i < D`.
                let d_i = unsafe { *self.data.storage.get_unchecked(i, i) };
                y[i] = y[i] / d_i;
            }

            for step in 0..D {
                let i = D - 1 - step;
                let mut sum = y[i];
                for k in (i + 1)..D {
                    // Safety: `k < D`, `i < D` — reads `L[k][i]`, which is
                    // `L^T[i][k]`.
                    let l_ki =
                        unsafe { *self.data.storage.get_unchecked(k, i) };
                    sum = sum - (l_ki * y[k]);
                }
                y[i] = sum;
            }

            for i in 0..D {
                // Safety: `i < D`, `c < COLS`.
                unsafe {
                    *b.storage.get_unchecked_mut(i, c) = y[i];
                }
            }
        }
    }
}
