//! # Linear Algebra
//!
//! Generic, storage-decoupled `Matrix<T, R, C, S>` built on top of the
//! `Storage`/`StorageMut`/`ContiguousStorage` trait family
//! ([`crate::math::storage`]) and the `Dim` compile-time dimension system
//! ([`crate::math::num_types`]).
//!
//! One `Matrix<T, R, C, S>` implementation — arithmetic, transposition,
//! factorization — operates over any conforming storage backend (stack
//! array, borrowed view). Arithmetic operators and decompositions read
//! elements through [`Storage::get_unchecked`]/[`StorageMut::get_unchecked_mut`]
//! rather than assuming a fixed physical layout, so mixed-layout operands
//! (e.g. one side a [`Matrix::transpose_view`]) are handled correctly with
//! no special-casing.
#![allow(
    clippy::arbitrary_source_item_ordering,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::similar_names,
    clippy::needless_range_loop,
    clippy::type_complexity,
    clippy::doc_markdown,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::option_if_let_else,
    clippy::must_use_candidate,
    clippy::many_single_char_names,
    clippy::collapsible_if,
    clippy::use_self,
    clippy::too_many_arguments,
    clippy::missing_const_for_fn,
    clippy::cast_lossless
)]

pub mod decomposition;
pub mod specialized;

#[cfg(any(test, feature = "ets"))]
/// Matrix module unit tests.
pub mod tests;

pub use decomposition::{
    CholeskyDecomposition, LdltDecomposition, LuDecomposition, QrDecomposition,
};
pub use specialized::{LowerTriangular, Symmetric, UpperTriangular};

use crate::math::num_traits::{One, Scalar, Zero};
use crate::math::num_types::{Const, Dim};
use crate::math::ops::{Add, Mul, Neg, Sub};
use crate::math::storage::{
    ArrayStorage, ColMajor, ContiguousStorage, ContiguousStorageMut,
    DenseStorage, DenseStorageMut, RowMajor, StaticStorageView, Storage,
    StorageInit, StorageMut, Trans,
};
use crate::math::subprograms::{DefaultBlas, level1::Axpy, level3::Gemm};
use core::marker::PhantomData;

/// A type-safe, storage-decoupled matrix.
///
/// `R`/`C` are compile-time dimensions ([`Dim`]); `S` determines where and
/// how elements are physically stored. See [`Owned`] for the default
/// stack-based backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
#[allow(clippy::type_complexity)]
pub struct Matrix<T, R: Dim, C: Dim, S: Storage<T, R, C>> {
    storage: S,
    _marker: PhantomData<(T, R, C)>,
}

/// The default owned, stack-based `Matrix`.
///
/// Fixed to the `Const<N>` bridge so its backing `ArrayStorage<T, R, C>` can
/// use plain `const usize` row/column parameters (`ArrayStorage` cannot be
/// generic over an arbitrary `Dim` type — see `matrix-design.md` §4.1).
pub type Owned<T, const R: usize, const C: usize> =
    Matrix<T, Const<R>, Const<C>, ArrayStorage<T, R, C>>;

/// A zero-copy, non-owning, immutable matrix view.
pub type MatrixSlice<'a, T, R, C, O = ColMajor> =
    Matrix<T, R, C, StaticStorageView<'a, T, R, C, O>>;

////////////////////////////////////////////////////////////////////////////////
// Constructors
////////////////////////////////////////////////////////////////////////////////

impl<T, const R: usize, const C: usize> Owned<T, R, C>
where
    Const<R>: Dim,
    Const<C>: Dim,
{
    /// Builds a matrix from a raw column-major `[[T; R]; C]` array.
    #[must_use]
    pub const fn from_array(data: [[T; R]; C]) -> Self {
        Self {
            storage: ArrayStorage::from_array(data),
            _marker: PhantomData,
        }
    }

    /// Builds an all-zero matrix using `T::ZERO`.
    ///
    /// `const fn` so static matrices can be placed directly in read-only
    /// memory.
    #[must_use]
    pub const fn zero() -> Self
    where
        T: Zero + Copy,
    {
        Self {
            storage: ArrayStorage::from_array([[T::ZERO; R]; C]),
            _marker: PhantomData,
        }
    }
}

impl<T, const D: usize> Owned<T, D, D>
where
    Const<D>: Dim,
{
    /// Builds the `D x D` multiplicative-identity matrix.
    #[must_use]
    pub const fn identity() -> Self
    where
        T: Zero + One + Copy,
    {
        let mut data = [[T::ZERO; D]; D];
        let mut j = 0;
        while j < D {
            data[j][j] = T::ONE;
            j += 1;
        }
        Self {
            storage: ArrayStorage::from_array(data),
            _marker: PhantomData,
        }
    }

    /// Builds a `D x D` diagonal matrix from `values`, filling off-diagonal
    /// elements with `T::ZERO`.
    #[must_use]
    pub const fn diagonal(values: [T; D]) -> Self
    where
        T: Zero + Copy,
    {
        let mut data = [[T::ZERO; D]; D];
        let mut j = 0;
        while j < D {
            data[j][j] = values[j];
            j += 1;
        }
        Self {
            storage: ArrayStorage::from_array(data),
            _marker: PhantomData,
        }
    }
}

impl<T, R: Dim, C: Dim, S: Storage<T, R, C>> Matrix<T, R, C, S> {
    /// Wraps a custom storage backend in a Matrix.
    pub const fn from_storage(storage: S) -> Self {
        Self {
            storage,
            _marker: PhantomData,
        }
    }

    /// Borrows the underlying storage backend.
    pub const fn storage(&self) -> &S {
        &self.storage
    }

    /// Mutably borrows the underlying storage backend.
    pub fn storage_mut(&mut self) -> &mut S {
        &mut self.storage
    }

    /// Unwraps the underlying storage backend.
    pub fn into_storage(self) -> S {
        self.storage
    }
}

impl<T, R, C, S> Matrix<T, R, C, S>
where
    R: Dim,
    C: Dim,
    S: Storage<T, R, C> + StorageInit<T, R, C>,
{
    /// Builds a matrix element-by-element from row/column indices.
    pub fn from_fn(f: impl FnMut(usize, usize) -> T) -> Self {
        Self {
            storage: S::from_fn(f),
            _marker: PhantomData,
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Access
////////////////////////////////////////////////////////////////////////////////

impl<T, R: Dim, C: Dim, S: Storage<T, R, C>> Matrix<T, R, C, S> {
    /// Number of logical columns.
    #[must_use]
    pub fn cols(&self) -> usize {
        self.storage.cols()
    }

    /// Returns a reference to the element at `(i, j)`, or `None` if out of
    /// bounds.
    #[must_use]
    pub fn get(&self, i: usize, j: usize) -> Option<&T> {
        self.storage.get(i, j)
    }

    /// Number of logical rows.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.storage.rows()
    }
}

impl<T, R: Dim, C: Dim, S: StorageMut<T, R, C>> Matrix<T, R, C, S> {
    /// Returns a mutable reference to the element at `(i, j)`, or `None` if
    /// out of bounds.
    pub fn get_mut(&mut self, i: usize, j: usize) -> Option<&mut T> {
        self.storage.get_mut(i, j)
    }
}

impl<T, R: Dim, C: Dim, S: Storage<T, R, C> + ContiguousStorage<T>>
    Matrix<T, R, C, S>
{
    /// Exposes a safe contiguous slice view of matrix memory.
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        self.storage.as_slice()
    }
}

impl<T, R: Dim, C: Dim, S: StorageMut<T, R, C> + ContiguousStorageMut<T>>
    Matrix<T, R, C, S>
{
    /// Exposes a safe mutable contiguous slice view of matrix memory.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.storage.as_mut_slice()
    }
}

impl<T, M: Dim, N: Dim, SA> Matrix<T, M, N, SA>
where
    T: Scalar + Copy,
    SA: DenseStorage<T, R = M, C = N>,
{
    /// Multiplies `self` by `rhs` into caller-provided `out` destination buffer using a specific BLAS engine.
    pub fn mul_into_with<B: Gemm<T, SA, SB, SC>, P: Dim, SB, SC>(
        &self,
        rhs: &Matrix<T, N, P, SB>,
        out: &mut Matrix<T, M, P, SC>,
    ) where
        SB: DenseStorage<T, R = N, C = P>,
        SC: DenseStorageMut<T, R = M, C = P>,
    {
        B::gemm(
            Trans::NoTrans,
            Trans::NoTrans,
            T::ONE,
            &self.storage,
            &rhs.storage,
            T::ZERO,
            &mut out.storage,
        );
    }

    /// Multiplies `self` by `rhs` into caller-provided `out` destination buffer using the default BLAS engine.
    pub fn mul_into<P: Dim, SB, SC>(
        &self,
        rhs: &Matrix<T, N, P, SB>,
        out: &mut Matrix<T, M, P, SC>,
    ) where
        SB: DenseStorage<T, R = N, C = P>,
        SC: DenseStorageMut<T, R = M, C = P>,
    {
        self.mul_into_with::<DefaultBlas, P, SB, SC>(rhs, out);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Operators
//
// Implemented for `&Matrix` (matching every worked example in
// `matrix-design.md`, e.g. `&diff * p_pred`), not for owned `Matrix`, so
// arithmetic never has to move a potentially large (up to 63KB, C-3) stack
// matrix. Delegated directly to canonical BLAS Level 1/3 subprogram kernels
// ([`DefaultBlas`]), ensuring zero-overhead compilation while enabling
// hardware-accelerated drop-in backends.
////////////////////////////////////////////////////////////////////////////////

impl<'b, T, const R: usize, const C: usize, S, S2>
    Add<&'b Matrix<T, Const<R>, Const<C>, S2>>
    for &Matrix<T, Const<R>, Const<C>, S>
where
    T: Scalar + Copy,
    Const<R>: Dim,
    Const<C>: Dim,
    S: DenseStorage<T, R = Const<R>, C = Const<C>>,
    S2: DenseStorage<T, R = Const<R>, C = Const<C>>,
{
    type Output = Owned<T, R, C>;

    fn add(self, rhs: &'b Matrix<T, Const<R>, Const<C>, S2>) -> Self::Output {
        let mut out = Owned::<T, R, C>::zero();
        DefaultBlas::axpy(T::ONE, &self.storage, &mut out.storage);
        DefaultBlas::axpy(T::ONE, &rhs.storage, &mut out.storage);
        out
    }
}

impl<'b, T, const R: usize, const C: usize, S, S2>
    Sub<&'b Matrix<T, Const<R>, Const<C>, S2>>
    for &Matrix<T, Const<R>, Const<C>, S>
where
    T: Scalar + Copy,
    Const<R>: Dim,
    Const<C>: Dim,
    S: DenseStorage<T, R = Const<R>, C = Const<C>>,
    S2: DenseStorage<T, R = Const<R>, C = Const<C>>,
{
    type Output = Owned<T, R, C>;

    fn sub(self, rhs: &'b Matrix<T, Const<R>, Const<C>, S2>) -> Self::Output {
        let mut out = Owned::<T, R, C>::zero();
        DefaultBlas::axpy(T::ONE, &self.storage, &mut out.storage);
        DefaultBlas::axpy(T::ZERO - T::ONE, &rhs.storage, &mut out.storage);
        out
    }
}

impl<T, const R: usize, const C: usize, S> Neg
    for &Matrix<T, Const<R>, Const<C>, S>
where
    T: Scalar + Copy,
    Const<R>: Dim,
    Const<C>: Dim,
    S: DenseStorage<T, R = Const<R>, C = Const<C>>,
{
    type Output = Owned<T, R, C>;

    fn neg(self) -> Self::Output {
        let mut out = Owned::<T, R, C>::zero();
        DefaultBlas::axpy(T::ZERO - T::ONE, &self.storage, &mut out.storage);
        out
    }
}

/// Matrix-matrix multiplication (`(M x N) * (N x P) -> (M x P)`), dimension
/// checked at compile time. Covers matrix-vector multiplication as the
/// `P == 1` case — no separate impl is needed since a vector is just a
/// `Matrix<T, N, 1, _>`.
impl<'b, T, const M: usize, const N: usize, const P: usize, S, S2>
    Mul<&'b Matrix<T, Const<N>, Const<P>, S2>>
    for &Matrix<T, Const<M>, Const<N>, S>
where
    T: Scalar + Copy,
    Const<M>: Dim,
    Const<N>: Dim,
    Const<P>: Dim,
    S: DenseStorage<T, R = Const<M>, C = Const<N>>,
    S2: DenseStorage<T, R = Const<N>, C = Const<P>>,
{
    type Output = Owned<T, M, P>;

    fn mul(self, rhs: &'b Matrix<T, Const<N>, Const<P>, S2>) -> Self::Output {
        let mut out = Owned::<T, M, P>::zero();
        self.mul_into(rhs, &mut out);
        out
    }
}

////////////////////////////////////////////////////////////////////////////////
// Transposition (Owned only — every design-doc example uses plain owned
// matrices; a fully storage-generic version needs a `Transposed<O>` layout
// marker not otherwise motivated by current call sites).
////////////////////////////////////////////////////////////////////////////////

impl<T: Copy, const R: usize, const C: usize> Owned<T, R, C>
where
    Const<R>: Dim,
    Const<C>: Dim,
{
    /// Creates a zero-copy, non-destructive transposed view over `self`.
    ///
    /// Reinterprets the same column-major flat data as a `C x R`, row-major
    /// view — the standard zero-copy transpose trick — without allocation
    /// or copying.
    #[must_use]
    #[allow(clippy::type_complexity)]
    pub fn transpose_view(
        &self,
    ) -> MatrixSlice<'_, T, Const<C>, Const<R>, RowMajor> {
        // Safety: `self.as_slice()` always holds exactly `R * C` elements
        // (`ArrayStorage`'s `ContiguousStorage` invariant), which equals
        // `C * R` for the transposed view being constructed here.
        let storage =
            unsafe { StaticStorageView::new_unchecked(self.as_slice()) };
        Matrix {
            storage,
            _marker: PhantomData,
        }
    }

    /// Writes the transpose of `self` into a caller-provided destination,
    /// avoiding a stack return.
    pub fn transpose_into(&self, dest: &mut Owned<T, C, R>) {
        for i in 0..R {
            for j in 0..C {
                // Safety: `i < R`, `j < C` for `self`; the same pair,
                // swapped, is `j < C`, `i < R` for `dest`'s `Const<C>`,
                // `Const<R>` shape.
                unsafe {
                    let value = *self.storage.get_unchecked(i, j);
                    *dest.storage.get_unchecked_mut(j, i) = value;
                }
            }
        }
    }

    /// Returns a new, transposed matrix (convenience API for small shapes).
    #[must_use]
    pub fn transpose(&self) -> Owned<T, C, R>
    where
        T: Zero,
    {
        let mut out = Owned::<T, C, R>::zero();
        self.transpose_into(&mut out);
        out
    }
}

impl<T: Copy, const D: usize> Owned<T, D, D>
where
    Const<D>: Dim,
{
    /// Performs an in-place transposition for square matrices.
    pub fn transpose_mut(&mut self) {
        for i in 0..D {
            for j in (i + 1)..D {
                // Safety: `i < D` and `j < D` per the loop bounds.
                unsafe {
                    let a = *self.storage.get_unchecked(i, j);
                    let b = *self.storage.get_unchecked(j, i);
                    *self.storage.get_unchecked_mut(i, j) = b;
                    *self.storage.get_unchecked_mut(j, i) = a;
                }
            }
        }
    }
}
