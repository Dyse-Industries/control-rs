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
    clippy::arithmetic_side_effects
)]

pub mod decomposition;
pub mod specialized;

#[cfg(any(test, feature = "hil"))]
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
    ArrayStorage, ColMajor, ContiguousStorage, ContiguousStorageMut, RowMajor,
    Storage, StorageInit, StorageMut, StorageView,
};
use core::marker::PhantomData;

/// A type-safe, storage-decoupled matrix.
///
/// `R`/`C` are compile-time dimensions ([`Dim`]); `S` determines where and
/// how elements are physically stored. See [`Owned`] for the default
/// stack-based backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    Matrix<T, R, C, StorageView<'a, T, R, C, O>>;

////////////////////////////////////////////////////////////////////////////////
// Constructors
////////////////////////////////////////////////////////////////////////////////

impl<T, const R: usize, const C: usize> Owned<T, R, C>
where
    Const<R>: Dim,
    Const<C>: Dim,
{
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

impl<T, R: Dim, C: Dim, S: ContiguousStorage<T, R, C>> Matrix<T, R, C, S> {
    /// Exposes a safe contiguous slice view of matrix memory.
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        self.storage.as_slice()
    }
}

impl<T, R: Dim, C: Dim, S: ContiguousStorageMut<T, R, C>> Matrix<T, R, C, S> {
    /// Exposes a safe mutable contiguous slice view of matrix memory.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.storage.as_mut_slice()
    }
}

////////////////////////////////////////////////////////////////////////////////
// Operators
//
// Implemented for `&Matrix` (matching every worked example in
// `matrix-design.md`, e.g. `&diff * p_pred`), not for owned `Matrix`, so
// arithmetic never has to move a potentially large (up to 63KB, C-3) stack
// matrix. Implemented as direct loops over `Storage::get_unchecked` rather
// than routed through `subprograms`'s `GEMV`/`GEMM`: `Matrix` has no backend
// type parameter to select a `GEMM<T>` implementor (those exist only for
// concrete `f32`/`f64`, not generic `T: Scalar`), and two operands can
// legitimately carry different storage layouts (e.g. one side a
// `transpose_view()`) that a single shared `order` parameter can't express.
// Each operand's own `Storage::offset` already encodes its layout, so this
// delivers on the "any conforming backend" promise directly.
////////////////////////////////////////////////////////////////////////////////

impl<'b, T, const R: usize, const C: usize, S, S2>
    Add<&'b Matrix<T, Const<R>, Const<C>, S2>>
    for &Matrix<T, Const<R>, Const<C>, S>
where
    T: Scalar + Copy,
    Const<R>: Dim,
    Const<C>: Dim,
    S: Storage<T, Const<R>, Const<C>>,
    S2: Storage<T, Const<R>, Const<C>>,
{
    type Output = Owned<T, R, C>;

    fn add(self, rhs: &'b Matrix<T, Const<R>, Const<C>, S2>) -> Self::Output {
        let mut out = Owned::<T, R, C>::zero();
        for i in 0..R {
            for j in 0..C {
                // Safety: `i < R` and `j < C` per the loop bounds; `self`,
                // `rhs` and `out` all share `Const<R>`/`Const<C>`.
                unsafe {
                    let value = *self.storage.get_unchecked(i, j)
                        + *rhs.storage.get_unchecked(i, j);
                    *out.storage.get_unchecked_mut(i, j) = value;
                }
            }
        }
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
    S: Storage<T, Const<R>, Const<C>>,
    S2: Storage<T, Const<R>, Const<C>>,
{
    type Output = Owned<T, R, C>;

    fn sub(self, rhs: &'b Matrix<T, Const<R>, Const<C>, S2>) -> Self::Output {
        let mut out = Owned::<T, R, C>::zero();
        for i in 0..R {
            for j in 0..C {
                // Safety: see `Add` above.
                unsafe {
                    let value = *self.storage.get_unchecked(i, j)
                        - *rhs.storage.get_unchecked(i, j);
                    *out.storage.get_unchecked_mut(i, j) = value;
                }
            }
        }
        out
    }
}

impl<T, const R: usize, const C: usize, S> Neg
    for &Matrix<T, Const<R>, Const<C>, S>
where
    T: Scalar + Copy,
    Const<R>: Dim,
    Const<C>: Dim,
    S: Storage<T, Const<R>, Const<C>>,
{
    type Output = Owned<T, R, C>;

    fn neg(self) -> Self::Output {
        let mut out = Owned::<T, R, C>::zero();
        for i in 0..R {
            for j in 0..C {
                // Safety: `i < R` and `j < C` per the loop bounds, matching
                // `self`'s and `out`'s shared `Const<R>`/`Const<C>`.
                unsafe {
                    *out.storage.get_unchecked_mut(i, j) =
                        T::ZERO - *self.storage.get_unchecked(i, j);
                }
            }
        }
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
    S: Storage<T, Const<M>, Const<N>>,
    S2: Storage<T, Const<N>, Const<P>>,
{
    type Output = Owned<T, M, P>;

    fn mul(self, rhs: &'b Matrix<T, Const<N>, Const<P>, S2>) -> Self::Output {
        let mut out = Owned::<T, M, P>::zero();
        for i in 0..M {
            for j in 0..P {
                let mut sum = T::ZERO;
                for k in 0..N {
                    // Safety: `i < M`, `k < N` within `self`'s bounds;
                    // `k < N`, `j < P` within `rhs`'s bounds.
                    unsafe {
                        sum = sum
                            + *self.storage.get_unchecked(i, k)
                                * *rhs.storage.get_unchecked(k, j);
                    }
                }
                // Safety: `i < M` and `j < P` per the loop bounds.
                unsafe {
                    *out.storage.get_unchecked_mut(i, j) = sum;
                }
            }
        }
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
        let storage = unsafe { StorageView::new_unchecked(self.as_slice()) };
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
