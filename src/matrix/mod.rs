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
//!
//! Dimension mismatches fail at compile time:
//!
//! ```compile_fail
//! use control_rs::matrix::Owned;
//! let a = Owned::<f64, 2, 2>::identity();
//! let b = Owned::<f64, 3, 1>::zero();
//! let _ = &a * &b;
//! ```
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
    QrDecompositionRect,
};
pub use specialized::{LowerTriangular, Symmetric, UpperTriangular};

use crate::math::StorageResult;
use crate::math::num_traits::{Float, One, Scalar, Zero};
use crate::math::num_types::{Const, Dim};
use crate::math::ops::{Add, Mul, Neg, Sub};
use crate::math::storage::{
    ArrayStorage, ColMajor, ContiguousStorage, ContiguousStorageMut,
    DenseStorage, DenseStorageMut, RowArrayStorage, RowMajor,
    StaticStorageView, StaticStorageViewMut, Storage, StorageInit, StorageMut,
    StorageView, StorageViewMut, Trans,
};
use crate::math::subprograms::{
    DefaultBlas,
    level1::{Axpy, Scal},
    level3::Gemm,
};
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

/// Column-major owning stack matrix (`matrix-design.md` §4.1).
pub type ArrayMatrix<T, const R: usize, const C: usize> =
    Matrix<T, Const<R>, Const<C>, ArrayStorage<T, R, C>>;

/// The default owned, stack-based `Matrix`.
///
/// Fixed to the `Const<N>` bridge so its backing `ArrayStorage<T, R, C>` can
/// use plain `const usize` row/column parameters (`ArrayStorage` cannot be
/// generic over an arbitrary `Dim` type — see `matrix-design.md` §4.1).
pub type Owned<T, const R: usize, const C: usize> = ArrayMatrix<T, R, C>;

/// A zero-copy, non-owning, immutable contiguous matrix view.
pub type MatrixSlice<'a, T, R, C, O = ColMajor> =
    Matrix<T, R, C, StaticStorageView<'a, T, R, C, O>>;

/// A zero-copy, non-owning, mutable contiguous matrix view.
pub type MatrixSliceMut<'a, T, R, C, O = ColMajor> =
    Matrix<T, R, C, StaticStorageViewMut<'a, T, R, C, O>>;

/// Strided (possibly non-contiguous) immutable view.
pub type MatrixView<'a, T, R, C> = Matrix<T, R, C, StorageView<'a, T, R, C>>;

/// Strided (possibly non-contiguous) mutable view.
pub type MatrixViewMut<'a, T, R, C> =
    Matrix<T, R, C, StorageViewMut<'a, T, R, C>>;

/// Row-major owning stack matrix.
pub type RowArrayMatrix<T, const R: usize, const C: usize> =
    Matrix<T, Const<R>, Const<C>, RowArrayStorage<T, R, C>>;
/// Alias for [`RowArrayMatrix`].
pub type RowOwned<T, const R: usize, const C: usize> = RowArrayMatrix<T, R, C>;

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
        Self::from_storage(ArrayStorage::zero())
    }
}

impl<T: Copy, const R: usize, const C: usize> Owned<T, R, C>
where
    Const<R>: Dim,
    Const<C>: Dim,
{
    /// Constructs an owned matrix from row-major nested array literals.
    ///
    /// Converts the input `[[T; C]; R]` row-major data into the underlying column-major
    /// storage format `[[T; R]; C]` at compile time or runtime without dynamic allocation.
    ///
    /// # Generic Arguments
    /// * `T` - Scalar element type (must implement [`Copy`]).
    /// * `R` - Number of rows.
    /// * `C` - Number of columns.
    ///
    /// # Arguments
    /// * `data` - Row-major nested arrays `[[T; C]; R]`.
    ///
    /// # Returns
    /// * `Self` - The constructed matrix backed by contiguous stack array storage.
    ///
    ///
    ///
    ///
    ///
    /// # Example
    /// ```
    /// use control_rs::matrix::Owned;
    ///
    /// let mat = Owned::<f64, 2, 3>::from_row_arrays([
    ///     [1.0, 2.0, 3.0],
    ///     [4.0, 5.0, 6.0],
    /// ]);
    /// assert_eq!(mat.get(0, 0), Some(&1.0));
    /// assert_eq!(mat.get(0, 2), Some(&3.0));
    /// assert_eq!(mat.get(1, 0), Some(&4.0));
    /// ```
    #[must_use]
    pub const fn from_row_arrays(data: [[T; C]; R]) -> Self {
        let mut col_major = [[data[0][0]; R]; C];
        let mut i = 0;
        while i < R {
            let mut j = 0;
            while j < C {
                col_major[j][i] = data[i][j];
                j += 1;
            }
            i += 1;
        }
        Self::from_array(col_major)
    }

    /// Returns the matrix contents as column-major nested arrays.
    ///
    /// Returns a copy of the underlying `[[T; R]; C]` storage buffer.
    ///
    /// # Returns
    /// * `[[T; R]; C]` - The matrix elements organized in column-major order.
    ///
    ///
    ///
    ///
    ///
    /// # Example
    /// ```
    /// use control_rs::matrix::Owned;
    ///
    /// let mat = Owned::<f64, 2, 2>::from_row_arrays([[1.0, 2.0], [3.0, 4.0]]);
    /// assert_eq!(mat.to_array(), [[1.0, 3.0], [2.0, 4.0]]);
    /// ```
    #[must_use]
    pub const fn to_array(&self) -> [[T; R]; C] {
        self.storage.to_array()
    }

    /// Returns the matrix contents as standard row-major nested arrays.
    ///
    /// Transposes the internal column-major data into a row-major `[[T; C]; R]` array
    /// without heap allocation, suitable for serialization and display.
    ///
    /// # Returns
    /// * `[[T; C]; R]` - The matrix elements organized in row-major order.
    ///
    ///
    ///
    ///
    ///
    /// # Example
    /// ```
    /// use control_rs::matrix::Owned;
    ///
    /// let mat = Owned::<f64, 2, 2>::from_array([[1.0, 3.0], [2.0, 4.0]]);
    /// assert_eq!(mat.to_row_arrays(), [[1.0, 2.0], [3.0, 4.0]]);
    /// ```
    #[must_use]
    pub const fn to_row_arrays(&self) -> [[T; C]; R] {
        let col_data = self.storage.as_array();
        let mut row_major = [[col_data[0][0]; C]; R];
        let mut i = 0;
        while i < R {
            let mut j = 0;
            while j < C {
                row_major[i][j] = col_data[j][i];
                j += 1;
            }
            i += 1;
        }
        row_major
    }
}

impl<T, const N: usize> Owned<T, N, 1>
where
    Const<N>: Dim,
{
    /// Constructs an $N \times 1$ column vector from a 1D array.
    ///
    /// # Generic Arguments
    /// * `T` - Scalar element type.
    /// * `N` - Dimension (number of rows) of the column vector.
    ///
    /// # Arguments
    /// * `data` - A fixed-size 1D array `[T; N]` containing column elements.
    ///
    /// # Returns
    /// * `Self` - An $N \times 1$ column vector matrix.
    ///
    ///
    ///
    ///
    ///
    /// # Example
    /// ```
    /// use control_rs::matrix::Owned;
    ///
    /// let col = Owned::<f64, 3, 1>::from_column([1.0, 2.0, 3.0]);
    /// assert_eq!(col.get(0, 0), Some(&1.0));
    /// assert_eq!(col.get(2, 0), Some(&3.0));
    /// ```
    #[must_use]
    pub const fn from_column(data: [T; N]) -> Self {
        Self::from_storage(ArrayStorage::from_column(data))
    }
}

impl<T: Copy, const N: usize> Owned<T, 1, N>
where
    Const<N>: Dim,
{
    /// Constructs a $1 \times N$ row vector from a 1D array.
    ///
    /// # Generic Arguments
    /// * `T` - Scalar element type (must implement [`Copy`]).
    /// * `N` - Dimension (number of columns) of the row vector.
    ///
    /// # Arguments
    /// * `data` - A fixed-size 1D array `[T; N]` containing row elements.
    ///
    /// # Returns
    /// * `Self` - A $1 \times N$ row vector matrix.
    ///
    ///
    ///
    ///
    ///
    /// # Example
    /// ```
    /// use control_rs::matrix::Owned;
    ///
    /// let row = Owned::<f64, 1, 3>::from_row([1.0, 2.0, 3.0]);
    /// assert_eq!(row.get(0, 0), Some(&1.0));
    /// assert_eq!(row.get(0, 2), Some(&3.0));
    /// ```
    #[must_use]
    pub const fn from_row(data: [T; N]) -> Self {
        Self::from_storage(ArrayStorage::from_row(data))
    }
}

impl<T: Copy> Owned<T, 1, 1> {
    /// Constructs a $1 \times 1$ scalar matrix.
    ///
    /// # Generic Arguments
    /// * `T` - Scalar element type (must implement [`Copy`]).
    ///
    /// # Arguments
    /// * `val` - The single scalar value.
    ///
    /// # Returns
    /// * `Self` - A $1 \times 1$ matrix wrapping `val`.
    ///
    ///
    ///
    ///
    ///
    /// # Example
    /// ```
    /// use control_rs::matrix::Owned;
    ///
    /// let s = Owned::<f64, 1, 1>::scalar(42.0);
    /// assert_eq!(s.get(0, 0), Some(&42.0));
    /// ```
    #[must_use]
    pub const fn scalar(val: T) -> Self {
        Self::from_array([[val]])
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
        Self::from_storage(ArrayStorage::identity())
    }

    /// Builds a `D x D` diagonal matrix from `values`, filling off-diagonal
    /// elements with `T::ZERO`.
    #[must_use]
    pub const fn diagonal(values: [T; D]) -> Self
    where
        T: Zero + Copy,
    {
        Self::from_storage(ArrayStorage::diagonal(values))
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

    /// Writes `val` at `(i, j)`.
    ///
    /// # Errors
    /// Returns [`StorageError::OutOfBounds`] if either index is out of bounds.
    pub fn set(&mut self, i: usize, j: usize, val: T) -> StorageResult<()> {
        self.storage.set(i, j, val)
    }
}

impl<T: Copy, const R: usize, const C: usize, S>
    Matrix<T, Const<R>, Const<C>, S>
where
    Const<R>: Dim,
    Const<C>: Dim,
    S: StorageMut<T, Const<R>, Const<C>>,
{
    /// Copies `src` into this matrix starting at `(row, col)`.
    pub fn write_block<const BR: usize, const BC: usize, S2>(
        &mut self,
        row: usize,
        col: usize,
        src: &Matrix<T, Const<BR>, Const<BC>, S2>,
    ) where
        Const<BR>: Dim,
        Const<BC>: Dim,
        S2: Storage<T, Const<BR>, Const<BC>>,
    {
        for i in 0..BR {
            for j in 0..BC {
                if let (Some(target), Some(&v)) =
                    (self.get_mut(row + i, col + j), src.get(i, j))
                {
                    *target = v;
                }
            }
        }
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

    /// Contiguous immutable [`MatrixSlice`].
    #[must_use]
    pub fn slice(&self) -> MatrixSlice<'_, T, R, C> {
        let storage = unsafe {
            StaticStorageView::new_unchecked(self.storage.as_slice())
        };
        Matrix::from_storage(storage)
    }
}

impl<T, R: Dim, C: Dim, S: StorageMut<T, R, C> + ContiguousStorageMut<T>>
    Matrix<T, R, C, S>
{
    /// Exposes a safe mutable contiguous slice view of matrix memory.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.storage.as_mut_slice()
    }

    /// Contiguous mutable [`MatrixSliceMut`].
    pub fn slice_mut(&mut self) -> MatrixSliceMut<'_, T, R, C> {
        let storage = unsafe {
            StaticStorageViewMut::new_unchecked(self.storage.as_mut_slice())
        };
        Matrix::from_storage(storage)
    }
}

impl<T, M: Dim, N: Dim, SA> Matrix<T, M, N, SA>
where
    T: Scalar + Copy,
    SA: DenseStorage<T, R = M, C = N>,
{
    /// Multiplies `self` by `rhs` into caller-provided `out` using backend `B`.
    pub fn mul_into_with<B, P: Dim, SB, SC>(
        &self,
        rhs: &Matrix<T, N, P, SB>,
        out: &mut Matrix<T, M, P, SC>,
    ) where
        B: Gemm<T, SA, SB, SC>,
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

    /// Multiplies `self` by `rhs` into caller-provided `out` via [`DefaultBlas`].
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

impl<T, const R: usize, const C: usize, S> Mul<T>
    for &Matrix<T, Const<R>, Const<C>, S>
where
    T: Scalar + Copy,
    Const<R>: Dim,
    Const<C>: Dim,
    S: DenseStorage<T, R = Const<R>, C = Const<C>>,
{
    type Output = Owned<T, R, C>;

    fn mul(self, rhs: T) -> Self::Output {
        let mut out = Owned::<T, R, C>::zero();
        DefaultBlas::axpy(T::ONE, &self.storage, &mut out.storage);
        DefaultBlas::scal(rhs, &mut out.storage);
        out
    }
}

impl<T, const R: usize, const C: usize, S> Mul<&T>
    for &Matrix<T, Const<R>, Const<C>, S>
where
    T: Scalar + Copy,
    Const<R>: Dim,
    Const<C>: Dim,
    S: DenseStorage<T, R = Const<R>, C = Const<C>>,
{
    type Output = Owned<T, R, C>;

    fn mul(self, rhs: &T) -> Self::Output {
        self * *rhs
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

impl<T, R: Dim, C: Dim, S: DenseStorage<T, R = R, C = C>> Matrix<T, R, C, S> {
    /// Zero-copy strided view of the full matrix.
    #[must_use]
    pub fn view(&self) -> MatrixView<'_, T, R, C> {
        let storage = unsafe {
            StorageView::new_with_strides_unchecked(
                self.storage.as_ptr(),
                self.storage.r_stride(),
                self.storage.c_stride(),
            )
        };
        Matrix::from_storage(storage)
    }

    /// Zero-copy reversed-index view.
    #[must_use]
    pub fn reverse_view(&self) -> MatrixView<'_, T, R, C> {
        Matrix::from_storage(self.storage.reverse_view())
    }

    /// Rectangular window of size `R2 × C2` starting at `(origin_row, origin_col)`.
    #[must_use]
    pub fn submatrix<const R2: usize, const C2: usize>(
        &self,
        origin_row: usize,
        origin_col: usize,
    ) -> Option<MatrixView<'_, T, Const<R2>, Const<C2>>>
    where
        Const<R2>: Dim,
        Const<C2>: Dim,
    {
        if origin_row.checked_add(R2)? > R::USIZE
            || origin_col.checked_add(C2)? > C::USIZE
        {
            return None;
        }
        let rs = self.storage.r_stride();
        let cs = self.storage.c_stride();
        let ptr = unsafe {
            self.storage.as_ptr().offset(
                origin_row.cast_signed() * rs + origin_col.cast_signed() * cs,
            )
        };
        let storage = unsafe {
            StorageView::<T, Const<R2>, Const<C2>>::new_with_strides_unchecked(
                ptr, rs, cs,
            )
        };
        Some(Matrix::from_storage(storage))
    }
}

impl<T, R: Dim, C: Dim, S: DenseStorageMut<T, R = R, C = C>>
    Matrix<T, R, C, S>
{
    /// Zero-copy mutable strided view of the full matrix.
    pub fn view_mut(&mut self) -> MatrixViewMut<'_, T, R, C> {
        let storage = unsafe {
            StorageViewMut::new_with_strides_unchecked(
                self.storage.as_mut_ptr(),
                self.storage.r_stride(),
                self.storage.c_stride(),
            )
        };
        Matrix::from_storage(storage)
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

impl<T: Scalar + Copy, const N: usize, S> Matrix<T, Const<N>, Const<N>, S>
where
    Const<N>: Dim,
    S: Storage<T, Const<N>, Const<N>>,
{
    /// Sum of diagonal entries.
    #[must_use]
    pub fn trace(&self) -> T {
        let mut s = T::ZERO;
        for i in 0..N {
            if let Some(&v) = self.get(i, i) {
                s = s + v;
            }
        }
        s
    }
}

impl<T: Float + Copy, const N: usize, S> Matrix<T, Const<N>, Const<N>, S>
where
    Const<N>: Dim,
    S: Storage<T, Const<N>, Const<N>>,
{
    /// Induced infinity norm (maximum absolute row sum).
    #[must_use]
    pub fn inf_norm(&self) -> T {
        let mut best = T::ZERO;
        for i in 0..N {
            let mut row = T::ZERO;
            for j in 0..N {
                if let Some(&v) = self.get(i, j) {
                    row = row + v.abs();
                }
            }
            if row > best {
                best = row;
            }
        }
        best
    }
}

impl<T: Float + Copy, const N: usize> Owned<T, N, N>
where
    Const<N>: Dim,
{
    /// Padé $[6/6]$ scaling-and-squaring matrix exponential.
    #[must_use]
    pub fn expm(&self) -> Self {
        let two = T::ONE + T::ONE;
        let theta = two + T::ONE;
        let mut a_scaled = *self;
        let mut s = 0_u32;
        while a_scaled.inf_norm() > theta && s < 16 {
            a_scaled = &a_scaled * (T::ONE / two);
            s += 1;
        }

        let a2 = &a_scaled * &a_scaled;
        let a4 = &a2 * &a2;
        let a6 = &a4 * &a2;
        let (b0, b1, b2, b3, b4, b5, b6) = pade6_coeffs::<T>();
        let u_inner = &(&(&Owned::<T, N, N>::identity() * b1) + &(&a2 * b3))
            + &(&a4 * b5);
        let u = &a_scaled * &u_inner;
        let v = &(&(&(&Owned::<T, N, N>::identity() * b0) + &(&a2 * b2))
            + &(&a4 * b4))
            + &(&a6 * b6);

        let mut r = &v + &u;
        if let Ok(lu) = LuDecomposition::decompose(&v - &u) {
            let _ = lu.solve_mut(&mut r);
        }
        for _ in 0..s {
            r = &r * &r;
        }
        r
    }
}

fn pade6_coeffs<T: Float + Copy>() -> (T, T, T, T, T, T, T) {
    let two = T::ONE + T::ONE;
    (
        T::ONE,
        T::ONE / two,
        T::from_usize(5) / T::from_usize(44),
        T::ONE / T::from_usize(66),
        T::ONE / T::from_usize(792),
        T::ONE / T::from_usize(15_840),
        T::ONE / T::from_usize(665_280),
    )
}
