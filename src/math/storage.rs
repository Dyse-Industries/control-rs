//! Storage backends for `R x C` grids of data, decoupled from dimension
//! checking.
//!
//! `Storage`/`StorageMut`/`ContiguousStorage`/`ContiguousStorageMut` tie the
//! compile-time `Dim` system ([`crate::math::num_types`]) to physical
//! memory. `offset()` is the single point of control for a backend's memory
//! layout; higher-level code accesses elements through it rather than
//! assuming a layout.
//! Includes dense strided backends (`ArrayStorage`, `RowArrayStorage`,
//! `StorageView`), packed structured storage (`DiagonalStorage`,
//! `SymmetricPackedStorage`, `HermitianPackedStorage`, `TriangularPackedStorage`),
//! compressed sparse backends (`ArrayCsrStorage`, `ArrayCscStorage`, `ArrayCooStorage`),
//! and 1-D sparse vectors (`ArraySparseVector`, `ViewSparseVector`).
//!
//! Runtime-stride views do not implement [`ContiguousStorage`]:
//!
//! ```compile_fail
//! use control_rs::math::storage::{
//!     ArrayStorage, ContiguousStorage, DenseStorage,
//! };
//!
//! fn needs_contiguous<S: ContiguousStorage<f64>>(_: &S) {}
//!
//! let a = ArrayStorage::<f64, 2, 2>::from_array([[1.0, 2.0], [3.0, 4.0]]);
//! let rev = a.reverse_view();
//! needs_contiguous(&rev);
//! ```
//!
//! Owning packed constructors const-assert `PACKED_LEN = N(N+1)/2`:
//!
//! ```compile_fail
//! use control_rs::math::storage::{SymmetricPackedStorage, UpLo};
//!
//! let _ = SymmetricPackedStorage::<f32, 4, 9>::new([0.0; 9], UpLo::Upper);
//! ```
#![allow(clippy::arbitrary_source_item_ordering)]
#![allow(clippy::indexing_slicing)]
#![allow(clippy::arithmetic_side_effects)]
#![allow(clippy::eq_op)]
#![allow(clippy::too_many_arguments)]

type ColMajorArray<T, const R: usize, const C: usize> = [[T; R]; C];
type RowMajorArray<T, const R: usize, const C: usize> = [[T; C]; R];
type ViewMarker<'a, T, R, C> = PhantomData<(&'a [T], R, C)>;
type ViewMarkerMut<'a, T, R, C> = PhantomData<(&'a mut [T], R, C)>;
type StorageMarker<R, C, O> = PhantomData<(R, C, O)>;
type SlicePair<'a, T> = (&'a [usize], &'a [T]);

use crate::math::num_traits::{One, Scalar, Zero};
use crate::math::num_types::{Const, Dim};
pub use crate::math::{
    ConversionError, ConversionResult, StorageError, StorageResult,
};
use core::marker::PhantomData;
use core::mem::MaybeUninit;

type UninitArray<T, const N: usize> = MaybeUninit<[T; N]>;

/// Returns whether every `(r, c)` in an `rows × cols` window with the given
/// strides lands inside a slice of length `len` (offsets measured from index 0).
fn strided_window_fits(
    len: usize,
    r_stride: isize,
    c_stride: isize,
    rows: usize,
    cols: usize,
) -> bool {
    for r in 0..rows {
        for c in 0..cols {
            let Some(off) =
                r.cast_signed().checked_mul(r_stride).and_then(|row| {
                    c.cast_signed().checked_mul(c_stride).map(|col| row + col)
                })
            else {
                return false;
            };
            if off < 0 {
                return false;
            }
            if off.cast_unsigned() >= len {
                return false;
            }
        }
    }
    true
}

/// Helper function to reverse arrays given to `Polynomial::new()`
#[allow(clippy::indexing_slicing, clippy::arithmetic_side_effects)]
#[inline]
pub const fn reverse_array<T: Copy, const N: usize>(input: [T; N]) -> [T; N] {
    let mut output = input;
    let mut i = 0;
    while i < N / 2 {
        let tmp = output[i];
        output[i] = output[N - 1 - i];
        output[N - 1 - i] = tmp;
        i += 1;
    }

    output
}

/// Initialize an array from an iterator.
///
/// # Safety
/// The iterator must have **at least** `N` elements or this assumes an
/// uninitialized value is initialized (undefined behavior).
///
/// # Panics
/// Panics in debug builds if the iterator is shorter than `N`.
#[allow(clippy::arithmetic_side_effects)]
pub(crate) unsafe fn array_from_iterator<I, T, const N: usize>(
    iterator: I,
) -> [T; N]
where
    I: IntoIterator<Item = T>,
{
    let mut maybe_uninit_array: UninitArray<T, N> = MaybeUninit::uninit();
    let arr_ptr = maybe_uninit_array.as_mut_ptr().cast::<T>();
    let mut write_counter = 0;
    for (i, b) in (0..N).zip(iterator) {
        unsafe {
            arr_ptr.add(i).write(b);
        }
        write_counter += 1;
    }
    debug_assert_eq!(write_counter, N);
    unsafe { maybe_uninit_array.assume_init() }
}

////////////////////////////////////////////////////////////////////////////////
// Operational & Structural Enumerations
////////////////////////////////////////////////////////////////////////////////

/// Memory layout convention for a contiguous 2-D storage backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixLayout {
    /// Elements of a row are contiguous in memory.
    RowMajor = 1,
    /// Elements of a column are contiguous in memory.
    ColMajor = 2,
}

/// Zero-sized type-level tag for a contiguous backend's memory layout.
pub trait LayoutMarker: 'static {
    /// The [`MatrixLayout`] this marker represents.
    const ORDER: MatrixLayout;
    /// Maps a logical `(i, j)` index into an `R x C` grid to a physical
    /// element offset under this layout.
    fn offset(rows: usize, cols: usize, i: usize, j: usize) -> isize;
}

/// [`LayoutMarker`] tag: elements of a column are contiguous in memory.
pub struct ColMajor;

impl LayoutMarker for ColMajor {
    const ORDER: MatrixLayout = MatrixLayout::ColMajor;
    #[allow(clippy::arithmetic_side_effects)]
    fn offset(rows: usize, _cols: usize, i: usize, j: usize) -> isize {
        (j * rows + i).cast_signed()
    }
}

/// [`LayoutMarker`] tag: elements of a row are contiguous in memory.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct RowMajor;

impl LayoutMarker for RowMajor {
    const ORDER: MatrixLayout = MatrixLayout::RowMajor;
    #[allow(clippy::arithmetic_side_effects)]
    fn offset(_rows: usize, cols: usize, i: usize, j: usize) -> isize {
        (i * cols + j).cast_signed()
    }
}

/// Upper or lower triangular storage format selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpLo {
    /// Upper triangular matrix or portion.
    Upper,
    /// Lower triangular matrix or portion.
    Lower,
}

/// Unit or non-unit diagonal selector for triangular matrices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Diag {
    /// Diagonal elements are explicitly stored and arbitrary.
    NonUnit,
    /// Diagonal elements are assumed to be ONE and not stored/modified.
    Unit,
}

/// Left or right operand position for matrix multiplication routines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    /// Operation applies on the left: $C \leftarrow \alpha A B + \beta C$.
    Left,
    /// Operation applies on the right: $C \leftarrow \alpha B A + \beta C$.
    Right,
}

/// Transposition and conjugate transposition operation selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trans {
    /// No transpose: $\text{op}(A) = A$.
    NoTrans,
    /// Transpose: $\text{op}(A) = A^T$.
    Trans,
    /// Conjugate transpose (Adjoint): $\text{op}(A) = A^H$.
    ConjTrans,
}

////////////////////////////////////////////////////////////////////////////////
// Core Strided & Contiguous Storage Hierarchy
////////////////////////////////////////////////////////////////////////////////

/// Read access to a 2D grid of elements with compile-time dimensions.
///
/// # Safety
/// `as_ptr()` must point to an addressable buffer and `offset(r, c)` must produce
/// a valid pointer within that buffer for all `r < Self::R::USIZE` and `c < Self::C::USIZE`.
pub unsafe trait DenseStorage<T> {
    /// Associated compile-time row dimension.
    type R: Dim;
    /// Associated compile-time column dimension.
    type C: Dim;

    /// Number of logical rows (`Self::R::USIZE`).
    fn rows(&self) -> usize {
        Self::R::USIZE
    }

    /// Number of logical columns (`Self::C::USIZE`).
    fn cols(&self) -> usize {
        Self::C::USIZE
    }

    /// Returns the stride in elements between consecutive rows.
    fn r_stride(&self) -> isize;

    /// Returns the stride in elements between consecutive columns.
    fn c_stride(&self) -> isize;

    /// Returns a raw pointer to the backend's first addressable element.
    fn as_ptr(&self) -> *const T;

    /// Alias for [`DenseStorage::as_ptr`].
    fn ptr(&self) -> *const T {
        self.as_ptr()
    }

    /// Maps a logical `(row, column)` index to a physical element offset from [`DenseStorage::as_ptr`].
    #[inline(always)]
    #[allow(clippy::arithmetic_side_effects)]
    fn offset(&self, r: usize, c: usize) -> isize {
        (r.cast_signed() * self.r_stride())
            + (c.cast_signed() * self.c_stride())
    }

    /// Returns a reference to the element at `(r, c)` without bounds checking.
    ///
    /// # Safety
    /// `r < self.rows()` and `c < self.cols()` must hold.
    #[inline(always)]
    unsafe fn get_unchecked(&self, r: usize, c: usize) -> &T {
        let off = self.offset(r, c);
        unsafe { &*self.as_ptr().offset(off) }
    }

    /// Returns a reference to the element at `(r, c)` or `None` if either index is out of bounds.
    fn get(&self, r: usize, c: usize) -> Option<&T> {
        if r < self.rows() && c < self.cols() {
            Some(unsafe { self.get_unchecked(r, c) })
        } else {
            None
        }
    }

    /// Returns a zero-copy transposed view of the storage grid.
    #[allow(clippy::type_complexity)]
    fn transpose_view<'a>(&'a self) -> StorageView<'a, T, Self::C, Self::R>
    where
        Self: 'a,
    {
        unsafe {
            StorageView::new_with_strides_unchecked(
                self.as_ptr(),
                self.c_stride(),
                self.r_stride(),
            )
        }
    }

    /// Returns a zero-copy reversed view of the storage grid.
    #[allow(clippy::type_complexity)]
    fn reverse_view<'a>(&'a self) -> StorageView<'a, T, Self::R, Self::C>
    where
        Self: 'a,
    {
        let off = self.offset(
            Self::R::USIZE.saturating_sub(1),
            Self::C::USIZE.saturating_sub(1),
        );
        unsafe {
            StorageView::new_with_strides_unchecked(
                self.as_ptr().offset(off),
                -self.r_stride(),
                -self.c_stride(),
            )
        }
    }
}

/// Mutable access to a 2D grid of elements with compile-time dimensions.
///
/// # Safety
/// `as_mut_ptr()` must return a pointer to the same backing memory `as_ptr()` describes,
/// with unique (exclusive) access for the lifetime of the borrow.
pub unsafe trait DenseStorageMut<T>: DenseStorage<T> {
    /// Returns a raw mutable pointer to the backend's first addressable element.
    fn as_mut_ptr(&mut self) -> *mut T;

    /// Alias for [`DenseStorageMut::as_mut_ptr`].
    fn ptr_mut(&mut self) -> *mut T {
        self.as_mut_ptr()
    }

    /// Returns a mutable reference to the element at `(r, c)` without bounds checking.
    ///
    /// # Safety
    /// `r < self.rows()` and `c < self.cols()` must hold.
    #[inline(always)]
    unsafe fn get_mut_unchecked(&mut self, r: usize, c: usize) -> &mut T {
        let off = self.offset(r, c);
        unsafe { &mut *self.as_mut_ptr().offset(off) }
    }

    /// Alias for [`DenseStorageMut::get_mut_unchecked`] for backwards compatibility.
    ///
    /// # Safety
    /// `r < self.rows()` and `c < self.cols()` must hold.
    #[inline(always)]
    unsafe fn get_unchecked_mut(&mut self, r: usize, c: usize) -> &mut T {
        unsafe { self.get_mut_unchecked(r, c) }
    }

    /// Returns a mutable reference to the element at `(r, c)` or `None` if out of bounds.
    fn get_mut(&mut self, r: usize, c: usize) -> Option<&mut T> {
        if r < self.rows() && c < self.cols() {
            Some(unsafe { self.get_mut_unchecked(r, c) })
        } else {
            None
        }
    }

    /// Updates the element at `(r, c)` with `val`.
    ///
    /// # Errors
    /// Returns [`StorageError::OutOfBounds`] if either index is out of bounds.
    fn set(&mut self, r: usize, c: usize, val: T) -> StorageResult<()> {
        if r < self.rows() && c < self.cols() {
            unsafe {
                self.set_unchecked(r, c, val);
            }
            Ok(())
        } else {
            Err(StorageError::OutOfBounds)
        }
    }

    /// Updates the element at `(r, c)` without bounds checking.
    ///
    /// # Safety
    /// `r < self.rows()` and `c < self.cols()` must hold.
    #[inline(always)]
    unsafe fn set_unchecked(&mut self, r: usize, c: usize, val: T) {
        unsafe {
            *self.get_mut_unchecked(r, c) = val;
        }
    }

    /// Returns a mutable zero-copy transposed view of the storage grid.
    #[allow(clippy::type_complexity)]
    fn transpose_mut_view<'a>(
        &'a mut self,
    ) -> StorageViewMut<'a, T, Self::C, Self::R>
    where
        Self: 'a,
    {
        unsafe {
            StorageViewMut::new_with_strides_unchecked(
                self.as_mut_ptr(),
                self.c_stride(),
                self.r_stride(),
            )
        }
    }

    /// Returns a mutable zero-copy reversed view of the storage grid.
    #[allow(clippy::type_complexity)]
    fn reverse_mut_view<'a>(
        &'a mut self,
    ) -> StorageViewMut<'a, T, Self::R, Self::C>
    where
        Self: 'a,
    {
        let off = self.offset(
            Self::R::USIZE.saturating_sub(1),
            Self::C::USIZE.saturating_sub(1),
        );
        unsafe {
            StorageViewMut::new_with_strides_unchecked(
                self.as_mut_ptr().offset(off),
                -self.r_stride(),
                -self.c_stride(),
            )
        }
    }
}

/// Marker for backends whose elements are laid out with
/// no padding or stride gaps in column-major or row-major order.
///
/// # Safety
/// `as_slice()` must return exactly `Self::R::USIZE * Self::C::USIZE` contiguous elements.
pub unsafe trait ContiguousStorage<T>: DenseStorage<T> {
    /// The memory layout convention followed by this backend.
    const ORDER: MatrixLayout;

    /// Returns the backend's elements as a single contiguous slice.
    fn as_slice(&self) -> &[T];
}

/// Mutable contiguous storage.
///
/// # Safety
/// Combines [`DenseStorageMut`] and [`ContiguousStorage`] invariants.
pub unsafe trait ContiguousStorageMut<T>:
    ContiguousStorage<T> + DenseStorageMut<T>
{
    /// Returns the backend's elements as a single mutable contiguous slice.
    fn as_mut_slice(&mut self) -> &mut [T];
}

/// Backwards-compatibility trait for code expecting `Storage<T, R, C>`.
///
/// # Safety
/// Implementors must uphold all safety invariants of [`DenseStorage`].
pub unsafe trait Storage<T, R: Dim, C: Dim>:
    DenseStorage<T, R = R, C = C>
{
}
unsafe impl<T, R: Dim, C: Dim, S: DenseStorage<T, R = R, C = C>>
    Storage<T, R, C> for S
{
}

/// Backwards-compatibility trait for code expecting `StorageMut<T, R, C>`.
///
/// # Safety
/// Implementors must uphold all safety invariants of [`DenseStorageMut`].
pub unsafe trait StorageMut<T, R: Dim, C: Dim>:
    DenseStorageMut<T, R = R, C = C> + Storage<T, R, C>
{
}
unsafe impl<T, R: Dim, C: Dim, S: DenseStorageMut<T, R = R, C = C>>
    StorageMut<T, R, C> for S
{
}

////////////////////////////////////////////////////////////////////////////////
// Initialization Strategies
////////////////////////////////////////////////////////////////////////////////

/// Safe initialization strategies for a storage backend without requiring `T: Default`.
pub trait StorageInit<T, R: Dim, C: Dim>: Sized {
    /// Builds a backend element-by-element from row/column indices.
    fn from_fn(f: impl FnMut(usize, usize) -> T) -> Self;

    /// Builds a backend by cloning `val` into every element.
    fn from_element(val: T) -> Self
    where
        T: Clone,
    {
        Self::from_fn(|_, _| val.clone())
    }

    /// Builds a backend filled with the additive identity.
    #[must_use]
    fn zeros() -> Self
    where
        T: Zero,
    {
        Self::from_fn(|_, _| T::ZERO)
    }

    /// Builds a backend representing the multiplicative identity matrix.
    #[must_use]
    fn identity() -> Self
    where
        T: Zero + One,
    {
        Self::from_fn(|i, j| if i == j { T::ONE } else { T::ZERO })
    }
}

////////////////////////////////////////////////////////////////////////////////
// ArrayStorage (Column-Major Default) & RowArrayStorage (Row-Major)
////////////////////////////////////////////////////////////////////////////////

/// Owning column-major stack storage backed by `[[T; R]; C]`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArrayStorage<T, const R: usize, const C: usize> {
    data: ColMajorArray<T, R, C>,
}

impl<T, const R: usize, const C: usize> ArrayStorage<T, R, C> {
    /// Builds a backend directly from a column-major `[[T; R]; C]` array.
    #[must_use]
    pub const fn from_array(data: ColMajorArray<T, R, C>) -> Self {
        Self { data }
    }

    /// Returns the backend's elements as a flat slice.
    #[must_use]
    pub const fn as_slice(&self) -> &[T] {
        self.data.as_flattened()
    }

    /// Returns the backend's elements as a mutable flat slice.
    #[must_use]
    pub const fn as_mut_slice(&mut self) -> &mut [T] {
        self.data.as_flattened_mut()
    }

    /// Builds an all-zero storage backend.
    #[must_use]
    pub const fn zero() -> Self
    where
        T: Zero + Copy,
    {
        Self::from_array([[T::ZERO; R]; C])
    }
}

impl<T, const N: usize> ArrayStorage<T, N, 1> {
    /// Builds a column vector backend from a 1D array of elements.
    #[must_use]
    pub const fn from_column(data: [T; N]) -> Self {
        Self::from_array([data])
    }
}

impl<T: Copy, const N: usize> ArrayStorage<T, 1, N> {
    /// Builds a row vector backend from a 1D array of elements.
    #[must_use]
    pub const fn from_row(data: [T; N]) -> Self {
        let mut arr = [[data[0]]; N];
        let mut j = 0;
        while j < N {
            arr[j][0] = data[j];
            j += 1;
        }
        Self::from_array(arr)
    }
}

impl<T, const D: usize> ArrayStorage<T, D, D> {
    /// Builds the `D x D` multiplicative identity storage backend.
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
        Self::from_array(data)
    }

    /// Builds a `D x D` diagonal storage backend from `values`.
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
        Self::from_array(data)
    }
}

unsafe impl<T, const R: usize, const C: usize> DenseStorage<T>
    for ArrayStorage<T, R, C>
where
    Const<R>: Dim,
    Const<C>: Dim,
{
    type R = Const<R>;
    type C = Const<C>;

    fn r_stride(&self) -> isize {
        1
    }

    fn c_stride(&self) -> isize {
        R.cast_signed()
    }

    fn as_ptr(&self) -> *const T {
        self.data.as_ptr().cast()
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn offset(&self, r: usize, c: usize) -> isize {
        (c * R + r).cast_signed()
    }

    unsafe fn get_unchecked(&self, r: usize, c: usize) -> &T {
        let off = self.offset(r, c);
        unsafe { &*self.as_ptr().offset(off) }
    }
}

unsafe impl<T, const R: usize, const C: usize> DenseStorageMut<T>
    for ArrayStorage<T, R, C>
where
    Const<R>: Dim,
    Const<C>: Dim,
{
    fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr().cast()
    }

    unsafe fn get_mut_unchecked(&mut self, r: usize, c: usize) -> &mut T {
        let off = self.offset(r, c);
        unsafe { &mut *self.as_mut_ptr().offset(off) }
    }
}

unsafe impl<T, const R: usize, const C: usize> ContiguousStorage<T>
    for ArrayStorage<T, R, C>
where
    Const<R>: Dim,
    Const<C>: Dim,
{
    const ORDER: MatrixLayout = MatrixLayout::ColMajor;

    fn as_slice(&self) -> &[T] {
        self.data.as_flattened()
    }
}

unsafe impl<T, const R: usize, const C: usize> ContiguousStorageMut<T>
    for ArrayStorage<T, R, C>
where
    Const<R>: Dim,
    Const<C>: Dim,
{
    fn as_mut_slice(&mut self) -> &mut [T] {
        self.data.as_flattened_mut()
    }
}

impl<T, const R: usize, const C: usize> StorageInit<T, Const<R>, Const<C>>
    for ArrayStorage<T, R, C>
where
    Const<R>: Dim,
    Const<C>: Dim,
{
    fn from_fn(mut f: impl FnMut(usize, usize) -> T) -> Self {
        Self {
            data: core::array::from_fn(|j| core::array::from_fn(|i| f(i, j))),
        }
    }
}

/// Owning row-major stack storage backed by `[[T; C]; R]`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowArrayStorage<T, const R: usize, const C: usize> {
    data: RowMajorArray<T, R, C>,
}

impl<T, const R: usize, const C: usize> RowArrayStorage<T, R, C> {
    /// Builds a backend directly from a row-major `[[T; C]; R]` array.
    #[must_use]
    pub const fn from_array(data: RowMajorArray<T, R, C>) -> Self {
        Self { data }
    }

    /// Returns the backend's elements as a flat slice.
    #[must_use]
    pub const fn as_slice(&self) -> &[T] {
        self.data.as_flattened()
    }

    /// Returns the backend's elements as a mutable flat slice.
    #[must_use]
    pub const fn as_mut_slice(&mut self) -> &mut [T] {
        self.data.as_flattened_mut()
    }

    /// Builds an all-zero row-major storage backend.
    #[must_use]
    pub const fn zero() -> Self
    where
        T: Zero + Copy,
    {
        Self::from_array([[T::ZERO; C]; R])
    }
}

impl<T, const N: usize> RowArrayStorage<T, 1, N> {
    /// Builds a row vector backend from a 1D array of elements.
    #[must_use]
    pub const fn from_row(data: [T; N]) -> Self {
        Self::from_array([data])
    }
}

impl<T: Copy, const N: usize> RowArrayStorage<T, N, 1> {
    /// Builds a column vector backend from a 1D array of elements.
    #[must_use]
    pub const fn from_column(data: [T; N]) -> Self {
        let mut arr = [[data[0]]; N];
        let mut i = 0;
        while i < N {
            arr[i][0] = data[i];
            i += 1;
        }
        Self::from_array(arr)
    }
}

impl<T, const D: usize> RowArrayStorage<T, D, D> {
    /// Builds the `D x D` multiplicative identity row-major storage backend.
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
        Self::from_array(data)
    }

    /// Builds a `D x D` diagonal row-major storage backend from `values`.
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
        Self::from_array(data)
    }
}

unsafe impl<T, const R: usize, const C: usize> DenseStorage<T>
    for RowArrayStorage<T, R, C>
where
    Const<R>: Dim,
    Const<C>: Dim,
{
    type R = Const<R>;
    type C = Const<C>;

    fn r_stride(&self) -> isize {
        C.cast_signed()
    }

    fn c_stride(&self) -> isize {
        1
    }

    fn as_ptr(&self) -> *const T {
        self.data.as_ptr().cast()
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn offset(&self, r: usize, c: usize) -> isize {
        (r * C + c).cast_signed()
    }

    unsafe fn get_unchecked(&self, r: usize, c: usize) -> &T {
        let off = self.offset(r, c);
        unsafe { &*self.as_ptr().offset(off) }
    }
}

unsafe impl<T, const R: usize, const C: usize> DenseStorageMut<T>
    for RowArrayStorage<T, R, C>
where
    Const<R>: Dim,
    Const<C>: Dim,
{
    fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr().cast()
    }

    unsafe fn get_mut_unchecked(&mut self, r: usize, c: usize) -> &mut T {
        let off = self.offset(r, c);
        unsafe { &mut *self.as_mut_ptr().offset(off) }
    }
}

unsafe impl<T, const R: usize, const C: usize> ContiguousStorage<T>
    for RowArrayStorage<T, R, C>
where
    Const<R>: Dim,
    Const<C>: Dim,
{
    const ORDER: MatrixLayout = MatrixLayout::RowMajor;

    fn as_slice(&self) -> &[T] {
        self.data.as_flattened()
    }
}

unsafe impl<T, const R: usize, const C: usize> ContiguousStorageMut<T>
    for RowArrayStorage<T, R, C>
where
    Const<R>: Dim,
    Const<C>: Dim,
{
    fn as_mut_slice(&mut self) -> &mut [T] {
        self.data.as_flattened_mut()
    }
}

impl<T, const R: usize, const C: usize> StorageInit<T, Const<R>, Const<C>>
    for RowArrayStorage<T, R, C>
where
    Const<R>: Dim,
    Const<C>: Dim,
{
    fn from_fn(mut f: impl FnMut(usize, usize) -> T) -> Self {
        Self {
            data: core::array::from_fn(|i| core::array::from_fn(|j| f(i, j))),
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// StorageView / StorageViewMut (Strided & Transpose Views)
////////////////////////////////////////////////////////////////////////////////

/// Zero-copy non-owning immutable view with arbitrary `isize` strides.
#[derive(Debug)]
#[allow(clippy::type_complexity)]
pub struct StorageView<'a, T, R: Dim, C: Dim> {
    ptr: *const T,
    r_stride: isize,
    c_stride: isize,
    _marker: ViewMarker<'a, T, R, C>,
}

impl<'a, T, R: Dim, C: Dim> StorageView<'a, T, R, C> {
    /// Wraps `data` with explicit row/column strides.
    ///
    /// # Errors
    /// Returns [`ConversionError::DimensionMismatch`] if `data` cannot cover
    /// the strided `R × C` window.
    pub fn new_with_strides(
        data: &'a [T],
        r_stride: isize,
        c_stride: isize,
    ) -> ConversionResult<Self> {
        if strided_window_fits(
            data.len(),
            r_stride,
            c_stride,
            R::USIZE,
            C::USIZE,
        ) {
            Ok(unsafe {
                Self::new_with_strides_unchecked(
                    data.as_ptr(),
                    r_stride,
                    c_stride,
                )
            })
        } else {
            Err(ConversionError::DimensionMismatch)
        }
    }

    /// Constructs a view from a raw pointer and explicit row/column strides.
    ///
    /// # Safety
    /// `ptr` with `r_stride` and `c_stride` must safely address `R x C` elements for lifetime `'a`.
    #[must_use]
    pub const unsafe fn new_with_strides_unchecked(
        ptr: *const T,
        r_stride: isize,
        c_stride: isize,
    ) -> Self {
        Self {
            ptr,
            r_stride,
            c_stride,
            _marker: PhantomData,
        }
    }
}

unsafe impl<T, R: Dim, C: Dim> DenseStorage<T> for StorageView<'_, T, R, C> {
    type R = R;
    type C = C;

    fn r_stride(&self) -> isize {
        self.r_stride
    }

    fn c_stride(&self) -> isize {
        self.c_stride
    }

    fn as_ptr(&self) -> *const T {
        self.ptr
    }
}

/// Zero-copy non-owning mutable view with arbitrary `isize` strides.
#[derive(Debug)]
#[allow(clippy::type_complexity)]
pub struct StorageViewMut<'a, T, R: Dim, C: Dim> {
    ptr: *mut T,
    r_stride: isize,
    c_stride: isize,
    _marker: ViewMarkerMut<'a, T, R, C>,
}

impl<'a, T, R: Dim, C: Dim> StorageViewMut<'a, T, R, C> {
    /// Wraps `data` with explicit row/column strides.
    ///
    /// # Errors
    /// Returns [`ConversionError::DimensionMismatch`] if `data` cannot cover
    /// the strided `R × C` window.
    pub fn new_with_strides(
        data: &'a mut [T],
        r_stride: isize,
        c_stride: isize,
    ) -> ConversionResult<Self> {
        if strided_window_fits(
            data.len(),
            r_stride,
            c_stride,
            R::USIZE,
            C::USIZE,
        ) {
            Ok(unsafe {
                Self::new_with_strides_unchecked(
                    data.as_mut_ptr(),
                    r_stride,
                    c_stride,
                )
            })
        } else {
            Err(ConversionError::DimensionMismatch)
        }
    }

    /// Constructs a mutable view from a raw pointer and explicit row/column strides.
    ///
    /// # Safety
    /// `ptr` with `r_stride` and `c_stride` must safely address `R x C` elements for lifetime `'a`.
    #[must_use]
    pub const unsafe fn new_with_strides_unchecked(
        ptr: *mut T,
        r_stride: isize,
        c_stride: isize,
    ) -> Self {
        Self {
            ptr,
            r_stride,
            c_stride,
            _marker: PhantomData,
        }
    }
}

unsafe impl<T, R: Dim, C: Dim> DenseStorage<T> for StorageViewMut<'_, T, R, C> {
    type R = R;
    type C = C;

    fn r_stride(&self) -> isize {
        self.r_stride
    }

    fn c_stride(&self) -> isize {
        self.c_stride
    }

    fn as_ptr(&self) -> *const T {
        self.ptr.cast_const()
    }
}

unsafe impl<T, R: Dim, C: Dim> DenseStorageMut<T>
    for StorageViewMut<'_, T, R, C>
{
    fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
}

////////////////////////////////////////////////////////////////////////////////
// StaticStorageView / StaticStorageViewMut (LayoutMarker Views)
////////////////////////////////////////////////////////////////////////////////

/// Non-owning view parameterized by [`LayoutMarker`].
#[derive(Debug)]
#[allow(clippy::type_complexity)]
pub struct StaticStorageView<'a, T, R: Dim, C: Dim, O: LayoutMarker> {
    data: &'a [T],
    _marker: StorageMarker<R, C, O>,
}

impl<'a, T, R: Dim, C: Dim, O: LayoutMarker> StaticStorageView<'a, T, R, C, O> {
    /// Wraps `data` as an `R x C` view with layout marker `O`.
    ///
    /// # Errors
    /// Returns [`ConversionError::DimensionMismatch`] if `data.len() != R::USIZE * C::USIZE`.
    pub const fn new(data: &'a [T]) -> ConversionResult<Self> {
        match R::USIZE.checked_mul(C::USIZE) {
            Some(expected_len) => {
                if expected_len == data.len() {
                    Ok(Self {
                        data,
                        _marker: PhantomData,
                    })
                } else {
                    Err(ConversionError::DimensionMismatch)
                }
            }
            None => Err(ConversionError::DimensionMismatch),
        }
    }

    /// Internal fast path without size check.
    ///
    /// # Safety
    /// `data.len()` must equal `R::USIZE * C::USIZE`.
    #[must_use]
    pub const unsafe fn new_unchecked(data: &'a [T]) -> Self {
        Self {
            data,
            _marker: PhantomData,
        }
    }
}

unsafe impl<T, R: Dim, C: Dim, O: LayoutMarker> DenseStorage<T>
    for StaticStorageView<'_, T, R, C, O>
{
    type R = R;
    type C = C;

    fn r_stride(&self) -> isize {
        match O::ORDER {
            MatrixLayout::ColMajor => 1,
            MatrixLayout::RowMajor => C::USIZE.cast_signed(),
        }
    }

    fn c_stride(&self) -> isize {
        match O::ORDER {
            MatrixLayout::ColMajor => R::USIZE.cast_signed(),
            MatrixLayout::RowMajor => 1,
        }
    }

    fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    fn offset(&self, r: usize, c: usize) -> isize {
        O::offset(R::USIZE, C::USIZE, r, c)
    }

    unsafe fn get_unchecked(&self, r: usize, c: usize) -> &T {
        let off = self.offset(r, c);
        unsafe { &*self.as_ptr().offset(off) }
    }
}

unsafe impl<T, R: Dim, C: Dim, O: LayoutMarker> ContiguousStorage<T>
    for StaticStorageView<'_, T, R, C, O>
{
    const ORDER: MatrixLayout = O::ORDER;

    fn as_slice(&self) -> &[T] {
        self.data
    }
}

/// Mutable non-owning view parameterized by [`LayoutMarker`].
#[derive(Debug)]
#[allow(clippy::type_complexity)]
pub struct StaticStorageViewMut<'a, T, R: Dim, C: Dim, O: LayoutMarker> {
    data: &'a mut [T],
    _marker: StorageMarker<R, C, O>,
}

impl<'a, T, R: Dim, C: Dim, O: LayoutMarker>
    StaticStorageViewMut<'a, T, R, C, O>
{
    /// Wraps `data` as a mutable `R x C` view with layout marker `O`.
    ///
    /// # Errors
    /// Returns [`ConversionError::DimensionMismatch`] if `data.len() != R::USIZE * C::USIZE`.
    pub const fn new(data: &'a mut [T]) -> ConversionResult<Self> {
        match R::USIZE.checked_mul(C::USIZE) {
            Some(expected_len) => {
                if expected_len == data.len() {
                    Ok(Self {
                        data,
                        _marker: PhantomData,
                    })
                } else {
                    Err(ConversionError::DimensionMismatch)
                }
            }
            None => Err(ConversionError::DimensionMismatch),
        }
    }

    /// Internal fast path without size check.
    ///
    /// # Safety
    /// `data.len()` must equal `R::USIZE * C::USIZE`.
    #[must_use]
    pub const unsafe fn new_unchecked(data: &'a mut [T]) -> Self {
        Self {
            data,
            _marker: PhantomData,
        }
    }
}

unsafe impl<T, R: Dim, C: Dim, O: LayoutMarker> DenseStorage<T>
    for StaticStorageViewMut<'_, T, R, C, O>
{
    type R = R;
    type C = C;

    fn r_stride(&self) -> isize {
        match O::ORDER {
            MatrixLayout::ColMajor => 1,
            MatrixLayout::RowMajor => C::USIZE.cast_signed(),
        }
    }

    fn c_stride(&self) -> isize {
        match O::ORDER {
            MatrixLayout::ColMajor => R::USIZE.cast_signed(),
            MatrixLayout::RowMajor => 1,
        }
    }

    fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    fn offset(&self, r: usize, c: usize) -> isize {
        O::offset(R::USIZE, C::USIZE, r, c)
    }

    unsafe fn get_unchecked(&self, r: usize, c: usize) -> &T {
        let off = self.offset(r, c);
        unsafe { &*self.as_ptr().offset(off) }
    }
}

unsafe impl<T, R: Dim, C: Dim, O: LayoutMarker> DenseStorageMut<T>
    for StaticStorageViewMut<'_, T, R, C, O>
{
    fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }
}

unsafe impl<T, R: Dim, C: Dim, O: LayoutMarker> ContiguousStorage<T>
    for StaticStorageViewMut<'_, T, R, C, O>
{
    const ORDER: MatrixLayout = O::ORDER;

    fn as_slice(&self) -> &[T] {
        self.data
    }
}

unsafe impl<T, R: Dim, C: Dim, O: LayoutMarker> ContiguousStorageMut<T>
    for StaticStorageViewMut<'_, T, R, C, O>
{
    fn as_mut_slice(&mut self) -> &mut [T] {
        self.data
    }
}

////////////////////////////////////////////////////////////////////////////////
// Packed Structured Storage Hierarchy
////////////////////////////////////////////////////////////////////////////////

/// Read access to packed structured storage ($N \times N$ structured matrices).
///
/// # Safety
/// Implementors must ensure `as_slice()` provides valid backing memory for the packed representation
/// and that index calculations do not cause out-of-bounds reads.
pub unsafe trait PackedStorage<T> {
    /// Associated compile-time matrix dimension.
    type N: Dim;

    /// Dimension $N$ of the square structured matrix (`Self::N::USIZE`).
    fn dim(&self) -> usize {
        Self::N::USIZE
    }

    /// Storage format triangle selection (Upper or Lower).
    fn uplo(&self) -> UpLo;

    /// Returns the underlying flat packed 1-D buffer.
    fn as_slice(&self) -> &[T];

    /// Returns the 1-D packed index for coordinate `(i, j)` if physically stored.
    fn packed_index(&self, i: usize, j: usize) -> Option<usize>;

    /// Computes the 1-D packed index without bounds checking.
    fn packed_index_unchecked(&self, i: usize, j: usize) -> usize;

    /// Evaluates the algebraic matrix entry $A_{i,j}$ (applying symmetry, conjugation, or zeros).
    fn value(&self, i: usize, j: usize) -> Option<T>;

    /// Evaluates the algebraic matrix entry $A_{i,j}$ without bounds checking.
    fn value_unchecked(&self, i: usize, j: usize) -> T;
}

/// Mutable access to physically stored entries of a packed structured matrix.
///
/// # Safety
/// Implementors must ensure `as_mut_slice()` provides exclusive, valid mutable access to the backing buffer.
pub unsafe trait PackedStorageMut<T>: PackedStorage<T> {
    /// Returns the underlying flat packed 1-D buffer as mutable.
    fn as_mut_slice(&mut self) -> &mut [T];

    /// Updates the physical storage slot at `(i, j)`.
    ///
    /// # Errors
    /// Returns [`StorageError::OutOfBounds`] if either index is out of bounds,
    /// or [`StorageError::InvalidStructuralInvariant`] if attempting to modify
    /// an element not physically stored.
    fn set(&mut self, i: usize, j: usize, val: T) -> StorageResult<()>;

    /// Updates the physical storage slot at `(i, j)` without bounds checking.
    ///
    /// # Safety
    /// `(i, j)` must be in the physical triangle.
    unsafe fn set_unchecked(&mut self, i: usize, j: usize, val: T);
}

/// Diagonal storage for $N \times N$ matrices storing exactly $N$ elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiagonalStorage<T, const N: usize> {
    data: [T; N],
}

impl<T, const N: usize> DiagonalStorage<T, N> {
    /// Creates diagonal storage from an array of $N$ diagonal elements.
    #[must_use]
    pub const fn from_array(data: [T; N]) -> Self {
        Self { data }
    }
}

impl<T: Scalar, const N: usize> DiagonalStorage<T, N>
where
    Const<N>: Dim,
{
    /// Projects the diagonal of a dense $N \times N$ matrix.
    ///
    /// Off-diagonal entries of `dense` are discarded.
    ///
    /// # Errors
    /// This projection is total with respect to shape; it returns [`Ok`].
    pub fn from_dense_diagonal(
        dense: &ArrayStorage<T, N, N>,
    ) -> StorageResult<Self> {
        let data: [T; N] = core::array::from_fn(|i| unsafe {
            dense.get_unchecked(i, i).clone()
        });
        Ok(Self::from_array(data))
    }
}

unsafe impl<T: Scalar, const N: usize> PackedStorage<T>
    for DiagonalStorage<T, N>
where
    Const<N>: Dim,
{
    type N = Const<N>;

    fn uplo(&self) -> UpLo {
        UpLo::Upper
    }

    fn as_slice(&self) -> &[T] {
        &self.data
    }

    fn packed_index(&self, i: usize, j: usize) -> Option<usize> {
        if i == j && i < N { Some(i) } else { None }
    }

    fn packed_index_unchecked(&self, i: usize, _j: usize) -> usize {
        i
    }

    fn value(&self, i: usize, j: usize) -> Option<T> {
        if i < N && j < N {
            Some(self.value_unchecked(i, j))
        } else {
            None
        }
    }

    #[allow(clippy::indexing_slicing)]
    fn value_unchecked(&self, i: usize, j: usize) -> T {
        if i == j {
            self.data[i].clone()
        } else {
            T::ZERO
        }
    }
}

unsafe impl<T: Scalar, const N: usize> PackedStorageMut<T>
    for DiagonalStorage<T, N>
where
    Const<N>: Dim,
{
    fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    #[allow(clippy::indexing_slicing)]
    fn set(&mut self, i: usize, j: usize, val: T) -> StorageResult<()> {
        if i >= N || j >= N {
            return Err(StorageError::OutOfBounds);
        }
        if i != j {
            return Err(StorageError::InvalidStructuralInvariant);
        }
        self.data[i] = val;
        Ok(())
    }

    #[allow(clippy::indexing_slicing)]
    unsafe fn set_unchecked(&mut self, i: usize, _j: usize, val: T) {
        self.data[i] = val;
    }
}

/// Symmetric packed storage for $N \times N$ matrices storing $N(N+1)/2$ elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SymmetricPackedStorage<T, const N: usize, const PACKED_LEN: usize> {
    data: [T; PACKED_LEN],
    uplo: UpLo,
}

impl<T, const N: usize, const PACKED_LEN: usize>
    SymmetricPackedStorage<T, N, PACKED_LEN>
{
    /// Creates symmetric packed storage from a flat packed buffer.
    #[must_use]
    pub const fn new(data: [T; PACKED_LEN], uplo: UpLo) -> Self {
        const { assert!(PACKED_LEN == N * (N + 1) / 2) };
        Self { data, uplo }
    }
}

impl<T: Scalar, const N: usize, const PACKED_LEN: usize>
    SymmetricPackedStorage<T, N, PACKED_LEN>
where
    Const<N>: Dim,
{
    /// Projects one triangle of a dense $N \times N$ matrix into packed
    /// symmetric storage.
    ///
    /// `uplo` names the stored triangle. A mismatched triangle on `dense` is
    /// a caller error, not a [`StorageError`].
    ///
    /// # Errors
    /// This projection is total with respect to shape; it returns [`Ok`].
    pub fn from_dense_triangle(
        dense: &ArrayStorage<T, N, N>,
        uplo: UpLo,
    ) -> StorageResult<Self> {
        let mut data: [T; PACKED_LEN] = core::array::from_fn(|_| T::ZERO);
        copy_dense_triangle(dense, uplo, &mut data, false);
        Ok(Self::new(data, uplo))
    }
}

#[allow(clippy::arithmetic_side_effects, clippy::indexing_slicing)]
unsafe impl<T: Scalar, const N: usize, const PACKED_LEN: usize> PackedStorage<T>
    for SymmetricPackedStorage<T, N, PACKED_LEN>
where
    Const<N>: Dim,
{
    type N = Const<N>;

    fn uplo(&self) -> UpLo {
        self.uplo
    }

    fn as_slice(&self) -> &[T] {
        &self.data
    }

    fn packed_index(&self, i: usize, j: usize) -> Option<usize> {
        if i >= N || j >= N {
            return None;
        }
        match self.uplo {
            UpLo::Upper => {
                if i <= j {
                    Some(i + (j * (j + 1)) / 2)
                } else {
                    None
                }
            }
            UpLo::Lower => {
                if i >= j {
                    Some(i - j + (j * (2 * N - j + 1)) / 2)
                } else {
                    None
                }
            }
        }
    }

    fn packed_index_unchecked(&self, i: usize, j: usize) -> usize {
        match self.uplo {
            UpLo::Upper => i + (j * (j + 1)) / 2,
            UpLo::Lower => i - j + (j * (2 * N - j + 1)) / 2,
        }
    }

    fn value(&self, i: usize, j: usize) -> Option<T> {
        if i < N && j < N {
            Some(self.value_unchecked(i, j))
        } else {
            None
        }
    }

    fn value_unchecked(&self, i: usize, j: usize) -> T {
        match self.uplo {
            UpLo::Upper => {
                let (r, c) = if i <= j { (i, j) } else { (j, i) };
                let idx = r + (c * (c + 1)) / 2;
                self.data[idx].clone()
            }
            UpLo::Lower => {
                let (r, c) = if i >= j { (i, j) } else { (j, i) };
                let idx = r - c + (c * (2 * N - c + 1)) / 2;
                self.data[idx].clone()
            }
        }
    }
}

#[allow(clippy::indexing_slicing)]
unsafe impl<T: Scalar, const N: usize, const PACKED_LEN: usize>
    PackedStorageMut<T> for SymmetricPackedStorage<T, N, PACKED_LEN>
where
    Const<N>: Dim,
{
    fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    fn set(&mut self, i: usize, j: usize, val: T) -> StorageResult<()> {
        if let Some(idx) = self.packed_index(i, j) {
            self.data[idx] = val;
            Ok(())
        } else if i < N && j < N {
            Err(StorageError::InvalidStructuralInvariant)
        } else {
            Err(StorageError::OutOfBounds)
        }
    }

    unsafe fn set_unchecked(&mut self, i: usize, j: usize, val: T) {
        let idx = self.packed_index_unchecked(i, j);
        self.data[idx] = val;
    }
}

/// Hermitian packed storage enforcing conjugate reflection $A_{i,j} = \overline{A_{j,i}}$
/// and real diagonal elements ($\text{Im}(A_{i,i}) = 0$).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HermitianPackedStorage<T, const N: usize, const PACKED_LEN: usize> {
    data: [T; PACKED_LEN],
    uplo: UpLo,
}

impl<T, const N: usize, const PACKED_LEN: usize>
    HermitianPackedStorage<T, N, PACKED_LEN>
{
    /// Creates Hermitian packed storage.
    #[must_use]
    pub const fn new(data: [T; PACKED_LEN], uplo: UpLo) -> Self {
        const { assert!(PACKED_LEN == N * (N + 1) / 2) };
        Self { data, uplo }
    }
}

impl<T: Scalar, const N: usize, const PACKED_LEN: usize>
    HermitianPackedStorage<T, N, PACKED_LEN>
where
    Const<N>: Dim,
{
    /// Projects one triangle of a dense $N \times N$ matrix into packed
    /// Hermitian storage.
    ///
    /// `uplo` names the stored triangle. A non-real diagonal returns
    /// [`StorageError::InvalidHermitianDiagonal`].
    ///
    /// # Errors
    /// Returns [`StorageError::InvalidHermitianDiagonal`] if any stored
    /// diagonal entry is not equal to its conjugate.
    pub fn from_dense_triangle(
        dense: &ArrayStorage<T, N, N>,
        uplo: UpLo,
    ) -> StorageResult<Self> {
        for k in 0..N {
            let val = unsafe { dense.get_unchecked(k, k).clone() };
            if val.clone() != val.conj() {
                return Err(StorageError::InvalidHermitianDiagonal);
            }
        }
        let mut data: [T; PACKED_LEN] = core::array::from_fn(|_| T::ZERO);
        copy_dense_triangle(dense, uplo, &mut data, false);
        Ok(Self::new(data, uplo))
    }
}

#[allow(clippy::arithmetic_side_effects, clippy::indexing_slicing)]
unsafe impl<T: Scalar, const N: usize, const PACKED_LEN: usize> PackedStorage<T>
    for HermitianPackedStorage<T, N, PACKED_LEN>
where
    Const<N>: Dim,
{
    type N = Const<N>;

    fn uplo(&self) -> UpLo {
        self.uplo
    }

    fn as_slice(&self) -> &[T] {
        &self.data
    }

    fn packed_index(&self, i: usize, j: usize) -> Option<usize> {
        if i >= N || j >= N {
            return None;
        }
        match self.uplo {
            UpLo::Upper => {
                if i <= j {
                    Some(i + (j * (j + 1)) / 2)
                } else {
                    None
                }
            }
            UpLo::Lower => {
                if i >= j {
                    Some(i - j + (j * (2 * N - j + 1)) / 2)
                } else {
                    None
                }
            }
        }
    }

    fn packed_index_unchecked(&self, i: usize, j: usize) -> usize {
        match self.uplo {
            UpLo::Upper => i + (j * (j + 1)) / 2,
            UpLo::Lower => i - j + (j * (2 * N - j + 1)) / 2,
        }
    }

    fn value(&self, i: usize, j: usize) -> Option<T> {
        if i < N && j < N {
            Some(self.value_unchecked(i, j))
        } else {
            None
        }
    }

    fn value_unchecked(&self, i: usize, j: usize) -> T {
        match self.uplo {
            UpLo::Upper => {
                if i <= j {
                    let idx = i + (j * (j + 1)) / 2;
                    self.data[idx].clone()
                } else {
                    let idx = j + (i * (i + 1)) / 2;
                    self.data[idx].clone().conj()
                }
            }
            UpLo::Lower => {
                if i >= j {
                    let idx = i - j + (j * (2 * N - j + 1)) / 2;
                    self.data[idx].clone()
                } else {
                    let idx = j - i + (i * (2 * N - i + 1)) / 2;
                    self.data[idx].clone().conj()
                }
            }
        }
    }
}

#[allow(clippy::indexing_slicing)]
unsafe impl<T: Scalar, const N: usize, const PACKED_LEN: usize>
    PackedStorageMut<T> for HermitianPackedStorage<T, N, PACKED_LEN>
where
    Const<N>: Dim,
{
    fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    fn set(&mut self, i: usize, j: usize, val: T) -> StorageResult<()> {
        if i >= N || j >= N {
            return Err(StorageError::OutOfBounds);
        }
        if i == j && val.clone() != val.clone().conj() {
            return Err(StorageError::InvalidHermitianDiagonal);
        }
        if let Some(idx) = self.packed_index(i, j) {
            self.data[idx] = val;
            Ok(())
        } else {
            Err(StorageError::InvalidStructuralInvariant)
        }
    }

    unsafe fn set_unchecked(&mut self, i: usize, j: usize, val: T) {
        let idx = self.packed_index_unchecked(i, j);
        self.data[idx] = val;
    }
}

/// Triangular packed storage for $N \times N$ matrices with optional unit diagonal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TriangularPackedStorage<T, const N: usize, const PACKED_LEN: usize> {
    data: [T; PACKED_LEN],
    uplo: UpLo,
    diag: Diag,
}

impl<T, const N: usize, const PACKED_LEN: usize>
    TriangularPackedStorage<T, N, PACKED_LEN>
{
    /// Creates triangular packed storage.
    #[must_use]
    pub const fn new(data: [T; PACKED_LEN], uplo: UpLo, diag: Diag) -> Self {
        const { assert!(PACKED_LEN == N * (N + 1) / 2) };
        Self { data, uplo, diag }
    }
}

impl<T: Scalar, const N: usize, const PACKED_LEN: usize>
    TriangularPackedStorage<T, N, PACKED_LEN>
where
    Const<N>: Dim,
{
    /// Projects one triangle of a dense $N \times N$ matrix into packed
    /// triangular storage.
    ///
    /// `uplo` and `diag` name the stored part. When `diag` is [`Diag::Unit`],
    /// diagonal slots stay implicit (`T::ONE`) and are not copied from
    /// `dense`.
    ///
    /// # Errors
    /// This projection is total with respect to shape; it returns [`Ok`].
    pub fn from_dense_triangle(
        dense: &ArrayStorage<T, N, N>,
        uplo: UpLo,
        diag: Diag,
    ) -> StorageResult<Self> {
        let mut data: [T; PACKED_LEN] = core::array::from_fn(|_| T::ZERO);
        copy_dense_triangle(dense, uplo, &mut data, diag == Diag::Unit);
        Ok(Self::new(data, uplo, diag))
    }
}

#[allow(clippy::arithmetic_side_effects, clippy::indexing_slicing)]
unsafe impl<T: Scalar, const N: usize, const PACKED_LEN: usize> PackedStorage<T>
    for TriangularPackedStorage<T, N, PACKED_LEN>
where
    Const<N>: Dim,
{
    type N = Const<N>;

    fn uplo(&self) -> UpLo {
        self.uplo
    }

    fn as_slice(&self) -> &[T] {
        &self.data
    }

    fn packed_index(&self, i: usize, j: usize) -> Option<usize> {
        if i >= N || j >= N {
            return None;
        }
        match self.uplo {
            UpLo::Upper => {
                if i <= j {
                    Some(i + (j * (j + 1)) / 2)
                } else {
                    None
                }
            }
            UpLo::Lower => {
                if i >= j {
                    Some(i - j + (j * (2 * N - j + 1)) / 2)
                } else {
                    None
                }
            }
        }
    }

    fn packed_index_unchecked(&self, i: usize, j: usize) -> usize {
        match self.uplo {
            UpLo::Upper => i + (j * (j + 1)) / 2,
            UpLo::Lower => i - j + (j * (2 * N - j + 1)) / 2,
        }
    }

    fn value(&self, i: usize, j: usize) -> Option<T> {
        if i < N && j < N {
            Some(self.value_unchecked(i, j))
        } else {
            None
        }
    }

    fn value_unchecked(&self, i: usize, j: usize) -> T {
        if self.diag == Diag::Unit && i == j {
            return T::ONE;
        }
        match self.uplo {
            UpLo::Upper => {
                if i <= j {
                    let idx = i + (j * (j + 1)) / 2;
                    self.data[idx].clone()
                } else {
                    T::ZERO
                }
            }
            UpLo::Lower => {
                if i >= j {
                    let idx = i - j + (j * (2 * N - j + 1)) / 2;
                    self.data[idx].clone()
                } else {
                    T::ZERO
                }
            }
        }
    }
}

#[allow(clippy::indexing_slicing)]
unsafe impl<T: Scalar, const N: usize, const PACKED_LEN: usize>
    PackedStorageMut<T> for TriangularPackedStorage<T, N, PACKED_LEN>
where
    Const<N>: Dim,
{
    fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    fn set(&mut self, i: usize, j: usize, val: T) -> StorageResult<()> {
        if i >= N || j >= N {
            return Err(StorageError::OutOfBounds);
        }
        if self.diag == Diag::Unit && i == j {
            return Err(StorageError::ImmutableUnitDiagonal);
        }
        if let Some(idx) = self.packed_index(i, j) {
            self.data[idx] = val;
            Ok(())
        } else {
            Err(StorageError::InvalidStructuralInvariant)
        }
    }

    unsafe fn set_unchecked(&mut self, i: usize, j: usize, val: T) {
        let idx = self.packed_index_unchecked(i, j);
        self.data[idx] = val;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Non-Owning Packed Structured Views
////////////////////////////////////////////////////////////////////////////////

/// Non-owning zero-copy view of a diagonal matrix.
#[derive(Debug)]
pub struct DiagonalView<'a, T, const N: usize> {
    data: &'a [T],
}

impl<'a, T, const N: usize> DiagonalView<'a, T, N> {
    /// Creates a diagonal view from a slice of length $N$.
    ///
    /// # Errors
    /// Returns [`ConversionError::DimensionMismatch`] if `data.len() != N`.
    pub const fn new(data: &'a [T]) -> ConversionResult<Self> {
        if data.len() == N {
            Ok(Self { data })
        } else {
            Err(ConversionError::DimensionMismatch)
        }
    }
}

unsafe impl<T: Scalar, const N: usize> PackedStorage<T>
    for DiagonalView<'_, T, N>
where
    Const<N>: Dim,
{
    type N = Const<N>;

    fn uplo(&self) -> UpLo {
        UpLo::Upper
    }

    fn as_slice(&self) -> &[T] {
        self.data
    }

    fn packed_index(&self, i: usize, j: usize) -> Option<usize> {
        if i == j && i < N { Some(i) } else { None }
    }

    fn packed_index_unchecked(&self, i: usize, _j: usize) -> usize {
        i
    }

    fn value(&self, i: usize, j: usize) -> Option<T> {
        if i < N && j < N {
            Some(self.value_unchecked(i, j))
        } else {
            None
        }
    }

    #[allow(clippy::indexing_slicing)]
    fn value_unchecked(&self, i: usize, j: usize) -> T {
        if i == j {
            self.data[i].clone()
        } else {
            T::ZERO
        }
    }
}

/// Mutable non-owning zero-copy view of a diagonal matrix.
#[derive(Debug)]
pub struct DiagonalViewMut<'a, T, const N: usize> {
    data: &'a mut [T],
}

impl<'a, T, const N: usize> DiagonalViewMut<'a, T, N> {
    /// Creates a mutable diagonal view from a slice of length $N$.
    ///
    /// # Errors
    /// Returns [`ConversionError::DimensionMismatch`] if `data.len() != N`.
    pub const fn new(data: &'a mut [T]) -> ConversionResult<Self> {
        if data.len() == N {
            Ok(Self { data })
        } else {
            Err(ConversionError::DimensionMismatch)
        }
    }
}

unsafe impl<T: Scalar, const N: usize> PackedStorage<T>
    for DiagonalViewMut<'_, T, N>
where
    Const<N>: Dim,
{
    type N = Const<N>;

    fn uplo(&self) -> UpLo {
        UpLo::Upper
    }

    fn as_slice(&self) -> &[T] {
        self.data
    }

    fn packed_index(&self, i: usize, j: usize) -> Option<usize> {
        if i == j && i < N { Some(i) } else { None }
    }

    fn packed_index_unchecked(&self, i: usize, _j: usize) -> usize {
        i
    }

    fn value(&self, i: usize, j: usize) -> Option<T> {
        if i < N && j < N {
            Some(self.value_unchecked(i, j))
        } else {
            None
        }
    }

    #[allow(clippy::indexing_slicing)]
    fn value_unchecked(&self, i: usize, j: usize) -> T {
        if i == j {
            self.data[i].clone()
        } else {
            T::ZERO
        }
    }
}

unsafe impl<T: Scalar, const N: usize> PackedStorageMut<T>
    for DiagonalViewMut<'_, T, N>
where
    Const<N>: Dim,
{
    fn as_mut_slice(&mut self) -> &mut [T] {
        self.data
    }

    #[allow(clippy::indexing_slicing)]
    fn set(&mut self, i: usize, j: usize, val: T) -> StorageResult<()> {
        if i >= N || j >= N {
            return Err(StorageError::OutOfBounds);
        }
        if i != j {
            return Err(StorageError::InvalidStructuralInvariant);
        }
        self.data[i] = val;
        Ok(())
    }

    #[allow(clippy::indexing_slicing)]
    unsafe fn set_unchecked(&mut self, i: usize, _j: usize, val: T) {
        self.data[i] = val;
    }
}

/// Non-owning view of symmetric packed storage for $N \times N$ matrices ($N(N+1)/2$ elements).
#[derive(Debug)]
pub struct SymmetricPackedView<'a, T, const N: usize> {
    data: &'a [T],
    uplo: UpLo,
}

impl<'a, T, const N: usize> SymmetricPackedView<'a, T, N> {
    /// Creates a symmetric packed view from a slice of length $N(N+1)/2$.
    ///
    /// # Errors
    /// Returns [`ConversionError::DimensionMismatch`] if `data.len() != N * (N + 1) / 2`.
    pub const fn new(data: &'a [T], uplo: UpLo) -> ConversionResult<Self> {
        if data.len() == (N * (N + 1)) / 2 {
            Ok(Self { data, uplo })
        } else {
            Err(ConversionError::DimensionMismatch)
        }
    }
}

#[allow(clippy::arithmetic_side_effects, clippy::indexing_slicing)]
unsafe impl<T: Scalar, const N: usize> PackedStorage<T>
    for SymmetricPackedView<'_, T, N>
where
    Const<N>: Dim,
{
    type N = Const<N>;

    fn uplo(&self) -> UpLo {
        self.uplo
    }

    fn as_slice(&self) -> &[T] {
        self.data
    }

    fn packed_index(&self, i: usize, j: usize) -> Option<usize> {
        if i >= N || j >= N {
            return None;
        }
        match self.uplo {
            UpLo::Upper => {
                if i <= j {
                    Some(i + (j * (j + 1)) / 2)
                } else {
                    None
                }
            }
            UpLo::Lower => {
                if i >= j {
                    Some(i - j + (j * (2 * N - j + 1)) / 2)
                } else {
                    None
                }
            }
        }
    }

    fn packed_index_unchecked(&self, i: usize, j: usize) -> usize {
        match self.uplo {
            UpLo::Upper => i + (j * (j + 1)) / 2,
            UpLo::Lower => i - j + (j * (2 * N - j + 1)) / 2,
        }
    }

    fn value(&self, i: usize, j: usize) -> Option<T> {
        if i < N && j < N {
            Some(self.value_unchecked(i, j))
        } else {
            None
        }
    }

    fn value_unchecked(&self, i: usize, j: usize) -> T {
        match self.uplo {
            UpLo::Upper => {
                let (r, c) = if i <= j { (i, j) } else { (j, i) };
                let idx = r + (c * (c + 1)) / 2;
                self.data[idx].clone()
            }
            UpLo::Lower => {
                let (r, c) = if i >= j { (i, j) } else { (j, i) };
                let idx = r - c + (c * (2 * N - c + 1)) / 2;
                self.data[idx].clone()
            }
        }
    }
}

/// Mutable non-owning view of symmetric packed storage for $N \times N$ matrices ($N(N+1)/2$ elements).
#[derive(Debug)]
pub struct SymmetricPackedViewMut<'a, T, const N: usize> {
    data: &'a mut [T],
    uplo: UpLo,
}

impl<'a, T, const N: usize> SymmetricPackedViewMut<'a, T, N> {
    /// Creates a mutable symmetric packed view from a slice of length $N(N+1)/2$.
    ///
    /// # Errors
    /// Returns [`ConversionError::DimensionMismatch`] if `data.len() != N * (N + 1) / 2`.
    pub const fn new(data: &'a mut [T], uplo: UpLo) -> ConversionResult<Self> {
        if data.len() == (N * (N + 1)) / 2 {
            Ok(Self { data, uplo })
        } else {
            Err(ConversionError::DimensionMismatch)
        }
    }
}

#[allow(clippy::arithmetic_side_effects, clippy::indexing_slicing)]
unsafe impl<T: Scalar, const N: usize> PackedStorage<T>
    for SymmetricPackedViewMut<'_, T, N>
where
    Const<N>: Dim,
{
    type N = Const<N>;

    fn uplo(&self) -> UpLo {
        self.uplo
    }

    fn as_slice(&self) -> &[T] {
        self.data
    }

    fn packed_index(&self, i: usize, j: usize) -> Option<usize> {
        if i >= N || j >= N {
            return None;
        }
        match self.uplo {
            UpLo::Upper => {
                if i <= j {
                    Some(i + (j * (j + 1)) / 2)
                } else {
                    None
                }
            }
            UpLo::Lower => {
                if i >= j {
                    Some(i - j + (j * (2 * N - j + 1)) / 2)
                } else {
                    None
                }
            }
        }
    }

    fn packed_index_unchecked(&self, i: usize, j: usize) -> usize {
        match self.uplo {
            UpLo::Upper => i + (j * (j + 1)) / 2,
            UpLo::Lower => i - j + (j * (2 * N - j + 1)) / 2,
        }
    }

    fn value(&self, i: usize, j: usize) -> Option<T> {
        if i < N && j < N {
            Some(self.value_unchecked(i, j))
        } else {
            None
        }
    }

    fn value_unchecked(&self, i: usize, j: usize) -> T {
        match self.uplo {
            UpLo::Upper => {
                let (r, c) = if i <= j { (i, j) } else { (j, i) };
                let idx = r + (c * (c + 1)) / 2;
                self.data[idx].clone()
            }
            UpLo::Lower => {
                let (r, c) = if i >= j { (i, j) } else { (j, i) };
                let idx = r - c + (c * (2 * N - c + 1)) / 2;
                self.data[idx].clone()
            }
        }
    }
}

#[allow(clippy::indexing_slicing)]
unsafe impl<T: Scalar, const N: usize> PackedStorageMut<T>
    for SymmetricPackedViewMut<'_, T, N>
where
    Const<N>: Dim,
{
    fn as_mut_slice(&mut self) -> &mut [T] {
        self.data
    }

    fn set(&mut self, i: usize, j: usize, val: T) -> StorageResult<()> {
        if let Some(idx) = self.packed_index(i, j) {
            self.data[idx] = val;
            Ok(())
        } else if i < N && j < N {
            Err(StorageError::InvalidStructuralInvariant)
        } else {
            Err(StorageError::OutOfBounds)
        }
    }

    unsafe fn set_unchecked(&mut self, i: usize, j: usize, val: T) {
        let idx = self.packed_index_unchecked(i, j);
        self.data[idx] = val;
    }
}

/// Non-owning view of Hermitian packed storage for $N \times N$ matrices ($N(N+1)/2$ elements).
#[derive(Debug)]
pub struct HermitianPackedView<'a, T, const N: usize> {
    data: &'a [T],
    uplo: UpLo,
}

impl<'a, T, const N: usize> HermitianPackedView<'a, T, N> {
    /// Creates a Hermitian packed view from a slice of length $N(N+1)/2$.
    ///
    /// # Errors
    /// Returns [`ConversionError::DimensionMismatch`] if `data.len() != N * (N + 1) / 2`.
    pub const fn new(data: &'a [T], uplo: UpLo) -> ConversionResult<Self> {
        if data.len() == (N * (N + 1)) / 2 {
            Ok(Self { data, uplo })
        } else {
            Err(ConversionError::DimensionMismatch)
        }
    }
}

#[allow(clippy::arithmetic_side_effects, clippy::indexing_slicing)]
unsafe impl<T: Scalar, const N: usize> PackedStorage<T>
    for HermitianPackedView<'_, T, N>
where
    Const<N>: Dim,
{
    type N = Const<N>;

    fn uplo(&self) -> UpLo {
        self.uplo
    }

    fn as_slice(&self) -> &[T] {
        self.data
    }

    fn packed_index(&self, i: usize, j: usize) -> Option<usize> {
        if i >= N || j >= N {
            return None;
        }
        match self.uplo {
            UpLo::Upper => {
                if i <= j {
                    Some(i + (j * (j + 1)) / 2)
                } else {
                    None
                }
            }
            UpLo::Lower => {
                if i >= j {
                    Some(i - j + (j * (2 * N - j + 1)) / 2)
                } else {
                    None
                }
            }
        }
    }

    fn packed_index_unchecked(&self, i: usize, j: usize) -> usize {
        match self.uplo {
            UpLo::Upper => i + (j * (j + 1)) / 2,
            UpLo::Lower => i - j + (j * (2 * N - j + 1)) / 2,
        }
    }

    fn value(&self, i: usize, j: usize) -> Option<T> {
        if i < N && j < N {
            Some(self.value_unchecked(i, j))
        } else {
            None
        }
    }

    fn value_unchecked(&self, i: usize, j: usize) -> T {
        match self.uplo {
            UpLo::Upper => {
                if i <= j {
                    let idx = i + (j * (j + 1)) / 2;
                    self.data[idx].clone()
                } else {
                    let idx = j + (i * (i + 1)) / 2;
                    self.data[idx].clone().conj()
                }
            }
            UpLo::Lower => {
                if i >= j {
                    let idx = i - j + (j * (2 * N - j + 1)) / 2;
                    self.data[idx].clone()
                } else {
                    let idx = j - i + (i * (2 * N - i + 1)) / 2;
                    self.data[idx].clone().conj()
                }
            }
        }
    }
}

/// Mutable non-owning view of Hermitian packed storage for $N \times N$ matrices ($N(N+1)/2$ elements).
#[derive(Debug)]
pub struct HermitianPackedViewMut<'a, T, const N: usize> {
    data: &'a mut [T],
    uplo: UpLo,
}

impl<'a, T, const N: usize> HermitianPackedViewMut<'a, T, N> {
    /// Creates a mutable Hermitian packed view from a slice of length $N(N+1)/2$.
    ///
    /// # Errors
    /// Returns [`ConversionError::DimensionMismatch`] if `data.len() != N * (N + 1) / 2`.
    pub const fn new(data: &'a mut [T], uplo: UpLo) -> ConversionResult<Self> {
        if data.len() == (N * (N + 1)) / 2 {
            Ok(Self { data, uplo })
        } else {
            Err(ConversionError::DimensionMismatch)
        }
    }
}

#[allow(clippy::arithmetic_side_effects, clippy::indexing_slicing)]
unsafe impl<T: Scalar, const N: usize> PackedStorage<T>
    for HermitianPackedViewMut<'_, T, N>
where
    Const<N>: Dim,
{
    type N = Const<N>;

    fn uplo(&self) -> UpLo {
        self.uplo
    }

    fn as_slice(&self) -> &[T] {
        self.data
    }

    fn packed_index(&self, i: usize, j: usize) -> Option<usize> {
        if i >= N || j >= N {
            return None;
        }
        match self.uplo {
            UpLo::Upper => {
                if i <= j {
                    Some(i + (j * (j + 1)) / 2)
                } else {
                    None
                }
            }
            UpLo::Lower => {
                if i >= j {
                    Some(i - j + (j * (2 * N - j + 1)) / 2)
                } else {
                    None
                }
            }
        }
    }

    fn packed_index_unchecked(&self, i: usize, j: usize) -> usize {
        match self.uplo {
            UpLo::Upper => i + (j * (j + 1)) / 2,
            UpLo::Lower => i - j + (j * (2 * N - j + 1)) / 2,
        }
    }

    fn value(&self, i: usize, j: usize) -> Option<T> {
        if i < N && j < N {
            Some(self.value_unchecked(i, j))
        } else {
            None
        }
    }

    fn value_unchecked(&self, i: usize, j: usize) -> T {
        match self.uplo {
            UpLo::Upper => {
                if i <= j {
                    let idx = i + (j * (j + 1)) / 2;
                    self.data[idx].clone()
                } else {
                    let idx = j + (i * (i + 1)) / 2;
                    self.data[idx].clone().conj()
                }
            }
            UpLo::Lower => {
                if i >= j {
                    let idx = i - j + (j * (2 * N - j + 1)) / 2;
                    self.data[idx].clone()
                } else {
                    let idx = j - i + (i * (2 * N - i + 1)) / 2;
                    self.data[idx].clone().conj()
                }
            }
        }
    }
}

#[allow(clippy::indexing_slicing)]
unsafe impl<T: Scalar, const N: usize> PackedStorageMut<T>
    for HermitianPackedViewMut<'_, T, N>
where
    Const<N>: Dim,
{
    fn as_mut_slice(&mut self) -> &mut [T] {
        self.data
    }

    fn set(&mut self, i: usize, j: usize, val: T) -> StorageResult<()> {
        if i >= N || j >= N {
            return Err(StorageError::OutOfBounds);
        }
        if i == j && val.clone() != val.clone().conj() {
            return Err(StorageError::InvalidHermitianDiagonal);
        }
        if let Some(idx) = self.packed_index(i, j) {
            self.data[idx] = val;
            Ok(())
        } else {
            Err(StorageError::InvalidStructuralInvariant)
        }
    }

    unsafe fn set_unchecked(&mut self, i: usize, j: usize, val: T) {
        let idx = self.packed_index_unchecked(i, j);
        self.data[idx] = val;
    }
}

/// Non-owning view of triangular packed storage for $N \times N$ matrices ($N(N+1)/2$ elements).
#[derive(Debug)]
pub struct TriangularPackedView<'a, T, const N: usize> {
    data: &'a [T],
    uplo: UpLo,
    diag: Diag,
}

impl<'a, T, const N: usize> TriangularPackedView<'a, T, N> {
    /// Creates a triangular packed view from a slice of length $N(N+1)/2$.
    ///
    /// # Errors
    /// Returns [`ConversionError::DimensionMismatch`] if `data.len() != N * (N + 1) / 2`.
    pub const fn new(
        data: &'a [T],
        uplo: UpLo,
        diag: Diag,
    ) -> ConversionResult<Self> {
        if data.len() == (N * (N + 1)) / 2 {
            Ok(Self { data, uplo, diag })
        } else {
            Err(ConversionError::DimensionMismatch)
        }
    }
}

#[allow(clippy::arithmetic_side_effects, clippy::indexing_slicing)]
unsafe impl<T: Scalar, const N: usize> PackedStorage<T>
    for TriangularPackedView<'_, T, N>
where
    Const<N>: Dim,
{
    type N = Const<N>;

    fn uplo(&self) -> UpLo {
        self.uplo
    }

    fn as_slice(&self) -> &[T] {
        self.data
    }

    fn packed_index(&self, i: usize, j: usize) -> Option<usize> {
        if i >= N || j >= N {
            return None;
        }
        match self.uplo {
            UpLo::Upper => {
                if i <= j {
                    Some(i + (j * (j + 1)) / 2)
                } else {
                    None
                }
            }
            UpLo::Lower => {
                if i >= j {
                    Some(i - j + (j * (2 * N - j + 1)) / 2)
                } else {
                    None
                }
            }
        }
    }

    fn packed_index_unchecked(&self, i: usize, j: usize) -> usize {
        match self.uplo {
            UpLo::Upper => i + (j * (j + 1)) / 2,
            UpLo::Lower => i - j + (j * (2 * N - j + 1)) / 2,
        }
    }

    fn value(&self, i: usize, j: usize) -> Option<T> {
        if i < N && j < N {
            Some(self.value_unchecked(i, j))
        } else {
            None
        }
    }

    fn value_unchecked(&self, i: usize, j: usize) -> T {
        if self.diag == Diag::Unit && i == j {
            return T::ONE;
        }
        match self.uplo {
            UpLo::Upper => {
                if i <= j {
                    let idx = i + (j * (j + 1)) / 2;
                    self.data[idx].clone()
                } else {
                    T::ZERO
                }
            }
            UpLo::Lower => {
                if i >= j {
                    let idx = i - j + (j * (2 * N - j + 1)) / 2;
                    self.data[idx].clone()
                } else {
                    T::ZERO
                }
            }
        }
    }
}

/// Mutable non-owning view of triangular packed storage for $N \times N$ matrices ($N(N+1)/2$ elements).
#[derive(Debug)]
pub struct TriangularPackedViewMut<'a, T, const N: usize> {
    data: &'a mut [T],
    uplo: UpLo,
    diag: Diag,
}

impl<'a, T, const N: usize> TriangularPackedViewMut<'a, T, N> {
    /// Creates a mutable triangular packed view from a slice of length $N(N+1)/2$.
    ///
    /// # Errors
    /// Returns [`ConversionError::DimensionMismatch`] if `data.len() != N * (N + 1) / 2`.
    pub const fn new(
        data: &'a mut [T],
        uplo: UpLo,
        diag: Diag,
    ) -> ConversionResult<Self> {
        if data.len() == (N * (N + 1)) / 2 {
            Ok(Self { data, uplo, diag })
        } else {
            Err(ConversionError::DimensionMismatch)
        }
    }
}

#[allow(clippy::arithmetic_side_effects, clippy::indexing_slicing)]
unsafe impl<T: Scalar, const N: usize> PackedStorage<T>
    for TriangularPackedViewMut<'_, T, N>
where
    Const<N>: Dim,
{
    type N = Const<N>;

    fn uplo(&self) -> UpLo {
        self.uplo
    }

    fn as_slice(&self) -> &[T] {
        self.data
    }

    fn packed_index(&self, i: usize, j: usize) -> Option<usize> {
        if i >= N || j >= N {
            return None;
        }
        match self.uplo {
            UpLo::Upper => {
                if i <= j {
                    Some(i + (j * (j + 1)) / 2)
                } else {
                    None
                }
            }
            UpLo::Lower => {
                if i >= j {
                    Some(i - j + (j * (2 * N - j + 1)) / 2)
                } else {
                    None
                }
            }
        }
    }

    fn packed_index_unchecked(&self, i: usize, j: usize) -> usize {
        match self.uplo {
            UpLo::Upper => i + (j * (j + 1)) / 2,
            UpLo::Lower => i - j + (j * (2 * N - j + 1)) / 2,
        }
    }

    fn value(&self, i: usize, j: usize) -> Option<T> {
        if i < N && j < N {
            Some(self.value_unchecked(i, j))
        } else {
            None
        }
    }

    fn value_unchecked(&self, i: usize, j: usize) -> T {
        if self.diag == Diag::Unit && i == j {
            return T::ONE;
        }
        match self.uplo {
            UpLo::Upper => {
                if i <= j {
                    let idx = i + (j * (j + 1)) / 2;
                    self.data[idx].clone()
                } else {
                    T::ZERO
                }
            }
            UpLo::Lower => {
                if i >= j {
                    let idx = i - j + (j * (2 * N - j + 1)) / 2;
                    self.data[idx].clone()
                } else {
                    T::ZERO
                }
            }
        }
    }
}

#[allow(clippy::indexing_slicing)]
unsafe impl<T: Scalar, const N: usize> PackedStorageMut<T>
    for TriangularPackedViewMut<'_, T, N>
where
    Const<N>: Dim,
{
    fn as_mut_slice(&mut self) -> &mut [T] {
        self.data
    }

    fn set(&mut self, i: usize, j: usize, val: T) -> StorageResult<()> {
        if i >= N || j >= N {
            return Err(StorageError::OutOfBounds);
        }
        if self.diag == Diag::Unit && i == j {
            return Err(StorageError::ImmutableUnitDiagonal);
        }
        if let Some(idx) = self.packed_index(i, j) {
            self.data[idx] = val;
            Ok(())
        } else {
            Err(StorageError::InvalidStructuralInvariant)
        }
    }

    unsafe fn set_unchecked(&mut self, i: usize, j: usize, val: T) {
        let idx = self.packed_index_unchecked(i, j);
        self.data[idx] = val;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Sparse Storage & Sparse Vector Hierarchy
////////////////////////////////////////////////////////////////////////////////

/// Read access to 2-D sparse matrix formats.
///
/// # Safety
/// Implementors must ensure `rows()`, `cols()`, and `nnz()` report accurate dimensions
/// and stored entries, and that non-zero slots represent valid memory.
pub unsafe trait SparseStorage<T> {
    /// Associated compile-time row dimension.
    type R: Dim;
    /// Associated compile-time column dimension.
    type C: Dim;

    /// Number of logical rows (`Self::R::USIZE`).
    fn rows(&self) -> usize {
        Self::R::USIZE
    }

    /// Number of logical columns (`Self::C::USIZE`).
    fn cols(&self) -> usize {
        Self::C::USIZE
    }

    /// Number of stored non-zero entries.
    fn nnz(&self) -> usize;

    /// Returns the element at `(r, c)` if present, or `Some(T::ZERO)` / `None` if out of bounds.
    fn get(&self, r: usize, c: usize) -> Option<T>;
}

/// Mutable access to existing non-zero values of a sparse matrix.
///
/// # Safety
/// Implementors must ensure `values_mut()` accurately references the allocated
/// non-zero values and that `set_unchecked` writes only to valid memory.
pub unsafe trait SparseStorageMut<T>: SparseStorage<T> {
    /// Returns a mutable slice of all stored non-zero values.
    fn values_mut(&mut self) -> &mut [T];

    /// Returns a mutable reference to the non-zero value at `(r, c)` if allocated.
    fn get_mut(&mut self, r: usize, c: usize) -> Option<&mut T>;

    /// Updates the existing non-zero value at `(r, c)`.
    ///
    /// # Errors
    /// Returns [`StorageError::OutOfBounds`] if either index is out of bounds,
    /// or [`StorageError::InvalidStructuralInvariant`] if attempting to modify
    /// a structural zero entry that is not stored.
    fn set(&mut self, r: usize, c: usize, val: T) -> StorageResult<()>;

    /// Updates the non-zero value at `(r, c)` without bounds checking.
    ///
    /// # Safety
    /// `(r, c)` must correspond to an allocated non-zero slot.
    unsafe fn set_unchecked(&mut self, r: usize, c: usize, val: T);
}

/// Compressed Sparse Row (CSR) storage contract.
///
/// # Safety
/// Implementors must ensure `row_offsets` has length `R + 1`, `col_indices` and `values`
/// have length `nnz`, and column indices are sorted and valid for each row.
pub unsafe trait CsrStorage<T>: SparseStorage<T> {
    /// Array of row offsets of length $R + 1$.
    fn row_offsets(&self) -> &[usize];

    /// Array of column indices of length $\text{nnz}$.
    fn col_indices(&self) -> &[usize];

    /// Array of non-zero values of length $\text{nnz}$.
    fn values(&self) -> &[T];

    /// Slices the column indices and non-zero values for row `r`.
    #[allow(clippy::type_complexity)]
    fn row_slice(&self, r: usize) -> Option<SlicePair<'_, T>> {
        if r < self.rows() {
            Some(self.row_slice_unchecked(r))
        } else {
            None
        }
    }

    /// Slices the column indices and values for row `r` without bounds checking.
    #[allow(clippy::indexing_slicing)]
    fn row_slice_unchecked(&self, r: usize) -> SlicePair<'_, T> {
        let offsets = self.row_offsets();
        let start = offsets[r];
        let end = offsets[r + 1];
        (&self.col_indices()[start..end], &self.values()[start..end])
    }
}

/// Compressed Sparse Column (CSC) storage contract.
///
/// # Safety
/// Implementors must ensure `col_offsets` has length `C + 1`, `row_indices` and `values`
/// have length `nnz`, and row indices are sorted and valid for each column.
pub unsafe trait CscStorage<T>: SparseStorage<T> {
    /// Array of column offsets of length $C + 1$.
    fn col_offsets(&self) -> &[usize];

    /// Array of row indices of length $\text{nnz}$.
    fn row_indices(&self) -> &[usize];

    /// Array of non-zero values of length $\text{nnz}$.
    fn values(&self) -> &[T];
}

/// 1-D sparse vector abstraction.
///
/// # Safety
/// Implementors must ensure `indices` and `values` have length `nnz` and indices are
/// strictly within `0..N`.
pub unsafe trait SparseVectorStorage<T> {
    /// Associated compile-time vector dimension.
    type N: Dim;

    /// Logical length $N$ of the vector (`Self::N::USIZE`).
    fn len(&self) -> usize {
        Self::N::USIZE
    }

    /// Number of stored non-zero elements.
    fn nnz(&self) -> usize;

    /// Slices the non-zero indices.
    fn indices(&self) -> &[usize];

    /// Slices the non-zero values.
    fn values(&self) -> &[T];

    /// Returns `true` if vector contains no non-zero elements.
    fn is_empty(&self) -> bool {
        self.nnz() == 0
    }
}

/// Fixed-capacity stack-allocated Coordinate list (COO) buffer for incremental assembly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArrayCooStorage<
    T,
    const R: usize,
    const C: usize,
    const MAX_NNZ: usize,
> {
    row_indices: [usize; MAX_NNZ],
    col_indices: [usize; MAX_NNZ],
    values: [T; MAX_NNZ],
    nnz: usize,
}

impl<T: Zero, const R: usize, const C: usize, const MAX_NNZ: usize>
    ArrayCooStorage<T, R, C, MAX_NNZ>
{
    /// Creates an empty COO storage buffer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            row_indices: [0; MAX_NNZ],
            col_indices: [0; MAX_NNZ],
            values: core::array::from_fn(|_| T::ZERO),
            nnz: 0,
        }
    }

    /// Pushes a new coordinate triplet `(r, c, val)` into the buffer.
    ///
    /// # Errors
    /// Returns [`StorageError::OutOfBounds`] if either index is out of bounds,
    /// or [`StorageError::CapacityExceeded`] if the buffer is at capacity.
    #[allow(clippy::indexing_slicing, clippy::arithmetic_side_effects)]
    pub fn push(&mut self, r: usize, c: usize, val: T) -> StorageResult<()> {
        if r >= R || c >= C {
            return Err(StorageError::OutOfBounds);
        }
        if self.nnz >= MAX_NNZ {
            return Err(StorageError::CapacityExceeded);
        }
        self.row_indices[self.nnz] = r;
        self.col_indices[self.nnz] = c;
        self.values[self.nnz] = val;
        self.nnz += 1;
        Ok(())
    }
}

impl<T: Zero, const R: usize, const C: usize, const MAX_NNZ: usize> Default
    for ArrayCooStorage<T, R, C, MAX_NNZ>
{
    fn default() -> Self {
        Self::new()
    }
}

unsafe impl<T: Scalar, const R: usize, const C: usize, const MAX_NNZ: usize>
    SparseStorage<T> for ArrayCooStorage<T, R, C, MAX_NNZ>
where
    Const<R>: Dim,
    Const<C>: Dim,
{
    type R = Const<R>;
    type C = Const<C>;

    fn nnz(&self) -> usize {
        self.nnz
    }

    #[allow(clippy::indexing_slicing)]
    fn get(&self, r: usize, c: usize) -> Option<T> {
        if r >= R || c >= C {
            return None;
        }
        let mut acc = T::ZERO;
        for i in 0..self.nnz {
            if self.row_indices[i] == r && self.col_indices[i] == c {
                acc = acc + self.values[i].clone();
            }
        }
        Some(acc)
    }
}

/// Fixed-capacity stack-allocated Compressed Sparse Row (CSR) storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArrayCsrStorage<
    T,
    const R: usize,
    const C: usize,
    const MAX_NNZ: usize,
    const R1: usize,
> {
    row_offsets: [usize; R1],
    col_indices: [usize; MAX_NNZ],
    values: [T; MAX_NNZ],
    nnz: usize,
}

impl<
    T: Scalar,
    const R: usize,
    const C: usize,
    const MAX_NNZ: usize,
    const R1: usize,
> ArrayCsrStorage<T, R, C, MAX_NNZ, R1>
{
    /// Constructs a canonical CSR matrix by sorting and accumulating an [`ArrayCooStorage`] buffer.
    ///
    /// # Errors
    /// Returns [`StorageError::OutOfBounds`] if any coordinate in `coo` is out of bounds.
    #[allow(
        clippy::indexing_slicing,
        clippy::arithmetic_side_effects,
        clippy::needless_range_loop,
        clippy::too_many_lines
    )]
    pub fn from_coo(
        coo: &ArrayCooStorage<T, R, C, MAX_NNZ>,
    ) -> StorageResult<Self> {
        debug_assert_eq!(R1, R + 1);

        // Pass 1: Histogram & row counts
        let mut row_counts = [0usize; R];
        for k in 0..coo.nnz {
            let r = coo.row_indices[k];
            let c = coo.col_indices[k];
            if r >= R || c >= C {
                return Err(StorageError::OutOfBounds);
            }
            row_counts[r] += 1;
        }

        // Prefix sums -> row_offsets
        let mut row_offsets = [0usize; R1];
        row_offsets[0] = 0;
        for i in 0..R {
            row_offsets[i + 1] = row_offsets[i] + row_counts[i];
        }

        // Pass 2: Bucket distribution
        let mut current_offsets = [0usize; R];
        current_offsets.copy_from_slice(&row_offsets[..R]);

        let mut temp_cols = [0usize; MAX_NNZ];
        let mut temp_vals: [T; MAX_NNZ] = core::array::from_fn(|_| T::ZERO);

        for k in 0..coo.nnz {
            let r = coo.row_indices[k];
            let dest = current_offsets[r];
            temp_cols[dest] = coo.col_indices[k];
            temp_vals[dest] = coo.values[k].clone();
            current_offsets[r] += 1;
        }

        // Pass 3: In-row sorting & duplicate accumulation
        let mut final_offsets = [0usize; R1];
        let mut final_cols = [0usize; MAX_NNZ];
        let mut final_vals: [T; MAX_NNZ] = core::array::from_fn(|_| T::ZERO);
        let mut write_idx = 0;

        for r in 0..R {
            final_offsets[r] = write_idx;
            let start = row_offsets[r];
            let end = row_offsets[r + 1];
            let len = end - start;

            if len > 0 {
                // Insertion sort by column index within row
                for i in start + 1..end {
                    let mut j = i;
                    while j > start && temp_cols[j - 1] > temp_cols[j] {
                        temp_cols.swap(j - 1, j);
                        temp_vals.swap(j - 1, j);
                        j -= 1;
                    }
                }

                // Accumulate duplicates
                let mut curr_col = temp_cols[start];
                let mut curr_val = temp_vals[start].clone();

                for i in start + 1..end {
                    let next_col = temp_cols[i];
                    let next_val = temp_vals[i].clone();
                    if next_col == curr_col {
                        curr_val = curr_val + next_val;
                    } else {
                        final_cols[write_idx] = curr_col;
                        final_vals[write_idx] = curr_val;
                        write_idx += 1;
                        curr_col = next_col;
                        curr_val = next_val;
                    }
                }
                final_cols[write_idx] = curr_col;
                final_vals[write_idx] = curr_val;
                write_idx += 1;
            }
        }
        final_offsets[R] = write_idx;

        Ok(Self {
            row_offsets: final_offsets,
            col_indices: final_cols,
            values: final_vals,
            nnz: write_idx,
        })
    }
}

unsafe impl<
    T: Scalar,
    const R: usize,
    const C: usize,
    const MAX_NNZ: usize,
    const R1: usize,
> SparseStorage<T> for ArrayCsrStorage<T, R, C, MAX_NNZ, R1>
where
    Const<R>: Dim,
    Const<C>: Dim,
{
    type R = Const<R>;
    type C = Const<C>;

    fn nnz(&self) -> usize {
        self.nnz
    }

    #[allow(clippy::indexing_slicing)]
    fn get(&self, r: usize, c: usize) -> Option<T> {
        if r >= R || c >= C {
            return None;
        }
        let (cols, vals) = self.row_slice(r)?;
        for (col, val) in cols.iter().zip(vals.iter()) {
            if *col == c {
                return Some(val.clone());
            }
        }
        Some(T::ZERO)
    }
}

unsafe impl<
    T: Scalar,
    const R: usize,
    const C: usize,
    const MAX_NNZ: usize,
    const R1: usize,
> CsrStorage<T> for ArrayCsrStorage<T, R, C, MAX_NNZ, R1>
where
    Const<R>: Dim,
    Const<C>: Dim,
{
    fn row_offsets(&self) -> &[usize] {
        &self.row_offsets
    }

    #[allow(clippy::indexing_slicing)]
    fn col_indices(&self) -> &[usize] {
        &self.col_indices[..self.nnz]
    }

    #[allow(clippy::indexing_slicing)]
    fn values(&self) -> &[T] {
        &self.values[..self.nnz]
    }
}

unsafe impl<
    T: Scalar,
    const R: usize,
    const C: usize,
    const MAX_NNZ: usize,
    const R1: usize,
> SparseStorageMut<T> for ArrayCsrStorage<T, R, C, MAX_NNZ, R1>
where
    Const<R>: Dim,
    Const<C>: Dim,
{
    #[allow(clippy::indexing_slicing)]
    fn values_mut(&mut self) -> &mut [T] {
        &mut self.values[..self.nnz]
    }

    #[allow(clippy::indexing_slicing)]
    fn get_mut(&mut self, r: usize, c: usize) -> Option<&mut T> {
        if r >= R || c >= C {
            return None;
        }
        let start = self.row_offsets[r];
        let end = self.row_offsets[r + 1];
        for idx in start..end {
            if self.col_indices[idx] == c {
                return Some(&mut self.values[idx]);
            }
        }
        None
    }

    fn set(&mut self, r: usize, c: usize, val: T) -> StorageResult<()> {
        self.get_mut(r, c).map_or_else(
            || {
                if r < R && c < C {
                    Err(StorageError::InvalidStructuralInvariant)
                } else {
                    Err(StorageError::OutOfBounds)
                }
            },
            |entry| {
                *entry = val;
                Ok(())
            },
        )
    }

    #[allow(clippy::indexing_slicing)]
    unsafe fn set_unchecked(&mut self, r: usize, c: usize, val: T) {
        let start = self.row_offsets[r];
        let end = self.row_offsets[r + 1];
        for idx in start..end {
            if self.col_indices[idx] == c {
                self.values[idx] = val;
                return;
            }
        }
    }
}

/// Fixed-capacity stack-allocated Compressed Sparse Column (CSC) storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArrayCscStorage<
    T,
    const R: usize,
    const C: usize,
    const MAX_NNZ: usize,
    const C1: usize,
> {
    col_offsets: [usize; C1],
    row_indices: [usize; MAX_NNZ],
    values: [T; MAX_NNZ],
    nnz: usize,
}

unsafe impl<
    T: Scalar,
    const R: usize,
    const C: usize,
    const MAX_NNZ: usize,
    const C1: usize,
> SparseStorage<T> for ArrayCscStorage<T, R, C, MAX_NNZ, C1>
where
    Const<R>: Dim,
    Const<C>: Dim,
{
    type R = Const<R>;
    type C = Const<C>;

    fn nnz(&self) -> usize {
        self.nnz
    }

    #[allow(clippy::indexing_slicing)]
    fn get(&self, r: usize, c: usize) -> Option<T> {
        if r >= R || c >= C {
            return None;
        }
        let start = self.col_offsets[c];
        let end = self.col_offsets[c + 1];
        for idx in start..end {
            if self.row_indices[idx] == r {
                return Some(self.values[idx].clone());
            }
        }
        Some(T::ZERO)
    }
}

unsafe impl<
    T: Scalar,
    const R: usize,
    const C: usize,
    const MAX_NNZ: usize,
    const C1: usize,
> CscStorage<T> for ArrayCscStorage<T, R, C, MAX_NNZ, C1>
where
    Const<R>: Dim,
    Const<C>: Dim,
{
    fn col_offsets(&self) -> &[usize] {
        &self.col_offsets
    }

    #[allow(clippy::indexing_slicing)]
    fn row_indices(&self) -> &[usize] {
        &self.row_indices[..self.nnz]
    }

    #[allow(clippy::indexing_slicing)]
    fn values(&self) -> &[T] {
        &self.values[..self.nnz]
    }
}

unsafe impl<
    T: Scalar,
    const R: usize,
    const C: usize,
    const MAX_NNZ: usize,
    const C1: usize,
> SparseStorageMut<T> for ArrayCscStorage<T, R, C, MAX_NNZ, C1>
where
    Const<R>: Dim,
    Const<C>: Dim,
{
    #[allow(clippy::indexing_slicing)]
    fn values_mut(&mut self) -> &mut [T] {
        &mut self.values[..self.nnz]
    }

    #[allow(clippy::indexing_slicing)]
    fn get_mut(&mut self, r: usize, c: usize) -> Option<&mut T> {
        if r >= R || c >= C {
            return None;
        }
        let start = self.col_offsets[c];
        let end = self.col_offsets[c + 1];
        for idx in start..end {
            if self.row_indices[idx] == r {
                return Some(&mut self.values[idx]);
            }
        }
        None
    }

    fn set(&mut self, r: usize, c: usize, val: T) -> StorageResult<()> {
        self.get_mut(r, c).map_or_else(
            || {
                if r < R && c < C {
                    Err(StorageError::InvalidStructuralInvariant)
                } else {
                    Err(StorageError::OutOfBounds)
                }
            },
            |entry| {
                *entry = val;
                Ok(())
            },
        )
    }

    #[allow(clippy::indexing_slicing)]
    unsafe fn set_unchecked(&mut self, r: usize, c: usize, val: T) {
        let start = self.col_offsets[c];
        let end = self.col_offsets[c + 1];
        for idx in start..end {
            if self.row_indices[idx] == r {
                self.values[idx] = val;
                return;
            }
        }
    }
}

/// Fixed-capacity stack-allocated 1-D sparse vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArraySparseVector<T, const N: usize, const MAX_NNZ: usize> {
    indices: [usize; MAX_NNZ],
    values: [T; MAX_NNZ],
    nnz: usize,
}

impl<T: Zero, const N: usize, const MAX_NNZ: usize>
    ArraySparseVector<T, N, MAX_NNZ>
{
    /// Creates an empty sparse vector.
    #[must_use]
    pub fn new() -> Self {
        Self {
            indices: [0; MAX_NNZ],
            values: core::array::from_fn(|_| T::ZERO),
            nnz: 0,
        }
    }

    /// Pushes a non-zero element `(idx, val)`.
    ///
    /// # Errors
    /// Returns [`StorageError::OutOfBounds`] if `idx >= N`,
    /// or [`StorageError::CapacityExceeded`] if the vector is at capacity.
    #[allow(clippy::indexing_slicing, clippy::arithmetic_side_effects)]
    pub fn push(&mut self, idx: usize, val: T) -> StorageResult<()> {
        if idx >= N {
            return Err(StorageError::OutOfBounds);
        }
        if self.nnz >= MAX_NNZ {
            return Err(StorageError::CapacityExceeded);
        }
        self.indices[self.nnz] = idx;
        self.values[self.nnz] = val;
        self.nnz += 1;
        Ok(())
    }
}

impl<T: Zero, const N: usize, const MAX_NNZ: usize> Default
    for ArraySparseVector<T, N, MAX_NNZ>
{
    fn default() -> Self {
        Self::new()
    }
}

unsafe impl<T: Scalar, const N: usize, const MAX_NNZ: usize>
    SparseVectorStorage<T> for ArraySparseVector<T, N, MAX_NNZ>
where
    Const<N>: Dim,
{
    type N = Const<N>;

    fn nnz(&self) -> usize {
        self.nnz
    }

    #[allow(clippy::indexing_slicing)]
    fn indices(&self) -> &[usize] {
        &self.indices[..self.nnz]
    }

    #[allow(clippy::indexing_slicing)]
    fn values(&self) -> &[T] {
        &self.values[..self.nnz]
    }
}

/// Zero-copy non-owning view of a 1-D sparse vector.
#[derive(Debug)]
pub struct ViewSparseVector<'a, T, const N: usize> {
    indices: &'a [usize],
    values: &'a [T],
}

impl<'a, T, const N: usize> ViewSparseVector<'a, T, N> {
    /// Creates a sparse vector view from parallel slices.
    ///
    /// # Errors
    /// Returns [`ConversionError::DimensionMismatch`] if `indices.len() != values.len()`.
    pub const fn new(
        indices: &'a [usize],
        values: &'a [T],
    ) -> ConversionResult<Self> {
        if indices.len() == values.len() {
            Ok(Self { indices, values })
        } else {
            Err(ConversionError::DimensionMismatch)
        }
    }
}

unsafe impl<T: Scalar, const N: usize> SparseVectorStorage<T>
    for ViewSparseVector<'_, T, N>
where
    Const<N>: Dim,
{
    type N = Const<N>;

    fn nnz(&self) -> usize {
        self.indices.len()
    }

    fn indices(&self) -> &[usize] {
        self.indices
    }

    fn values(&self) -> &[T] {
        self.values
    }
}

////////////////////////////////////////////////////////////////////////////////
// Layout Conversion Traits
////////////////////////////////////////////////////////////////////////////////

/// Copies the `uplo` triangle of `dense` into a packed buffer using the
/// packed-index map in `storage-design.md` §4.4.
#[allow(clippy::arithmetic_side_effects, clippy::indexing_slicing)]
fn copy_dense_triangle<T: Scalar, const N: usize, const PACKED_LEN: usize>(
    dense: &ArrayStorage<T, N, N>,
    uplo: UpLo,
    data: &mut [T; PACKED_LEN],
    skip_diagonal: bool,
) where
    Const<N>: Dim,
{
    for j in 0..N {
        for i in 0..N {
            let on_triangle = match uplo {
                UpLo::Upper => i <= j,
                UpLo::Lower => i >= j,
            };
            if !on_triangle || (skip_diagonal && i == j) {
                continue;
            }
            let idx = match uplo {
                UpLo::Upper => i + (j * (j + 1)) / 2,
                UpLo::Lower => i - j + (j * (2 * N - j + 1)) / 2,
            };
            data[idx] = unsafe { dense.get_unchecked(i, j).clone() };
        }
    }
}

/// Conversion to dense array storage.
pub trait ToDenseStorage<Dense> {
    /// Converts `self` to dense storage format.
    ///
    /// # Errors
    /// Returns [`StorageError`] if conversion fails due to dimension mismatch or capacity constraints.
    fn to_dense(&self) -> StorageResult<Dense>;
}

/// Conversion to Compressed Sparse Row (CSR) format.
pub trait ToCsrStorage<Csr> {
    /// Converts `self` into CSR format.
    ///
    /// # Errors
    /// Returns [`StorageError`] if the layout conversion fails.
    fn to_csr(&self) -> StorageResult<Csr>;
}

/// Conversion to Compressed Sparse Column (CSC) format.
pub trait ToCscStorage<Csc> {
    /// Converts `self` into CSC format.
    ///
    /// # Errors
    /// Returns [`StorageError`] if the layout conversion fails.
    fn to_csc(&self) -> StorageResult<Csc>;
}

impl<T: Scalar, const N: usize> ToDenseStorage<ArrayStorage<T, N, N>>
    for DiagonalStorage<T, N>
where
    Const<N>: Dim,
{
    fn to_dense(&self) -> StorageResult<ArrayStorage<T, N, N>> {
        Ok(ArrayStorage::from_fn(|i, j| self.value_unchecked(i, j)))
    }
}

impl<T: Scalar, const N: usize, const PACKED_LEN: usize>
    ToDenseStorage<ArrayStorage<T, N, N>>
    for SymmetricPackedStorage<T, N, PACKED_LEN>
where
    Const<N>: Dim,
{
    fn to_dense(&self) -> StorageResult<ArrayStorage<T, N, N>> {
        Ok(ArrayStorage::from_fn(|i, j| self.value_unchecked(i, j)))
    }
}

impl<T: Scalar, const N: usize, const PACKED_LEN: usize>
    ToDenseStorage<ArrayStorage<T, N, N>>
    for HermitianPackedStorage<T, N, PACKED_LEN>
where
    Const<N>: Dim,
{
    fn to_dense(&self) -> StorageResult<ArrayStorage<T, N, N>> {
        Ok(ArrayStorage::from_fn(|i, j| self.value_unchecked(i, j)))
    }
}

impl<T: Scalar, const N: usize, const PACKED_LEN: usize>
    ToDenseStorage<ArrayStorage<T, N, N>>
    for TriangularPackedStorage<T, N, PACKED_LEN>
where
    Const<N>: Dim,
{
    fn to_dense(&self) -> StorageResult<ArrayStorage<T, N, N>> {
        Ok(ArrayStorage::from_fn(|i, j| self.value_unchecked(i, j)))
    }
}

impl<T: Scalar, const R: usize, const C: usize, const MAX_NNZ: usize>
    ToDenseStorage<ArrayStorage<T, R, C>> for ArrayCooStorage<T, R, C, MAX_NNZ>
where
    Const<R>: Dim,
    Const<C>: Dim,
{
    fn to_dense(&self) -> StorageResult<ArrayStorage<T, R, C>> {
        let mut dense = ArrayStorage::zeros();
        for k in 0..self.nnz {
            let r = self.row_indices[k];
            let c = self.col_indices[k];
            let val = self.values[k].clone();
            let current: &T = unsafe { dense.get_unchecked(r, c) };
            unsafe {
                dense.set_unchecked(r, c, current.clone() + val);
            }
        }
        Ok(dense)
    }
}

impl<
    T: Scalar,
    const R: usize,
    const C: usize,
    const MAX_NNZ: usize,
    const R1: usize,
> ToDenseStorage<ArrayStorage<T, R, C>>
    for ArrayCsrStorage<T, R, C, MAX_NNZ, R1>
where
    Const<R>: Dim,
    Const<C>: Dim,
{
    fn to_dense(&self) -> StorageResult<ArrayStorage<T, R, C>> {
        let mut dense = ArrayStorage::zeros();
        for r in 0..R {
            let start = self.row_offsets[r];
            let end = self.row_offsets[r + 1];
            for k in start..end {
                let c = self.col_indices[k];
                let val = self.values[k].clone();
                unsafe {
                    dense.set_unchecked(r, c, val);
                }
            }
        }
        Ok(dense)
    }
}

impl<
    T: Scalar,
    const R: usize,
    const C: usize,
    const MAX_NNZ: usize,
    const C1: usize,
> ToDenseStorage<ArrayStorage<T, R, C>>
    for ArrayCscStorage<T, R, C, MAX_NNZ, C1>
where
    Const<R>: Dim,
    Const<C>: Dim,
{
    fn to_dense(&self) -> StorageResult<ArrayStorage<T, R, C>> {
        let mut dense = ArrayStorage::zeros();
        for c in 0..C {
            let start = self.col_offsets[c];
            let end = self.col_offsets[c + 1];
            for k in start..end {
                let r = self.row_indices[k];
                let val = self.values[k].clone();
                unsafe {
                    dense.set_unchecked(r, c, val);
                }
            }
        }
        Ok(dense)
    }
}

impl<
    T: Scalar,
    const R: usize,
    const C: usize,
    const MAX_NNZ: usize,
    const R1: usize,
> ToCsrStorage<ArrayCsrStorage<T, R, C, MAX_NNZ, R1>>
    for ArrayCooStorage<T, R, C, MAX_NNZ>
{
    /// Converts `self` into CSR format.
    ///
    /// # Errors
    /// Returns [`StorageError::OutOfBounds`] if any coordinate is out of bounds.
    fn to_csr(&self) -> StorageResult<ArrayCsrStorage<T, R, C, MAX_NNZ, R1>> {
        ArrayCsrStorage::from_coo(self)
    }
}

impl<
    T: Scalar,
    const R: usize,
    const C: usize,
    const MAX_NNZ: usize,
    const C1: usize,
> ArrayCscStorage<T, R, C, MAX_NNZ, C1>
{
    /// Constructs a canonical CSC matrix by sorting and accumulating an [`ArrayCooStorage`] buffer.
    ///
    /// # Errors
    /// Returns [`StorageError::OutOfBounds`] if any coordinate in `coo` is out of bounds.
    #[allow(
        clippy::indexing_slicing,
        clippy::arithmetic_side_effects,
        clippy::needless_range_loop,
        clippy::manual_memcpy,
        clippy::too_many_lines
    )]
    pub fn from_coo(
        coo: &ArrayCooStorage<T, R, C, MAX_NNZ>,
    ) -> StorageResult<Self> {
        debug_assert_eq!(C1, C + 1);

        // Pass 1: Histogram & column counts
        let mut col_counts = [0usize; C];
        for k in 0..coo.nnz {
            let r = coo.row_indices[k];
            let c = coo.col_indices[k];
            if r >= R || c >= C {
                return Err(StorageError::OutOfBounds);
            }
            col_counts[c] += 1;
        }

        // Prefix sums -> col_offsets
        let mut col_offsets = [0usize; C1];
        col_offsets[0] = 0;
        for j in 0..C {
            col_offsets[j + 1] = col_offsets[j] + col_counts[j];
        }

        // Pass 2: Bucket distribution
        let mut current_offsets = [0usize; C];
        current_offsets.copy_from_slice(&col_offsets[..C]);

        let mut temp_rows = [0usize; MAX_NNZ];
        let mut temp_vals: [T; MAX_NNZ] = core::array::from_fn(|_| T::ZERO);

        for k in 0..coo.nnz {
            let c = coo.col_indices[k];
            let dest = current_offsets[c];
            temp_rows[dest] = coo.row_indices[k];
            temp_vals[dest] = coo.values[k].clone();
            current_offsets[c] += 1;
        }

        // Pass 3: In-column sorting & duplicate accumulation
        let mut final_offsets = [0usize; C1];
        let mut final_rows = [0usize; MAX_NNZ];
        let mut final_vals: [T; MAX_NNZ] = core::array::from_fn(|_| T::ZERO);
        let mut write_idx = 0;

        for c in 0..C {
            final_offsets[c] = write_idx;
            let start = col_offsets[c];
            let end = col_offsets[c + 1];
            let len = end - start;

            if len > 0 {
                // Collect row indices and values for this column
                let mut col_rows = [0usize; MAX_NNZ];
                let mut col_vals: [T; MAX_NNZ] =
                    core::array::from_fn(|_| T::ZERO);
                for i in 0..len {
                    col_rows[i] = temp_rows[start + i];
                    col_vals[i] = temp_vals[start + i].clone();
                }

                // Simple insertion sort by row index
                for i in 1..len {
                    let key_row = col_rows[i];
                    let key_val = col_vals[i].clone();
                    let mut j = i;
                    while j > 0 && col_rows[j - 1] > key_row {
                        col_rows[j] = col_rows[j - 1];
                        col_vals[j] = col_vals[j - 1].clone();
                        j -= 1;
                    }
                    col_rows[j] = key_row;
                    col_vals[j] = key_val;
                }

                // Accumulate duplicates
                let mut curr_row = col_rows[0];
                let mut curr_val = col_vals[0].clone();

                for i in 1..len {
                    if col_rows[i] == curr_row {
                        curr_val = curr_val + col_vals[i].clone();
                    } else {
                        final_rows[write_idx] = curr_row;
                        final_vals[write_idx] = curr_val;
                        write_idx += 1;
                        curr_row = col_rows[i];
                        curr_val = col_vals[i].clone();
                    }
                }
                final_rows[write_idx] = curr_row;
                final_vals[write_idx] = curr_val;
                write_idx += 1;
            }
        }
        final_offsets[C] = write_idx;

        Ok(Self {
            col_offsets: final_offsets,
            row_indices: final_rows,
            values: final_vals,
            nnz: write_idx,
        })
    }
}

impl<
    T: Scalar,
    const R: usize,
    const C: usize,
    const MAX_NNZ: usize,
    const C1: usize,
> ToCscStorage<ArrayCscStorage<T, R, C, MAX_NNZ, C1>>
    for ArrayCooStorage<T, R, C, MAX_NNZ>
{
    /// Converts `self` into CSC format.
    ///
    /// # Errors
    /// Returns [`StorageError::OutOfBounds`] if any coordinate is out of bounds.
    fn to_csc(&self) -> StorageResult<ArrayCscStorage<T, R, C, MAX_NNZ, C1>> {
        ArrayCscStorage::from_coo(self)
    }
}

////////////////////////////////////////////////////////////////////////////////
// Scratch Pivot Buffer
////////////////////////////////////////////////////////////////////////////////

/// A fixed-capacity scratch buffer of row/column indices for pivoting algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PivotStorage<const N: usize> {
    indices: [usize; N],
}

impl<const N: usize> PivotStorage<N> {
    /// Builds the identity permutation `[0, 1, ..., N - 1]`.
    #[must_use]
    #[allow(clippy::indexing_slicing, clippy::arithmetic_side_effects)]
    pub const fn identity() -> Self {
        let mut indices = [0usize; N];
        let mut i = 0;
        while i < N {
            indices[i] = i;
            i += 1;
        }
        Self { indices }
    }

    /// Returns the current permutation as a slice.
    #[must_use]
    pub const fn as_slice(&self) -> &[usize] {
        &self.indices
    }

    /// Returns the current permutation as a mutable slice.
    pub const fn as_mut_slice(&mut self) -> &mut [usize] {
        &mut self.indices
    }

    /// Swaps two entries in the permutation.
    pub const fn swap(&mut self, a: usize, b: usize) {
        self.indices.swap(a, b);
    }
}
