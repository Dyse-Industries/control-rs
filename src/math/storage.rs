//! Storage backends for `R x C` grids of data, decoupled from dimension
//! checking.
//!
//! `Storage`/`StorageMut`/`ContiguousStorage`/`ContiguousStorageMut` tie the
//! compile-time `Dim` system ([`crate::math::num_types`]) to physical
//! memory. `offset()` is the single point of control for a backend's memory
//! layout; higher-level code accesses elements through it rather than
//! assuming a layout.
//!
//! Also home to array-initialization helpers inspired by
//! [array-init](https://github.com/Manishearth/array-init/) and
//! [nalgebra](https://docs.rs/nalgebra/latest/nalgebra/).
#![allow(clippy::arbitrary_source_item_ordering)]

use crate::math::num_traits::{One, Zero};
use crate::math::num_types::{Const, Dim};
use crate::math::{ConversionError, ConversionResult};
use core::marker::PhantomData;
use core::mem::MaybeUninit;

type UninitArray<T, const N: usize> = MaybeUninit<[T; N]>;

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
/// # Generic Arguments
/// * `I` - Any collection that implements [`IntoIterator<item=T>`].
/// * `T` - Field type of the array.
/// * `N` - Capacity of the array.
///
/// # Arguments
/// * `iterator` - Collection of `T`.
///
/// # Returns
/// * `initialized_array` - An array filled with elements from the iterator.
///
/// # Safety
/// * The iterator must have **at least** `N` elements, or this assumes an uninitialized
///   value is initialized (resulting in undefined behavior).
///
/// # Panics
/// * This function panics in debug builds if the safety criterion is not met.
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
// Core Storage Trait Hierarchy
////////////////////////////////////////////////////////////////////////////////

/// Memory layout convention for a contiguous 2-D storage backend.
///
/// Read off `ContiguousStorage::ORDER` by BLAS-style subprograms
/// (`GEMV`/`GEMM`/`AXPY`) rather than assumed, so swapping a backend's
/// default layout is a zero-call-site-change operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixLayout {
    /// Elements of a row are contiguous in memory.
    RowMajor,
    /// Elements of a column are contiguous in memory.
    ColMajor,
}

/// Read access to an `R x C` grid of `T`, decoupled from where the data
/// physically lives.
///
/// `C = U1` (or `Const<1>`) for 1-D containers such as polynomials.
/// Implementors define memory layout entirely through [`Storage::offset`];
/// every other access method is expressed in terms of it.
///
/// # Safety
/// `offset(i, j)` must return a value such that
/// `self.ptr().offset(self.offset(i, j))` is a valid, in-bounds pointer to
/// the element at logical position `(i, j)` for every `i < R::DIM` and
/// `j < C::DIM`.
pub unsafe trait Storage<T, R: Dim, C: Dim> {
    /// Returns a raw pointer to the backend's first addressable element.
    fn ptr(&self) -> *const T;
    /// Maps a logical `(row, column)` index to a physical element offset,
    /// in units of `T`, from [`Storage::ptr`].
    ///
    /// Returning `isize` rather than `usize` lets implementations encode
    /// reversed or non-standard layouts (e.g. a negative stride) without
    /// physically rearranging the data.
    fn offset(&self, i: usize, j: usize) -> isize;
    /// Returns a reference to the element at `(i, j)` without bounds
    /// checking.
    ///
    /// # Safety
    /// `i < self.rows()` and `j < self.cols()` must hold.
    unsafe fn get_unchecked(&self, i: usize, j: usize) -> &T;
    /// Returns a reference to the element at `(i, j)`, or `None` if either
    /// index is out of bounds.
    fn get(&self, i: usize, j: usize) -> Option<&T> {
        if i < R::DIM && j < C::DIM {
            // Safety: just checked `i < R::DIM` and `j < C::DIM` above.
            Some(unsafe { self.get_unchecked(i, j) })
        } else {
            None
        }
    }
    /// Number of logical rows.
    fn rows(&self) -> usize {
        R::DIM
    }
    /// Number of logical columns.
    fn cols(&self) -> usize {
        C::DIM
    }
}

/// Mutable access to an `R x C` grid of `T`.
///
/// Kept as an independent capability from [`ContiguousStorage`] (FR-4 of
/// `storage-trait-design.md`): a backend can be mutable without being
/// contiguous (e.g. a strided view) or contiguous without being mutable
/// (e.g. a `const`-baked Flash/ROM backend).
///
/// # Safety
/// `ptr_mut()` must return a pointer to the same backing memory `ptr()`
/// describes, with unique (exclusive) access for the lifetime of the
/// borrow.
pub unsafe trait StorageMut<T, R: Dim, C: Dim>:
    Storage<T, R, C>
{
    /// Returns a raw mutable pointer to the backend's first addressable
    /// element.
    fn ptr_mut(&mut self) -> *mut T;
    /// Returns a mutable reference to the element at `(i, j)` without
    /// bounds checking.
    ///
    /// # Safety
    /// `i < self.rows()` and `j < self.cols()` must hold.
    unsafe fn get_unchecked_mut(&mut self, i: usize, j: usize) -> &mut T;
    /// Returns a mutable reference to the element at `(i, j)`, or `None` if
    /// either index is out of bounds.
    fn get_mut(&mut self, i: usize, j: usize) -> Option<&mut T> {
        if i < R::DIM && j < C::DIM {
            // Safety: just checked `i < R::DIM` and `j < C::DIM` above.
            Some(unsafe { self.get_unchecked_mut(i, j) })
        } else {
            None
        }
    }
}

/// Marker for backends whose `R::DIM * C::DIM` elements are laid out with
/// no padding or stride gaps, enabling flat-slice access and direct BLAS
/// interop.
///
/// # Safety
/// `as_slice()` must return exactly `R::DIM * C::DIM` elements, ordered
/// consistently with `Storage::offset` and `ORDER`.
pub unsafe trait ContiguousStorage<T, R: Dim, C: Dim>:
    Storage<T, R, C>
{
    /// The memory layout this backend's contiguous elements follow.
    const ORDER: MatrixLayout;
    /// Returns the backend's elements as a single contiguous slice.
    fn as_slice(&self) -> &[T];
}

/// Mutable, contiguous access to an `R x C` grid of `T`.
///
/// # Safety
/// Combines [`StorageMut`] and [`ContiguousStorage`]'s invariants: their
/// contracts must both hold simultaneously for the same backing memory.
pub unsafe trait ContiguousStorageMut<T, R: Dim, C: Dim>:
    StorageMut<T, R, C> + ContiguousStorage<T, R, C>
{
    /// Returns the backend's elements as a single mutable contiguous slice.
    fn as_mut_slice(&mut self) -> &mut [T];
}

////////////////////////////////////////////////////////////////////////////////
// Initialization
////////////////////////////////////////////////////////////////////////////////

/// Safe initialization strategies for a storage backend, none of which
/// require `T: Default`.
///
/// # Note
/// None of these methods are `const fn`, including `from_element`/`zeros`/
/// `identity` (NFR-3 asks for constness "where possible" only). Calling a
/// generic trait-bound associated item (`T::ZERO`, `Clone::clone`) from a
/// `const fn` requires `const_trait_impl`, which is still unstable (tracked
/// in `num-traits-design.md`'s own Risk 4) — `ArrayStorage` instead exposes
/// a `const fn from_array` inherent constructor for the fully-const path.
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
// ArrayStorage: the default owning, stack-based backend
////////////////////////////////////////////////////////////////////////////////

/// The default owning storage backend: a column-major, stack-allocated
/// `R x C` array.
///
/// Parameterized over plain `const R: usize, const C: usize` rather than
/// `Dim`-typed row/column parameters, so its backing array can be written
/// exactly as `[[T; R]; C]` (Rust cannot use an associated const of a
/// generic `Dim` type as an array length on stable). It implements the
/// `Dim`-generic [`Storage`] trait family specifically for `Const<R>`/
/// `Const<C>` — `num_types.rs`'s own const-generic ergonomics bridge (FR-4
/// of `num-types-design.md`) — the same pattern nalgebra's `Const<T>` uses
/// to compute `Dim::value()` with no type-level detour.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::type_complexity)]
pub struct ArrayStorage<T, const R: usize, const C: usize> {
    data: [[T; R]; C],
}

impl<T, const R: usize, const C: usize> ArrayStorage<T, R, C> {
    /// Builds a backend directly from a column-major `[[T; R]; C]` array.
    #[must_use]
    #[allow(clippy::type_complexity)]
    pub const fn from_array(data: [[T; R]; C]) -> Self {
        Self { data }
    }
}

// Safety: `offset` maps every `(i, j)` with `i < R` and `j < C` to a
// distinct, in-bounds index of the `R * C`-element `data` array.
unsafe impl<T, const R: usize, const C: usize> Storage<T, Const<R>, Const<C>>
    for ArrayStorage<T, R, C>
where
    Const<R>: Dim,
    Const<C>: Dim,
{
    fn ptr(&self) -> *const T {
        self.data.as_ptr().cast()
    }
    #[allow(clippy::arithmetic_side_effects)]
    fn offset(&self, i: usize, j: usize) -> isize {
        (j * R + i).cast_signed()
    }
    unsafe fn get_unchecked(&self, i: usize, j: usize) -> &T {
        let offset = self.offset(i, j);
        // Safety: delegated to the caller per this method's own contract.
        unsafe { &*self.ptr().offset(offset) }
    }
}

// Safety: `ptr_mut` addresses the same `data` array `ptr` does, with
// exclusive access for the duration of the `&mut self` borrow.
unsafe impl<T, const R: usize, const C: usize> StorageMut<T, Const<R>, Const<C>>
    for ArrayStorage<T, R, C>
where
    Const<R>: Dim,
    Const<C>: Dim,
{
    fn ptr_mut(&mut self) -> *mut T {
        self.data.as_mut_ptr().cast()
    }
    unsafe fn get_unchecked_mut(&mut self, i: usize, j: usize) -> &mut T {
        let offset = self.offset(i, j);
        // Safety: delegated to the caller per this method's own contract.
        unsafe { &mut *self.ptr_mut().offset(offset) }
    }
}

// Safety: `[[T; R]; C]` has no padding between rows or columns (Rust array
// layout guarantee), so `as_slice` covers exactly `R * C` contiguous
// elements ordered consistently with `offset`'s column-major mapping.
unsafe impl<T, const R: usize, const C: usize>
    ContiguousStorage<T, Const<R>, Const<C>> for ArrayStorage<T, R, C>
where
    Const<R>: Dim,
    Const<C>: Dim,
{
    const ORDER: MatrixLayout = MatrixLayout::ColMajor;
    fn as_slice(&self) -> &[T] {
        self.data.as_flattened()
    }
}

unsafe impl<T, const R: usize, const C: usize>
    ContiguousStorageMut<T, Const<R>, Const<C>> for ArrayStorage<T, R, C>
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

////////////////////////////////////////////////////////////////////////////////
// StorageView / StorageViewMut: zero-copy, non-owning backends
////////////////////////////////////////////////////////////////////////////////

/// A zero-copy, non-owning, immutable view over an `R x C` grid backed by
/// existing memory.
///
/// Generic over any `Dim` (not just `Const<N>`), since it borrows rather
/// than owns its backing array.
#[derive(Debug)]
pub struct StorageView<'a, T, R: Dim, C: Dim> {
    data: &'a [T],
    order: MatrixLayout,
    #[allow(clippy::type_complexity)]
    _marker: PhantomData<(R, C)>,
}

impl<'a, T, R: Dim, C: Dim> StorageView<'a, T, R, C> {
    /// Wraps `data` as an `R x C` view with the given layout.
    ///
    /// # Errors
    /// Returns [`ConversionError::DimensionMismatch`] if
    /// `data.len() != R::DIM * C::DIM`.
    pub fn new(data: &'a [T], order: MatrixLayout) -> ConversionResult<Self> {
        if R::DIM.checked_mul(C::DIM) == Some(data.len()) {
            Ok(Self {
                data,
                order,
                _marker: PhantomData,
            })
        } else {
            Err(ConversionError::DimensionMismatch)
        }
    }
}

// Safety: `offset` maps every in-bounds `(i, j)` to a distinct index within
// `data`, whose length was checked against `R::DIM * C::DIM` at construction.
unsafe impl<T, R: Dim, C: Dim> Storage<T, R, C> for StorageView<'_, T, R, C> {
    fn ptr(&self) -> *const T {
        self.data.as_ptr()
    }
    #[allow(clippy::arithmetic_side_effects)]
    fn offset(&self, i: usize, j: usize) -> isize {
        match self.order {
            MatrixLayout::ColMajor => (j * R::DIM + i).cast_signed(),
            MatrixLayout::RowMajor => (i * C::DIM + j).cast_signed(),
        }
    }
    unsafe fn get_unchecked(&self, i: usize, j: usize) -> &T {
        let offset = self.offset(i, j);
        // Safety: delegated to the caller per this method's own contract.
        unsafe { &*self.ptr().offset(offset) }
    }
}

// Safety: `data` covers exactly `R::DIM * C::DIM` contiguous elements
// (checked at construction), ordered consistently with `offset`.
unsafe impl<T, R: Dim, C: Dim> ContiguousStorage<T, R, C>
    for StorageView<'_, T, R, C>
{
    const ORDER: MatrixLayout = MatrixLayout::ColMajor;
    fn as_slice(&self) -> &[T] {
        self.data
    }
}

/// A zero-copy, non-owning, mutable view over an `R x C` grid backed by
/// existing memory.
#[derive(Debug)]
pub struct StorageViewMut<'a, T, R: Dim, C: Dim> {
    data: &'a mut [T],
    order: MatrixLayout,
    #[allow(clippy::type_complexity)]
    _marker: PhantomData<(R, C)>,
}

impl<'a, T, R: Dim, C: Dim> StorageViewMut<'a, T, R, C> {
    /// Wraps `data` as a mutable `R x C` view with the given layout.
    ///
    /// # Errors
    /// Returns [`ConversionError::DimensionMismatch`] if
    /// `data.len() != R::DIM * C::DIM`.
    pub fn new(
        data: &'a mut [T],
        order: MatrixLayout,
    ) -> ConversionResult<Self> {
        if R::DIM.checked_mul(C::DIM) == Some(data.len()) {
            Ok(Self {
                data,
                order,
                _marker: PhantomData,
            })
        } else {
            Err(ConversionError::DimensionMismatch)
        }
    }
}

// Safety: see `StorageView`'s `Storage` impl — the mapping and bounds
// argument are identical, only mutability differs.
unsafe impl<T, R: Dim, C: Dim> Storage<T, R, C>
    for StorageViewMut<'_, T, R, C>
{
    fn ptr(&self) -> *const T {
        self.data.as_ptr()
    }
    #[allow(clippy::arithmetic_side_effects)]
    fn offset(&self, i: usize, j: usize) -> isize {
        match self.order {
            MatrixLayout::ColMajor => (j * R::DIM + i).cast_signed(),
            MatrixLayout::RowMajor => (i * C::DIM + j).cast_signed(),
        }
    }
    unsafe fn get_unchecked(&self, i: usize, j: usize) -> &T {
        let offset = self.offset(i, j);
        // Safety: delegated to the caller per this method's own contract.
        unsafe { &*self.ptr().offset(offset) }
    }
}

// Safety: `ptr_mut` addresses the same `data` slice `ptr` does, with
// exclusive access for the duration of the `&mut self` borrow.
unsafe impl<T, R: Dim, C: Dim> StorageMut<T, R, C>
    for StorageViewMut<'_, T, R, C>
{
    fn ptr_mut(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }
    unsafe fn get_unchecked_mut(&mut self, i: usize, j: usize) -> &mut T {
        let offset = self.offset(i, j);
        // Safety: delegated to the caller per this method's own contract.
        unsafe { &mut *self.ptr_mut().offset(offset) }
    }
}

// Safety: `data` covers exactly `R::DIM * C::DIM` contiguous elements
// (checked at construction), ordered consistently with `offset`.
unsafe impl<T, R: Dim, C: Dim> ContiguousStorage<T, R, C>
    for StorageViewMut<'_, T, R, C>
{
    const ORDER: MatrixLayout = MatrixLayout::ColMajor;
    fn as_slice(&self) -> &[T] {
        self.data
    }
}

unsafe impl<T, R: Dim, C: Dim> ContiguousStorageMut<T, R, C>
    for StorageViewMut<'_, T, R, C>
{
    fn as_mut_slice(&mut self) -> &mut [T] {
        self.data
    }
}

////////////////////////////////////////////////////////////////////////////////
// PivotStorage: scratch-data category (FR-3)
////////////////////////////////////////////////////////////////////////////////

/// A fixed-capacity scratch buffer of row/column indices, for
/// pivoting-style algorithms (e.g. partial-pivot LU decomposition).
///
/// Indices, not `T`-typed matrix elements, are what this backend stores, so
/// it does not implement the [`Storage`] family; it is a distinct
/// "scratch-data" backend category (FR-3 of `storage-trait-design.md`).
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
