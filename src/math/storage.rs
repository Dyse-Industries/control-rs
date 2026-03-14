//! # Static Storage and Matrix Traits
//!
//! These traits are essential for interfacing with linear algebra backends (like BLAS or LAPACK)
//! and ensuring consistent memory management across the library.

use crate::math::num_types::{Const, Dim};
use core::mem::MaybeUninit;

/// Represents the memory layout of a matrix or tensor for BLAS operations.
///
/// This enum specifies how multidimensional data is mapped to linear memory.
/// *   `RowMajor`: Elements of a row are contiguous.
/// *   `ColMajor`: Elements of a column are contiguous.
///
/// # Example
/// ```
/// use control_rs::math::storage::MatrixLayout;
///
/// let layout = MatrixLayout::RowMajor;
/// assert_eq!(layout as i32, 101);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)] // Ensures compatibility if passed directly to C FFI
pub enum MatrixLayout {
    /// Row-major layout (C-style). Elements of a row are contiguous in memory.
    /// CBLAS equivalent: `CblasRowMajor` (101)
    RowMajor = 101,

    /// Column-major layout (Fortran-style). Elements of a column are contiguous in memory.
    /// CBLAS equivalent: `CblasColMajor` (102)
    ColMajor = 102,
}

/// Defines the interface for accessing static storage.
///
/// This trait provides methods to get raw pointers to the underlying data storage.
/// It is primarily used for FFI interactions or low-level memory manipulation where
/// direct pointer access is required.
///
/// # Generic Arguments
/// * `T` - The type of elements stored.
///
/// # Example
/// ```
/// use control_rs::math::storage::{StaticArray, StaticStorage};
///
/// let mut storage = StaticArray([1, 2, 3]);
/// let ptr = storage.get_ptr();
/// let mut_ptr = storage.get_mut_ptr();
///
/// unsafe {
///     assert_eq!(*ptr, 1);
///     *mut_ptr.add(1) = 5;
///     assert_eq!(storage.0[1], 5);
/// }
///
/// for (i, &val) in storage.iter().enumerate() {
///     assert_eq!(val, storage.0[i]);
/// }
/// ```
pub trait StaticStorage<T: 'static> {
    /// Returns a mutable raw pointer to the stored data.
    ///
    /// # Returns
    /// * `*mut T` - A mutable raw pointer to the beginning of the data.
    ///
    /// # Safety
    /// This function does not use `unsafe` code, but the returned pointer must be used with care.
    /// The caller must ensure that:
    /// *   The pointer is not dereferenced after the storage is dropped.
    /// *   Aliasing rules are respected if creating multiple mutable references.
    fn get_mut_ptr(&mut self) -> *mut T;

    /// Returns a const raw pointer to the stored data.
    ///
    /// # Returns
    /// * `*const T` - A const raw pointer to the beginning of the data.
    ///
    /// # Safety
    /// This function does not use `unsafe` code.
    fn get_ptr(&self) -> *const T;
    /// Returns an iterator over the stored elements.
    fn iter(&self) -> impl Iterator<Item = &T>;
}

/// Extension trait for initializing storage from an uninitialized state.
///
/// # Example
/// ```
/// use control_rs::math::storage::{UninitStaticArray, UninitStaticStorage, StaticStorage};
///
/// let data = [1, 2, 3, 4];
/// let initialized_storage = unsafe {
///     UninitStaticArray::<i32, 4>::unchecked_from_iterator::<_, 4>(data.into_iter())
/// };
///
/// assert_eq!(initialized_storage.0, [1, 2, 3, 4]);
/// ```
pub trait UninitStaticStorage<T: 'static> {
    /// Associated type representing the initialized variant of this type.
    type Initialized: StaticStorage<T>;

    /// Assumes the storage is fully initialized.
    ///
    /// # Safety
    /// The caller must ensure that all elements up to `CAPACITY` have been written to.
    unsafe fn assume_init(self) -> Self::Initialized;

    /// Helper to get a pointer to the uninitialized data safely.
    fn get_uninit_mut_ptr(&mut self) -> *mut T;

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
    /// * The iterator must have **at least** `N` elements, or this will assume an uninitialized
    ///   value is initialized (resulting in UB).
    ///
    /// # Panics
    /// * This function will panic in debug builds if the safety criterion is not met.
    unsafe fn unchecked_from_iterator<I, const N: usize>(
        iterator: I,
    ) -> Self::Initialized
    where
        Self: Sized,
        I: IntoIterator<Item = T>,
    {
        let mut maybe_uninit = Self::uninit();
        let arr_ptr = maybe_uninit.get_uninit_mut_ptr();
        let mut write_counter: usize = 0;
        for (i, b) in (0..N).zip(iterator.into_iter()) {
            unsafe {
                arr_ptr.add(i).write(b);
            }
            write_counter = write_counter.saturating_add(1);
        }
        debug_assert!(
            write_counter == N,
            "Incorrect number of elements in iter."
        );
        unsafe { maybe_uninit.assume_init() }
    }

    /// Creates a new, uninitialized instance of the storage.
    fn uninit() -> Self;
}

/// Defines the interface for matrix storage with dimensions and layout.
///
/// This trait extends `StaticStorage` to include matrix-specific properties such as
/// dimensions (rows, columns) and helper methods for index calculation.
///
/// # Generic Arguments
/// * `T` - The type of elements stored in the matrix.
/// * `R` - The dimension type representing the number of rows (must implement `Dim`).
/// * `C` - The dimension type representing the number of columns (must implement `Dim`).
///
/// # Example
/// ```
/// use control_rs::math::num_types::U2;
/// use control_rs::math::storage::{MatrixStorage, StaticArray, StaticStorage};
///
/// // A 2x2 matrix backed by a StaticArray of size 4
/// pub struct MyMatrix([f32; 4]);
///
/// impl StaticStorage<f32> for MyMatrix {
///     fn get_mut_ptr(&mut self) -> *mut f32 { self.0.as_mut_ptr() }
///     fn get_ptr(&self) -> *const f32 { self.0.as_ptr() }
///     fn iter(&self) -> impl Iterator<Item=&f32> { self.0.iter() }
/// }
///
/// // Implement MatrixStorage for a 2x2 matrix
/// impl MatrixStorage<f32, U2, U2> for MyMatrix {}
///
/// let matrix = MyMatrix([1.0, 2.0, 3.0, 4.0]);
///
/// assert_eq!(matrix.rows(), 2);
/// assert_eq!(matrix.cols(), 2);
/// ```
pub trait MatrixStorage<T: 'static, R: Dim, C: Dim> {
    /// Returns the number of columns in the matrix.
    ///
    /// # Returns
    /// * `usize` - The number of columns defined by type `C`.
    ///
    /// # Panics
    /// This function does not panic.
    ///
    /// # Safety
    /// This function does not use `unsafe` code.
    fn cols(&self) -> usize {
        C::DIM
    }

    /// Calculates the 1D flat index for a 2D matrix given a (row, col) coordinate.
    ///
    /// This function computes the index in the underlying linear storage corresponding
    /// to the given row and column indices, taking into account the matrix layout.
    ///
    /// # Arguments
    /// * `row` - The target row index (0-indexed).
    /// * `col` - The target column index (0-indexed).
    /// * `layout` - The memory layout of the matrix (`RowMajor` or `ColMajor`).
    ///
    /// # Returns
    /// * `usize` - The 1D index in the flat storage.
    ///
    /// # Panics
    /// This function panics if `row` or `col` are out of bounds and debug assertions are enabled.
    ///
    /// # Safety
    /// This function does not use `unsafe` code.
    ///
    /// # Example
    /// ```
    /// use control_rs::math::num_types::{U2, U3};
    /// use control_rs::math::storage::{MatrixLayout, MatrixStorage, StaticStorage};
    ///
    /// // Dummy struct for a 2x3 matrix
    /// struct MyMatrix;
    ///
    /// // Dummy impl of StaticStorage
    /// impl StaticStorage<f32> for MyMatrix {
    ///     fn get_mut_ptr(&mut self) -> *mut f32 { unimplemented!() }
    ///     fn get_ptr(&self) -> *const f32 { unimplemented!() }
    ///     fn iter(&self) -> impl Iterator<Item=&f32> { [].iter() }
    /// }
    ///
    /// // Implement MatrixStorage to get rows() and cols()
    /// impl MatrixStorage<f32, U2, U3> for MyMatrix {}
    ///
    /// let matrix = MyMatrix;
    /// let row = 1;
    /// let col = 1;
    ///
    /// // For a 2x3 matrix, rows=2, cols=3.
    /// // Row-major index for (1, 1) is (1 * 3) + 1 = 4
    /// assert_eq!(matrix.linear_index_unchecked(row, col, MatrixLayout::RowMajor), 4);
    ///
    /// // Column-major index for (1, 1) is (1 * 2) + 1 = 3
    /// assert_eq!(matrix.linear_index_unchecked(row, col, MatrixLayout::ColMajor), 3);
    /// ```
    #[allow(clippy::arithmetic_side_effects)]
    fn linear_index_unchecked(
        &self,
        row: usize,
        col: usize,
        layout: MatrixLayout,
    ) -> usize {
        debug_assert!(row < self.rows(), "Row index out of bounds");
        debug_assert!(col < self.cols(), "Column index out of bounds");

        match layout {
            MatrixLayout::RowMajor => (row * self.cols()) + col,
            MatrixLayout::ColMajor => (col * self.rows()) + row,
        }
    }

    /// Returns the number of rows in the matrix.
    ///
    /// # Returns
    /// * `usize` - The number of rows defined by type `R`.
    ///
    /// # Panics
    /// This function does not panic.
    ///
    /// # Safety
    /// This function does not use `unsafe` code.
    fn rows(&self) -> usize {
        R::DIM
    }
}

/// A concrete implementation of static storage using a fixed-size array.
///
/// This struct wraps a standard Rust array `[T; N]` and implements `StaticStorage`.
/// It serves as the backing store for fixed-size matrices.
///
/// # Generic Arguments
/// * `T` - The type of elements stored.
/// * `N` - The size of the array.
///
/// # Example
/// ```
/// use control_rs::math::storage::{StaticArray, StaticStorage};
///
/// let mut storage = StaticArray([1, 2, 3]);
/// let ptr = storage.get_ptr();
/// let mut_ptr = storage.get_mut_ptr();
///
/// unsafe {
///     assert_eq!(*ptr.add(1), 2);
///     *mut_ptr.add(1) = 5;
///     assert_eq!(storage.0[1], 5);
/// }
/// ```
pub struct StaticArray<T, const N: usize>(pub [T; N]);

/// Convenient alias for nested storage.
pub type StaticMatrix<T, const R: usize, const C: usize> =
    StaticArray<[T; R], C>;

type MaybeUninitArray<T, const N: usize> = MaybeUninit<[T; N]>;

/// A concrete implementation of uninitialized static storage using `MaybeUninit`.
///
/// This struct wraps a `MaybeUninit<[T; N]>` and implements `UninitStaticStorage`.
/// It is used to construct an initialized `StaticArray` safely.
///
/// # Generic Arguments
/// * `T` - The type of elements to be stored.
/// * `N` - The size of the array.
///
/// # Example
/// ```
/// use control_rs::math::storage::{UninitStaticArray, UninitStaticStorage, StaticStorage};
///
/// let mut uninit_storage = UninitStaticArray::<i32, 3>::uninit();
/// let ptr = uninit_storage.get_uninit_mut_ptr();
///
/// unsafe {
///     ptr.add(0).write(10);
///     ptr.add(1).write(20);
///     ptr.add(2).write(30);
/// }
///
/// let initialized_storage = unsafe { uninit_storage.assume_init() };
/// assert_eq!(initialized_storage.0, [10, 20, 30]);
/// ```
pub struct UninitStaticArray<T, const N: usize>(MaybeUninitArray<T, N>);

impl<T: 'static, const N: usize> StaticStorage<T> for StaticArray<T, N> {
    /// Returns a mutable raw pointer to the underlying array.
    ///
    /// # Returns
    /// * `*mut T` - Mutable pointer to the array data.
    ///
    /// # Panics
    /// This function does not panic.
    ///
    /// # Safety
    /// This function does not use `unsafe` code.
    fn get_mut_ptr(&mut self) -> *mut T {
        self.0.as_mut_ptr()
    }

    /// Returns a const raw pointer to the underlying array.
    ///
    /// # Returns
    /// * `*const T` - Const pointer to the array data.
    ///
    /// # Panics
    /// This function does not panic.
    ///
    /// # Safety
    /// This function does not use `unsafe` code.
    fn get_ptr(&self) -> *const T {
        self.0.as_ptr()
    }
    fn iter(&self) -> impl Iterator<Item = &T> {
        self.0.iter()
    }
}

impl<T: 'static, const N: usize> UninitStaticStorage<T>
    for UninitStaticArray<T, N>
{
    type Initialized = StaticArray<T, N>;

    /// Assumes the storage is fully initialized and unwraps it into a `StaticArray`.
    ///
    /// # Safety
    /// The caller must ensure that all elements up to `N` have been properly initialized.
    unsafe fn assume_init(self) -> Self::Initialized {
        unsafe { StaticArray(self.0.assume_init()) }
    }

    /// Returns a mutable raw pointer to the underlying uninitialized array elements.
    fn get_uninit_mut_ptr(&mut self) -> *mut T {
        // as_mut_ptr() returns *mut StaticArray<T, N>.
        // We cast it to *mut T to point to the first element's memory space.
        self.0.as_mut_ptr().cast::<T>()
    }

    /// Creates a new, uninitialized instance of the static array storage.
    fn uninit() -> Self {
        Self(MaybeUninit::uninit())
    }
}

impl<T: 'static, const R: usize, const C: usize>
    MatrixStorage<T, Const<R>, Const<C>> for StaticMatrix<T, R, C>
where
    Const<R>: Dim,
    Const<C>: Dim,
{
}
