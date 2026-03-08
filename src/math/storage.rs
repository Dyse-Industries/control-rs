//! # Static Storage and Matrix Traits
//!
//! This module defines the core traits and types for managing static storage in the control system.
//! It provides abstractions for:
//! *   **Static Storage**: Accessing raw pointers to underlying data.
//! *   **Matrix Storage**: Handling matrix dimensions and memory layouts (Row-Major vs. Column-Major).
//! *   **Static Arrays**: Concrete implementations of storage using fixed-size arrays.
//!
//! These traits are essential for interfacing with linear algebra backends (like BLAS or LAPACK)
//! and ensuring consistent memory management across the library.

use crate::math::num_types::Dim;

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
pub trait StaticStorage<T> {
    /// Returns a mutable raw pointer to the stored data.
    ///
    /// # Returns
    /// * `*mut T` - A mutable raw pointer to the beginning of the data.
    ///
    /// # Panics
    /// This function does not panic.
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
    /// # Panics
    /// This function does not panic.
    ///
    /// # Safety
    /// This function does not use `unsafe` code.
    fn get_ptr(&self) -> *const T;
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
pub trait MatrixStorage<T, R: Dim, C: Dim>: StaticStorage<T> {
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
    /// use control_rs::math::storage::{MatrixStorage, MatrixLayout};
    /// use control_rs::math::num_types::Dim;
    ///
    /// fn test_access<R: Dim, C: Dim, S: MatrixStorage<f32, R, C>>(mat: S, row: usize, col: usize) {
    ///     let idx = mat.linear_index_unchecked(row, col, MatrixLayout::RowMajor);
    ///     assert_eq!(idx, 2); // Row 1, Col 0 in 2x2 RowMajor is index 2
    /// }
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
/// unsafe {
///     assert_eq!(*storage.get_ptr(), 1);
/// }
/// ```
pub struct StaticArray<T, const N: usize>(pub [T; N]);

impl<T, const N: usize> StaticStorage<T> for StaticArray<T, N> {
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
}
