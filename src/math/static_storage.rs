//! # Common traits for static storage.
//!
//! This module defines traits for accessing static storage and matrix properties.

use crate::math::num_types::Dim;

/// Represents the memory layout of a matrix or tensor for BLAS operations.
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
/// This trait provides methods to obtain static references and raw pointers to the underlying data.
/// Implementors must ensure that the data lives for the `'static` lifetime.
///
/// # Generic Arguments
/// * `T` - The type of elements stored.
pub trait StaticStorage<T: 'static> {
    /// Returns a mutable raw pointer to the stored data.
    ///
    /// # Returns
    /// * `*mut T` - A mutable raw pointer to the data.
    fn get_mut_ptr(&mut self) -> *mut T;

    /// Returns a raw pointer to the stored data.
    ///
    /// # Returns
    /// * `*const T` - A raw pointer to the data.
    fn get_ptr(&self) -> *const T;

    /// Returns a mutable static reference to the stored data.
    ///
    /// # Returns
    /// * `&'static mut T` - A mutable reference to the data.
    fn get_static_mut(&mut self) -> &'static mut T;

    /// Returns a static reference to the stored data.
    ///
    /// # Returns
    /// * `&'static T` - A reference to the data.
    fn get_static_ref(&self) -> &'static T;
}

/// Defines the interface for matrix storage with dimensions and layout.
///
/// This trait extends `StaticStorage` to include matrix-specific properties such as
/// dimensions (rows, columns) and memory layout (row-major or column-major).
///
/// # Generic Arguments
/// * `T` - The type of elements stored in the matrix.
/// * `R` - The dimension type representing the number of rows.
/// * `C` - The dimension type representing the number of columns.
pub trait MatrixStorage<T: 'static, R: Dim, C: Dim>: StaticStorage<T> {
    /// The memory layout of the matrix.
    const LAYOUT: MatrixLayout;
    /// The number of columns in the matrix.
    const NUM_COLS: C;
    /// The number of rows in the matrix.
    const NUM_ROWS: R;

    /// Returns the number of columns in the matrix.
    ///
    /// # Returns
    /// * `usize` - The number of columns.
    fn cols(&self) -> usize {
        Self::NUM_COLS.value()
    }

    /// Returns the memory layout as an integer compatible with CBLAS.
    ///
    /// # Returns
    /// * `i32` - The layout value (101 for `RowMajor`, 102 for `ColMajor`).
    fn layout(&self) -> i32 {
        Self::LAYOUT as i32
    }

    /// Calculates the 1D flat index for a 2D matrix given a (row, col) coordinate.
    ///
    /// This function computes the index in the underlying linear storage corresponding
    /// to the given row and column indices, taking into account the matrix layout.
    ///
    /// # Arguments
    /// * `row` - The target row index (0-indexed).
    /// * `col` - The target column index (0-indexed).
    ///
    /// # Returns
    /// * `usize` - The 1D index in the flat storage.
    ///
    /// # Panics
    /// This function panics in debug builds if `row` or
    /// `col` are out of bounds and debug assertions are
    /// enabled.
    ///
    /// # Safety
    /// This function does not use `unsafe` code.
    #[allow(clippy::arithmetic_side_effects)]
    fn linear_index_unchecked(&self, row: usize, col: usize) -> usize {
        debug_assert!(row < self.rows(), "Row index out of bounds");
        debug_assert!(col < self.cols(), "Column index out of bounds");

        match Self::LAYOUT {
            MatrixLayout::RowMajor => (row * self.cols()) + col,
            MatrixLayout::ColMajor => (col * self.rows()) + row,
        }
    }

    /// Returns the number of rows in the matrix.
    ///
    /// # Returns
    /// * `usize` - The number of rows.
    fn rows(&self) -> usize {
        Self::NUM_ROWS.value()
    }
}
