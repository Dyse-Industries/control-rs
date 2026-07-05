#![allow(clippy::arbitrary_source_item_ordering, clippy::type_complexity)]

use crate::math::ArithmeticError;

/// A column-major M x N static `Matrix` allocated on the stack.
/// Inner array layout: C elements of type [T; R] (column-major).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Matrix<T, const R: usize, const C: usize> {
    pub(crate) data: [[T; R]; C],
}

impl<T, const R: usize, const C: usize> Matrix<T, R, C> {
    /// Returns the number of columns.
    #[inline(always)]
    pub const fn cols(&self) -> usize {
        C
    }

    /// Safe element access returning a reference to the element.
    #[inline(always)]
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        self.data.get(col).and_then(|col_arr| col_arr.get(row))
    }

    /// Safe mutable element access.
    #[inline(always)]
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        self.data
            .get_mut(col)
            .and_then(|col_arr| col_arr.get_mut(row))
    }

    /// Create a new `Matrix` wrapper around a nested column-major array.
    #[inline(always)]
    pub const fn new(data: [[T; R]; C]) -> Self {
        Self { data }
    }

    /// Returns the number of rows.
    #[inline(always)]
    pub const fn rows(&self) -> usize {
        R
    }
}

// --- Dimensional & View Aliases ---

/// General M x N static matrix allocated on the stack.
pub type MatrixMN<T, const R: usize, const C: usize> = Matrix<T, R, C>;

/// Square static matrix allocated on the stack.
pub type SquareMatrix<T, const D: usize> = Matrix<T, D, D>;

/// Column Vector (D x 1) allocated on the stack.
pub type Vector<T, const D: usize> = Matrix<T, D, 1>;

/// Row Vector (1 x D) allocated on the stack.
pub type RowVector<T, const D: usize> = Matrix<T, 1, D>;

// --- Mathematical Operations ---

impl<T, const R: usize, const C: usize> core::ops::AddAssign<&Self>
    for Matrix<T, R, C>
where
    T: core::ops::AddAssign<T> + Copy,
{
    #[allow(clippy::arithmetic_side_effects)]
    #[inline]
    fn add_assign(&mut self, rhs: &Self) {
        for (dst_col, src_col) in self.data.iter_mut().zip(rhs.data.iter()) {
            for (d, s) in dst_col.iter_mut().zip(src_col.iter()) {
                *d += *s;
            }
        }
    }
}

impl<T, const R: usize, const C: usize> core::ops::SubAssign<&Self>
    for Matrix<T, R, C>
where
    T: core::ops::SubAssign<T> + Copy,
{
    #[allow(clippy::arithmetic_side_effects)]
    #[inline]
    fn sub_assign(&mut self, rhs: &Self) {
        for (dst_col, src_col) in self.data.iter_mut().zip(rhs.data.iter()) {
            for (d, s) in dst_col.iter_mut().zip(src_col.iter()) {
                *d -= *s;
            }
        }
    }
}

// --- Structural Specializations ---

/// Upper triangular matrix wrapper (elements below the diagonal are logically zero).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UpperTriangular<T, const D: usize> {
    pub(crate) matrix: Matrix<T, D, D>,
}

/// Lower triangular matrix wrapper (elements above the diagonal are logically zero).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LowerTriangular<T, const D: usize> {
    pub(crate) matrix: Matrix<T, D, D>,
}

/// Symmetric matrix wrapper (elements satisfy A[i, j] == A[j, i]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Symmetric<T, const D: usize> {
    pub(crate) matrix: Matrix<T, D, D>,
}

impl<T, const D: usize> UpperTriangular<T, D> {
    /// Returns the number of columns.
    #[inline(always)]
    pub const fn cols(&self) -> usize {
        D
    }

    /// Safe read-only element access.
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        self.matrix.get(row, col)
    }

    /// Mutably accesses an element in the upper triangle.
    /// Returns `None` if the element is in the strictly lower triangle.
    #[inline]
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        if row <= col {
            self.matrix.get_mut(row, col)
        } else {
            None
        }
    }

    /// Construct an `UpperTriangular` matrix.
    ///
    /// # Errors
    /// This function returns an `ArithmeticError` for compatibility, but is statically guaranteed to succeed.
    #[inline]
    pub const fn new(matrix: Matrix<T, D, D>) -> Result<Self, ArithmeticError> {
        Ok(Self { matrix })
    }

    /// Returns the number of rows.
    #[inline(always)]
    pub const fn rows(&self) -> usize {
        D
    }
}

impl<T, const D: usize> LowerTriangular<T, D> {
    /// Returns the number of columns.
    #[inline(always)]
    pub const fn cols(&self) -> usize {
        D
    }

    /// Safe read-only element access.
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        self.matrix.get(row, col)
    }

    /// Mutably accesses an element in the lower triangle.
    /// Returns `None` if the element is in the strictly upper triangle.
    #[inline]
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        if row >= col {
            self.matrix.get_mut(row, col)
        } else {
            None
        }
    }

    /// Construct a `LowerTriangular` matrix.
    ///
    /// # Errors
    /// This function returns an `ArithmeticError` for compatibility, but is statically guaranteed to succeed.
    #[inline]
    pub const fn new(matrix: Matrix<T, D, D>) -> Result<Self, ArithmeticError> {
        Ok(Self { matrix })
    }

    /// Returns the number of rows.
    #[inline(always)]
    pub const fn rows(&self) -> usize {
        D
    }
}

impl<T, const D: usize> Symmetric<T, D> {
    /// Returns the number of columns.
    #[inline(always)]
    pub const fn cols(&self) -> usize {
        D
    }

    /// Safe read-only element access.
    #[inline(always)]
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        self.matrix.get(row, col)
    }

    /// Construct a `Symmetric` matrix.
    ///
    /// # Errors
    /// This function returns an `ArithmeticError` for compatibility, but is statically guaranteed to succeed.
    #[inline]
    pub const fn new(matrix: Matrix<T, D, D>) -> Result<Self, ArithmeticError> {
        Ok(Self { matrix })
    }

    /// Returns the number of rows.
    #[inline(always)]
    pub const fn rows(&self) -> usize {
        D
    }
}

impl<T: Copy, const D: usize> Symmetric<T, D> {
    /// Set an element in the symmetric matrix, updating both logical locations to maintain symmetry.
    ///
    /// # Errors
    /// Returns `ArithmeticError::DomainViolation` if `row >= D` or `col >= D`.
    #[inline]
    pub fn set(
        &mut self,
        row: usize,
        col: usize,
        val: T,
    ) -> Result<(), ArithmeticError> {
        if row < D && col < D {
            if let Some(cell) =
                self.matrix.data.get_mut(col).and_then(|c| c.get_mut(row))
            {
                *cell = val;
            }
            if let Some(cell) =
                self.matrix.data.get_mut(row).and_then(|r| r.get_mut(col))
            {
                *cell = val;
            }
            Ok(())
        } else {
            Err(ArithmeticError::DomainViolation)
        }
    }
}
