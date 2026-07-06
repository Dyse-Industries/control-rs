//! # Matrix
//!
//! A module providing a stack-allocated, column-major `Matrix` implementation
//! and structural specializations like `UpperTriangular`, `LowerTriangular`, and `Symmetric`.

use crate::math::ArithmeticError;
use core::ops::{AddAssign, SubAssign};

/// Unit and HIL test suites for matrices.
#[cfg(any(test, feature = "hil"))]
pub mod tests;

/// A column-major M x N static `Matrix` allocated on the stack.
/// Inner array layout: C elements of type [T; R] (column-major).
///
/// # Clippy Allow explanation
/// We allow `clippy::type_complexity` because nested arrays are the most direct, zero-cost representation
/// of stack-allocated column-major matrices in a no-std context.
#[allow(clippy::type_complexity)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Matrix<T, const R: usize, const C: usize> {
    pub(crate) data: [[T; R]; C],
}

/// General M x N static matrix allocated on the stack.
pub type MatrixMN<T, const R: usize, const C: usize> = Matrix<T, R, C>;

/// Square static matrix allocated on the stack.
pub type SquareMatrix<T, const D: usize> = Matrix<T, D, D>;

/// Column Vector (D x 1) allocated on the stack.
pub type Vector<T, const D: usize> = Matrix<T, D, 1>;

/// Row Vector (1 x D) allocated on the stack.
pub type RowVector<T, const D: usize> = Matrix<T, 1, D>;

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

impl<T, const R: usize, const C: usize> Matrix<T, R, C> {
    /// Returns a flat mutable slice of the matrix data in column-major order.
    #[allow(clippy::arithmetic_side_effects)]
    #[inline]
    pub const fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: `[[T; R]; C]` has the exact same memory layout and size as `[T; R * C]`.
        unsafe {
            core::slice::from_raw_parts_mut(
                self.data.as_mut_ptr().cast::<T>(),
                R * C,
            )
        }
    }

    /// Returns a flat read-only slice of the matrix data in column-major order.
    #[allow(clippy::arithmetic_side_effects)]
    #[inline]
    pub const fn as_slice(&self) -> &[T] {
        // SAFETY: `[[T; R]; C]` has the exact same memory layout and size as `[T; R * C]`.
        unsafe {
            core::slice::from_raw_parts(self.data.as_ptr().cast::<T>(), R * C)
        }
    }

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
    ///
    /// # Clippy Allow explanation
    /// Allowed locally because column-major nesting represents the matrix layout precisely.
    #[allow(clippy::type_complexity)]
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

impl<T, const R: usize, const C: usize> core::ops::AddAssign<&Self>
    for Matrix<T, R, C>
where
    T: crate::math::num_traits::Ring,
{
    /// Performs element-wise matrix addition in place using standard BLAS AXPY subprograms.
    #[inline]
    fn add_assign(&mut self, rhs: &Self) {
        use crate::math::subprograms::BasicSubPrograms;
        use crate::math::subprograms::level1::AXPY;
        BasicSubPrograms::axpy(T::ONE, rhs.as_slice(), self.as_mut_slice());
    }
}

impl<T, const R: usize, const C: usize> core::ops::AddAssign<Self>
    for Matrix<T, R, C>
where
    T: crate::math::num_traits::Ring,
{
    /// Performs element-wise matrix addition in place using standard BLAS AXPY subprograms.
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.add_assign(&rhs);
    }
}

impl<T, const R: usize, const C: usize> core::ops::SubAssign<&Self>
    for Matrix<T, R, C>
where
    T: crate::math::num_traits::Ring + crate::math::ops::Neg<Output = T>,
{
    /// Performs element-wise matrix subtraction in place using standard BLAS AXPY subprograms.
    ///
    /// # Clippy Allow explanation
    /// We allow `clippy::arithmetic_side_effects` here because negating `T::ONE` to pass to `axpy`
    /// is a standard algebraic representation of negation/subtraction.
    #[allow(clippy::arithmetic_side_effects)]
    #[inline]
    fn sub_assign(&mut self, rhs: &Self) {
        use crate::math::subprograms::BasicSubPrograms;
        use crate::math::subprograms::level1::AXPY;
        BasicSubPrograms::axpy(-T::ONE, rhs.as_slice(), self.as_mut_slice());
    }
}

impl<T, const R: usize, const C: usize> core::ops::SubAssign<Self>
    for Matrix<T, R, C>
where
    T: crate::math::num_traits::Ring + crate::math::ops::Neg<Output = T>,
{
    /// Performs element-wise matrix subtraction in place using standard BLAS AXPY subprograms.
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.sub_assign(&rhs);
    }
}

impl<T, const R: usize, const C: usize> core::ops::Add<Self> for Matrix<T, R, C>
where
    T: crate::math::num_traits::Ring,
{
    type Output = Self;

    #[inline]
    fn add(mut self, rhs: Self) -> Self::Output {
        self.add_assign(&rhs);
        self
    }
}

impl<T, const R: usize, const C: usize> core::ops::Add<&Self>
    for Matrix<T, R, C>
where
    T: crate::math::num_traits::Ring,
{
    type Output = Self;

    #[inline]
    fn add(mut self, rhs: &Self) -> Self::Output {
        self.add_assign(rhs);
        self
    }
}

impl<T, const R: usize, const C: usize> core::ops::Sub<Self> for Matrix<T, R, C>
where
    T: crate::math::num_traits::Ring + crate::math::ops::Neg<Output = T>,
{
    type Output = Self;

    #[inline]
    fn sub(mut self, rhs: Self) -> Self::Output {
        self.sub_assign(&rhs);
        self
    }
}

impl<T, const R: usize, const C: usize> core::ops::Sub<&Self>
    for Matrix<T, R, C>
where
    T: crate::math::num_traits::Ring + crate::math::ops::Neg<Output = T>,
{
    type Output = Self;

    #[inline]
    fn sub(mut self, rhs: &Self) -> Self::Output {
        self.sub_assign(rhs);
        self
    }
}

impl<T, const R: usize, const C: usize> core::ops::MulAssign<T>
    for Matrix<T, R, C>
where
    T: crate::math::num_traits::Ring,
{
    /// Scales the matrix by a scalar in-place using standard BLAS SCAL subprograms.
    #[inline]
    fn mul_assign(&mut self, rhs: T) {
        use crate::math::subprograms::BasicSubPrograms;
        use crate::math::subprograms::level1::SCAL;
        BasicSubPrograms::scal(rhs, self.as_mut_slice());
    }
}

impl<T, const R: usize, const C: usize> core::ops::Mul<T> for Matrix<T, R, C>
where
    T: crate::math::num_traits::Ring,
{
    type Output = Self;

    /// Scales the matrix by a scalar.
    #[allow(clippy::arithmetic_side_effects)]
    #[inline]
    fn mul(mut self, rhs: T) -> Self::Output {
        self *= rhs;
        self
    }
}

impl<T, const R: usize, const C: usize> core::ops::DivAssign<T>
    for Matrix<T, R, C>
where
    T: crate::math::num_traits::Field,
{
    /// Divides the matrix by a scalar in-place using standard BLAS SCAL subprograms with reciprocal.
    #[allow(clippy::arithmetic_side_effects)]
    #[inline]
    fn div_assign(&mut self, rhs: T) {
        use crate::math::subprograms::BasicSubPrograms;
        use crate::math::subprograms::level1::SCAL;
        BasicSubPrograms::scal(T::one() / rhs, self.as_mut_slice());
    }
}

impl<T, const R: usize, const C: usize> core::ops::Div<T> for Matrix<T, R, C>
where
    T: crate::math::num_traits::Field,
{
    type Output = Self;

    /// Divides the matrix by a scalar.
    #[allow(clippy::arithmetic_side_effects)]
    #[inline]
    fn div(mut self, rhs: T) -> Self::Output {
        self /= rhs;
        self
    }
}

impl<T, const R: usize, const C: usize, const C2: usize>
    core::ops::Mul<&Matrix<T, C, C2>> for Matrix<T, R, C>
where
    T: crate::math::num_traits::Ring + Copy,
{
    type Output = Matrix<T, R, C2>;

    /// Multiplies two matrices using standard BLAS GEMM subprograms.
    #[inline]
    fn mul(self, rhs: &Matrix<T, C, C2>) -> Self::Output {
        use crate::math::subprograms::BasicSubPrograms;
        use crate::math::subprograms::level3::GEMM;
        let mut out = Matrix::new([[T::ZERO; R]; C2]);
        BasicSubPrograms::gemm(
            T::ONE,
            rhs.as_slice(),
            self.as_slice(),
            T::ZERO,
            out.as_mut_slice(),
            C2,
            R,
            C,
        );
        out
    }
}

impl<T, const R: usize, const C: usize, const C2: usize>
    core::ops::Mul<Matrix<T, C, C2>> for Matrix<T, R, C>
where
    T: crate::math::num_traits::Ring + Copy,
{
    type Output = Matrix<T, R, C2>;

    /// Multiplies two matrices using standard BLAS GEMM subprograms.
    #[allow(clippy::arithmetic_side_effects)]
    #[inline]
    fn mul(self, rhs: Matrix<T, C, C2>) -> Self::Output {
        self * &rhs
    }
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
