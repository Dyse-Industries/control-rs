# Matrix Type & Structural Specializations (Redesigned)

## 1. Context & Objective

The objective of the `Matrix` struct is to provide an ergonomic, type-safe, and zero-cost representation for static, stack-allocated mathematical matrices.

By using Rust's nested arrays `[[T; R]; C]`, we represent a matrix in column-major order (an array of `C` columns, where each column is an array of `R` elements) without requiring dynamic allocation or unsafe storage abstractions. This completely removes the complex storage traits, simplifying compile-time checking and generic parameters.

---

## 2. The Matrix Struct & Aliases

The primary `Matrix` struct is defined as:

```rust
/// A column-major M x N static matrix allocated on the stack.
/// Inner array layout: C elements of type [T; R] (column-major).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Matrix<T, const R: usize, const C: usize> {
    pub(crate) data: [[T; R]; C],
}

impl<T, const R: usize, const C: usize> Matrix<T, R, C> {
    /// Create a new Matrix from a nested column-major array.
    #[inline(always)]
    pub const fn new(data: [[T; R]; C]) -> Self {
        Self { data }
    }

    /// Returns the number of rows.
    #[inline(always)]
    pub const fn rows(&self) -> usize {
        R
    }

    /// Returns the number of columns.
    #[inline(always)]
    pub const fn cols(&self) -> usize {
        C
    }

    /// Safe element access returning a reference to the element.
    #[inline(always)]
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if row < R && col < C {
            Some(&self.data[col][row])
        } else {
            None
        }
    }

    /// Safe mutable element access.
    #[inline(always)]
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        if row < R && col < C {
            Some(&mut self.data[col][row])
        } else {
            None
        }
    }
}
```

### 2.1. Convenience Aliases

With the removal of the storage traits, we no longer carry a `const N: usize` generic parameter. The aliases are simplified:

```rust
/// General M x N static matrix allocated on the stack.
pub type MatrixMN<T, const R: usize, const C: usize> = Matrix<T, R, C>;

/// Square static matrix allocated on the stack.
pub type SquareMatrix<T, const D: usize> = Matrix<T, D, D>;

/// Column Vector (D x 1) allocated on the stack.
pub type Vector<T, const D: usize> = Matrix<T, D, 1>;

/// Row Vector (1 x D) allocated on the stack.
pub type RowVector<T, const D: usize> = Matrix<T, 1, D>;
```

---

## 3. Structural Specializations (Triangular & Symmetric)

Structural specializations enforce mathematical structures at compile-time and are represented as fundamentally different types wrapping `Matrix<T, D, D>`. This design simplifies implementation of specialized operators that mix specializations.

```rust
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
```

### 3.1. Construction and Invariant Enforcement

Because the specialized wrappers take `Matrix<T, D, D>` as input, they are guaranteed to be square at compile-time. No runtime dimension checks are required:

```rust
impl<T, const D: usize> UpperTriangular<T, D> {
    /// Construct an UpperTriangular matrix.
    pub fn new(matrix: Matrix<T, D, D>) -> Self {
        Self { matrix }
    }
}

impl<T, const D: usize> LowerTriangular<T, D> {
    /// Construct a LowerTriangular matrix.
    pub fn new(matrix: Matrix<T, D, D>) -> Self {
        Self { matrix }
    }
}

impl<T, const D: usize> Symmetric<T, D> {
    /// Construct a Symmetric matrix.
    pub fn new(matrix: Matrix<T, D, D>) -> Self {
        Self { matrix }
    }
}
```

### 3.2. Safe Access & Mutation Rules

* **UpperTriangular**: Element reads outside the upper triangle return the stored value (assumed zero if properly initialized). Mutation via `get_mut` returns `None` for the strictly lower triangle.
* **LowerTriangular**: Element reads outside the lower triangle return the stored value. Mutation via `get_mut` returns `None` for the strictly upper triangle.
* **Symmetric**: Element writes update both mirror elements `(row, col)` and `(col, row)` to maintain symmetry.

---

## 4. Mathematical Operations

All operations are designed for `no_std` environments, avoiding dynamic allocation.

### 4.1. In-place Addition (`AddAssign`)

Element-wise addition is implemented safely using loops over columns and rows:

```rust
impl<T, const R: usize, const C: usize> core::ops::AddAssign<&Matrix<T, R, C>> for Matrix<T, R, C>
where
    T: core::ops::AddAssign<T> + Copy,
{
    fn add_assign(&mut self, rhs: &Matrix<T, R, C>) {
        for (dst_col, src_col) in self.data.iter_mut().zip(rhs.data.iter()) {
            for (d, s) in dst_col.iter_mut().zip(src_col.iter()) {
                *d += *s;
            }
        }
    }
}
```