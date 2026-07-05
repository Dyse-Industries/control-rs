# Matrix Type & Specializations

**Implementation Order:** 1  
![Date Badge](https://img.shields.io/badge/Date-July_5,_2026-blue)
![Status Badge](https://img.shields.io/badge/Doc%20Status-Needs%20Review-yellow)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)


---

## 1. Context & Objective

The objective of the `Matrix` struct is to wrap the underlying `MatrixStorage`
traits and provide an ergonomic, type-safe, and zero-cost API for mathematical
operations and specialized matrix forms (e.g., vectors, square matrices, and
symmetric/triangular representations).

Because the storage layer is separated from mathematical operations, the
`Matrix` type acts as a thin wrapper boundary. It forwards memory access to the
storage backend while exposing safe algebraic interfaces to the user,
eliminating runtime overhead like virtual dispatch and redundant bounds checks.

---

## 2. The Matrix Struct & Aliases

By using an associated type `type Element` on the `MatrixStorage` trait, we
avoid carrying redundant generic parameters and `PhantomData` fields. The
primary `Matrix` struct is defined as:

```rust
use crate::math::storage::{MatrixStorage, MatrixStorageMut};

/// The primary matrix type wrapping a storage backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Matrix<S: MatrixStorage> {
    pub(crate) storage: S,
}

impl<S: MatrixStorage> Matrix<S> {
    /// Create a new Matrix wrapper around a storage backend.
    #[inline(always)]
    pub const fn new(storage: S) -> Self {
        Self { storage }
    }

    /// Returns the number of rows.
    #[inline(always)]
    pub fn rows(&self) -> usize {
        self.storage.rows()
    }

    /// Returns the number of columns.
    #[inline(always)]
    pub fn cols(&self) -> usize {
        self.storage.cols()
    }

    /// Safe element access returning a reference to the element.
    #[inline(always)]
    pub fn get(&self, row: usize, col: usize) -> Option<&S::Element> {
        self.storage.get(row, col)
    }
}

impl<S: MatrixStorageMut> Matrix<S> {
    /// Safe mutable element access.
    #[inline(always)]
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut S::Element> {
        self.storage.get_mut(row, col)
    }
}
```

### 2.1. Dimensional & View Aliases

To maintain usability, type aliases hide generic storage details from the user:

```rust
use crate::math::storage::{ArrayStorage, ViewStorage, ViewStorageMut};

/// General M x N static matrix allocated on the stack.
pub type MatrixMN<T, const R: usize, const C: usize> = Matrix<ArrayStorage<T, R, C>>;

/// Square static matrix allocated on the stack.
pub type SquareMatrix<T, const D: usize> = MatrixMN<T, D, D>;

/// Column Vector (D x 1) allocated on the stack.
pub type Vector<T, const D: usize> = MatrixMN<T, D, 1>;

/// Row Vector (1 x D) allocated on the stack.
pub type RowVector<T, const D: usize> = MatrixMN<T, 1, D>;

/// Borrowed read-only view of a matrix.
pub type MatrixView<'a, T> = Matrix<ViewStorage<'a, T>>;

/// Borrowed mutable view of a matrix.
pub type MatrixViewMut<'a, T> = Matrix<ViewStorageMut<'a, T>>;
```

---

## 3. Structural Specializations (Triangular & Symmetric)

To enforce mathematical structures at compile-time and enable specialized math
paths (e.g., Cholesky decomposition), we wrap matrices in invariant-bearing
types.

```rust
use crate::math::ArithmeticError;

/// Upper triangular matrix wrapper (elements below the diagonal are logically zero).
pub struct UpperTriangular<S: MatrixStorage>(Matrix<S>);

/// Lower triangular matrix wrapper (elements above the diagonal are logically zero).
pub struct LowerTriangular<S: MatrixStorage>(Matrix<S>);

/// Symmetric matrix wrapper (elements satisfy A[i, j] == A[j, i]).
pub struct Symmetric<S: MatrixStorage>(Matrix<S>);
```

### 3.1. Construction and Invariant Enforcement

Specialized wrappers require the underlying storage to be square. Construction
validates this invariant:

```rust
impl<S: MatrixStorage> UpperTriangular<S> {
    /// Construct an UpperTriangular matrix, verifying it is square.
    pub fn new(matrix: Matrix<S>) -> Result<Self, ArithmeticError> {
        if matrix.rows() == matrix.cols() {
            Ok(Self(matrix))
        } else {
            Err(ArithmeticError::DomainViolation)
        }
    }
}

impl<S: MatrixStorage> LowerTriangular<S> {
    /// Construct a LowerTriangular matrix, verifying it is square.
    pub fn new(matrix: Matrix<S>) -> Result<Self, ArithmeticError> {
        if matrix.rows() == matrix.cols() {
            Ok(Self(matrix))
        } else {
            Err(ArithmeticError::DomainViolation)
        }
    }
}

impl<S: MatrixStorage> Symmetric<S> {
    /// Construct a Symmetric matrix, verifying it is square.
    pub fn new(matrix: Matrix<S>) -> Result<Self, ArithmeticError> {
        if matrix.rows() == matrix.cols() {
            Ok(Self(matrix))
        } else {
            Err(ArithmeticError::DomainViolation)
        }
    }
}
```

### 3.2. Safe Mutation Rules

To preserve structural invariants, mutation must be restricted:

* **Triangular Matrices**: Mutable access (`get_mut`) is only allowed on
  elements within the active triangle.
  ```rust
  impl<S: MatrixStorageMut> UpperTriangular<S> {
      /// Mutably accesses an element in the upper triangle.
      /// Returns `None` if the element is in the strictly lower triangle.
      #[inline]
      pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut S::Element> {
          if row <= col {
              self.0.get_mut(row, col)
          } else {
              None
          }
      }
  }
  ```
* **Symmetric Matrices**: Exposing a direct `&mut T` reference to a single
  element allows a user to break symmetry. Instead, `Symmetric` does not
  implement standard mutable indexing. It exposes a safe write method that
  updates both mirror elements:
  ```rust
  impl<S: MatrixStorageMut> Symmetric<S>
  where
      S::Element: Copy,
  {
      /// Set an element in the symmetric matrix, updating both logical locations to maintain symmetry.
      pub fn set(&mut self, row: usize, col: usize, val: S::Element) -> Result<(), ArithmeticError> {
          if row < self.0.rows() && col < self.0.cols() {
              unsafe {
                  *self.0.storage.get_unchecked_mut(row, col) = val;
                  *self.0.storage.get_unchecked_mut(col, row) = val;
              }
              Ok(())
          } else {
              Err(ArithmeticError::DomainViolation)
          }
      }
  }
  ```

---

## 4. Mathematical Operations

All operations are designed for `no_std` environments, avoiding any dynamic
memory allocation.

### 4.1. In-place Addition (`AddAssign`)

Element-wise addition is implemented generically using row and column loops:

```rust
impl<S1, S2> core::ops::AddAssign<&Matrix<S2>> for Matrix<S1>
where
    S1: MatrixStorageMut,
    S2: MatrixStorage<Element=S1::Element>,
    S1::Element: core::ops::AddAssign<S2::Element> + Copy,
{
    fn add_assign(&mut self, rhs: &Matrix<S2>) {
        assert_eq!(self.rows(), rhs.rows());
        assert_eq!(self.cols(), rhs.cols());

        for r in 0..self.rows() {
            for c in 0..self.cols() {
                unsafe {
                    *self.storage.get_unchecked_mut(r, c) += *rhs.storage.get_unchecked(r, c);
                }
            }
        }
    }
}
```

### 4.2. Contiguous Fast-Path Optimization

If both operands are contiguous, algorithms can bypass double-loop coordinate
math and operate directly on flat slices to leverage SIMD or compiler loop
vectorization:

```rust
use crate::math::storage::{ContiguousStorage, ContiguousStorageMut};

/// Fast-path addition for contiguous column-major storage backends.
pub fn add_contiguous<S1, S2>(lhs: &mut Matrix<S1>, rhs: &Matrix<S2>)
where
    S1: ContiguousStorageMut,
    S2: ContiguousStorage<Element=S1::Element>,
    S1::Element: core::ops::AddAssign<S2::Element> + Copy,
{
    assert_eq!(lhs.rows(), rhs.rows());
    assert_eq!(lhs.cols(), rhs.cols());

    let dst = lhs.storage.as_mut_slice();
    let src = rhs.storage.as_slice();

    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d += *s;
    }
}
```

---

## 5. Testing, Verification, & Performance Goals

### 5.1. Test Strategy

1. **Unit Testing**:
    - Verify index offsets, strides, and dimensional bounds.
    - Assert compiler errors (via compile-fail tests) when trying to write to
      invalid regions of specialized wrappers.
2. **Invariant Testing**:
    - Confirm that `UpperTriangular` and `LowerTriangular` restrict mutable
      access to their valid boundaries.
    - Confirm that `Symmetric` updates both mirror elements upon mutation.
3. **Property Testing**:
    - Compare calculations (e.g., matrix-vector multiplication) across different
      backends (owned static matrix vs. strided sub-views) and assert identical
      outputs.
4. **Hardware Benchmarks**:
    - Execute test suites on Teensty 4.1 hardware to measure execution speed and
      stack bounds.

### 5.2. Numeric Guarantees

* **Integers and Deterministic Transforms**: Operations must match bit-for-bit
  identical results regardless of the storage backend.
* **Floating-Point Operations**: Checked using tolerance-based comparisons to
  account for floating-point non-associativity:
  $$\|A - B\|_{\infty} \le \epsilon$$

### 5.3. Bare-Metal Performance Targets

* **Zero-Overhead Abstraction**: The abstraction boundary must incur no
  measurable runtime latency compared to manual flat-array implementations.
* **Verification Method**: Inspect compiler-generated assembly to ensure bounds
  checking is fully optimized out in loop bodies, and compile-time generic
  monomorphization has eliminated all virtual dispatch symbols.