# Matrix Type & Structural Specializations (Design Document)

![Date Badge](https://img.shields.io/badge/Date-August_18,_2026-blue)
![Status Badge](https://img.shields.io/badge/Doc%20Status-Reviewed-yellow)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

---

### 1. Introduction

This module provides the matrix operations required by advanced control and
state estimation. Its architecture is directly modeled on
the `nalgebra` linear algebra crate (licensed under BSD-3-Clause, by Sébastien
Crozet).

The following elements of `nalgebra` are directly mirrored or adapted in our
architecture:

- **Matrix Signature**: `Matrix<T, R: Dim, C: Dim, S: MatrixStorage<T, R, C>>`
  is identical to `nalgebra`'s decoupled storage pattern §4.1.
- **Storage Trait Hierarchy**: Based on the three-tier hierarchy defined in
  `storage-subprograms-design.md` §4.1 (`Buffer`/
  `BufferMut`,
  `BlasStorage<T, R, C>`/`BlasStorageMut`,
  Tier-2 addressing branches `MatrixStorage<T, R, C>`/
  `PackedStorage`, and concrete leaves `Dense`, `Symmetric`,
  `Triangular`, `Diagonal`).
- **Dense Leaf Structure**: Owning dense storage is `DenseArray<T, const R,
  const C>` (`Dense<T, Const<R>, Const<C>, Array2<T, R, C>>`), wrapping
  `Array2`'s nested `[[T; R]; C]` (`storage-subprograms-design.md` §4.1.3).
  Capacity is two bare `const usize` parameters on the alias, never
  `Dim::USIZE`.
- **Matrix Views**: `MatrixSlice`/`MatrixSliceMut` wrap
  `DenseRef<'a, T, const R, const C>` and
  `DenseRefMut<'a, T, const R, const C>` (`Ref` over `&[[T; R]; C]`).
  There is no `StridedView` leaf.
- **Operand Model**: Typed `BlasStorage`/`MatrixStorage` plus nested
  `as_nested() -> &[[T; R]; C]` (and `&[T; N]` for vectors) are the
  subprogram floor. Algebraic transpose is a kernel `trans` flag.

---

### 2. Requirements

#### 2.1. Functional Requirements

- **FR-1 — Compile-Time Shape Verification**: Operations between matrix operands
  validate matching dimensions ($R_1 = R_2, C_1 = C_2$ for
  addition/subtraction; $C_1 = R_2$ for multiplication) at compile time via
  Peano type bounds (`Dim`).
- **FR-2 — Fallible Linear Algebra Solvers**: Matrix inversion and system
  solving ($A x = b$) over singular matrices return
  `Err(LinAlgError::SingularMatrix)` rather than panicking or producing invalid
  numerical outputs.
- **FR-3 — Value-Preserving Structure Conversions**: Conversions between dense,
  symmetric, triangular, and 1D polynomial/vector views preserve element values
  and coordinates without numerical truncation.

#### 2.2. Non-Functional Requirements

- **NFR-1 — Operation Complexity Guarantees**: Dense matrix
  multiplication ($M \times N \times P$) executes in $O(MNP)$ operations;
  triangular matrix multiplication and solves execute in $O(MN^2)$ operations;
  vector updates execute in $O(N)$ operations.
- **NFR-2 — Deterministic Fixed-Memory Execution**: Matrix operations perform
  zero heap allocations, executing in deterministic $O(1)$ stack memory.

#### 2.3. Constraints

- **C-1 — `#![no_std]` Environment**: Matrix operations execute without standard
  library dependencies.
- **C-2 — Zero Dynamic Allocation**: All storage is stack-based or statically
  borrowed.
- **C-3 — Dimension Ceiling**: Matrix dimensions are bounded by the `U127` Peano
  ceiling ($N \le 127$), capping maximum stack array size (e.g. 16,129 elements
  for $127 \times 127$).

---

### 3. Technical Overview

`Matrix` is a type-safe wrapper that provides compile time dimension and type
bounds to guarantee that unsafe subprograms are not misused. Beyond safety,
the API is also designed to make expressing and solving linear systems —
factorization, inversion and substitution — convenient for callers.

---

### 4. Core Architecture

The `Matrix` struct will be implemented in a new submodule: `src/matrix/mod.rs`.

#### 4.1. Generics Foundation & Sizing

The core `Matrix` structure decouples mathematical dimensions from physical
storage using the `MatrixStorage<T, R, C>` trait bound (
`storage-subprograms-design.md` Rev 1.16). `R` and `C` stay `Dim` on
`Matrix` itself. Owning aliases and nested `Ref` views take bare
`const usize` parameters (`storage-subprograms-design.md` §4.1.3):

```rust
pub struct Matrix<T, R: Dim, C: Dim, S: MatrixStorage<T, R, C>> {
    storage: S,
    _marker: core::marker::PhantomData<(R, C)>,
}

/// Default owned, stack-based `Matrix`. `R`/`C` are the alias's own const
/// generics; `Dim` slots on `Matrix`/`Dense` are `Const<R>`/`Const<C>`.
pub type ArrayMatrix<T, const R: usize, const C: usize> =
    Matrix<T, Const<R>, Const<C>, DenseArray<T, R, C>>;
pub type Owned<T, const R: usize, const C: usize> = ArrayMatrix<T, R, C>;
```

`DenseArray<T, const R, const C>` is `Dense<T, Const<R>, Const<C>, Array2<T, R,
C>>`. It does not accept a `Dim` type argument and does not project
`Dim::USIZE` into an array length.

Borrowed views use the same bare `const usize` parameters: `Ref` borrows
`[[T; R]; C]`, so `MatrixSlice` is not `Dim`-generic.

```rust
pub type MatrixSlice<'a, T, const R: usize, const C: usize> =
    Matrix<T, Const<R>, Const<C>, DenseRef<'a, T, R, C>>;
pub type MatrixSliceMut<'a, T, const R: usize, const C: usize> =
    Matrix<T, Const<R>, Const<C>, DenseRefMut<'a, T, R, C>>;
```

The point of decoupling storage is to have one `Matrix` implementation that
works for every `MatrixStorage` implementor.

#### 4.2. Memory Layout & Storage Strategy

`Matrix`'s own arithmetic never branches on layout — it reads `order()` from
whichever `MatrixStorage` backend it was instantiated with.

- **Cache Locality**: Matrix multiplication: Under column-major ordering, each
  column $ A_j $ is contiguous in memory, maximizing CPU cache hit rates.
- **BLAS Interoperability**: Column-major layout matches the standard convention
  of legacy BLAS/LAPACK (Anderson et al., 1999) and embedded DSP libraries (
  e.g., ARM CMSIS-DSP).

*The default storage backend (`DenseArray`) uses a **column-major** nested
array (`Array2<T, R, C>` = `[[T; R]; C]`).*

#### 4.3. Memory Representation, Slicing & Views

To ensure stable memory layout and compatibility with C-based hardware
libraries the matrix owns a contiguous array marked as `#[repr(C)]`.

Flat slice interfaces are exposed directly on `BlasStorage` / `BlasStorageMut`
(Tier 1 universal floor; `storage-subprograms-design.md` §4.1.1):

```rust
impl<T, R: Dim, C: Dim, S> Matrix<T, R, C, S>
where
    S: BlasStorage<T, R, C>,
{
    /// Exposes a safe contiguous slice view of matrix memory.
    pub fn as_slice(&self) -> &[T] {
        self.storage.as_slice()
    }
}

impl<T, R: Dim, C: Dim, S> Matrix<T, R, C, S>
where
    S: BlasStorageMut<T, R, C>,
{
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.storage.as_mut_slice()
    }
}

impl<T, const R: usize, const C: usize> ArrayMatrix<T, R, C> {
    /// Nested operand for subprogram kernels (`storage-subprograms-design.md` §4.2.2).
    pub fn as_nested(&self) -> &[[T; R]; C] {
        self.storage.as_nested()
    }
}
```

Owning kernels call `as_nested()` (`&[[T; R]; C]`). `as_slice()` remains the
inspection path on any `BlasStorage` backend, including views. There is no
flattened `&[T; R * C]` accessor.

##### 4.3.1. Zero-Copy Views (`DenseRef`)

`Matrix` builds on the nested-array view backends in
`storage-subprograms-design.md` §4.1.3. There is no `StridedView` and no
`transpose_view` storage type; algebraic transpose is a kernel `trans` flag.

Borrowed `DenseRef`/`DenseRefMut` leaves are constructed only through
`storage-subprograms-design.md` FR-6:

- `ArrayMatrix::view()` / `view_mut()` copy `R`/`C` from the owning alias's
  own const generics. `as_nested()` on the view is `&[[T; R]; C]`.
- `DenseArray::try_submatrix<const R2, const C2, const LDA>(origin)` takes
  caller-supplied shape and leading dimension. The runtime check is `origin`
  against `self`. Nested operand of the result is `&[[T; LDA]; C2]`
  (column panel when `origin.0 == 0` and `LDA` equals the parent's inner
  dimension). No public constructor accepts an independent `Dim` plus a
  raw `&[T]`.

`MatrixSlice` / `MatrixSliceMut` wrap those leaves.

- **In-Place Transposition**: For square matrices ($R = C$), in-place element
  swapping (`pub fn transpose_mut(&mut self)`) mutates elements directly within
  the existing memory layout. Copying `transpose` / `transpose_into` write a
  new buffer; they do not produce a transposed view type.

#### 4.4. Instantiation & Constructors

- `pub const fn zero() -> Self where T: Zero + Copy`: Instantiates an all-zero
  matrix using `T::ZERO` as the constant initialization value.
- `pub const fn identity() -> Self where T: Zero + One + Copy`: Instantiates an
  identity matrix (restricted to square shapes) by initializing elements to
  `T::ZERO` and filling the main diagonal with `T::ONE` via a const-evaluated
  loop.

- `pub const fn diagonal<const D: usize>(val: [T; D]) -> Owned<T, D, D>`:
  Constructs a diagonal matrix using the provided array of diagonal values and
  filling off-diagonal elements with `T::ZERO`.

- `pub fn from_fn<F>(mut f: F) -> Self where F: FnMut(usize, usize) -> T`:
  Generates a matrix using a coordinate-based mapping function at runtime.

_Implementation Note_: All static constructors are marked `const fn` to allow
placing static matrices directly in read-only flash memory. The scalar type
`T` must implement `Zero` and `One` from `crate::math::num_traits`. These
traits expose the associated constants `T::ZERO` and `T::ONE`.

#### 4.5. Operator Overloading

Overloads `Add`, `Sub`, `Neg`, and `Mul` from `core::ops`. Dimension rules are
statically enforced at compile-time. Conventionally these map to BLAS
subprograms in [subprograms.rs](../../src/math/subprograms.rs) (Anderson et
al., 1999; Golub & Van Loan, 2013; Demmel, 1997):

| Operator      | Subprogram              | Level | Binding                     |
|:--------------|:------------------------|:------|:----------------------------|
| `Add`         | `AXPY` (`y = a*x + y`)  | 1     | `a = T::ONE`                |
| `Sub`         | `AXPY`                  | 1     | `a = -T::ONE`               |
| `Neg`         | `SCAL`                  | 1     | `a = -T::ONE`               |
| `Mul<Matrix>` | `GEMM` (`C = αAB + βC`) | 3     | `α = T::ONE`, `β = T::ZERO` |
| `Mul<Vector>` | `GEMV` (`y = αAx + βy`) | 2     | `α = T::ONE`, `β = T::ZERO` |

`Mul<Matrix>` statically enforces $(M \times N) \times (N \times P) \to (M
\times P)$.

```rust
impl<T, M: Dim, N: Dim, P: Dim> Mul<Matrix<T, N, P>> for Matrix<T, M, N>
where
    T: Scalar + Add<Output=T> + Mul<Output=T> + Copy,
{
    type Output = Matrix<T, M, P>;
    // ...
}
```

##### 4.5.1. Required Subprogram Inventory

`Matrix` is a caller of `subprograms.rs`, not an extension of it. Every kernel
this design needs is one of the subprogram traits defined in
`storage-subprograms-design.md` §4.2.

| Subprogram | Level | Operation                      | Required by                                                                                                                                          |
|:-----------|:------|:-------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------|
| `AXPY`     | 1     | $y = a x + y$                  | `Add`, `Sub` (§4.5); reflector application in QR (§4.7)                                                                                              |
| `SCAL`     | 1     | $x = a x$                      | `Neg` (§4.5); pivot-row normalization in LU (§4.6); diagonal scaling in $LDL^T$ (§4.7)                                                               |
| `DOT`      | 1     | $x^T y$                        | Inner-product accumulation in forward/backward substitution (§4.7.2, §4.10.1) and in $LDL^T$/Cholesky diagonal updates (§4.7)                        |
| `NRM2`     | 1     | $\lVert x \rVert_2$            | Householder reflector construction in QR (§4.7)                                                                                                      |
| `IAMAX`    | 1     | $\arg\max_i \lvert x_i \rvert$ | Partial-pivot column search in LU (§4.6) and the symmetric pivot search in $LDL^T$, which inspects at most two columns per step (Greif et al., 2016) |
| `GER`      | 2     | $A = \alpha x y^T + A$         | LU decomposition trailing submatrix rank-1 updates (`lu_decompose_mut`; Decision D-2)                                                                |
| `TRSV`     | 2     | $\text{op}(A) x = b$           | Triangular linear system solves ($Ax = b$) and forward/backward substitutions (§4.7.2, §4.10.1; Decision D-2)                                        |
| `GEMV`     | 2     | $y = \alpha A x + \beta y$     | `Mul<Vector>` (§4.5); matrix-vector products in the solver paths (§4.7.2)                                                                            |
| `GEMM`     | 3     | $C = \alpha A B + \beta C$     | `Mul<Matrix>` (§4.5); trailing-submatrix block updates in LU, $LDL^T$ and QR                                                                         |

`SCAL` is defined in `subprograms.rs` and included in
`storage-subprograms-design.md` §4.2.1.

Per Decision D-2 (resolving B-2), `GER` (Level 2) and `TRSV` (Level 2) are
required routines:

- **`GER`** expresses LU elimination steps over trailing submatrices (
  `lu_decompose_mut`), avoiding full three-loop dispatch overhead of `GEMM`
  with $k = 1$.
- **`TRSV`** accelerates triangular solves and forward/backward substitutions in
  linear system solvers.

*Backend Selection*: `Matrix<T, R, C, S>` dispatches subprogram calls through
monomorphized subprogram traits backed by feature-selected implementations (
e.g., `#[cfg(feature = "accelerate")]` for hardware BLAS/DSP acceleration, with
default naive fallback bodies) without introducing a 5th generic backend
parameter `B` to the `Matrix` struct signature.

#### 4.6. Core Operations

- **Transposition**:
    - `pub fn transpose_into(&self, dest: &mut Matrix<T, C, R>)`: Writes the
      transposed matrix into a caller-provided destination buffer, avoiding
      stack returns.
    - `pub fn transpose_mut(&mut self)`: Performs an in-place transposition for
      square matrices ($R = C$).
    - `pub fn transpose(&self) -> Matrix<T, C, R>`: Returns a new transposed
      matrix on the stack (convenience API for small shapes).
    - Algebraic $A^T x$ / $A^T B$ without a new buffer: pass `trans` /
      `trans_a` into `GEMV`/`GEMM` (`storage-subprograms-design.md` §4.2.3).
      There is no `transpose_view` storage type.
- **Matrix Inversion & System Solving**:
    - _Explicit Decomposition Design_: Convenient signatures that mask
      heavy $O(N^3)$ operations behind stack-allocating value returns (such as
      `invert(&self) -> Result<Matrix<T, D, D>, LinAlgError>`) are explicitly
      rejected to prevent unexpected stack bloat in embedded runtimes.
    - `pub fn invert_mut(&mut self, pivots: &mut [usize; D]) -> Result<(), 
      LinAlgError>`: Inverts a square matrix purely in-place using
      caller-provided pivot scratch space.
    - `pub fn invert_into(&self, dest: &mut Matrix<T, D, D, S2>, pivots: 
    &mut [usize; D]) -> Result<(), LinAlgError>`: Computes the matrix inverse
      into a caller-provided destination matrix buffer.
    - **Symmetric Matrices**: Factorized via
      **$LDL^T$ Decomposition** ($A = L D L^T$).
    - **General Square Matrices**: Factorized via **LU Decomposition with
      Partial Pivoting** ($P A = L U$).
- **Determinant Calculation**:
    - `pub fn determinant(&self) -> T`: Computes $\det(A)$ in $O(N)$ time
      directly from the diagonal factors of an already-computed
      `LuDecomposition` or `LdltDecomposition` object.

#### 4.7. Matrix Decomposition Objects

Similar to structural specializations, matrix factorizations are exposed as
dedicated **Decomposition Objects**.

The struct definitions below match shipped code (`src/matrix/decomposition.rs`),
using concrete `const D: usize` parameters with `Const<D>: Dim` bounds so that
pivot scratch and temporary factors are stored in statically bounded stack
arrays:

```rust
/// LU Factorization with partial pivoting (PA = LU)
pub struct LuDecomposition<T, const D: usize>
where
    Const<D>: Dim,
{
    data: Owned<T, D, D>,
    pivots: [usize; D],
    row_exchanges: usize,
}

/// LDL^T Factorization for symmetric indefinite/positive-definite matrices (A = L D L^T)
pub struct LdltDecomposition<T, const D: usize>
where
    Const<D>: Dim,
{
    data: Owned<T, D, D>,
}

/// Cholesky Factorization for symmetric positive-definite matrices (A = L L^T)
pub struct CholeskyDecomposition<T, const D: usize>
where
    Const<D>: Dim,
{
    l: LowerTriangular<T, D>,
}

/// QR Factorization (A = Q R)
pub struct QrDecomposition<T, const R: usize, const C: usize>
where
    Const<R>: Dim,
    Const<C>: Dim,
{
    q: Owned<T, R, R>,
    r: UpperTriangular<T, R>,
}
```

##### 4.7.1. In-Place Factorization & Scratch Space API

To prevent stack bloat and avoid dynamic allocations, decomposition algorithms
provide in-place mutation methods where caller-provided scratch buffers or
mutable references act as the working state:

```rust
impl<T, const D: usize> Owned<T, D, D>
where
    Const<D>: Dim,
{
    /// Performs LU decomposition purely in-place on the matrix data.
    /// Overwrites self with L and U factors and populates the caller-provided pivot scratch array.
    pub fn lu_decompose_mut(&mut self, pivots: &mut [usize; D]) -> Result<usize, LinAlgError> {
        let mut row_exchanges = 0;
        // Low-level GETRF kernel execution using array bounds [usize; D]...
        Ok(row_exchanges)
    }

    /// Consumes the matrix to construct a stack-allocated LuDecomposition wrapper.
    pub fn into_lu(mut self) -> Result<LuDecomposition<T, D>, LinAlgError> {
        let mut pivots = [0usize; D];
        for i in 0..D {
            pivots[i] = i;
        }
        let row_exchanges = self.lu_decompose_mut(&mut pivots)?;
        Ok(LuDecomposition {
            data: self,
            pivots,
            row_exchanges,
        })
    }
}
```

##### 4.7.2. Linear System Solving via Decompositions

Decomposition objects expose specialized linear solver methods ($A x = b$)
utilizing forward and backward substitution over factor matrices:

```rust
impl<T, const D: usize> LuDecomposition<T, D>
where
    Const<D>: Dim,
{
    /// Solves A * x = b in-place by mutating the right-hand side vector b into the solution x.
    pub fn solve_mut<const COLS: usize>(&self, b: &mut Owned<T, D, COLS>) -> Result<(), LinAlgError>
    where
        Const<COLS>: Dim,
    {
        // 1. Permute rows of b according to self.pivots
        // 2. Solve L * y = P * b using forward substitution
        // 3. Solve U * x = y using backward substitution
        Ok(())
    }
}
```

#### 4.8. Interoperability & Conversions

##### 4.8.1. Conversion to Polynomial

A square matrix `Matrix<T, D, D, S>` converts to its characteristic polynomial
`Polynomial<T, <D as DimAdd<U1>>::Output>`.

- **Type Signature**:
  ```rust
  impl<T, D: Dim, S> TryFrom<Matrix<T, D, D, S>> for Polynomial<T, <D as DimAdd<U1>>::Output>
  where
      S: BlasStorage<T, D, D>,
      D: DimAdd<U1>,
      <D as DimAdd<U1>>::Output: Dim,
      T: Float + One + Copy,
  {
      type Error = ConversionError;
      // ...
  }
  ```
- **Behavior**: Coefficients are computed using the Faddeev-LeVerrier
  algorithm (Faddeev & Faddeeva, 1963).
- **Failure Condition**: Returns `ConversionError::DimensionMismatch` if the
  scalar type cannot perform division, if numerical overflow occurs or if
  capacity is insufficient.

##### 4.8.2. Conversion to Tensor

Converts a 2D matrix to a rank-2 `Tensor<T, Layout, B>`.

- **Type Signature**:
  ```rust
  impl<T, R: Dim, C: Dim, B: Buffer<T>, Layout: TensorLayout> From<Matrix<T, R, C, Dense<T, R, C, B>>> for Tensor<T, Layout, B>
  where
      Layout: TensorLayout<Size = <R as DimMul<C>>::Output>,
  {
      // Preserves backing buffer zero-copy when compile-time size and rank 2 match
  }
  ```
- **Behavior**: Maps column-major array storage directly into the flat buffer
  representation of the `Tensor`.
- **Infallible Compile-Time Bound**: Dimensions and rank are verified statically
  at compile time via `Layout: TensorLayout<Size = <R as DimMul<C>>::Output>`,
  eliminating runtime `LayoutMismatch` failure modes.

#### 4.9. Error Handling & Element Lookup

##### 4.9.1. Compile-Time Constraints

Dimension mismatches (e.g., adding matrices of different sizes or multiplying
incompatible dimensions) fail at compile-time. Rust's type checker prevents
compiling invalid math.

##### 4.9.2. Element Lookup & Leading-Dimension Offset

Element lookup on `Matrix<T, R, C, S>` where `S: MatrixStorage<T, R, C>`
evaluates coordinate offsets directly via `S::lda()` and `S::order()`:

```rust
impl<T, R: Dim, C: Dim, S> Matrix<T, R, C, S>
where
    S: MatrixStorage<T, R, C>,
{
    /// Safe bounds-checked coordinate lookup for MatrixStorage backends.
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if row >= R::USIZE || col >= C::USIZE {
            return None;
        }
        // Evaluates element offset according to storage layout order:
        // - Column-major: col * lda + row
        // - Row-major: row * lda + col
        let offset = if self.storage.order() == MatrixLayout::ColMajor {
            col * self.storage.lda() + row
        } else {
            row * self.storage.lda() + col
        };
        self.storage.as_slice().get(offset)
    }
}
```

*Codegen Constraint*: In release builds, a runtime branch `if row >= R::USIZE`
against generic `R: Dim` does not fold if `R` is an un-monomorphized type
parameter. Range-check elimination and branch folding require `R` and `C` to
resolve to fixed constants (`Const<N>`) so that nested array access (
`&[[T; R]; C]`) compiles to 0 branch instructions and 0 panic paths.

##### 4.9.3. Runtime Fallbacks

Dynamic operations that cannot be validated statically use soft failure paths:

- Matrix inversion returns a `Result<Self, LinAlgError>` instead of panicking,
  allowing control loops to handle singular conditions (e.g., falling back to a
  degraded state by returning `Err(LinAlgError::SingularMatrix)`).
- Boundary access returns `Option<&T>` via safe `get` methods.

#### 4.10. Structural Specializations & Extensions

Structural specializations are implemented as specialized storage leaves at Tier
3 (`Symmetric`, `Triangular` implementing `MatrixStorage`) paired with
high-level newtype wrappers around `Matrix`:

```rust
pub struct UpperTriangular<T, const D: usize, S = DenseArray<T, D, D>>(
    pub Matrix<T, Const<D>, Const<D>, S>,
);
pub struct LowerTriangular<T, const D: usize, S = DenseArray<T, D, D>>(
    pub Matrix<T, Const<D>, Const<D>, S>,
);
pub struct Symmetric<T, const D: usize, S = DenseArray<T, D, D>>(
    pub Matrix<T, Const<D>, Const<D>, S>,
);
```

This dual story provides complete consistency: storage leaves define physical
memory layout and bounds, while high-level newtype wrappers enforce mathematical
invariants, optimize solver algorithms ($LDL^T$, forward/backward substitution),
and dispatch specialized subprogram kernels.

Rather than packing triangular data (which requires complex
non-linear index mapping and prevents flat slicing), we wrap a full square
matrix. This trades memory space for cache friendliness and compatibility with
slice-based BLAS kernels.

##### 4.10.1. Forward and Backward Substitution Examples

```rust
/// Solves L * x = b for a lower triangular matrix where L is D x D and x, b are D x 1.
pub fn solve_lower_triangular<T, const D: usize>(
    l: &LowerTriangular<T, D>,
    b: &Matrix<T, Const<D>, U1>,
    tolerance: T,
) -> Result<Matrix<T, Const<D>, U1>, LinAlgError>
where
    Const<D>: Dim,
    T: Float + Copy,
{
    let n = D;
    let mut x = Matrix::<T, Const<D>, U1>::zero();
    let l_mat = &l.0;

    for i in 0..n {
        let l_ii = l_mat.get(i, i).copied().unwrap_or(T::ZERO);
        // Tolerance-based singularity check using the type's Signed abs()
        if l_ii.abs() < tolerance {
            return Err(LinAlgError::SingularMatrix);
        }
        let mut sum = T::ZERO;
        for j in 0..i {
            let l_ij = l_mat.get(i, j).copied().unwrap_or(T::ZERO);
            let x_j = x.get(j, 0).copied().unwrap_or(T::ZERO);
            sum = sum + l_ij * x_j;
        }
        let b_i = b.get(i, 0).copied().unwrap_or(T::ZERO);
        if let Some(x_i) = x.as_mut_slice().get_mut(i) {
            *x_i = (b_i - sum) / l_ii;
        }
    }
    Ok(x)
}
```

##### 4.10.2. Companion Matrix Root-Finding

For polynomial root-finding, the coefficients are mapped to a companion matrix
in upper Hessenberg form (strict zeros beneath the first lower subdiagonal).
Instead of using a general $O(N^3)$ QR algorithm, the solver exploits the
unitary-plus-rank-one structure (Aurentz et al., 2014). This reduces storage
requirements to $O(N)$ and computational complexity to $O(N^2)$ flops. Applying
a sequence of planar rotators guarantees normwise backward stability.

##### 4.10.3. Kalman Filter State Update

The following example demonstrates the proposed `Matrix` API when computing the
covariance update in a Kalman filter loop:
$$ P*{k|k} = (I - K_k H_k) P*{k|k-1} $$

```rust
use control_rs::matrix::{Matrix, Dim, U1};

pub fn kalman_covariance_update<T, S: Dim, O: Dim>(
    p_pred: &Matrix<T, S, S>,
    k: &Matrix<T, S, O>,
    h: &Matrix<T, O, S>,
) -> Matrix<T, S, S>
where
    T: Scalar + Copy,
    S: Dim,
    O: Dim,
    S: DimMul<S>,
    S: DimMul<O>,
    O: DimMul<S>,
{
    // Identity matrix I of state dimension S
    let i = Matrix::<T, S, S>::identity();

    // K * H -> S x S matrix
    let k_h = k * h;

    // I - K * H -> S x S matrix
    let diff = &i - &k_h;

    // (I - K * H) * P_pred -> S x S matrix
    &diff * p_pred
}
```

###### 4.10.4. Abstracting Target-Specific DSP / BLAS FFI

When hardware acceleration (e.g., CMSIS-DSP, ARM NEON, or vendor-specific
DSPLib) is enabled, underlying BLAS traits dispatch calls to FFI functions.

- **Wrapped Unsafe Functions**: External foreign function interfaces (FFI)
  accepting raw pointers.
- **Safety Preconditions & Invariants**:
    - C-based FFI routines do not perform bounds checking and assume that the
      caller has allocated sufficient, correctly-aligned memory.
    - The `Matrix` type acts as a guard by statically verifying all dimension
      constraints at compile time (using Peano types). It ensures that the
      buffers passed to FFI calls have the precise size expected by the hardware
      kernels, preventing memory corruption or CPU faults.

---

### 5. Alternatives

#### 5.1. Convenience Methods vs. Explicit Decompositions

We evaluated exposing convenient, immutable linear algebra signatures like
`invert(&self) -> Matrix<T, D, D>`. We explicitly rejected this pattern because:

- **Hidden Stack Allocations**: Returning new matrix structures from
  heavy $O(N^3)$ operations masks large internal stack allocations, risking
  unpredictable hard faults on stack-constrained embedded targets.
- **Redundant Factorization Computation**: Hiding factorizations behind
  convenience methods forces subsequent operations (e.g., calculating
  determinants or solving multiple right-hand side vectors) to recompute
  factorizations from scratch.
- **Explicit `no_alloc` Alternative**: Linear algebra operations require
  explicit decomposition objects (`LuDecomposition`, `LdltDecomposition`) or
  in-place mutation methods (`invert_mut`, `solve_mut`) using caller-provided
  scratch space.

#### 5.2. Const Generics vs. Type-Level Traits (`Dim`)

We evaluated using raw const generics (`[[T; R]; C]`) as the primary matrix
interface versus type-level dimension traits (`Dim`).

- **Raw Const Generics Limitations**: Stable Rust currently limits const generic
  arithmetic in public trait bounds (e.g., expressing that
  multiplying $M \times N$ by $N \times P$ yields $M \times P$).
- **Selected `Dim` + Decoupled Storage Architecture**: Combining the `Dim` trait
  system with the decoupled `MatrixStorage<T, R, C>` trait enables complex
  compile-time matrix arithmetic bounds while keeping storage backends
  completely pluggable.

#### 5.3. External Libraries (`nalgebra`)

Using external crates like `nalgebra` in `no_std` mode was considered and
bypassed for two primary reasons:

1. **Generic `const fn` Support on Stable Rust**: `nalgebra` relies on
   traits like `Default` which cannot be evaluated inside `const fn` on stable
   Rust.
2. **Custom `Zero` and `One` traits**: expose associated constants (`T::ZERO`/
   `T::ONE`).
3. **Audit Footprint**: Complete auditing of dependency source code is
   difficult with more dependencies. `nalgebra` has a large number of
   dependencies.

While `nalgebra` was bypassed as a direct dependency, the matrix architecture is
a direct adaptation of `nalgebra`'s design. Key design structures, trait
hierarchies, dimensions, and slicing properties are structurally modeled on
Sébastien Crozet's original architecture.

#### 5.4. Decoupled Storage Architecture

The physical memory layout is decoupled from mathematical dimensions via the
three-tier storage trait hierarchy (`storage-subprograms-design.md` §4.1):

- **Tier 0 Provenance (`Buffer`/`BufferMut`)**: Distinguishes an owning,
  stack-allocated
  buffer from borrowed (`Ref`, `RefMut`) nested arrays.
- **Tier 1 Universal Shape Floor (`BlasStorage`/`BlasStorageMut`)**: Mandates
  `as_slice()`
  and `as_mut_slice()` for universal contiguous memory access.
- **Tier 2 Addressing
  Branches (`MatrixStorage`/`PackedStorage`)**:
  Branch according to addressing capability: leading dimension (`lda`/`order`)
  or packed (`packed_index`/`IMPLICIT`).
- **Concrete
  Leaves (`Dense`, `Symmetric`, `Triangular`, `Diagonal`)**:
  Each implements exactly one Tier 2 branch. There is no `StridedView` leaf.

#### 5.5. Factorization & Inversion Algorithms

For solving linear systems and matrix inversion, the following factorization
algorithms were analyzed with their trade-offs for embedded deployment:

- **LU Factorization (with Partial Pivoting)**:
    - _Pros_: General-purpose; works on any non-singular square matrix. Pivoting
      prevents division by small values, preserving numeric stability (Golub &
      Van Loan, 2013).
    - _Cons_: Pivoting requires row-swapping logic, which complicates loop
      unrolling and SIMD optimization. It has a higher constant factor overhead
      than Cholesky/LDL^T ($O(2N^3/3)$ operations).
- **QR Factorization (via Givens Rotations or Householder Reflections)**:
    - _Pros_: Extremely stable numerically, even for poorly conditioned or
      singular-prone systems.
    - _Cons_: Highly computationally expensive ($O(4N^3/3)$ operations). Givens
      rotations require many square root and trigonometric function calls,
      making it slow on microcontrollers lacking hardware FPU support.
- **Cholesky Factorization ($LL^T$)**:
    - _Pros_: Highly efficient ($O(N^3/3)$ operations, half the operations of
      LU) and exhibits excellent numerical stability for positive-definite
      symmetric matrices.
    - _Cons_: Restricted strictly to symmetric positive-definite matrices.
      Requires calculating square roots for each diagonal element, which
      typically takes many CPU cycles and increases quantization errors in
      fixed-point representations.
- **$LDL^T$ Factorization**:
    - _Pros_: Chosen as the default solver for symmetric matrices. Like
      Cholesky, it requires only $O(N^3/3)$ operations. By decomposing the
      matrix into $L D L^T$ (where $L$ is unit lower-triangular and $D$ is
      diagonal), it completely avoids square root calculations. This preserves
      scaling boundaries in fixed-point formats and optimizes CPU cycle counts (
      Higham, 2002).
    - _Cons_: Restricted to symmetric matrices. If the matrix is near-singular
      or indefinite, it may suffer from numerical instability without complex
      block-pivoting algorithms (e.g., Bunch-Kaufman).
- **Normal Equation Solving (Forming $A^T A$)**:
    - _Pros_: Allows solving non-symmetric or rectangular systems ($A x = b$) by
      converting them to a symmetric system ($A^T A x = A^T b$) and applying
      efficient symmetric solvers (Cholesky/LDL^T).
    - _Cons_: Strongly avoided. Forming $A^T A$ squares the condition number of
      the matrix ($\kappa(A^T A) = \kappa(A)^2$), which halves the number of
      valid decimal digits in calculations and leads to severe precision loss.

#### 5.6. Matrix Multiplication Algorithms

To evaluate $C = A B$, several multiplication approaches were compared:

- **Naive Row-by-Column (Triple Loop, $O(N^3)$)**:
    - _Pros_: Tiny code footprint, no temporary buffer requirements,
      and trivial for the compiler to optimize or auto-vectorize for very small
      dimension limits ($N \le 8$).
    - _Cons_: For larger dimensions (e.g., $N = 32$), this approach suffers from
      high L1 cache miss rates due to non-contiguous memory access in
      column-major matrices.
- **Block-Based (Tiled) Multiplication**:
    - _Pros_: Restructures the triple loop into sub-matrix
      blocks ($k_c \times n_R$) to fit inside the CPU's cache line size,
      drastically reducing memory bus transactions for larger
      matrices ($N > 32$).
    - _Cons_: Adds complex index boundary math and loop nesting, which increases
      target binary size and introduces instruction overhead that outweighs
      cache benefits for small embedded matrices ($N \le 32$).
- **Vectorized SIMD / Hardware BLAS FFI**:
    - _Pros_: Directly utilizes SIMD registers (such as ARM NEON or CMSIS-DSP
      assembly instructions) to perform multiple multiply-accumulate operations
      per cycle.
    - _Cons_: Bypasses safe Rust controls by passing raw pointers to FFI
      functions. It is highly hardware-specific and requires fallback
      implementations for targets lacking SIMD engines.

#### 5.7. Determinant Calculation Algorithms

For computing $\det(A)$, two primary methods were analyzed:

- **Leibniz Formula / Cofactor Expansion**:
    - _Pros_: Does not require factorization or modifications to the matrix
      data. Highly efficient and division-free for tiny dimensions ($2 \times 2$
      or $3 \times 3$).
    - _Cons_: Factorial complexity ($O(N!)$). Computing the determinant of
      a $32 \times 32$ matrix using cofactor expansion is mathematically
      impossible in real-time.
- **Factorization-Based**:
    - _Pros_: Uses the LU or $LDL^T$ decomposition result. Since the determinant
      of a triangular matrix is the product of its diagonal elements, $\det(A)$
      is computed in $O(N)$ additional operations after factorization.
      Numerically stable and scales to $N=32$.
    - _Cons_: Requires running a full matrix factorization first, which is
      fallible (e.g., singular matrices return zero determinant or error).

---

### 6. Verification & Validation

The matrix implementation is verified and validated across four structured
pillars to guarantee mathematical correctness, embedded safety, and real-time
execution predictability.

#### 6.1. Verification Strategy

1. **Compile-Time Verification**:
    - Matrix dimension matching ($M \times N \times N \times P \to M \times P$)
      is strictly enforced by the Rust type system using Peano types (`Dim`),
      completely eliminating runtime dimension checks and preventing invalid
      pointer arithmetic at compile time.
2. **Property & Unit Testing**:
    - Host-based unit tests execute via `cargo test` to verify constructors,
      operators, triangular solvers, and slice bounds.
    - Property-based testing via `proptest` mathematically proves algebraic
      matrix identities (e.g., $(AB)^T = B^T A^T$, $A(B+C) = AB + AC$) over
      thousands of generated inputs.
    - Ill-conditioned, near-singular, and Hilbert matrices are tested to confirm
      safe, panic-free error degradation (`Err(LinAlgError::SingularMatrix)`).
3. **Hardware-in-the-Loop (HIL) & Cache Profiling**:
    - Cross-compiled binaries run on physical target microcontrollers (e.g., ARM
      Cortex-M4/M7) to profile L1 data/instruction cache misses (`I1mr`/`D1mr`),
      FPU cycle counts ($c_{\text{inner}}$), and hardware pipeline stall
      dependencies.
    - Cycle time for matrix multiplication is validated against the execution
      model:
      $$T \approx \frac{(n \cdot m \cdot k \cdot c_{\text{inner}}) + c_{\text{overhead}}}{f}$$
4. **Stack Bounds Verification**:
    - Inline stack-allocated matrix capacities are strictly capped
      at $127 \times 127$ elements ($R::USIZE \times C::USIZE \le 16{,}129$;
      C-3),
      matching the `Dim` system's `U127` ceiling rather than an independently
      chosen number.
    - This is a type-system bound, not a stack-safety guarantee: a
      $127 \times 127$ `f32` instance is
      $16{,}129 \times 4\text{ bytes} \approx 63\text{KB}$, well past typical
      2–8KB bare-metal stack budgets. `clippy::large_stack_arrays`
      (`storage-subprograms-design.md` §8) is the actual enforcement point for
      call-site instance size; CI must fail on any un-justified `#[allow]`
      of that lint.

#### 6.2. Validation Strategy

1. **Kalman Filter Covariance Update**: Validate end-to-end numeric integrity
   using the discrete Kalman filter covariance
   update ($P_{k\vert{}k} = (I - K_k H_k) P_{k\vert{}k-1}$).
   In state estimation (like a discrete Kalman filter running on a
   microcontroller), you must update the error covariance matrix using the
   formula $P_{k\vert{}k} = (I - K_k H_k) P_{k\vert{}k-1}$. This example
   demonstrates how the `Matrix` API handles matrix arithmetic and identity
   generation without heap allocation.

   ```rust
   use control_rs::matrix::{Matrix, Dim, U2, U1};

   /// Updates the 2x2 error covariance matrix for a 2D state vector (e.g., Position, Velocity)
   /// given a 1D measurement (e.g., GPS position).
   pub fn update_error_covariance(
       p_pred: &Matrix<f32, U2, U2>, // Predicted covariance (2x2)
       k: &Matrix<f32, U2, U1>,      // Kalman Gain (2x1)
       h: &Matrix<f32, U1, U2>,      // Observation model (1x2)
   ) -> Matrix<f32, U2, U2> {
       // 1. Generate a 2x2 Identity matrix
       let i = Matrix::<f32, U2, U2>::identity();

       // 2. Compute K * H -> (2x1) * (1x2) = (2x2)
       let k_h = k * h;

       // 3. Compute (I - K * H) -> (2x2)
       let diff = &i - &k_h;

       // 4. Compute final updated covariance: (I - K * H) * P_pred
       &diff * p_pred
   }
   ```

2. **External Integration**: Pass contiguous slice views (`as_slice()`) directly
   to hardware vendor libraries (ARM CMSIS-DSP, MCUXpresso DSPLib) without
   copying data.
3. **Control System Demos**: Execute step-response simulations and closed-loop
   state-space control loops in `examples/`.

---

### 7. Risks & Open Questions

- **Const Generics Complexity**: Stabilized const generics are still limited.
  Custom trait bounds (like `DimAdd`, `DimMul`) might increase compile times and
  create verbose error messages.
- **Precision vs. Performance Trade-off**: Deciding whether to utilize
  `-ffast-math` or rely on strict IEEE 754 compliance for float math.
- **Fixed-Point Precision Loss**: Truncation errors in Q31/Q15 accumulator
  scaling might lead to drift in high-frequency loops.

---

### 8. Development Plan

| Task / Feature               | Description                                                                             | Estimated Effort |
|:-----------------------------|:----------------------------------------------------------------------------------------|:-----------------|
| **Step 1: Core Layout**      | Define `Matrix` struct, column-major storage, and slice casting.                        | 1.0 Day          |
| **Step 2: Operators**        | Implement `Add`, `Sub`, `Mul` traits with compile-time checks.                          | 1.5 Days         |
| **Step 3: Solvers**          | Implement $LDL^T$ decomposition, LU, determinants, and matrix inversion.                | 2.0 Days         |
| **Step 4: Specializations**  | Create `UpperTriangular`, `LowerTriangular`, and `Symmetric` wrappers.                  | 1.0 Day          |
| **Step 5: Factorizations**   | Implement Cholesky and QR solvers.                                                      | 2.0 Days         |
| **Step 6: Verification**     | Set up `proptest` suites, ARM DWT cycle profiling, and Cachegrind setups.               | 1.5 Days         |
| **Step 7: Interoperability** | Implement conversions between `Matrix`, `Polynomial` (Faddeev-LeVerrier), and `Tensor`. | 2.0 Days         |

---

### 9. References

1. **Golub, G. H., & Van Loan, C. F. (2013).** _Matrix Computations_ (4th ed.).
   Johns Hopkins University Press. — Flop-count basis for in-place
   factorizations ($O(N^3/3)$ Cholesky/$LDL^T$, $O(2N^3/3)$ LU, $O(4N^3/3)$ QR).
2. **Anderson, E., et al. (1999).** _LAPACK Users' Guide_ (3rd ed.). SIAM. —
   Reference performance/blocking conventions behind BLAS-backed solver
   routines.
3. **Frison, G., et al. (2018).** BLASFEO: Basic Linear Algebra Subroutines for
   Embedded Optimization. _ACM Transactions on Mathematical Software_. — Direct
   embedded runtime benchmarks and panel-major vs. column-major memory-layout
   comparison.
4. **Bini, D. A., Boito, P., Eidelman, Y., Gemignani, L., & Gohberg, I. (2010).
   ** A Fast Implicit QR Eigenvalue Algorithm for Companion Matrices. _Linear
   Algebra and its Applications_, 432(8), 2006–2031. —
   Explicit $O(N^3) \to O(N^2)$ time and $O(N)$ space reduction for
   companion-matrix eigenvalue solving.
5. **Aurentz, J. L., Mach, T., Vandebril, R., & Watkins, D. S. (2014).** Fast
   and backward stable computation of roots of polynomials. _TW Reports_, KU
   Leuven. — Speed-vs-backward-stability trade-off evaluation for companion
   matrix polynomial rootfinding.
6. **Higham, N. J. (2002).** _Accuracy and Stability of Numerical Algorithms_ (
   2nd ed.). SIAM. — Condition-number and error-bound analysis underpinning the
   rule against forming $A^T A$ and pivoting stability rules.
7. **Yiu, J. (2013).** _The Definitive Guide to ARM Cortex-M3 and Cortex-M4
   Processors_ (3rd ed.). Newnes. — FPU register count (32 single-precision
   registers) and micro-architectural execution constraints.
8. **Demmel, J. W. (1997).** _Applied Numerical Linear Algebra_. SIAM. —
   Reference textbook for standard numerical linear algebra algorithms and
   conventional BLAS/LAPACK routine mapping.
9. **Faddeev, D. K., & Faddeeva, V. N. (1963).** _Computational Methods of
   Linear Algebra_. W. H. Freeman and Company. — Classical derivation behind the
   division-free Faddeev–LeVerrier matrix characteristic polynomial formulation.
10. **Claessen, K., & Hughes, J. (2000).** QuickCheck: A Lightweight Tool for
    Random Testing of Haskell Programs. _ACM SIGPLAN Notices_, 35(9), 268–279. —
    Random generation and shrinking methodology behind property-based test
    suites (`proptest`).
11. **Rust Project Developers. (2024).** _The Rustonomicon: The Dark Arts of
    Advanced and Unsafe Rust Programming_. — Memory-aliasing and layout
    guarantees underpinning the `BlasStorage`/`MatrixStorage` trait split and
    `#[repr(C)]`
    slice casting.
12. **ISO. (2018).** _ISO 26262-6:2018 Road vehicles — Functional safety — Part
    6: Product development at the software level_. — Automotive functional
    safety requirements governing static allocation and WCET determinism.
13. **RTCA / EUROCAE. (2011).** _DO-178C: Software Considerations in Airborne
    Systems and Equipment Certification_. — Airborne software verification and
    determinism standards.
14. **IEEE Computer Society. (2008).** _IEEE Standard for Software and System
    Test Documentation_ (IEEE Std 829-2008). — Software verification and test
    suite structure standards.
15. **control-rs. (2026).** `src/math/subprograms.rs`. — Level-1/2/3 subprogram
    trait definitions (`AXPY`, `SCAL`, `DOT`, `NRM2`, `IAMAX`, `GEMV`, `GEMM`);
    the inventory of kernels available to `Matrix` (§4.5.1).
16. **Greif, C., He, S., & Liu, P. (2016).** SYM-ILDL: Incomplete $LDL^T$
    Factorization of Symmetric Indefinite and Skew-Symmetric Matrices.
    _arXiv:1505.07589_. — $O(n)$ per-step pivot-search cost for symmetric
    partial pivoting, bounding the `IAMAX` work per elimination step.
17. **Higham, N. J., & Tisseur, F. (2000).** A Block Algorithm for Matrix
    1-Norm Estimation, with an Application to 1-Norm Pseudospectra. _SIAM
    Journal on Matrix Analysis and Applications_, 21(4).
    doi: 10.1137/S0895479899356080. — Multiple-right-hand-side triangular
    solves arising in LU-based solver paths.

---

### 10. Revision History

| Revision | Date            | Author          | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
|:---------|:----------------|:----------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1.0      | July 12, 2026   | @MitchellDScott | Initial draft outlining core concepts, layout, and operations.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| 1.1      | July 19, 2026   | @MitchellDScott | Restructured to new template; added embedded performance and verification details.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| 1.2      | July 25, 2026   | @MitchellDScott | Added supporting bibliography and inline citations.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| 1.3      | July 26, 2026   | @MitchellDScott | Added Decomposition Objects, zero-copy MatrixView wrappers, and no_alloc scratch space patterns.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| 1.4      | July 26, 2026   | @MitchellDScott | Harmonized with storage trait design doc; updated `Matrix` definition, bounds, decomposition rules, and V&V.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| 1.5      | July 26, 2026   | @MitchellDScott | Added comprehensive 3-tiered bibliography and inline citations across core architectural sections.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| 1.6      | August 1, 2026  | @MitchellDScott | Corrected `nalgebra` comparison claims; clarified storage-decoupling benefit; added system-solving convenience note.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| 1.7      | August 2, 2026  | @MitchellDScott | Propagated `num-traits-design.md` pivot; removed duplicate MatrixView definitions; relocated `ConversionError`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| 1.8      | August 10, 2026 | @MitchellDScott | Realigned with updated math-module code                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| 1.10     | August 11, 2026 | @MitchellDScott | Condensed §4.5 operator→subprogram mapping into a scannable table (`Add`/`Sub`→`AXPY`, `Neg`→`SCAL`, `Mul<Matrix>`→`GEMM`, `Mul<Vector>`→`GEMV`).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| 1.13     | August 15, 2026 | @mitchelldscott | Reverted Doc Status to Draft. `storage-subprograms-design.md` rev 1.2 (August 14, 2026) replaced the flat `Storage`/`StorageMut`/`ContiguousStorage`/`ContiguousStorageMut` hierarchy this document's §1, §4.1, §4.2, §4.3, §4.3.1 and §5.4 still describe with a four-tier `Buffer`/`BlasStorage`/{`MatrixStorage`, `PackedStorage`, `StridedStorage`}/leaf hierarchy; `ArrayStorage`, `StorageView` and `StorageViewMut` no longer name a type that hierarchy defines. No section content changed in this revision; reconciliation (renaming `Matrix`'s `S` bound to the correct Tier-2 branch and updating the architecture prose) is tracked as its own maintenance pass.                   |
| 1.14     | August 16, 2026 | @mitchelldscott | Propagated four-tier `BlasStorage` hierarchy (§1, §4.1, §4.3, §4.3.1, §5.4): updated `Matrix` bound to `MatrixStorage<T, R, C>`, default storage alias to `Dense<T, R, C, Array<T, N>>`, slice methods to `BlasStorage`, and view aliases to `Dense` / `StridedView`. Applied B-2/D-2 resolution (§4.5.1): added `GER` and `TRSV` to required subprogram inventory.                                                                                                                                                                                                                                                                                                                             |
| 1.15     | August 16, 2026 | @mitchelldscott | Harmonized with `storage-subprograms-design.md` Rev 1.4 (§1, §4.1): updated `Matrix` storage bound to single-parameter `S: MatrixStorage<T, R = R, C = C>` with associated `R`/`C` types, integrated `FixedBlasStorage<T>` array access (`as_array()`), and detailed monomorphized zero-cost subprogram delegation.                                                                                                                                                                                                                                                                                                                                                                             |
| 1.16     | August 16, 2026 | @mitchelldscott | Harmonized with `storage-subprograms-design.md` Rev 1.5 (§1, §4.1): updated `BlasStorage<T, R, C, Stride>` and `MatrixStorage<T, R, C>` to keep `R: Dim`, `C: Dim`, `Stride: Dim` as generic parameters on the storage trait itself, enabling one storage implementor (`Dense`, `Array`, `Ref`, `RefMut`) to be used for any compile-time shape and stride.                                                                                                                                                                                                                                                                                                                                     |
| 1.17     | August 16, 2026 | @mitchelldscott | Completed `/cr-design-doc` pass (§1-§10): standardized typed `BlasStorage`/`MatrixStorage` + `&[T; N]` operand model; restored unique `MX-FR-*`/`MX-NFR-*`/`MX-C-*` requirement IDs; specified feature-selected backend dispatch (`#[cfg(feature = "accelerate")]`); aligned decomposition signatures (`LuDecomposition`, `LdltDecomposition`, `CholeskyDecomposition`, `QrDecomposition`) with shipped `src/matrix/decomposition.rs` code (`Const<D>: Dim`); eliminated `lu_decompose_mut` runtime `assert!`; added Tier-2 storage branch element lookup composition (`get`) with `R::USIZE` codegen folding constraint; unified `Symmetric`/`Triangular` leaf + newtype specialization story. |
| 1.18     | August 16, 2026 | @mitchelldscott | Reconciled residual `Storage<T, R, C>` prose references in §5.1 and §8 with `MatrixStorage<T, R, C>` and `BlasStorage`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| 1.19     | August 16, 2026 | @mitchelldscott | Standardized requirement IDs in §2 from MX- prefixed tags to plain FR/NFR/C numbered format.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| 1.20     | August 16, 2026 | @mitchelldscott | Updated `Matrix` default storage parameter and view aliases (`Owned`, `MatrixSlice`, `MatrixSliceMut`) to convenience storage aliases (`DenseArray`, `DenseRef`, `DenseRefMut`).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| 1.21     | August 16, 2026 | @mitchelldscott | Encapsulated dimension multiplication inside `DenseArray<T, R, C>` in `storage.rs`, eliminating non-stable const generic math (`{ R * C }`) and extra capacity parameters from `Matrix`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| 1.22     | August 16, 2026 | @mitchelldscott | Removed obsolete `FixedBlasStorage` mention; decoupled `R: DimMul<C>` from `Matrix` struct definition; corrected decomposition `Owned<T, Const<D>, Const<D>>` type arguments; aligned `solve_lower_triangular` generic dimension parameters with `LowerTriangular<T, const D: usize>`; refined `Matrix::get` element lookup offset evaluation.                                                                                                                                                                                                                                                                                                                                                  |
| 1.23     | August 18, 2026 | @mitchelldscott | Reconciled `Matrix` struct storage bound to `S: MatrixStorage<T, R, C>` across §1 and §4.1; clarified `StridedMatrixSlice` as a standalone wrapper over `StridedView`; harmonized §4.9.2 coordinate lookup with Tier-2 `MatrixStorage`; aligned with `storage-subprograms-design.md` Rev 1.10 `&[T; N]` operand model.                                                                                                                                                                                                                                                                                                                                                                       |
| 1.24     | August 18, 2026 | @mitchelldscott | Aligned §4.8.2 `Matrix` → `Tensor` conversion to infallible `From` bounded by `TensorLayout<Size = <R as DimMul<C>>::Output>`, eliminating obsolete `LayoutMismatch` runtime check.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| 1.25     | August 18, 2026 | @mitchelldscott | Propagated `storage-subprograms-design.md` Rev 1.11–1.12: `ArrayMatrix`/`Owned` take `const R, const C` over `DenseArray<T, R, C>`/`Array2`; `Matrix` itself stays `Dim`-generic with no owning default; views via FR-6 `view()`/`try_submatrix()`; subprogram operands via `as_array()`. |
| 1.26     | August 18, 2026 | @mitchelldscott | Propagated `storage-subprograms-design.md` Rev 1.16: `as_nested()` over `&[[T; R]; C]`; caller-supplied `R2`/`C2`/`LDA` on `try_submatrix`; kernel `trans` only (no `transpose_view` / `StridedView`); `BlasStorage` without `Stride`; `IMPLICIT` packed branch. |
