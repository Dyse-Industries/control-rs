# Matrix Type & Structural Specializations (Design Document)

![Date Badge](https://img.shields.io/badge/Date-July_19,_2026-blue)
![Status Badge](https://img.shields.io/badge/Doc%20Status-Draft-orange)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

---

### 1. Introduction

This module provides the matrix operations required by advanced control and
state estimation. Its architecture is directly modeled on
the `nalgebra` linear algebra crate (licensed under BSD-3-Clause, by Sébastien
Crozet).

The following elements of `nalgebra` are directly mirrored or adapted in our
architecture:

- **Matrix Signature**: `Matrix<T, R: Dim, C: Dim, S: Storage<T, R, C>>` is
  identical to `nalgebra`'s.
- **Storage Trait Hierarchy**: `control-rs` extends this with its own
  `ContiguousStorage`/`ContiguousStorageMut` traits (see below).
- **ArrayStorage Structure**: `ArrayStorage<T, R, C>` wrapping `[[T; R]; C]`
  with `#[repr(transparent)]` (or `#[repr(C)]` on storage types) for contiguous
  stack storage.
- **Matrix Views**: `MatrixView` and `MatrixViewMut` providing zero-copy
  non-destructive window slices over memory, equivalent to `ViewStorage` and
  `ViewStorageMut`.

---

### 2. Requirements

#### 2.1. Functional Requirements

- **FR-1 — Compile-Time Sizing**: Enforce dimensions of arguments at compile
  time using `num_types`.
- **FR-2 — Static Constructors**: Provide compile-time evaluated constructors
  for zero, identity, and diagonal matrices.
- **FR-3 — Core Arithmetic**: Implement standard operator overloading for
  addition, subtraction, multiplication, and negation.
- **FR-4 — Matrix Operations**: Provide methods for transposition, determinant
  calculation, and inversion.
- **FR-5 — Specializations**: Support specialized structures (Upper Triangular,
  Lower Triangular, Symmetric) to dispatch optimized routines.
- **FR-6 — Coordinate-Based Instantiation**: Expose coordinate-based mapping
  functions to initialize elements by index.
- **FR-7 — Matrix Concatenation**: Implement a method to combine matrices and
  lists of matrices both vertically and horizontally.
- **FR-8 — Type Conversions**: Support conversions between `Matrix`,
  `Polynomial`, and `Tensor` representations for compatible ranks and sizes.

#### 2.2. Non-Functional Requirements

- **NFR-1 — Deterministic Execution**: Matrix operations must execute within
  predictable, deterministic timeframes.
- **NFR-2 — No Excessive Compile-Time Overhead**: `const fn` static constructors
  and dimension enforcement must not cause excessive compile-time increase or
  binary bloat.
- **NFR-3 — Specialization Optimization**: Structural specializations should run
  in about half the operations of a regular matrix multiplication.

#### 2.3. Constraints

- **C-1 — No-Std Environment**: The code must compile and run in `#![no_std]`
  environments without the Rust standard library.
- **C-2 — No Dynamic Allocation**: The module must not use a heap allocator; all
  memory allocations are static or stack-based.
- **C-3 — Memory Footprint**: Maximum matrix dimensions are limited
  to $32 \times 32$ elements, keeping a single instance under 4KB of stack (
  32-bit floats).

---

### 3. Technical Overview

`Matrix` is a type-safe wrapper that provides compile time dimension and type
bounds to guarantee that unsafe subprograms are not misused. Beyond safety,
the API is also designed to make expressing and solving linear systems —
factorization, inversion, and substitution — convenient for callers.

---

### 4. Core Architecture

#### 4.1. Generics Foundation & Sizing

The core `Matrix` structure decouples mathematical dimensions from physical
storage using the `Storage<T, R, C>` trait hierarchy:

```rust
pub struct Matrix<T, R: Dim, C: Dim, S: Storage<T, R, C> = ArrayStorage<T, R, C>> {
    storage: S,
    _marker: core::marker::PhantomData<(R, C)>,
}
```

The `Dim` trait and Peano number representations defined in
[num_types](../../src/math/num_types.rs) perform type-level
arithmetic (e.g., dimension addition or multiplication) to statically verify
shape changes at compile time, while `S` determines where and how elements are
physically stored in memory.

This is the actual point of decoupling storage from the matrix: one
`Matrix<T, R, C, S>` implementation — one set of arithmetic, factorization,
and conversion routines — operates unmodified over any conforming storage
backend (stack array, borrowed view, or memory-mapped DMA register), instead
of requiring a separate implementation per backend.

#### 4.2. Memory Layout & Storage Strategy

The default storage backend (`ArrayStorage`) uses a **column-major array
representation** (`[[T; R]; C]`). `Matrix`'s own arithmetic never branches 
on layout — it reads `S::ORDER` off whichever storage backend it was 
instantiated with and forwards it. This is what makes it possible to 
introduce a row-major backend later without touching a single `Matrix` 
operator implementation.

- **Cache Locality**: Matrix multiplication: Under column-major ordering, each
  column $ A_j $ is contiguous in memory, maximizing CPU cache hit rates.
- **BLAS Interoperability**: Column-major layout matches the standard convention
  of legacy BLAS/LAPACK (Anderson et al., 1999) and embedded DSP libraries (
  e.g., ARM CMSIS-DSP),
  allowing zero-copy routing to hardware-accelerated kernels. Panel-major vs.
  column-major memory-layout performance trade-offs for embedded optimization
  are evaluated in BLASFEO (Frison et al., 2018).

#### 4.3. Memory Representation, Slicing & Views

To ensure stable memory layout and compatibility with C-based hardware
libraries the matrix owns a contiguous array marked as `#[repr(C)]`.

Flat slice interfaces are safely gated behind the `ContiguousStorage` sub-traits
to prevent leaking padded or strided memory as valid contiguous slices:

```rust
impl<T, R: Dim, C: Dim, S> Matrix<T, R, C, S>
where
    S: ContiguousStorage<T, R, C>,
{
    /// Exposes a safe contiguous slice view of matrix memory.
    pub fn as_slice(&self) -> &[T] {
        self.storage.as_slice()
    }
}

impl<T, R: Dim, C: Dim, S> Matrix<T, R, C, S>
where
    S: ContiguousStorageMut<T, R, C>,
{
    /// Exposes a safe mutable contiguous slice view of matrix memory.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.storage.as_mut_slice()
    }
}
```

##### 4.3.1. Zero-Copy Non-Destructive Views (`MatrixView` & `MatrixViewMut`)

To support zero-copy operations (such as non-destructive transposition) without
requiring memory copies or stack allocations, `control-rs` provides
reference-holding view types:

```rust
pub struct MatrixView<'a, T, R: Dim, C: Dim> {
    data: &'a [T],
    row_stride: usize,
    col_stride: usize,
    _marker: core::marker::PhantomData<(R, C)>,
}

pub struct MatrixViewMut<'a, T, R: Dim, C: Dim> {
    data: &'a mut [T],
    row_stride: usize,
    col_stride: usize,
    _marker: core::marker::PhantomData<(R, C)>,
}

/// A high-level Matrix wrapper around an immutable MatrixView storage backend.
pub type MatrixSlice<'a, T, R, C> = Matrix<T, R, C, MatrixView<'a, T, R, C>>;

/// A high-level Matrix wrapper around a mutable MatrixViewMut storage backend.
pub type MatrixSliceMut<'a, T, R, C> = Matrix<T, R, C, MatrixViewMut<'a, T, R, C>>;
```

- **Non-Destructive Transposed Views**:
  `pub fn transpose_view(&self) -> MatrixSlice<'_, T, C, R>` creates a
  transposed $C \times R$ view over the original matrix data with swapped
  striding rules (`new_row_stride = old_col_stride` and
  `new_col_stride = old_row_stride`),
  incurring zero allocation or byte copying.
- **In-Place Transposition**: For square matrices ($R = C$), in-place element
  swapping (`pub fn transpose_mut(&mut self)`) mutates elements directly within
  the existing memory layout.

#### 4.4. Instantiation & Constructors

- `pub const fn zero() -> Self where T: Zero + Copy`: Instantiates an all-zero
  matrix using `T::ZERO` as the constant initialization value.
- `pub const fn identity() -> Self where T: Zero + One + Copy`: Instantiates an
  identity matrix (restricted to square shapes) by initializing elements to
  `T::ZERO` and filling the main diagonal with `T::ONE` via a const-evaluated
  loop.

- `pub const fn diagonal(val: [T; D::DIM]) -> Matrix<T, D, D>`:
  Constructs a diagonal matrix using the provided array of diagonal values and
  filling off-diagonal elements with `T::ZERO`.

- `pub fn from_fn<F>(mut f: F) -> Self where F: FnMut(usize, usize) -> T`:
  Generates a matrix using a coordinate-based mapping function at runtime.

_Implementation Note_: To support generic `const fn` initialization on stable
Rust, the scalar type `T` must implement the `Zero` and `One` traits from
`crate::math::num_traits`. These traits expose the associated constants
`T::ZERO` and `T::ONE`. All static constructors are marked `const fn` to allow
placing static matrices directly in read-only flash memory.

#### 4.5. Operator Overloading

Overloads `Add`, `Sub`, and `Mul` from `core::ops`. Dimension rules are
statically enforced at compile-time. Under the hood, these high-level operator
implementations follow standard numerical linear algebra conventions by mapping
directly to specific low-level BLAS subprograms (Anderson et al., 1999; Golub &
Van Loan, 2013; Demmel, 1997).
This mapping represents the industry-conventional approach to structuring basic
linear
algebra operations, rather than a custom design innovation:

- **Matrix Addition (`Add`) & Subtraction (`Sub`)**: Evaluated element-wise.
  Following standard convention, these operations map to the BLAS Level 1 *
  *`AXPY`** subprogram (
  `y = a*x + y` trait defined
  in [subprograms.rs](../../src/math/subprograms.rs)), where addition uses
  `a = T::ONE` and subtraction uses `a = -T::ONE`.
- **Matrix Negation (`Neg`)**: Maps to the standard BLAS Level 1 **`SCAL`**
  subprogram with
  `a = -T::ONE`.
- **Matrix-Matrix Multiplication (`Mul<Matrix>`)**: Statically enforces
  dimension matching (e.g. $(M \times N) \times (N \times P) \to (M \times P)$).
  It maps to the conventional BLAS Level 3 **`GEMM`** subprogram (
  `C = alpha*A*B + beta*C`
  trait in [subprograms.rs](../../src/math/subprograms.rs)) with
  `alpha = T::ONE` and `beta = T::ZERO`.
- **Matrix-Vector Multiplication (`Mul<Vector>`)**: Maps to the conventional
  BLAS Level 2
  **`GEMV`** subprogram (`y = alpha*A*x + beta*y` trait in
  [subprograms.rs](../../src/math/subprograms.rs)) with `alpha = T::ONE` and
  `beta = T::ZERO`.

Every mapping above passes `S::ORDER` (`storage-trait-design.md` FR-4)
through to the subprogram call as an explicit layout parameter; `Matrix`'s
operator implementations do not special-case column-major vs. row-major
storage. This closes a gap in a prior revision of this document, which
described these mappings as though `GEMV`/`GEMM` already matched
`ArrayStorage`'s column-major default — `subprograms.rs`'s current
implementations are row-major (`chunks_exact(cols)`), a mismatch
`storage-trait-design.md` NFR-6 now resolves by making layout an explicit
argument rather than a fixed assumption on either side.

```rust
impl<T, M: Dim, N: Dim, P: Dim> Mul<Matrix<T, N, P>> for Matrix<T, M, N>
where
    T: Copy + Zero + Add<Output=T> + Mul<Output=T>,
{
    type Output = Matrix<T, M, P>;
    // ...
}
```

#### 4.6. Core Operations

- **Transposition**:
    - `pub fn transpose_view(&self) -> MatrixSlice<'_, T, C, R>`: Creates a
      zero-copy, non-destructive transposed view over the original matrix
      without allocations or memory copies.
    - `pub fn transpose_into(&self, dest: &mut Matrix<T, C, R>)`: Writes the
      transposed matrix into a caller-provided destination buffer, avoiding
      stack returns.
    - `pub fn transpose_mut(&mut self)`: Performs an in-place transposition for
      square matrices ($R = C$).
    - `pub fn transpose(&self) -> Matrix<T, C, R>`: Consumes/borrows the matrix
      and returns a new transposed matrix on the stack (convenience API for
      small matrix shapes).
- **Matrix Inversion & System Solving**:
    - _Explicit Decomposition Design_: Convenient signatures that mask
      heavy $O(N^3)$ operations behind stack-allocating value returns (such as
      `invert(&self) -> Result<Matrix<T, D, D>, LinAlgError>`) are explicitly
      rejected to prevent unexpected stack bloat in embedded runtimes.
    -
  `pub fn invert_mut(&mut self, pivots: &mut [usize]) -> Result<(), LinAlgError>`:
  Inverts a square matrix purely in-place using caller-provided pivot scratch
  space.
    -
  `pub fn invert_into(&self, dest: &mut Matrix<T, D, D, S2>, pivots: &mut [usize]) -> Result<(), LinAlgError>`:
  Computes the matrix inverse into a caller-provided destination matrix
  buffer.
    - **Symmetric Matrices**: Factorized via **$LDL^T$ Decomposition
      ** ($A = L D L^T$).
    - **General Square Matrices**: Factorized via **LU Decomposition with
      Partial Pivoting** ($P A = L U$).
- **Determinant Calculation**:
    - `pub fn determinant(&self) -> T`: Computes $\det(A)$ in $O(N)$ time
      directly from the diagonal factors of an already-computed
      `LuDecomposition` or `LdltDecomposition` object.

#### 4.7. Matrix Decomposition Objects

Similar to structural specializations, matrix factorizations are exposed as
dedicated **Decomposition Objects**. These types encapsulate matrix factors
alongside statically bounded auxiliary state (e.g. permutation indices for
pivoting) without heap allocation (`no_alloc`).

```rust
/// LU Factorization with partial pivoting (PA = LU)
pub struct LuDecomposition<T, D: Dim, S = ArrayStorage<T, D, D>, P = PivotStorage<D>> {
    data: Matrix<T, D, D, S>,
    pivots: P, // Statically bounded pivot array (decoupled via storage trait)
    row_exchanges: usize,
}

/// LDL^T Factorization for symmetric indefinite/positive-definite matrices (A = L D L^T)
pub struct LdltDecomposition<T, D: Dim, S = ArrayStorage<T, D, D>> {
    data: Matrix<T, D, D, S>,
}

/// Cholesky Factorization for symmetric positive-definite matrices (A = L L^T)
pub struct CholeskyDecomposition<T, D: Dim, S = ArrayStorage<T, D, D>> {
    l: LowerTriangular<T, D, S>,
}

/// QR Factorization (A = Q R)
pub struct QrDecomposition<T, R: Dim, C: Dim, SQ = ArrayStorage<T, R, R>, SR = ArrayStorage<T, R, C>> {
    q: Matrix<T, R, R, SQ>,
    r: UpperTriangular<T, R, SR>, // Restricted to top R x C triangular factor
}
```

##### 4.7.1. In-Place Factorization & Scratch Space API

To prevent stack bloat and avoid dynamic allocations, decomposition algorithms
provide in-place mutation methods where caller-provided scratch buffers or
mutable references act as the working state:

```rust
impl<T, D: Dim, S> Matrix<T, D, D, S>
where
    S: Storage<T, D, D>,
{
    /// Performs LU decomposition purely in-place on the matrix data.
    /// Overwrites self with L and U factors and populates the caller-provided pivot scratch array.
    pub fn lu_decompose_mut(&mut self, pivots: &mut [usize]) -> Result<usize, LinAlgError> {
        assert_eq!(pivots.len(), D::DIM, "Pivot scratch buffer size mismatch");
        let mut row_exchanges = 0;
        // Low-level LAPACK/BLAS GETRF kernel execution on slice pointers...
        Ok(row_exchanges)
    }

    /// Consumes the matrix to construct a stack-allocated LuDecomposition wrapper.
    pub fn into_lu(mut self) -> Result<LuDecomposition<T, D>, LinAlgError> {
        let mut pivots = PivotStorage::<D>::default();
        let row_exchanges = self.lu_decompose_mut(pivots.as_mut_slice())?;
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
impl<T, D: Dim, S, P> LuDecomposition<T, D, S, P> {
    /// Solves A * x = b in-place by mutating the right-hand side vector b into the solution x.
    pub fn solve_mut<const COLS: usize>(&self, b: &mut Matrix<T, D, Const<COLS>>) -> Result<(), LinAlgError> {
        // 1. Permute rows of b according to self.pivots
        // 2. Solve L * y = P * b using forward substitution
        // 3. Solve U * x = y using backward substitution
        Ok(())
    }
}
```

#### 4.8. Interoperability & Conversions

##### 4.8.1. Conversion to Polynomial

A square matrix `Matrix<T, D, D>` converts to its characteristic polynomial
`Polynomial<T, <D as DimAdd<U1>>::Output>`.

- **Type Signature**:
  ```rust
  impl<T, D: Dim> TryFrom<Matrix<T, D, D>> for Polynomial<T, <D as DimAdd<U1>>::Output>
  where
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

Converts a 2D matrix to a rank-2 `Tensor<T, Layout>`.

- **Type Signature**:
  ```rust
  impl<T, R: Dim, C: Dim, Layout: TensorLayout> TryFrom<Matrix<T, R, C>> for Tensor<T, Layout>
  where
      Layout: TensorLayout<Size = <R as DimMul<C>>::Output>,
  {
      type Error = ConversionError;
      // ...
  }
  ```
- **Behavior**: Maps nested column-major arrays into the flat array
  representation of the `Tensor`.
- **Failure Condition**: Returns `ConversionError::LayoutMismatch` if
  `Layout::RANK != 2` or if the layout's dimensions do not match $ R \times C $.

#### 4.9. Error Handling & State Management

##### 4.9.1. Compile-Time Constraints

Dimension mismatches (e.g., adding matrices of different sizes or multiplying
incompatible dimensions) fail at compile-time. Rust's type checker prevents
compiling invalid math.

##### 4.9.2. Runtime Error Taxonomy

To supplement the crate's generic `ArithmeticError`, `control-rs` defines
dedicated error enums in the `math` module to represent linear algebra and
conversion failures:

```rust
/// Unified linear algebra errors supplementing ArithmeticError.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinAlgError {
    /// The matrix is singular (or near-singular under the given numerical tolerance).
    SingularMatrix,
    /// The matrix operation requires a square shape but a non-square shape was provided.
    NonSquareMatrix,
}

/// Representation and layout conversion errors supplementing ArithmeticError.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConversionError {
    /// Rank or coordinate dimensions do not align between Matrix/Tensor.
    LayoutMismatch,
    /// The polynomial is not monic (leading coefficient is not ONE), preventing companion matrix construction.
    NonMonicPolynomial,
    /// Dimension or capacity overflow/underflow during calculations.
    DimensionMismatch,
}
```

##### 4.9.3. Runtime Fallbacks

Dynamic operations that cannot be validated statically use soft failure paths:

- Matrix inversion returns a `Result<Self, LinAlgError>` instead of panicking,
  allowing control loops to handle singular conditions (e.g., falling back to a
  degraded state by returning `Err(LinAlgError::SingularMatrix)`).
- Boundary access returns `Option<&T>` via safe `get` methods.

#### 4.10. Structural Specializations & Extensions

Specialized matrices are implemented as new-type wrappers around `Matrix` to
enforce mathematical invariants and dispatch optimized routines:

```rust
pub struct UpperTriangular<T, D: Dim>(Matrix<T, D, D>);
pub struct LowerTriangular<T, D: Dim>(Matrix<T, D, D>);
pub struct Symmetric<T, D: Dim>(Matrix<T, D, D>);
```

Rather than packing triangular data (which requires complex
non-linear index mapping and prevents flat slicing), we wrap a full square
matrix. This trades memory space for cache friendliness and compatibility with
slice-based BLAS kernels.

##### 4.10.1. Forward and Backward Substitution Examples

```rust
/// Solves L * x = b for a lower triangular matrix where L is L::DIM x L::DIM and x, b are L::DIM x 1.
pub fn solve_lower_triangular<T, D: Dim>(
    l: &LowerTriangular<T, D>,
    b: &Matrix<T, D, U1>,
    tolerance: T,
) -> Result<Matrix<T, D, U1>, LinAlgError>
where
    T: Float + Copy,
{
    let n = D::DIM;
    let mut x = Matrix::<T, D, U1>::zero();
    let l_mat = &l.0;

    for i in 0..n {
        let l_ii = l_mat.data[i][i];
        // Tolerance-based singularity check using the type's Signed abs()
        if l_ii.abs() < tolerance {
            return Err(LinAlgError::SingularMatrix);
        }
        let mut sum = T::ZERO;
        for j in 0..i {
            sum = sum + l_mat.data[j][i] * x.data[0][j]; // column-major: data[col][row]
        }
        x.data[0][i] = (b.data[0][i] - sum) / l_ii;
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
use control_rs::math::matrix::{Matrix, Dim, U1};

pub fn kalman_covariance_update<T, S: Dim, O: Dim>(
    p_pred: &Matrix<T, S, S>,
    k: &Matrix<T, S, O>,
    h: &Matrix<T, O, S>,
) -> Matrix<T, S, S>
where
    T: Ring + Copy,
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
  system with the decoupled `Storage<T, R, C>` trait enables complex
  compile-time matrix arithmetic bounds while keeping storage backends
  completely pluggable.

#### 5.3. External Libraries (`nalgebra`)

Using external crates like `nalgebra` in `no_std` mode was considered and
bypassed for two primary reasons:

1. **Generic `const fn` Support on Stable Rust**: Baking static matrices into
   read-only Flash memory requires `const fn` constructors. `nalgebra` relies on
   traits like `Default` which cannot be evaluated inside `const fn` on stable
   Rust. The custom `Zero` and `One` traits in `crate::math::num_traits` expose
   associated constants (`T::ZERO`/`T::ONE`), enabling compile-time matrix
   initialization into ROM.
2. **Safety-Critical Audit Footprint**: Certification standards (ISO 26262,
   DO-178C) require complete auditing of dependency source code. Bypassing
   external libraries keeps the audit footprint minimal and strictly focused on
   safety invariants.

While `nalgebra` was bypassed as a direct dependency for the reasons above, the
matrix architecture implemented in `control-rs` is a direct adaptation of
`nalgebra`'s design. Key design structures, trait hierarchies, dimensions, and
slicing properties are structurally modeled on Sébastien Crozet's original
architecture.

#### 5.4. Decoupled Storage Architecture

The physical memory layout is abstracted from mathematical dimensions via the
`Storage<T, R, C>` trait hierarchy (`Storage`, `StorageMut`).

- **Base `Storage<T, R, C>`**: Encapsulates raw pointer access (`ptr`,
  `ptr_mut`), offset calculation (`offset`), and unchecked indexing (
  `get_unchecked`).
- **Marker `ContiguousStorage<T, R, C>`**: Restricts slice coercion (`as_slice`)
  strictly to contiguous memory backends (`ArrayStorage`, `MatrixView`),
  protecting strided or padded memory layouts from slice-based data corruption.
- **Custom Backends**: Enables user-defined backends including read-only Flash
  wrappers, DMA memory pools, and borrowed `MatrixView`/`MatrixViewMut`
  wrappers.

#### 5.5. Memory Layout Alternatives

- **Row-Major Layout**: Row-major layouts provide spatial locality when
  accessing rows.
- **Panel-Major Layout (BLASFEO style)**: BLASFEO stores matrices in
  fixed-height panels with column-major layouts inside. This avoids data-packing
  overhead for cache-resident matrices but requires non-contiguous mapping and
  index arithmetic. It prevents exposing zero-copy flat slice APIs (`&[T]`)
  without allocation, which conflicts with safe API requirements.

#### 5.6. Factorization & Inversion Algorithms

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

#### 5.7. Matrix Multiplication Algorithms

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

#### 5.8. Determinant Calculation Algorithms

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
      at $32 \times 32$ elements ($R::DIM \times C::DIM \le 1024$),
      ensuring that a single float matrix instance never
      exceeds 4KB of stack
      space ($1024 \times 4\text{ bytes} = 4096\text{ bytes}$).

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
   use control_rs::math::matrix::{Matrix, Dim, U2, U1};

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
    guarantees underpinning the `Storage<T, R, C>` trait split and `#[repr(C)]`
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

---

### 10. Revision History

| Revision | Date           | Author          | Description                                                                                                          |
|:---------|:---------------|:----------------|:---------------------------------------------------------------------------------------------------------------------|
| 1.0      | July 12, 2026  | @MitchellDScott | Initial draft outlining core concepts, layout, and operations.                                                       |
| 1.1      | July 19, 2026  | @MitchellDScott | Restructured to new template; added embedded performance and verification details.                                   |
| 1.2      | July 25, 2026  | @MitchellDScott | Added supporting bibliography and inline citations.                                                                  |
| 1.3      | July 26, 2026  | @MitchellDScott | Added Decomposition Objects, zero-copy MatrixView wrappers, and no_alloc scratch space patterns.                     |
| 1.4      | July 26, 2026  | @MitchellDScott | Harmonized with storage trait design doc; updated `Matrix` definition, bounds, decomposition rules, and V&V.         |
| 1.5      | July 26, 2026  | @MitchellDScott | Added comprehensive 3-tiered bibliography and inline citations across core architectural sections.                   |
| 1.6      | August 1, 2026 | @MitchellDScott | Corrected `nalgebra` comparison claims; clarified storage-decoupling benefit; added system-solving convenience note. |
| 1.7      | August 2, 2026 | @MitchellDScott | Propagated `num-traits-design.md` pivot; removed duplicate MatrixView definitions; relocated `ConversionError`.      |
