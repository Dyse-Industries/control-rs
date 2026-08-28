# Matrix Type & Structural Specializations (Design Document)

![Date Badge](https://img.shields.io/badge/Date-August_25,_2026-blue)
![Status Badge](https://img.shields.io/badge/Doc%20Status-Approved-green)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

---

### 1. Introduction

This module provides statically-typed, zero-allocation matrix representations
and linear algebra routines for real-time control, estimation, and signal
processing.

Primary usage scenarios:

- **Kalman Filtering & Covariance Propagation**: Evaluating time
  updates ($P_{k|k-1} = A P_{k-1} A^T + Q$) and measurement
  updates ($K = P H^T (H P H^T + R)^{-1}$) on fixed-size state estimates with
  deterministic execution time and zero heap allocation.
- **Coordinate Transformations & Kinematics**: Multiplying vectors by spatial
  rotation and transformation matrices ($v' = R v$) in robotics and flight
  control loops.
- **Direct Linear Solves & Least-Squares Estimation**: Solving systems of linear
  equations ($A x = b$) and computing factorizations (LU, Cholesky, $LDL^T$, QR)
  for model-predictive control and parameter identification.
- **Structured Covariance Storage**: Storing and operating on symmetric
  positive-definite covariance matrices using packed storage representations to
  cut memory consumption in resource-constrained embedded systems.

---

### 2. Requirements

#### 2.1. Functional Requirements

- **FR-1 — Compile-Time Shape Verification**: Matrix operations validate operand
  dimension compatibility at compile time before execution. A dimension
  mismatch (such as multiplying a $2 \times 2$ matrix by a $3 \times 1$ vector)
  must prevent compilation rather than panicking at runtime in embedded control
  loops.
- **FR-2 — Matrix Algebra & Linear Transformations**: Matrices provide standard
  matrix-vector and matrix-matrix
  products ($y = \alpha A x + \beta y$, $C = \alpha A B + \beta C$), addition,
  subtraction, negation, transposition, and scaling. Operations must
  mathematically match standard linear algebra without requiring intermediate
  heap buffers.
- **FR-3 — Fallible Factorizations & Direct Solvers**: Direct linear system
  solvers ($A x = b$), triangular solvers, and matrix factorizations (LU,
  Cholesky, $LDL^T$, QR) return explicit error variants (`LinAlgError`) when
  inputs are singular, rank-deficient, or non-positive-definite (Anderson et
  al., 1999). Numerical failures must never panic or silently corrupt downstream
  state estimates.
- **FR-4 — Coordinate Element Access**: Element indexing by 2D
  coordinates $(i, j)$ accesses the correct logical entry for both row-major and
  column-major representations. Out-of-bounds indices return `None` on checked
  access and fail at compile time for fixed-index lookups.
- **FR-5 — Structural Specializations**: Symmetric, Hermitian, triangular, and
  diagonal matrix structures provide element access and tailored algebraic
  operations that exploit symmetry and sparsity without allocating full dense
  storage.
- **FR-6 — Zero-Copy Submatrix Views**: Callers can extract rectangular
  sub-windows, row slices, and column vectors without copying underlying matrix
  data. Slices preserve stride semantics for zero-copy kernel execution.

#### 2.2. Non-Functional Requirements

- **NFR-1 — Zero-Allocation Deterministic Execution**: Matrix operations and
  factorizations over statically sized dimensions execute entirely on stack or
  borrowed memory without dynamic allocation.
- **NFR-2 — Interoperable C-ABI Layout**: Contiguous matrix layouts expose flat
  slice views conforming to standard C-ABI contiguous array layouts for hardware
  DMA and vendor DSP library interoperability.
- **NFR-3 — Predictable Real-Time Latency**: Factorization and solver execution
  loops address memory without runtime bounds checking or unwinding landing
  pads, guaranteeing bounded Worst-Case Execution Time (WCET).

#### 2.3. Constraints

- **C-1 — Stable Rust Toolchain**: Code must compile on `stable` Rust without
  requiring incomplete experimental features (`generic_const_exprs`).
- **C-2 — Stack Footprint Limit**: Matrix dimension capacities are statically
  bounded ($R, C \le 128$) to ensure stack allocations do not exceed embedded
  microcontroller memory limits.
- **C-3 — `#![no_std]` Environment**: Operates strictly in `#![no_std]` without
  standard library dependencies.
- **C-4 — In-Place Factorization Mutability**: In-place matrix decompositions
  require mutable dense storage and cannot be executed over immutable or
  packed-view backends.

---

### 3. Technical Overview

`Matrix<T, R, C, S>` provides statically sized matrix algebra and direct
factorizations parameterized over scalar type `T`, dimensions `R: Dim, C: Dim`,
and storage backend `S`. Built directly on `control-rs::math`'s decoupled
storage hierarchy (`storage-design.md`) and BLAS/LAPACK subprograms (
`subprograms-design.md`), it serves as the single unified matrix wrapper across
owning stack arrays (`ArrayStorage`, `RowArrayStorage`), borrowed zero-copy
views (`StorageView`, `StorageViewMut`), and packed structured storage (
`DiagonalStorage`, `SymmetricPackedStorage`, `TriangularPackedStorage`).

Packed and structured matrices are provided as type aliases over `Matrix`,
preserving zero-cost abstraction without wrapper-level duplication or runtime
layout-matching branching. Linear algebra operations delegate directly to
zero-overhead subprogram kernels, returning typed `LinAlgError` variants on
numerical singularity.

---

### 4. Architecture

The `Matrix` struct will be implemented in a new submodule: `src/matrix/mod.rs`.

#### 4.1. Core Matrix Types & Storage Hierarchy

```rust
pub struct Matrix<T, R: Dim, C: Dim, S> {
    storage: S,
    _marker: PhantomData<(T, R, C)>,
}

// Column-Major Dense Matrix Aliases
pub type ArrayMatrix<T, const R: usize, const C: usize> =
Matrix<T, Const<R>, Const<C>, ArrayStorage<T, R, C>>;
pub type Owned<T, const R: usize, const C: usize> = ArrayMatrix<T, R, C>;

pub type MatrixSlice<'a, T, R, C> = Matrix<T, R, C, StorageView<'a, T, R, C>>;
pub type MatrixSliceMut<'a, T, R, C> =
Matrix<T, R, C, StorageViewMut<'a, T, R, C>>;

// Row-Major Dense Matrix Aliases
pub type RowArrayMatrix<T, const R: usize, const C: usize> =
Matrix<T, Const<R>, Const<C>, RowArrayStorage<T, R, C>>;
pub type RowOwned<T, const R: usize, const C: usize> = RowArrayMatrix<T, R, C>;

pub type RowMatrixSlice<'a, T, R, C> =
Matrix<T, R, C, StorageView<'a, T, R, C>>;
pub type RowMatrixSliceMut<'a, T, R, C> =
Matrix<T, R, C, StorageViewMut<'a, T, R, C>>;

// Packed and Structured Matrix Aliases
pub type PackedMatrix<T, const N: usize, S> =
Matrix<T, Const<N>, Const<N>, S>;
pub type SymmetricPacked<T, const N: usize, const L: usize> =
Matrix<T, Const<N>, Const<N>, SymmetricPackedStorage<T, N, L>>;
pub type HermitianPacked<T, const N: usize, const L: usize> =
Matrix<T, Const<N>, Const<N>, HermitianPackedStorage<T, N, L>>;
pub type TriangularPacked<T, const N: usize, const L: usize> =
Matrix<T, Const<N>, Const<N>, TriangularPackedStorage<T, N, L>>;
pub type DiagonalMatrix<T, const N: usize> =
Matrix<T, Const<N>, Const<N>, DiagonalStorage<T, N>>;
```

`ArrayStorage<T, R, C>` and `RowArrayStorage<T, R, C>` take bare
`const usize` capacities and implement `DenseStorage<T>` at
`type R = Const<R>; type C = Const<C>;` (`storage-design.md` FR-1, C-4).
Neither projects `Dim::USIZE` into an array length, so neither requires
`generic_const_exprs`.

`StorageView`/`StorageViewMut` are `Dim`-generic rather than
const-generic (`storage-design.md` FR-2): they carry runtime `isize`
strides, so one view type covers column-major, row-major, transposed and
reversed windows without a separate alias per ordering.

The point of decoupling storage is to have one single `Matrix` struct
implementation that works across every dense, strided, and packed storage leaf.

##### 4.1.1. Storage Capabilities and Trait Bounds

Methods on `Matrix<T, R, C, S>` are enabled via trait bounds on `S`:

| Bound                  | Operations                                                                            |
|:-----------------------|:--------------------------------------------------------------------------------------|
| `DenseStorage<T>`      | Strided lookup (`get`, `get_unchecked`), Level 2/3 kernel calls, views (§4.5, §4.9.2) |
| `DenseStorageMut<T>`   | Mutable access (`get_mut`, `get_mut_unchecked`, `set`), in-place factorization (§4.7) |
| `ContiguousStorage<T>` | `as_slice()`, `const ORDER: MatrixLayout`, C-ABI and FFI hand-off (§4.3)              |
| `PackedStorage<T>`     | Packed lookup (`packed_index`, `value`, `value_unchecked`), `uplo()` (§4.9.2)         |
| `PackedStorageMut<T>`  | Physical-slot mutation (`set`, `set_unchecked`) with structural rejection (§4.9.2)    |

#### 4.2. Memory Layout & Storage Strategy

`Matrix`'s own arithmetic never branches on layout. Ordering is not a kernel
argument and not a runtime query: it is carried by the leaf's strides.
`DenseStorage` exposes `r_stride() -> isize` and `c_stride() -> isize`, and
the address of $(r, c)$ is $r \cdot RS + c \cdot CS$ (`storage-design.md`
C-1). Column-major leaves report $RS = 1, CS = R$; row-major leaves report
$RS = C, CS = 1$. Leaves that are additionally contiguous expose
`const ORDER: MatrixLayout` through `ContiguousStorage<T>` for C-ABI and FFI
hand-off only (§4.3).

- **Cache Locality**: Under column-major ordering (`ArrayStorage`), each
  column $ A_j $ is contiguous in memory. Under row-major ordering
  (`RowArrayStorage`), each row $ A_i $ is contiguous in memory, enabling
  row-strided vectorization.
- **BLAS Interoperability**: Column-major layout matches standard Fortran/LAPACK
  conventions (Anderson et al., 1999) and embedded DSP libraries (ARM
  CMSIS-DSP).
  Row-major layout allows direct zero-copy interfacing with C/C++ libraries and
  hardware sensor/actuator rasters without transposition.

*The default storage backend (`Owned`) aliases **column-major**
`ArrayStorage<T, R, C>` (`[[T; R]; C]`).*

#### 4.3. Memory Representation, Slicing & Views

To ensure stable memory layout and compatibility with C-based hardware
libraries, the owning array leaves store a padding-free nested array and
implement the contiguity markers of `storage-design.md` FR-2.

Slice access is a capability of `ContiguousStorage<T>`, not of every dense
leaf: a `StorageView` with a non-unit stride has no contiguous slice to
return. The accessors are therefore bounded on the marker:

```rust
impl<T, R: Dim, C: Dim, S> Matrix<T, R, C, S>
where
    S: ContiguousStorage<T, R=R, C=C>,
{
    /// Padding-free slice of matrix memory, in `S::ORDER`.
    pub fn as_slice(&self) -> &[T] {
        self.storage.as_slice()
    }

    pub const fn order(&self) -> MatrixLayout {
        S::ORDER
    }
}

impl<T, R: Dim, C: Dim, S> Matrix<T, R, C, S>
where
    S: ContiguousStorageMut<T, R=R, C=C>,
{
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.storage.as_mut_slice()
    }
}
```

Two access paths, one per consumer:

| Accessor               | Bound               | Consumer                                             |
|:-----------------------|:--------------------|:-----------------------------------------------------|
| `as_slice`             | `ContiguousStorage` | Inspection, C-ABI and hardware-backend hand-off      |
| storage operand (`&S`) | `DenseStorage`      | Every Level 1/2/3 kernel (`Axpy`, `Gemv`, `Gemm`, …) |

Each has a `_mut` counterpart (`as_mut_slice`, `&mut S`) on the
corresponding `ContiguousStorageMut`/`DenseStorageMut` bound.

Subprogram traits are parameterized over the storage types themselves
(`subprograms-design.md` FR-9): a call site passes `&self.storage`, and the
kernel reads shape from `S::R`/`S::C` and addresses through `as_ptr()` plus
the leaf's strides. Kernels do not take flattened or nested array operands,
so call sites carry no `Const<R>: DimMul<Const<C>, Output = …>` bound and no
flattened `&[T; R * C]` accessor blocked by `generic_const_exprs`
(`storage-design.md` §5).

##### 4.3.1. Zero-Copy Views (`StorageView`)

`Matrix` builds on the strided view backends of `storage-design.md` FR-2.
`StorageView<'a, T, R, C>` carries arbitrary `isize` strides: an arbitrary
submatrix window is a pointer offset plus the parent's own strides, with no
padded leading dimension to leave untouched and no restriction to column
panels.

- `ArrayMatrix::view()` / `view_mut()` borrow the whole buffer at the owning
  alias's shape, with $RS = 1, CS = R$.
- `Matrix::submatrix::<R2, C2>(origin)` offsets the base pointer by
  $r_0 \cdot RS + c_0 \cdot CS$ and keeps the parent strides. The window is
  fallible in the offset alone ($r_0 + R2 \le R$, $c_0 + C2 \le C$), and
  wrapping an erased-length slice returns
  `ConversionError::DimensionMismatch` (`storage-design.md` §4.6).
- **Zero-Copy Transposition**: `transpose_view()` swaps strides and
  dimensions (`storage-design.md` FR-2), so $A^T$ needs no buffer. Algebraic
  transposition inside a kernel remains the `Trans` flag; the two are
  complementary, not alternatives.
- **Reversed Views**: `reverse_view()` sets a negative row stride over a
  tail-offset pointer, giving BLAS's $INCX < 0$ semantics without a copy.

`MatrixSlice` / `MatrixSliceMut` wrap those leaves.

- **In-Place Transposition**: For square matrices ($R = C$), in-place element
  swapping (`pub fn transpose_mut(&mut self)`) mutates elements directly within
  the existing memory layout. Copying `transpose` / `transpose_into` write a
  new buffer. Neither is required to *read* $A^T$, which `transpose_view()`
  now provides at zero cost.

#### 4.4. Instantiation & Constructors

- `pub const fn zero() -> Self where T: Zero + Copy`: Instantiates an all-zero
  matrix using `T::ZERO` as the constant initialization value.
- `pub const fn identity() -> Self where T: Zero + One + Copy`: Instantiates an
  identity matrix (restricted to square shapes) by initializing elements to
  `T::ZERO` and filling the main diagonal with `T::ONE` via a const-evaluated
  loop.

- `pub const fn diagonal<const D: usize>(val: [T; D]) -> Owned<T, D, D>`:
  Constructs a dense $D \times D$ matrix from the provided diagonal values,
  filling off-diagonal elements with `T::ZERO`. The $O(D^2)$-space dense
  form is what level-2/3 kernels can consume directly.
- `pub const fn packed_diagonal<const D: usize>(val: [T; D])
  -> DiagonalMatrix<T, D>`: Constructs the $O(D)$-space `DiagonalStorage`
  leaf as a `DiagonalMatrix` alias (§4.1.1). Off-diagonal coordinates are
  unstored
  and evaluate algebraically to `T::ZERO` (§4.9.2). This backend reaches no
  dense Level 2/3 kernel, because it implements `PackedStorage`, not
  `DenseStorage`; packed operands instead reach the packed kernels
  (`Spmv`, `Hpmv`, `Tpmv`, `Tpsv`; `subprograms-design.md` FR-3), or are
  converted to a dense leaf through `ToDenseStorage`
  (`storage-design.md` FR-7) before matrix-matrix arithmetic.

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
| `Add`         | `Axpy` (`y = αx + y`)   | 1     | `α = T::ONE`                |
| `Sub`         | `Axpy`                  | 1     | `α = -T::ONE`               |
| `Neg`         | `Scal`                  | 1     | `α = -T::ONE`               |
| `Mul<Matrix>` | `Gemm` (`C = αAB + βC`) | 3     | `α = T::ONE`, `β = T::ZERO` |
| `Mul<Vector>` | `Gemv` (`y = αAx + βy`) | 2     | `α = T::ONE`, `β = T::ZERO` |

`Sub` needs no extra bound: `Sub<Output = Self>` is already a `Scalar`
supertrait (`num-traits-design.md` §4.1). `Neg` and the $\alpha = -1$
bindings need a negatable scalar, so they bound `T: Scalar + Signed`, which
excludes unsigned integers. `Complex<T>` is `AdditiveGroup` but deliberately
not `Signed` (`num-traits-design.md` §4.3), so complex negation is written
`T::ZERO - x` and stays at `T: Scalar`.

`Mul<Matrix>` statically enforces $(M \times N) \times (N \times P) \to (M
\times P)$. The owning output leaf forces the operator impls onto the
const-generic aliases: naming `ArrayStorage<T, {M::USIZE}, {P::USIZE}>` in a
`Dim`-generic impl would be a parameter-dependent const expression, which
C-1 and `storage-design.md` NFR-2 forbid.

```rust
// Dim-generic form: caller supplies the destination, no owning output named.
impl<T, M: Dim, N: Dim, P: Dim, SA, SB> Matrix<T, M, N, SA>
where
    T: Scalar,
    SA: DenseStorage<T, R=M, C=N>,
    SB: DenseStorage<T, R=N, C=P>,
{
    pub fn mul_into<SC>(&self, rhs: &Matrix<T, N, P, SB>, out: &mut Matrix<T, M, P, SC>)
    where
        SC: DenseStorageMut<T, R=M, C=P>
    {}
}

// Operator sugar, defined where the output length is a plain const usize.
impl<T: Scalar, const M: usize, const N: usize, const P: usize>
Mul<ArrayMatrix<T, N, P>> for ArrayMatrix<T, M, N>
{
    type Output = ArrayMatrix<T, M, P>;
    // delegates to mul_into over a fresh ArrayStorage<T, M, P>
}
```

`T: Scalar` is the ring bound: the operators are defined for integers,
fixed-point `Quantized`, floats and `Complex<T>` alike, and none of them
requires `Div` (`num-traits-design.md` FR-2).

##### 4.5.1. Required Subprogram Inventory

`Matrix` calls the Level-1/2/3 and LAPACK kernels in
`subprograms-design.md`; it does not reimplement them.

##### 4.5.2. Operand Derivation at the Call Site

Kernels take typed storage operands, so layout parameters are properties
the kernel reads off the operand's own type rather than const generics at
the call site (FR-5; `subprograms-design.md` FR-9). `Matrix` supplies:

| Kernel input            | Source                                                                |
|:------------------------|:----------------------------------------------------------------------|
| Operand `&A` / `&mut Y` | `&self.storage`, typed `S: DenseStorage<T>`                           |
| Shape                   | `S::R::USIZE`, `S::C::USIZE` — monomorphization constants             |
| Addressing              | `as_ptr()` with `r_stride()` / `c_stride()` (`storage-design.md` C-1) |
| `trans` / `ta`, `tb`    | The algebraic operation, never the layout (§4.6)                      |
| `uplo`, `diag`, `side`  | The structural intent of the call (`UpLo`, `Diag`, `Side`)            |
| $\alpha$, $\beta$       | Scalars of type `T`, or `T::Real` on the real-scaled routines         |

Row traversal of column-major storage needs no gather into scratch and no
separate increment parameter: a row is a `StorageView` over the same buffer
with the strides swapped, which the Level 1 kernels consume directly
(`storage-design.md` FR-2).

*Backend Selection*: `Matrix<T, R, C, S>` dispatches through associated
functions on a backend marker type — `DefaultBlas` for the pure-Rust
reference path, `CmsisDspBlas` / `NmsisDspBlas` under their target features
(`subprograms-design.md` §4.8). The backend is fixed by the target triple at
compile time, so it is not a 5th generic parameter on the `Matrix` struct.

#### 4.6. Core Operations

- **Transposition**:
    - `pub fn transpose_into(&self, dest: &mut Matrix<T, C, R>)`: Writes the
      transposed matrix into a caller-provided destination buffer, avoiding
      stack returns.
    - `pub fn transpose_mut(&mut self)`: Performs an in-place transposition for
      square matrices ($R = C$).
    - `pub fn transpose(&self) -> Matrix<T, C, R>`: Returns a new transposed
      matrix on the stack (convenience API for small shapes).
    - Algebraic $A^T x$ / $A^T B$ without a new buffer: pass
      `Trans::Trans` (or `Trans::ConjTrans` for the adjoint $A^H$) into
      `Gemv`/`Gemm` (`subprograms-design.md` C-2). Alternatively
      `transpose_view()` produces a stride-swapped `StorageView` with no
      copy (§4.3.1); the adjoint is only available as `Trans::ConjTrans`,
      since conjugation has no representation as a stride.
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

Every factorization mutates its factors in place, so each decomposition
object holds an owning strided leaf implementing `DenseStorageMut<T>` (C-3).
A packed input is converted to a dense working copy through `ToDenseStorage`
(`storage-design.md` FR-7) before factorization, except where a packed
LAPACK routine exists: `Pptrf`/`Pptrs` factor and solve directly in the
packed triangle (`subprograms-design.md` FR-6, FR-7).

The decomposition objects wrap the LAPACK subprogram traits rather than
reimplementing them. `into_lu` calls `Getrf`, `solve_mut` calls `Getrs`,
the Cholesky path calls `Potrf`/`Potrs`, and QR calls
`Geqrf` followed by `Ormqr` (real) or `Unmqr` (complex). Each returns
`LinAlgResult<()>`, whose arms are `NotPositiveDefinite`, `SingularMatrix`,
`WorkspaceTooSmall` and `MaxIterationsReached` (`error-design.md` §3).

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
    pub fn lu_decompose_mut(&mut self, pivots: &mut [usize; D]) -> LinAlgResult<usize> {
        // `Getrf` writes the factors in place and records row swaps in `ipiv`
        // (`subprograms-design.md` FR-6).
        Backend::getrf(&mut self.storage, pivots)?;
        Ok(count_row_exchanges(pivots))
    }

    /// Consumes the matrix to construct a stack-allocated LuDecomposition wrapper.
    pub fn into_lu(mut self) -> LinAlgResult<LuDecomposition<T, D>> {
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
    pub fn solve_mut<const COLS: usize>(&self, b: &mut Owned<T, D, COLS>) -> LinAlgResult<()>
    where
        Const<COLS>: Dim,
    {
        // `Getrs` applies P, then the forward and backward triangular solves
        // (`subprograms-design.md` FR-7).
        Backend::getrs(
            Trans::NoTrans,
            &self.data.storage,
            &self.pivots,
            &mut b.storage,
        )
    }
}
```

#### 4.8. Interoperability & Conversions

##### 4.8.1. Conversion to Polynomial

A square matrix `Matrix<T, D, D, S>` converts to its characteristic polynomial
`Polynomial<T, <D as DimAdd<Const<1>>>::Output>`.

- **Type Signature**:
  ```rust
  impl<T, D: Dim, S> TryFrom<Matrix<T, D, D, S>> for Polynomial<T, <D as DimAdd<Const<1>>>::Output>
  where
      S: DenseStorage<T, R = D, C = D>,
      D: DimAdd<Const<1>>,
      <D as DimAdd<Const<1>>>::Output: Dim,
      T: Scalar + Div<Output = T>,
  {
      type Error = ConversionError;
      // ...
  }
  ```
- **Behavior**: Coefficients are computed using the Faddeev-LeVerrier
  algorithm (Faddeev & Faddeeva, 1963). The recurrence divides by the step
  index, hence the `Div` bound; `T: Scalar` alone excludes division
  (`num-traits-design.md` §4.1, Alternative 3), and integer scalars route
  through `TryDiv` instead of this conversion.
- **Failure Condition**: Returns `ConversionError::DimensionMismatch` when
  the coefficient capacity erased from the destination type cannot hold
  $D + 1$ terms. `ConversionError` is defined once in `src/math/mod.rs`
  (`error-design.md` FR-1).

##### 4.8.2. Conversion to Tensor

Converts a 2D matrix to a rank-2 `Tensor<T, Layout, B>`.

- **Type Signature**:
  ```rust
  impl<T, R: Dim, C: Dim, S, Layout: TensorLayout> From<Matrix<T, R, C, S>> for Tensor<T, Layout, S>
  where
      S: ContiguousStorage<T, R = R, C = C>,
      Layout: TensorLayout<Size = <R as DimMul<C>>::Output>,
  {
      // Preserves backing buffer zero-copy when compile-time size and rank 2 match
  }
  ```
- **Behavior**: Maps the leaf's padding-free slice directly into the flat
  buffer representation of the `Tensor`. The `ContiguousStorage` bound is
  what makes the mapping zero-copy: a strided `StorageView` has no such
  slice and converts by element copy instead.
- **Infallible Compile-Time Bound**: Dimensions and rank are verified statically
  at compile time via `Layout: TensorLayout<Size = <R as DimMul<C>>::Output>`.
  This conversion cannot produce `ConversionError::LayoutMismatch`
  (`error-design.md` §3).

#### 4.9. Error Handling & Element Lookup

##### 4.9.1. Compile-Time Constraints

Dimension mismatches (e.g., adding matrices of different sizes or multiplying
incompatible dimensions) fail at compile-time. Rust's type checker prevents
compiling invalid math.

##### 4.9.2. Element Lookup Across Both Storage Subsystems

`storage-design.md` FR-1, FR-2, FR-3, and FR-7 define logical coordinate
resolution on `DenseStorage` and `PackedStorage`. Each wrapper exposes those
accessors by delegating directly to `self.storage` (FR-4), eliminating
wrapper-level layout matching and runtime index arithmetic:

```rust
impl<T, R: Dim, C: Dim, S> Matrix<T, R, C, S>
where
    S: DenseStorage<T, R=R, C=C>,
{
    /// Strided lookup. Delegates to `self.storage.get(row, col)`.
    #[inline(always)]
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        self.storage.get(row, col)
    }

    /// # Safety
    /// `row < R::USIZE` and `col < C::USIZE` must hold.
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, row: usize, col: usize) -> &T {
        unsafe { self.storage.get_unchecked(row, col) }
    }
}

impl<T, R: Dim, C: Dim, S> Matrix<T, R, C, S>
where
    S: DenseStorageMut<T, R=R, C=C>,
{
    #[inline(always)]
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        self.storage.get_mut(row, col)
    }

    /// Checked write. `Err(StorageError::OutOfBounds)` on an invalid coordinate.
    #[inline(always)]
    pub fn set(&mut self, row: usize, col: usize, val: T) -> StorageResult<()> {
        self.storage.set(row, col, val)
    }

    /// # Safety
    /// `row < R::USIZE` and `col < C::USIZE` must hold.
    #[inline(always)]
    pub unsafe fn get_mut_unchecked(&mut self, row: usize, col: usize) -> &mut T {
        unsafe { self.storage.get_mut_unchecked(row, col) }
    }
}

impl<T, N: Dim, S> PackedMatrix<T, N, S>
where
    S: PackedStorage<T, N=N>,
    T: Copy,
{
    /// Algebraic entry evaluation. Applies the leaf's structural invariant:
    /// symmetric reflection, Hermitian conjugate reflection, unit diagonal,
    /// or a structural `T::ZERO` off-triangle.
    #[inline(always)]
    pub fn value(&self, row: usize, col: usize) -> Option<T> {
        self.storage.value(row, col)
    }

    /// Physical slot lookup. `None` for a coordinate in the implicit half.
    #[inline(always)]
    pub fn packed_index(&self, row: usize, col: usize) -> Option<usize> {
        self.storage.packed_index(row, col)
    }

    /// # Safety
    /// `row < N::USIZE` and `col < N::USIZE` must hold.
    #[inline(always)]
    pub unsafe fn value_unchecked(&self, row: usize, col: usize) -> T {
        unsafe { self.storage.value_unchecked(row, col) }
    }
}
```

The signatures differ deliberately. `get` returns `Option<&T>` because every
in-bounds coordinate of a strided backend names a stored element. `value`
returns `Option<T>` by value because a structurally-implied entry (a
reflected element, a unit diagonal, an off-triangle zero) is computed, not
addressed, so no reference to it exists (`storage-design.md` §4.3). `None`
means out of bounds in both, never "structurally zero".

Packed mutation is separate again: `PackedStorageMut::set` writes physical
slots only and rejects structural violations with
`StorageError::ImmutableUnitDiagonal` or
`StorageError::InvalidHermitianDiagonal` (`storage-design.md` FR-3, §4.4).

*Codegen Advantage*: Delegating coordinate lookup to the concrete leaves lets
each monomorphized implementation compile without dead layout branches —
`ArrayStorage` folds $r \cdot 1 + c \cdot R$ directly, while
`SymmetricPackedStorage` handles triangular index reflection natively. Hot
arithmetic paths bypass lookup entirely and run inside the subprogram
kernels, whose unchecked accessors carry no bounds checks (NFR-3).

##### 4.9.3. Runtime Fallbacks

Dynamic operations that cannot be validated statically use soft failure paths:

- Matrix inversion returns `LinAlgResult<()>` instead of panicking, allowing
  control loops to handle singular conditions (e.g., falling back to a
  degraded state on `Err(LinAlgError::SingularMatrix)`).
- Boundary access returns `Option<&T>` (`get`, strided backends) or
  `Option<T>` (`value`, packed backends); `None` denotes an out-of-bounds
  coordinate only (§4.9.2).
- Checked writes return `StorageResult<()>`; the structural arms are listed
  in `error-design.md` §3.

#### 4.10. Structural Specializations & Extensions

Structural specializations pair a storage leaf with a high-level newtype
wrapper. `storage-design.md` FR-3 provides four packed leaves
(`SymmetricPackedStorage`, `HermitianPackedStorage`,
`TriangularPackedStorage`, `DiagonalStorage`); the strided branch provides
the full-square leaves. The newtypes below are distinct items from the
storage leaves and live in `crate::matrix` rather than
`crate::math::storage`.

```rust
// Full-square strided form: consumes the dense Level 2/3 kernels directly.
pub struct UpperTriangular<T, const D: usize, S = ArrayStorage<T, D, D>>(
    pub Matrix<T, Const<D>, Const<D>, S>,
);
pub struct LowerTriangular<T, const D: usize, S = ArrayStorage<T, D, D>>(
    pub Matrix<T, Const<D>, Const<D>, S>,
);
pub struct Symmetric<T, const D: usize, S = ArrayStorage<T, D, D>>(
    pub Matrix<T, Const<D>, Const<D>, S>,
);
```

This dual story provides complete consistency: storage leaves define physical
memory layout and bounds, while high-level newtype wrappers enforce mathematical
invariants, optimize solver algorithms ($LDL^T$, forward/backward substitution),
and dispatch specialized subprogram kernels.

Both forms are first-class: a full-square wrapper trades $N^2$ storage for
the dense kernels (`Trmv`, `Trsv`, `Symv`, `Hemv`); the packed aliases
(§4.1.1) trade a non-linear index map for $N(N+1)/2$ storage and reach the
packed kernels (`Tpmv`, `Tpsv`, `Spmv`, `Hpmv`; `subprograms-design.md`
FR-3). Hardware acceleration is not the deciding factor between them, since
`subprograms-design.md` §4.8 delegates the packed routines to `DefaultBlas`
on every backend. Choose packed when the $\approx 2\times$ space saving
matters and dense when the operand feeds a Level 3 routine.

##### 4.10.1. Forward and Backward Substitution

Substitution delegates to `Trsv` (§4.5.1). The wrapper's job is the
singularity screen and the operand derivation; it does not re-implement the
inner loop, and in particular does not address elements through `get` in the
hot path (§4.9.2, NFR-3):

```rust
/// Solves L * x = b in place for a lower triangular D x D factor.
pub fn solve_lower_triangular_mut<T, const D: usize>(
    l: &LowerTriangular<T, D>,
    b: &mut ArrayMatrix<T, D, 1>,
    tolerance: T::Real,
) -> LinAlgResult<()>
where
    Const<D>: Dim,
    T: Scalar + Div<Output=T>,
    T::Real: Radical + PartialOrd,
{
    // Diagonal screen. `abs2()` is re² + im², so the comparison needs no
    // square root and stays valid for complex factors
    // (`num-traits-design.md` FR-4).
    let a = &l.0.storage;
    for i in 0..D {
        if unsafe { a.get_unchecked(i, i) }.abs2() < tolerance {
            return Err(LinAlgError::SingularMatrix);
        }
    }
    // op(L) x = b over the lower triangle, non-unit diagonal. `Backend` is
    // the target-selected implementor (§4.5.2).
    Backend::trsv(
        UpLo::Lower,
        Trans::NoTrans,
        Diag::NonUnit,
        a,
        &mut b.storage,
    );
    Ok(())
}
```

`Trsv` overwrites the right-hand side operand with the solution. A caller
needing the original `b` clones it first; the design does not offer a
non-destructive triangular solve, for the stack-allocation reason §5.1
gives. `Trsv` is a field kernel, hence the `Div` bound: integer and
`Quantized` scalars have no total division and do not reach this path
(`num-traits-design.md` Alternative 3).

##### 4.10.2. Companion Matrix Root-Finding

For polynomial root-finding, the coefficients are mapped to a companion matrix
in upper Hessenberg form (strict zeros beneath the first lower subdiagonal).
Instead of using a general $O(N^3)$ QR algorithm, the solver exploits the
unitary-plus-rank-one structure (Aurentz et al., 2014). This reduces storage
requirements to $O(N)$ and computational complexity to $O(N^2)$ flops. Applying
a sequence of planar rotators guarantees normwise backward stability.

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
  system with the decoupled `DenseStorage<T>` trait, whose `type R`/`type C`
  are themselves `Dim`, enables compile-time matrix arithmetic bounds while
  keeping storage backends pluggable. Array leaves still take bare
  `const usize` capacities and bridge through `Const<N>: Dim`
  (`num-types-design.md` FR-3), so no array length is a parameter-dependent
  const expression.

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

The physical memory layout is decoupled from mathematical dimensions by the
three storage subsystems of `storage-design.md` §3:

- **Dense Strided (`DenseStorage`/`DenseStorageMut`)**: `type R: Dim`,
  `type C: Dim`, `isize` strides, `as_ptr()`, and dual checked/unchecked
  accessors. Contiguity is an orthogonal marker
  (`ContiguousStorage`/`ContiguousStorageMut`) carrying `as_slice()` and
  `const ORDER: MatrixLayout`. Leaves: `ArrayStorage`, `RowArrayStorage`,
  `StorageView`, `StorageViewMut`.
- **Packed Structured (`PackedStorage`/`PackedStorageMut`)**: `type N: Dim`,
  with physical slot lookup (`packed_index`) decoupled from algebraic
  evaluation (`value`). Leaves: `SymmetricPackedStorage`,
  `HermitianPackedStorage`, `TriangularPackedStorage`, `DiagonalStorage`,
  plus their typed views.
- **Sparse (`SparseStorage`, `CsrStorage`, `CscStorage`,
  `SparseVectorStorage`)**: Outside this document's scope; `Matrix` does not
  wrap a sparse leaf.

Ways of exposing the dense/packed split on the wrapper:

| Alternative                                                        | Status                                                                                                                                                                                                                                                                                                           |
|:-------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Adopted: Single `Matrix` struct with trait-gated `impl` blocks** | Defines one unified `Matrix<T, R, C, S>` struct. Trait bounds on `S` (`DenseStorage`, `PackedStorage`, `ContiguousStorage`) enable the appropriate methods per backend, with aliases (`PackedMatrix`, `DiagonalMatrix`, `MatrixSlice`, `Owned`) providing ergonomic naming without wrapper duplication (§4.1.1). |
| Separate struct wrappers (`Matrix` and `PackedMatrix`)             | Rejected: duplicates constructor, conversion, and inspection surface across two distinct wrapper types.                                                                                                                                                                                                          |
| A fifth `Matrix` parameter selecting the branch                    | Rejected: encodes in a generic what the leaf's own trait impl already decides; doubles every signature.                                                                                                                                                                                                          |
| Wrapper-level coordinate resolution branching                      | Rejected: incurs layout matching and arithmetic on every lookup; delegating to storage leaves enables zero-branch, monomorphized indexing.                                                                                                                                                                       |

#### 5.5. Factorization, Multiplication and Determinant Algorithm Choices

$LDL^T$ is the default solver for symmetric matrices: $O(N^3/3)$ operations,
no square-root evaluations, and no convenience `invert()` (Higham, 2002).
Near-singular or indefinite symmetric matrices are not handled via
block-pivoting (e.g. Bunch-Kaufman); callers needing that fall back to LU.
General non-symmetric systems use LU with partial pivoting ($O(2N^3/3)$); QR
is reserved for ill-conditioned or non-square systems ($O(4N^3/3)$).
Forming $A^T A$ to reduce a rectangular system to a symmetric one is
rejected: it squares the condition number ($\kappa(A^T A) = \kappa(A)^2$),
halving the number of valid decimal digits (Higham, 2002).

Matrix multiplication uses the naive triple loop at the crate's target
dimensions ($N \le 32$); block-tiled and SIMD/FFI variants are not adopted,
since their cache and code-size benefits do not offset their overhead at
this scale. Determinant is read from the LU or $LDL^T$ factorization's
diagonal product ($O(N)$ after factorization) rather than computed by
cofactor expansion ($O(N!)$, intractable past $N=3$).

---

### 6. Verification & Validation

#### 6.1. Objectives

- Demonstrate compile-time rejection of mismatched operand dimensions.
- Demonstrate numerical accuracy and backward stability across matrix
  multiplication, transposition, triangular substitution, and direct
  factorizations (LU, Cholesky, $LDL^T$, QR).
- Demonstrate exact coordinate access and zero-copy slicing semantics across
  both dense and packed storage subsystems.
- Demonstrate zero dynamic heap allocation in `#![no_std]` execution and bounded
  stack memory consumption.
- Demonstrate deterministic latency and absence of runtime panic landing pads
  across direct solvers and factorizations.

#### 6.2. Methods

| Method                    | Mechanism                                                | Requirements discharged  |
|:--------------------------|:---------------------------------------------------------|:-------------------------|
| Compile-time shape check  | Type-level `Dim` assertions, `compile_fail` doctests     | FR-1, C-1, C-4           |
| Requirements-based test   | `#[test]` unit tests over edge cases and singular inputs | FR-3, FR-4, FR-5, C-2    |
| Property-based test       | `proptest` suites verifying algebraic invariants         | FR-2, FR-6               |
| Doctest                   | Runnable doc examples in rustdoc                         | FR-2, FR-4               |
| Back-to-back comparison   | `examples/numerical-models/python3/matrix.py` vs `src/matrix.rs` JSON; [`numerical-models-design.md`](numerical-models-design.md) §6.3 | FR-2, FR-3               |
| Resource usage evaluation | `no_alloc` audit, `size_of` assertions, stack analysis                       | NFR-1, NFR-2, C-2, C-3   |
| On-target execution       | ETS suites under QEMU and Teensy hardware                | NFR-3                    |
| Coverage measurement      | `cargo coverage` reporting statement and branch metrics  | FR-1..FR-6, NFR-1..NFR-3 |

#### 6.3. Acceptance Criteria

| Claim                           | Oracle                              | Measure                     | Bound                                                                  | Justification                                                              |
|:--------------------------------|:------------------------------------|:----------------------------|:-----------------------------------------------------------------------|:---------------------------------------------------------------------------|
| LU factorization residual       | Manufactured solution / prototype   | Residual test ratio         | $r = \frac{\|A - LU\|_\infty}{N \|A\|_\infty \epsilon} < 20.0$         | Standard LAPACK test ratio threshold (Anderson et al., 1999; Higham, 2002) |
| Cholesky factorization residual | Manufactured SPD matrix / prototype | Residual test ratio         | $r = \frac{\|A - L L^T\|_\infty}{N \|A\|_\infty \epsilon} < 20.0$      | Backward error bound for Cholesky (Golub & Van Loan, 2013)                 |
| QR factorization orthogonality  | Identity $Q^T Q = I$                | Absolute error              | $\|Q^T Q - I\|_\infty < N \epsilon$                                    | Householder QR backward stability (Higham, 2002)                           |
| Linear system solve $A x = b$   | Closed-form manufactured solution   | Residual test ratio         | $\frac{\|A x - b\|_\infty}{\|A\|_\infty \|x\|_\infty \epsilon} < 20.0$ | Standard linear system backward error (Higham, 2002)                       |
| Transposition identity          | $(AB)^T = B^T A^T$                  | Exact equality / Rel. error | $0$ (exact) / $\le 2\epsilon$                                          | Algebraic ring property over floating-point / integer scalars              |
| Coordinate retrieval            | In-bounds / Out-of-bounds queries   | Exact equality              | `Some(v)` (exact) / `None`                                             | Structural indexing invariants across row/col dense and packed layouts     |
| Singular matrix detection       | Known rank-deficient matrices       | Exact equality              | `Err(LinAlgError::SingularMatrix)`                                     | Determinant threshold / zero-pivot detection in factorization              |
| Zero-allocation guarantee       | Host memory allocator interception  | Exact equality              | 0 heap allocations                                                     | NFR-1 `#![no_std]` invariant                                               |

#### 6.4. Traceability

| Requirement                                     | Method                                           | Artifact                                               |
|:------------------------------------------------|:-------------------------------------------------|:-------------------------------------------------------|
| FR-1 — Compile-Time Shape Verification          | Compile-time shape check                         | rustdoc `compile_fail` doctests in `src/matrix/mod.rs`              |
| FR-2 — Matrix Algebra & Linear Transformations  | Property-based test, Back-to-back comparison     | `src/matrix/tests/matrix_tests.rs::prop_add_associativity`          |
| FR-3 — Fallible Factorizations & Direct Solvers | Requirements-based test, Back-to-back comparison | `src/matrix/tests/matrix_tests.rs::test_lu_solve_mut`               |
| FR-4 — Coordinate Element Access                | Requirements-based test                          | `src/matrix/tests/matrix_tests.rs::test_coordinate_access`          |
| FR-5 — Structural Specializations               | Property-based test, Requirements-based test     | `src/matrix/tests/matrix_tests.rs::test_symmetric_packed`           |
| FR-6 — Zero-Copy Submatrix Views                | Property-based test, Requirements-based test     | `src/matrix/tests/matrix_tests.rs::test_strided_submatrix`          |
| NFR-1 — Zero-Allocation Deterministic Execution | Resource usage evaluation                        | `#![no_std]` host check & `size_of` assertions                      |
| NFR-2 — Interoperable C-ABI Layout              | Resource usage evaluation                        | `src/matrix/tests/matrix_tests.rs::test_c_abi_layout`               |
| NFR-3 — Predictable Real-Time Latency           | On-target execution                              | ETS suite `matrix_test_suite`                                       |
| C-1 — Stable Rust Toolchain                     | Compile-time shape check                         | Workspace build on `stable` Rust                       |
| C-2 — Stack Footprint Limit                     | Resource usage evaluation                        | `clippy::large_stack_arrays` CI check                  |
| C-3 — `#![no_std]` Environment                  | Resource usage evaluation                        | Compilation under `#![no_std]` target triples          |
| C-4 — In-Place Factorization Mutability         | Compile-time shape check                         | Type bound verification on `DenseStorageMut`           |

#### 6.5. Coverage

- **Target**: $\ge 90\%$ statement coverage, $\ge 85\%$ branch coverage reported
  via `cargo coverage`.
- **Excluded**: Target-specific ARM assembly branches tested exclusively under
  on-target ETS execution, and non-functional `core::fmt::Debug`
  implementations.

#### 6.6. Validation

- **Matrix Arithmetic, Linear Solves, & Inversion**: End-to-end numeric integrity
  verification in `examples/numerical-models/src/matrix.rs` executing matrix
  construction, arithmetic (`+`, `-`, `*`), transposition, $LU$ decomposition
  solving $Ax = b$, matrix inversion with identity check ($A \cdot A^{-1} = I$),
  Hilbert $n=8$ solve/inverse (residual and $\tau\kappa\varepsilon$), and timed
  GEMM $n=64$, without dynamic heap allocation.
- **Hardware DSP Interoperability**: Slicing contiguous memory (`as_slice()`) to
  pass directly into CMSIS-DSP vector routines without intermediate buffers.

#### 6.7. Not Verified

- Dynamic sparse linear solves are not verified in this document (deferred per
  §8 open questions to dedicated sparse linear dynamics scoping).
- Trans-architecture floating-point bitwise equivalence is not claimed across
  differing hardware FPU implementations (FMA vs non-FMA rounding differences).
- $1024\times 1024$ GEMM/LU cache-stress is not in the example crate; host
  generators use Hilbert $n=8$ and GEMM $n=64$
  ([`numerical-models-design.md`](numerical-models-design.md) §6.6). MCU C-2
  ($R, C \le 128$) is unchanged.

---

### 7. Performance & Resource Considerations

- **Stack Overhead**: Inline stack-allocated matrix capacities are strictly
  capped at $128 \times 128$ elements ($R::USIZE \times C::USIZE \le 16{,}384$),
  matching the `Const<N>: Dim` range of `num-types-design.md` C-1 and C-3.
- **Static Memory Footprint**: Dense array storage
  requires $R \times C \times \text{size\_of}(T)$ bytes on stack;
  symmetric/triangular packed storage
  requires $\frac{N(N+1)}{2} \times \text{size\_of}(T)$ bytes, achieving
  a $\approx 50\%$ RAM saving for covariance representations.
- **Zero-Copy Views**: `MatrixSlice` and `MatrixSliceMut` occupy 2 pointer words
  plus stride metadata, incurring zero allocation and constant-time setup.
- **Worst-Case Execution Time (WCET)**: Subprogram invocations delegate directly
  to non-allocating unrolled loops without panic landing pads or runtime
  branching overhead.

---

### 8. Risks & Open Questions

- **Const Generics Compile Overhead**: Trait bounds with type-level dimension
  arithmetic (`DimAdd`, `DimMul`) may increase compile times or produce complex
  diagnostics on compilation failure.
- **Fixed-Point & Quantized Precision Loss**: Truncation and rounding errors in
  fixed-point / Q-format accumulators may accumulate drift in high-rate control
  loops.
- **Hermitian Specialization Wrapper Scoping**: `Complex<T>` satisfies
  `T: Scalar`, admitting complex matrices across ring operators and
  factorizations. Whether dedicated high-level wrappers for Hermitian
  structures (analogous to `Symmetric`) are necessary beyond type aliases is an
  open API design question.
- **Sparse Dynamics Matrix Scoping**: Sparse matrix storage (`storage-design.md`
  FR-11..FR-15) and SpBLAS routines (`subprograms-design.md` FR-5) remain
  unconsumed by the core `Matrix` wrapper. Scoping whether sparse linear
  dynamics belong in `Matrix` or directly in `state-space-design.md` is an open
  question.

---

### 9. Development Plan

| Task / Feature               | Description                                                                                                                                                                                                                             | Estimated Effort |
|:-----------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
| **Step 1: Core Layout**      | Define unified `Matrix<T, R, C, S>` struct and storage aliases (`Owned`, `MatrixSlice`, `DiagonalMatrix`, `SymmetricPacked`, etc.); implement delegating lookup accessors (`get`/`set`/`value`) and the `ContiguousStorage` slice path. | 2.0 Days         |
| **Step 2: Operators**        | Implement `Add`, `Sub`, `Neg`, `Mul` over `T: Scalar` with compile-time shape checks, passing typed storage operands per §4.5.2.                                                                                                        | 1.5 Days         |
| **Step 3: Solvers**          | Wrap `Getrf`/`Getrs` for LU, add $LDL^T$, determinants, and in-place inversion over `DenseStorageMut`.                                                                                                                                  | 2.0 Days         |
| **Step 4: Specializations**  | Create `UpperTriangular`, `LowerTriangular`, `Symmetric` wrappers and their packed counterparts.                                                                                                                                        | 1.5 Days         |
| **Step 5: Factorizations**   | Wrap `Potrf`/`Potrs` (Cholesky, real and complex) and `Geqrf`/`Ormqr`/`Unmqr` (QR) with typed workspaces.                                                                                                                               | 2.0 Days         |
| **Step 6: Verification**     | Set up `proptest` suites, dual-subsystem and strided-view coverage (§6.1), complex-scalar cases, ARM DWT cycle profiling, and Cachegrind setups per [`vv-standards.md`](../vv-standards.md).                                                                  | 2.5 Days         |
| **Step 7: Interoperability** | Implement conversions between `Matrix`, `Polynomial` (Faddeev-LeVerrier), and `Tensor`.                                                                                                                                                 | 2.0 Days         |

---

### 10. References

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
10. **Greif, C., He, S., & Liu, P. (2016).** SYM-ILDL: Incomplete $LDL^T$
    Factorization of Symmetric Indefinite and Skew-Symmetric Matrices. _arXiv:
    1505.07589_. — $O(n)$ per-step pivot-search cost for symmetric partial
    pivoting, bounding the `Iamax` work per elimination step.
11. **Higham, N. J., & Tisseur, F. (2000).** A Block Algorithm for Matrix 1-Norm
    Estimation, with an Application to 1-Norm Pseudospectra. _SIAM Journal on
    Matrix Analysis and Applications_, 21(4). doi: 10.1137/S0895479899356080. —
    Multiple-right-hand-side triangular solves arising in LU-based solver paths.
12. **PLASMA (Univ. of Tennessee Innovative Computing Laboratory). (2025).**
    `plasma_2.4.5/include/cblas.h` Source File. _PLASMA
    Documentation_. [Online].
    Available: https://icl.utk.edu/plasma/docs/cblas_8h_source.html. Accessed:
    Aug. 8, 2026. — `CBLAS_ORDER` as a per-call argument rather than a routine
    property, and the `lda`/`ldb`/`ldc` positions in `cblas_sgemm`, behind
    §4.2's layout-forwarding rule and §4.5.2's operand table.

---

### 11. Revision History

| Revision | Date            | Author          | Description                                                                                                                                           |
|:---------|:----------------|:----------------|:------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1.0      | July 12, 2026   | @MitchellDScott | Initial draft: core matrix representation, operations, and zero-allocation scratch patterns.                                                          |
| 1.1      | July 26, 2026   | @MitchellDScott | Matrix decompositions: added LU, QR, Cholesky, and LDLT factorizations with LAPACK subroutine mapping.                                                |
| 1.2      | August 16, 2026 | @MitchellDScott | Storage subsystem parameterization: integrated decoupled storage traits (`DenseStorage`, `PackedStorage`) with static array backends.                |
| 1.3      | August 19, 2026 | @MitchellDScott | View & layout abstractions: added column-major and row-major storage types, submatrices, and strided views.                                          |
| 1.4      | August 25, 2026 | @MitchellDScott | Unified matrix representation: consolidated `Matrix<T, R, C, S>` struct across dense and packed backends with specialized type aliases.              |
| 1.5      | August 25, 2026 | @MitchellDScott | V&V standardization: upgraded test oracles, residual bounds ($\tau = 20.0$), and structured matrix verification.                                      |
| 1.6      | August 26, 2026 | @MitchellDScott | Storage view retarget: updated references to `StorageView`/`StorageViewMut` and `Const<N>` dimensions.                                                |
| 1.7      | August 26, 2026 | @MitchellDScott | Collapsed subprogram inventory; crate-wide standards cite `vv-standards.md`.                                                                          |
| 1.8      | August 28, 2026 | @MitchellDScott | Host-scale V&V: Hilbert and $1000\times 1000$ rows; umbrella $\tau\kappa\varepsilon$ and Instant timing. Caps unchanged.                             |
| 1.9      | August 28, 2026 | @MitchellDScott | Host-scale $1024\times 1024$ `ArrayStorage` (no heap); C-2 MCU cap unchanged.                                                                        |
| 1.10     | August 28, 2026 | @MitchellDScott | Example crate: Hilbert $n=8$ and timed GEMM $n=64$; $1024\times 1024$ remains out. Caps unchanged.                                                  |

