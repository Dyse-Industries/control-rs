# Matrix Type & Structural Specializations (Design Document)

![Date Badge](https://img.shields.io/badge/Date-August_24,_2026-blue)
![Status Badge](https://img.shields.io/badge/Doc%20Status-Draft-orange)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

---

### 1. Introduction

This module provides the matrix operations required by advanced control and
state estimation. Its architecture is modeled on the `nalgebra` linear algebra
crate (licensed under BSD-3-Clause, by Sébastien Crozet).

The following elements of `nalgebra` are directly mirrored or adapted in our
architecture:

- **Matrix Signature**: mirrors `nalgebra`'s decoupled storage pattern (§4.1).
  Shape lives on the storage leaf's associated types; the wrapper re-exposes
  it as `R`/`C` struct parameters bound to those associated types
  (§4.1, §4.9.2).
- **Storage Trait Hierarchy**: The dense and packed subsystems of
  `storage-design.md` (Rev 1.8) share no supertrait. `DenseStorage<T>`
  carries `type R: Dim`, `type C: Dim`, `isize` strides, `as_ptr()`,
  `get`/`get_unchecked`; `DenseStorageMut<T>` adds `as_mut_ptr()`,
  `get_mut`/`get_mut_unchecked` and `set`. `PackedStorage<T>` carries
  `type N: Dim`, `uplo()`, `packed_index`/`packed_index_unchecked` and the
  algebraic `value`/`value_unchecked`. Packed leaves are
  `SymmetricPackedStorage`, `HermitianPackedStorage`,
  `TriangularPackedStorage` and `DiagonalStorage`.
- **Dense Storage**: `ArrayStorage<T, R, C>`'s nested `[[T; R]; C]`
  (column-major) and `RowArrayStorage<T, R, C>`'s nested `[[T; C]; R]`
  (row-major). Capacities are bare `const usize` parameters on the leaves,
  which implement the trait at `type R = Const<R>; type C = Const<C>;`
  (`storage-design.md` C-4).
- **Matrix Views**: `MatrixSlice`/`MatrixSliceMut` over
  `ViewStorage`/`ViewStorageMut`, whose arbitrary `isize` strides carry both
  orderings and zero-copy transposition (`storage-design.md` FR-5, FR-6).

---

### 2. Requirements

#### 2.1. Functional Requirements

- **FR-1 — Compile-Time Shape Verification**: Operand dimensions are
  validated at compile time via `Dim` type bounds.
- **FR-2 — Fallible Linear Algebra Solvers**: Matrix inversion and system
  solving ($A x = b$) over singular matrices return `Err` rather than
  panicking or producing invalid numerical outputs.
- **FR-3 — Value-Preserving Structure Conversions**: Conversions between dense,
  symmetric, triangular, and 1D polynomial/vector views preserve element values
  and coordinates without numerical truncation.
- **FR-4 — Branchless Delegated Coordinate Lookup**: `Matrix` resolves a logical
  coordinate $(i, j)$ by delegating directly to `DenseStorage::get` (strided
  backends) and `PackedStorage::value` (packed backends), eliminating
  wrapper-level layout matching branches and index arithmetic.
- **FR-5 — Upstream Kernel Precondition Enforcement**: `Matrix` is the
  enforcement point for the subprogram operand preconditions
  (`subprograms-design.md` C-1). Kernels receive typed storage operands
  whose shape and strides come from the operand's own type; no call site
  accepts a raw slice or a dynamic stride.
- **FR-6 — Static Dimension Inference**: Operators infer output dimensions
  from input operands at compile time without dynamic capacity checks.

#### 2.2. Non-Functional Requirements

- **NFR-1 — Zero Dynamic Allocation**: All core operations execute strictly
  on stack-allocated buffers (`#![no_std]`).
- **NFR-2 — Memory Layout Guarantees**: Matrices over a leaf that also
  implements `ContiguousStorage<T>` expose a padding-free slice
  (`as_slice()`) conforming to C-ABI layout (`storage-design.md` FR-3).
- **NFR-3 — Zero Inlined Bounds Checks in Solvers**: Hot substitution loops
  and factorization steps address memory through subprogram kernels
  (`Trsv`, `Geru`, `Gemm`) whose unchecked accessors compile to branchless
  pointer arithmetic, allowing LLVM to prove bounds safety without emitting
  runtime checks or panic paths (`storage-design.md` NFR-3).

#### 2.3. Constraints

- **C-1 — Stable Rust Toolchain**: Code must compile on `stable` Rust without
  relying on `generic_const_exprs`.
- **C-2 — BLAS Level Alignment**: Level 2 and Level 3 linear algebra operations
  map directly onto the subprogram traits of `subprograms-design.md` §4.2
  (`Gemv`, `Gemm`, `Trsv`, `Geru`, …), dispatched as associated functions on
  a backend marker type.
- **C-3 — In-Place Factorization Restricted to Mutable Dense**: In-place
  factorizations require `DenseStorageMut<T>`, implemented by the owning
  array leaves (`ArrayStorage`, `RowArrayStorage`) and `ViewStorageMut`, not
  by the packed or sparse leaves.

---

### 3. Technical Overview

`Matrix<T, R, C, S>` is generic over element type `T`, row and column
dimensions `R: Dim, C: Dim`, and a strided storage backend
`S: DenseStorage<T, R = R, C = C>`. The struct places no constraint on
whether the underlying storage is owning or borrowed, column-major or
row-major: `ArrayStorage`, `RowArrayStorage`, `ViewStorage` and
`ViewStorageMut` are all admissible leaves, and their `isize` strides carry
the ordering.

Packed structured matrices are a sibling wrapper, `PackedMatrix<T, N, S>`
over `S: PackedStorage<T, N = N>` (§4.1.1). `storage-design.md` Rev 1.8
decouples the dense and packed subsystems entirely — they share no
supertrait — so no single struct bound admits both, and the per-branch split
replaces the former Tier-1 floor.

`Matrix` delegates element lookup directly to its storage leaf without
wrapper-level branching, enabling monomorphized zero-overhead indexing across
both orderings.

---

### 4. Architecture

The `Matrix` struct will be implemented in a new submodule: `src/matrix/mod.rs`.

#### 4.1. Core Matrix Types & Storage Hierarchy

```rust
pub struct Matrix<T, R: Dim, C: Dim, S: DenseStorage<T, R = R, C = C>> {
    storage: S,
    _marker: PhantomData<(T, R, C)>,
}

// Column-Major Matrix Aliases
pub type ArrayMatrix<T, const R: usize, const C: usize> =
    Matrix<T, Const<R>, Const<C>, ArrayStorage<T, R, C>>;
pub type Owned<T, const R: usize, const C: usize> = ArrayMatrix<T, R, C>;

pub type MatrixSlice<'a, T, R, C> = Matrix<T, R, C, ViewStorage<'a, T, R, C>>;
pub type MatrixSliceMut<'a, T, R, C> =
    Matrix<T, R, C, ViewStorageMut<'a, T, R, C>>;

// Row-Major Matrix Aliases
pub type RowArrayMatrix<T, const R: usize, const C: usize> =
    Matrix<T, Const<R>, Const<C>, RowArrayStorage<T, R, C>>;
pub type RowOwned<T, const R: usize, const C: usize> = RowArrayMatrix<T, R, C>;
```

`ArrayStorage<T, R, C>` and `RowArrayStorage<T, R, C>` take bare
`const usize` capacities and implement `DenseStorage<T>` at
`type R = Const<R>; type C = Const<C>;` (`storage-design.md` FR-4, C-4).
Neither projects `Dim::USIZE` into an array length, so neither requires
`generic_const_exprs`.

`ViewStorage`/`ViewStorageMut` are `Dim`-generic rather than
const-generic (`storage-design.md` FR-5): they carry runtime `isize`
strides, so one view type covers column-major, row-major, transposed and
reversed windows without a separate alias per ordering.

The point of decoupling storage is to have one `Matrix` implementation that
works for every strided leaf.

##### 4.1.1. Dense Struct Bound and the Packed Sibling

`storage-design.md` Rev 1.8 organizes storage into three decoupled
subsystems (dense strided, packed structured, sparse) with **no common
supertrait**. The former Tier-1 `MatrixStorage` floor no longer exists, so
there is no bound that admits a strided leaf and a packed leaf at once.
`Matrix` therefore names the dense branch, and packed structured matrices
get a sibling wrapper:

```rust
pub struct PackedMatrix<T, N: Dim, S: PackedStorage<T, N = N>> {
    storage: S,
    _marker: PhantomData<(T, N)>,
}

pub type SymmetricPacked<T, const N: usize, const L: usize> =
    PackedMatrix<T, Const<N>, SymmetricPackedStorage<T, N, L>>;
pub type HermitianPacked<T, const N: usize, const L: usize> =
    PackedMatrix<T, Const<N>, HermitianPackedStorage<T, N, L>>;
pub type TriangularPacked<T, const N: usize, const L: usize> =
    PackedMatrix<T, Const<N>, TriangularPackedStorage<T, N, L>>;
pub type DiagonalMatrix<T, const N: usize> =
    PackedMatrix<T, Const<N>, DiagonalStorage<T, N>>;
```

Capability per bound:

| Bound                  | Operations                                                                                     |
|:-----------------------|:-----------------------------------------------------------------------------------------------|
| `DenseStorage<T>`      | Strided lookup (`get`, `get_unchecked`), Level 2/3 kernel calls, views (§4.5, §4.9.2)          |
| `DenseStorageMut<T>`   | Mutable access (`get_mut`, `get_mut_unchecked`, `set`), in-place factorization (§4.7)          |
| `ContiguousStorage<T>` | `as_slice()`, `const ORDER: MatrixLayout`, C-ABI and FFI hand-off (§4.3)                       |
| `PackedStorage<T>`     | Packed lookup (`packed_index`, `value`, `value_unchecked`), `uplo()` (§4.9.2)                  |
| `PackedStorageMut<T>`  | Physical-slot mutation (`set`, `set_unchecked`) with structural rejection (§4.9.2)             |

This reverses the Rev 1.31 decision recorded in §5.4, which chose a single
struct over the shared floor. That floor is gone; §5.4 records the reversal
and §7 tracks the surface it costs.

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
  conventions (Anderson et al., 1999) and embedded DSP libraries (ARM CMSIS-DSP).
  Row-major layout allows direct zero-copy interfacing with C/C++ libraries and
  hardware sensor/actuator rasters without transposition.

*The default storage backend (`Owned`) aliases **column-major**
`ArrayStorage<T, R, C>` (`[[T; R]; C]`).*

#### 4.3. Memory Representation, Slicing & Views

To ensure stable memory layout and compatibility with C-based hardware
libraries, the owning array leaves store a padding-free nested array and
implement the contiguity markers of `storage-design.md` FR-3.

Slice access is a capability of `ContiguousStorage<T>`, not of every dense
leaf: a `ViewStorage` with a non-unit stride has no contiguous slice to
return. The accessors are therefore bounded on the marker:

```rust
impl<T, R: Dim, C: Dim, S> Matrix<T, R, C, S>
where
    S: ContiguousStorage<T, R = R, C = C>,
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
    S: ContiguousStorageMut<T, R = R, C = C>,
{
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.storage.as_mut_slice()
    }
}
```

Two access paths, one per consumer:

| Accessor                | Bound                  | Consumer                                              |
|:------------------------|:-----------------------|:------------------------------------------------------|
| `as_slice`              | `ContiguousStorage`    | Inspection, C-ABI and hardware-backend hand-off       |
| storage operand (`&S`)  | `DenseStorage`         | Every Level 1/2/3 kernel (`Axpy`, `Gemv`, `Gemm`, …)  |

Each has a `_mut` counterpart (`as_mut_slice`, `&mut S`) on the
corresponding `ContiguousStorageMut`/`DenseStorageMut` bound.

Kernels no longer take flattened or nested array operands. Subprogram traits
are parameterized over the storage types themselves
(`subprograms-design.md` FR-9), so a call site passes `&self.storage` and the
kernel reads shape from `S::R`/`S::C` and addresses through
`as_ptr()` plus the leaf's strides. This removes the former
`as_array::<N>()` / `as_nested()` pair along with the
`Const<R>: DimMul<Const<C>, Output = …>` bound each call site had to carry,
and it removes the flattened `&[T; R * C]` accessor that
`generic_const_exprs` blocked in the first place (`storage-design.md` §5).

##### 4.3.1. Zero-Copy Views (`ViewStorage`)

`Matrix` builds on the strided view backends of `storage-design.md` FR-5 and
FR-6. `ViewStorage<'a, T, R, C>` carries arbitrary `isize` strides, which
subsumes the former column-panel restriction: an arbitrary submatrix window
is a pointer offset plus the parent's own strides, with no padded leading
dimension to leave untouched.

- `ArrayMatrix::view()` / `view_mut()` borrow the whole buffer at the owning
  alias's shape, with $RS = 1, CS = R$.
- `Matrix::submatrix::<R2, C2>(origin)` offsets the base pointer by
  $r_0 \cdot RS + c_0 \cdot CS$ and keeps the parent strides. The window is
  fallible in the offset alone ($r_0 + R2 \le R$, $c_0 + C2 \le C$), and
  wrapping an erased-length slice returns
  `ConversionError::DimensionMismatch` (`storage-design.md` §4.6).
- **Zero-Copy Transposition**: `transpose_view()` swaps strides and
  dimensions (`storage-design.md` FR-6), so $A^T$ needs no buffer. Algebraic
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
  leaf as a `PackedMatrix` (§4.1.1). Off-diagonal coordinates are unstored
  and evaluate algebraically to `T::ZERO` (§4.9.2). This backend reaches no
  dense Level 2/3 kernel, because it implements `PackedStorage`, not
  `DenseStorage`; packed operands instead reach the packed kernels
  (`Spmv`, `Hpmv`, `Tpmv`, `Tpsv`; `subprograms-design.md` FR-3), or are
  converted to a dense leaf through `ToDenseStorage`
  (`storage-design.md` FR-16) before matrix-matrix arithmetic.

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
    SA: DenseStorage<T, R = M, C = N>,
    SB: DenseStorage<T, R = N, C = P>,
{
    pub fn mul_into<SC>(&self, rhs: &Matrix<T, N, P, SB>, out: &mut Matrix<T, M, P, SC>)
    where
        SC: DenseStorageMut<T, R = M, C = P>;
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

`Matrix` is a caller of `subprograms.rs`, not an extension of it. Every kernel
this design needs is one of the subprogram traits defined in
`subprograms-design.md` §4.2 and §4.3.

| Subprogram      | Level | Operation                      | Required by                                                                                                                       |
|:----------------|:------|:-------------------------------|:------------------------------------------------------------------------------------------------------------------------------------|
| `Axpy`          | 1     | $y \leftarrow \alpha x + y$    | `Add`, `Sub` (§4.5); reflector application in QR (§4.7)                                                                           |
| `Scal`          | 1     | $x \leftarrow \alpha x$        | `Neg` (§4.5); pivot-row normalization in LU (§4.6); diagonal scaling in $LDL^T$ (§4.7)                                            |
| `Dotu` / `Dotc` | 1     | $x^T y$ / $x^H y$              | Inner-product accumulation in substitution (§4.7.2, §4.10.1) and $LDL^T$/Cholesky diagonal updates (§4.7). `Dotc` on complex operands |
| `Nrm2`          | 1     | $\lVert x \rVert_2$            | Householder reflector construction in QR (§4.7)                                                                                   |
| `Iamax`         | 1     | $\arg\max_i (\lvert \text{Re} \rvert + \lvert \text{Im} \rvert)$ | Partial-pivot column search in LU (§4.6) and the symmetric pivot search in $LDL^T$ (Greif et al., 2016) |
| `Geru` / `Gerc` | 2     | $A \leftarrow \alpha x y^T + A$ | LU trailing-submatrix rank-1 updates. `Gerc` supplies the conjugated form on complex operands                                     |
| `Trsv`          | 2     | $x \leftarrow \text{op}(A)^{-1} x$ | Triangular solves and forward/backward substitution (§4.7.2, §4.10.1)                                                         |
| `Gemv`          | 2     | $y \leftarrow \alpha \text{op}(A) x + \beta y$ | `Mul<Vector>` (§4.5); matrix-vector products in the solver paths (§4.7.2)                                             |
| `Gemm`          | 3     | $C \leftarrow \alpha \text{op}(A)\text{op}(B) + \beta C$ | `Mul<Matrix>` (§4.5); trailing-submatrix block updates in LU, $LDL^T$ and QR                                |

Scalar bounds follow `subprograms-design.md` §4.1. The ring kernels
(`Axpy`, `Scal`, `Dotu`, `Dotc`, `Geru`, `Gerc`, `Gemv`, `Gemm`) require
`T: Scalar`; the field kernels (`Nrm2`, `Trsv`) require `T: Scalar + Div`
with `T::Real: Radical`. `T: Float` is `f32`/`f64` only and is never used as
a stand-in for `Complex<T>` (`num-traits-design.md` FR-5).

`Geru`/`Gerc` express LU elimination over trailing submatrices, avoiding the
three-loop dispatch overhead of `Gemm` with $k = 1$. `Trsv` accelerates
triangular solves and substitution in the linear system solvers.

The dense factorizations are no longer hand-rolled here. `subprograms-design.md`
FR-6 and FR-7 supply `Getrf`/`Getrs` (LU with partial pivoting),
`Potrf`/`Potrs` (Cholesky) and `Geqrf`/`Ormqr`/`Unmqr` (Householder QR) as
subprogram traits returning `LinAlgResult<()>`, so §4.7's decomposition
objects are wrappers over those kernels rather than independent
implementations (§4.7).

`Syrk`/`Trsm` remain outside this inventory: the §4.7 factorizations are
unblocked, right-looking variants whose trailing updates are rank-1
(`Geru`) or matrix-matrix (`Gemm`). They become required only if a blocked
variant is added, which the stack ceiling of §6.1 does not currently
motivate.

##### 4.5.2. Operand Derivation at the Call Site

Kernels take typed storage operands, so the layout parameters that were
formerly const generics at the call site are now properties the kernel reads
off the operand's own type (FR-5; `subprograms-design.md` FR-9). `Matrix`
supplies:

| Kernel input          | Source                                                                     |
|:----------------------|:----------------------------------------------------------------------------|
| Operand `&A` / `&mut Y` | `&self.storage`, typed `S: DenseStorage<T>`                                |
| Shape                 | `S::R::USIZE`, `S::C::USIZE` — monomorphization constants                   |
| Addressing            | `as_ptr()` with `r_stride()` / `c_stride()` (`storage-design.md` C-1)      |
| `trans` / `ta`, `tb`  | The algebraic operation, never the layout (§4.6)                            |
| `uplo`, `diag`, `side` | The structural intent of the call (`UpLo`, `Diag`, `Side`)                 |
| $\alpha$, $\beta$     | Scalars of type `T`, or `T::Real` on the real-scaled routines               |

Row traversal of column-major storage needs no gather into scratch and no
separate increment parameter: a row is a `ViewStorage` over the same buffer
with the strides swapped, which the Level 1 kernels consume directly
(`storage-design.md` FR-5, FR-6).

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
      `transpose_view()` produces a stride-swapped `ViewStorage` with no
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
(`storage-design.md` FR-16) before factorization, except where a packed
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
`Polynomial<T, <D as DimAdd<U1>>::Output>`.

- **Type Signature**:
  ```rust
  impl<T, D: Dim, S> TryFrom<Matrix<T, D, D, S>> for Polynomial<T, <D as DimAdd<U1>>::Output>
  where
      S: DenseStorage<T, R = D, C = D>,
      D: DimAdd<U1>,
      <D as DimAdd<U1>>::Output: Dim,
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
  what makes the mapping zero-copy: a strided `ViewStorage` has no such
  slice and converts by element copy instead.
- **Infallible Compile-Time Bound**: Dimensions and rank are verified statically
  at compile time via `Layout: TensorLayout<Size = <R as DimMul<C>>::Output>`.
  `ConversionError::LayoutMismatch` no longer exists (`error-design.md` §3).

#### 4.9. Error Handling & Element Lookup

##### 4.9.1. Compile-Time Constraints

Dimension mismatches (e.g., adding matrices of different sizes or multiplying
incompatible dimensions) fail at compile-time. Rust's type checker prevents
compiling invalid math.

##### 4.9.2. Element Lookup Across Both Storage Subsystems

`storage-design.md` FR-1, FR-2, FR-7 and FR-8 define logical coordinate
resolution on `DenseStorage` and `PackedStorage`. Each wrapper exposes those
accessors by delegating directly to `self.storage` (FR-4), eliminating
wrapper-level layout matching and runtime index arithmetic:

```rust
impl<T, R: Dim, C: Dim, S> Matrix<T, R, C, S>
where
    S: DenseStorage<T, R = R, C = C>,
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
    S: DenseStorageMut<T, R = R, C = C>,
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
    S: PackedStorage<T, N = N>,
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
`StorageError::InvalidHermitianDiagonal` (`storage-design.md` FR-8, §4.4).

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
wrapper. `storage-design.md` FR-9 provides four packed leaves
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

Both forms are now first-class. A full-square wrapper trades $N^2$ storage for
the dense kernels (`Trmv`, `Trsv`, `Symv`, `Hemv`); the packed sibling
(§4.1.1) trades a non-linear index map for $N(N+1)/2$ storage and reaches the
packed kernels (`Tpmv`, `Tpsv`, `Spmv`, `Hpmv`;
`subprograms-design.md` FR-3), which the previous four-tier hierarchy did not
offer. Hardware acceleration is no longer the deciding factor, since
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
    T: Scalar + Div<Output = T>,
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
      constraints at compile time (using `Dim` types). It ensures that the
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
  `ViewStorage`, `ViewStorageMut`.
- **Packed Structured (`PackedStorage`/`PackedStorageMut`)**: `type N: Dim`,
  with physical slot lookup (`packed_index`) decoupled from algebraic
  evaluation (`value`). Leaves: `SymmetricPackedStorage`,
  `HermitianPackedStorage`, `TriangularPackedStorage`, `DiagonalStorage`,
  plus their typed views.
- **Sparse (`SparseStorage`, `CsrStorage`, `CscStorage`,
  `SparseVectorStorage`)**: Outside this document's scope; `Matrix` does not
  wrap a sparse leaf.

Ways of exposing the dense/packed split on the wrapper:

| Alternative                                     | Status                                                                                                                                                                    |
|:------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| A shared storage floor named on one `Matrix` struct | **No longer available.** `storage-design.md` Rev 1.8 gives the dense and packed subsystems no common supertrait, so no such bound can be written                          |
| A fifth `Matrix` parameter selecting the branch | Rejected: encodes in a generic what the leaf's own trait impl already decides; doubles every signature                                                                    |
| **Adopted: per-branch wrappers (`Matrix` / `PackedMatrix`)** | Duplicates the constructor and inspection surface, which is the cost the upstream split imposes. In exchange each wrapper's accessors are exactly its branch's (§4.1.1) |
| Wrapper-level coordinate resolution branching   | Rejected: incurs layout matching and arithmetic on every lookup; delegating to storage leaves enables zero-branch, monomorphized indexing                                 |

This reverses the Rev 1.31 choice of a single struct over the Tier-1 floor.
The reversal is forced by the upstream split, not by a new preference here.

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

The matrix implementation is verified and validated across five structured
pillars to guarantee mathematical correctness, embedded safety, and real-time
execution predictability.

#### 6.1. Verification Strategy

1. **Compile-Time Verification**:
    - Matrix dimension matching ($M \times N \times N \times P \to M \times P$)
      is strictly enforced by the Rust type system using `Dim` types,
      completely eliminating runtime dimension checks and preventing invalid
      pointer arithmetic at compile time.
2. **Storage-Branch Coverage**:
    - Coordinate lookup (§4.9.2) is tested on both subsystems: `get` over
      `ArrayStorage`, `RowArrayStorage` and `ViewStorage`, and `value` over
      each packed leaf, asserting that an in-bounds implicit coordinate
      evaluates to its structural entry (reflected, conjugate-reflected,
      unit, or zero) and that only out-of-bounds coordinates yield `None`.
    - Submatrix views (§4.3.1) are passed to `Gemv`/`Gemm` and compared
      against the same window copied into a fresh `ArrayStorage`, asserting
      that a non-unit stride changes no result.
    - `transpose_view()` operands are asserted equal to the corresponding
      `Trans::Trans` kernel call, and `reverse_view()` operands to the
      $INCX < 0$ reference (`storage-design.md` §6.1 Level 2).
    - Both wrappers are exercised over `f64`, an integer scalar, and
      `Complex<f64>`, confirming the ring kernels admit all three and the
      field kernels reject the integer scalar at compile time.
    - The subprogram verification suite (`subprograms-design.md` §6.1) is the
      reference oracle for kernel correctness; `Matrix`'s own suite asserts
      the call sites pass the right operands, not that the kernels are
      correct.
3. **Property & Unit Testing**:
    - Host-based unit tests execute via `cargo test` to verify constructors,
      operators, triangular solvers, and slice bounds.
    - Property-based testing via `proptest` mathematically proves algebraic
      matrix identities (e.g., $(AB)^T = B^T A^T$, $A(B+C) = AB + AC$) over
      thousands of generated inputs.
    - Ill-conditioned, near-singular, and Hilbert matrices are tested to confirm
      safe, panic-free error degradation (`Err(LinAlgError::SingularMatrix)`).
4. **Hardware-in-the-Loop (HIL) & Cache Profiling**:
    - Cross-compiled binaries run on physical target microcontrollers (e.g., ARM
      Cortex-M4/M7) to profile L1 data/instruction cache misses (`I1mr`/`D1mr`),
      FPU cycle counts ($c_{\text{inner}}$), and hardware pipeline stall
      dependencies.
    - Cycle time for matrix multiplication is validated against the execution
      model:
      $$T \approx \frac{(n \cdot m \cdot k \cdot c_{\text{inner}}) + c_{\text{overhead}}}{f}$$
5. **Stack Bounds Verification**:
    - Inline stack-allocated matrix capacities are strictly capped
      at $128 \times 128$ elements ($R::USIZE \times C::USIZE \le 16{,}384$),
      matching the `Const<N>: Dim` range of `num-types-design.md` C-1 and the
      `Const`×`Const` product bound of C-3 rather than an independently
      chosen number.
    - This is a type-system bound, not a stack-safety guarantee: a
      $128 \times 128$ `f32` instance is
      $16{,}384 \times 4\text{ bytes} \approx 64\text{KB}$, well past typical
      2–8KB bare-metal stack budgets. `clippy::large_stack_arrays` is the
      actual enforcement point for call-site instance size; CI must fail on
      any un-justified `#[allow]` of that lint. `storage-design.md` §6.1
      Level 5 carries the corresponding per-target stack-budget check.

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
- **Layout query owner resolved**: ordering is carried by the leaf's `isize`
  strides and, for contiguous leaves, `ContiguousStorage::ORDER`. Coordinate
  resolution is encapsulated in the storage accessors, so no
  `match self.layout()` branch survives in `Matrix` (§4.2, §4.9.2).
- **Packed mutation resolved**: `PackedStorageMut` (`storage-design.md` FR-8)
  ships with the packed leaves, so `PackedMatrix` has a typed in-place write
  from the start. Writes are restricted to physical slots and reject
  structural violations by `StorageError` arm (§4.9.2).
- **Per-branch wrapper surface (new, this revision)**: splitting `Matrix` and
  `PackedMatrix` (§4.1.1) duplicates the constructor, conversion and
  inspection surface across two types. How much of that surface is worth
  sharing through a crate-private helper trait — as opposed to writing it
  twice — is unresolved and should be settled before implementation.
- **Sparse-backed matrices unscoped**: `storage-design.md` FR-11 to FR-15
  define a full sparse subsystem with its own SpBLAS kernels
  (`subprograms-design.md` FR-5). No wrapper in this document consumes it.
  Whether sparse dynamics matrices belong here or in
  `state-space-design.md` is an open scoping question.
- **Complex scalars reach `Matrix` for the first time**: `Complex<T>`
  satisfies `T: Scalar` (`num-traits-design.md` §4.3), so every ring
  operator and kernel in this document now admits it. The specializations of
  §4.10 name only the symmetric and triangular cases; the Hermitian
  equivalents (`HermitianPacked`, `Hemv`/`Hemm`/`Herk`) have no wrapper here
  yet.

---

### 8. Development Plan

| Task / Feature               | Description                                                                                                                                                                                        | Estimated Effort |
|:-----------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
| **Step 1: Core Layout**      | Define `Matrix` over `DenseStorage<T, R = R, C = C>` and the `PackedMatrix` sibling; delegating lookup accessors (`get`/`set`/`value`) and the `ContiguousStorage` slice path.                     | 2.0 Days         |
| **Step 2: Operators**        | Implement `Add`, `Sub`, `Neg`, `Mul` over `T: Scalar` with compile-time shape checks, passing typed storage operands per §4.5.2.                                                                   | 1.5 Days         |
| **Step 3: Solvers**          | Wrap `Getrf`/`Getrs` for LU, add $LDL^T$, determinants, and in-place inversion over `DenseStorageMut`.                                                                                             | 2.0 Days         |
| **Step 4: Specializations**  | Create `UpperTriangular`, `LowerTriangular`, `Symmetric` wrappers and their packed counterparts.                                                                                                   | 1.5 Days         |
| **Step 5: Factorizations**   | Wrap `Potrf`/`Potrs` (Cholesky, real and complex) and `Geqrf`/`Ormqr`/`Unmqr` (QR) with typed workspaces.                                                                                          | 2.0 Days         |
| **Step 6: Verification**     | Set up `proptest` suites, dual-subsystem and strided-view coverage (§6.1), complex-scalar cases, ARM DWT cycle profiling, and Cachegrind setups.                                                   | 2.5 Days         |
| **Step 7: Interoperability** | Implement conversions between `Matrix`, `Polynomial` (Faddeev-LeVerrier), and `Tensor`.                                                                                                            | 2.0 Days         |

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
    guarantees underpinning the `DenseStorage`/`ContiguousStorage` marker
    split and padding-free slice casting.
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
    and LAPACK trait definitions (`Axpy`, `Scal`, `Dotu`, `Dotc`, `Nrm2`,
    `Iamax`, `Gemv`, `Gemm`, `Trsv`, `Getrf`, `Potrf`, `Geqrf`); the inventory
    of kernels available to `Matrix` (§4.5.1).
16. **Greif, C., He, S., & Liu, P. (2016).** SYM-ILDL: Incomplete $LDL^T$
    Factorization of Symmetric Indefinite and Skew-Symmetric Matrices.
    _arXiv:1505.07589_. — $O(n)$ per-step pivot-search cost for symmetric
    partial pivoting, bounding the `Iamax` work per elimination step.
17. **Higham, N. J., & Tisseur, F. (2000).** A Block Algorithm for Matrix
    1-Norm Estimation, with an Application to 1-Norm Pseudospectra. _SIAM
    Journal on Matrix Analysis and Applications_, 21(4).
    doi: 10.1137/S0895479899356080. — Multiple-right-hand-side triangular
    solves arising in LU-based solver paths.
18. **PLASMA (Univ. of Tennessee Innovative Computing Laboratory). (2025).**
    `plasma_2.4.5/include/cblas.h` Source File. _PLASMA Documentation_.
    [Online]. Available:
    https://icl.utk.edu/plasma/docs/cblas_8h_source.html. Accessed:
    Aug. 8, 2026. — `CBLAS_ORDER` as a per-call argument rather than a
    routine property, and the `lda`/`ldb`/`ldc` positions in `cblas_sgemm`,
    behind §4.2's layout-forwarding rule and §4.5.2's operand table.

---

### 10. Revision History

| Revision | Date            | Author          | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
|:---------|:----------------|:----------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1.0      | July 12, 2026   | @MitchellDScott | Initial draft outlining core concepts, layout, and operations.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| 1.1      | July 19, 2026   | @MitchellDScott | Restructured to new template; added embedded performance and verification details.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| 1.2      | July 25, 2026   | @MitchellDScott | Added supporting bibliography and inline citations.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| 1.3      | July 26, 2026   | @MitchellDScott | Added Decomposition Objects, zero-copy MatrixView wrappers, and no_alloc scratch space patterns.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| 1.4      | July 26, 2026   | @MitchellDScott | Harmonized with storage trait design doc; updated `Matrix` definition, bounds, decomposition rules, and V&V.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| 1.5      | July 26, 2026   | @MitchellDScott | Added comprehensive 3-tiered bibliography and inline citations across core architectural sections.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| 1.6      | August 1, 2026  | @MitchellDScott | Corrected `nalgebra` comparison claims; clarified storage-decoupling benefit; added system-solving convenience note.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| 1.7      | August 2, 2026  | @MitchellDScott | Propagated `num-traits-design.md` pivot; removed duplicate MatrixView definitions; relocated `ConversionError`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| 1.8      | August 10, 2026 | @MitchellDScott | Realigned with updated math-module code                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| 1.10     | August 11, 2026 | @MitchellDScott | Condensed §4.5 operator→subprogram mapping into a scannable table (`Add`/`Sub`→`AXPY`, `Neg`→`SCAL`, `Mul<Matrix>`→`GEMM`, `Mul<Vector>`→`GEMV`).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| 1.13     | August 15, 2026 | @mitchelldscott | Reverted Doc Status to Draft. `storage-subprograms-design.md` rev 1.2 (August 14, 2026) replaced the flat `Storage`/`StorageMut`/`ContiguousStorage`/`ContiguousStorageMut` hierarchy this document's §1, §4.1, §4.2, §4.3, §4.3.1 and §5.4 still describe with a four-tier `Buffer`/`BlasStorage`/{`MatrixStorage`, `PackedStorage`, `StridedStorage`}/leaf hierarchy; `ArrayStorage`, `StorageView` and `StorageViewMut` no longer name a type that hierarchy defines. No section content changed in this revision; reconciliation (renaming `Matrix`'s `S` bound to the correct Tier-2 branch and updating the architecture prose) is tracked as its own maintenance pass.                                                                                            |
| 1.14     | August 16, 2026 | @mitchelldscott | Propagated four-tier `BlasStorage` hierarchy (§1, §4.1, §4.3, §4.3.1, §5.4): updated `Matrix` bound to `MatrixStorage<T, R, C>`, default storage alias to `Dense<T, R, C, Array<T, N>>`, slice methods to `BlasStorage`, and view aliases to `Dense` / `StridedView`. Applied B-2/D-2 resolution (§4.5.1): added `GER` and `TRSV` to required subprogram inventory.                                                                                                                                                                                                                                                                                                                                                                                                      |
| 1.15     | August 16, 2026 | @mitchelldscott | Harmonized with `storage-subprograms-design.md` Rev 1.4 (§1, §4.1): updated `Matrix` storage bound to single-parameter `S: MatrixStorage<T, R = R, C = C>` with associated `R`/`C` types, integrated `FixedBlasStorage<T>` array access (`as_array()`), and detailed monomorphized zero-cost subprogram delegation.                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| 1.16     | August 16, 2026 | @mitchelldscott | Harmonized with `storage-subprograms-design.md` Rev 1.5 (§1, §4.1): updated `BlasStorage<T, R, C, Stride>` and `MatrixStorage<T, R, C>` to keep `R: Dim`, `C: Dim`, `Stride: Dim` as generic parameters on the storage trait itself, enabling one storage implementor (`Dense`, `Array`, `Ref`, `RefMut`) to be used for any compile-time shape and stride.                                                                                                                                                                                                                                                                                                                                                                                                              |
| 1.17     | August 16, 2026 | @mitchelldscott | Completed `/cr-design-doc` pass (§1-§10): standardized typed `BlasStorage`/`MatrixStorage` + `&[T; N]` operand model; restored unique `MX-FR-*`/`MX-NFR-*`/`MX-C-*` requirement IDs; specified feature-selected backend dispatch (`#[cfg(feature = "accelerate")]`); aligned decomposition signatures (`LuDecomposition`, `LdltDecomposition`, `CholeskyDecomposition`, `QrDecomposition`) with shipped `src/matrix/decomposition.rs` code (`Const<D>: Dim`); eliminated `lu_decompose_mut` runtime `assert!`; added Tier-2 storage branch element lookup composition (`get`) with `R::USIZE` codegen folding constraint; unified `Symmetric`/`Triangular` leaf + newtype specialization story.                                                                          |
| 1.18     | August 16, 2026 | @mitchelldscott | Reconciled residual `Storage<T, R, C>` prose references in §5.1 and §8 with `MatrixStorage<T, R, C>` and `BlasStorage`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| 1.19     | August 16, 2026 | @mitchelldscott | Standardized requirement IDs in §2 from MX- prefixed tags to plain FR/NFR/C numbered format.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| 1.20     | August 16, 2026 | @mitchelldscott | Updated `Matrix` default storage parameter and view aliases (`Owned`, `MatrixSlice`, `MatrixSliceMut`) to convenience storage aliases (`DenseArray`, `DenseRef`, `DenseRefMut`).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| 1.21     | August 16, 2026 | @mitchelldscott | Encapsulated dimension multiplication inside `DenseArray<T, R, C>` in `storage.rs`, eliminating non-stable const generic math (`{ R * C }`) and extra capacity parameters from `Matrix`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| 1.22     | August 16, 2026 | @mitchelldscott | Removed obsolete `FixedBlasStorage` mention; decoupled `R: DimMul<C>` from `Matrix` struct definition; corrected decomposition `Owned<T, Const<D>, Const<D>>` type arguments; aligned `solve_lower_triangular` generic dimension parameters with `LowerTriangular<T, const D: usize>`; refined `Matrix::get` element lookup offset evaluation.                                                                                                                                                                                                                                                                                                                                                                                                                           |
| 1.23     | August 18, 2026 | @mitchelldscott | Reconciled `Matrix` struct storage bound to `S: MatrixStorage<T, R, C>` across §1 and §4.1; clarified `StridedMatrixSlice` as a standalone wrapper over `StridedView`; harmonized §4.9.2 coordinate lookup with Tier-2 `MatrixStorage`; aligned with `storage-subprograms-design.md` Rev 1.10 `&[T; N]` operand model.                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| 1.24     | August 18, 2026 | @mitchelldscott | Aligned §4.8.2 `Matrix` → `Tensor` conversion to infallible `From` bounded by `TensorLayout<Size = <R as DimMul<C>>::Output>`, eliminating obsolete `LayoutMismatch` runtime check.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| 1.25     | August 18, 2026 | @mitchelldscott | Propagated `storage-subprograms-design.md` Rev 1.11–1.12: `ArrayMatrix`/`Owned` take `const R, const C` over `DenseArray<T, R, C>`/`Array2`; `Matrix` itself stays `Dim`-generic with no owning default; views via FR-6 `view()`/`try_submatrix()`; subprogram operands via `as_array()`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| 1.26     | August 18, 2026 | @mitchelldscott | Propagated `storage-subprograms-design.md` Rev 1.16: `as_nested()` over `&[[T; R]; C]`; caller-supplied `R2`/`C2`/`LDA` on `try_submatrix`; kernel `trans` only (no `transpose_view` / `StridedView`); `BlasStorage` without `Stride`; `IMPLICIT` packed branch.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| 1.27     | August 19, 2026 | @mitchelldscott | Propagated `storage-subprograms-design.md` Rev 1.21 (Approved). Struct bound lowered to `BlasStorage<T, R, C>` with Tier-2 capability required per `impl` block (§4.1.1, §5.4). Accepted ownership of per-$(i, j)$ coordinate lookup for both branches (§4.9.2, FR-4) and of kernel operand preconditions (FR-5, §4.5.2), with `S::LDA` replacing the `lda()` query. Added the `Diagonal`-backed constructor and packed `value()` accessor; recorded `SYRK`/`TRSM` as not required while factorizations stay unblocked; restricted in-place paths to `Dense` (C-4); moved substitution off `get()` onto `TRSV` and fixed-array operands (NFR-3, §4.10.1). Added storage-branch, column-panel and row-strided verification (§6.1) and three upstream open questions (§7). |
| 1.28     | August 19, 2026 | @mitchelldscott | Delegated logical `(i, j)` coordinate lookup from `Matrix` to `MatrixStorage` (`get`/`get_unchecked`, `get_mut`/`get_unchecked_mut`) and `PackedStorage` (`value`/`value_unchecked`) concrete leaves (FR-4); eliminated wrapper-level `match self.layout()` branching and offset calculation; updated §1, §2.1, §3, §4.1.1, §4.9.2, §5.4, §7, and §8.                                                                                                                                                                                                                                                                                                                                                                                                                    |
| 1.29     | August 19, 2026 | @mitchelldscott | Integrated separate Column-Major and Row-Major storage types (`ColDense`/`RowDense`, `DenseColArray`/`DenseRowArray`, `RowOwned`, `RowMatrixSlice`, `RowMatrixSliceMut`) into matrix alias definitions, kernel layout forwarding, and documentation; updated §1, §4.1, §4.2, §5.4, and §10. |
| 1.30     | August 20, 2026 | @mitchelldscott | Propagated `storage-subprograms-design.md` Rev 1.31: `PackedStorage` is the packed addressing branch with Phase-1 `Diagonal` leaf; `SP`/`TP` added on the same branch after storage Phase 1; `PackedStorageMut` deferred until those leaves. |
| 1.31     | August 20, 2026 | @mitchelldscott | Renamed `BlasStorage`/`BlasStorageMut` -> `MatrixStorage`/`MatrixStorageMut` (universal floor) and the prior `MatrixStorage`/`MatrixStorageMut` (leading-dimension branch) -> `DenseStorage`/`DenseStorageMut`, matching `storage-subprograms-design.md` Rev 1.31; updated §1, §4.1, §4.1.1, §4.9.2, and the Alternatives table. |
| 1.32     | August 24, 2026 | @mitchelldscott | Retargeted onto the split math designs. `storage-subprograms-design.md` citations resolve to `storage-design.md` Rev 1.8 and `subprograms-design.md` Rev 1.6. Replaced the four-tier `Buffer`/`MatrixStorage`/`DenseStorage`/`PackedStorage` hierarchy with the dense/packed/sparse subsystems: `Matrix` binds `DenseStorage<T, R = R, C = C>`, `PackedMatrix` binds `PackedStorage<T, N = N>` (§4.1.1, §5.4), since the upstream split leaves no shared floor. `isize` strides and `transpose_view`/`reverse_view` replace `type LDA` and the column-panel restriction (§4.2, §4.3.1); typed storage operands replace `as_array`/`as_nested` and the call-site const generics (§4.3, §4.5.2). Subprogram names lowercased onto the trait surface and factorizations delegated to `Getrf`/`Potrf`/`Geqrf` (§4.5.1, §4.7). Scalar bounds moved to `T: Scalar` (ring) and `T: Scalar + Div` (field) per `num-traits-design.md` FR-5, admitting `Complex<T>`. Errors resolve to the four crate-wide enums of `error-design.md`. Status stays Draft. |
