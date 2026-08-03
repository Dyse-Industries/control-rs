# Storage Trait (Design Document)

![Date Badge](https://img.shields.io/badge/Date-August_1,_2026-blue)
![Status Badge](https://img.shields.io/badge/Doc%20Status-Draft-orange)
![Author Badge](https://img.shields.io/badge/Author-@mitchelldscott-blueviolet)

---

### 1. Introduction

`control-rs` is a `no_std` control systems library targeting bare-metal
microcontrollers. Every data structure that holds numerical arrays —
polynomials,
state-space models, transfer functions, and the eventual matrix type — must
decide _where_ that data lives in memory. Today, the codebase has three
incompatible answers to this question:

| Module                     | Storage Mechanism          | Dimension Encoding       |
|:---------------------------|:---------------------------|:-------------------------|
| `Polynomial` trait         | `&[T]` / `&mut [T]` slices | Runtime (slice length)   |
| `StaticPolynomial` (draft) | `[T; N::DIM]` inline array | Type-level `Dim` (Peano) |
| BLAS subprograms           | `&[T]` / `&mut [T]` slices | Runtime `usize` params   |
| DSP helpers                | `&[T; N]` array refs       | `const` generics         |

The `polynomial`, `state_space`, and `transfer_function` modules are currently
commented out in `lib.rs`. The active `Polynomial` trait exposes raw slice
accessors (`coefficients(&self) -> &[T]`), while the module-level docs in
`math/mod.rs` sketch a `StaticPolynomial<C: Dim>` backed by the Peano type
system. The BLAS subprograms operate on flat `&[T]` slices with explicit
runtime dimension parameters. No concrete `Matrix` or `Vector` type exists
anywhere in the codebase.

This means there is currently **no unified abstraction over where numerical
data lives in memory**. Every future type — matrices, state-space models,
transfer functions — will need to independently solve the same problem.

More critically, **stack allocation is the only option**. This works for small
structures (a 5th-order polynomial, a 4×4 transformation matrix) but is fatal
on embedded targets when the data grows. A 100×100 covariance matrix for a
Kalman filter requires 80 kB of `f64` storage — far exceeding the 2–8 kB
stack typical of Cortex-M0/M3 targets. Attempting to stack-allocate it causes
an immediate hard fault.

The `Storage` trait provides a single abstraction over _where_ data lives while
preserving the library's compile-time dimension guarantees. By parameterizing
data containers over a storage backend, all mathematical implementations become
agnostic to the allocation strategy — inline arrays, static buffers, or
borrowed slices — without any changes to the algorithms themselves.

---

### 2. Requirements

#### 2.1 Functional Requirements

##### FR-1: Core Storage Trait

The base `Storage<T, R, C>` trait defines the minimal interface that **every**
backend must implement — including non-contiguous layouts (strided, padded,
transposed views, DMA buffers). It contains only the methods that are valid
for _any_ memory arrangement:

- `ptr(&self) -> *const T` — raw pointer to the first element of the
  backing allocation. Required for DMA and FFI.
- `ptr_mut(&mut self) -> *mut T` — mutable raw pointer to the first element.
- `offset(&self, i: usize, j: usize) -> isize` — maps a logical matrix
  index `(i, j)` to a signed pointer displacement (in elements) from
  `ptr()`. The element at row `i`, column `j` lives at
  `ptr().offset(self.offset(i, j))`. Returning `isize` rather than `usize`
  allows implementations to encode reversed or non-standard layouts (e.g.,
  a negative stride) without physically rearranging the data.
- `get_unchecked(i, j) -> &T` — unchecked element access (`unsafe`).
  Provided as a default implementation:
  `&*self.ptr().offset(self.offset(i, j))`.
- `get_unchecked_mut(i, j) -> &mut T` — unchecked mutable element access
  (`unsafe`). Provided as a default implementation via `ptr_mut()` +
  `offset()`.

`offset()` is the single point of control for the storage's memory layout.
Swapping the `offset()` implementation is sufficient to change:

- **Row-major vs. column-major**: a row-major backend returns
  `(i * C::DIM + j) as isize`; a column-major backend returns
  `(j * R::DIM + i) as isize`.
- **Reversed storage**: returning
  `((R::DIM * C::DIM - 1) - (i * C::DIM + j)) as isize`
  reverses the element order without moving any data. A reversed polynomial
  or a flipped matrix row is just a different `offset()` implementation over
  the same buffer.
- **Strided / padded layouts**: hardware DMA buffers or SIMD-aligned rows
  can insert padding by adjusting the stride term in the offset calculation.

The following methods are explicitly **excluded** from the base trait:

- **`as_slice()` / `as_mut_slice()`** — a Rust slice (`&[T]`) guarantees
  contiguous memory. Exposing slices from a padded or strided backend would
  leak padding bytes as valid matrix elements, causing silent data corruption.
  These belong on the `ContiguousStorage` sub-trait (FR-4).
- **`get(i, j)` / `get_mut(i, j)`** — bounds checking is a _logical_
  operation against the matrix dimensions `R` and `C`, not a property of
  the storage layout. The `Matrix<T, R, C, S>` wrapper implements these by
  checking `i < R::DIM && j < C::DIM` and delegating to
  `Storage::get_unchecked()`. Keeping bounds checks out of the trait
  reduces boilerplate for custom hardware backends.

##### FR-2: Type-Level Dimension Encoding

The trait **must** be parameterized by the existing `Dim` type system:

```text
Storage<T, R: Dim, C: Dim>
```

where `R` is the row dimension and `C` is the column dimension. For 1-D
containers (polynomials, vectors), `C = U1`. The total element count
`N = R::DIM * C::DIM` is evaluated entirely at compile time via `DimMul`.
This means:

- Dimension mismatches in matrix operations (e.g., multiplying a 3×4 by a 5×2)
  are caught as compile errors — not runtime panics.
- The `Dim` trait's `DIM` associated constant provides the runtime value where
  needed (e.g., BLAS `m`, `n`, `k` parameters) without any additional
  bookkeeping.

##### FR-3: Core Storage Implementations

The module **must** ship the following concrete backends. Together they cover
the three fundamental ownership models (owned-stack, borrowed-immutable,
borrowed-mutable) and a scratch-data category for decomposition algorithms.

**FR-3a: Stack-Based Array Storage**

`ArrayStorage<T, R, C>` is the default owning backend. Internally it stores
a column-major 2-D const-generic array `[[T; R]; C]` and is annotated with
`#[repr(C)]` to guarantee a contiguous, predictable memory layout. Because
the columns are the outer dimension, column `j` is a single contiguous
`[T; R]` slice — matching the access pattern required by column-major BLAS
and enabling direct `as_slice()` over the entire allocation.

`ArrayStorage` implements `Storage`, `StorageMut`, `ContiguousStorage`, and
`ContiguousStorageMut`.

**FR-3b: Zero-Copy View Storage**

The module provides two non-owning storage types that hold references to
existing memory:

- `MatrixView<'a, T, R, C>` — wraps `&'a [T]`. Implements `Storage` and
  `ContiguousStorage`.
- `MatrixViewMut<'a, T, R, C>` — wraps `&'a mut [T]`. Implements all four
  traits in the hierarchy.

Views allow operations such as transposition by swapping the stride rules
in `offset()` without allocating new memory or copying any bytes. A
transposed view simply returns `(j * R::DIM + i)` instead of
`(i * C::DIM + j)`, reinterpreting the same underlying buffer.

**This document is the sole owner of the `MatrixView`/`MatrixViewMut`
struct definitions** (fields, `offset()` implementation, transposition
mechanics). Consuming documents (`matrix-design.md` and any future
`tensor-design.md`/`polynomial-design.md` revision) reference these types
by name and define only their own wrapper aliases on top
(`Matrix<T, R, C, MatrixView<'a, T, R, C>>`, etc.) — they must not restate
or re-derive the struct's field layout. `matrix-design.md` previously
carried its own competing struct sketch for these two types; that
duplication is removed as of `matrix-design.md`'s current revision (see
that document's revision history) specifically to close this gap.

**FR-3c: Auxiliary / Scratch Storage**

Decomposition algorithms (LU, QR, Cholesky) require auxiliary bookkeeping
arrays (e.g., permutation indices for pivoting). The module **must** provide
statically bounded scratch types such as `PivotStorage<D: Dim>` that hold
this auxiliary data on the stack without triggering dynamic heap allocations.
These types are _not_ general-purpose matrix storage — they do not implement
the full `Storage` trait — but they share the same `Dim`-parameterized,
`no_alloc` design philosophy.

##### FR-3d: Hardware-Specific Backends (Via the Storage Trait)

Because the storage trait is decoupled from the matrix math, the module is
specifically designed to allow users to plug in target-specific backends.
The following extension points **must** be supported by the trait's design
(even though the library does not ship these implementations):

- **Flash (Read-Only) Memory Wrappers** — users can define storage that
  points directly to read-only Flash memory. The module supports this by
  ensuring all static constructors (`zero()`, `identity()`, `diagonal()`)
  are implemented as `const fn`, allowing matrices to be baked directly
  into ROM. A Flash backend implements `Storage` and `ContiguousStorage`
  but **not** `StorageMut`.
- **DMA-Safe Static Memory Pools** — the architecture supports custom
  storage types for tightly coupled, custom-aligned memory buffers. This
  is required to safely hand off contiguous slice pointers to C-based
  hardware DSP/BLAS kernels and to implement true DMA double-buffering
  without memory tearing. A DMA backend implements the full trait hierarchy
  and may add alignment guarantees beyond what `#[repr(C)]` provides.

##### FR-4: Trait Hierarchy

The trait hierarchy **must** distinguish along two orthogonal axes:
mutability (read-only vs. read-write) and contiguity (strided vs. contiguous).

```text
                    Storage<T, R, C>
                   /                \
     StorageMut<T, R, C>    ContiguousStorage<T, R, C>
                   \                /
              ContiguousStorageMut<T, R, C>
```

- **`Storage<T, R, C>`** — the base trait (FR-1). Provides `ptr`, `offset`,
  and unchecked element access. Every backend implements this.
- **`StorageMut<T, R, C>: Storage<T, R, C>`** — extends with `ptr_mut` and
  `get_unchecked_mut`. Only backends with mutable access implement this.
  This distinction prevents, at compile time, code from attempting to mutate
  read-only lookup tables or const-embedded data.
- **`ContiguousStorage<T, R, C>: Storage<T, R, C>`** — a marker sub-trait
  for backends that guarantee all `R::DIM * C::DIM` elements are laid out
  contiguously in memory (no padding, no stride gaps). Provides:
    - `as_slice(&self) -> &[T]` — safe because the trait bound guarantees
      contiguity.
      Backends like `ArrayStorage` and `MatrixView` implement this. Strided,
      padded, or DMA-aligned backends do **not**.
    - `const ORDER: MatrixLayout` — declares which physical layout the flat
      slice from `as_slice()` represents (`MatrixLayout::RowMajor` or
      `MatrixLayout::ColMajor`). This is the mechanism that lets FR-6's BLAS
      interop work for *any* contiguous backend, not just column-major ones —
      see FR-6 and NFR-6.
- **`ContiguousStorageMut<T, R, C>: StorageMut + ContiguousStorage`** —
  combines mutability and contiguity. Provides:
    - `as_mut_slice(&mut self) -> &mut [T]`
      Backends like `ArrayStorage` and `MatrixViewMut` implement this.

This hierarchy allows the `Matrix` wrapper to conditionally expose methods.
For example, `Matrix<T, R, C, S>` only implements `as_slice()` when
`S: ContiguousStorage<T, R, C>`, and BLAS interop (FR-6) requires this bound.

##### FR-5: Initialization Strategies

Each backend **must** support safe initialization without requiring
`Default` on `T`:

- **From a closure**: `fn from_fn(f: impl FnMut(usize, usize) -> T) -> Self`
  — fills element-by-element using row/column indices.
- **From a value**: `fn from_element(val: T) -> Self where T: Clone` — fills
  every element with a clone of `val`.
- **Zeroed**: `fn zeros() -> Self where T: Zero` — fills with the additive
  identity.
- **Identity (square only)**: `fn identity() -> Self where T: Zero + One, R = C`
  — ones on the diagonal, zeros elsewhere.
- **From a flat slice (borrowed backends)**: wraps an existing `&[T]` or
  `&mut [T]`, with a debug assertion on length.

##### FR-6: BLAS Interoperability

Any `ContiguousStorage` backend **must** be directly passable to the existing
BLAS subprogram traits (`AXPY`, `DOT`, `GEMV`, `GEMM`) via its `as_slice()` /
`as_mut_slice()` methods. No intermediate copies. No adapters. The
`ContiguousStorage` sub-trait's slice views are the bridge between
type-checked matrix code and raw-slice BLAS kernels.

**Layout genericity is a hard requirement, not an implementation detail**:
a caller must be able to swap a backend's physical layout (row-major vs.
column-major) — or introduce a second `ArrayStorage`-like backend with the
opposite layout — without changing a single line of `Matrix`'s arithmetic
implementations. This is achieved by threading `ContiguousStorage::ORDER`
(FR-4) through to the BLAS call as an explicit parameter, the same role
CBLAS's `Order` argument (`CblasRowMajor`/`CblasColMajor`) plays ahead of
every `cblas_*gemm`/`*gemv` call: `Matrix`'s operator implementations read
`S::ORDER` off their own storage backend and pass it straight through, and
never branch on layout themselves. Concretely, `GEMV`/`GEMM`/`AXPY` gain an
`order: MatrixLayout` parameter (alongside the existing `rows`/`cols`
parameters, which already serve the role CBLAS splits across `lda`/`ldb`/
`ldc` for a densely-packed contiguous backend); a single implementation
branches on `order` internally rather than requiring one dispatch-wrapper
per storage type. This is the direct fix for the mismatch recorded in NFR-6.

Non-contiguous backends (strided, padded) interact with BLAS only through
element-wise access or by copying into a contiguous temporary — this is an
explicit, visible cost rather than a silent abstraction leak.

> **Citation gap**: the CBLAS `Order`-parameter convention itself is not yet
> backed by a source in `research/results/matrix.json` (that file's BLAS
> citations cover accumulator width and LAPACK/BLASFEO layout trade-offs, not
> the `Order`-as-argument mechanism specifically). Treat the mechanism above
> as architecturally decided but the citation as open — see §8.

##### FR-7: Retrofit Existing Types

The `Polynomial<T, N>` type and the `StateSpace<T, NX, NU, NY>` type **must**
be expressible in terms of the new storage trait without changing their
public API. This means the trait must be general enough that:

- `Polynomial<T, N>` can be internally backed by `ArrayStorage<T, N, U1>`.
- `StateSpace` matrices can each be backed by an independent `Storage`
  implementor.

This is a migration path, not an immediate requirement — existing code
continues to compile during the transition.

---

#### 2.2 Non-Functional Requirements

##### NFR-1: Zero-Cost Abstraction

The `Storage` trait **must** compile down to the same machine code as
hand-written array access. Specifically:

- All trait method calls **must** be monomorphized and inlined by the compiler.
- No vtable dispatch. No `dyn` trait objects. All dispatch is static.
- The resulting binary size and execution time must be equivalent (within
  measurement noise) to code written directly against `[T; N]`.

This is verified by inspecting the generated assembly for representative
operations (element access, slice creation, BLAS calls) and confirming that
the trait boundary introduces zero additional instructions.

##### NFR-2: `no_std` + `no_alloc` Compatibility

The storage trait and all provided backends **must** be fully functional
in a `#![no_std]` environment with no allocator. Specifically:

- No dependency on `alloc`, `std`, `Box`, `Vec`, `Rc`, `Arc`, or any
  heap-allocating type.
- No use of `HashMap`, `BTreeMap`, or any collection requiring dynamic memory.
- All code compiles with `#[deny(unsafe_op_in_unsafe_fn)]` and passes the
  project's existing Clippy configuration (including
  `clippy::large_stack_arrays`).

A future `alloc` feature gate may add a `HeapStorage<T, R, C>` backed by
`Vec<T>`, but this is explicitly out of scope for the initial implementation.

##### NFR-3: Const-Constructibility

Where possible, backend constructors **must** be `const fn`. This is
essential for the Flash/ROM backend use case (FR-3d): matrices baked into
read-only Flash must be fully evaluable at compile time. The following
constructors must be `const`:

- `ArrayStorage::from_array()` / `ArrayStorage::new()`
- `zero()` — fills with the additive identity.
- `identity()` — ones on the diagonal, zeros elsewhere.
- `diagonal()` — arbitrary diagonal values, zeros elsewhere.

```rust
static GRAVITY: Matrix<f32, U3, U1, ArrayStorage<f32, U3, U1>> =
    Matrix::from_array([[0.0, 0.0, -9.81]]);
```

##### NFR-4: Safety Discipline

The trait exposes two categories of access:

1. **Safe methods** — bounds-checked `get(i, j)` and `get_mut(i, j)` live on
   the `Matrix` wrapper (not the storage trait). `as_slice()` and
   `as_mut_slice()` are safe because they are gated behind the
   `ContiguousStorage` marker trait.
2. **Unsafe methods** (`get_unchecked`, `get_unchecked_mut`, `ptr`, `ptr_mut`)
   — no bounds checking. Callers must uphold the documented safety invariants.

The `# Safety` contract for each unsafe method must be documented per the
project's `doc-standards.md` (§3.2).

##### NFR-5: Minimal Trait Bounds on `T`

The `Storage` trait itself **must not** require `T: Default`, `T: Clone`,
`T: Copy`, or any arithmetic trait. Only specific _constructors_ (e.g.,
`zeros()`) add bounds as needed. This ensures the storage layer works with:

- Primitive floats (`f32`, `f64`)
- Fixed-point types (`i16`, custom Q-format wrappers)
- Complex numbers (`Complex<f32>`)
- Any user-defined `Scalar` type

##### NFR-6: Memory Layout Guarantees

`ArrayStorage<T, R, C>`'s *default* choice is column-major (Fortran-order)
layout via `[[T; R]; C]` annotated with `#[repr(C)]`. Element `(i, j)` of an
`R × C` matrix is at linear index `j * R::DIM + i`. This is the standard
convention for BLAS/LAPACK and enables column `j` to be accessed as a single
contiguous `[T; R]` slice. It is a default, not a hard-coded assumption
baked into `Matrix`: per FR-4/FR-6, `ArrayStorage` declares
`ContiguousStorage::ORDER = MatrixLayout::ColMajor`, and every downstream
consumer (BLAS subprogram calls, `Matrix`'s own arithmetic) reads that
constant rather than assuming it.

The `offset()` abstraction (FR-1) allows individual backends to implement
any layout. The default `ArrayStorage::offset()` returns
`(j * R::DIM + i) as isize` (column-major). View types can override this
to provide transposed (row-major) access without copying data.

> **Resolved (supersedes the prior note in this section)**: The existing
> `GEMV`/`GEMM` implementations in `subprograms.rs` currently use row-major
> access patterns (`chunks_exact(cols)`), which do not match
> `ArrayStorage`'s column-major default — a real, currently-live mismatch
> between the two modules. An earlier revision of this document proposed
> resolving it by rewriting the subprograms to hard-code column-major
> access. That approach was rejected: it would re-introduce exactly the kind
> of hard-coded-layout assumption this trait hierarchy exists to eliminate,
> and it would make a future row-major backend (e.g. a DMA buffer populated
> by a peripheral that only produces row-major data) impossible to support
> without a parallel set of subprogram implementations. FR-6's
> `ContiguousStorage::ORDER`-as-parameter mechanism is the adopted fix
> instead: one `GEMV`/`GEMM`/`AXPY` implementation, parameterized by
> `order`, serves every contiguous layout. This is a `subprograms.rs`
> change, deferred to implementation — not made in this revision, which is
> documentation-only.

---

#### 2.3 Constraints

##### C-1: Rust Edition 2024

The implementation targets Rust edition 2024 as specified in the workspace
`Cargo.toml`. This enables `const` generics and other recent language features
but imposes any limitations of the current stable toolchain.

##### C-2: Peano Dimension Ceiling

The current `Dim` type system defines aliases up to `U32`. The storage trait
inherits this limit: inline-backed matrices are restricted to dimensions where
`R::DIM * C::DIM ≤ 1024` (32×32). Borrowed-slice backends (`SliceMut`,
`SliceRef`) are not subject to this limit since they do not use the dimension
in a const-generic array context. This ceiling also keeps `DimMul`'s recursive
trait resolution (`S<N> * M = M + (N * M)`) well within the compiler's default
recursion limit, so a `Const<N>`-based reimplementation is unnecessary.

##### C-3: Clippy Compliance

The implementation must pass the workspace Clippy configuration. Of particular
relevance:

- `clippy::large_stack_arrays = "deny"` — inline backends for large dimensions
  must _not_ trigger this lint. The `clippy.toml` does not override the default
  threshold (512 KiB), so `Inline` is safe for all dimensions where
  `R::DIM * C::DIM * size_of::<T>() < 524_288`. In practice the Peano ceiling
  (C-2) is the binding constraint, not this lint.
- `clippy::indexing_slicing = "deny"` — all element access must use checked
  methods or documented `unsafe` blocks.
- `clippy::arithmetic_side_effects = "deny"` — index calculations must use
  checked arithmetic or be provably safe (const-evaluated).
- `too-many-arguments-threshold = 4` — trait methods must not exceed 4
  parameters. The existing `GEMV`/`GEMM` traits suppress this with
  `#[allow(clippy::too_many_arguments)]`; new trait methods should stay
  within the limit.
- `too-many-lines-threshold = 60` — implementations should be factored into
  small, testable functions.

##### C-4: No Breaking Changes to Existing Public API

The `Storage` trait is additive. Existing public types (`Polynomial`,
`StateSpace`, `ArithmeticError`, `Map`, etc.) must continue to compile and
function identically. The trait is introduced as a new module within `math`,
and existing types may optionally adopt it behind a feature gate or through
a phased migration.

---

### 3. Technical Overview

This project introduces a trait-based storage abstraction into the `math`
module of `control-rs`. It sits below the (future) `Matrix<T, R, C, S>` type
and above the raw BLAS subprograms. The primary expertise required is:

- **Rust type-level programming**: deep familiarity with the existing Peano
  `Dim` system, associated types, and trait bounds.
- **Unsafe Rust**: raw pointer manipulation for `ptr()` / `ptr_mut()` and
  `MaybeUninit`-based initialization (extending the existing
  `array_from_iterator` pattern).
- **Embedded systems memory architecture**: understanding of stack vs. static
  allocation, linker sections (`.bss`, `.data`), and DMA buffer placement.
- **BLAS conventions**: column-major layout, leading dimension semantics, and
  how the existing `GEMV`/`GEMM` traits consume slice arguments.

---

### 4. Core Architecture

The storage layer decouples _where_ matrix data lives in memory from _how_
mathematical dimensions are verified, via a trait hierarchy parameterized over
the existing `Dim` system.

```text
                    Storage<T, R, C>
                   /                \
     StorageMut<T, R, C>    ContiguousStorage<T, R, C>
                   \                /
              ContiguousStorageMut<T, R, C>
```

- **`offset()` as the single layout control point:** every backend implements
  `ptr()` / `ptr_mut()` plus `offset(i, j) -> isize`, mapping a logical
  `(row, col)` index to a pointer displacement. Row-major, column-major,
  reversed, and strided/padded layouts are all expressed by swapping this one
  method; no other code changes.
- **Mutability and contiguity as orthogonal bounds:** `StorageMut` gates write
  access; `ContiguousStorage` / `ContiguousStorageMut` gate `as_slice()` /
  `as_mut_slice()` behind a compile-time guarantee of gap-free memory. This
  keeps padded or strided backends from leaking uninitialized bytes through a
  safe slice API, and lets `Matrix<T, R, C, S>` conditionally expose
  slice-based methods based on which bounds `S` satisfies.
- **Backend catalog:** `ArrayStorage` (owned, column-major by default,
  `#[repr(C)]`), `MatrixView` / `MatrixViewMut` (borrowed, zero-copy, support
  non-destructive transposition via `offset()`), and `PivotStorage` (bounded
  scratch space for decomposition bookkeeping — not a full `Storage`
  implementor). Flash/ROM and DMA-pool backends are supported extension
  points (FR-3d) rather than shipped types.
- **Layout as data, not assumption:** `MatrixLayout` (`RowMajor` /
  `ColMajor`) is a plain two-variant enum. Every `ContiguousStorage`
  backend declares its layout via the `ORDER` associated const (FR-4);
  BLAS-facing subprogram traits accept `order: MatrixLayout` as a
  parameter (FR-6) instead of assuming one. `Matrix`'s own code reads
  `S::ORDER` and forwards it — it never hard-codes either layout, which is
  what makes swapping `ArrayStorage`'s default column-major choice for a
  row-major (or other) backend a zero-`Matrix`-code-change operation.

---

### 5. Alternatives

- **Convenience Methods vs. Explicit Decompositions:** Convenient immutable APIs
  that hide heavy $O(n^3)$ operations behind signatures like
  `invert(&self) -> Matrix`
  were explicitly rejected. Hiding factorizations masks expensive internal stack
  allocations and silent matrix copies. Instead, operations require explicit
  decomposition objects (`LuDecomposition`, `LdltDecomposition`) or
  caller-managed
  in-place mutation (`lu_decompose_mut`, `solve_mut`).
- **Const Generics vs. Type-Level Traits (`Dim`):** Native const generic arrays
  (`[[T; R]; C]`) without type-level dimension bounds were evaluated against
  trait-bound dimension wrappers. The `Dim` trait combined with decoupled
  `Storage`
  was selected to support complex compile-time dimension arithmetic (
  e.g., $M \times N$
  and $N \times P$ matrix multiplication bounds) on stable Rust edition 2024.
- **External Libraries (`nalgebra`):** Bypassing external dependencies like
  `nalgebra` (even in `no_std` mode) minimizes the safety-critical audit
  footprint under standards such as ISO 26262 / DO-178C, and guarantees
  `const fn`
  constructors required for baking static matrices directly into ROM/Flash.
  However, the custom storage trait design here is a direct, `#![no_std]`
  adaptation
  of the decoupled storage architecture from `nalgebra` (BSD-3-Clause, by
  Sébastien Crozet),
  adopting its core layout patterns, trait structures (`Storage`, `StorageMut`),
  and slice coercion interfaces.

---

### 6. Verification & Validation

Verification and validation follow four rigorous pillars tailored for
safety-critical
embedded control systems:

1. **Compile-Time Verification:** Dimension mismatches and memory bounds are
   strictly
   enforced by the Rust type system using Peano types (`Dim`), preventing
   invalid pointer
   arithmetic or dimension collisions at compile time.
2. **Property & Unit Testing:** Logic is validated on the host via `cargo test`
   and
   `proptest` to mathematically prove algebraic identities (
   e.g., $(AB)^T = B^T A^T$)
   and verify safe, panic-free degradation for ill-conditioned or singular
   matrices.
3. **Hardware-in-the-Loop (HIL):** Deferred until a hardware-specific backend
   (Flash, DMA) ships. Not applicable to the initial `ArrayStorage` / view /
   scratch backends, which are fully verified on host. Once a target-specific
   backend lands, cross-compiled code will be profiled on actual hardware for
   L1 cache misses, FPU cycle counts, and pipeline dependencies.
4. **Stack Bounds Verification:** Matrix capacities are strictly capped
   at $32 \times 32$
   elements to mathematically guarantee that a single matrix instance never
   exceeds 4KB
   of stack space, preventing stack overflow hard-faults on embedded
   controllers.

---

### 7. Performance & Resource Considerations

- **Stack usage**: the primary motivating concern. `ArrayStorage` backends
  must remain within the `clippy::large_stack_arrays` threshold. All larger
  allocations must use view storage or static buffers.
- **Compile time**: deep Peano type recursion can slow the trait solver.
  Implementations should prefer `Const<N>` + `DimMul` over deeply nested
  `S<S<S<...>>>` where possible.
- **Code size**: monomorphization of `Storage` methods for every `(T, R, C)`
  combination can bloat the binary. Critical methods should be marked
  `#[inline]` judiciously, and shared logic should be factored into
  non-generic helper functions operating on `&[T]`.

---

### 8. Risks & Open Questions

**Lifetime ergonomics for borrowed backends:** `MatrixViewMut` carries a
lifetime `'a`, which propagates into any containing type (e.g.,
`Matrix<f32, U100, U100, MatrixViewMut<'_, f32, U100, U100>>`). A type-alias
convenience layer (e.g., `type MatSliceMut<'a, T, R, C> = Matrix<T, R, C,
MatrixViewMut<'a, T, R, C>>`) is deferred until real call sites show the
verbosity is a practical problem rather than a cosmetic one.

**CBLAS `Order`-parameter citation is open:** FR-6's layout-genericity
mechanism (`ContiguousStorage::ORDER` threaded through to `GEMV`/`GEMM`/
`AXPY` as an explicit parameter) is architecturally decided but not yet
backed by a source in `research/results/matrix.json` — that file's existing
BLAS citations (Anderson et al. 1999; BLASFEO) cover accumulator width and
panel-vs-column-major layout trade-offs, not the `Order`-as-argument
convention itself. A `/cr-research` pass adding a citation for the CBLAS
`Order` parameter (or an equivalent embedded-DSP precedent, e.g. how
CMSIS-DSP's matrix functions handle layout) should run before this section
is treated as final.

---

### 9. Development Plan

| Task / Feature                              | Description                                                                                                                | Estimated Effort |
|:---------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------|:-----------------|
| Phase 1: Trait Hierarchy & Core Backend     | Define `Storage`, `StorageMut`, `ContiguousStorage`, `ContiguousStorageMut`; implement `ArrayStorage<T, R, C>`             | M                 |
| Phase 2: View & Scratch Storage             | Implement `MatrixView`, `MatrixViewMut`, and `PivotStorage<D>`                                                              | M                 |
| Phase 3: Initialization & BLAS Integration  | Add `from_fn`, `from_element`, `zeros`, `identity`, `diagonal` constructors; verify `as_slice()` interop with `GEMV`/`GEMM` | M                 |
| Phase 4: `Matrix` Wrapper & Retrofit        | Introduce `Matrix<T, R, C, S>`; migrate `Polynomial` and `StateSpace` onto storage backends                                 | L                 |

---

### 10. Revision History

| Date       | Author          | Description                                                                                  |
|:-----------|:----------------|:---------------------------------------------------------------------------------------------|
| 2026-07-26 | @MitchellDScott | Initial draft                                                                                |
| 2026-07-26 | @MitchellDScott | Expanded Core Architecture, Alternatives, and 4-pillar V&V sections to align with matrix doc |
| 2026-08-01 | @MitchellDScott | Rewrote Core Architecture to remove decomposition-scope drift and duplicate FR text; folded resolved Risks items into Requirements; collapsed Development Plan into 4 phases; caveated HIL applicability |
| 2026-08-02 | @MitchellDScott | Made this document the sole owner of `MatrixView`/`MatrixViewMut` struct definitions (`matrix-design.md` no longer redefines them); added `ContiguousStorage::ORDER` (FR-4) and reworked FR-6/NFR-6 so BLAS interop takes an explicit layout parameter instead of assuming `ArrayStorage`'s column-major default, superseding the prior "rewrite subprograms to column-major" note; added a Risks entry for the still-open CBLAS `Order`-parameter citation. |
