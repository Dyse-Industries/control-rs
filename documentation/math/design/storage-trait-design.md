# Storage Trait (Design Document)

![Date Badge](https://img.shields.io/badge/Date-August_1,_2026-blue)
![Status Badge](https://img.shields.io/badge/Doc%20Status-Reviewed-yellow)
![Author Badge](https://img.shields.io/badge/Author-@mitchelldscott-blueviolet)

---

### 1. Introduction

Each numerical model and algorithm utilizes generic dimensions (`Dim`) to enable
size checking at compile time rather than runtime. The `Storage` trait provides
the abstraction that ties `Dim` to physical memory and standardizes how higher
level code accesses data.

---

### 2. Requirements

#### 2.1 Functional Requirements

##### FR-1: Core Storage Trait

At minimum, a backend must expose: raw-pointer access to its backing allocation;
a mapping from a logical `(row, column)` index to a physical memory location,
general enough to represent row-major, column-major, reversed, and
strided/padded layouts through one interface; and unchecked (`unsafe`) element
access built on top of that mapping.

##### FR-2: Type-Level Dimension Encoding

The trait must be parameterized by compile-time row and column dimensions,
using the existing `Dim` type system, so that dimension mismatches in
matrix operations (e.g., multiplying a 3×4 by a 5×2) are rejected as
compile errors rather than surfaced as runtime panics. The compile-time
dimension representation must still expose its value at runtime without
additional bookkeeping, since the existing BLAS subprograms take dimension
parameters (`m`, `n`, `k`) at runtime.

##### FR-3: Core Storage Implementations

The module must ship backends covering the three fundamental ownership
models (owned-stack, borrowed-immutable, borrowed-mutable), plus a
scratch-data category for decomposition algorithms, and must support
user-defined hardware-specific backends as extension points.

**FR-3a: Stack-Based Array Storage**

The module must provide a default, owning backend that stores its elements
inline on the stack, guarantees a contiguous and predictable memory layout
suitable for BLAS interop, and supports both mutable and immutable access.

**FR-3b: Zero-Copy View Storage**

The module must provide two non-owning backends that borrow existing
memory without copying — one immutable, one mutable. Views must support
layout-changing operations such as transposition without allocating memory
or copying data.

##### FR-4: Trait Hierarchy

The trait hierarchy must distinguish two independent capabilities: mutable
access and guaranteed contiguity.

##### FR-5: Initialization Strategies

Every backend must support safe initialization with a constant `T`. At minimum
this includes: construction from a single repeated value, zero-filled
construction and identity construction.

##### FR-6: BLAS Interoperability

Any backend that guarantees contiguous memory must be directly usable by
the existing BLAS subprogram traits (`AXPY`, `DOT`, `GEMV`, `GEMM`) without
intermediate copies or adapter code.

**Layout genericity is a hard requirement, not an implementation detail**:
a caller must be able to swap a backend's physical layout, or introduce an
alternative backend with the opposite layout, without changing any of
`Matrix`'s arithmetic implementations. Non-contiguous backends interact
with BLAS only through element-wise access or an explicit, visible copy
into a contiguous temporary — never a silent one.

##### FR-7: Retrofit Existing Types

The existing `Polynomial<T, N>` and `StateSpace<T, NX, NU, NY>` types must
be expressible in terms of the new storage trait without changing their
public API. This is a migration path, not an immediate requirement.

---

#### 2.2 Non-Functional Requirements

##### NFR-1: Zero-Cost Abstraction

The `Storage` trait must compile down to the same machine code as
handwritten array access.

##### NFR-2: `no_std` + `no_alloc` Compatibility

The storage trait and all provided backends must be fully functional in a
`#![no_std]` environment with no allocator:

##### NFR-3: Const-Constructibility

Storage constructors should be constant when possible except `from_fn(|i,
j| -> T)`.

##### NFR-4: Safety Discipline

Access must be split into two categories: safe, bounds-checked access, and
unsafe, unchecked raw-memory access.

---

#### 2.3 Constraints

##### C-1: Peano Dimension Ceiling

The current `Dim` type system defines aliases up to `U32`. The storage trait
inherits this limit: inline-backed matrices are restricted to dimensions where
`R::DIM * C::DIM ≤ 1024` (32×32).

##### C-2: Clippy Compliance

The implementation must pass the workspace Clippy configuration. Of particular
relevance:

- `clippy::large_stack_arrays = "deny"` (rust-clippy, 2026a) — inline backends
  for large dimensions must _not_ trigger this lint. Clippy's own default
  `array_size_threshold` is 16 KiB (`16 * 1024` bytes) (rust-clippy, 2026b),
  and `clippy.toml` does not override it. Rather than shrinking C-1's
  dimension ceiling to accommodate every possible scalar width, any
  `ArrayStorage<T, R, C>` instantiation whose size reaches the
  `large_stack_arrays` threshold requires an explicit
  `#[allow(clippy::large_stack_arrays)]`.
- `clippy::indexing_slicing = "deny"` (rust-clippy, 2026c) — all element
  access must use checked methods or documented `unsafe` blocks.

---

### 3. Technical Overview

This project introduces a trait-based storage abstraction into the `math`
module of `control-rs`. It sits below the (future) `Matrix<T, R, C, S>` type
and above the raw BLAS subprograms.

---

### 4. Core Architecture

The storage layer decouples _where_ matrix data lives in memory from _how_
mathematical dimensions are verified, via a trait hierarchy parameterized over
the existing `Dim` system:

```text
                    Storage<T, R, C>
                   /                \
     StorageMut<T, R, C>    ContiguousStorage<T, R, C>
                   \                /
              ContiguousStorageMut<T, R, C>
```

**`offset()` is the single point of control for a memory layout.**

`Storage<T, R: Dim, C: Dim>` (`C = U1` for 1-D containers such as
polynomials) provides

* `ptr(&self) -> *const T`
* `offset(&self, i: usize, j: usize) -> isize`
* `get_unchecked(i, j) -> &T`

which can be used to access elements as

* `&*self.ptr().offset(self.offset(i, j))`

> Returning `isize` rather than `usize` from `offset()` lets implementations
> encode reversed or non-standard layouts (e.g. a negative stride) without
> physically rearranging the data.

`as_slice()`/`as_mut_slice()`  are not possible with non-contiguous storage
and `get()`/`get_mut()` performs bounds checks that will be done by the
higher level types.

**Mutability and contiguity as orthogonal bounds.**
`StorageMut<T, R, C>: Storage<T, R, C>` adds

* `ptr_mut(&mut self) -> *mut T`
* `get_unchecked_mut`

so read-only backends cannot be mutated.

`ContiguousStorage<T, R, C>: Storage<T, R, C>` is a marker sub-trait for
backends whose `R::DIM * C::DIM` elements are laid out with
no padding or stride gaps; it provides `as_slice(&self) -> &[T]` and
`const ORDER: MatrixLayout`.
`ContiguousStorageMut<T, R, C>: StorageMut + ContiguousStorage`
combines both, adding `as_mut_slice(&mut self) -> &mut [T]`.

**Backend catalog.**

`ArrayStorage<T, R, C>` is the default owning backend:
a column-major 2-D const-generic array `[[T; R]; C]` annotated with
`#[repr(C)]` for a contiguous, predictable layout.

`StorageView<'a, T, R, C>` and `StorageViewMut<'a, T, R, C>` are the zero-copy,
non-owning backends.

Flash/ROM and DMA-pool backends are supported extension points rather than
shipped types. A Flash backend implements `Storage` and `ContiguousStorage`
but not `StorageMut`, relying on `const fn` constructors to bake matrices
directly into ROM — mirroring `cortex-m-rt`'s `link_section` + `MaybeUninit`
static-placement pattern for routing data into a target-specific memory region
at link time (cortex-m-rt, 2026):

```rust
static GRAVITY: Matrix<f32, U3, U1, ArrayStorage<f32, U3, U1>> =
    Matrix::from_array([[0.0, 0.0, -9.81]]);
```

**Initialization.** Every backend supports construction without requiring `T:
Default`

```rust
// element-by-element using row/column indices
fn from_fn(f: impl FnMut(usize, usize) -> T) -> Self { /** */ }

// fills every element with a clone
fn from_element(val: T) -> Self where
    T: Clone
{ /** */ }

fn zeros() -> Self
where
    T: Zero
{ /** */ }

fn identity() -> Self where
    T: Zero + One
{ /** */ }
```

**BLAS interoperability and layout genericity.**  `cblas.h` defines
`enum CBLAS_ORDER {CblasRowMajor, CblasColMajor}` and threads it as the
first argument to every such call (Netlib, n.d.), alongside the
`lda`/`ldb`/`ldc` leading-dimension parameters (Dongarra et al., 1990) that
this design's `rows`/`cols` parameters play the equivalent role for.

`GEMV`/`GEMM`/`AXPY` expect an `order: MatrixLayout` parameter. A single
implementation branches on `order` internally rather than requiring one
dispatch-wrapper per storage type, and `Matrix`'s operator implementations read
`S::ORDER` off their own storage backend and pass it straight through, never
branching on layout themselves.

CMSIS-DSP's matrix functions take the opposite approach
—`arm_matrix_instance_f32` (Arm, 2022) and `arm_mat_mult_f32` (Arm, 2015) fix
a single row-major layout with no `Order`-equivalent parameter — which is the
embedded-DSP counterpoint this design deliberately rejects in favor of the CBLAS
convention.

**Layout is part of the storage type.** `MatrixLayout` (`RowMajor`/`ColMajor`)
is a plain two-variant enum.

This is a default declared via `ContiguousStorage::ORDER = 
MatrixLayout::ColMajor`, not a hard-coded assumption baked into `Matrix`: every
downstream consumer reads that constant rather than assuming it, which is what
makes swapping the default for a row-major (or other) backend a
zero-`Matrix`-code-change operation.

---

### 5. Alternatives

- **Const Generics vs. Type-Level Traits (`Dim`):** Native const generic arrays
  (`[[T; R]; C]`) without type-level dimension bounds were evaluated against
  trait-bound dimension wrappers. The `Dim` trait combined with decoupled
  `Storage` was selected to support complex compile-time dimension arithmetic (
  e.g., $M \times N$ and $N \times P$ matrix multiplication bounds) on stable
  Rust edition 2024.
- **Compile-Time vs. Runtime Dimension Checking:** This design instead rejects
  dimension mismatches at compile time through `Dim` (FR-2), trading the
  flexibility of runtime-determined shapes for the elimination of an entire
  class of error-handling code paths and runtime branches in safety-critical
  control loops.
- **Sealed vs. Open Trait Hierarchy:**
  `Storage`/`StorageMut`/`ContiguousStorage`/`ContiguousStorageMut` are
  intentionally left open so downstream users can implement custom
  backends (FR-3d), which a sealed hierarchy would prevent.

---

### 6. Verification & Validation

Verification and validation follow four rigorous pillars tailored for
safety-critical embedded control systems:

1. **Compile-Time Verification:** Dimension mismatches and memory bounds are
   strictly enforced by the Rust type system using Peano types (`Dim`).
2. **Property & Unit Testing:** Logic is validated on the host via `cargo test`
   and `proptest`.
3. **Hardware-in-the-Loop (HIL):** Measure baseline for overhead from
   implementation.
4. **Stack Bounds Verification:** Storage is strictly capped
   at $32 \times 32$ elements (see num-types).

---

### 7. Performance & Resource Considerations

- **Stack usage**: the primary motivating concern. `ArrayStorage` backends
  must remain within the `clippy::large_stack_arrays` threshold
  (rust-clippy, 2026a). All larger allocations must use view storage or
  static buffers.
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

**Stack-size headroom narrows for wide scalar types:** C-3's
`clippy::large_stack_arrays` analysis (rust-clippy, 2026a) shows a 32×32
`Complex<f64>` matrix lands exactly on Clippy's 16 KiB lint threshold
(rust-clippy, 2026b), with zero margin, despite
`Complex<f32>` being explicitly in scope per NFR-5. C-3's
explicit-`#[allow]`-with-justification requirement addresses this for
individual call sites, but the underlying fact — that C-2's fixed 32×32
ceiling has no reserved headroom once `size_of::<T>()` exceeds 8 bytes —
is not otherwise flagged anywhere a reader would find it before hitting the
lint.

---

### 9. Development Plan

| Task / Feature                             | Description                                                                                                                 | Estimated Effort |
|:-------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------|:-----------------|
| Phase 1: Trait Hierarchy & Core Backend    | Define `Storage`, `StorageMut`, `ContiguousStorage`, `ContiguousStorageMut`; implement `ArrayStorage<T, R, C>`              | M                |
| Phase 2: View & Scratch Storage            | Implement `MatrixView`, `MatrixViewMut`, and `PivotStorage<D>`                                                              | M                |
| Phase 3: Initialization & BLAS Integration | Add `from_fn`, `from_element`, `zeros`, `identity`, `diagonal` constructors; verify `as_slice()` interop with `GEMV`/`GEMM` | M                |
| Phase 4: `Matrix` Wrapper & Retrofit       | Introduce `Matrix<T, R, C, S>`; migrate `Polynomial` and `StateSpace` onto storage backends                                 | L                |

---

### 10. References

1. **rust-clippy (2026a).** _clippy_lints/src/large_stack_arrays.rs_ [Source
   code]. rust-lang/rust-clippy.
   https://raw.githubusercontent.com/rust-lang/rust-clippy/master/clippy_lints/src/large_stack_arrays.rs
   — Declares the `large_stack_arrays` lint (pedantic group, not deny-by-default
   upstream) that C-2 and the Risks section require an explicit `#[allow]` for.
2. **rust-clippy (2026b).** _clippy_config/src/conf.rs_ [Source code].
   rust-lang/rust-clippy.
   https://raw.githubusercontent.com/rust-lang/rust-clippy/master/clippy_config/src/conf.rs
   — Source of the `array_size_threshold` default (`16 * 1024` = 16 KiB) that
   C-2's Complex\<f64\> headroom analysis is based on.
3. **rust-clippy (2026c).** _clippy_lints/src/indexing_slicing.rs_ [Source
   code]. rust-lang/rust-clippy.
   https://raw.githubusercontent.com/rust-lang/rust-clippy/master/clippy_lints/src/indexing_slicing.rs
   — Declares the `indexing_slicing` lint (restriction group, allow-by-default
   upstream) behind C-2's checked-access requirement.
4. **cortex-m-rt (2026).** _cortex_m_rt_ (version 0.7.6) [Software
   documentation]. rust-embedded/cortex-m, docs.rs.
   https://docs.rs/cortex-m-rt/latest/cortex_m_rt/ — Source of the
   `link_section` + `MaybeUninit` static-placement pattern that this design's
   Flash/ROM backend extension point mirrors.
5. **Netlib (n.d.).** _cblas.h_ (reference CBLAS header) [Source code].
   https://www.netlib.org/blas/cblas.h (accessed Aug. 6, 2026) — Defines
   `enum CBLAS_ORDER {CblasRowMajor, CblasColMajor}` and threads it as the
   first argument to every `cblas_*gemm`/`cblas_*gemv` call, the precedent
   this design's layout-genericity mechanism follows.
6. **Dongarra, J. J., Du Croz, J., Duff, I. S., & Hammarling, S. (1990).** A
   Set of Level 3 Basic Linear Algebra Subprograms. _ACM Transactions on
   Mathematical Software_, 16(1), 1–17. — Origin of the `dgemm`/`LDA`/`LDB`/
   `LDC` leading-dimension convention this design's `rows`/`cols` parameters
   play the equivalent role for.
7. **Arm Limited (2022).** _Include/dsp/matrix_functions.h_ (version
   V1.10.1) [Source code]. ARM-software/CMSIS-DSP.
   https://raw.githubusercontent.com/ARM-software/CMSIS-DSP/main/Include/dsp/matrix_functions.h
   — Current `arm_matrix_instance_f32` definition; fixes a single row-major
   layout with no `Order`-equivalent parameter.
8. **Arm Limited (2015).**
   _CMSIS/DSP_Lib/Source/MatrixFunctions/arm_mat_mult_f32.c_
   (version V.1.4.5) [Source code]. ARM-software/CMSIS_4.
   https://raw.githubusercontent.com/ARM-software/CMSIS_4/master/CMSIS/DSP_Lib/Source/MatrixFunctions/arm_mat_mult_f32.c
   — Legacy `arm_mat_mult_f32` signature corroborating the fixed row-major,
   no-`Order`-parameter convention.

---

### 11. Revision History

| Date       | Author          | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
|:-----------|:----------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 2026-07-26 | @MitchellDScott | Initial draft                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| 2026-07-26 | @MitchellDScott | Expanded Core Architecture, Alternatives, and 4-pillar V&V sections to align with matrix doc                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| 2026-08-01 | @MitchellDScott | Rewrote Core Architecture to remove decomposition-scope drift and duplicate FR text; folded resolved Risks items into Requirements; collapsed Development Plan into 4 phases; caveated HIL applicability                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| 2026-08-02 | @MitchellDScott | Made this document the sole owner of `MatrixView`/`MatrixViewMut` struct definitions (`matrix-design.md` no longer redefines them); added `ContiguousStorage::ORDER` (FR-4) and reworked FR-6/NFR-6 so BLAS interop takes an explicit layout parameter instead of assuming `ArrayStorage`'s column-major default, superseding the prior "rewrite subprograms to column-major" note; added a Risks entry for the still-open CBLAS `Order`-parameter citation.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| 2026-08-02 | @MitchellDScott | Clarified in FR-7 that FR-6's layout-genericity mechanism is a property of `Storage` itself, not Matrix-specific, and applies uniformly to `Polynomial` and future `Storage`-backed models.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| 2026-08-07 | @MitchellDScott | Backfilled citations from `/cr-research` passes 3-4 (`research/results/storage-trait.json`): corrected nalgebra's license to Apache-2.0 and clarified which nalgebra version each borrowed trait name follows; reworded the nalgebra Alternatives entry to credit the hierarchy concept rather than exact struct-layout fidelity; closed the CBLAS `Order`-parameter citation gap in FR-6 and removed the corresponding Risks entry; added citations to FR-3c (`nalgebra::PermutationSequence`/`LU`) and FR-3d (`cortex-m-rt`, `embedded-dma`, `aligned`); corrected C-3's Clippy `large_stack_arrays` default from 512 KiB to the actual 16 KiB and added an explicit-`#[allow]`-with-justification requirement (mirroring NFR-4's `# Safety` convention) for scalar types that reach the threshold; added a Risks entry for the resulting narrow stack-size headroom on wide scalar types (e.g. `Complex<f64>`); added two Alternatives entries (compile-time vs. runtime dimension checking; sealed vs. open trait hierarchy). |
| 2026-08-07 | @MitchellDScott | Reorganized per the template's section intent: Requirements (§2) now state quantified, behavioral goals only — concrete trait/struct signatures, `offset()` layout formulas, initialization function signatures, BLAS parameter-threading mechanics, backend type names, and code examples moved to Core Architecture (§4). Removed the duplicate trait-hierarchy diagram from FR-4 (single copy retained in §4). Removed a stale `SliceMut`/`SliceRef` reference in C-2 (the module never defines those names; replaced with a generic reference to borrowed-view backends). Moved the `subprograms.rs` layout-mismatch discussion from an NFR-6 blockquote to a new Alternatives entry, reframed as a technical tradeoff rather than a revision narrative. Moved NFR-1's assembly-inspection verification method to §6 (Compile-Time Verification pillar).                                                                                                                                                                      |
| 2026-08-07 | @MitchellDScott | Updated citations to the new author-year + numbered References standard (matching `matrix-design.md`'s §9 convention), following `research/results/storage-trait.json`'s migration to structured bibliographic `source` objects. Added inline `(cite_author, year)` citations to C-2's Clippy lint/threshold claims, §4's `cortex-m-rt` Flash/DMA pattern, §4's CBLAS `Order`-parameter and BLAS leading-dimension (`LDA`/`LDB`/`LDC`) claims, §4's CMSIS-DSP row-major-only counterpoint, and the two Risks/Performance restatements of the Clippy threshold. Added a new §10 References section (8 entries) and renumbered Revision History to §11. No factual claims were added, removed, or reworded — only citation apparatus.                                                                                                                                                                                                                                                                                               |
