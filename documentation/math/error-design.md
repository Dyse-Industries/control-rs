# Crate-Wide Error Module (Design Document)

![Date Badge](https://img.shields.io/badge/Date-August_22,_2026-blue)
![Status Badge](https://img.shields.io/badge/Doc%20Status-Approved-green)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

---

### 1. Introduction

`control-rs::math` exposes four crate-wide error types, defined once in
`src/math/mod.rs`:

- `ArithmeticError` — Scalar `Try*` failures (`math::ops`), consumed by
  `complex_num.rs` and `assert.rs`. Unchanged by this revision.
- `ConversionError` — Representation and value-validity failures whose
  producing conversion's type signature cannot already rule them out.
- `StorageError` — Indexing, capacity, and structural-invariant failures on
  dense, packed, and sparse backends (`storage-design.md`).
- `LinAlgError` — Computational failures of factorizations, solvers, and
  spectral decompositions (`subprograms-design.md`).

`storage-design.md` and `subprograms-design.md` split the former
`storage-subprograms-design.md` and each introduce error variants. This
revision is the canonical home for those types (FR-1). A condition already
pinned by a `where` bound is a compile error, not an `Err` arm (FR-2).
LAPACK's `INFO` split is the layering rule: illegal arguments versus
failure in the course of computation (Anderson et al., 1999).

---

### 2. Requirements

#### 2.1 Functional Requirements

- **FR-1 — Single Definition for Shared Error Types**: An error type
  consumed by more than one sibling module (`storage`, `subprograms`,
  `Matrix`, `Polynomial`, `Tensor`, `StateSpace`, `TransferFunction`) is
  defined once, here. Single-consumer enums (e.g. `DivisionError` in
  `polynomial-design.md` §4.8.2) stay in their owning module.
- **FR-2 — No Statically-Decidable Failure Modes**: A variant may be
  returned only for a condition that is not already provable from the
  producer's generic bounds. If a bound already guarantees the condition,
  the API is infallible (`From`, associated-function kernel, or
  `debug_assert` at a kernel boundary) with respect to that condition.
- **FR-3 — Layered Failure Classes**: Map each runtime failure to one
  enum by class, not by producing file:
  - scalar arithmetic → `ArithmeticError`;
  - erased-length / representation conversion → `ConversionError`;
  - storage index, capacity, structural invariant → `StorageError`;
  - factorization / solver / eigensolver computation → `LinAlgError`.
    The same named condition (`DimensionMismatch`) must not appear on more
    than one crate-wide enum.

#### 2.2 Non-Functional Requirements

- **NFR-1 — Convention Compliance**: Follows the crate-wide `thiserror`-enum
  convention already established by `matrix-design.md` and
  `state-space-design.md`.
- **NFR-2 — Shipped Producers Unchanged Except Dead `LinAlgError` Arms**:
  `StorageView` / `StorageViewMut::new` (`src/math/storage.rs`;
  `ViewStorage` / `ViewStorageMut` in `storage-design.md`) remain the only
  shipped `ConversionError` producers. Their `data.len()` check stays a
  runtime `Err`: `&[T]` erases length from its type, and
  `&[T; R::USIZE * C::USIZE]` requires `generic_const_exprs`. `StorageError`
  has no shipped producers. Shipped `LinAlgError` producers return only
  `SingularMatrix`. `NonSquareMatrix` has Display coverage and no producer;
  removing it is not a change to a live failure path. As-yet-unimplemented
  `Matrix` / `Polynomial` / `Tensor` `TryFrom` conversions remain as
  specified in `../numerical-models/matrix-design.md`,
  `../numerical-models/polynomial-design.md` and
  `../numerical-models/tensor-design.md`.

---

### 3. Technical Overview

```rust
/// Representation and value-validity conversion errors, shared across
/// `Matrix`, `Polynomial`, `Tensor`, and fallible view wrapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConversionError {
    /// Buffer length or conversion capacity is incompatible with the
    /// destination shape (erased from the producing type).
    DimensionMismatch,
    /// The polynomial is not monic (leading coefficient is not ONE),
    /// preventing companion matrix construction.
    NonMonicPolynomial,
}

/// Indexing, capacity, and structural-invariant failures on storage backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageError {
    /// Logical `(r, c)` (or packed `(i, j)`) exceeds the backend's dimensions.
    OutOfBounds,
    /// `nnz` would exceed the compile-time `MAX_NNZ` of a stack sparse leaf.
    CapacityExceeded,
    /// Write to a unit-diagonal slot of `TriangularPackedStorage`.
    ImmutableUnitDiagonal,
    /// Write of a non-real value to a Hermitian diagonal slot.
    InvalidHermitianDiagonal,
    /// Write to an unallocated sparse coordinate, or a compressed buffer
    /// that violates its offset/index contract.
    InvalidStructuralInvariant,
}

/// Computational failures of factorizations, solvers, and spectral decompositions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinAlgError {
    /// Cholesky (`Potrf` / `Pptrf`) encountered a non-positive pivot.
    NotPositiveDefinite,
    /// LU (`Getrf`) or a triangular solve encountered an exact-zero pivot.
    SingularMatrix,
    /// Caller-provided `tau` / `work` / `ipiv` slice is shorter than required.
    WorkspaceTooSmall,
    /// Jacobi eigensolver (`Syev` / `Heev`) exhausted its iteration bound.
    MaxIterationsReached,
}

pub type ConversionResult<T> = Result<T, ConversionError>;
pub type StorageResult<T> = Result<T, StorageError>;
pub type LinAlgResult<T> = Result<T, LinAlgError>;
```

`LayoutMismatch` remains removed from `ConversionError` (Rev 1.2–1.4).
`StorageError` is new. `LinAlgError` drops `NonSquareMatrix` and does not
gain `DimensionMismatch`.

---

### 4. Architecture

```mermaid
flowchart TD
    F[Failure condition] --> S{Provable from generic bounds?}
    S -- yes --> C[Trait bound / marker.\nFrom or infallible kernel.\nWrong types: compile error.]
    S -- no --> K{Failure class}
    K -- scalar arithmetic --> A[ArithmeticError]
    K -- erased length or value-validity conversion --> V[ConversionError]
    K -- index / capacity / structural invariant --> ST[StorageError]
    K -- factorization / solver / eigensolver computation --> L[LinAlgError]
```

#### 4.1 Layering

ndarray keeps a dedicated shape/layout error whose `ErrorKind` distinguishes
"incompatible shape" from "overflow when computing offset, length, etc."
(ndarray, 2026). ndarray-linalg then composes that shape error with LAPACK
failure codes in a separate `LinalgError` (ndarray-linalg, 2026). LAPACK
itself splits the diagnostic argument: `INFO < 0` means an illegal argument
and no computation; `INFO > 0` means failure in the course of computation
(Anderson et al., 1999). This design uses three public enums for those
layers rather than wrapping, because `control-rs` dimensions are `Dim`
parameters rather than runtime ndarray shapes: a wrap would reintroduce a
shape variant on `LinAlgError` that FR-2 already forbids.

Subprogram kernels assume valid operand dimensions (`subprograms-design.md`
C-1) and keep `debug_assert_eq!` at the kernel boundary. High-level
containers enforce shape statically (`subprograms-design.md` NFR-2). Eigen
states the same split: many conditions on fixed-size objects "can and
should be detected at compile time" (Eigen, 2026). uom rejects illegal
dimensional conversions at compile time (`error[E0308]`) with "zero runtime
cost over using the raw storage type" (uom, 2026).

#### 4.2 `ConversionError`

**`DimensionMismatch` stays.** Producers are value- or slice-length-dependent:

- `ViewStorage` / `ViewStorageMut::new` wrap a runtime `&[T]`; length is
  not part of the type (NFR-2; `storage-design.md` §3.1, §4.6).
- `Matrix → Polynomial` (Faddeev–LeVerrier,
  `../numerical-models/matrix-design.md` §4.8.1) fails "if the scalar type
  cannot perform division, if numerical overflow occurs or if capacity is
  insufficient" — a numeric-value condition, not a `Dim` mismatch.
- Dense ↔ packed ↔ sparse conversions whose destination capacity is a
  runtime `nnz` against a typed `MAX_NNZ` still use `StorageError`
  (`CapacityExceeded`); a true shape incompatibility that survives the
  type signature (erased view length, DSP convolution against a runtime
  slice) uses `ConversionError::DimensionMismatch`.
  `polynomial-design.md` §4.5 returns that arm from `Convolution`.

**`NonMonicPolynomial` stays.** `Polynomial → Matrix` companion-form
conversion (`../numerical-models/polynomial-design.md` §4.7.1) fails when
the leading coefficient is not `T::ONE` — a property of runtime
coefficient values, invisible to `N: Dim`. nalgebra's factorization APIs
keep an equivalent value-dependent check at runtime: `Cholesky::new`
"Returns `None` if the input matrix is not definite-positive" (nalgebra,
2026), with no compile-time alternative offered.

**`LayoutMismatch` stays removed.** Rank and size of
`Matrix` / `Polynomial` / `Tensor` conversions are `TensorLayout<Size = …>`
bounds. If that bound holds, a size mismatch cannot occur (FR-2). Rank is
an associated constant of `Layout`. Both belong in the type system.
Infallible `From` conversions (e.g.
`From<Matrix<T, R, C, …>> for Tensor<T, Layout, B>` where
`Layout: TensorLayout<Size = <R as DimMul<C>>::Output>`) fail at compile
time (`error[E0277]` / `error[E0308]`).

#### 4.3 `StorageError`

Adopted from `storage-design.md` §3.3 / §4.6, with one deletion:
`DimensionMismatch` is not duplicated here (FR-3). Remaining variants are
all runtime properties of a live buffer, not of `Dim`:

| Variant                      | Producer                                                                                                                  | Why runtime                                                                                                           |
| :--------------------------- | :------------------------------------------------------------------------------------------------------------------------ | :-------------------------------------------------------------------------------------------------------------------- |
| `OutOfBounds`                | `StorageMut::set`, `PackedStorageMut::set`, `SparseStorageMut::set`, `ArrayCooStorage::push`, `ArrayCsrStorage::from_coo` | `(r, c)` is a value; checked accessors return `Result` at library boundaries (`storage-design.md` FR-2, FR-8, FR-12). |
| `CapacityExceeded`           | `ArrayCooStorage::push`; dense → sparse when `nnz > MAX_NNZ`                                                              | `MAX_NNZ` is a type parameter; live `nnz` is not (ndarray's offset/length overflow class; ndarray, 2026).             |
| `ImmutableUnitDiagonal`      | `TriangularPackedStorage::set` with `Diag::Unit`                                                                          | Unit diagonal is a construction flag; the write is a value at `(i, i)`.                                               |
| `InvalidHermitianDiagonal`   | `HermitianPackedStorage::set` on the diagonal                                                                             | $\mathrm{Im}(A_{i,i}) = 0$ is a value invariant (`storage-design.md` FR-18).                                          |
| `InvalidStructuralInvariant` | `SparseStorageMut::set` on an unallocated coordinate; malformed CSR/CSC offsets                                           | Sparsity pattern is data, not a `Dim`.                                                                                |

Checked `get` continues to return `Option<&T>` (`None` = missing or
out-of-bounds). `set` returns `Result<(), StorageError>` because a failed
write must distinguish bounds, capacity, and invariant classes.
Unchecked accessors stay `unsafe` and infallible
(`storage-design.md` C-3).

Typed layout conversions whose shapes are in the type
(`SymmetricPackedStorage<T, N, L>` → `ArrayStorage<T, N, N>`) are `From`,
not `TryFrom` (FR-2). `ToDenseStorage` / `FromDenseStorage` return
`StorageError` only for capacity and structural failures, not for a
`Dim` mismatch. `storage-design.md` §3.3 / §4.6 match this enum: no
`StorageError::DimensionMismatch` arm.

#### 4.4 `LinAlgError`

Computational class only (`INFO > 0`; Anderson et al., 1999):

- **`NotPositiveDefinite`**: `Potrf` / `Pptrf` verify $L_{k,k} > 0$ before
  the square-root step (`subprograms-design.md` §4.3). Distinct from
  `SingularMatrix` so a Kalman / MPC loop can retry a different
  factorization. nalgebra collapses this to `Option` (nalgebra, 2026);
  §5 rejects that collapse.
- **`SingularMatrix`**: `Getrf` exact-zero pivot; shipped LU / LDLT / QR
  substitution screens in `src/matrix/decomposition.rs` and
  `src/matrix/specialized.rs`. Stays.
- **`WorkspaceTooSmall`**: `Geqrf` / `Ormqr` / `Unmqr` / `Syev` / `Heev`
  take `&mut [T]` (and `ipiv: &mut [usize]`). Slice length is erased, so
  the check is `INFO < 0`-shaped but cannot be a `Dim` bound unless the
  signatures switch to `[T; N]` (`subprograms-design.md` §4.3, §8).
- **`MaxIterationsReached`**: Jacobi `Syev` / `Heev` on the stack
  (`subprograms-design.md` §4.3). Iteration count is data-dependent.

**Not on `LinAlgError`:**

- **`DimensionMismatch`** — not on `LinAlgError`.
  `subprograms-design.md` C-1 (kernels assume valid dimensions) and NFR-2
  (containers enforce shape statically) make illegal operand shape a
  compile error or a `debug_assert` at the kernel boundary, not a solver
  `Err` (Eigen, 2026; Anderson et al., 1999). DSP convolution against a
  runtime slice is `ConversionError::DimensionMismatch`
  (`polynomial-design.md` §4.5).
- **`NonSquareMatrix`** — shipped, never produced. Square factorizations
  are `Matrix<T, D, D>` / `Const<D>: Dim`. ndarray-linalg's `NotSquare`
  exists because ndarray shapes are runtime (ndarray-linalg, 2026);
  that rationale does not apply here (FR-2).

Shipped `CholeskyDecomposition` / `LdltDecomposition` currently map a
non-positive pivot to `SingularMatrix`. `Potrf` uses
`NotPositiveDefinite`. Whether Matrix wrappers switch when they delegate
to `Potrf` is a `matrix-design.md` change, not this module's.

---

### 5. Alternatives

- **Do nothing (status quo)**: Keep `LayoutMismatch` as a blanket runtime
  check covering both rank and size; leave `StorageError` unspecified in
  this module; keep shipped `LinAlgError::{NonSquareMatrix, SingularMatrix}`.
  Rejected: `storage-design.md` and `subprograms-design.md` now name
  additional shared failure modes (FR-1); `LayoutMismatch` still violates
  FR-2 for rank/size.
- **Keep `DimensionMismatch` on all three enums**. Rejected:
  FR-3; callers cannot match one condition; LAPACK and ndarray-linalg
  already separate shape from computation (Anderson et al., 1999;
  ndarray-linalg, 2026). Sibling UMLs omit the arm
  (`storage-design.md` §3.3, `subprograms-design.md` §3.3).
- **Fold `StorageError` into `ConversionError`**. Rejected: capacity and
  Hermitian-diagonal writes are not conversions. ndarray keeps overflow
  and incompatible-shape as distinct `ErrorKind`s on a layout type, and
  still does not fold those into LAPACK computational codes (ndarray,
  2026; ndarray-linalg, 2026).
- **Wrap `StorageError` / `ConversionError` inside `LinAlgError`**,
  following ndarray-linalg's `Shape` / `Lapack` composition
  (ndarray-linalg, 2026). Rejected: wrapping reintroduces a shape arm on
  the solver type; kernel preconditions are compile-time (`subprograms-design.md`
  C-1).
- **Collapse `LinAlgError` (and `ConversionError`) to `Option`**,
  following nalgebra's `Cholesky::new -> Option<Self>` (nalgebra, 2026).
  Rejected: `storage_tests.rs` and downstream callers already branch on
  _which_ condition failed; ndarray, ndarray-linalg and LAPACK all keep a
  structured, multi-variant error — ndarray's `ErrorKind` distinguishes
  "incompatible shape" from "overflow when computing offset, length, etc."
  (ndarray, 2026), ndarray-linalg's `LinalgError` composes a shape variant
  with a wrapped LAPACK code (ndarray-linalg, 2026), and LAPACK's `INFO`
  convention separates illegal arguments from computational failure by
  sign (Anderson et al., 1999). Distinguishing `NotPositiveDefinite` from
  `SingularMatrix` is the same requirement at the solver layer.
- **Per-strategy modules** (saturating/wrapping/strict), following the
  `fixed` crate's `Saturating`/`Wrapping`/`Strict` split (fixed, 2026).
  Rejected: that pattern trades a single fallible operation for several
  infallible ones under different numeric policies — applicable to
  `ArithmeticError`'s overflow domain, not to conversion, storage, or
  factorization domains.
- **Approximate instead of error**, following micromath's infallible,
  precision-traded approximations (micromath, 2023). Not applicable:
  dimension, monic-ness, structural invariants, and singularity are
  correctness properties with no valid approximate answer.
- **Fold `ConversionError` into `ArithmeticError` (considered, deferred;
  carried over from rev 1.1)**: `ArithmeticError`'s existing variants
  (`DivisionByZero`, `Overflow`, `DomainViolation`, `PrecisionLoss`,
  `Underflow`, `Saturation`) are scalar-arithmetic-shaped, not
  layout/value/storage-shaped.
- **Keep `NonSquareMatrix`** because ndarray-linalg exposes `NotSquare`
  (ndarray-linalg, 2026). Rejected: that variant serves dynamically sized
  arrays; `control-rs` square solvers are typed `D × D` (FR-2).

---

### 6. Verification & Validation

1. `StorageView` / `StorageViewMut::new`'s `DimensionMismatch` path keeps
   its existing success/failure unit test pairs and proptest coverage
   (`src/math/tests/storage_tests.rs`) — unaffected (NFR-2). After the
   `ViewStorage` rename, the same pairs move with the constructors.
2. When `Matrix → Polynomial` and `Polynomial → Matrix` land, each
   `ConversionError` variant they produce (`DimensionMismatch`,
   `NonMonicPolynomial`) needs a dedicated failure-path unit test, matching
   the existing pattern in `src/math/mod.rs`'s `Display` tests and
   `storage_tests.rs`. Convolution length failure tests
   `ConversionError::DimensionMismatch`, not `LinAlgError`.
3. When the `From` + `TensorLayout<Size = …>` conversions (§4.2) land, add
   a `compile_fail` doctest demonstrating that a `Layout` whose `Size`
   does not match the source shape fails to compile rather than returning
   `Err`, matching the `compile_fail` doctest pattern already used in
   `src/math/num_types` for subtraction underflow.
4. Add `Display` / `Error` unit tests for every `StorageError` variant when
   the enum lands. When storage Phases 2–4 land (`storage-design.md` §8),
   each producer listed in §4.3 needs a dedicated failure-path test:
   `OutOfBounds` (`set` / `push` / `from_coo`), `CapacityExceeded` (`push`
   and dense → sparse), `ImmutableUnitDiagonal`,
   `InvalidHermitianDiagonal`, `InvalidStructuralInvariant`.
5. When LAPACK Phase 4 lands (`subprograms-design.md` §9):
   `Potrf` / `Pptrf` return `Err(LinAlgError::NotPositiveDefinite)` on a
   non-SPD / non-HPD matrix; `Getrf` returns `Err(LinAlgError::SingularMatrix)`
   on a singular matrix; a short `work` / `tau` / `ipiv` slice returns
   `WorkspaceTooSmall`; a Jacobi budget of zero (or a non-converging
   fixture if one is specified later) returns `MaxIterationsReached`.
   Remove the `NonSquareMatrix` Display test with the variant.
6. `compile_fail` (or type-level) coverage that a BLAS `Gemv` /
   `Gemm` impl does not return `Result` for shape: mismatched `Dim`
   parameters fail to compile. Kernel-boundary mismatches remain
   `debug_assert` only (debug builds).

No HIL of the error enums themselves; HIL of kernels is
`subprograms-design.md` §6.1.5.

---

### 7. Performance & Resource Considerations

Removing `LayoutMismatch` drops a runtime `RANK` branch from every `Tensor`
conversion. Omitting `DimensionMismatch` from `StorageError` and
`LinAlgError` keeps BLAS inner loops and `set` match arms free of a
dead shape class — zero runtime cost on `ArrayStorage` kernels, consistent
with `subprograms-design.md` NFR-3. `StorageError` and the extra
`LinAlgError` variants are `Copy` enums; they add no allocation.

---

### 8. Risks & Open Questions

- **Downstream Tensor conversions (closed, Rev 1.3)**:
  `../numerical-models/matrix-design.md` §4.8.2,
  `../numerical-models/polynomial-design.md` §4.7.2 and
  `../numerical-models/tensor-design.md` §4.11 specify infallible `From`
  bounded by `TensorLayout<Size = …>`. Rank-marker traits (`Rank1Layout` /
  `Rank2Layout`) are not part of that surface; `Size` is the bound.
- **Sibling-doc drift (closed, this revision)**:
  `storage-design.md` §3.3 / §4.6 omits `StorageError::DimensionMismatch`.
  `subprograms-design.md` §3.3 omits `LinAlgError::DimensionMismatch`.
  `polynomial-design.md` §4.5 returns `ConversionError::DimensionMismatch`
  from `Convolution`. FR-3 holds at those three sites.
- **`faer-rs` unresearched (open, low priority)**:
  `documentation/math/research/error.json` query 9 could not resolve
  faer-rs's dimension-mismatch convention from its crate-level docs. Not
  pursued further — Eigen and the ndarray/LAPACK family already establish
  the relevant precedent for both branches of §4 (statically-decidable vs.
  value-dependent).
- **Matrix Cholesky mapping (open)**: Shipped
  `CholeskyDecomposition` / `LdltDecomposition` report a non-positive
  pivot as `SingularMatrix`. `Potrf` reports `NotPositiveDefinite`.
  Whether `matrix-design.md` wrappers switch when they call `Potrf` is
  deferred to that document.
- **Workspace signatures (assumption)**: `subprograms-design.md` keeps
  `tau` / `work` / `ipiv` as slices, so `WorkspaceTooSmall` stays. If those
  arguments become `[T; N]` / `[usize; N]`, the variant becomes dead under
  FR-2 and is removed.
- **Assumption**: No `StorageError` producers exist in shipped code
  (confirmed by repository search). Adding the enum is not a breaking
  change. Removing `LinAlgError::NonSquareMatrix` is a public-enum break
  with no live producer (NFR-2).

---

### 9. Development Plan

| Task / Feature                     | Description                                                                                                                                              | Estimated Effort           |
| :--------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------- |
| Step 1: Narrow `ConversionError`   | Remove `LayoutMismatch` from the enum (§3); update `Display`/`Error` impls and their tests in `src/math/mod.rs`.                                         | Complete                   |
| Step 2: Rank-marker traits         | Superseded. Rank and size are `TensorLayout<Size = …>` bounds (§4.2), not a `Rank1Layout` / `Rank2Layout` family.                                        | —                          |
| Step 3: Align dependent model docs | `matrix-design.md` §4.8.2, `polynomial-design.md` §4.7.2, `tensor-design.md` §4.11 use the `From` + `Size` shape.                                        | Complete                   |
| Step 4: Add `StorageError`         | Land the enum, `StorageResult`, `Display`/`Error` impls, and Display tests in `src/math/mod.rs` (§3, §4.3). View constructors stay on `ConversionError`. | 2                          |
| Step 5: Align `LinAlgError`        | Add `NotPositiveDefinite`, `WorkspaceTooSmall`, `MaxIterationsReached`; remove `NonSquareMatrix`; keep `SingularMatrix`. Update Display tests.           | 2                          |
| Step 6: Producer tests             | Storage Phases 2–4 and subprograms Phase 4 attach the failure-path tests in §6 items 4–5.                                                                | — (owned by those designs) |

---

### 10. Revision History

| Revision | Date            | Author          | Description                                                                                                                                                                                                                                                                                                                                                                                                                               |
| :------- | :-------------- | :-------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1.0      | August 2, 2026  | @MitchellDScott | Initial stub relocating `ConversionError` out of `matrix-design.md` per review feedback; minimal pending research.                                                                                                                                                                                                                                                                                                                        |
| 1.1      | August 9, 2026  | @MitchellDScott | Review and corrections.                                                                                                                                                                                                                                                                                                                                                                                                                   |
| 1.2      | August 18, 2026 | @MitchellDScott | Corrected §1's error-type count (three, not two: added `LinAlgError`). Added FR-2/NFR-2, §4 Architecture, §6 Verification & Validation, §7 Performance and §8 Risks & Open Questions. Removed `LayoutMismatch`, moving its rank/size checks into `TensorLayout` trait bounds (§4); added comparative alternatives (§5) grounded in `research/error.json`. Reverted Doc Status to Draft pending re-review of the `LayoutMismatch` removal. |
| 1.3      | August 18, 2026 | @MitchellDScott | Reconciled Doc Status badge to Reviewed-yellow; aligned downstream cross-model conversions in `matrix-design.md`, `polynomial-design.md`, and `tensor-design.md` to infallible compile-time `From` conversions via `TensorLayout<Size = ...>`, eliminating `LayoutMismatch` across design specifications and `src/math/mod.rs`.                                                                                                           |
| 1.4      | August 18, 2026 | @MitchellDScott | Closed stale §8/§9 items against Rev 1.3: downstream `From` + `Size` alignment is complete; rank-marker traits are superseded; V&V item 3 tests the `Size` bound, not `RankNLayout`.                                                                                                                                                                                                                                                      |
| 1.5      | August 22, 2026 | @MitchellDScott | Canonicalized `StorageError` from `storage-design.md` (without duplicating `DimensionMismatch`) and realigned `LinAlgError` with `subprograms-design.md` computational failures (`NotPositiveDefinite`, `WorkspaceTooSmall`, `MaxIterationsReached`); dropped dead `NonSquareMatrix`; FR-3 forbids cloning `DimensionMismatch` across enums.                                                                                              |
| 1.6      | August 22, 2026 | @MitchellDScott | Closed sibling-doc drift: storage, subprograms, and polynomial Convolution match FR-3. `NonSquareMatrix` remains absent from `LinAlgError`.                                                                                                                                                                                                                                                                                               |

---

## References

[1] E. Anderson, Z. Bai, C. Bischof, S. Blackford, J. Dongarra, J. Du Croz,
A. Greenbaum, S. Hammarling, A. McKenney, and D. Sorensen, _LAPACK
Users' Guide_, 3rd ed. Philadelphia, PA, USA: SIAM, 1999.

[2] ndarray, _ndarray_ (Version 0.17.2). [Online]. Available:
https://docs.rs/ndarray/latest/ndarray/. Accessed: Aug. 18, 2026.

[3] ndarray-linalg, _ndarray-linalg_ (Version 0.18.1). [Online]. Available:
https://docs.rs/ndarray-linalg/latest/ndarray_linalg/error/enum.LinalgError.html.
Accessed: Aug. 18, 2026.

[4] Eigen, "Assertions," _Eigen documentation (nightly)_. [Online].
Available: https://libeigen.gitlab.io/eigen/docs-nightly/TopicAssertions.html.
Accessed: Aug. 18, 2026.

[5] uom, _uom_ (Version 0.38.0). [Online]. Available:
https://docs.rs/uom/latest/uom/. Accessed: Aug. 18, 2026.

[6] nalgebra, _nalgebra_ (Version 0.35.0). [Online]. Available:
https://docs.rs/nalgebra/latest/nalgebra/. Accessed: Aug. 18, 2026.

[7] fixed, _fixed_ (Version 1.31.0). [Online]. Available:
https://docs.rs/fixed. Accessed: Aug. 18, 2026.

[8] micromath, _micromath_ (Version 2.1.0). [Online]. Available:
https://docs.rs/micromath/latest/micromath/. Accessed: Aug. 18, 2026.
