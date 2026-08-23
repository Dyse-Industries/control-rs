# Polynomial Type (Design Document)

![Date Badge](https://img.shields.io/badge/Date-August_20,_2026-blue)
![Status Badge](https://img.shields.io/badge/Doc%20Status-Draft-orange)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

---

### 1. Introduction

This module provides single-variable polynomial representation for FIR/IIR
filter coefficients, Tustin/ZOH discretization of continuous models,
cubic/quintic trajectory generation and companion-matrix root-finding for
characteristic polynomials produced by `Matrix`. Its architecture directly
reuses the storage hierarchy `matrix-design.md` §4.1 builds on
(`storage-subprograms-design.md`, Reviewed, Rev 1.31) rather than defining a
parallel storage abstraction, fixing `C = U1` to specialize it to a single
column.

The following elements are directly shared with, or adapted from, `Matrix`'s
architecture:

- **Signature Pattern**: `Polynomial<T, N: Dim, S: MatrixStorage<T, N, U1>>`
  mirrors `Matrix<T, R, C, S: MatrixStorage<T, R, C>>` with `C` fixed to `U1`
  (`matrix-design.md` §4.1). Owning stack storage is
  `DenseVectorArray<T, const N>` (`Const<N>` in the `Dim` slot).
- **Storage Trait Reuse**: No new storage trait is introduced; `Polynomial`
  is `Matrix`'s $N \times 1$ column case, inheriting
  `DenseVectorArray<T, const N>`,
  `DenseVectorRef<'a, T, N>` / `DenseVectorRefMut<'a, T, N>`,
  and the `MatrixStorage<T, N, U1>` universal floor as-is.
- **Single Addressing Branch**: A single column reaches only the
  leading-dimension branch, where `DenseStorage`'s associated
  `type LDA = Const<N>` (`storage-subprograms-design.md` §4.1.2). The packed
  branch (`PackedStorage`, `IMPLICIT`; `Diagonal` now, `SP`/`TP` later) has
  no $N \times 1$ analogue, so
  `matrix-design.md` §4.9.2's second lookup accessor has no counterpart here.
- **Level-1 Operand Contract**: Coefficients reach kernels as
  `as_array::<N>() -> &[T; N]` with `INC_X = 1` and `BUF_X = N`
  (`storage-subprograms-design.md` §4.2.2). No level-2/3 nested operand
  arises from a single column.
- **Ownership Aliases**: `ArrayPolynomial`/`PolynomialView`/
  `PolynomialViewMut` mirror `matrix-design.md`'s `Owned`/`MatrixSlice`/
  `MatrixSliceMut` aliases (§4.1, §4.3.1).

---

### 2. Requirements

#### 2.1. Functional Requirements

- **FR-1 — Degree-Indexed Storage**: Polynomials store coefficients in
  ascending degree order ($P(s) = a_0 + a_1 s + \dots + a_N s^N$), where
  index $i$ corresponds to degree $i$.
- **FR-2 — Exact Horner Evaluation Complexity**: Evaluating $P(s)$ at a scalar
  point $s$ executes in exactly $N$ multiply-accumulate (FMA) operations for a
  degree-$N$ polynomial.
- **FR-3 — Mathematical Convolution Multiplication**: Polynomial multiplication
  of degree-$N$ and degree-$M$ polynomials yields a degree-$(N+M)$ polynomial
  whose coefficients equal the linear discrete convolution of the inputs.
- **FR-4 — Fallible Root-Finding and Division**: Root-finding algorithms and
  division operations over zero polynomials return explicit error variants (
  `PolynomialError`) rather than panicking or producing `NaN`.
- **FR-5 — Fixed-Extent Kernel Operands**: Every subprogram call site derives
  its extents ($N$, `INC_X`, `BUF_X`) from the coefficient storage type, not
  from a caller-supplied scalar, discharging the caller-side half of
  `storage-subprograms-design.md` C-4 for the $N \times 1$ case.

#### 2.2. Non-Functional Requirements

- **NFR-1 — Data-Independent Evaluation Flow**: Polynomial evaluation cost
  scales linearly with degree $N$ independent of coefficient numerical values.
- **NFR-2 — Deterministic Fixed-Memory Operations**: Polynomial arithmetic and
  evaluation execute in $O(1)$ stack memory with zero dynamic allocations.

#### 2.3. Constraints

- **C-1 — `#![no_std]` Environment**: Operates without the Rust standard
  library.
- **C-2 — Zero Dynamic Allocation**: All storage is stack-allocated or
  statically borrowed.
- **C-3 — Coefficient Ordering**: Coefficients are stored in ascending order of
  powers, a logical indexing rule independent of storage physical layout.
- **C-4 — Capacity Bound**: Maximum polynomial capacity (coefficient count $N$)
  is bounded by `Const<N>: Dim` in `num-types-design.md` C-1 ($N \le 1024$).
  Max representable degree is 1023. Stack footprint, not the trait solver, limits
  larger degrees.

---

### 3. Technical Overview

`Polynomial` is a type-safe, degree-aware wrapper that reuses `Matrix`'s
storage abstraction as its $N \times 1$ column case. Beyond
compile-time capacity safety, the API is designed to make coefficient-domain
operations — evaluation, arithmetic, division and companion-matrix
root-finding — convenient and allocation-free for callers across the FIR/IIR,
discretization and trajectory-generation use cases named in §1. Because the
storage hierarchy is specified to monomorphize without vtables and to compile
to zero-branch, zero-panic-path code over compile-time array extents
(`storage-subprograms-design.md` NFR-4, §7.1), reusing it here carries no
additional runtime cost beyond what `Matrix` already pays.

---

### 4. Core Architecture

The `Polynomial` struct is implemented in `src/polynomial/mod.rs`, replacing
the module's current pre-`Storage` stub (a bare `Polynomial<T>` trait over
`&[T]`, with no `Storage` integration) with the design below.

#### 4.1. Generics Foundation & Sizing

The core `Polynomial` structure decouples mathematical dimensions from
physical storage using the same Tier-1 bound `Matrix` names on its struct
(`matrix-design.md` §4.1.1), with `C` fixed to `U1`:

```rust
pub struct Polynomial<T, N: Dim, S: MatrixStorage<T, N, U1>> {
    storage: S,
    _marker: core::marker::PhantomData<N>,
}
```

Operations needing a leading dimension (level-2 kernel calls on the
companion-matrix path, §4.7.1) require `S: DenseStorage<T, N, U1>` on their
own `impl` block, as `Matrix` does. For a single column the two bounds admit
the same leaves, so the split costs nothing here; it keeps the two documents'
storage story identical.

Here, `N` represents the capacity (number of coefficients, maximum possible
degree is $N - 1$) and `S` defines where the coefficients reside (e.g. stack
`DenseVectorArray`, borrowed `DenseVectorRef` view or static Flash memory).
`DenseVectorArray` takes a bare `const usize`, so it is not a valid default
for `N: Dim`.

#### 4.2. Coefficient Layout & Storage Strategy

By parameterizing `Polynomial` over `MatrixStorage<T, N, U1>`, `control-rs`
supports multiple ownership models without duplicating algebraic logic:

```rust
/// Owning polynomial backed by a stack array — `N` is the alias's own const generic.
pub type ArrayPolynomial<T, const N: usize> =
    Polynomial<T, Const<N>, DenseVectorArray<T, N>>;

/// Zero-copy read-only borrowed polynomial view.
pub type PolynomialView<'a, T, const N: usize> =
    Polynomial<T, Const<N>, DenseVectorRef<'a, T, N>>;

/// Zero-copy mutable borrowed polynomial view.
pub type PolynomialViewMut<'a, T, const N: usize> =
    Polynomial<T, Const<N>, DenseVectorRefMut<'a, T, N>>;
```

Coefficients are stored in **ascending order of powers**:
$$ p(x) = c_0 + c_1 x + c_2 x^2 + \dots + c_{N-1} x^{N-1} $$
where index `i` maps to the coefficient of $x^i$. This is a logical
convention (`index = degree of term`) that `Polynomial` itself defines and
fully resolves; it says nothing about physical memory layout, which remains
`Storage`'s concern.

- **Ascending Power Storage Rationale**:
    - Direct index-to-exponent mapping: element at index `i` corresponds
      directly to $x^i$.
    - Zero-cost padding: Adding polynomials of differing capacities aligns
      coefficients naturally without element shifting.
    - **Ecosystem Consistency**: Of the Rust polynomial-adjacent crates with
      directly evidenced coefficient-ordering documentation this pass —
      `polynomial` (lib.rs, 2023) and `aberth` (docs.rs, 2026) — both use
      ascending-degree storage. This is a small, directly-verified sample,
      not a claim about the wider crate ecosystem; it supports "idiomatic,
      not crate-specific" without overstating coverage.

**Physical layout is storage's concern, not `Polynomial`'s.** Cold paths
address coefficients through a bounds-checked accessor and hot paths through
`as_array::<N>()` (§4.3), never through a raw offset, so `Polynomial` is
agnostic to whichever concrete backend it is instantiated over. Row-major and
column-major degenerate to the same addressing scheme for a $C = U1$
container, so the layout distinction `matrix-design.md` §4.2 draws between
them (cache locality across columns, BLAS interoperability) does not apply
here in the same way. The storage hierarchy still permits mixing and
matching backends under `Polynomial` to the extent a single-column shape
allows.

#### 4.3. Memory Representation & Slicing

CONTIGUOUS slice interfaces are safely exposed when the storage backend
implements `MatrixStorage` or `MatrixStorageMut`, gated the same way
`matrix-design.md` §4.3 gates `Matrix::as_slice`/`as_mut_slice`:

```rust
impl<T, N: Dim, S> Polynomial<T, N, S>
where
    S: MatrixStorage<T, N, U1>,
{
    /// Exposes a safe contiguous slice view of coefficient memory.
    pub fn as_slice(&self) -> &[T] {
        self.storage.as_slice()
    }
}

impl<T, N: Dim, S> Polynomial<T, N, S>
where
    S: MatrixStorageMut<T, N, U1>,
{
    /// Exposes a safe mutable contiguous slice view of coefficient memory.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.storage.as_mut_slice()
    }
}

impl<T, const N: usize> ArrayPolynomial<T, N> {
    /// Fixed-extent level-1 operand (`storage-subprograms-design.md` §4.2.2).
    /// The storage accessor is generic in the extent; on this alias `N` is
    /// already concrete, so `Polynomial`'s wrapper takes no turbofish.
    /// `N = N * 1` holds by `as_array`'s own `DimMul` bound, so no
    /// `generic_const_exprs` gate applies.
    pub fn as_array(&self) -> &[T; N] {
        self.storage.as_array::<N>()
    }
}
```

Level-1 kernels and `Convolution` take the fixed-extent form,
`as_array::<N>() -> &[T; N]`, rather than the unsized slice: the array's
length is in the signature, so LLVM folds the loop bounds checks
(`storage-subprograms-design.md` §4.2.2, §7.1). `as_slice()` remains the
inspection and FFI hand-off path. Both are contiguous, which is what lets
`mul_with_conv` (§4.5) and `evaluate` (§4.6) vectorize over coefficients and
pass into hardware DSP kernels without copying (NFR-1).

Unlike `Matrix` (`matrix-design.md` §4.3.1), `Polynomial` does not define a
zero-copy transposed view: transposition is not a meaningful operation on
an $N \times 1$ shape, so that subsection of `Matrix`'s architecture has no
analogue here.

Borrowed views are constructed only through `storage-subprograms-design.md`
FR-6: `ArrayPolynomial::view()` / `view_mut()` copy `N` from the owning
alias's const generic. There is no `from_slice(&[T])` constructor that
pairs an independent `N: Dim` with a raw slice.

#### 4.4. Instantiation & Constructors

- `pub const fn constant(val: T) -> ArrayPolynomial<T, 1> where T: Copy`:
  Constructs a degree-0 polynomial containing a single coefficient.
- `pub const fn line(c0: T, c1: T) -> ArrayPolynomial<T, 2> where T: Copy`:
  Constructs a degree-1 linear polynomial $c_0 + c_1 x`.
- `pub const fn from_coefficients<const N: usize>(data: [T; N]) -> ArrayPolynomial<T, N>`:
  Constructs an owning stack polynomial from an array of coefficients.
- `pub const fn from_storage(storage: S) -> Self`: Constructs a polynomial
  wrapping a custom storage backend `S`.
- `pub fn from_fn<const N: usize, F>(f: F) -> ArrayPolynomial<T, N> where F: FnMut(usize) -> T`:
  Generates coefficients via a mapping closure, mirroring `Matrix::from_fn`
  (`matrix-design.md` §4.4).

_Implementation Note_: All `const fn` constructors allow placing static
polynomials directly in read-only Flash memory, matching `Matrix`'s
constructor rationale (`matrix-design.md` §4.4).

#### 4.5. Operator Overloading & Multiplication

Overloads standard traits (`Add`, `Sub`, `Neg`), coefficient-wise with
zero-padding to the larger operand's capacity. Polynomial multiplication
provides two interfaces:

1. `mul_poly`: Static multiplication returning a combined capacity bound:
   ```rust
   impl<T, N: Dim, S1: MatrixStorage<T, N, U1>> Polynomial<T, N, S1> {
       pub fn mul_poly<M: Dim, S2: MatrixStorage<T, M, U1>, SOut>(
           &self,
           other: &Polynomial<T, M, S2>,
       ) -> Polynomial<T, <<N as DimAdd<M>>::Output as DimSub<U1>>::Output, SOut>
       where
           N: DimAdd<M>,
           <N as DimAdd<M>>::Output: DimSub<U1>,
           <N as DimAdd<M>>::Output: Dim,
           SOut: DenseStorage<T, <<N as DimAdd<M>>::Output as DimSub<U1>>::Output, U1>,
           T: Copy + Zero + Add<Output = T> + Mul<Output = T>,
       { /* ... */ }
   }
   ```
   The returned capacity `N + M - 1` is an exact bound, not an
   approximation: ascending-power zero-padding (§4.2) guarantees any
   coefficient above the true product degree evaluates to `T::zero()`
   rather than a silently wrong value. The return type does not name
   `DenseVectorArray<T, Out>`: that alias takes a bare `const usize`, not a
   `Dim` associated type (`storage-subprograms-design.md` §4.1.3). The
   `ArrayPolynomial<T, const N>` convenience impl builds the owning result
   through `from_fn` at the monomorphized `Const<N>`/`Const<M>` call site.
2. `mul_with_conv`: Decouples arithmetic from representation by leveraging
   the `Convolution<T: Float>` trait
   ([`src/math/dsp.rs`](../../src/math/dsp.rs)) and underlying
   hardware-optimized DSP kernels. `Convolution<T>` is shipped code
   bound on `T: Float`. The default implementation (`convolve_input`)
   verifies length bounds statically or returns
   `Err(ConversionError::DimensionMismatch)`
   (`error-design.md` FR-3),
   guaranteeing a panic-free execution path under release optimization.

#### 4.6. Core Operations

- **Horner's Method Evaluation**: Evaluates $p(x)$ using the recurrence
  $p(x) = c_0 + x(c_1 + x(c_2 + \dots))$ over the fixed-extent operand
  `as_array::<N>()`, descending from index $N-1$. The array extent is in the
  signature, so the loop carries no bounds check and no panic path
  (`storage-subprograms-design.md` NFR-4). Minimizes floating-point rounding
  error and operation count to $N-1$ additions and multiplications
  (Horner, 1819).
  The computed result is exact for a polynomial whose coefficients are
  relatively perturbed by at most $\gamma_{2n} = 2nu / (1 - 2nu)$ from
  $p$'s true coefficients, where $u$ is unit roundoff (Higham, 2002, Ch.
    5) — a small, degree-linear backward-error bound quantifying the
       "minimizes rounding error" claim above.
- **Polynomial Division (`div_rem`)**: Computes quotient and remainder with
  statically checked degree bounds (`Q = N - M + 1`, `R = M - 1`):
  ```rust
  impl<T, N: Dim, S: MatrixStorage<T, N, U1>> Polynomial<T, N, S> {
      pub fn div_rem<M: Dim, Sm: MatrixStorage<T, M, U1>, Q: Dim, R: Dim>(
          &self,
          divisor: &Polynomial<T, M, Sm>,
      ) -> Result<(Polynomial<T, Q>, Polynomial<T, R>), DivisionError> { /* ... */ }
  }
  ```
  `DivisionError` covers the hard case (an exactly-zero leading divisor
  coefficient or a degree mismatch), not the soft case: `div_rem`'s
  repeated subtract-and-rescale steps degrade continuously in accuracy as
  the divisor's leading coefficient shrinks relative to its other
  coefficients.
- **Calculus Operations**: Analytical derivative and integral methods
  returning statically resized polynomial bounds. Zero-location properties
  and stability bounds underpin root-finding correctness (Henrici, 1974).

#### 4.7. Interoperability & Conversions

##### 4.7.1. Conversion to Matrix (Companion Form)

A monic polynomial of degree $n = N - 1$ converts to its $n \times n$
companion matrix in Controllable Canonical Form, enabling $O(N^2)$-time,
$O(N)$-space companion-matrix QR rootfinding.

- **Type Signature**:
  ```rust
  impl<T, N: Dim, S: MatrixStorage<T, N, U1>> TryFrom<Polynomial<T, N, S>>
  for Matrix<T, <N as DimSub<U1>>::Output, <N as DimSub<U1>>::Output>
  where
      N: DimSub<U1>,
      <N as DimSub<U1>>::Output: Dim,
      T: Zero + One + Signed + Copy,
  {
      type Error = ConversionError;
      // ...
  }
  ```
  Unlike the inverse `Matrix` → `Polynomial` conversion
  (`matrix-design.md` §4.8.1), which requires `T: Float` for its
  division-based Faddeev–LeVerrier recursion, this conversion only places
  and negates coefficients into matrix cells — no division — so the
  narrower `Signed` bound suffices.
- **Behavior**: Coefficients are placed directly into the companion form
  (Bini et al., 2010 established the $O(N^2)$-time algorithm; Aurentz et
  al., 2015 reduced its storage from $O(N^2)$ to $O(N)$ while preserving
  backward stability). The structural guarantees (upper Hessenberg,
  unitary-plus-rank-one) hold regardless of which row/column convention
  places the coefficients, since all such placements are related by
  transpose/permutation. Root-finding accuracy is a matrix-level
  guarantee, not a coefficient-level one: Aurentz et al.'s backward-
  stability result guarantees the computed eigenvalues are exact for a
  matrix *near* the companion matrix, not that they are exact for a
  polynomial near $p$'s own coefficients (Aurentz et al., 2018) — callers
  root-finding on coefficient-sensitive input should read accuracy claims
  with that distinction in mind.
- **Failure Condition**: Returns `ConversionError::NonMonicPolynomial` if
  the leading coefficient is not `T::ONE`.

This conversion builds the companion matrix directly from coefficients — a
different, better-conditioned operation than transforming a *general*
state-space system into Controllable Canonical Form via its (often
near-singular) controllability matrix (`state-space-design.md` §5.5's
canonical-transformation realization path).

The inverse conversion (`Matrix` → characteristic `Polynomial` via the
Faddeev–LeVerrier algorithm) is specified in `matrix-design.md` §4.8.1,
which shares this document's Faddeev & Faddeeva (1963) citation for that
reason.

##### 4.7.2. Conversion to Tensor

Converts flat coefficient data into a 1D `Tensor<T, Layout, B>`, mirroring
`matrix-design.md` §4.8.2's `Matrix` → `Tensor` conversion.

- **Type Signature**:
  ```rust
  impl<T, N: Dim, B: Buffer<T>, Layout: TensorLayout> From<Polynomial<T, N, Dense<T, N, U1, B>>> for Tensor<T, Layout, B>
  where
      Layout: TensorLayout<Size = N>,
  {
      // Preserves backing buffer zero-copy when compile-time size and rank 1 match
  }
  ```
- **Behavior**: Maps coefficient storage directly into the flat buffer
  representation of the `Tensor`.
- **Infallible Compile-Time Bound**: Dimensions and rank are verified statically
  at compile time via `Layout: TensorLayout<Size = N>`, eliminating runtime
  `LayoutMismatch` failure modes.

#### 4.8. Error Handling & State Management

##### 4.8.1. Compile-Time Constraints

Capacity mismatches during polynomial arithmetic are rejected at compile
time via `Dim` type constraints, the same mechanism `matrix-design.md`
§4.9.1 uses for `Matrix` dimension mismatches.

##### 4.8.2. Runtime Error Taxonomy

`ConversionError` is shared across `Matrix`, `Polynomial` and `Tensor`
conversions and is defined once, canonically, in
[`error-design.md`](../math/error-design.md) §3 — not restated here.

`DivisionError` is specific to `Polynomial` and is defined here, matching
`matrix-design.md` §4.9.2's precedent of defining single-consumer error
enums directly in their owning design doc:

```rust
/// Errors returned by Polynomial::div_rem, supplementing ConversionError.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DivisionError {
    /// The divisor's leading coefficient is exactly zero.
    ZeroLeadingCoefficient,
    /// The divisor's degree exceeds the dividend's degree.
    DegreeMismatch,
}
```

##### 4.8.3. Runtime Fallbacks & Caveats

- Bounds-checked element access returns `Option<&T>`, matching
  `matrix-design.md` §4.9.3's `get`-method pattern.
- **Near-Singular Divisor**: `div_rem` accuracy degrades continuously as
  the divisor's leading coefficient shrinks toward (but does not reach)
  zero — a documented conditioning caveat, not a `DivisionError` variant
  (§4.6).
- **Host/Design-Time Scope**: `div_rem` and companion-matrix root-finding
  have no established fixed-point (Q31/Q15) numerical precedent in DSP
  reference libraries (unlike Horner evaluation and convolution, both
  standard fixed-point DSP primitives). These two operations are intended
  for floating-point, design-time use (e.g. offline controller synthesis,
  coefficient generation), not on-target fixed-point runtime paths.
- **Panic Path in `mul_with_conv`'s Dependency**: shipped
  `Convolution::convolve_input` (`src/math/dsp.rs:196-210`) panics via
  `assert!` on an undersized caller-provided output buffer, violating the
  crate's no-panic-outside-tests-and-examples rule (`CLAUDE.md`) and
  `storage-subprograms-design.md` NFR-4. `mul_with_conv` (§4.5) delegates to
  it directly. The correction (`assert!` → `debug_assert!`, matching the
  "Panics (Debug only)" convention `GEMV`/`GEMM` already use in
  `subprograms.rs`) is a required pre-implementation fix, tracked in §7.

---

### 5. Alternatives

#### 5.1. Aberth–Ehrlich Simultaneous Iteration (rejected for root-finding)

The Rust `aberth` crate (docs.rs, 2026) demonstrates a viable, `no_std`,
array-backed alternative to companion-matrix QR eigenvalue root-finding,
using simultaneous Aberth–Ehrlich iteration (cubic convergence for simple
roots). It was not chosen because its convergence rate is data-dependent
(cubic for simple roots, only linear for multiple or tightly clustered
roots), violating §2.2's Deterministic Execution non-functional
requirement. The companion-matrix QR approach (Bini et al., 2010; Aurentz
et al., 2015) keeps root-finding structurally consistent with the rest of
the crate's fixed-operation-count posture — the same determinism-over-
convergence-speed tradeoff `matrix-design.md` §5.5 applies when preferring
$LDL^T$ over pivoted alternatives for embedded targets.

#### 5.2. FFT-Based Polynomial Multiplication (rejected for `mul_poly`/

`mul_with_conv`)

Asymptotically faster than direct $O(N \times M)$ summation for large
degree, but numerically stable only when both operands' coefficients are of
comparable magnitude (van der Hoeven, 2008) — an assumption this crate
cannot make about arbitrary user-supplied coefficients. This is consistent
with CMSIS-DSP's own guidance that direct convolution, not an FFT-based
approach, is appropriate below its documented long-vector cutoff (ARM
CMSIS-DSP, 2025), which comfortably covers this crate's 127-element
capacity ceiling (§2.3 C-4).

#### 5.3. Single Unified Multiplication Method (rejected)

Merging `mul_poly` and `mul_with_conv` into one method was considered,
since they are currently algorithmically identical (§4.5). They are kept
separate so that `mul_with_conv` alone can later delegate to a hardware- or
fixed-point-specialized `Convolution<T>` implementation without changing
`mul_poly`'s own, strictly broader bound (`T: Copy + Zero + Add<Output=T> +
Mul<Output=T>`, §4.5 — no `Float` required, so `mul_poly` already works
for fixed-point and integer `T` today) or requiring downstream callers of
`mul_poly` to opt into `Convolution<T>`'s narrower, `Float`-only
specialization.

#### 5.4. Companion Form as the Only `Matrix` Conversion (rejected)

Letting `TryFrom<Polynomial>` mean companion form exclusively (§4.7.1)
conflates two operations with different cost, fallibility and shape.
Companion construction is $n \times n$, fallible on a non-monic input, and
reorders coefficients; a value-preserving column copy is $N \times 1$,
infallible, and moves the same buffer `Polynomial` already owns. Both are
provided: the column copy is the `From` conversion to
`Matrix<T, N, U1>` (a $\Theta(1)$ storage move, since
`DenseVectorArray<T, N>` and a single-column `DenseArray<T, N, 1>` share the
`Array<T, N>` buffer), and companion form stays the named, fallible
`TryFrom`. Overloading one conversion for both would force callers wanting
coefficients as a vector to accept a `ConversionError` they cannot
trigger.

---

### 6. Verification & Validation

#### 6.1. Verification Strategy

1. **Compile-Time Verification**: Capacity/degree mismatches are rejected
   by `Dim` type constraints (§4.8.1), eliminating a class of runtime
   bounds errors before they compile.
2. **Host/Target Tests**: Unit tests executed on host and qemu targets,
   matching `matrix-design.md` §6.1's testing tiers.
3. **Operand & Codegen Checks**: `as_array::<N>()` is exercised for every
   owning and borrowed alias, asserting `N` resolves to the alias's own const
   generic; `evaluate` and `mul_with_conv` are disassembled at
   `opt-level=3` to confirm zero panic paths survive the level-1 operand
   form (§4.3; `storage-subprograms-design.md` §7.1).
4. **Property-Based Testing**: `proptest` validation for commutativity
   ($P+Q=Q+P$), distributivity ($P(Q+R) = PQ + PR$) and division
   invariants ($P = QD + R$), adopting the random-generation framing
   associated with QuickCheck (Claessen & Hughes, 2000). This citation is
   carried forward from `matrix-design.md` §9 ref. 10 for the same
   methodology; the paper's own text remains paywalled and was not
   independently verified this research pass
   (`research/polynomial.json` unresolved_query_notes).

#### 6.2. Validation Strategy

**Cubic Spline Trajectory Generation**: For robotics and CNC path planning,
smooth motion paths are often generated using cubic splines. This example
uses `Polynomial` to store a pre-computed cubic trajectory and evaluate the
robot's position at a specific time step $t$ using Horner's method:

```rust
use control_rs::math::polynomial::{Polynomial, ArrayPolynomial};
use control_rs::math::num_types::U4;

/// Evaluates a cubic spline trajectory: p(t) = c_0 + c_1*t + c_2*t^2 + c_3*t^3
pub fn evaluate_trajectory(time_sec: f32) -> f32 {
    // Initialize the trajectory polynomial with ascending power coefficients
    // For example: 0.0m initial pos, 1.5m/s velocity, 0.2m/s^2 accel, -0.05m/s^3 jerk
    let trajectory: ArrayPolynomial<f32, U4> = Polynomial::from_coefficients(
        [0.0, 1.5, 0.2, -0.05]
    );

    // Evaluate the polynomial at the given time step
    // Horner's method ensures this takes exactly 3 additions and 3 multiplications.
    trajectory.evaluate(time_sec)
}
```

---

### 7. Risks & Open Questions

- **Horner Evaluation Fixed-Point Renormalization**: Unlike a single matrix
  multiply-accumulate or DSP convolution sum (both of which can use one
  wide accumulator truncated once at the end), Horner's recurrence feeds
  each step's result into the next step's multiplicand, requiring a
  Q-format rescale after *every* multiply-add rather than once per
  operation. No CMSIS-DSP or equivalent fixed-point reference
  implementation for Horner-style evaluation was found during research.
  `matrix-design.md` §7 flags the analogous Q31/Q15 accumulator-truncation
  risk for `Matrix` without resolving it either — this is a shared open
  risk across both types, not one `Matrix` has already solved.
- **`div_rem` / Root-Finding Fixed-Point Scope**: Should the host/design-
  time scoping in §4.8.3 be stated as a hard constraint (compile-time
  bound to floating-point `T`) or left as a documentation-only
  recommendation pending a concrete fixed-point use case?
- **`Convolution::convolve_input` Panic Path**: `src/math/dsp.rs:196-210`
  panics via `assert!` on an undersized output buffer, violating the crate's
  no-panic rule and `storage-subprograms-design.md` NFR-4. Correct it
  (`assert!` → `debug_assert!`) before or during `/cr-implement` (§4.8.3).
- **`Convolution` Call-Site Migration**: `num-traits-design.md` §9 Phase 3
  lists `Convolution`/`FFT`/`Discrete` (`dsp.rs`) as migration targets.
  Shipped source already declares `Convolution<T: Float>`; confirm at
  `/cr-implement` time whether any further re-bind is outstanding beyond
  that bound.
- **`Convolution` Operand Form**: `Convolution<T>` predates the level-1
  operand contract and takes slices, not `&[T; N]`
  (`storage-subprograms-design.md` §4.2.2). Whether `mul_with_conv`
  re-binds it to fixed-extent arrays, or keeps the slice form and accepts
  the residual bounds checks on that path alone, is unresolved and belongs
  to the `dsp.rs` owner rather than this document.

---

### 8. Development Plan

| Task / Feature                              | Description                                                                                                                                  | Estimated Effort |
|:--------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
| **Step 1: Storage & Constructors**          | `Polynomial<T, N, S>` struct over `MatrixStorage<T, N, U1>`, `ArrayPolynomial`/`PolynomialView`/`PolynomialViewMut` aliases, `as_array`/`as_slice` accessors, §4.4 constructors. | 1.5 Days |
| **Step 2: Core Arithmetic**                 | `Add`/`Sub`/`Neg` operator overloads, `mul_poly`, `mul_with_conv` via `Convolution<T>`.                                                      | 2.0 Days         |
| **Step 3: Evaluation, Calculus & Division** | Horner `evaluate`, derivative/integral methods, `div_rem` with `DivisionError` and the near-singular caveat.                                 | 2.5 Days         |
| **Step 4: Interoperability**                | Companion-`Matrix` `TryFrom` conversion, column-copy `From` conversion (§5.4), `Tensor` conversion, cross-check against `matrix-design.md`'s reverse Faddeev–LeVerrier conversion. | 2.0 Days |
| **Step 5: Verification**                    | `proptest` algebraic invariants, host/qemu unit tests, release-codegen check that `evaluate` retains zero panic paths, cubic-spline trajectory validation example. | 2.0 Days |

---

### 9. References

1. **Rust `polynomial` crate contributors. (2023).** _polynomial_: a
   no-std library for manipulating polynomials (Version 0.2.6). [Online].
   Available: https://lib.rs/crates/polynomial. Accessed: Aug. 7, 2026.
2. **ickk. (2026).** _aberth_: Aberth–Ehrlich simultaneous polynomial
   root-finding (Version 0.4.1). [Online]. Available:
   https://docs.rs/aberth/latest/aberth/. Accessed: Aug. 7, 2026.
3. **Horner, W. G. (1819).** A New Method of Solving Numerical Equations of
   All Orders, by Continuous Approximation. _Philosophical Transactions of
   the Royal Society of London_, 109, 308–335. — Origin of the $N-1$
   operation evaluation scheme; direct FLOP-count justification for
   `evaluate`.
4. **Higham, N. J. (2002).** _Accuracy and Stability of Numerical
   Algorithms_, 2nd ed., Ch. 5 "Polynomials". SIAM. — Backward-error bound
   $\gamma_{2n}$ for Horner's method, quantifying the accuracy claim in
   §4.6.
5. **Henrici, P. (1974).** _Applied and Computational Complex Analysis,
   Volume 1: Power Series, Integration, Conformal Mapping, Location of
   Zeros_. Wiley. (Cited edition: 1988 Wiley Classics Library reprint,
   ISBN 0471608416.) — Zero-location theory underpinning root-finding
   correctness and region bounds.
6. **Bini, D. A., Boito, P., Eidelman, Y., Gemignani, L., & Gohberg, I.
   (2010).** A Fast Implicit QR Eigenvalue Algorithm for Companion
   Matrices. _Linear Algebra and its Applications_, 432(8), 2006–2031. —
   Establishes the $O(N^2)$-time companion-matrix rootfinding algorithm
   (using $O(N^2)$ storage) for characteristic polynomials. Cited via its
   bibliographic record in Aurentz et al. (2018)'s own reference list; the
   paper's own text was not independently retrieved this pass (paywalled).
7. **Aurentz, J. L., Mach, T., Vandebril, R., & Watkins, D. S. (2015).**
   Fast and Backward Stable Computation of Roots of Polynomials. _SIAM
   Journal on Matrix Analysis and Applications_, 36(3), 942–973. — Reduces
   the Bini et al. (2010) companion-matrix rootfinder's storage from
   $O(N^2)$ to $O(N)$ while preserving backward stability. Complexity and
   backward-stability claims corroborated via nhigham.com and Aurentz et
   al. (2018)'s own citation record; the paper's own abstract was not
   independently retrieved this pass (paywalled).
8. **Aurentz, J. L., Mach, T., Vandebril, R., & Watkins, D. S. (2018).**
   Fast and Backward Stable Computation of Roots of Polynomials, Part II:
   Backward Error Analysis; Companion Matrix and Companion Pencil. _SIAM
   Journal on Matrix Analysis and Applications_, 1245–1269,
   doi: 10.1137/17M1152802. — Formalizes the distinction between backward
   stability with respect to the companion matrix and backward stability
   with respect to the polynomial's own coefficients (§4.7.1).
9. **Faddeev, D. K., & Faddeeva, V. N. (1963).** _Computational Methods of
   Linear Algebra_. W. H. Freeman and Company. — Trace-based derivation of
   the Faddeev–LeVerrier algorithm; shared with `matrix-design.md` §4.8.1,
   which specifies the inverse (`Matrix` → characteristic `Polynomial`)
   conversion.
10. **van der Hoeven, J. (2008).** Making fast multiplication of
    polynomials numerically stable. [Online]. Available:
    https://www.texmacs.org/joris/stablemult/stablemult.html. Accessed:
    Aug. 7, 2026. — Numerical-stability precondition (comparable-magnitude
    coefficients) for FFT-based polynomial multiplication, motivating its
    rejection in §5.2.
11. **ARM. (2025).** Convolution. _CMSIS-DSP_ (Version 1.15.0). [Online].
    Available:
    https://arm-software.github.io/CMSIS-DSP/v1.15.0/group__Conv.html.
    Accessed: Aug. 7, 2026. — Long-vector FFT cutoff guidance and Q15/Q31
    fixed-point accumulator/overflow behavior for direct convolution,
    referenced in §5.2.
12. **Claessen, K., & Hughes, J. (2000).** QuickCheck: A Lightweight Tool
    for Random Testing of Haskell Programs. _ACM SIGPLAN Notices_, 35(9),
    268–279. — Property-based testing methodology framing `proptest`
    algebraic invariant checks (§6.1); citation carried forward unverified
    (paywalled) per `research/polynomial.json`.

---

### 10. Revision History

| Revision | Date            | Author          | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
|:---------|:----------------|:----------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1.0      | July 12, 2026   | @MitchellDScott | Initial draft with static array layout.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| 1.1      | July 26, 2026   | @MitchellDScott | Integrated `Storage` trait hierarchy to support borrowed zero-copy views and ROM storage.                                                                                                                                                                                                                                                                                                                                                                                                                               |
| 1.2      | July 26, 2026   | @MitchellDScott | Added inline academic citations and 3-tiered references section.                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| 1.4      | August 2, 2026  | @MitchellDScott | Added citations; clarified companion-form conditioning; documented `div_rem` caveats; added Alternatives, Risks, and Development Plan.                                                                                                                                                                                                                                                                                                                                                                                  |
| 1.5      | August 2, 2026  | @MitchellDScott | Propagated `num-traits-design.md` pivot to companion-matrix bound; relocated `ConversionError`; corrected a factual error.                                                                                                                                                                                                                                                                                                                                                                                              |
| 1.6      | August 2, 2026  | @MitchellDScott | Separated coefficient-ordering convention from `Storage` layout; cross-referenced `storage-trait-design.md` instead of restating it.                                                                                                                                                                                                                                                                                                                                                                                    |
| 1.7      | August 11, 2026 | @MitchellDScott | Restructured to match `matrix-design.md`'s section skeleton (Technical Overview, Core Architecture subsections, flat References list); corrected the `Convolution<T>` bound (`Real` → shipped `Float`) and `num-traits-design.md`'s now-Approved status; fixed the `matrix-design.md` §5.2→§5.5 cross-reference and the unresolved `van der Hoeven` citation; narrowed the ecosystem-consistency claim to directly-evidenced crates; flattened the References list and resolved every entry against an inline citation. |
| 1.8      | August 16, 2026 | @mitchelldscott | Propagated four-tier `BlasStorage` hierarchy (§1, §2.1, §2.2, §3, §4.1, §4.2): updated `Polynomial` bound to `MatrixStorage<T, N, U1>`, default storage alias to `Dense<T, N, U1, Array<T, N>>`, and view aliases `PolynomialView`/`PolynomialViewMut` to `Dense` view leaves (`Ref`/`RefMut`).                                                                                                                                                                                                                         |
| 1.9      | August 16, 2026 | @mitchelldscott | Harmonized with `storage-subprograms-design.md` Rev 1.4 (§1, §4.1): updated `Polynomial` storage bound to single-parameter `S: MatrixStorage<T, R = N, C = U1>` with associated `R`/`C` types and `FixedBlasStorage<T>` array access (`as_array()`).                                                                                                                                                                                                                                                                    |
| 1.10     | August 16, 2026 | @mitchelldscott | Harmonized with `storage-subprograms-design.md` Rev 1.5 (§1, §4.1): updated `Polynomial` storage bound to `MatrixStorage<T, N, U1>` where `N: Dim` and `U1` are generic trait parameters on `MatrixStorage`.                                                                                                                                                                                                                                                                                                            |
| 1.11     | August 16, 2026 | @mitchelldscott | Reconciled residual `Storage` and `ContiguousStorage`/`ContiguousStorageMut` bounds in §1, §4.3, §4.7 with `BlasStorage<T, N, U1>` / `MatrixStorage<T, N, U1>`.                                                                                                                                                                                                                                                                                                                                                         |
| 1.12     | August 16, 2026 | @mitchelldscott | Refactored §2 to outcome-focused requirements (Horner FMA op count, convolution multiplication, error contracts).                                                                                                                                                                                                                                                                                                                                                                                                       |
| 1.13     | August 16, 2026 | @mitchelldscott | Updated `Polynomial` generic defaults and type aliases (`ArrayPolynomial`, `PolynomialView`, `PolynomialViewMut`) to convenience storage aliases (`DenseVectorArray`, `DenseVectorRef`, `DenseVectorRefMut`).                                                                                                                                                                                                                                                                                                           |
| 1.14     | August 16, 2026 | @mitchelldscott | Encapsulated 1D capacity evaluation (`N::USIZE`) inside `DenseVectorArray<T, N>`, eliminating extra capacity parameters from `Polynomial`.                                                                                                                                                                                                                                                                                                                                                                              |
| 1.15     | August 16, 2026 | @mitchelldscott | Removed obsolete `FixedBlasStorage` reference from §1 Storage Trait Reuse overview.                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| 1.16     | August 18, 2026 | @mitchelldscott | Aligned §4.7.2 `Polynomial` → `Tensor` conversion to infallible `From` bounded by `TensorLayout<Size = N>`, eliminating obsolete `LayoutMismatch` runtime check.                                                                                                                                                                                                                                                                                                                                                         |
| 1.17     | August 18, 2026 | @mitchelldscott | Propagated `storage-subprograms-design.md` Rev 1.11–1.12: `ArrayPolynomial<T, const N>` over `DenseVectorArray<T, N>`; removed `from_slice` (FR-6); `mul_poly` no longer names `DenseVectorArray` at a `Dim` associated type; dropped the stale `ConversionError` inline-definition note. |
| 1.18     | August 18, 2026 | @mitchelldscott | Propagated storage Rev 1.16: `PolynomialView`/`PolynomialViewMut` take `const N` over `DenseVectorRef`/`VectorRef` (`&[T; N]`). |
| 1.19     | August 19, 2026 | @mitchelldscott | Propagated `storage-subprograms-design.md` Rev 1.21 (Approved). Struct bound lowered to `BlasStorage<T, N, U1>` with `MatrixStorage` required per `impl` block, mirroring `matrix-design.md` §4.1.1; recorded that a single column reaches only the leading-dimension branch (`type LDA = Const<N>`) and has no packed analogue. Added the level-1 operand contract (`as_array::<N>()`, `INC_X = 1`, `BUF_X = N`) as FR-5 and moved Horner evaluation onto it (§4.3, §4.6); relaxed the companion conversion to `BlasStorage`; reframed §5.4 as the companion-versus-column-copy tradeoff; added operand and release-codegen verification (§6.1); corrected stale requirement IDs (`ST-NFR-1`, `NFR-3`) and removed draft-revision narration from §7. |
| 1.20     | August 20, 2026 | @mitchelldscott | Propagated `storage-subprograms-design.md` Rev 1.31: packed branch is `PackedStorage` with Phase-1 `Diagonal` and planned `SP`/`TP`; still no $N \times 1$ packed analogue. |
| 1.21     | August 20, 2026 | @mitchelldscott | Renamed `BlasStorage` -> `MatrixStorage` (universal floor) and the prior `MatrixStorage` -> `DenseStorage` (leading-dimension branch), matching `storage-subprograms-design.md` Rev 1.31 and `matrix-design.md` Rev 1.31; updated §1, struct bound, and companion-conversion bound. |
| 1.22     | August 22, 2026 | @MitchellDScott | `Convolution` length failure returns `ConversionError::DimensionMismatch`, not `LinAlgError::DimensionMismatch` (`error-design.md` FR-3). Storage hierarchy citations remain on the deleted `storage-subprograms-design.md`; this document stays Draft pending a dedicated retarget to `storage-design.md`. |
| 1.23     | August 22, 2026 | @MitchellDScott | C-4 capacity bound cites `Const<N>: Dim` (`num-types-design.md` C-1), not a dense `U*` alias range. |

---
