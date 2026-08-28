# Polynomial Type (Design Document)

![Date Badge](https://img.shields.io/badge/Date-August_25,_2026-blue)
![Status Badge](https://img.shields.io/badge/Doc%20Status-Approved-green)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

---

### 1. Introduction

This module provides statically-typed, single-variable polynomial
representations and polynomial arithmetic for digital filter design, trajectory
interpolation, discretization algebra, and root finding.

Primary usage scenarios:

- **Digital Filter Implementation**: Representing FIR and IIR digital filter
  numerator and denominator coefficient vectors ($B(z)$ and $A(z)$) for signal
  processing pipelines.
- **Continuous System Discretization**: Performing algebraic polynomial
  transformations (e.g., Tustin bilinear
  transform $s \leftarrow \frac{2}{T_s} \frac{z-1}{z+1}$ and Zero-Order Hold) to
  map continuous Laplace-domain transfer functions into discrete $Z$-domain
  equivalents.
- **Motion Profile & Trajectory Generation**: Evaluating cubic, quintic, and
  higher-order spline polynomials for smooth robotic joint position, velocity,
  and jerk profiles.
- **Characteristic Polynomials & Stability Analysis**: Constructing
  characteristic polynomials ($\det(\lambda I - A) = 0$) and computing
  polynomial roots via companion matrix eigen-decomposition to evaluate
  closed-loop system poles and stability margins.

---

### 2. Requirements

#### 2.1. Functional Requirements

- **FR-1 — Ascending Degree Coefficient Indexing**: Single-variable polynomials
  represent $P(s) = \sum_{i=0}^{N-1} a_i s^i$ where index $i$ corresponds directly
  to the coefficient of $s^i$. Indexing must preserve degree semantics across
  differentiation, integration, and evaluation.
- **FR-2 — Real-Time Horner Evaluation**: Evaluating $P(s)$ at a scalar
  point $s$ executes in exactly $N-1$ multiply-accumulate (FMA) steps for an
  $N$-coefficient polynomial of degree $N-1$ (Higham, 2002). Evaluation must execute
  in constant working memory without dynamic allocation.
- **FR-3 — Discrete Convolution Multiplication**: Multiplying degree-$(N-1)$ and
  degree-$(M-1)$ polynomials ($N$ and $M$ coefficients) yields an exact
  degree-$(N+M-2)$ polynomial ($N+M-1$ coefficients) whose coefficients equal the
  linear discrete convolution of the inputs (ARM, 2025). Coefficients must be
  computed without numerical truncation across supported scalar types.
- **FR-4 — Fallible Polynomial Division**: Polynomial Euclidean
  division ($A(s) = Q(s) D(s) + R(s)$) returns the quotient and remainder, or
  returns a typed error variant when dividing by a zero polynomial.
- **FR-5 — Companion Matrix Realization**: Polynomials construct companion
  matrices whose eigenvalues equal the polynomial roots. Conversion must return
  an explicit error if the leading coefficient is zero (Aurentz et al., 2018).
- **FR-6 — Discretization & Trajectory Transforms**: Evaluates trajectory
  splines (cubic and quintic) and parameter substitutions (e.g. bilinear
  transform $s \to \frac{2}{T_s}\frac{z-1}{z+1}$) without heap allocation.

#### 2.2. Non-Functional Requirements

- **NFR-1 — Data-Independent Evaluation Latency**: Polynomial evaluation cycle
  count depends only on degree $N$, not on numerical coefficient values,
  ensuring predictable Worst-Case Execution Time (WCET).
- **NFR-2 — Memory Footprint Predictability**: Polynomial operations execute
  within compile-time-bounded stack frames without dynamic heap growth.

#### 2.3. Constraints

- **C-1 — Maximum Degree Bound**: Polynomial degree is statically
  bounded ($N \le 1024$) to prevent stack overflow on microcontroller targets (
  `num-types-design.md` C-1).
- **C-2 — `#![no_std]` / Zero Heap Allocation**: All polynomial representations
  and operations operate strictly on fixed stack arrays or borrowed memory
  slices.

---

### 3. Technical Overview

`Polynomial<T, N, S>` provides a statically sized, degree-aware polynomial
representation over scalar type `T`, dimension `N: Dim`, and strided storage
backend `S: DenseStorage<T, R = N, C = Const<1>>`. By specializing `Matrix`'s storage
hierarchy to a single column, it reuses `ArrayStorage<T, N, 1>` for owning
values and `StorageView<'a, T, N, Const<1>>` for borrowed slices.

The module provides Horner polynomial evaluation, discrete convolution
multiplication, fallible polynomial division, trajectory spline generation, and
companion-matrix conversion for root finding, while operating entirely within
`#![no_std]` stack allocations.

---

### 4. Core Architecture

The `Polynomial` struct is implemented in `src/polynomial/mod.rs`, replacing
the module's current pre-`Storage` stub (a bare `Polynomial<T>` trait over
`&[T]`, with no `Storage` integration) with the design below.

#### 4.1. Generics Foundation & Sizing

The core `Polynomial` structure decouples mathematical dimensions from
physical storage using the same bound `Matrix` names on its struct
(`matrix-design.md` §4.1.1), with the column dimension fixed to `Const<1>`:

```rust
pub struct Polynomial<T, N: Dim, S: DenseStorage<T, R=N, C=Const<1>>> {
    storage: S,
    _marker: core::marker::PhantomData<N>,
}
```

Operations needing a padding-free slice (FFI hand-off, §4.3) additionally
require `S: ContiguousStorage<T>` on their own `impl` block, as `Matrix`
does. Every owning single-column leaf satisfies both, so the split costs
nothing here; it keeps the two documents' storage story identical.

Here, `N` represents the capacity (number of coefficients, maximum possible
degree is $N - 1$) and `S` defines where the coefficients reside (e.g. stack
`ArrayStorage<T, N, 1>`, a borrowed `StorageView` or static Flash memory).
`ArrayStorage` takes bare `const usize` capacities, so it is not a valid
default for `N: Dim`.

#### 4.2. Coefficient Layout & Storage Strategy

By parameterizing `Polynomial` over `DenseStorage<T, R = N, C = Const<1>>`,
`control-rs` supports multiple ownership models without duplicating
algebraic logic:

```rust
/// Owning polynomial backed by a stack array — `N` is the alias's own const generic.
pub type ArrayPolynomial<T, const N: usize> =
Polynomial<T, Const<N>, ArrayStorage<T, N, 1>>;

/// Zero-copy read-only borrowed polynomial view.
pub type PolynomialView<'a, T, N> =
Polynomial<T, N, StorageView<'a, T, N, Const<1>>>;

/// Zero-copy mutable borrowed polynomial view.
pub type PolynomialViewMut<'a, T, N> =
Polynomial<T, N, StorageViewMut<'a, T, N, Const<1>>>;
```

Coefficients are stored in **ascending order of powers**:
$$ p(x) = c_0 + c_1 x + c_2 x^2 + \dots + c_{N-1} x^{N-1} $$
where index `i` maps to the coefficient of $x^i$. This is a logical
convention (`index = degree of term`) that `Polynomial` itself defines and
fully resolves; it says nothing about physical memory layout, which remains
the storage leaf's concern.

- **Ascending Power Storage Rationale**:
    - Direct index-to-exponent mapping: element at index `i` corresponds
      directly to $x^i$.
    - Zero-cost padding: Adding polynomials of differing capacities aligns
      coefficients naturally without element shifting.
    - **Ecosystem Consistency**: Standard polynomial libraries (`polynomial`,
      `aberth`) use ascending-degree indexing where index `i` corresponds to the
      coefficient of $x^i$.

**Physical layout is storage's concern, not `Polynomial`'s.** Cold paths
address coefficients through a bounds-checked accessor and hot paths hand
the typed storage operand to a kernel (§4.3), never through a raw offset, so
`Polynomial` is agnostic to whichever concrete backend it is instantiated
over. Row-major and column-major degenerate to the same addressing scheme
for a single-column container, so the layout distinction
`matrix-design.md` §4.2 draws between them (cache locality across columns,
BLAS interoperability) does not apply here in the same way. A non-unit row
stride remains representable through `StorageView`, which is how a
coefficient run embedded in a larger buffer is borrowed without a copy.

#### 4.3. Memory Representation & Slicing

Padding-free slice interfaces are exposed when the storage backend
implements `ContiguousStorage` or `ContiguousStorageMut`, gated the same way
`matrix-design.md` §4.3 gates `Matrix::as_slice`/`as_mut_slice`:

```rust
impl<T, N: Dim, S> Polynomial<T, N, S>
where
    S: ContiguousStorage<T, R=N, C=Const<1>>,
{
    /// Exposes a safe contiguous slice view of coefficient memory.
    pub fn as_slice(&self) -> &[T] {
        self.storage.as_slice()
    }
}

impl<T, N: Dim, S> Polynomial<T, N, S>
where
    S: ContiguousStorageMut<T, R=N, C=Const<1>>,
{
    /// Exposes a safe mutable contiguous slice view of coefficient memory.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.storage.as_mut_slice()
    }
}
```

Level 1 kernels take the typed storage operand directly
(`subprograms-design.md` FR-9): shape comes from `S::R::USIZE` and
addressing from `as_ptr()` plus the leaf's strides, both monomorphization
constants on an owning array leaf, so LLVM folds the loop bounds
(`storage-design.md` NFR-3), with no `as_array::<N>() -> &[T; N]` accessor
or `DimMul` bound required at the call site. `as_slice()` is the inspection
and FFI hand-off path, which is what lets
`mul_with_conv` (§4.5) and `evaluate` (§4.6) pass into hardware DSP kernels
without copying (NFR-1).

Unlike `Matrix` (`matrix-design.md` §4.3.1), `Polynomial` does not define a
zero-copy transposed view: transposition is not a meaningful operation on
an $N \times 1$ shape, so that subsection of `Matrix`'s architecture has no
analogue here.

Borrowed views are constructed through `storage-design.md` FR-2:
`ArrayPolynomial::view()` / `view_mut()` copy `N` from the owning alias's
const generic. `StorageView::new` is the only path that wraps an
erased-length slice, and it is fallible with
`ConversionError::DimensionMismatch` (`storage-design.md` §4.6).

#### 4.4. Instantiation & Constructors

- `pub const fn constant(val: T) -> ArrayPolynomial<T, 1> where T: Copy`:
  Constructs a degree-0 polynomial containing a single coefficient.
- `pub const fn line(c0: T, c1: T) -> ArrayPolynomial<T, 2> where T: Copy`:
  Constructs a degree-1 linear polynomial $c_0 + c_1 x$.
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
   impl<T, N: Dim, S1: DenseStorage<T, R = N, C = Const<1>>> Polynomial<T, N, S1> {
       pub fn mul_poly<M: Dim, S2: DenseStorage<T, R = M, C = Const<1>>, SOut>(
           &self,
           other: &Polynomial<T, M, S2>,
       ) -> Polynomial<T, <<N as DimAdd<M>>::Output as DimSub<Const<1>>>::Output, SOut>
       where
           N: DimAdd<M>,
           <N as DimAdd<M>>::Output: DimSub<Const<1>>,
           <N as DimAdd<M>>::Output: Dim,
           SOut: DenseStorageMut<
               T,
               R = <<N as DimAdd<M>>::Output as DimSub<Const<1>>>::Output,
               C = Const<1>,
           >,
           T: Scalar,
       { /* ... */ }
   }
   ```
   The returned capacity `N + M - 1` is an exact bound, not an
   approximation: ascending-power zero-padding (§4.2) guarantees any
   coefficient above the true product degree evaluates to `T::zero()`
   rather than a silently wrong value. The return type does not name
   `ArrayStorage<T, Out, 1>`: that leaf takes bare `const usize` capacities,
   not a `Dim` associated type (`storage-design.md` FR-1, C-4). The
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
  $p(x) = c_0 + x(c_1 + x(c_2 + \dots))$ over the typed storage operand,
  descending from index $N-1$. The extent is `S::R::USIZE`, a
  monomorphization constant, so the loop carries no bounds check and no
  panic path (`storage-design.md` NFR-3). Minimizes floating-point rounding
  error and operation count to $N-1$ additions and multiplications
  (Horner, 1819). The recurrence is multiply-add only, so it holds at
  `T: Scalar` and admits integer, fixed-point and complex coefficients.
  The computed result is exact for a polynomial whose coefficients are
  relatively perturbed by at most $\gamma_{2n} = 2nu / (1 - 2nu)$ from
  $p$'s true coefficients, where $u$ is unit roundoff (Higham, 2002, Ch.
    5) — a small, degree-linear backward-error bound quantifying the
       "minimizes rounding error" claim above.
- **Trajectory Splines & Bilinear Substitution (FR-6)**:
  ```rust
  impl<T: Copy + Zero + One> ArrayPolynomial<T, 4> {
      /// Cubic Hermite segment on $t \in [0, 1]$ from endpoints
      /// $(p_0, v_0)$ and $(p_1, v_1)$.
      pub fn cubic(p0: T, p1: T, v0: T, v1: T) -> Self { /* ... */ }
  }
  impl<T: Copy + Zero + One> ArrayPolynomial<T, 6> {
      /// Quintic Hermite segment on $t \in [0, 1]$ from endpoints
      /// $(p_0, v_0, a_0)$ and $(p_1, v_1, a_1)$.
      pub fn quintic(p0: T, p1: T, v0: T, v1: T, a0: T, a1: T) -> Self { /* ... */ }
  }
  impl<T: Float + Copy, const N: usize> ArrayPolynomial<T, N> {
      /// Substitute $s = \frac{2}{T_s}\frac{z-1}{z+1}$ and clear $(z+1)^{N-1}$.
      pub fn compose_bilinear(&self, sample_time: T) -> Self { /* ... */ }
  }
  ```
- **Polynomial Division (`div_rem`)**: Computes quotient and remainder with
  statically checked degree bounds (`Q = N - M + 1`, `R = M - 1`):
  ```rust
  impl<T, N: Dim, S: DenseStorage<T, R = N, C = Const<1>>> Polynomial<T, N, S> {
      pub fn div_rem<M: Dim, Sm: DenseStorage<T, R = M, C = Const<1>>, Q: Dim, R: Dim>(
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
  impl<T, N: Dim, S: DenseStorage<T, R = N, C = Const<1>>> TryFrom<Polynomial<T, N, S>>
  for Matrix<T, <N as DimSub<Const<1>>>::Output, <N as DimSub<Const<1>>>::Output>
  where
      N: DimSub<Const<1>>,
      <N as DimSub<Const<1>>>::Output: Dim,
      T: Zero + One + Signed + Copy,
  {
      type Error = ConversionError;
      // ...
  }
  ```
  This conversion only places and negates coefficients, so `T: Signed`
  suffices; the inverse Faddeev–LeVerrier conversion is specified in
  `matrix-design.md` §4.8.1.
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

##### 4.7.2. Conversion to Tensor

Converts flat coefficient data into a 1D `Tensor<T, Layout, B>`, mirroring
`matrix-design.md` §4.8.2's `Matrix` → `Tensor` conversion.

- **Type Signature**:
  ```rust
  impl<T, N: Dim, S, Layout: TensorLayout> From<Polynomial<T, N, S>> for Tensor<T, Layout, S>
  where
      S: ContiguousStorage<T, R = N, C = Const<1>>,
      Layout: TensorLayout<Size = N>,
  {
      // Preserves backing buffer zero-copy when compile-time size and rank 1 match
  }
  ```
- **Behavior**: Maps the leaf's padding-free slice directly into the flat
  buffer representation of the `Tensor`. The `ContiguousStorage` bound is
  what makes the mapping zero-copy; a strided `StorageView` has no such
  slice and converts by element copy
  (`tensor-design.md` §4.1, `FlatBuffer<T>`).
- **Infallible Compile-Time Bound**: Dimensions and rank are verified statically
  at compile time via `Layout: TensorLayout<Size = N>`.
  This conversion cannot produce `ConversionError::LayoutMismatch`
  (`error-design.md` §3).

#### 4.8. Error Handling & State Management

##### 4.8.1. Compile-Time Constraints

Capacity mismatches during polynomial arithmetic are rejected at compile
time via `Dim` type constraints, the same mechanism `matrix-design.md`
§4.9.1 uses for `Matrix` dimension mismatches.

##### 4.8.2. Runtime Error Taxonomy

`ConversionError` is shared across `Matrix`, `Polynomial` and `Tensor`
conversions and is defined once, canonically, in
[`error-design.md`](../math/error-design.md) §3 — not restated here.

`DivisionError` is specific to `Polynomial` and stays in this module, which
`error-design.md` FR-1 states as the rule: an error type consumed by more
than one sibling module is defined once in `src/math/mod.rs`, while a
single-consumer enum stays with its owner.

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
  `Convolution::convolve_input` ([`src/math/dsp.rs`](../../src/math/dsp.rs)) panics via
  `assert!` on an undersized caller-provided output buffer, violating the
  crate's no-panic-outside-tests-and-examples rule ([`CLAUDE.md`](../../CLAUDE.md)) and
  `subprograms-design.md` NFR-3. `mul_with_conv` (§4.5) delegates to it
  directly. The correction (`assert!` → `debug_assert!`, matching the
  `debug_assert_eq!` precondition convention `Gemv`/`Gemm` already use in
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

#### 5.2. FFT-Based Polynomial Multiplication (rejected for `mul_poly`/`mul_with_conv`)

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
Mul<Output=T>`, satisfied by `T: Scalar`, §4.5 — no `Float` required, so
`mul_poly` already works for fixed-point, integer and complex `T` today) or
requiring downstream callers of
`mul_poly` to opt into `Convolution<T>`'s narrower, `Float`-only
specialization.

#### 5.4. Companion Form as the Only `Matrix` Conversion (rejected)

Letting `TryFrom<Polynomial>` mean companion form exclusively (§4.7.1)
conflates two operations with different cost, fallibility and shape.
Companion construction is $n \times n$, fallible on a non-monic input, and
reorders coefficients; a value-preserving column copy is $N \times 1$,
infallible, and moves the same buffer `Polynomial` already owns. Both are
provided: the column copy is the `From` conversion to
`Matrix<T, N, Const<1>>` (a $\Theta(1)$ storage move, since both wrappers hold the
same `ArrayStorage<T, N, 1>` leaf), and companion form stays the named,
fallible
`TryFrom`. Overloading one conversion for both would force callers wanting
coefficients as a vector to accept a `ConversionError` they cannot
trigger.

---

### 6. Verification & Validation

#### 6.1. Objectives

- Demonstrate compile-time verification of polynomial capacity and degree
  bounds.
- Demonstrate numerical accuracy and backward error bounds ($\gamma_{2n}$) for
  Horner's method evaluation.
- Demonstrate exact algebraic ring properties (commutativity, associativity,
  distributivity) for addition, subtraction, and multiplication.
- Demonstrate numerical correctness of companion matrix root-finding conversions
  and polynomial calculus.
- Demonstrate zero dynamic heap allocation in `#![no_std]` execution and
  deterministic real-time performance.

#### 6.2. Methods

| Method                    | Mechanism                                                  | Requirements discharged  |
|:--------------------------|:-----------------------------------------------------------|:-------------------------|
| Compile-time shape check  | Type-level `Dim` sizing and `compile_fail` doctests        | FR-1, C-1                |
| Requirements-based test   | `#[test]` unit tests over boundary conditions and division | FR-2, FR-4, FR-5, FR-6   |
| Property-based test       | `proptest` suites verifying ring algebraic invariants      | FR-2, FR-3               |
| Doctest                   | Runnable rustdoc examples                                  | FR-2, FR-5               |
| Back-to-back comparison   | `examples/numerical-models/python3/polynomial.py` vs `src/polynomial.rs` JSON; [`numerical-models-design.md`](numerical-models-design.md) §6.3 | FR-2, FR-3, FR-6         |
| Resource usage evaluation | `no_alloc` audit, `size_of` assertions, stack analysis                           | NFR-2, C-2               |
| On-target execution       | ETS suites under QEMU and Teensy hardware                  | NFR-1                    |
| Coverage measurement      | `cargo coverage` reporting statement and branch metrics    | FR-1..FR-6, NFR-1..NFR-2 |

#### 6.3. Acceptance Criteria

| Claim                             | Oracle                                             | Measure                     | Bound                                                                                                     | Justification                                                  |
|:----------------------------------|:---------------------------------------------------|:----------------------------|:----------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------|
| Horner evaluation backward error  | Manufactured roots / prototype                     | Relative error              | $\|p(x) - \hat{p}(x)\| \le \gamma_{2n} \tilde{p}(\|x\|)$ where $\gamma_k = \frac{k\epsilon}{1-k\epsilon}$ | Backward stability of Horner's method (Higham, 2002)           |
| Polynomial multiplication         | Convolution algebraic definition                   | Absolute error              | $\|c_k - \sum a_i b_{k-i}\|_\infty \le (N+M)\epsilon$                                                     | Discrete convolution arithmetic bound (Higham, 2002)           |
| Division remainder relation       | Identity $A(x) = Q(x)B(x) + R(x)$                  | Exact equality / Rel. error | $\|A - (QB + R)\|_\infty \le \max(N, M)\epsilon$                                                          | Euclidean division invariant                                   |
| Polynomial derivative             | Analytic power rule $\frac{d}{dx} x^k = k x^{k-1}$ | Exact equality              | $0$ (exact for integer/fixed-point)                                                                       | Exact algebraic derivative definition                          |
| Companion matrix eigenvalues      | Known root sets                                    | Absolute error              | $\|\lambda_i - r_i\| \le \mathcal{O}(\epsilon \kappa(p))$                                                 | Backward stable companion matrix pencil (Aurentz et al., 2018) |
| Zero leading denominator division | Divisor with zero leading coefficient              | Exact equality              | `Err(DivisionError::ZeroLeadingCoefficient)`                                                              | Precondition failure contract                                  |
| Zero-allocation execution         | Host allocator interception                        | Exact equality              | 0 heap allocations                                                                                        | NFR-1 `#![no_std]` invariant                                   |

#### 6.4. Traceability

| Requirement                                     | Method                                       | Artifact                                                  |
|:------------------------------------------------|:---------------------------------------------|:----------------------------------------------------------|
| FR-1 — Ascending Degree Coefficient Indexing    | Compile-time shape check                     | rustdoc `compile_fail` doctests in `src/polynomial/mod.rs`            |
| FR-2 — Real-Time Horner Evaluation              | Requirements-based test, Property-based test | `src/polynomial/tests/polynomial_tests.rs::test_polynomial_evaluation` |
| FR-3 — Discrete Convolution Multiplication      | Property-based test, Back-to-back comparison | `src/polynomial/tests/polynomial_tests.rs::test_polynomial_multiplication` |
| FR-4 — Fallible Polynomial Division             | Requirements-based test                      | `src/polynomial/tests/polynomial_tests.rs::test_polynomial_div_rem`   |
| FR-5 — Companion Matrix Realization             | Back-to-back comparison                      | `src/polynomial/tests/polynomial_tests.rs::test_companion_matrix`     |
| FR-6 — Discretization & Trajectory Transforms   | Requirements-based test, Doctest             | `src/polynomial/tests/polynomial_tests.rs::test_cubic_quintic_bilinear` |
| NFR-1 — Data-Independent Evaluation Latency     | On-target execution                          | ETS disassembly audit for zero panic landing pads         |
| NFR-2 — Memory Footprint Predictability         | Resource usage evaluation                    | `#![no_std]` host allocator audit                         |
| C-1 — Maximum Degree Bound                      | Compile-time shape check                     | `clippy::large_stack_arrays` CI check                     |
| C-2 — `#![no_std]` / Zero Heap Allocation       | Resource usage evaluation                    | Compilation under `#![no_std]` targets                    |

#### 6.5. Coverage

- **Target**: $\ge 90\%$ statement coverage, $\ge 85\%$ branch coverage reported
  via `cargo coverage`.
- **Excluded**: Target-specific assembly branches tested exclusively via ETS and
  debug formatting routines (`core::fmt::Debug`).

#### 6.6. Validation

- **Polynomial Evaluation, Calculus, & Companion Realization**: Verification of
  degree-bounded polynomial construction, real and complex Horner evaluation,
  analytical differentiation/integration, polynomial multiplication, Euclidean division,
  Frobenius companion matrix formulation, and clustered-root Horner
  $p(x)=(x-1)^8(x-1.01)^8$ on a 128-point sweep in
  `examples/numerical-models/src/polynomial.rs`.

#### 6.7. Not Verified

- Root finding for ill-conditioned polynomials with high multiplicity roots (
  where condition number $\kappa(p) \to \infty$) is not guaranteed to achieve
  backward stability without multi-precision arithmetic. The example crate
  evaluates clustered-root Horner at degree 16
  ([`numerical-models-design.md`](numerical-models-design.md) §6.6); root
  *finding* for $\kappa(p)\to\infty$ is still not claimed.
- Fixed-point Horner evaluation without per-iteration dynamic scaling may suffer
  precision degradation for dynamic ranges $> 2^{16}$.

---

### 7. Performance & Resource Considerations

- **Memory Footprint**: Owning `ArrayPolynomial<T, N>` occupies
  exactly $N \times \text{size\_of}(T)$ bytes on stack without heap overhead.
- **Horner FLOP Count**: `evaluate` executes in exactly $N-1$ additions
  and $N-1$ multiplications (utilizing hardware FMA instructions on ARM
  Cortex-M4/M7).
- **Zero-Copy Views**: `PolynomialView` and `PolynomialViewMut` occupy only 2
  pointer words plus stride metadata, avoiding buffer copying.

---

### 8. Risks & Open Questions

- **Horner Evaluation Fixed-Point Renormalization**: Unlike single-pass dot
  products that maintain a wide accumulator, Horner's method feeds the
  accumulator back into the multiplicand at each step, requiring per-iteration
  scaling in fixed-point / Q-format representations.
- **Root-Finding Precision Scope**: Polynomial root finding via companion matrix
  QR iterations requires floating-point scalar support (`T: Float`); integer
  and fixed-point coefficient types are out of scope for eigenvalue extraction.

---

### 9. Development Plan

| Task / Feature                              | Description                                                                                                                                                                                     | Estimated Effort |
|:--------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
| **Step 1: Storage & Constructors**          | `Polynomial<T, N, S>` struct over `DenseStorage<T, R = N, C = Const<1>>`, `ArrayPolynomial`/`PolynomialView`/`PolynomialViewMut` aliases, the `ContiguousStorage` slice accessors, §4.4 constructors. | 1.5 Days         |
| **Step 2: Core Arithmetic**                 | `Add`/`Sub`/`Neg` operator overloads, `mul_poly`, `mul_with_conv` via `Convolution<T>`.                                                                                                         | 2.0 Days         |
| **Step 3: Evaluation, Calculus & Division** | Horner `evaluate`, derivative/integral methods, `div_rem` with `DivisionError` and the near-singular caveat.                                                                                    | 2.5 Days         |
| **Step 4: Interoperability**                | Companion-`Matrix` `TryFrom` conversion, column-copy `From` conversion (§5.4), `Tensor` conversion, cross-check against `matrix-design.md`'s reverse Faddeev–LeVerrier conversion.              | 2.0 Days         |
| **Step 5: Verification**                    | `proptest` algebraic invariants, host/qemu unit tests, release-codegen check that `evaluate` retains zero panic paths, cubic-spline trajectory validation example per [`vv-standards.md`](../vv-standards.md).        | 2.0 Days         |

---

### 10. References

1. **Rust `polynomial` crate contributors. (2023).** _polynomial_: a no-std
   library for manipulating polynomials (Version 0.2.6). [Online].
   Available: https://lib.rs/crates/polynomial. Accessed: Aug. 7, 2026.
2. **ickk. (2026).** _aberth_: Aberth–Ehrlich simultaneous polynomial
   root-finding (Version 0.4.1). [Online].
   Available: https://docs.rs/aberth/latest/aberth/. Accessed: Aug. 7, 2026.
3. **Horner, W. G. (1819).** A New Method of Solving Numerical Equations of All
   Orders, by Continuous Approximation. _Philosophical Transactions of the Royal
   Society of London_, 109, 308–335. — Origin of the $N-1$ operation evaluation
   scheme; direct FLOP-count justification for `evaluate`.
4. **Higham, N. J. (2002).** _Accuracy and Stability of Numerical Algorithms_,
   2nd ed., Ch. 5 "Polynomials". SIAM. — Backward-error bound $\gamma_{2n}$ for
   Horner's method, quantifying the accuracy claim in §4.6.
5. **Henrici, P. (1974).** _Applied and Computational Complex Analysis, Volume
   1: Power Series, Integration, Conformal Mapping, Location of Zeros_. Wiley. (
   Cited edition: 1988 Wiley Classics Library reprint, ISBN 0471608416.) —
   Zero-location theory underpinning root-finding correctness and region bounds.
6. **Bini, D. A., Boito, P., Eidelman, Y., Gemignani, L., & Gohberg, I. (2010).
   ** A Fast Implicit QR Eigenvalue Algorithm for Companion Matrices. _Linear
   Algebra and its Applications_, 432(8), 2006–2031. — $O(N^2)$-time
   companion-matrix QR exploiting unitary-plus-rank-one structure.
7. **Aurentz, J. L., Mach, T., Vandebril, R., & Watkins, D. S. (2015).** Fast
   and Backward Stable Computation of Roots of Polynomials. _SIAM Journal on
   Matrix Analysis and Applications_, 36(3), 942–973. — Reduces Bini et al.
   (2010) companion-matrix storage from $O(N^2)$ to $O(N)$ while preserving
   backward stability.
8. **Aurentz, J. L., Mach, T., Vandebril, R., & Watkins, D. S. (2018).** Fast
   and Backward Stable Computation of Roots of Polynomials, Part II: Backward
   Error Analysis; Companion Matrix and Companion Pencil. _SIAM Journal on
   Matrix Analysis and Applications_, 1245–1269, doi: 10.1137/17M1152802. —
   Formalizes the distinction between backward stability with respect to the
   companion matrix and backward stability with respect to the polynomial's own
   coefficients (§4.7.1).
9. **Faddeev, D. K., & Faddeeva, V. N. (1963).** _Computational Methods of
   Linear Algebra_. W. H. Freeman and Company. — Trace-based derivation of the
   Faddeev–LeVerrier algorithm; shared with `matrix-design.md` §4.8.1, which
   specifies the inverse (`Matrix` → characteristic `Polynomial`) conversion.
10. **van der Hoeven, J. (2008).** Making fast multiplication of polynomials
    numerically stable. [Online].
    Available: https://www.texmacs.org/joris/stablemult/stablemult.html.
    Accessed: Aug. 7, 2026. — Numerical-stability precondition (
    comparable-magnitude coefficients) for FFT-based polynomial multiplication,
    motivating its rejection in §5.2.
11. **ARM. (2025).** Convolution. _CMSIS-DSP_ (Version 1.15.0). [Online].
    Available: https://arm-software.github.io/CMSIS-DSP/v1.15.0/group__Conv.html.
    Accessed: Aug. 7, 2026. — Long-vector FFT cutoff guidance and Q15/Q31
    fixed-point accumulator/overflow behavior for direct convolution, referenced
    in §5.2.
12. **Claessen, K., & Hughes, J. (2000).** QuickCheck: A Lightweight Tool for
    Random Testing of Haskell Programs. _ACM SIGPLAN Notices_, 35(9), 268–279. —
    Property-based testing methodology framing `proptest` algebraic invariant
    checks (§6.1).

### 11. Revision History

| Revision | Date            | Author          | Description                                                                                                                           |
|:---------|:----------------|:----------------|:--------------------------------------------------------------------------------------------------------------------------------------|
| 1.0      | July 12, 2026   | @MitchellDScott | Initial draft with static array layout and basic polynomial arithmetic.                                                               |
| 1.1      | August 16, 2026 | @MitchellDScott | Storage hierarchy integration: bound polynomial storage to decoupled `DenseStorage` and enabled zero-copy view leaves.                |
| 1.2      | August 19, 2026 | @MitchellDScott | Algorithms & operands: specified Horner evaluation, convolution multiplication, and companion matrix state-space realization.         |
| 1.3      | August 24, 2026 | @MitchellDScott | Generic scalar bounds: generalized arithmetic to `T: Scalar` with complex coefficient support.                                        |
| 1.4      | August 25, 2026 | @MitchellDScott | V&V standardization: aligned test oracles with backward error bounds ($\gamma_{2n}$).                                                 |
| 1.5      | August 26, 2026 | @MitchellDScott | Storage view retarget: updated references to `StorageView`/`StorageViewMut` and `Const<1>` dimensions.                                |
| 1.6      | August 26, 2026 | @MitchellDScott | Trimmed companion/Faddeev–LeVerrier comparison; de-duplicated Bini/Aurentz reference blurbs.                                          |
| 1.7      | August 28, 2026 | @MitchellDScott | Host-scale V&V: clustered-root Horner ($N>50$); umbrella $\tau\kappa\varepsilon$ and Instant timing. Caps unchanged.                 |
| 1.8      | August 28, 2026 | @MitchellDScott | Example crate: clustered-root Horner degree 16 (128-point sweep) and Instant timings. Caps unchanged.                                |
