# Polynomial Type (Design Document)

![Date Badge](https://img.shields.io/badge/Date-August_2,_2026-blue)
![Status Badge](https://img.shields.io/badge/Doc%20Status-Draft-orange)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

---

### 1. Introduction

The `Polynomial` type in `control-rs` provides a statically sized,
single-variable polynomial, generic over a `Storage<T, N, U1>` backend of
capacity `N` (maximum representable degree `N - 1`). It backs four concrete
use cases: FIR/IIR filter coefficient representation and evaluation,
Tustin/ZOH discretization of continuous models, cubic/quintic trajectory
generation and companion-matrix root finding for characteristic polynomials
produced by `Matrix`.

Requiring no heap allocation and no forced stack ownership is a crate-wide
guarantee (see [README](../../README.md)), not something specific to
`Polynomial`. What is specific to this module is the ascending-power
coefficient layout and the degree-aware capacity arithmetic described in the
sections below.

---

### 2. Requirements

#### 2.1 Functional Requirements

- **FR-1 — Compile-Time Sizing**: Enforce polynomial capacity/degree bounds at
  compile time using `num_types` (`Dim`).
- **FR-2 — Constructors**: Provide `const fn` constructors for low-degree cases
  and general capacity, plus runtime constructors over borrowed memory.
- **FR-3 — Core Arithmetic**: Implement operator overloading for addition,
  subtraction and negation (`Add`, `Sub`, `Neg`).
- **FR-4 — Multiplication**: Provide `mul_poly` (statically-sized) and
  `mul_with_conv` (DSP-kernel convolution via `Convolution<T>`) paths.
- **FR-5 — Evaluation**: Evaluate $p(x)$ using Horner's method in exactly $N-1$
  multiply-adds.
- **FR-6 — Division**: Implement `div_rem` to compute quotient and remainder
  polynomials with statically resized capacity bounds.
- **FR-7 — Calculus Operations**: Implement analytical derivative and integral
  methods returning statically resized polynomial bounds.
- **FR-8 — Type Conversions**: Support conversion to a companion `Matrix` (for
  root-finding) and to a rank-1 `Tensor`.

#### 2.2 Non-Functional Requirements

- **NFR-1 — Deterministic Execution**: Horner evaluation and `div_rem` execute
  in a fixed, data-independent operation count for a given capacity `N`.
- **NFR-2 — Zero-Cost Storage Abstraction**: The `Storage<T, N, U1>` abstraction
  must monomorphize and inline without vtables or dynamic dispatch.
- **NFR-3 — Vectorization-Friendly Layout**: Contiguous storage backends must
  allow compiler SIMD auto-vectorization over coefficient slices.

#### 2.3 Constraints

- **C-1 — No-Std Environment**: The code must compile and run in `#![no_std]`
  environments without the Rust standard library (crate-wide rule).
- **C-2 — No Dynamic Allocation**: The module must not use a heap allocator; all
  memory allocations are static or stack-based.
- **C-3 — Coefficient Ordering**: Coefficients are stored in ascending order of
  powers, a logical indexing rule independent of `Storage`'s physical
  layout ([storage-trait-design.md](../math/storage-trait-design.md) FR-1/FR-6);
  this is the canonical statement other models cross-reference.
- **C-4 — Capacity Bound**: Maximum polynomial capacity is limited to 128
  elements (crate-level capacity table).

---

### 3. Core Architecture & Memory Layout

#### 3.1 Generics Foundation & Storage Strategy

The core `Polynomial` structure decouples mathematical dimensions from physical
storage using the `Storage<T, R, C>` trait hierarchy (with $R = N$
and $C = U1$):

```rust
pub struct Polynomial<T, N: Dim, S: Storage<T, N, U1> = ArrayStorage<T, N, U1>> {
    storage: S,
    _marker: core::marker::PhantomData<N>,
}
```

Here, `N` represents the capacity (number of coefficients, maximum possible
degree is $N - 1$) and `S` defines where the coefficients reside (e.g. stack
`ArrayStorage`, borrowed `MatrixView` or static Flash memory).

#### 3.2 Storage Backends & Zero-Copy Views

By parameterizing `Polynomial` over `Storage<T, N, U1>`, `control-rs` supports
multiple ownership models without duplicating algebraic logic:

```rust
/// Owning polynomial backed by column-major stack array
pub type ArrayPolynomial<T, N> = Polynomial<T, N, ArrayStorage<T, N, U1>>;

/// Zero-copy read-only borrowed polynomial view over &[T]
pub type PolynomialView<'a, T, N> = Polynomial<T, N, MatrixView<'a, T, N, U1>>;

/// Zero-copy mutable borrowed polynomial view over &mut [T]
pub type PolynomialViewMut<'a, T, N> = Polynomial<T, N, MatrixViewMut<'a, T, N, U1>>;
```

#### 3.3 Coefficient Memory Layout

Coefficients are stored in **ascending order of powers**:
$$ p(x) = c_0 + c_1 x + c_2 x^2 + \dots + c_{N-1} x^{N-1} $$
where index `i` maps to the coefficient of $x^i$. This is a logical
convention (`index = degree of term`) that `Polynomial` itself defines and
fully resolves — it says nothing about physical memory layout.

- **Ascending Power Storage Rationale**:
    - Direct index-to-exponent mapping: element at index `i` corresponds
      directly to $x^i$.
    - Zero-cost padding: Adding polynomials of differing capacities aligns
      coefficients naturally without element shifting.
    - **Ecosystem Consistency**: Of the four Rust polynomial crates surveyed
      during research (`polynomial`, `polynomials`, `polynomial-roots`,
      `aberth` — `research/results/polynomial.json`), all four use
      ascending-degree storage. That is a consistent signal across a small,
      non-exhaustive sample, not a claim that every polynomial crate in the
      ecosystem does the same; it supports "idiomatic, not crate-specific"
      without overstating the survey's coverage.

**Physical layout is a separate, already-solved concern.** `Polynomial`
addresses coefficients only through `Storage`'s logical `(i, 0)` interface
(`Storage::get_unchecked`, never a raw offset), so it is agnostic to whichever
concrete backend it is instantiated over — `ArrayStorage`, `MatrixView`/
`MatrixViewMut` or a custom row-major, column-major or ROM-backed
implementor all swap in without changing `Polynomial`'s arithmetic. This is
the same `Storage`-level layout genericity `storage-trait-design.md` FR-6/FR-7
specifies for `Matrix`, applied here without modification. Row-major and
column-major degenerate to the same addressing scheme for a `C = U1`
container, so the distinction that matters in practice is narrower than for
`Matrix` — but the underlying `Storage` abstraction still permits mixing and
matching backends under `Polynomial` to that extent.

#### 3.4 Memory Representation & Slicing

Contiguous slice interfaces are safely exposed when the storage backend
implements `ContiguousStorage` or `ContiguousStorageMut`:

```rust
impl<T, N: Dim, S> Polynomial<T, N, S>
where
    S: ContiguousStorage<T, N, U1>,
{
    pub fn as_slice(&self) -> &[T] {
        self.storage.as_slice()
    }
}

impl<T, N: Dim, S> Polynomial<T, N, S>
where
    S: ContiguousStorageMut<T, N, U1>,
{
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.storage.as_mut_slice()
    }
}
```

---

### 4. API Specification

#### 4.1 Instantiation & Constructors

- `pub const fn constant(val: T) -> Polynomial<T, U1> where T: Copy`: Constructs
  a degree-0 polynomial containing a single coefficient.
- `pub const fn line(c0: T, c1: T) -> Polynomial<T, U2> where T: Copy`:
  Constructs a degree-1 polynomial $[c_0, c_1]$ ($c_0 + c_1 x$).
-

`pub const fn from_coefficients(data: [T; N::DIM]) -> Polynomial<T, N, ArrayStorage<T, N, U1>>`:
Constructs an owning stack polynomial.

- `pub const fn from_storage(storage: S) -> Self`: Constructs a polynomial
  wrapping a custom storage backend `S`.
- `pub fn from_slice(slice: &'a [T]) -> PolynomialView<'a, T, N>`: Constructs a
  borrowed zero-copy view over existing memory.
-

`pub fn from_fn<F>(f: F) -> Polynomial<T, N, ArrayStorage<T, N, U1>> where F: FnMut(usize) -> T`:
Generates coefficients via a mapping closure.

#### 4.2 Operator Overloading

Overloads standard traits (`Add`, `Sub`, `Neg`). Polynomial multiplication
provides two interfaces:

1. `mul_poly`: Static multiplication returning a combined capacity bound:
   ```rust
   impl<T, N: Dim, S1: Storage<T, N, U1>> Polynomial<T, N, S1> {
       pub fn mul_poly<M: Dim, S2: Storage<T, M, U1>>(
           &self,
           other: &Polynomial<T, M, S2>,
       ) -> Polynomial<T, <<N as DimAdd<M>>::Output as DimSub<U1>>::Output, ArrayStorage<T, <<N as DimAdd<M>>::Output as DimSub<U1>>::Output, U1>>
       where
           N: DimAdd<M>,
           <N as DimAdd<M>>::Output: DimSub<U1>,
           <N as DimAdd<M>>::Output: Dim,
           T: Copy + Zero + Add<Output = T> + Mul<Output = T>,
       { /* ... */ }
   }
   ```
   The returned capacity `N + M - 1` is an exact bound, not an approximation:
   ascending-power zero-padding (§3.3) guarantees any coefficient above the
   true product degree evaluates to `T::zero()` rather than a silently wrong
   value.
2. `mul_with_conv`: Decouples arithmetic from representation by leveraging the
   `Convolution<T>` trait ([`src/math/dsp.rs`](../../src/math/dsp.rs)) and
   underlying hardware-optimized DSP kernels. `Convolution<T>` is already
   shipped code, bounded on `T: Real` (the pre-`num-traits-design.md`-pivot
   hierarchy, §9) — this document references the trait as it exists today
   rather than renaming its bound to `Float`; `Real`'s call sites, including
   this one, are subject to the same migration `num-traits-design.md` §4.1
   already flags for `subprograms.rs`'s `AXPY`/`GEMM`, to be resolved once
   that document completes its own research/design pass (§9). The trait's
   current default implementation (`convolve_input`) performs the same
   direct summation, in the same accumulation order, as `mul_poly` — the
   two paths are algorithmically identical today. The separation exists to
   let `mul_with_conv` delegate to a future hardware- or fixed-point-
   specialized `Convolution<T>` implementation without changing `mul_poly`'s
   generic behavior; see §7 and §10 for the numerical implications once such
   a specialization exists.

#### 4.3 Core Operations

- **Horner's Method Evaluation**:
  Evaluates $p(x)$ using the recurrence
  relation $p(x) = c_0 + x(c_1 + x(c_2 + \dots))$ directly via storage element
  access (`Storage::get_unchecked`). Minimizes floating-point rounding errors
  and operational count to $N-1$ additions and multiplications (Horner, 1819).
  The computed result is exact for a polynomial whose coefficients are
  relatively perturbed by at most $\gamma_{2n} = 2nu / (1 - 2nu)$ from $p$'s
  true coefficients, where $u$ is unit roundoff (Higham, 2002, Ch. 5) — a
  small, degree-linear backward-error bound that quantifies the "minimizes
  rounding errors" claim above.
- **Polynomial Division (`div_rem`)**:
  Computes quotient and remainder:
  ```rust
  impl<T, N: Dim, S: Storage<T, N, U1>> Polynomial<T, N, S> {
      pub fn div_rem<M: Dim, Sm: Storage<T, M, U1>, Q: Dim, R: Dim>(
          &self,
          divisor: &Polynomial<T, M, Sm>,
      ) -> Result<(Polynomial<T, Q>, Polynomial<T, R>), DivisionError> { /* ... */ }
  }
  ```
  `DivisionError` covers the hard case (an exactly-zero leading divisor
  coefficient or a degree mismatch), not the soft case: `div_rem`'s repeated
  subtract-and-rescale steps degrade continuously in accuracy as the
  divisor's leading coefficient shrinks relative to its other coefficients,
  the same conditioning mechanism as scalar division by a small number.
  Treated as a documented caveat rather than a distinct error variant,
  consistent with `matrix-design.md`'s handling of near-singular
  factorization inputs (see §5.2).
- **Calculus Operations**:
  Analytical derivative and integral methods returning statically resized
  polynomial bounds. Zero-location properties and stability bounds underpin
  root-finding correctness (Henrici, 1974).

#### 4.4 Interoperability & Conversions

##### 4.4.1 Companion Matrix Conversion

A monic polynomial of degree $n = N - 1$ converts to its $n \times n$ companion
matrix in Controllable Canonical Form, enabling $O(N^2)$-time, $O(N)$-space
companion matrix QR rootfinding (Bini et al., 2010 established the $O(N^2)$
-time algorithm; Aurentz et al., 2015 reduced its storage from $O(N^2)$ to
$O(N)$ while preserving backward stability):

```rust
impl<T, N: Dim, S: Storage<T, N, U1>> TryFrom<Polynomial<T, N, S>>
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

`ConversionError` is defined once, canonically, in
[`error-design.md`](../math/error-design.md) — shared with
`Matrix` and `Tensor`'s conversions — not restated here.

This conversion builds the companion matrix directly from coefficients — a
different, better-conditioned operation than transforming a *general*
state-space system into Controllable Canonical Form via its (often
near-singular) controllability matrix (`state-space-design.md`'s
realization path); the Bini/Aurentz algorithms' structural guarantees
(upper Hessenberg, unitary-plus-rank-one) hold regardless of which
row/column convention places the coefficients, since all such placements
are related by transpose/permutation. Root-finding accuracy is a
matrix-level guarantee, not a coefficient-level one: Aurentz et al.'s
backward-stability result guarantees the computed eigenvalues are exact for
a matrix *near* the companion matrix, not that they are exact for a
polynomial near $p$'s own coefficients (Aurentz et al., 2018) — callers
root-finding on coefficient-sensitive input should read accuracy claims
with that distinction in mind.

The inverse conversion (`Matrix` → characteristic `Polynomial` via the
Faddeev–LeVerrier algorithm) is specified in `matrix-design.md`, which shares
this document's Faddeev & Faddeeva (1963) citation for that reason.

##### 4.4.2 Tensor Conversion

Converts flat coefficient data into a 1D `Tensor<T, Layout>`.

---

### 5. Error Handling & State Management

#### 5.1 Compile-Time Constraints

Capacity mismatches during polynomial arithmetic are rejected at compile time
via Peano type constraints.

#### 5.2 Runtime Error Handling

- Zero division or degree mismatch returns `Result<..., DivisionError>`.
- Bounds checked element access returns `Option<&T>`.
- **Near-Singular Divisor**: `div_rem` accuracy degrades continuously as the
  divisor's leading coefficient shrinks toward (but does not reach) zero — a
  documented conditioning caveat, not a `DivisionError` variant (§4.3).
- **Host/Design-Time Scope**: `div_rem` and companion-matrix root-finding
  have no established fixed-point (Q31/Q15) numerical precedent in DSP
  reference libraries (unlike Horner evaluation and convolution, both of
  which are standard fixed-point DSP primitives). These two operations are
  intended for floating-point, design-time use (e.g. offline controller
  synthesis, coefficient generation), not on-target fixed-point runtime
  paths.
- **Pre-Existing Defect in `mul_with_conv`'s Dependency (not fixed in this
  revision)**: `Convolution::convolve_input` (`src/math/dsp.rs`, already
  shipped) panics via `assert!` when the caller-provided output buffer is
  undersized, which violates the crate's no-panic-outside-tests-and-examples
  rule (`CLAUDE.md`). `mul_with_conv` (§4.2) delegates to it directly. This
  document is design-only and does not modify `src/`; the required fix
  (most likely `assert!` → `debug_assert!`, matching the "Panics (Debug
  only)" convention `GEMV`/`GEMM` already use in `subprograms.rs`) is
  recorded as a required pre-implementation correction to apply at
  `/cr-implement` time, not deferred indefinitely — see §9.

---

### 6. Testing & Validation Framework

#### 6.1. Verification Strategy

- **Host/Target Tests**: Unit tests executed on host and qemu targets.
- **Property-Based Testing**: `proptest` validation for
  commutativity ($P+Q=Q+P$), distributivity ($P(Q+R) = PQ + PR$) and division
  invariants ($P = QD + R$), adopting the random generation methodology of
  QuickCheck (Claessen & Hughes, 2000).

#### 6.2. Validation Strategy

- **Cubic Spline Trajectory Generation**:
  For robotics and CNC path planning, smooth motion paths are often generated
  using cubic splines. This example uses the `Polynomial` type to store a
  pre-computed cubic trajectory and efficiently evaluates the robot's position
  at a specific time step $t$ using Horner's method.

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

### 7. Alternatives

- **Aberth–Ehrlich Simultaneous Iteration (rejected for root-finding)**: The
  Rust `aberth` crate demonstrates a viable, `no_std`, array-backed
  alternative to companion-matrix QR eigenvalue root-finding, using
  simultaneous Aberth–Ehrlich iteration (cubic convergence for simple roots).
  It was not chosen because its convergence rate is data-dependent (cubic for
  simple roots, only linear for multiple or tightly clustered roots),
  violating §2.2's Deterministic Execution non-functional requirement. The
  companion-matrix QR approach (Bini et al., 2010; Aurentz et al., 2015)
  keeps root-finding structurally consistent with the rest of the crate's
  fixed-operation-count posture.
- **FFT-Based Polynomial Multiplication (rejected
  for `mul_poly`/`mul_with_conv`)**:
  Asymptotically faster than direct $O(N \times M)$ summation for large
  degree, but numerically stable only when both operands' coefficients are
  of comparable magnitude (van der Hoeven) — an assumption this crate cannot
  make about arbitrary user-supplied coefficients. This is consistent with
  CMSIS-DSP's own guidance that direct convolution, not an FFT-based
  approach, is appropriate below its documented long-vector cutoff, which
  comfortably covers this crate's 128-element capacity ceiling (§2.3).
- **Single Unified Multiplication Method (rejected)**: Merging `mul_poly` and
  `mul_with_conv` into one method was considered, since they are currently
  algorithmically identical (§4.2). They are kept separate so that
  `mul_with_conv` alone can later delegate to a hardware- or
  fixed-point-specialized `Convolution<T>` implementation without changing
  `mul_poly`'s own, strictly broader bound (`T: Copy + Zero + Add<Output=T>
    + Mul<Output=T>`, §4.2 — no `Real`/`Float` required, so `mul_poly` already
  works for fixed-point and integer `T` today) or requiring downstream
  callers of `mul_poly` to opt into `Convolution<T>`'s narrower, `Real`-only
      specialization.

---

### 8. Performance & Resource Considerations

- **Zero-Cost Abstraction**: Storage abstraction monomorphizes and inlines
  without vtables or dynamic allocation.
- **Vectorization**: Contiguous storage backends enable compiler SIMD
  auto-vectorization over coefficient slices.

---

### 9. Risks & Open Questions

- **Horner Evaluation Fixed-Point Renormalization**: Unlike Matrix
  multiply-accumulate or DSP convolution (both of which can use a single
  wide accumulator truncated once at the end), Horner's recurrence feeds each
  step's result into the next step's multiplicand, requiring a Q-format
  rescale after *every* multiply-add rather than once per operation. No
  CMSIS-DSP or equivalent fixed-point reference implementation for
  Horner-style evaluation was found during research. This is an open
  question distinct from and not resolved by, the wide-accumulator
  convention already adopted for Matrix in `matrix-design.md` §7.
- **`div_rem` / Root-Finding Fixed-Point Scope**: Should the host/design-time
  scoping in §5.2 be stated as a hard constraint (compile-time bound to
  floating-point `T`) or left as a documentation-only recommendation pending
  a concrete fixed-point use case?
- **`Convolution::convolve_input` Panic Path**: Must be corrected
  (`assert!` → `debug_assert!`) before or during `/cr-implement`; tracked
  here rather than silently assumed away, since this document does not
  touch `src/` (§5.2).
- **`num-traits-design.md` Dependency Is Provisional**: `Convolution<T:
  Real>` (§4.2) references shipped code's current, pre-pivot bound; once
  `num-traits-design.md` completes its own `/cr-research`/`/cr-design-doc`
  pass (it has not yet had either — see that document's own status), this
  call site and any others bound on the retired `Ring`/`Field`/`Real`
  hierarchy will need a follow-up pass.

---

### 10. Development Plan

| Task / Feature                          | Description                                                                                                                                  | Estimated Effort |
|:----------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
| Step 1: Storage & Constructors          | `Polynomial<T, N, S>` struct, `ArrayPolynomial`/`PolynomialView`/`PolynomialViewMut` aliases, §4.1 constructors.                             | 1.5 Days         |
| Step 2: Core Arithmetic                 | `Add`/`Sub`/`Neg` operator overloads, `mul_poly`, `mul_with_conv` via `Convolution<T>`.                                                      | 2.0 Days         |
| Step 3: Evaluation, Calculus & Division | Horner `evaluate`, derivative/integral methods, `div_rem` with `DivisionError` and the near-singular caveat.                                 | 2.5 Days         |
| Step 4: Interoperability                | Companion-`Matrix` `TryFrom` conversion, `Tensor` conversion, cross-check against `matrix-design.md`'s reverse Faddeev–LeVerrier conversion. | 1.5 Days         |
| Step 5: Verification                    | `proptest` algebraic invariants, host/qemu unit tests, cubic-spline trajectory validation example.                                           | 1.5 Days         |

---

### 11. References

#### 11.1. Practical

1. **Bini, D. A., Boito, P., Eidelman, Y., Gemignani, L., & Gohberg, I. (2010).
   ** A Fast Implicit QR Eigenvalue Algorithm for Companion Matrices. *Linear
   Algebra and its Applications*, 432(8), 2006–2031. — Establishes the
   $O(N^2)$-time companion matrix rootfinding algorithm (using $O(N^2)$
   storage) for characteristic polynomials.
2. **Horner, W. G. (1819).** A New Method of Solving Numerical Equations of All
   Orders, by Continuous Approximation. *Philosophical Transactions of the Royal
   Society of London*, 109, 308–335. — Origin of the $N-1$ operation evaluation
   scheme; direct FLOP-count justification for `evaluate`.
3. **Aurentz, J. L., Mach, T., Vandebril, R., & Watkins, D. S. (2015).** Fast
   and Backward Stable Computation of Roots of Polynomials. *SIAM Journal on
   Matrix Analysis and Applications*, 36(3), 942–973. — Reduces the Bini et
   al. (2010) companion matrix rootfinder's storage from $O(N^2)$ to $O(N)$
   while preserving backward stability, via a Givens-rotator-plus-rank-one
   representation.
4. **Aurentz, J. L., Mach, T., Vandebril, R., & Watkins, D. S. (2018).** Fast
   and Backward Stable Computation of Roots of Polynomials, Part II: Backward
   Error Analysis; Companion Matrix and Companion Pencil. *SIAM Journal on
   Matrix Analysis and Applications*. — Formalizes the distinction between
   backward stability with respect to the companion matrix and backward
   stability with respect to the polynomial's own coefficients (§4.4.1).

#### 11.2. Theoretical

5. **Higham, N. J. (2002).** *Accuracy and Stability of Numerical Algorithms*,
   2nd ed., Chapter 5 "Polynomials". SIAM. — Backward-error bound
   $\gamma_{2n}$ for Horner's method, quantifying the accuracy claim in §4.3.
6. **Henrici, P. (1974).** *Applied and Computational Complex Analysis, Volume
   1*. Wiley. — Zero-location theory underpinning root-finding correctness and
   region bounds.
7. **Faddeev, D. K., & Faddeeva, V. N. (1963).** *Computational Methods of
   Linear Algebra*. W. H. Freeman and Company. — Trace-based derivation of
   the Faddeev–LeVerrier algorithm; shared with `matrix-design.md`, which
   specifies the inverse (`Matrix` → characteristic `Polynomial`) conversion.

#### 11.3. Standards, Safety and Verification

8. **Claessen, K., & Hughes, J. (2000).** QuickCheck: A Lightweight Tool for
   Random Testing of Haskell Programs. *ACM SIGPLAN Notices*, 35(9), 268–279. —
   Property-based testing methodology driving `proptest` algebraic invariant
   checks.
9. **Rust Project Developers. (2024).** *The Rustonomicon: The Dark Arts of
   Advanced and Unsafe Rust Programming*. — Memory safety and layout guarantees
   for slice conversions.
10. **ISO. (2018).** *ISO 26262-6:2018 Road vehicles — Functional safety — Part
    6: Product development at the software level*. — Embedded functional safety
    compliance guidelines.
11. **RTCA / EUROCAE. (2011).** *DO-178C: Software Considerations in Airborne
    Systems and Equipment Certification*. — Safety-critical software engineering
    rules.

---

### 12. Revision History

| Revision | Date           | Author          | Description                                                                                                                           |
|:---------|:---------------|:----------------|:--------------------------------------------------------------------------------------------------------------------------------------|
| 1.0      | July 12, 2026  | @MitchellDScott | Initial draft with static array layout.                                                                                               |
| 1.1      | July 26, 2026  | @MitchellDScott | Integrated `Storage` trait hierarchy to support borrowed zero-copy views and ROM storage.                                             |
| 1.2      | July 26, 2026  | @MitchellDScott | Added inline academic citations and 3-tiered references section.                                                                      |
| 1.3      | August 1, 2026 | @MitchellDScott | Restructured Requirements to the crate-wide template; made coefficient ordering the canonical cross-referenced statement.             |
| 1.4      | August 2, 2026 | @MitchellDScott | Added citations; clarified companion-form conditioning; documented `div_rem` caveats; added Alternatives, Risks and Development Plan. |
| 1.5      | August 2, 2026 | @MitchellDScott | Propagated `num-traits-design.md` pivot to companion-matrix bound; relocated `ConversionError`; corrected a factual error.            |
| 1.6      | August 2, 2026 | @MitchellDScott | Separated coefficient-ordering convention from `Storage` layout; cross-referenced `storage-trait-design.md` instead of restating it.  |
