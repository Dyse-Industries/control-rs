# Polynomial Type (Design Document)

![Date Badge](https://img.shields.io/badge/Date-July_26,_2026-blue)
![Status Badge](https://img.shields.io/badge/Doc%20Status-Draft-orange)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

---

### 1. Introduction

The `Polynomial` type in `control-rs` provides a statically sized,
single-variable polynomial, generic over a `Storage<T, N, U1>` backend of
capacity `N` (maximum representable degree `N - 1`). It backs four concrete
use cases: FIR/IIR filter coefficient representation and evaluation,
Tustin/ZOH discretization of continuous models, cubic/quintic trajectory
generation, and companion-matrix root finding for characteristic polynomials
produced by `Matrix`.

Requiring no heap allocation and no forced stack ownership is a crate-wide
guarantee (see [README](../../README.md)), not something specific to
`Polynomial`. What is specific to this module is the ascending-power
coefficient layout and the degree-aware capacity arithmetic described in the
sections below.

---

### 2. Requirements

#### 2.1 Functional Requirements

- **Compile-Time Sizing**: Enforce polynomial capacity/degree bounds at
  compile time using [num_types](../../src/math/num_types.rs) (`Dim`).
- **Constructors**: Provide `const fn` constructors for common low-degree
  cases (`constant`, `line`) and general capacity (`from_coefficients`), plus
  runtime constructors over borrowed memory (`from_slice`, `from_fn`).
- **Core Arithmetic**: Implement operator overloading for polynomial
  addition, subtraction, and negation (`Add`, `Sub`, `Neg`).
- **Multiplication**: Provide two multiplication paths — `mul_poly` for
  statically-sized, capacity-bound multiplication, and `mul_with_conv` for
  DSP-kernel-backed convolution via the `Convolution<T>` trait.
- **Evaluation**: Evaluate $p(x)$ using Horner's method in exactly $N-1$
  multiply-adds.
- **Division**: Implement `div_rem` to compute quotient and remainder
  polynomials with statically resized capacity bounds.
- **Calculus Operations**: Implement analytical derivative and integral
  methods returning statically resized polynomial bounds.
- **Type Conversions**: Support conversion to a companion `Matrix` (for
  root-finding) and to a rank-1 `Tensor`.

#### 2.2 Non-Functional Requirements

- **Deterministic Execution**: Horner evaluation and `div_rem` must execute in
  a fixed, data-independent number of operations for a given capacity `N`.
- **Zero-Cost Storage Abstraction**: The `Storage<T, N, U1>` abstraction must
  monomorphize and inline without vtables or dynamic dispatch.
- **Vectorization-Friendly Layout**: Contiguous storage backends must allow
  compiler SIMD auto-vectorization over coefficient slices.

#### 2.3 Constraints

- **No-Std Environment**: The code must compile and run in `#![no_std]`
  environments without the Rust standard library (crate-wide rule, see
  [README](../../README.md)).
- **No Dynamic Allocation**: The module must not use a heap allocator; all
  memory allocations must be static or stack-based (crate-wide rule).
- **Coefficient Ordering**: Coefficients are stored in **ascending order of
  powers** (see §3.3 for the full layout rationale). This is the canonical
  statement of the convention; other numerical models that share it (e.g.
  `transfer-function-design.md`) reference this section rather than restating
  it.
- **Capacity Bound**: Maximum polynomial capacity is limited to 128 elements
  (see the crate-level capacity table in [README](../../README.md)).

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
degree is $N - 1$), and `S` defines where the coefficients reside (e.g. stack
`ArrayStorage`, borrowed `MatrixView`, or static Flash memory).

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
where index `i` maps to the coefficient of $x^i$.

- **Ascending Power Storage Rationale**:
    - Direct index-to-exponent mapping: element at index `i` corresponds
      directly to $x^i$.
    - Zero-cost padding: Adding polynomials of differing capacities aligns
      coefficients naturally without element shifting.

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
2. `mul_with_conv`: Decouples arithmetic from representation by leveraging the
   `Convolution<T>` trait and underlying hardware-optimized DSP kernels.

#### 4.3 Core Operations

- **Horner's Method Evaluation**:
  Evaluates $p(x)$ using the recurrence
  relation $p(x) = c_0 + x(c_1 + x(c_2 + \dots))$ directly via storage element
  access (`Storage::get_unchecked`). Minimizes floating-point rounding errors
  and operational count to $N-1$ additions and multiplications (Horner, 1819).
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
- **Calculus Operations**:
  Analytical derivative and integral methods returning statically resized
  polynomial bounds. Zero-location properties and stability bounds underpin
  root-finding correctness (Henrici, 1974).

#### 4.4 Interoperability & Conversions

##### 4.4.1 Companion Matrix Conversion

A monic polynomial of degree $n = N - 1$ converts to its $n \times n$ companion
matrix in Controllable Canonical Form, enabling $O(N^2)$ companion matrix QR
rootfinding algorithms (Bini et al., 2010; Aurentz et al., 2014):

```rust
impl<T, N: Dim, S: Storage<T, N, U1>> TryFrom<Polynomial<T, N, S>>
for Matrix<T, <N as DimSub<U1>>::Output, <N as DimSub<U1>>::Output>
where
    N: DimSub<U1>,
    <N as DimSub<U1>>::Output: Dim,
    T: Zero + One + Copy + Neg<Output=T> + PartialEq,
{
    type Error = ConversionError;
    // ...
}
```

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

---

### 6. Testing & Validation Framework

#### 6.1. Verification Strategy

- **Host/Target Tests**: Unit tests executed on host and qemu targets.
- **Property-Based Testing**: `proptest` validation for
  commutativity ($P+Q=Q+P$), distributivity ($P(Q+R) = PQ + PR$), and division
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

### 7. Performance & Resource Considerations

- **Zero-Cost Abstraction**: Storage abstraction monomorphizes and inlines
  without vtables or dynamic allocation.
- **Vectorization**: Contiguous storage backends enable compiler SIMD
  auto-vectorization over coefficient slices.

---

### 8. References

#### 8.1. Practical

1. **Bini, D. A., Boito, P., Eidelman, Y., Gemignani, L., & Gohberg, I. (2010).
   ** A Fast Implicit QR Eigenvalue Algorithm for Companion Matrices. *Linear
   Algebra and its Applications*, 432(8), 2006–2031. — The $O(N^2)$
   -time / $O(N)$-space companion matrix rootfinding algorithm used for
   characteristic polynomials.
2. **Horner, W. G. (1819).** A New Method of Solving Numerical Equations of All
   Orders, by Continuous Approximation. *Philosophical Transactions of the Royal
   Society of London*, 109, 308–335. — Origin of the $N-1$ operation evaluation
   scheme; direct FLOP-count justification for `evaluate`.
3. **Aurentz, J. L., Mach, T., Vandebril, R., & Watkins, D. S. (2014).** Fast
   and backward stable computation of roots of polynomials. *TW Reports*, KU
   Leuven. — Backward stability analysis and speed trade-offs for companion
   matrix root solvers.

#### 8.2. Theoretical

4. **Henrici, P. (1974).** *Applied and Computational Complex Analysis, Volume
   1*. Wiley. — Zero-location theory underpinning root-finding correctness and
   region bounds.
5. **Faddeev, D. K., & Faddeeva, V. N. (1963).** *Computational Methods of
   Linear Algebra*. W. H. Freeman and Company. — Trace-based derivation of
   Faddeev–LeVerrier algorithm for characteristic polynomial generation.

#### 8.3. Standards, Safety and Verification

6. **Claessen, K., & Hughes, J. (2000).** QuickCheck: A Lightweight Tool for
   Random Testing of Haskell Programs. *ACM SIGPLAN Notices*, 35(9), 268–279. —
   Property-based testing methodology driving `proptest` algebraic invariant
   checks.
7. **Rust Project Developers. (2024).** *The Rustonomicon: The Dark Arts of
   Advanced and Unsafe Rust Programming*. — Memory safety and layout guarantees
   for slice conversions.
8. **ISO. (2018).** *ISO 26262-6:2018 Road vehicles — Functional safety — Part
   6: Product development at the software level*. — Embedded functional safety
   compliance guidelines.
9. **RTCA / EUROCAE. (2011).** *DO-178C: Software Considerations in Airborne
   Systems and Equipment Certification*. — Safety-critical software engineering
   rules.

---

### 9. Revision History

| Date          | Author          | Description                                                                                         |
|:--------------|:----------------|:----------------------------------------------------------------------------------------------------|
| July 12, 2026 | @MitchellDScott | Initial draft with static array layout.                                                             |
| July 26, 2026 | @MitchellDScott | Integrated `Storage<T, N, U1>` trait hierarchy to support borrowed zero-copy views and ROM storage. |
| July 26, 2026 | @MitchellDScott | Added inline academic citations and 3-tiered references section.                                    |
| August 1, 2026 | @MitchellDScott | Restructured section 2 to match the crate-wide Requirements template (Functional/Non-Functional/Constraints), rewrote the introduction with concrete use cases, and made the coefficient-ordering constraint the canonical statement for cross-referencing from other models. |
