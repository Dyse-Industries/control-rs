# State-Space Model Type (Design Document)

![Date Badge](https://img.shields.io/badge/Date-July_26,_2026-blue)
![Status Badge](https://img.shields.io/badge/Doc%20Status-Draft-orange)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

---

### 1. Introduction

The `StateSpace` module provides a statically sized, type-safe representation of
continuous-time and discrete-time linear time-invariant (LTI) state-space models
for control systems engineering, signal processing, and state estimation (e.g.,
Kalman filtering, LQR synthesis, and observer design).

A continuous-time LTI state-space system is governed by:
$$\dot{x}(t) = A x(t) + B u(t)$$
$$y(t) = C x(t) + D u(t)$$

A discrete-time LTI state-space system is governed by:
$$x[k+1] = A x[k] + B u[k]$$
$$y[k] = C x[k] + D u[k]$$

where $x \in \mathbb{R}^{N_x}$ is the state vector, $u \in \mathbb{R}^{N_u}$ is
the input vector, $y \in \mathbb{R}^{N_y}$ is the output vector,
and $A, B, C, D$ are system, input, output, and feedforward matrices of
sizes $(N_x \times N_x)$, $(N_x \times N_u)$, $(N_y \times N_x)$,
and $(N_y \times N_u)$ respectively.

Following the design philosophy established by `TransferFunction` and `Matrix`,
`StateSpace` is a **standalone, generic container** built directly on top of
four generic storage backends (`Sa`, `Sb`, `Sc`, `Sd`) implementing the
`Storage` trait, rather than storing four `Matrix<T, R, C, S>` fields
directly. Avoiding heap allocation is the crate-wide `no_alloc` rule (see
[README](../../README.md)), not a `StateSpace`-specific decision; what is
specific here is *why* raw storage backends are used instead of full `Matrix`
wrappers: it avoids redundant type-wrapper parameters on every field, and it
lets each matrix ($A$, $B$, $C$, $D$) opt into a different storage strategy
independently — for example a stack-allocated $A$ alongside a
zero/identity-view $D$ — without changing the `StateSpace` signature (see §6,
Alternative C). `StateSpace` still leverages the high-level `Matrix` type and
zero-copy `MatrixView` wrappers to execute linear algebra operations safely
and conveniently, while retaining direct access to lower-level Peano
dimension traits (`Dim`), storage traits, and BLAS kernels.

This architecture achieves:

1. **Zero Dynamic Allocation (`#![no_std]`)**: Storage backends can be inline
   arrays, static ROM tables, or borrowed slice views.
2. **Heterogeneous Storage**: Each matrix ($A, B, C, D$) can utilize a distinct
   storage type (e.g., stack-allocated `ArrayStorage` for $A$, and a
   zero/identity virtual storage view for $D$).
3. **Safe High-Level Algebra**: Exposes matrix views (`MatrixView` /
   `MatrixViewMut`) and leverages `Matrix` operations for state propagation,
   system interconnection, and linear transformations.
4. **Compile-Time Dimension Safety**: Enforces matrix dimension
   compatibility ($N_x, N_u, N_y$) at compile time using Peano arithmetic (
   `DimAdd`, `DimSub`, `DimMul`).

---

### 2. Requirements

#### 2.1 Functional Requirements

##### FR-1: Decoupled Storage Parameterization

The `StateSpace` type must accept four independent storage backends for
matrices $A, B, C, D$:

```rust
pub struct StateSpace<
    T,
    NX: Dim,
    NU: Dim,
    NY: Dim,
    Sa: Storage<T, NX, NX>,
    Sb: Storage<T, NX, NU>,
    Sc: Storage<T, NY, NX>,
    Sd: Storage<T, NY, NU>,
> {}
```

##### FR-2: Domain Encoding (Continuous vs. Discrete)

The type must encode domain information (continuous $s$-domain vs. discrete $z$
-domain) via an optional sampling period `sample_time: Option<T>`.

##### FR-3: Interoperability with `Matrix` Type & BLAS Kernels

High-level operations (state propagation, matrix multiplication, coordinate
transformations) must be capable of utilizing the `Matrix` abstraction (e.g.,
via `MatrixView` or temporary `Matrix` wrappers) while underlying data access
maps directly to `Storage` traits and BLAS level 1/2/3 subprograms (`GEMV`,
`GEMM`, `AXPY`).

##### FR-4: Time-Domain State Propagation

- **Discrete Step**: Advance discrete state vector $x[k+1] = A x[k] + B u[k]$
  and calculate output $y[k] = C x[k] + D u[k]$.
- **Continuous Derivative**: Evaluate state
  derivative $\dot{x}(t) = A x(t) + B u(t)$ and output $y(t) = C x(t) + D u(t)$.

##### FR-5: System Interconnections

Provide compile-time dimension-checked functions for:

- **Series (Cascade)**: $S_2 \circ S_1$ combining
  states $x = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}$.
- **Parallel**: $S_1 + S_2$ combining outputs $y = y_1 + y_2$.
- **Feedback (Closed-Loop)**: Negative/positive feedback around systems with $D$
  -matrix feedforward resolution.

##### FR-6: Discretization & Continuous Transformations

- **Zero-Order Hold (ZOH)**: Matrix
  exponential $A_d = e^{A T_s}$, $B_d = \left(\int_0^{T_s} e^{A \tau} d\tau\right) B$.
- **Bilinear (Tustin) Transform**: Discretization with optional frequency
  pre-warping.

##### FR-7: Canonical Form Transformations & Structural Analysis

- Conversions to/from Controllable Canonical Form (CCF) and Observable Canonical
  Form (OCF).
- Similarity transformations $A' = T A T^{-1}$, $B' = T B$, $C' = C T^{-1}$.
- Computation of Controllability Matrix $\mathcal{C}$ and Observability
  Matrix $\mathcal{O}$.

##### FR-8: Transfer Function Conversion

Bidirectional conversion between SISO/MIMO `StateSpace` models and
`TransferFunction` models ($H(s) = C (sI - A)^{-1} B + D$).

#### 2.2 Non-Functional Requirements

##### NFR-1: Zero Dynamic Allocation (`#![no_std]`, `no_alloc`)

All operations must execute deterministically without relying on heap
allocation (`Vec`, `Box`).

##### NFR-2: Zero-Cost Abstraction

Storage abstraction and Peano dimension bounds must monomorphize completely,
matching hand-optimized raw array operations.

##### NFR-3: Contiguity-Gated Slice & Matrix Accessors

Safe slice and raw-matrix view accessors (`a_slice()`, `a_matrix()`, etc.) must
be exposed if and only if the underlying storage backends implement
`ContiguousStorage`.

#### 2.3 Constraints

- State dimension `NX`, input dimension `NU`, and output dimension `NY` must
  satisfy `Dim` bounds (`NX >= 1`, `NU >= 1`, `NY >= 1`).
- Matrices must conform to strict physical layout rules:
    - $A$: `NX` $\times$ `NX`
    - $B$: `NX` $\times$ `NU`
    - $C$: `NY` $\times$ `NX`
    - $D$: `NY` $\times$ `NU`

---

### 3. Technical Overview

`StateSpace` acts as a domain-aware state-space container over generic matrix
storage backends `Sa`, `Sb`, `Sc`, `Sd`.

It integrates cleanly with existing `control-rs` modules:

- **`crate::math::num_types`**: Peano arithmetic (`Dim`, `DimAdd`, `DimSub`,
  `DimMul`, `U1`, etc.) for compile-time shape verification.
- **`crate::math::storage`**: Storage traits (`Storage`, `StorageMut`,
  `ContiguousStorage`, `ContiguousStorageMut`).
- **`crate::math::matrix`**: `Matrix<T, R, C, S>`, `MatrixView<'a, T, R, C>`,
  `MatrixViewMut<'a, T, R, C>`, `MatrixSlice<'a, T, R, C>`, and
  `MatrixSliceMut<'a, T, R, C>` for safe, high-level matrix operations.
- **`crate::math::subprograms`**: BLAS Level 1/2/3 kernels (`GEMV`, `GEMM`,
  `AXPY`, `TRSM`).

---

### 4. Core Architecture

#### 4.1 Type Signature & Storage Layout

```rust
pub struct StateSpace<
    T,
    NX: Dim,
    NU: Dim,
    NY: Dim,
    Sa: Storage<T, NX, NX> = ArrayStorage<T, NX, NX>,
    Sb: Storage<T, NX, NU> = ArrayStorage<T, NX, NU>,
    Sc: Storage<T, NY, NX> = ArrayStorage<T, NY, NX>,
    Sd: Storage<T, NY, NU> = ArrayStorage<T, NY, NU>,
> {
    a_storage: Sa,
    b_storage: Sb,
    c_storage: Sc,
    d_storage: Sd,
    sample_time: Option<T>, // None = Continuous (s-domain), Some(Ts) = Discrete (z-domain)
    _marker: core::marker::PhantomData<(NX, NU, NY)>,
}
```

- **State Matrix Storage (`Sa`)**: Holds $N_x \times N_x$ elements for $A$.
- **Input Matrix Storage (`Sb`)**: Holds $N_x \times N_u$ elements for $B$.
- **Output Matrix Storage (`Sc`)**: Holds $N_y \times N_x$ elements for $C$.
- **Feedforward Matrix Storage (`Sd`)**: Holds $N_y \times N_u$ elements
  for $D$.
- **Sampling Time (`sample_time`)**: `None` specifies continuous-time
  state-space; `Some(Ts)` specifies discrete-time state-space with period $T_s$.

#### 4.2 Storage Backends & Convenient Type Aliases

```rust
/// Owning state-space model with stack-allocated arrays
pub type ArrayStateSpace<T, NX, NU, NY> = StateSpace<
    T,
    NX,
    NU,
    NY,
    ArrayStorage<T, NX, NX>,
    ArrayStorage<T, NX, NU>,
    ArrayStorage<T, NY, NX>,
    ArrayStorage<T, NY, NU>,
>;

/// Zero-copy borrowed read-only state-space view over &[T] slices
pub type StateSpaceView<'a, T, NX, NU, NY> = StateSpace<
    T,
    NX,
    NU,
    NY,
    MatrixView<'a, T, NX, NX>,
    MatrixView<'a, T, NX, NU>,
    MatrixView<'a, T, NY, NX>,
    MatrixView<'a, T, NY, NU>,
>;

/// Zero-copy borrowed mutable state-space view over &mut [T] slices
pub type StateSpaceViewMut<'a, T, NX, NU, NY> = StateSpace<
    T,
    NX,
    NU,
    NY,
    MatrixViewMut<'a, T, NX, NX>,
    MatrixViewMut<'a, T, NX, NU>,
    MatrixViewMut<'a, T, NY, NX>,
    MatrixViewMut<'a, T, NY, NU>,
>;
```

#### 4.3 Safe `Matrix` Integration & Zero-Copy Matrix Views

While data is stored inside storage backends (`Sa`, `Sb`, `Sc`, `Sd`),
`StateSpace` exposes methods to treat each backend as a zero-cost `MatrixView`
or `MatrixViewMut`, enabling full reuse of `Matrix` operations (multiplication,
addition, transposition, solver routines):

```rust
impl<T, NX: Dim, NU: Dim, NY: Dim, Sa, Sb, Sc, Sd> StateSpace<T, NX, NU, NY, Sa, Sb, Sc, Sd>
where
    Sa: Storage<T, NX, NX>,
    Sb: Storage<T, NX, NU>,
    Sc: Storage<T, NY, NX>,
    Sd: Storage<T, NY, NU>,
{
    /// Exposes system matrix A as a high-level Matrix view.
    pub fn a_matrix(&self) -> MatrixSlice<'_, T, NX, NX> {
        MatrixSlice::from_view(MatrixView::from_storage(&self.a_storage))
    }

    /// Exposes input matrix B as a high-level Matrix view.
    pub fn b_matrix(&self) -> MatrixSlice<'_, T, NX, NU> {
        MatrixSlice::from_view(MatrixView::from_storage(&self.b_storage))
    }

    /// Exposes output matrix C as a high-level Matrix view.
    pub fn c_matrix(&self) -> MatrixSlice<'_, T, NY, NX> {
        MatrixSlice::from_view(MatrixView::from_storage(&self.c_storage))
    }

    /// Exposes feedforward matrix D as a high-level Matrix view.
    pub fn d_matrix(&self) -> MatrixSlice<'_, T, NY, NU> {
        MatrixSlice::from_view(MatrixView::from_storage(&self.d_storage))
    }
}
```

---

### 5. API Specification & Operations

#### 5.1 Constructors

- **From Storage**:
  `pub const fn from_storage(a: Sa, b: Sb, c: Sc, d: Sd, sample_time: Option<T>) -> Self`
- **Owning Array Constructor**:
  `pub fn from_arrays(a: [T; NX::DIM * NX::DIM], b: [T; NX::DIM * NU::DIM], c: [T; NY::DIM * NX::DIM], d: [T; NY::DIM * NU::DIM], sample_time: Option<T>) -> ArrayStateSpace<T, NX, NU, NY>`
- **Slice View Constructor**:
  `pub fn from_slices(a: &'a [T], b: &'a [T], c: &'a [T], d: &'a [T], sample_time: Option<T>) -> StateSpaceView<'a, T, NX, NU, NY>`

#### 5.2 State Propagation & Time-Domain Simulation

##### Discrete State Step ($x[k+1] = A x[k] + B u[k]$)

Given current state vector $x \in \mathbb{R}^{N_x}$ and input
vector $u \in \mathbb{R}^{N_u}$, compute next state $x_{next}$ and
output $y \in \mathbb{R}^{N_y}$:

```rust
impl<T, NX: Dim, NU: Dim, NY: Dim, Sa, Sb, Sc, Sd> StateSpace<T, NX, NU, NY, Sa, Sb, Sc, Sd>
where
    Sa: Storage<T, NX, NX>,
    Sb: Storage<T, NX, NU>,
    Sc: Storage<T, NY, NX>,
    Sd: Storage<T, NY, NU>,
{
    /// Evaluates one discrete step update: x_next = A*x + B*u, y = C*x + D*u
    pub fn step<Sx, Su>(
        &self,
        x: &Matrix<T, NX, U1, Sx>,
        u: &Matrix<T, NU, U1, Su>,
    ) -> (ArrayMatrix<T, NX, U1>, ArrayMatrix<T, NY, U1>)
    where
        Sx: Storage<T, NX, U1>,
        Su: Storage<T, NU, U1>,
        T: Copy + Zero + Add<Output=T> + Mul<Output=T>,
    {
        let x_next = self.a_matrix() * x + self.b_matrix() * u;
        let y = self.c_matrix() * x + self.d_matrix() * u;
        (x_next, y)
    }
}
```

##### Continuous State Derivative ($\dot{x} = A x + B u$)

Evaluates state derivative $\dot{x}(t)$ for numerical integration routines (
e.g., Runge-Kutta 4th Order).

#### 5.3 System Interconnections

##### Series (Cascade) Connection

Connecting output of System 1 ($N_{x1}, N_u, N_y$) to input of System
2 ($N_{x2}, N_y, N_{out}$):
Total state size: $N_x = N_{x1} + N_{x2}$.

$$\begin{bmatrix} \dot{x}_1 \\ \dot{x}_2 \end{bmatrix} = \begin{bmatrix} A_1 & 0 \\ B_2 C_1 & A_2 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} + \begin{bmatrix} B_1 \\ B_2 D_1 \end{bmatrix} u_1$$
$$y = \begin{bmatrix} D_2 C_1 & C_2 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} + D_2 D_1 u_1$$

##### Parallel Connection

Connecting two systems with identical input and output
dimensions ($N_{x1} + N_{x2}$ states):
$$A_{par} = \begin{bmatrix} A_1 & 0 \\ 0 & A_2 \end{bmatrix}, \quad B_{par} = \begin{bmatrix} B_1 \\ B_2 \end{bmatrix}, \quad C_{par} = \begin{bmatrix} C_1 & C_2 \end{bmatrix}, \quad D_{par} = D_1 + D_2$$

##### Feedback Connection

Closed-loop negative feedback around system $S_1$ and controller/feedback
system $S_2$.

#### 5.4 Discretization (ZOH & Bilinear)

Converts a continuous `StateSpace` model ($T_s = \text{None}$) into a discrete
`StateSpace` model ($T_s = \text{Some}(T_s)$):

- **ZOH Discretization**: Form augmented
  matrix $M = \begin{bmatrix} A & B \\ 0 & 0 \end{bmatrix} \in \mathbb{R}^{(N_x + N_u) \times (N_x + N_u)}$
  and compute matrix
  exponential $e^{M T_s} = \begin{bmatrix} A_d & B_d \\ 0 & I \end{bmatrix}$ via
  scaling-and-squaring with Padé approximation (Moler & Van Loan, 2003).
- **Tustin Discretization**: Uses matrix inversion / triangular solver
  for $(I - \frac{T_s}{2} A)^{-1}$ (Ogata, 2010).

#### 5.5 Canonical Transformations & Transfer Function Equivalences

- **Similarity Transformation**: Given state transformation
  matrix $T \in \mathbb{R}^{N_x \times N_x}$ (Åström & Murray, 2021):
  $$A' = T A T^{-1}, \quad B' = T B, \quad C' = C T^{-1}, \quad D' = D$$
- **Controllability Matrix**:
  $$\mathcal{C} = \begin{bmatrix} B & AB & A^2B & \dots & A^{N_x-1}B \end{bmatrix}$$
  Determines system controllability (Kailath, 1980).
- **Observability Matrix**:
  $$\mathcal{O} = \begin{bmatrix} C \\ CA \\ CA^2 \\ \vdots \\ CA^{N_x-1} \end{bmatrix}$$
  Determines system observability (Kailath, 1980).
- **Transfer Function Conversion**:
  $$H(s) = C (s I - A)^{-1} B + D = \frac{C \text{adj}(sI - A) B + D \det(sI - A)}{\det(sI - A)}$$
  Evaluated using Hessenberg reduction and triangular solves rather than
  explicit inversion to preserve numerical stability (Golub & Van Loan, 2013).

---

### 6. Alternatives

| Alternative                                             | Description                                                                           | Pros                                                                                                   | Cons                                                                                   |
|:--------------------------------------------------------|:--------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------|
| **A. Monolithic Storage Field**                         | Storing a single combined block matrix $\begin{bmatrix} A & B \\ C & D \end{bmatrix}$ | Single contiguous allocation                                                                           | Inflexible when $D$ matrix is zero/identity view, awkward individual slice borrowing   |
| **B. Wrapping 4 `Matrix` Structs Directly**             | Storing `a: Matrix<T, NX, NX, Sa>`, `b: Matrix...`                                    | Built-in matrix methods on fields                                                                      | Redundant type wrapper parameters, extra boilerplate vs storing raw `Storage` backends |
| **C. Generic Storage Backends + `MatrixView` (Chosen)** | Storing raw `Sa, Sb, Sc, Sd` backends and creating `MatrixView` wrappers on demand    | Maximum flexibility (`#![no_std]`, ROM, stack, zero-matrix storage), symmetric with `TransferFunction` | Requires `a_matrix()` accessors for high-level matrix algebra                          |

---

### 7. Verification & Validation

#### 7.1. Verification Strategy

1. **Unit Testing (`src/state_space/tests/`)**:
    - Verify step response of known continuous and discrete systems against
      analytical solutions.
    - Verify similarity transformations preserve transfer function frequency
      response $H(j\omega)$.
2. **Property-Based Testing (`proptest`)**:
    - Verify series and parallel interconnections maintain algebraic identity
      equivalence using QuickCheck random generator principles (Claessen &
      Hughes, 2000).
    - Verify discretization followed by continuous approximation converges
      as $T_s \to 0$.
3. **HIL Verification (`control-rs-hil`)**:
    - Test real-time execution of discrete state
      propagation $x[k+1] = A x[k] + B u[k]$ on microcontroller targets (
      Cortex-M) within deterministic deadline bounds (DO-178C, ISO 26262).

#### 7.2. Validation Strategy

- **Kinematic Object Tracking**:
  A practical use of a discrete `StateSpace` model is predicting the future
  position and velocity of an object given its acceleration. This models the
  kinematic system $x[k+1] = A x[k] + B u[k]$ where $x$ contains position and
  velocity, and $u$ is the acceleration input.

  ```rust
  use control_rs::state_space::{StateSpace, ArrayStateSpace};
  use control_rs::math::num_types::{U2, U1};
  use control_rs::math::matrix::{Matrix, ArrayMatrix};

  /// Instantiates a 1D kinematic tracking model (Position, Velocity) and predicts the next state.
  pub fn predict_next_kinematic_state(
      current_state: &Matrix<f32, U2, U1>, 
      acceleration: f32,
      dt: f32
  ) -> ArrayMatrix<f32, U2, U1> {
      // 1. Define the kinematic matrices for a given time step `dt`
      // A = [1, dt; 0, 1], B = [0.5 * dt^2; dt], C = [1, 0], D = [0]
      let sys: ArrayStateSpace<f32, U2, U1, U1> = StateSpace::from_arrays(
          [1.0, 0.0, dt, 1.0],         // A matrix (Column-major layout)
          [0.5 * dt * dt, dt],         // B matrix
          [1.0, 0.0],                  // C matrix (Extracts position)
          [0.0],                       // D matrix
          Some(dt)                     // Discrete system
      );

      // 2. Format the input vector u[k]
      let mut u_k = Matrix::<f32, U1, U1>::zero();
      u_k.as_mut_slice()[0] = acceleration;

      // 3. Propagate the state forward one step
      let (x_next, _y) = sys.step(current_state, &u_k);
      x_next
  }
  ```

---

### 8. Performance & Resource Considerations

- **Stack Allocation Limits**: Large state vectors (e.g., $N_x = 32$)
  require $32 \times 32 = 1024$ elements for matrix $A$. Storing via `Storage`
  enables static buffer placement (`StaticStorage`) or borrowed heap views (
  `HeapStorage` when `alloc` enabled), preventing embedded stack overflow.
- **Matrix Exponential Performance**: ZOH discretization of $A$ uses
  scaling-and-squaring Padé approximation using BLAS level 3 `GEMM` kernels (
  Moler & Van Loan, 2003).

---

### 9. Risks & Open Questions

> [!IMPORTANT]
> **Matrix Inversion & Numerical Stability**
> Explicit matrix inversion for transfer function
> conversion ($C(sI-A)^{-1}B + D$) or Tustin transformation can be
> ill-conditioned
> for high state dimensions ($N_x > 10$). Implementation should prefer QR /
> Hessenberg decomposition or LU triangular solvers (`TRSM`) over direct matrix
> inversion (Golub & Van Loan, 2013).

> [!NOTE]
> **Sparse and Structured Storage Optimization**
> Many real-world systems have sparse $A$ matrices (e.g., companion form or
> tridiagonal systems) or zero $D$ matrices. The `Storage` trait
> parameterization
> allows introducing `ZeroStorage` or `SparseStorage` backends in the future
> without changing the `StateSpace` type signature.

---

### 10. References

#### 10.1. Practical

1. **Moler, C., & Van Loan, C. (2003).** Nineteen Dubious Ways to Compute the
   Exponential of a Matrix, Twenty-Five Years Later. *SIAM Review*, 45(1),
   3–49. — Comparative complexity and accuracy analysis justifying
   scaling-and-squaring + Padé approximation over the augmented block matrix.
2. **Golub, G. H., & Van Loan, C. F. (2013).** *Matrix Computations* (4th ed.).
   Johns Hopkins University Press. — Complexity basis for using triangular
   solves and Hessenberg reduction instead of direct matrix inversion when
   evaluating $H(s)=C(sI-A)^{-1}B+D$.

#### 10.2. Theoretical

3. **Kailath, T. (1980).** *Linear Systems*. Prentice-Hall. — Definitional
   source for controllability/observability matrices and canonical realizations.
4. **Ogata, K. (2010).** *Modern Control Engineering* (5th ed.). Prentice
   Hall. — LTI state-space formulation and block-diagram interconnection
   algebra.
5. **Åström, K. J., & Murray, R. M. (2021).** *Feedback Systems: An Introduction
   for Scientists and Engineers* (2nd ed.). Princeton University Press. —
   Similarity transformations and closed-loop feedback derivations.

#### 10.3. Standards, Safety and Verification

6. **Claessen, K., & Hughes, J. (2000).** QuickCheck: A Lightweight Tool for
   Random Testing of Haskell Programs. *ACM SIGPLAN Notices*, 35(9), 268–279. —
   Property-based testing methodology driving `proptest` suites.
7. **Rust Project Developers. (2024).** *The Rustonomicon: The Dark Arts of
   Advanced and Unsafe Rust Programming*. — Memory safety and pointer aliasing
   guarantees underpinning generic storage wrappers.
8. **ISO. (2018).** *ISO 26262-6:2018 Road vehicles — Functional safety — Part
   6: Product development at the software level*.
9. **RTCA / EUROCAE. (2011).** *DO-178C: Software Considerations in Airborne
   Systems and Equipment Certification*.
10. **IEEE Computer Society. (2008).** *IEEE Standard for Software and System
    Test Documentation* (IEEE Std 829-2008).

---

### 11. Development Plan

| Task / Feature                                | Description                                                                                                                         | Estimated Effort |
|:----------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
| **Step 1: Core Struct & Constructors**        | Implement `StateSpace<T, NX, NU, NY, Sa, Sb, Sc, Sd>` and basic array/slice constructors in `src/state_space/mod.rs`                | 1 Day            |
| **Step 2: Matrix Views & Basic Operations**   | Implement `a_matrix()`, `b_matrix()`, `c_matrix()`, `d_matrix()` views and time-domain simulation (`step()`, continuous derivative) | 1 Day            |
| **Step 3: System Interconnections**           | Implement `series`, `parallel`, and `feedback` methods with compile-time Peano dimension math                                       | 2 Days           |
| **Step 4: Discretization Algorithms**         | Implement ZOH matrix exponential and Bilinear (Tustin) discretization methods                                                       | 2 Days           |
| **Step 5: Structural Analysis & Conversions** | Implement controllability/observability matrix generation, similarity transforms, and transfer function conversions                 | 2 Days           |
| **Step 6: Tests & Documentation**             | Add comprehensive unit tests, doctests, and update crate-level documentation                                                        | 1 Day            |

---

### 12. Revision History

- **July 26, 2026**: Initial draft outline of the `StateSpace` model design
  document.
- **July 26, 2026**: Added inline academic citations and 3-tiered references
  section (@MitchellDScott).
- **August 1, 2026**: Separated the crate-wide `no_alloc` rule from the
  `StateSpace`-specific justification for using raw storage backends instead
  of wrapping `Matrix` fields directly (@MitchellDScott).
