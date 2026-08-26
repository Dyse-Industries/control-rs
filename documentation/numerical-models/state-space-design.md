# State-Space Model Type (Design Document)

![Date Badge](https://img.shields.io/badge/Date-August_25,_2026-blue)
![Status Badge](https://img.shields.io/badge/Doc%20Status-Approved-green)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

---

### 1. Introduction

The `StateSpace` module provides a statically sized, type-safe representation of
continuous-time and discrete-time linear time-invariant (LTI) state-space models
for control systems engineering, real-time simulation, and state estimation.

Continuous-time LTI dynamics:
$$\dot{x}(t) = A x(t) + B u(t), \quad y(t) = C x(t) + D u(t)$$

Discrete-time LTI dynamics:
$$x[k+1] = A x[k] + B u[k], \quad y[k] = C x[k] + D u[k]$$

where $x \in \mathbb{R}^{N_x}$ is the state vector, $u \in \mathbb{R}^{N_u}$ is
the control input vector, $y \in \mathbb{R}^{N_y}$ is the measurement output
vector, and $A, B, C, D$ are system matrices of
dimensions $(N_x \times N_x)$, $(N_x \times N_u)$, $(N_y \times N_x)$,
and $(N_y \times N_u)$.

Primary usage scenarios:

- **Real-Time Plant Simulation**: Stepping state equations forward at
  deterministic sampling intervals ($T_s$) in flight and motor controllers
  without heap allocation.
- **State Estimation & Observer Design**: Executing Kalman filter prediction
  updates and Luenberger observer
  corrections ($\hat{x}_{k+1} = (A - L C)\hat{x}_k + B u_k + L y_k$).
- **LTI System Interconnections**: Constructing composite feedback loops,
  cascade series connections, and parallel plants at compile-time validated
  dimensions.
- **Discretization of Physical Models**: Converting continuous differential
  models ($A, B$) into discrete transition
  matrices ($A_d = e^{A T_s}, B_d = \int_0^{T_s} e^{A \tau} d\tau B$) using
  matrix exponentials and Zero-Order Hold (ZOH).

---

### 2. Requirements

#### 2.1. Functional Requirements

- **FR-1 — Continuous & Discrete LTI Representation**: Represents Linear
  Time-Invariant (LTI) systems with continuous
  dynamics ($\dot{x} = A x + B u, y = C x + D u$) or discrete
  updates ($x_{k+1} = A x_k + B u_k, y_k = C x_k + D u_k$) with explicit sample
  time $T_s$. Dimension compatibility ($N_x, N_u, N_y$) must be validated at
  compile time.
- **FR-2 — Deterministic State Propagation**: Evaluates single-step state
  propagation ($x_{k+1}$) and output generation ($y_k$) given current
  state $x_k$ and input $u_k$. Systems with zero direct feedthrough ($D = 0$)
  must skip the feedforward multiply-accumulate without runtime branch overhead.
- **FR-3 — System Interconnection Algebra**: Supports series ($G_2 \cdot G_1$),
  parallel ($G_1 + G_2$), and feedback ($G / (I + G H)$) interconnections,
  deriving composite state dimensions at compile time. Feedback interconnections
  evaluate algebraic loop solvability and return
  `Err(StateSpaceError::AlgebraicLoop)` when $(I + D_1 D_2)$ is singular.
- **FR-4 — Continuous-to-Discrete Discretization**: Discretizes continuous-time
  state-space models using Zero-Order Hold (ZOH via scaling-and-squaring matrix
  exponential) and Bilinear / Tustin transformations (Van Loan, 1978; Higham,
  2005). Ill-conditioned discretization transformations return explicit error
  variants.
- **FR-5 — Coordinate Similarity Transformations**: Computes coordinate basis
  changes ($z = T x$) producing transformed system
  matrices ($\tilde{A} = T A T^{-1}$, $\tilde{B} = T B$, $\tilde{C} = C T^{-1}$, $\tilde{D} = D$),
  returning an error if transformation matrix $T$ is singular.
- **FR-6 — Controllability, Observability & Transfer Function Conversion**:
  Generates controllability and observability
  matrices ($[B, AB, \dots, A^{n-1}B]$ and $[C; CA; \dots; CA^{n-1}]$) and
  provides conversion to SISO `TransferFunction` models.

#### 2.2. Non-Functional Requirements

- **NFR-1 — Single-Step Execution Complexity**: Discrete single-step propagation
  executes in $O(N_x^2 + N_x N_u + N_y N_x + N_y N_u)$ operations with zero
  dynamic memory allocations.
- **NFR-2 — Bounded Stack Overhead**: System representations and intermediate
  interconnection matrices maintain bounded stack footprints suited for embedded
  real-time tasks.

#### 2.3. Constraints

- **C-1 — Non-Zero State Dimensions**: State, input, and output dimensions must
  satisfy $N_x \ge 1, N_u \ge 1, N_y \ge 1$.
- **C-2 — Dimension Capacity Limits**: State dimensions are
  bounded ($N_x \le 32$, $N_u, N_y \le 16$) to ensure real-time determinism and
  prevent stack overflow.
- **C-3 — `#![no_std]` Environment**: Operates strictly in `#![no_std]` without
  standard library dependencies.

---

### 3. Technical Overview

`StateSpace<T, Nx, Nu, Ny, Sa, Sb, Sc, Sd>` acts as a statically sized LTI
state-space container over four generic matrix storage backends `Sa`, `Sb`,
`Sc`, `Sd` implementing `DenseStorage`. By operating directly over raw storage
backends rather than four full `Matrix` structs, it enables heterogeneous
storage configurations (such as static ROM arrays for system matrices and
zero-sized backends for direct feedthrough $D = 0$).

The module exposes `MatrixSlice` views on demand for linear algebra, and
provides real-time time-domain state propagation, system interconnection
algebra (series, parallel, feedback with algebraic loop detection), Van Loan
scaling-and-squaring matrix exponential ZOH discretization, similarity
coordinate transforms, and SISO transfer function conversions.
and `MatrixSliceMut<'a, T, R, C>` for safe, high-level matrix operations.
These accessors (§4.3) are layout-agnostic per `storage-design.md`
§4.1, where ordering is carried by the leaf's `isize` strides rather than
by a layout flag: `a_matrix()`/`b_matrix()`/`c_matrix()`/`d_matrix()` and the
arithmetic `step()`/`derivative()` build on top of never assume
`Sa`/`Sb`/`Sc`/`Sd` are column-major — swapping any one of them for a
row-major backend requires no change here.

- **`crate::math::subprograms`**: BLAS Level 1/2/3 kernels (`Gemv`, `Gemm`,
  `Axpy`) and the LAPACK factorization and solver traits
  (`Getrf`/`Getrs`, `Potrf`/`Potrs`, `Geqrf`, `Syev`/`Heev`;
  `subprograms-design.md` FR-6 to FR-8). Solves (§4.4, §4.7, §4.9) call
  those traits rather than re-implementing substitution;
  `Trsv` and `Trsm` both exist in `subprograms-design.md` FR-2 and FR-4 and
  are available where a bare triangular solve is wanted.

Transcendental operations required by §4.8/§4.9 (matrix exponential scaling
factors, `tan` for Tustin pre-warping) project onto `T::Real`, which the
analytic traits `Radical`, `Exponential` and `Trig` cover
(`num-traits-design.md` §4.1). `T: Float` is `f32`/`f64` only and is not a
bound that admits `Complex<T>` (FR-5), so those sites bind
`T: Scalar + Div` with `T::Real: Trig` rather than `T: Float`.

---

### 4. Core Architecture

#### 4.1 Type Signature & Storage Layout

```rust
// Dim-generic core. Owning defaults cannot be `ArrayStorage<T, NX, NX>`:
// that leaf takes bare `const usize` parameters, not `Dim` types.
pub struct StateSpaceCore<
    T,
    NX: Dim,
    NU: Dim,
    NY: Dim,
    Sa: DenseStorage<T, R=NX, C=NX>,
    Sb: DenseStorage<T, R=NX, C=NU>,
    Sc: DenseStorage<T, R=NY, C=NX>,
    Sd: DenseStorage<T, R=NY, C=NU>,
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
// Owning stack allocation. `NX`/`NU`/`NY` are the alias's own const generics;
// `Dim` slots are `Const<NX>`/`Const<NU>`/`Const<NY>`.
pub type StateSpace<T, const NX: usize, const NU: usize, const NY: usize> = StateSpaceCore<
    T,
    Const<NX>,
    Const<NU>,
    Const<NY>,
    DenseArray<T, NX, NX>,
    DenseArray<T, NX, NU>,
    DenseArray<T, NY, NX>,
    DenseArray<T, NY, NU>,
>;

/// Sibling model alias for standard stack-allocated array storage
pub type ArrayStateSpace<T, const NX: usize, const NU: usize, const NY: usize> =
StateSpace<T, NX, NU, NY>;

/// Zero-copy borrowed read-only view.
pub type StateSpaceView<'a, T, const NX: usize, const NU: usize, const NY: usize> = StateSpaceCore<
    T,
    Const<NX>,
    Const<NU>,
    Const<NY>,
    StorageView<'a, T, Const<NX>, Const<NX>>,
    StorageView<'a, T, Const<NX>, Const<NU>>,
    StorageView<'a, T, Const<NY>, Const<NX>>,
    StorageView<'a, T, Const<NY>, Const<NU>>,
>;

/// Zero-copy borrowed mutable view.
pub type StateSpaceViewMut<'a, T, const NX: usize, const NU: usize, const NY: usize> = StateSpaceCore<
    T,
    Const<NX>,
    Const<NU>,
    Const<NY>,
    StorageViewMut<'a, T, Const<NX>, Const<NX>>,
    StorageViewMut<'a, T, Const<NX>, Const<NU>>,
    StorageViewMut<'a, T, Const<NY>, Const<NX>>,
    StorageViewMut<'a, T, Const<NY>, Const<NU>>,
>;
```

`StorageView`/`StorageViewMut` are themselves `DenseStorage` implementors
(`storage-design.md` FR-2), so `StateSpaceView`'s `a_matrix()` (§4.3)
constructs a `MatrixSlice` directly over the existing borrowed view rather
than performing a redundant copy — the accessor borrows straight through, it
does not re-wrap. Views are strided, not necessarily contiguous, so they
carry no `as_slice()`; the accessors that need one bound
`ContiguousStorage` separately (§4.3). Built-in time-domain simulation
methods (`step()`, `derivative()`) delegate directly to the single-source
subprogram kernels (`Gemv`, `Axpy`) defined in `src/math/subprograms.rs`.

#### 4.3 Safe `Matrix` Integration & Zero-Copy Matrix Views

While data is stored inside storage backends (`Sa`, `Sb`, `Sc`, `Sd`),
`StateSpaceCore` exposes methods to treat each backend as a zero-cost `Dense`
view
or `MatrixSlice`, enabling full reuse of `Matrix` operations (multiplication,
addition, transposition, solver routines). Per §4.1 and NFR-2, these accessors
are
bounded on `DenseStorage`/`DenseStorageMut`:

```rust
impl<T, const NX: usize, const NU: usize, const NY: usize, Sa, Sb, Sc, Sd>
StateSpaceCore<T, Const<NX>, Const<NU>, Const<NY>, Sa, Sb, Sc, Sd>
where
    Sa: DenseStorage<T, R=Const<NX>, C=Const<NX>>,
    Sb: DenseStorage<T, R=Const<NX>, C=Const<NU>>,
    Sc: DenseStorage<T, R=Const<NY>, C=Const<NX>>,
    Sd: DenseStorage<T, R=Const<NY>, C=Const<NU>>,
{
    /// Exposes system matrix A as a high-level Matrix view.
    pub fn a_matrix(&self) -> MatrixSlice<'_, T, NX, NX> {
        MatrixSlice::from_storage(&self.a_storage)
    }

    /// Exposes input matrix B as a high-level Matrix view.
    pub fn b_matrix(&self) -> MatrixSlice<'_, T, NX, NU> {
        MatrixSlice::from_storage(&self.b_storage)
    }

    /// Exposes output matrix C as a high-level Matrix view.
    pub fn c_matrix(&self) -> MatrixSlice<'_, T, NY, NX> {
        MatrixSlice::from_storage(&self.c_storage)
    }

    /// Exposes feedforward matrix D as a high-level Matrix view.
    pub fn d_matrix(&self) -> MatrixSlice<'_, T, NY, NU> {
        MatrixSlice::from_storage(&self.d_storage)
    }
}
```

A general (non-contiguous) accessor path is left unspecified in this
revision; callers holding a non-contiguous backend must materialize an owned
`Matrix` explicitly rather than obtaining a borrowed view.

#### 4.4 Error Handling

Following the crate-wide error strategy, fallible operations return
`Result<T, Error>` via a crate-local `thiserror` enum rather than panicking.
Two operations are fallible:

- **Feedback (§4.7)**: forms the loop
  matrix $F = I - \text{sign} \cdot D_2 D_1 \in \mathbb{R}^{N_u \times N_u}$
  and solves against it rather than inverting explicitly, matching
  `python-control`'s `feedback()` (`control/statesp.py`, §11.1 ref. 5).
  $F$ singular to working precision is a
  genuine closed-loop ill-posedness, not an internal error.
- **Tustin Discretization (§4.8)**: solves against $(I - \frac{T_s}{2}A)$,
  singular exactly when $\frac{2}{T_s}$ is an eigenvalue of $A$ (MathWorks,
  *Continuous-Discrete Conversion Methods*).

```rust
#[derive(Debug, thiserror::Error)]
pub enum StateSpaceError {
    #[error("feedback loop matrix (I - sign*D2*D1) is singular to working precision"
    )]
    SingularLoopMatrix,
    #[error("Tustin discretization operator (I - Ts/2 * A) is singular to working precision"
    )]
    SingularDiscretizationOperator,
}
```

Both variants reuse the rank/singularity detection machinery already
Near-singular (but not exactly singular) evaluation of $H(s)$ near a pole of
$A$ (§4.9) is *not* treated as a constructor- or call-time error — a valid
`StateSpace` may still be evaluated at an ill-conditioned frequency point,
which is documented behavior (§4.9) rather than a distinct error variant,
matching the posture already adopted for `TransferFunction::evaluate_complex`
(`transfer-function-design.md` §4.5).

#### 4.5 Instantiation & Constructors

- **From Storage**:
  `pub const fn from_storage(a: Sa, b: Sb, c: Sc, d: Sd, sample_time: Option<T>) -> Self`
- **Owning Array Constructor**:
  ```rust
  pub fn from_arrays<const NX: usize, const NU: usize, const NY: usize>(
      a: [[T; NX]; NX],
      b: [[T; NX]; NU],
      c: [[T; NY]; NX],
      d: [[T; NY]; NU],
      sample_time: Option<T>,
  ) -> ArrayStateSpace<T, NX, NU, NY>
  ```
  Nested arrays match `ArrayStorage`'s `[[T; R]; C]` buffer
  (`storage-design.md` FR-2). Lengths are the alias's own const generics,
  not `Dim::USIZE` products.
- **Borrowed views**: `ArrayStateSpace::view()` / `view_mut()`
  (`storage-design.md` FR-2). Wrapping an erased-length slice goes through
  `StorageView::new`, which is fallible with
  `ConversionError::DimensionMismatch`.

##### 4.5.1 Data-Driven State-Space Factories [Proposal (not in evidence)]

To support higher-level subspace identification, modal realization, and
experimental system identification, dedicated data-driven **Object Factories**
estimate $(A, B, C, D)$ matrices directly from sampled input-output data or
Markov parameter sequences without polluting `StateSpaceCore` (Van Overschee &
De Moor, 1994; Verhaegen & Dewilde, 1992; Juang & Pappa, 1985; Juang et al.,
1993; Ljung, 2001, 2003; Qin, 2006; De Schutter, 2000):

- **ERA Realization Factory (`EraStateSpaceFactory`)**: Realizes a discrete-time
  minimal state-space model $(A, B, C, D)$ from Markov parameters $Y_k = C
  A^{k-1} B$ by forming shifted block-Hankel matrices $H_0, H_1$ and computing
  their singular value decomposition (SVD) with order-truncation (Juang & Pappa,
  1985; De Schutter, 2000).
- **MOESP Subspace Estimator (`MoespStateSpaceEstimator`)**: Realizes
  finite-dimensional state-space models from input-output data $(u_k, y_k)$ via
  $LQ$ decomposition of block-Hankel data matrices without non-linear
  optimization (Verhaegen & Dewilde, 1992; Qin, 2006).
- **N4SID Subspace Estimator (`N4sidStateSpaceEstimator`)**: Estimates state
  sequences and system matrices through oblique projection of past and future
  input-output block-Hankel matrices (Van Overschee & De Moor, 1994; Ljung,
  2003).

_Detailed standalone design, algorithmic steps, and API signatures are specified
in `documentation/control-toolboxes/sysid-design.md`._

#### 4.6 State Propagation & Time-Domain Simulation

##### Discrete State Step ($x[k+1] = A x[k] + B u[k]$)

Given current state vector $x \in \mathbb{R}^{N_x}$ and input
vector $u \in \mathbb{R}^{N_u}$, compute next state $x_{next}$ and
output $y \in \mathbb{R}^{N_y}$:

```rust
impl<T, NX: Dim, NU: Dim, NY: Dim, Sa, Sb, Sc, Sd> StateSpaceCore<T, NX, NU, NY, Sa, Sb, Sc, Sd>
where
    Sa: DenseStorage<T, R=NX, C=NX>,
    Sb: DenseStorage<T, R=NX, C=NU>,
    Sc: DenseStorage<T, R=NY, C=NX>,
    Sd: DenseStorage<T, R=NY, C=NU>,
{
    pub fn step<Sx, Su>(
        &self,
        x: &Matrix<T, NX, Const<1>, Sx>,
        u: &Matrix<T, NU, Const<1>, Su>,
    ) -> (ArrayMatrix<T, NX, Const<1>>, ArrayMatrix<T, NY, Const<1>>)
    where
        Sx: DenseStorage<T, R=NX, C=Const<1>>,
        Su: DenseStorage<T, R=NU, C=Const<1>>,
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
e.g., Runge-Kutta 4th Order). Symmetric to `step()` and subject to the same
domain-mismatch caveat in reverse.

#### 4.7 System Interconnections

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

Closed-loop negative/positive feedback around system $S_1$ and
controller/feedback system $S_2$.

#### 4.8 Discretization (ZOH & Bilinear)

Converts a continuous `StateSpace` model ($T_s = \text{None}$) into a discrete
`StateSpace` model ($T_s = \text{Some}(T_s)$).

##### Zero-Order Hold (ZOH)

Form the augmented
matrix $M = \begin{bmatrix} A & B \\ 0 & 0 \end{bmatrix} \in \mathbb{R}^{(N_x + N_u) \times (N_x + N_u)}$
and compute its
matrix
exponential $e^{M T_s} = \begin{bmatrix} \Phi & \Gamma \\ 0 & I \end{bmatrix}$ (
Van
Loan, 1978).

##### Bilinear (Tustin) Transform

Approximates $s \approx \frac{2}{T_s} \frac{z-1}{z+1}$ algebraically.

#### 4.9 Canonical Transformations & Transfer Function Equivalences

##### Similarity Transformation

Given state coordinate change $z = T x$ with non-singular $T \in
\mathbb{R}^{N_x \times N_x}$:
$$A' = T A T^{-1}, \quad B' = T B, \quad C' = C T^{-1}, \quad D' = D$$

##### Controllability & Observability Matrices

**API commitment**: implement explicit power construction.

##### Transfer Function Conversion

**API commitment**: implement
$$H(s) = C (s I - A)^{-1} B + D = \frac{C \text{adj}(sI - A) B + D \det(sI - A)}{\det(sI - A)}$$

---

### 5. Alternatives

| Alternative                                              | Description                                                                           | Pros                                                                                                                                                                                                                                  | Cons                                                                                                                                                                                                                                                   |
|:---------------------------------------------------------|:--------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **A. Monolithic Storage Field**                          | Storing a single combined block matrix $\begin{bmatrix} A & B \\ C & D \end{bmatrix}$ | Single contiguous allocation                                                                                                                                                                                                          | Cannot specialize a structurally zero $D$ (common for strictly proper plants); the series/parallel/feedback formulas (§4.7) assemble $A, B, C, D$ from independently shaped sub-blocks and would need awkward strided writes into one contiguous block |
| **B. Wrapping 4 `Matrix` Structs Directly**              | Storing `a: Matrix<T, NX, NX, Sa>`, `b: Matrix...`                                    | Built-in matrix methods on fields                                                                                                                                                                                                     | Redundant type wrapper parameters, extra boilerplate vs storing raw `DenseStorage` backends; the zero-$D$ specialization is awkward since the wrapper's own type parameters still have to be named                                                     |
| **C. Generic Storage Backends + `MatrixSlice` (Chosen)** | Storing raw `Sa, Sb, Sc, Sd` backends and creating `MatrixSlice` wrappers on demand   | Maximum flexibility (`#![no_std]`, ROM, stack, zero-matrix storage); expresses a zero-sized $D$ specialization that eliminates a `Gemm` per `step()`, which neither A nor B can express as cleanly; symmetric with `TransferFunction` | Requires `a_matrix()`/etc. accessors gated on `DenseStorage` (§4.3) for high-level matrix algebra                                                                                                                                                      |

The strongest evidence for Alternative C is the zero-$D$ specialization
above. The same "avoid wrapping the natural lower-level peer type" decision
is made independently by `transfer-function-design.md` §6 for
`TransferFunction` vs. `Polynomial`, for compatible reasons.

---

### 6. Verification & Validation

#### 6.1. Objectives

- Demonstrate compile-time verification of state, input, and output dimension
  constraints.
- Demonstrate mathematical exactness and backward stability of time-domain
  simulation (`step()`, `derivative()`).
- Demonstrate numerical accuracy of ZOH discretization via Van Loan augmented
  matrix exponential with Higham Padé scaling.
- Demonstrate algebraic exactness of series, parallel, and feedback block
  interconnections.
- Demonstrate stability invariance under similarity transforms ($z = Tx$).
- Demonstrate zero dynamic heap allocation in `#![no_std]` execution and
  deterministic real-time latency.

#### 6.2. Methods

| Method                    | Mechanism                                                                        | Requirements discharged  |
|:--------------------------|:---------------------------------------------------------------------------------|:-------------------------|
| Compile-time shape check  | Type-level `Dim` assertions, `compile_fail` doctests                             | FR-1, C-1, C-2, C-4      |
| Requirements-based test   | `#[test]` unit tests over standard physical models and singular cases            | FR-2, FR-3, FR-4, FR-5   |
| Property-based test       | `proptest` suites verifying interconnection identities and similarity invariants | FR-3, FR-5               |
| Doctest                   | Runnable doc examples in rustdoc                                                 | FR-2, FR-6               |
| Back-to-back comparison   | `/cr-prototype numerical-models/state-space` and MATLAB/`python-control` oracles | FR-2, FR-4, FR-5         |
| Resource usage evaluation | `no_alloc` audit, `size_of` assertions, stack analysis                           | NFR-1, NFR-3, C-2, C-3   |
| On-target execution       | ETS suites under QEMU and Teensy hardware                                        | NFR-2                    |
| Coverage measurement      | `cargo coverage` reporting statement and branch metrics                          | FR-1..FR-6, NFR-1..NFR-3 |

#### 6.3. Acceptance Criteria

| Claim                              | Oracle                                            | Measure        | Bound                                                                                             | Justification                                               |
|:-----------------------------------|:--------------------------------------------------|:---------------|:--------------------------------------------------------------------------------------------------|:------------------------------------------------------------|
| ZOH discretization residual        | Van Loan exact matrix exponential                 | Relative error | $\frac{\|\hat{A}_d - e^{A T_s}\|_\infty}{\|e^{A T_s}\|_\infty \epsilon} < 20.0$                   | Van Loan (1978) & Higham (2005) Padé scaling error bound    |
| Step response propagation          | Closed-form analytic linear solution              | Absolute error | $\|x[k] - x_{\text{exact}}[k]\|_\infty \le k \gamma_{N_x} \|A\|_\infty^k \|x[0]\|_\infty$         | Matrix recurrence error propagation (Higham, 2002)          |
| Similarity transform invariance    | Characteristic polynomial invariance              | Relative error | $\frac{\|\det(sI - A) - \det(sI - T A T^{-1})\|_\infty}{\|\det(sI - A)\|_\infty} \le 10 \epsilon$ | Spectral invariance under coordinate change (Ogata, 2010)   |
| Feedback loop singularity          | Singular algebraic loop matrix $(I + D_1 D_2)$    | Exact equality | `Err(StateSpaceError::SingularLoopMatrix)`                                                        | Precondition failure contract                               |
| Tustin discretization singularity  | Singular bilinear operator $(I - \frac{T_s}{2}A)$ | Exact equality | `Err(StateSpaceError::SingularDiscretizationOperator)`                                            | Solvability precondition contract                           |
| Long-horizon fixed-point recursion | Saturating recursion across $10^5$ steps          | Exact equality | Zero unbounded overflow drift / limit cycles                                                      | Mullis & Roberts (1976), Hwang (1977) fixed-point stability |
| Zero-allocation guarantee          | Host memory allocator interception                | Exact equality | 0 heap allocations                                                                                | NFR-1 `#![no_std]` invariant                                |

#### 6.4. Traceability

| Requirement                                  | Method                                           | Artifact                                                     |
|:---------------------------------------------|:-------------------------------------------------|:-------------------------------------------------------------|
| FR-1 — Compile-Time Static Sizing            | Compile-time shape check                         | `tests/state_space_shape_fail.rs` (`compile_fail` doctests)  |
| FR-2 — Time-Domain Simulation                | Requirements-based test, Back-to-back comparison | `tests/state_space_sim.rs::test_step_response`               |
| FR-3 — System Interconnection Algebra        | Property-based test, Back-to-back comparison     | `tests/state_space_interconnect.rs::prop_series_parallel`    |
| FR-4 — Discretization Algorithms             | Requirements-based test, Back-to-back comparison | `tests/state_space_discretize.rs::test_van_loan_zoh`         |
| FR-5 — Model Transformations & Realizations  | Property-based test, Requirements-based test     | `tests/state_space_transforms.rs::test_similarity_transform` |
| FR-6 — Non-Allocating Storage Abstraction    | Resource usage evaluation                        | `tests/state_space_storage.rs::test_dense_storage_views`     |
| NFR-1 — Zero Dynamic Heap Allocation         | Resource usage evaluation                        | `#![no_std]` host allocator check                            |
| NFR-2 — Deterministic Real-Time Execution    | On-target execution                              | ETS test suite `state_space::bench_step_latency`             |
| NFR-3 — Interoperable Matrix Storage & Views | Resource usage evaluation                        | `tests/state_space_interop.rs::test_matrix_interop`          |
| C-1 — LTI Standard Form                      | Compile-time shape check                         | Type definitions for linear systems                          |
| C-2 — Dimension Capacity Bounds              | Resource usage evaluation                        | `clippy::large_stack_arrays` CI check                        |
| C-3 — `#![no_std]` Compatibility             | Resource usage evaluation                        | Compilation under `#![no_std]` target triples                |
| C-4 — Stable Rust Toolchain                  | Compile-time shape check                         | Cargo workspace build on `stable` Rust                       |

#### 6.5. Coverage

- **Target**: $\ge 90\%$ statement coverage, $\ge 85\%$ branch coverage reported
  via `cargo coverage`.
- **Excluded**: Hardware ETS cycle benchmarking loops and debug display
  implementations (`core::fmt::Debug`).

#### 6.6. Validation

- **Kinematic Object Tracking**: Validation of continuous and discrete 1D
  kinematic object tracking predicting position and velocity in
  `examples/kinematic_tracker.rs`.
- **Closed-Loop DC Motor Control**: Step-response and disturbance rejection
  validation in `examples/dc_motor_lqr.rs`.

#### 6.7. Not Verified

- Descriptor state-space formulations ($E \dot{x} = A x + B u$) are excluded and
  not verified in this revision.
- MIMO transfer-function conversions using minimal McMillan-degree state
  realizations are deferred to future revisions.

---

### 7. Performance & Resource Considerations

- **Stack Allocation Limits**: Large state vectors (e.g., $N_x = 32$)
  require $32 \times 32 = 1024$ elements for matrix $A$, exactly
  `matrix-design.md` §2.3's per-matrix budget. Storing via `DenseStorage`
  enables static buffer placement or borrowed views, preventing embedded stack
  overflow for $A$/$B$/$C$/$D$ themselves.
- **ZOH Is Not Accommodated at $N_x = 32$**: the augmented matrix $M$
  is $(N_x + N_u)$-square (§4.8), so a 32-state system cannot be ZOH-discretized
  with even a single input without flattening bounds.
- **ZOH Workspace Multiplier**: scaling-and-squaring at a competitive Padé
  degree requires on the order of six additional $(N_x+N_u)$-square temporaries
  beyond $M$ itself. A 24-state, 8-input system needs
  roughly $7 \times 32 \times 32 \times 4\ \text{bytes} \approx 28\ \text{KB}$
  of `f32` transient workspace (Higham, 2005).
- **Interconnection Depth**: series/parallel/feedback each produce
  an $N_{x1}+N_{x2}$-state result. Stack workspace for the combined $A$ matrix
  is bounded by the static capacity constraints of `DenseStorage`.
- **Matrix Exponential Performance**: ZOH discretization of $A$ uses
  scaling-and-squaring Padé approximation using the Level 3 `Gemm` kernel (see
  §4.8 for the specific Padé-degree and overscaling considerations).

---

### 8. Risks & Open Questions

- **Derived-Dimension Alias Gaps**: FR-2 ($N_{x1}+N_{x2}$), ZOH ($N_x+N_u$) and
  matrix multiplications ($N_x \cdot N_u$, $N_y \cdot N_x$) each derive a
  dimension larger than $N_x$. Binary `Dim` encoding (`num-types-design.md`
  C-1/C-2) no longer rejects these at a `U127` solver ceiling; unnamed products
  are valid. `Const<N>: Dim` is still missing for some flattened sizes
  in $1025..16383$. Stack workspace, not the trait solver, is the remaining
  limit for large $N_x$.
- **Matrix Inversion & Numerical Stability**: Explicit matrix inversion for
  Tustin discretization or transfer-function conversion is ill-conditioned for
  high state dimensions. The solve-over-inversion preference (§4.4, §4.7, §4.9)
  is fully resolved by reference-implementation precedent.
- **Sparse and Structured Storage Optimization**: A structurally zero $D$ matrix
  is a common, compile-time-knowable, immediately resolvable win. A sparse or
  companion-form $A$ is a different question and stays deferred.
- **Analytic Scalar Bounds**: §4.8/§4.9's transcendental sites (matrix
  exponential scaling, Tustin pre-warping) bind `T: Scalar + Div` with
  `T::Real: Radical`/`Trig` rather than `T: Float`, which `num-traits-design.md`
  FR-5 restricts to `f32`/`f64`. Projecting onto `T::Real` keeps those paths
  open to `Complex<T>` plants.
- **Descriptor Form (`E` Matrix) Scope**: Descriptor/DAE
  systems ($E\dot{x} = Ax + Bu$) are out of scope for this revision (C-1, §4.1).
- **Domain Encoding Ambiguity**: The runtime `sample_time: Option<T>` does not
  prevent calling `step()` on a continuous-time model or `derivative()`-style
  evaluation on a discrete one (§4.6). Whether to introduce a type-level or ZST
  domain marker is left open.
- **Fixed-Point Recursion Noise & Limit Cycles**: A wide-accumulator convention
  for a single `Gemm` is necessary but not sufficient for `step()`, which is a
  recursion. Overflow handling in this path must saturate rather than wrap.
- **FR-4 Scope (MIMO Transfer Function Conversion)**: This revision scopes FR-4
  to SISO (§4.9). Extending to MIMO is tracked as future work.

---

### 9. Development Plan

| Task / Feature                                | Description                                                                                                                                                                                                                         | Estimated Effort |
|:----------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
| **Step 1: Core Struct & Constructors**        | Implement `StateSpaceCore<T, NX, NU, NY, Sa, Sb, Sc, Sd>`, `StateSpace<T, NX, NU, NY>` alias, basic array/slice constructors and `StateSpaceError` (§4.4) in `src/state_space/mod.rs`.                                              | 1.0 Day          |
| **Step 2: Matrix Views & Basic Operations**   | Implement `a_matrix()`/`b_matrix()`/`c_matrix()`/`d_matrix()` bounded on `DenseStorage` per §4.3 and time-domain simulation (`step()`, derivative).                                                                                 | 1.0 Day          |
| **Step 3: System Interconnections**           | Implement `series`, `parallel` and fallible `feedback` (loop-matrix solve, `StateSpaceError::SingularLoopMatrix`) with compile-time `Dim` arithmetic.                                                                               | 2.0 Days         |
| **Step 4: Discretization**                    | Implement ZOH (Van Loan augmented matrix, scaling-and-squaring with precision-dependent Padé degree selection §4.8, Al-Mohy/Higham overscaling correction) and fallible Tustin (`StateSpaceError::SingularDiscretizationOperator`). | 4.5 Days         |
| **Step 5: Structural Analysis & Conversions** | Implement controllability/observability matrix generation (scoped as definitional, §4.9), similarity transforms ($z=Tx$) and Hessenberg-reduction-based SISO transfer function conversion.                                          | 3.0 Days         |
| **Step 6: Tests & Documentation**             | Unit tests, `proptest` suites, `python-control`/MATLAB cross-validation, long-horizon fixed-point recursion tests and crate-level documentation per `vv-standards.md`.                                                              | 3.0 Days         |

---

### 10. References

1. **Van Loan, C. F. (1978).** Computing integrals involving the matrix
   exponential. *IEEE Transactions on Automatic Control*, AC-23(3), 395–404.
2. **Moler, C., & Van Loan, C. (2003).** Nineteen Dubious Ways to Compute the
   Exponential of a Matrix, Twenty-Five Years Later. *SIAM Review*, 45(1), 3–49.
3. **Higham, N. J. (2005).** The Scaling and Squaring Method for the Matrix
   Exponential Revisited. *SIAM J. Matrix Anal. Appl.*, 26(4), 1179–1193.
4. **Al-Mohy, A. H., & Higham, N. J. (2009).** A New Scaling and Squaring
   Algorithm for the Matrix Exponential. *SIAM J. Matrix Anal. Appl.*, 31(3),
   970–989.
5. **python-control developers. (2026).** `control/statesp.py`, python-control
   library source. [Online].
   Available: https://github.com/python-control/python-control/blob/main/control/statesp.py.
6. **python-control developers. (2026).** `control.StateSpace`, python-control
   documentation. [Online].
   Available: https://python-control.readthedocs.io/en/latest/generated/control.StateSpace.html.
7. **python-control developers. (2026).** `control.similarity_transform`,
   python-control documentation. [Online].
   Available: https://python-control.readthedocs.io/en/latest/generated/control.similarity_transform.html.
8. **python-control developers. (2026).** `control.ss2tf`, python-control
   documentation. [Online].
   Available: https://python-control.readthedocs.io/en/latest/generated/control.ss2tf.html.
9. **MathWorks. (2026).** `c2d` — Convert model from continuous to discrete
   time, MATLAB documentation. [Online].
   Available: https://www.mathworks.com/help/control/ref/dynamicsystem.c2d.html.
10. **MathWorks. (2026).** *Continuous-Discrete Conversion Methods*, MATLAB &
    Simulink documentation. [Online].
    Available: https://www.mathworks.com/help/control/ug/continuous-discrete-conversion-methods.html.
11. **MathWorks. (2026).** `obsv` — Observability of state-space model, MATLAB
    documentation. [Online].
    Available: https://www.mathworks.com/help/control/ref/statespacemodel.obsv.html.
12. **MathWorks. (2026).** `ctrb` — Controllability of state-space model, MATLAB
    documentation. [Online].
    Available: https://www.mathworks.com/help/control/ref/statespacemodel.ctrb.html.
13. **MathWorks. (2026).** `ctrbf` — Compute controllability staircase form,
    MATLAB documentation. [Online].
    Available: https://www.mathworks.com/help/control/ref/ctrbf.html.
14. **MathWorks. (2026).** *State-Space Realizations* (`canon`, `compreal`,
    `modalreal`), MATLAB & Simulink documentation. [Online].
    Available: https://www.mathworks.com/help/control/ug/canonical-state-space-realizations.html.
15. **MathWorks. (2026).** `canon` — (Not recommended) Canonical state-space
    realization, MATLAB documentation. [Online].
    Available: https://www.mathworks.com/help/ident/ref/dynamicsystem.canon.html.
16. **MathWorks. (2026).** `prescale` — Optimal scaling of state-space models,
    MATLAB documentation. [Online].
    Available: https://www.mathworks.com/help/control/ref/ss.prescale.html.
17. **MathWorks. (2026).** *Scaling State-Space Models*, MATLAB & Simulink
    documentation. [Online].
    Available: https://www.mathworks.com/help/control/ug/scaling-state-space-models.html.
18. **MathWorks. (2026).** `ss2ss` — State coordinate transformation for
    state-space model, MATLAB documentation. [Online].
    Available: https://www.mathworks.com/help/ident/ref/statespacemodel.ss2ss.html.
19. **MathWorks. (2026).** `sparss` — Sparse first-order state-space model,
    MATLAB documentation. [Online].
    Available: https://www.mathworks.com/help/control/ref/sparss.html.
20. **python-control issue #116 / SLICOT `TB05AD`. (2026).** Discussion of using
    SLICOT's `TB05AD` in `StateSpace.freqresp`. [Online].
    Available: https://github.com/python-control/python-control/issues/116.
21. **ACM. (2011).** A Note on Shifted Hessenberg Systems and Frequency Response
    Computation. *ACM Transactions on Mathematical Software*, 38(2), doi:
    10.1145/2049673.2049676.
22. **ARM Ltd. (2025).** *Matrix Multiplication*, CMSIS-DSP
    documentation. [Online].
    Available: https://arm-software.github.io/CMSIS-DSP/main/group__MatrixMult.html.
23. **`control_systems_torbox` contributors. (2026).** `control_systems_torbox`
    crate documentation. [Online].
    Available: https://docs.rs/control_systems_torbox.
24. **rdesarz. (2026).** `control-sys-rs`: A Control System library implemented
    in Rust. [Online]. Available: https://github.com/rdesarz/control-sys-rs.
25. **sunsided. (2026).** `minikalman-rs`: Fixed- and floating-point Kalman
    filters for resource-constrained environments. [Online].
    Available: https://github.com/sunsided/minikalman-rs.
26. **strawlab. (2026).** `adskalman-rs`: Kalman filter and RTS smoothing in
    Rust. [Online]. Available: https://github.com/strawlab/adskalman-rs.
27. **Kailath, T. (1980).** *Linear Systems*. Prentice-Hall.
28. **Ogata, K. (2010).** *Modern Control Engineering* (5th ed.). Prentice Hall.
29. **Åström, K. J., & Murray, R. M. (2021).** *Feedback Systems: An
    Introduction for Scientists and Engineers* (2nd ed.). Princeton University
    Press.
30. **Golub, G. H., & Van Loan, C. F. (2013).** *Matrix Computations* (4th ed.).
    Johns Hopkins University Press.
31. **Mullis, C. T., & Roberts, R. A. (1976).** Synthesis of minimum roundoff
    noise fixed point digital filters. *IEEE Transactions on Circuits and
    Systems*, 23(9), 551–562.
32. **Yang, S., & Jones, C. N. (2026).** Numerically Reliable Brunovsky
    Transformations.
33. **Claessen, K., & Hughes, J. (2000).** QuickCheck: A Lightweight Tool for
    Random Testing of Haskell Programs. *ACM SIGPLAN Notices*, 35(9), 268–279.
34. **Rust Project Developers. (2024).** *The Rustonomicon: The Dark Arts of
    Advanced and Unsafe Rust Programming*.
35. **ISO. (2018).** *ISO 26262-6:2018 Road vehicles — Functional safety — Part
    6: Product development at the software level*.
36. **RTCA / EUROCAE. (2011).** *DO-178C: Software Considerations in Airborne
    Systems and Equipment Certification*.
37. **IEEE Computer Society. (2008).** *IEEE Standard for Software and System
    Test Documentation* (IEEE Std 829-2008).
38. **Van Overschee, P., & De Moor, B. (1994).** N4SID: Subspace Algorithms for
    the Identification of Combined Deterministic-Stochastic Systems.
    *Automatica*, 30(1), 75–93, doi: 10.1016/0005-1098(94)90046-6.
39. **Verhaegen, M., & Dewilde, P. (1992).** Subspace Model Identification Part
    1. The output-error state-space model identification class of algorithms.
       *International Journal of Control*, 56(5), 1187–1210, doi:
       10.1080/00207179208934363.
40. **Juang, J.-N., & Pappa, R. S. (1985).** An eigensystem realization
    algorithm for modal parameter identification and model reduction. *Journal
    of
    Guidance, Control, and Dynamics*, 8(5), 620–627, doi: 10.2514/3.20031.
41. **Juang, J.-N., Phan, M., Horta, L. G., & Longman, R. W. (1993).**
    Identification of observer/Kalman filter Markov parameters - Theory and
    experiments. *Journal of Guidance, Control, and Dynamics*, 16(2), 320–329,
    doi: 10.2514/3.21006.
42. **Qin, S. J. (2006).** An overview of subspace identification. *Computers &
    Chemical Engineering*, 30(10–12), 1502–1513, doi:
    10.1016/j.compchemeng.2006.05.034.
43. **De Schutter, B. (2000).** Minimal state-space realization in linear
    system theory: an overview. *Journal of Computational and Applied
    Mathematics*, 121(1–2), 331–354, doi: 10.1016/S0377-0427(00)00341-1.
44. **Ljung, L. (2001).** Black-box models from input-output data. In *40th
    IEEE Conference on Decision and Control*, Orlando, FL, USA.
45. **Ljung, L. (2003).** Subspace methods for system identification. In *SYSID
    2003: 13th IFAC Symposium on System Identification*, Rotterdam, The
    Netherlands.

---

### 11. Revision History

| Revision | Date            | Author          | Description                                                                                                                           |
|:---------|:----------------|:----------------|:--------------------------------------------------------------------------------------------------------------------------------------|
| 1.0      | July 26, 2026   | @MitchellDScott | Initial draft: LTI state-space representations, layouts, and continuous/discrete models.                                               |
| 1.1      | August 16, 2026 | @MitchellDScott | Storage & subprogram integration: bound system matrices ($A, B, C, D$) to decoupled `DenseStorage` and LAPACK solvers.               |
| 1.2      | August 25, 2026 | @MitchellDScott | Discretization & simulation: added zero-order hold (ZOH), bilinear/Tustin discretization, and algebraic interconnects.                 |
| 1.3      | August 25, 2026 | @MitchellDScott | V&V standardization: aligned test oracles with matrix exponential and Padé approximation error bounds.                                |
| 1.4      | August 26, 2026 | @MitchellDScott | Storage view retarget: updated references to `StorageView`/`StorageViewMut` and `DenseStorage` traits.                                |
