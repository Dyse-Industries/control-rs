# State-Space Model Type (Design Document)

![Date Badge](https://img.shields.io/badge/Date-August_2,_2026-blue)
![Status Badge](https://img.shields.io/badge/Doc%20Status-Draft-orange)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

---

### 1. Introduction

The `StateSpace` module provides a statically sized, type-safe representation of
continuous-time and discrete-time linear time-invariant (LTI) state-space models
for control systems engineering, signal processing and state estimation (e.g.,
Kalman filtering, LQR synthesis and observer design).

A continuous-time LTI state-space system is governed by:
$$\dot{x}(t) = A x(t) + B u(t)$$
$$y(t) = C x(t) + D u(t)$$

A discrete-time LTI state-space system is governed by:
$$x[k+1] = A x[k] + B u[k]$$
$$y[k] = C x[k] + D u[k]$$

where $x \in \mathbb{R}^{N_x}$ is the state vector, $u \in \mathbb{R}^{N_u}$ is
the input vector, $y \in \mathbb{R}^{N_y}$ is the output vector,
and $A, B, C, D$ are system, input, output and feedforward matrices of
sizes $(N_x \times N_x)$, $(N_x \times N_u)$, $(N_y \times N_x)$,
and $(N_y \times N_u)$ respectively.

Following the design philosophy established by `TransferFunction` and `Matrix`,
`StateSpace` is a **standalone, generic container** built directly on top of
four generic storage backends (`Sa`, `Sb`, `Sc`, `Sd`) implementing the
`Storage` trait, rather than storing four `Matrix<T, R, C, S>` fields
directly. Avoiding heap allocation is the crate-wide `no_alloc` rule (see
[README](../../README.md)), not a `StateSpace`-specific decision; what is
specific here is *why* raw storage backends are used instead of full `Matrix`
wrappers: it avoids redundant type-wrapper parameters on every field and it
lets each matrix ($A$, $B$, $C$, $D$) opt into a different storage strategy
independently without changing the `StateSpace` signature (see §6, Alternative
C). The concrete payoff: a zero-sized storage backend can specialize a
structurally zero $D$ matrix, eliminating one `GEMM` per `step()` for any
strictly proper plant.

Among the four Rust control/estimation crates surveyed during research
(§11.1 refs. 23–26), none combine a no_std/no_alloc target, compile-time-sized
state dimensions and an LTI container offering interconnection,
discretization and structural analysis — this is a statement about that
survey, not a claim of exhaustive ecosystem coverage. The two crates providing a
genuine
state-space model type (`control_systems_torbox`, `rdesarz/control-sys-rs`)
are both `nalgebra`-backed with dynamically-sized matrices; the two
embedded-first `no_std` crates surveyed (`sunsided/minikalman-rs`,
`strawlab/adskalman-rs`) provide statically allocated filter buffers and
matrix operations but no LTI model type at all — the transition matrix is
supplied directly by the caller with no notion of sample time, discretization,
or interconnection algebra. Every embedded user of those crates therefore
hand-discretizes a continuous plant offline and hardcodes the resulting
$A_d, B_d$ (§11.1 refs. 23–26). `StateSpace` fills exactly that gap.

`StateSpace` still leverages the high-level `Matrix` type and zero-copy
`MatrixView` wrappers to execute linear algebra operations safely and
conveniently, while retaining direct access to lower-level Peano dimension
traits (`Dim`), storage traits and BLAS kernels.

This architecture achieves:

1. **Zero Dynamic Allocation (`#![no_std]`)**: Storage backends can be inline
   arrays, static ROM tables or borrowed slice views.
2. **Heterogeneous Storage**: Each matrix ($A, B, C, D$) can utilize a distinct
   storage type (e.g., stack-allocated `ArrayStorage` for $A$ and a
   zero/identity virtual storage view for $D$).
3. **Safe High-Level Algebra**: Exposes matrix views (`MatrixView` /
   `MatrixViewMut`) and leverages `Matrix` operations for state propagation,
   system interconnection and linear transformations.
4. **Compile-Time Dimension Safety**: Enforces matrix dimension
   compatibility ($N_x, N_u, N_y$) at compile time using Peano arithmetic (
   `DimAdd`, `DimSub`, `DimMul`).

---

### 2. Requirements

#### 2.1 Functional Requirements

- **FR-1 — Decoupled Storage Parameterization**: `StateSpace` must accept four
  independent `Storage` backends for $A, B, C, D$; descriptor-form
  systems ($E\dot{x} = Ax + Bu$) are out of scope (§9).
- **FR-2 — Domain Encoding**: The type must encode continuous vs. discrete
  domain via an optional sampling period `sample_time: Option<T>`, a runtime
  rather than type-level discriminant (§9).
- **FR-3 — Interoperability with `Matrix` & BLAS Kernels**: High-level
  operations must be able to use the `Matrix` abstraction while underlying data
  access maps directly to `Storage` and BLAS level 1/2/3 subprograms.
- **FR-4 — Time-Domain State Propagation**: Advance the discrete
  state ($x[k+1] = Ax[k]+Bu[k]$) and evaluate the continuous
  derivative ($\dot{x}(t) = Ax(t)+Bu(t)$), each with matching output equations.
- **FR-5 — System Interconnections**: Provide compile-time dimension-checked
  series, parallel and feedback interconnections, each with state capacity
  bound $N_{x1}+N_{x2}$ via `DimAdd`; feedback is fallible (§4.5, §5.3).
- **FR-6 — Discretization & Continuous Transformations**: Provide Zero-Order
  Hold via augmented-matrix exponentiation (Van Loan, 1978) and a fallible
  bilinear (Tustin) transform with optional pre-warping (§4.5, §5.4).
- **FR-7 — Canonical Form Transformations & Structural Analysis**: Provide
  CCF/OCF conversions, $z=Tx$ similarity transforms and definitional
  controllability/observability matrix construction, scoped to low-to-moderate
  system order (§5.5).
- **FR-8 — Transfer Function Conversion**: Bidirectional conversion between
  `StateSpace` and `TransferFunction` ($H(s) = C(sI-A)^{-1}B+D$), with SISO as
  the primary target; MIMO is future work (§9).

#### 2.2 Non-Functional Requirements

- **NFR-1 — Zero Dynamic Allocation**: All operations must execute
  deterministically without heap allocation (`Vec`, `Box`).
- **NFR-2 — Zero-Cost Abstraction**: Storage abstraction and Peano dimension
  bounds must monomorphize completely, matching hand-optimized raw array
  operations.
- **NFR-3 — Contiguity-Gated Slice & Matrix Accessors**: Safe slice/matrix-view
  accessors (`a_slice()`, `a_matrix()`) are exposed only if the backend
  implements `ContiguousStorage` (§4.3).

#### 2.3 Constraints

- **C-1 — Dimension Bounds**: State, input and output dimensions (`NX`, `NU`,
  `NY`) must each satisfy `Dim >= 1`, with $A$/$B$/$C$/$D$ conforming to strict
  `NX`/`NU`/`NY` layout rules.
- **C-2 — Derived-Dimension Peano Preconditions**: FR-5/FR-6/FR-7's derived
  dimensions ($N_{x1}+N_{x2}$, $N_x+N_u$, $N_x \cdot N_u$, $N_y \cdot N_x$) must
  each independently satisfy the `U32` Peano ceiling (`matrix-design.md` §2.3);
  none are enforced by a documented precondition today (§9).

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
  `MatrixViewMut<'a, T, R, C>`, `MatrixSlice<'a, T, R, C>` and
  `MatrixSliceMut<'a, T, R, C>` for safe, high-level matrix operations.
  These accessors (§4.3) are layout-agnostic per `storage-trait-design.md`
  FR-6: `a_matrix()`/`b_matrix()`/`c_matrix()`/`d_matrix()` and the
  arithmetic `step()`/`derivative()` build on top of never assume
  `Sa`/`Sb`/`Sc`/`Sd` are column-major — swapping any one of them for a
  row-major backend requires no change here.
- **`crate::math::subprograms`**: BLAS Level 1/2/3 kernels (`GEMV`, `GEMM`,
  `AXPY`). Solves (§4.4, §5.3, §5.5) are performed via forward/backward
  substitution over triangular factors (`matrix-design.md` §4.10.1), not a
  dedicated `TRSM` trait — no such trait exists in `subprograms.rs` or is
  specified elsewhere in this document family; a prior revision of this
  section referenced one in error.

Transcendental operations required by §5.4/§5.5 (matrix exponential scaling
factors, `tan` for Tustin pre-warping) are bounded against `Float`
in `crate::math::num_traits`. This bound is provisional in a stronger sense
than "pending re-approval": `num-traits-design.md` has not yet been through
`/cr-research` or `/cr-design-doc` at all (per the project's gated
pipeline) and will be revised independently of this document (§9).

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

`MatrixView`/`MatrixViewMut` are themselves contiguous, unit-stride storage
backends and implement `ContiguousStorage`/`ContiguousStorageMut`
(`matrix-design.md` §5.4). `StateSpaceView`'s `a_matrix()` (§4.3) therefore
constructs a `MatrixSlice` directly over the existing borrowed view rather
than performing a redundant copy — the accessor borrows straight through, it
does not re-wrap.

#### 4.3 Safe `Matrix` Integration & Zero-Copy Matrix Views

While data is stored inside storage backends (`Sa`, `Sb`, `Sc`, `Sd`),
`StateSpace` exposes methods to treat each backend as a zero-cost `MatrixView`
or `MatrixViewMut`, enabling full reuse of `Matrix` operations (multiplication,
addition, transposition, solver routines). Per NFR-3, these accessors are
bounded on `ContiguousStorage`/`ContiguousStorageMut`, not the weaker
`Storage`/`StorageMut` — a non-contiguous backend cannot safely yield the
borrowed slice `MatrixView` requires without copying:

```rust
impl<T, NX: Dim, NU: Dim, NY: Dim, Sa, Sb, Sc, Sd> StateSpace<T, NX, NU, NY, Sa, Sb, Sc, Sd>
where
    Sa: ContiguousStorage<T, NX, NX>,
    Sb: ContiguousStorage<T, NX, NU>,
    Sc: ContiguousStorage<T, NY, NX>,
    Sd: ContiguousStorage<T, NY, NU>,
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

A general (non-contiguous) accessor path is left unspecified in this
revision; callers holding a non-contiguous backend must materialize an owned
`Matrix` explicitly rather than obtaining a borrowed view.

#### 4.4 Error Handling

Following the crate-wide error strategy, fallible operations return
`Result<T, Error>` via a crate-local `thiserror` enum rather than panicking.
Two operations are fallible:

- **Feedback (§5.3)**: forms the loop
  matrix $F = I - \text{sign} \cdot D_2 D_1 \in \mathbb{R}^{N_u \times N_u}$
  and solves against it rather than inverting explicitly, matching
  `python-control`'s `feedback()` (`control/statesp.py`, §11.1 ref. 5).
  $F$ singular to working precision is a
  genuine closed-loop ill-posedness, not an internal error.
- **Tustin Discretization (§5.4)**: solves against $(I - \frac{T_s}{2}A)$,
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
established for `Matrix::invert`/`Matrix::solve`
(`LinAlgError::SingularMatrix`, `matrix-design.md` §4.9.2) rather than
introducing a parallel mechanism; the working-precision tolerance is a
parameter of the solve, not a fixed constant, matching MATLAB's `ctrbf`
default of $10 \cdot n \cdot \lVert A \rVert_1 \cdot \varepsilon$
(MathWorks, `ctrbf`, §11.1 ref. 13) as a precedent for how such a tolerance
should be expressed.

Near-singular (but not exactly singular) evaluation of $H(s)$ near a pole of
$A$ (§5.5) is *not* treated as a constructor- or call-time error — a valid
`StateSpace` may still be evaluated at an ill-conditioned frequency point,
which is documented behavior (§5.5) rather than a distinct error variant,
matching the posture already adopted for `TransferFunction::evaluate_complex`
(`transfer-function-design.md` §4.5).

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

`step()` is defined regardless of `sample_time` and its name asserts
discreteness that the type does not enforce; calling it on a model
constructed with `sample_time: None` compiles and silently reinterprets
$\dot{x}$ as $x[k+1]$. This is a known gap in FR-2's runtime domain encoding,
not resolved in this revision (§9, Domain Encoding Ambiguity).

##### Continuous State Derivative ($\dot{x} = A x + B u$)

Evaluates state derivative $\dot{x}(t)$ for numerical integration routines (
e.g., Runge-Kutta 4th Order). Symmetric to `step()` and subject to the same
domain-mismatch caveat in reverse.

#### 5.3 System Interconnections

##### Series (Cascade) Connection

Connecting output of System 1 ($N_{x1}, N_u, N_y$) to input of System
2 ($N_{x2}, N_y, N_{out}$):
Total state size: $N_x = N_{x1} + N_{x2}$.

$$\begin{bmatrix} \dot{x}_1 \\ \dot{x}_2 \end{bmatrix} = \begin{bmatrix} A_1 & 0 \\ B_2 C_1 & A_2 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} + \begin{bmatrix} B_1 \\ B_2 D_1 \end{bmatrix} u_1$$
$$y = \begin{bmatrix} D_2 C_1 & C_2 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} + D_2 D_1 u_1$$

Matches `python-control`'s `__mul__` block assembly exactly (`A = [[other.A, 0],
[self.B @ other.C, self.A]]`, `B = [other.B; self.B @ other.D]`,
`C = [self.D @ other.C, self.C]`, `D = self.D @ other.D`; `control/statesp.py`,
§11.1 ref. 5), including the state-ordering convention.

##### Parallel Connection

Connecting two systems with identical input and output
dimensions ($N_{x1} + N_{x2}$ states):
$$A_{par} = \begin{bmatrix} A_1 & 0 \\ 0 & A_2 \end{bmatrix}, \quad B_{par} = \begin{bmatrix} B_1 \\ B_2 \end{bmatrix}, \quad C_{par} = \begin{bmatrix} C_1 & C_2 \end{bmatrix}, \quad D_{par} = D_1 + D_2$$

Matches `python-control`'s `__add__` (`blkdiag(A1, A2)`, stacked $B$, stacked
$C$, summed $D$; `control/statesp.py`, §11.1 ref. 5) exactly.

##### Feedback Connection

Closed-loop negative/positive feedback around system $S_1$ and
controller/feedback system $S_2$. Unlike series and parallel, this is a
**fallible** operation (§4.4): form the loop
matrix $F = I - \text{sign} \cdot D_2 D_1 \in \mathbb{R}^{N_u \times N_u}$,
reject with `StateSpaceError::SingularLoopMatrix` if $F$ is singular to
working precision, then solve (never invert explicitly)
$[E_{D_2} \mid E_{C_2}] = F^{-1}[D_2 \mid C_2]$ as a single factorization
reused for both blocks, matching `python-control`'s
`feedback()` (`control/statesp.py`, §11.1 ref. 5). The resulting $A$, $B$,
$C$, $D$ blocks are assembled from products of $B_1/B_2$
with $E_{D_2}/E_{C_2}$. This mirrors the crate's existing preference for
triangular solves over explicit inversion (§5.5): the reference
implementation applies the same substitution at every point a textbook
formula shows an inverse — `feedback()` uses `solve(F, [D2 | C2])` rather
than `inv(F)` and `horner()` (§5.5) uses `solve(x*I - A, B)` rather than
`inv(sI - A)` (`control/statesp.py`, §11.1 ref. 5).

#### 5.4 Discretization (ZOH & Bilinear)

Converts a continuous `StateSpace` model ($T_s = \text{None}$) into a discrete
`StateSpace` model ($T_s = \text{Some}(T_s)$).

##### Zero-Order Hold (ZOH)

Form the augmented
matrix $M = \begin{bmatrix} A & B \\ 0 & 0 \end{bmatrix} \in \mathbb{R}^{(N_x + N_u) \times (N_x + N_u)}$
and compute its
exponential $e^{M T_s} = \begin{bmatrix} A_d & B_d \\ 0 & I \end{bmatrix}$
via scaling-and-squaring with Padé approximation. The augmented-matrix
construction is attributed to **Van Loan (1978)**, "Computing integrals
involving the matrix exponential" (§11.1 ref. 1), the origin of the
block-triangular auxiliary-matrix technique used here. Moler & Van Loan
(2003) (§11.1 ref. 2) is a comparative survey of matrix-exponential
*algorithms* and does not itself adjudicate the augmented-matrix
reformulation of the ZOH integral; the two are complementary, not
competing — Van Loan (1978) determines *what* is exponentiated,
scaling-and-squaring with Padé determines *how* the exponential of $M$ is
computed.

The augmented-matrix formulation is the correct choice because it never
inverts $A$: the closed-form alternative $B_d = A^{-1}(A_d - I)B$ requires $A$
nonsingular, which fails for every pure-integrator or kinematic model —
including this document's own §7.2 validation example
($A = \begin{bmatrix} 0 & 1 \\ 0 & 0 \end{bmatrix}$, nilpotent).

Two refinements are load-bearing for a from-scratch implementation:

- **Padé degree selection** (Higham, 2005, §11.1 ref. 3): sharp
  backward-error bounds select the Padé degree from $m \in \{3, 5, 7, 9, 13\}$
  via precision-dependent thresholds $\theta_m$. Degree 13 is optimal for
  IEEE double precision; single precision — the crate's primary target and
  the type used in §7.2's own example — admits a substantially lower optimal
  degree. The degree and scaling rule are therefore precision-dependent
  parameters this revision does not yet fix (§9).
- **Overscaling correction** (Al-Mohy & Higham, 2009, §11.1 ref. 4): a
  norm-based scaling rule keyed on $\lVert M \rVert$ overscales
  whenever $\lVert B \rVert \gg \lVert A \rVert$, since $M$'s block-triangular
  structure gives $\lVert M \rVert \ge \max(\lVert A \rVert, \lVert B \rVert)$
  and $B$ carries input-gain units unrelated to $A$'s time constants — the
  same light-years-vs-millimeters ill-scaling MATLAB documents for
  state-space models generally (MathWorks, *Scaling State-Space Models*,
  §11.1 ref. 17). Keying the scaling rule
  on $\{\lVert M^k \rVert^{1/k}\}$ instead of $\lVert M \rVert$ self-corrects
  for this, since the former is asymptotically governed by the spectral
  radius of $A$ (Al-Mohy & Higham, 2009, §11.1 ref. 4).

##### Bilinear (Tustin) Transform

$$A_d = (I - \tfrac{T_s}{2} A)^{-1} (I + \tfrac{T_s}{2} A)$$

Uses matrix inversion / triangular solver for $(I - \tfrac{T_s}{2} A)^{-1}$ (
Ogata, 2010). This is a **fallible** operation
(§4.4): $(I - \tfrac{T_s}{2}A)$ is singular exactly when $\tfrac{2}{T_s}$ is
an eigenvalue of $A$ and MATLAB documents the discrete-side symptom directly
— Tustin "is not defined for systems with poles at $z = -1$ and is
ill-conditioned for systems with poles near $z = -1$" (MathWorks,
*Continuous-Discrete Conversion Methods*, §11.1 ref. 10). MATLAB further
states that state vectors
are **not preserved** across a Tustin conversion; a Tustin-discretized
model's state is not the same physical quantity as the continuous model's
state and callers must not feed a continuous initial condition directly
into a Tustin-discretized model. Frequency pre-warping is optional but not
merely cosmetic — plain Tustin's frequency shift is documented as
"unacceptable for many applications" (`transfer-function-design.md` §5.4,
citing MathWorks *Discretize a Compensator*).

#### 5.5 Canonical Transformations & Transfer Function Equivalences

##### Similarity Transformation

**API commitment**: given $T \in \mathbb{R}^{N_x \times N_x}$, under the
convention $z = Tx$:
$$A' = T A T^{-1}, \quad B' = T B, \quad C' = C T^{-1}, \quad D' = D$$
implemented by solving $T A' = A T$-style systems rather than forming
$T^{-1}$ explicitly, matching this document's general solve-over-inversion
preference (§4.4, §5.3).

The $z = Tx$ convention is pinned explicitly, not left implicit, because it
is genuinely contested: the mirror-image textbook convention $x = Tz$
yields $A' = T^{-1}AT$, $B' = T^{-1}B$, $C' = CT$ and `python-control`
resolves the ambiguity with an explicit runtime `inverse` flag rather than
picking one silently (`control.similarity_transform`, §11.1 ref. 7).
Matches MATLAB's `ss2ss` (§11.1 ref. 18). A cross-validation harness (§7.1)
must pin this same convention. Accuracy scales with $\kappa(T)$ and is
exact only for orthogonal $T$; controllable/observable canonical form
(below) is the worst case for this, since its $T$ is built from the
controllability/observability matrix and inherits that matrix's
conditioning growth.

##### Controllability & Observability Matrices

**API commitment**: implement explicit power construction —
$$\mathcal{C} = \begin{bmatrix} B & AB & A^2B & \dots & A^{N_x-1}B \end{bmatrix}, \qquad \mathcal{O} = \begin{bmatrix} C \\ CA \\ CA^2 \\ \vdots \\ CA^{N_x-1} \end{bmatrix}$$
— scoped to definitional/structural use and small, hand-verifiable systems.
It is explicitly **not** the rank/controllability-testing method for
moderate $N_x$; the orthogonal-staircase alternative below is future work,
not this revision's implementation.

**Why**: MATLAB documents `obsv` as "not recommended for control design"
and $\mathcal{O}$ as "numerically singular for most systems with more than
a handful of states" (§11.1 ref. 11); `ctrb` guidance is the same, pointing
to `ctrbf`/`obsvf` (§11.1 refs. 12–13). Mechanism: for diagonalizable $A$,
$\mathcal{C}$'s $k$-th block scales with $\lambda_{max}^k/\lambda_{min}^k$,
so conditioning grows with the eigenvalue-spread raised to the $(N_x-1)$
power — consuming single-precision epsilon around $N_x \approx 8$ and
double-precision epsilon around $N_x \approx 17$, tighter than this
document's general inversion guidance (below). An orthogonal-staircase
construction (Van Dooren, 1978/1981, per MATLAB `ctrbf`/`obsvf`, §11.1
ref. 13) would share this section's Hessenberg-reduction kernel and, unlike
$\mathcal{C}$/$\mathcal{O}$, stays inside the Peano ceiling for any
expressible $N_x$ — recorded as future work (§9), not built here.

##### Transfer Function Conversion

**API commitment**: implement
$$H(s) = C (s I - A)^{-1} B + D = \frac{C \text{adj}(sI - A) B + D \det(sI - A)}{\det(sI - A)}$$
via Hessenberg reduction plus triangular solves (never explicit inversion),
scoped to **SISO** for this revision.

**Why Hessenberg over the alternatives** — two reasons: (1) *Amortization*
— $(sI-A)$ needs re-factoring at every frequency point under direct LU
(cubic cost/point), but Hessenberg reduction is shift-invariant, so it runs
once and each subsequent solve is quadratic (§11.1 ref. 21); a $K$-point
Bode sweep becomes one cubic reduction plus $K$ quadratic solves instead of
$K$ cubic solves. (2) *Determinism* — the Schur-form alternative is
likewise shift-invariant but needs the iterative QR eigenvalue algorithm
(unbounded iteration count, no static WCET bound, conflicting with this
crate's ISO 26262/DO-178C determinism posture, §10.3); Householder
Hessenberg reduction is finite and iteration-free. SLICOT's `TB05AD` —
`python-control`'s delegate when available — follows the same
reduce-once-reuse-across-frequencies structure (§11.1 ref. 20).

**Conditioning, briefly**: near-pole evaluation is better-conditioned than
the coefficient-based `TransferFunction` path with respect to *pole
representation* (poles are eigenvalues of $A$ via backward-stable
orthogonal methods, not roots of a coefficient vector —
`transfer-function-design.md` §5.5), but **not** with respect to
*proximity to a pole*: as $s \to \lambda_i(A)$, the solve's condition
number grows like $1/|s-\lambda_i|$, the same blowup the `TransferFunction`
path's Horner evaluation shows near a root of its denominator
(`transfer-function-design.md` §5.2). Neither conversion direction is a
lossless round trip. Scaling/balancing (SLICOT/MATLAB `prescale`, §11.1
refs. 16–17) and MIMO support (blocked on an SLICOT-`tb04ad`-equivalent and
colliding with the same Peano ceiling as $\mathcal{C}$/$\mathcal{O}$,
§11.1 ref. 8) are both left as future work (§9), not implemented here.

---

### 6. Alternatives

| Alternative                                             | Description                                                                           | Pros                                                                                                                                                                                                                                  | Cons                                                                                                                                                                                                                                                   |
|:--------------------------------------------------------|:--------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **A. Monolithic Storage Field**                         | Storing a single combined block matrix $\begin{bmatrix} A & B \\ C & D \end{bmatrix}$ | Single contiguous allocation                                                                                                                                                                                                          | Cannot specialize a structurally zero $D$ (common for strictly proper plants); the series/parallel/feedback formulas (§5.3) assemble $A, B, C, D$ from independently shaped sub-blocks and would need awkward strided writes into one contiguous block |
| **B. Wrapping 4 `Matrix` Structs Directly**             | Storing `a: Matrix<T, NX, NX, Sa>`, `b: Matrix...`                                    | Built-in matrix methods on fields                                                                                                                                                                                                     | Redundant type wrapper parameters, extra boilerplate vs storing raw `Storage` backends; the zero-$D$ specialization is awkward since the wrapper's own type parameters still have to be named                                                          |
| **C. Generic Storage Backends + `MatrixView` (Chosen)** | Storing raw `Sa, Sb, Sc, Sd` backends and creating `MatrixView` wrappers on demand    | Maximum flexibility (`#![no_std]`, ROM, stack, zero-matrix storage); expresses a zero-sized $D$ specialization that eliminates a `GEMM` per `step()`, which neither A nor B can express as cleanly; symmetric with `TransferFunction` | Requires `a_matrix()`/etc. accessors gated on `ContiguousStorage` (§4.3) for high-level matrix algebra                                                                                                                                                 |

The strongest evidence for Alternative C is the zero-$D$ specialization
above. The same "avoid wrapping the natural lower-level peer type" decision
is made independently by `transfer-function-design.md` §6 for
`TransferFunction` vs. `Polynomial`, for compatible reasons.

---

### 7. Verification & Validation

#### 7.1. Verification Strategy

1. **Unit Testing (`src/state_space/tests/`)**:
    - Verify step response of known continuous and discrete systems against
      analytical solutions.
    - Verify similarity transformations preserve transfer function frequency
      response $H(j\omega)$, with the $z = Tx$ convention (§5.5) pinned
      explicitly rather than assumed when comparing against a reference
      implementation.
    - Verify `feedback()` and Tustin discretization return
      `Err(StateSpaceError::SingularLoopMatrix)` /
      `Err(StateSpaceError::SingularDiscretizationOperator)` on constructed
      near-singular inputs (§4.4).
2. **Property-Based Testing (`proptest`)**:
    - Verify series and parallel interconnections maintain algebraic identity
      equivalence using QuickCheck random generator principles (Claessen &
      Hughes, 2000).
    - Verify discretization followed by continuous approximation converges
      as $T_s \to 0$.
3. **Cross-Validation**: Compare series/parallel/feedback block assembly,
   ZOH/Tustin discretization and similarity transforms against
   `python-control` and MATLAB reference outputs (§5.3, §5.4, §5.5).
4. **Fixed-Point Recursion Testing**: A single-step comparison against an
   analytical solution does not exercise `step()`'s behavior as a recursion.
   Repeated invocation re-quantizes the state vector to the storage format
   every iteration regardless of accumulator width, so quantization residues
   are fed back through $A$ and integrate rather than decay for any system
   with spectral radius near 1 — including this document's own §7.2 kinematic
   example, whose eigenvalues are both exactly 1 (Mullis & Roberts, 1976;
   Hwang, 1977). A long-horizon fixed-point recursion test (bounded roundoff
   accumulation, no overflow limit cycle under saturating arithmetic) is
   required in addition to single-step validation.
5. **HIL Verification (`control-rs-hil`)**:
    - Test real-time execution of discrete state
      propagation $x[k+1] = A x[k] + B u[k]$ on microcontroller targets (
      Cortex-M) within deterministic deadline bounds (DO-178C, ISO 26262).

#### 7.2. Validation Strategy

- **Kinematic Object Tracking**:
  A practical use of a discrete `StateSpace` model is predicting the future
  position and velocity of an object given its acceleration. This models the
  kinematic system $x[k+1] = A x[k] + B u[k]$ where $x$ contains position and
  velocity and $u$ is the acceleration input. Its system matrix $A$ is
  singular (nilpotent), which is precisely the case the augmented-matrix ZOH
  formulation (§5.4) handles and the closed-form alternative does not.

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
  require $32 \times 32 = 1024$ elements for matrix $A$, exactly
  `matrix-design.md` §2.3's per-matrix budget. Storing via `Storage` enables
  static buffer placement or borrowed views, preventing embedded stack
  overflow for $A$/$B$/$C$/$D$ themselves.
- **ZOH Is Not Accommodated at $N_x = 32$**: the augmented matrix $M$ is
  $(N_x + N_u)$-square (§5.4), so a 32-state system cannot be ZOH-discretized
  with even a single input — the `DimAdd` bound underlying $M$ simply fails
  to resolve, since no `U33` dimension alias exists. This is a compile-time
  wall, not a runtime concern and it is invisible at the type-signature
  level until the bound fails (§9).
- **ZOH Workspace Multiplier**: scaling-and-squaring at a competitive Padé
  degree requires on the order of six additional $(N_x+N_u)$-square
  temporaries beyond $M$ itself. A 24-state, 8-input system therefore needs
  roughly $7 \times 32 \times 32 \times 4\ \text{bytes} \approx 28\ \text{KB}$
  of `f32` transient workspace — several times the single-matrix budget above,
  and a materially larger stack-overflow risk on a Cortex-M target than the
  storage of $A$/$B$/$C$/$D$ alone (`storage-trait-design.md` C-2;
  `matrix-design.md` §2.3; Higham, 2005, §11.1 ref. 3, for the
  temporary-count estimate at Padé degree 13).
- **Interconnection Depth**: series/parallel/feedback each produce
  an $N_{x1}+N_{x2}$-state result. Cascading four 8-state
  blocks — an ordinary controller-plant-filter arrangement — reaches the
  32-state Peano ceiling; a fifth stage cannot be expressed. The combined
  system's $A$ matrix at that size is a 4 KB value returned by `step()`'s
  owning signature (§5.2), which `matrix-design.md` §5.1 already flags as a
  pattern to avoid for heavy operations on stack-constrained targets.
- **Matrix Exponential Performance**: ZOH discretization of $A$ uses
  scaling-and-squaring Padé approximation using BLAS level 3 `GEMM` kernels (
  see §5.4 for the specific Padé-degree and overscaling considerations).

---

### 9. Risks & Open Questions

> [!IMPORTANT]
> **Derived-Dimension Peano Ceiling Breaches**
> FR-5 ($N_{x1}+N_{x2}$), FR-6 ($N_x+N_u$) and FR-7 ($N_x \cdot N_u$,
> $N_y \cdot N_x$) each derive a dimension larger than $N_x$ that must
> independently fit the crate's `U32` Peano ceiling (`storage-trait-design.md`
> C-2). None of these preconditions is currently documented; each surfaces as
> an unresolvable trait bound rather than a diagnosed limit and this
> document's own §8 example ($N_x = 32$) cannot be ZOH-discretized at all.
> This is the finding most likely to force a structural change and it is
> independent of the storage architecture chosen in §6 — Alternatives A and B
> would hit it identically. Whether to raise the Peano ceiling or to document
> per-FR preconditions is an open question for the next revision.

> [!IMPORTANT]
> **Matrix Inversion & Numerical Stability**
> Explicit matrix inversion for Tustin discretization or transfer-function
> conversion is ill-conditioned for high state dimensions. The solve-over-
> inversion preference (§4.4, §5.3, §5.5) is fully resolved by reference-
> implementation precedent and should be treated as a design commitment.
> The *threshold* is not single-valued, however: general
> inversion/Tustin conditioning and explicit controllability/observability
> construction (§5.5) degrade at different rates and must be stated
> separately — informally, "$N_x > 10$" for the former and roughly
> "$N_x \gtrsim 8$ on `f32`" for the latter (§5.5). Both are precision-
> dependent and the exact crossover on real `f32` targets is a measurement
> item for the implementation/HIL phase, not something resolvable from
> literature alone.

> [!NOTE]
> **Sparse and Structured Storage Optimization**
> A structurally zero $D$ matrix is a common, compile-time-knowable,
> immediately resolvable win — it is the strongest concrete justification for
> Alternative C (§6) and should be an explicit specialization rather than an
> open-ended deferral. A sparse or companion-form $A$ is a different
> question and should stay deferred: the sparse $A$ forms most worth
> optimizing are companion/canonical forms, which §5.5's conditioning
> evidence recommends avoiding as a computational representation, not
> optimizing. MATLAB's own sparse precedent (`sparss`) is a genuinely
> separate model type built for large FEM-derived systems far outside this
> crate's embedded envelope and it additionally carries a descriptor $E$
> matrix — see below.

> [!NOTE]
> **`num-traits-design.md` Dependency Is Provisional**
> §5.4/§5.5's `Float` bound (matrix exponential scaling, Tustin pre-warping)
> depends on `num-traits-design.md`, which has not yet been through
> `/cr-research` or `/cr-design-doc` and will be revised independently of
> this document. Not a blocker for this document's own approval, but
> implementation must confirm the bound still resolves once that document
> completes its own gated-pipeline pass.

> [!NOTE]
> **Descriptor Form (`E` Matrix) Scope**
> Descriptor/DAE systems ($E\dot{x} = Ax + Bu$) are out of scope for this
> revision (FR-1). Unlike a sparse storage backend, adding $E$ later changes
> the `StateSpace` type signature rather than only adding a new `Storage`
> implementation. Whether descriptor form belongs on a future roadmap is an
> open scoping question, not a technical blocker.

> [!NOTE]
> **Domain Encoding Ambiguity**
> FR-2's runtime `sample_time: Option<T>` does not prevent calling `step()`
> on a continuous-time model or `derivative()`-style evaluation on a discrete
> one (§5.2); both compile and silently produce a semantically wrong result.
> `python-control` resolves this behaviorally with a single `dt`-aware method;
> `rdesarz/control-sys-rs` resolves it structurally with two distinct types (
> `ContinuousStateSpaceModel`/`DiscreteStateSpaceModel`). Neither approach is
> adopted in this revision; whether to introduce a type-level or ZST domain
> marker, consistent with the crate's general preference for compile-time
> enforcement (`matrix-design.md` §6.1), is left open.

> [!NOTE]
> **No Scaling/Balancing Story**
> Both reference toolboxes treat state-vector scaling as an automatic
> precondition for accurate frequency-domain computation (MATLAB `prescale`;
> SLICOT `TB05AD` balancing before Hessenberg reduction, §5.5). This document
> has no equivalent concept. A power-of-two diagonal balancing transform is
> rounding-error-free and embedded-friendly, but MATLAB itself cautions that
> naive balancing of $A$ alone can be harmful, which is why MATLAB ships the
> more careful `prescale` rather than raw balancing. Adopting a scaling step
> is left as future work.

> [!NOTE]
> **Fixed-Point Recursion Noise & Limit Cycles**
> A wide-accumulator convention for a single `GEMM` (e.g. CMSIS-DSP's 64-bit
> `arm_mat_mult_q31` vs. its 32-bit `arm_mat_mult_fast_q31`, §11.1 ref. 22) is
> necessary but not sufficient for `step()`, which is a recursion: the state
> vector is re-quantized to the storage format every iteration regardless of
> accumulator width, so residues are fed back through $A$ and integrate
> rather than decay for lightly damped or integrating modes. Per Mullis &
> Roberts (1976) and Hwang (1977), the steady-state roundoff-noise gain of a
> fixed-point state-space recursion is a property of the *realization* — the
> similarity transform of §5.5 — not of the transfer function it implements,
> and minimizing it also yields guaranteed immunity from autonomous overflow
> limit cycles under an $l_2$-norm dynamic-range scaling constraint. This
> puts the similarity transform in a different light than an analysis
> convenience: it is a numerical-quality lever for fixed-point deployments.
> Overflow handling in this path must saturate rather than wrap. §7.1 adds
> the corresponding long-horizon test requirement; the realization-selection
> question itself is left open.

> [!NOTE]
> **FR-8 Scope (MIMO Transfer Function Conversion)**
> This revision scopes FR-8 to SISO (§5.5). Extending to MIMO requires either
> an algorithm equivalent to SLICOT's `tb04ad` or an accepted non-minimal
> common-denominator representation whose storage collides with the Peano
> ceiling the same way $\mathcal{C}$/$\mathcal{O}$ do.
`transfer-function-design.md`
> should be revisited together with any future MIMO work, since it currently
> specifies only the SISO reverse path independently.

> [!NOTE]
> **On-Target vs. Host-Side ZOH Discretization**
> Given §8's workspace multiplier and Peano-ceiling interaction and given
> that every surveyed embedded-Rust precedent (`minikalman`, `adskalman`)
> has users hand-discretize a plant offline and hardcode the resulting
> $A_d, B_d$, whether on-target ZOH discretization is a hard requirement at
> all — versus a host-side/offline operation whose output is const-baked
> into ROM — is an open scoping question that would, if resolved toward
> host-side-only, substantially relax both the workspace and dimension-
> ceiling concerns above.

---

### 10. Development Plan

| Task / Feature                                | Description                                                                                                                                                                                                                         | Estimated Effort |
|:----------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
| **Step 1: Core Struct & Constructors**        | Implement `StateSpace<T, NX, NU, NY, Sa, Sb, Sc, Sd>`, basic array/slice constructors and `StateSpaceError` (§4.4) in `src/state_space/mod.rs`.                                                                                     | 1 Day            |
| **Step 2: Matrix Views & Basic Operations**   | Implement `a_matrix()`/`b_matrix()`/`c_matrix()`/`d_matrix()` bounded on `ContiguousStorage` per NFR-3 (§4.3) and time-domain simulation (`step()`, derivative).                                                                    | 1 Day            |
| **Step 3: System Interconnections**           | Implement `series`, `parallel` and fallible `feedback` (loop-matrix solve, `StateSpaceError::SingularLoopMatrix`) with compile-time Peano dimension math.                                                                           | 2 Days           |
| **Step 4: Discretization**                    | Implement ZOH (Van Loan augmented matrix, scaling-and-squaring with precision-dependent Padé degree selection §5.4, Al-Mohy/Higham overscaling correction) and fallible Tustin (`StateSpaceError::SingularDiscretizationOperator`). | 4.5 Days         |
| **Step 5: Structural Analysis & Conversions** | Implement controllability/observability matrix generation (scoped as definitional, §5.5), similarity transforms ($z=Tx$) and Hessenberg-reduction-based SISO transfer function conversion.                                          | 3.0 Days         |
| **Step 6: Tests & Documentation**             | Unit tests, `proptest` suites, `python-control`/MATLAB cross-validation (two external reference implementations), long-horizon fixed-point recursion tests and crate-level documentation.                                           | 3.0 Days         |

---

### 11. References

#### 11.1. Practical

1. **Van Loan, C. F. (1978).** Computing integrals involving the matrix
   exponential. *IEEE Transactions on Automatic Control*, AC-23(3), 395–404.
   — Origin of the augmented block-triangular auxiliary-matrix technique
   underlying ZOH discretization (§5.4).
2. **Moler, C., & Van Loan, C. (2003).** Nineteen Dubious Ways to Compute the
   Exponential of a Matrix, Twenty-Five Years Later. *SIAM Review*, 45(1),
   3–49. — Comparative survey of matrix-exponential algorithms; motivates
   scaling-and-squaring with Padé as the computational method for whatever
   matrix is being exponentiated (§5.4).
3. **Higham, N. J. (2005).** The Scaling and Squaring Method for the Matrix
   Exponential Revisited. *SIAM J. Matrix Anal. Appl.*, 26(4), 1179–1193. —
   Precision-dependent Padé degree selection ($\theta_m$ thresholds); the
   algorithm underlying MATLAB's `expm` (§5.4).
4. **Al-Mohy, A. H., & Higham, N. J. (2009).** A New Scaling and Squaring
   Algorithm for the Matrix Exponential. *SIAM J. Matrix Anal. Appl.*, 31(3),
   970–989. — Overscaling correction directly relevant to the augmented
   matrix's block-triangular, norm-imbalanced structure (§5.4).
5. **python-control developers.** `control/statesp.py`, python-control
   library source.
   <https://github.com/python-control/python-control/blob/main/control/statesp.py>
   — Reference series/parallel/feedback block assembly (§5.3), Hessenberg-
   fallback frequency-response evaluation (§5.5).
6. **python-control developers.** `control.StateSpace`, python-control
   documentation.
   <https://python-control.readthedocs.io/en/latest/generated/control.StateSpace.html>
   — `sample()`/`c2d` discretization method set (§5.4, §9).
7. **python-control developers.** `control.similarity_transform`,
   python-control documentation.
   <https://python-control.readthedocs.io/en/latest/generated/control.similarity_transform.html>
   — Explicit `inverse` convention flag motivating this document's explicit
   $z=Tx$ statement (§5.5).
8. **python-control developers.** `control.ss2tf`, python-control
   documentation.
   <https://python-control.readthedocs.io/en/latest/generated/control.ss2tf.html>
   — MIMO conversion's Slycot (`tb04ad`) dependency, motivating FR-8's SISO
   scope (§5.5, §9).
9. **MathWorks.** `c2d` — Convert model from continuous to discrete time,
   MATLAB documentation.
   <https://www.mathworks.com/help/control/ref/dynamicsystem.c2d.html>
   — Full discretization method set and restrictions (§5.4, §9).
10. **MathWorks.** *Continuous-Discrete Conversion Methods*, MATLAB &
    Simulink documentation.
    <https://www.mathworks.com/help/control/ug/continuous-discrete-conversion-methods.html>
    — ZOH/FOH exactness, Tustin ill-conditioning near $z=-1$, state
    non-preservation under Tustin (§5.4).
11. **MathWorks.** `obsv` — Observability of state-space model, MATLAB
    documentation.
    <https://www.mathworks.com/help/control/ref/statespacemodel.obsv.html>
    — "Not recommended for control design" guidance and numerical-singularity
    statement (§5.5).
12. **MathWorks.** `ctrb` — Controllability of state-space model, MATLAB
    documentation.
    <https://www.mathworks.com/help/control/ref/statespacemodel.ctrb.html>
    — Ill-conditioning guidance, points to `ctrbf` (§5.5).
13. **MathWorks.** `ctrbf` — Compute controllability staircase form, MATLAB
    documentation. <https://www.mathworks.com/help/control/ref/ctrbf.html>
    — Orthogonal-staircase mitigation and rank tolerance convention (§5.5).
14. **MathWorks.** *State-Space Realizations* (`canon`, `compreal`,
    `modalreal`), MATLAB & Simulink documentation.
    <https://www.mathworks.com/help/control/ug/canonical-state-space-realizations.html>
    — Companion-form conditioning caveat (§5.5).
15. **MathWorks.** `canon` — (Not recommended) Canonical state-space
    realization, MATLAB documentation.
    <https://www.mathworks.com/help/ident/ref/dynamicsystem.canon.html>
    — "Not recommended" status for companion-form realization (§5.5).
16. **MathWorks.** `prescale` — Optimal scaling of state-space models, MATLAB
    documentation. <https://www.mathworks.com/help/control/ref/ss.prescale.html>
    — Motivates the scaling/balancing open question (§5.5, §9).
17. **MathWorks.** *Scaling State-Space Models*, MATLAB & Simulink
    documentation.
    <https://www.mathworks.com/help/control/ug/scaling-state-space-models.html>
    — Relative-accuracy metric and ill-scaling example (§5.4, §5.5).
18. **MathWorks.** `ss2ss` — State coordinate transformation for state-space
    model, MATLAB documentation.
    <https://www.mathworks.com/help/ident/ref/statespacemodel.ss2ss.html>
    — Canonical statement of the $z=Tx$ similarity-transform convention
    (§5.5).
19. **MathWorks.** `sparss` — Sparse first-order state-space model, MATLAB
    documentation. <https://www.mathworks.com/help/control/ref/sparss.html>
    — Reference precedent for sparse state-space as a distinct model type
    carrying a descriptor $E$ matrix (§9).
20. **python-control issue #116** / SLICOT `TB05AD`. Discussion of using
    SLICOT's `TB05AD` in `StateSpace.freqresp`.
    <https://github.com/python-control/python-control/issues/116> —
    One-time Hessenberg reduction (with balancing) reused across all
    frequency points (§5.5).
21. A Note on Shifted Hessenberg Systems and Frequency Response Computation,
    *ACM Transactions on Mathematical Software*.
    <https://dl.acm.org/doi/pdf/10.1145/2049673.2049676> — Amortized
    quadratic-cost shifted-Hessenberg solves for repeated frequency-response
    evaluation (§5.5).
22. **ARM Ltd.** *Matrix Multiplication*, CMSIS-DSP documentation.
    <https://arm-software.github.io/CMSIS-DSP/main/group__MatrixMult.html> —
    `arm_mat_mult_q31` (64-bit accumulator) vs. `arm_mat_mult_fast_q31`
    (32-bit accumulator) convention motivating the fixed-point recursion risk
    discussed in §9; a single wide-accumulator `GEMM` does not by itself
    bound the roundoff accumulated across repeated `step()` calls (§7.1, §9).
23. **`control_systems_torbox`** crate documentation (docs.rs).
    <https://docs.rs/control_systems_torbox> — Rust ecosystem data point: a
    `nalgebra`-`DMatrix`-backed (dynamically sized), `netlib-src`-dependent
    state-space type, structurally the opposite of this crate's `no_std`/
    compile-time-sized target (§1, §6).
24. **`rdesarz/control-sys-rs`**, "A Control System library implemented in
    Rust". <https://github.com/rdesarz/control-sys-rs> — Rust ecosystem data
    point: exposes distinct `ContinuousStateSpaceModel`/
    `DiscreteStateSpaceModel` types with an explicit `from_continuous_zoh()`
    bridge, cited as prior art for the domain-encoding question in §9.
25. **`sunsided/minikalman-rs`**, "Fixed- and floating-point Kalman filters
    for resource-constrained environments".
    <https://github.com/sunsided/minikalman-rs> — `no_std`-by-default,
    statically allocated, Q16.16 fixed-point-capable Kalman filter buffers
    with no LTI model type of their own (§1, §9).
26. **`strawlab/adskalman-rs`**, Kalman filter and RTS smoothing in Rust.
    <https://github.com/strawlab/adskalman-rs> — Second `no_std`-capable,
    `nalgebra`-backed Kalman/RTS data point with no discretization,
    interconnection or structural-analysis operations of its own (§1).

#### 11.2. Theoretical

27. **Kailath, T. (1980).** *Linear Systems*. Prentice-Hall. — Definitional
    source for controllability/observability matrices and canonical
    realizations (§5.5).
28. **Ogata, K. (2010).** *Modern Control Engineering* (5th ed.). Prentice
    Hall. — LTI state-space formulation, block-diagram interconnection
    algebra and bilinear (Tustin) discretization (§5.3, §5.4).
29. **Åström, K. J., & Murray, R. M. (2021).** *Feedback Systems: An
    Introduction for Scientists and Engineers* (2nd ed.). Princeton
    University Press. — Similarity transformations and closed-loop feedback
    derivations (§5.3, §5.5).
30. **Golub, G. H., & Van Loan, C. F. (2013).** *Matrix Computations* (4th
    ed.). Johns Hopkins University Press. — Complexity basis for triangular
    solves and Hessenberg reduction over explicit inversion (§5.5).
31. **Mullis, C. T., & Roberts, R. A. (1976).** Synthesis of minimum roundoff
    noise fixed point digital filters. *IEEE Transactions on Circuits and
    Systems*, 23(9). **Hwang, S. Y. (1977).** Minimum uncorrelated unit noise
    in state-space digital filtering. *IEEE Transactions on Acoustics,
    Speech and Signal Processing*, 25(4), 273–281. — Realization-dependent
    fixed-point roundoff-noise gain and overflow-limit-cycle immunity (§7.1,
    §9).
32. **Yang, S., & Jones, C. N. (2026).** Numerically Reliable Brunovsky
    Transformations. — Exponential condition-number growth of the standard
    companion-form transformation with system dimension (§5.5).

#### 11.3. Standards, Safety and Verification

33. **Claessen, K., & Hughes, J. (2000).** QuickCheck: A Lightweight Tool for
    Random Testing of Haskell Programs. *ACM SIGPLAN Notices*, 35(9), 268–279. —
    Property-based testing methodology driving `proptest` suites (§7.1).
34. **Rust Project Developers. (2024).** *The Rustonomicon: The Dark Arts of
    Advanced and Unsafe Rust Programming*. — Memory safety and pointer aliasing
    guarantees underpinning generic storage wrappers.
35. **ISO. (2018).** *ISO 26262-6:2018 Road vehicles — Functional safety — Part
    6: Product development at the software level*.
36. **RTCA / EUROCAE. (2011).** *DO-178C: Software Considerations in Airborne
    Systems and Equipment Certification*.
37. **IEEE Computer Society. (2008).** *IEEE Standard for Software and System
    Test Documentation* (IEEE Std 829-2008).

---

### 12. Revision History

| Revision | Date           | Author          | Description                                                                                                          |
|:---------|:---------------|:----------------|:---------------------------------------------------------------------------------------------------------------------|
| 1.0      | July 26, 2026  | @MitchellDScott | Initial draft outline of the `StateSpace` model design document.                                                     |
| 1.1      | July 26, 2026  | @MitchellDScott | Added inline academic citations and 3-tiered references section.                                                     |
| 1.2      | August 1, 2026 | @MitchellDScott | Separated the crate-wide `no_alloc` rule from the `StateSpace`-specific raw-storage-backend justification.           |
| 1.3      | August 2, 2026 | @MitchellDScott | Corrected ZOH citation; added conditioning caveat; pinned similarity-transform convention; narrowed FR-8 to SISO.    |
| 1.4      | August 2, 2026 | @MitchellDScott | Removed unsupported `TRSM` reference; sharpened dependency note; restructured §5.5; revised effort estimates upward. |
