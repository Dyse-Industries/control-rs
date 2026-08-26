# Transfer Function Type (Design Document)

![Date Badge](https://img.shields.io/badge/Date-August_25,_2026-blue)
![Status Badge](https://img.shields.io/badge/Doc%20Status-Approved-green)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

---

### 1. Introduction

The `TransferFunction` module provides a statically sized, zero-allocation
representation of rational transfer functions $H(s) = \frac{B(s)}{A(s)}$ (
continuous time) and $H(z) = \frac{B(z)}{A(z)}$ (discrete time) for linear
time-invariant (LTI) control systems and digital signal processing.

Primary usage scenarios:

- **Frequency-Domain Analysis**: Evaluating frequency sweeps ($H(j\omega)$
  or $H(e^{j\omega T_s})$) for Bode magnitude/phase plots and stability margin
  determination.
- **Block Diagram Interconnection**: Computing closed-loop transfer functions
  via series multiplication ($G_1 \cdot G_2$), parallel addition ($G_1 + G_2$),
  and negative feedback loops ($\frac{G}{1 + G H}$).
- **Continuous-to-Discrete Discretization**: Transforming continuous-time filter
  and plant models into discrete $Z$-domain implementations via Tustin bilinear
  transform (with optional pre-warping) and Zero-Order Hold (ZOH).
- **Canonical State-Space Realization**: Converting rational transfer functions
  into controllable, observable, or modal canonical state-space realizations for
  time-domain simulation and state estimation.

---

### 2. Requirements

#### 2.1. Functional Requirements

- **FR-1 — Rational SISO Transfer Function Representation**: Represents
  continuous ($G(s) = N(s)/D(s)$) and discrete ($G(z) = N(z)/D(z)$) single-input
  single-output (SISO) transfer functions as numerator and denominator
  polynomials with explicit sample time $T_s$ for discrete systems.
- **FR-2 — Frequency Response Evaluation**: Evaluates complex frequency
  response $G(j\omega)$ or $G(e^{j\omega T_s})$ across specified frequency
  points, computing magnitude and phase. Evaluation must execute in $O(N + D)$
  multiply-accumulate operations per point using Horner's method without dynamic
  allocations.
- **FR-3 — Rational System Algebra**: Evaluates series ($G_1 \cdot G_2$),
  parallel ($G_1 + G_2$), and negative feedback ($G / (1 + G H)$)
  interconnections via discrete polynomial convolution, with output polynomial
  orders determined at compile time.
- **FR-4 — System Discretization**: Converts continuous transfer functions to
  discrete form using Bilinear (Tustin, with optional pre-warping) and
  Zero-Order Hold (ZOH) methods, returning an error when the transformation is
  ill-conditioned.
- **FR-5 — State-Space Canonical Realization**: Converts transfer functions into
  equivalent controllable canonical or observable canonical `StateSpace` models
  with state dimension equal to the denominator degree $n = D - 1$.

#### 2.2. Non-Functional Requirements

- **NFR-1 — Deterministic Fixed-Memory Execution**: Frequency evaluation and
  rational arithmetic execute entirely within stack-allocated buffers with zero
  heap allocations.
- **NFR-2 — Real-Time Frequency Sweep Throughput**: Frequency response
  calculations maintain linear computational scaling across frequency sweeps
  without heap fragmentation.

#### 2.3. Constraints

- **C-1 — Properness Precondition**: Denominator coefficient capacity $D$ and
  numerator capacity $N$ must satisfy $D \ge 1$ and $D \ge N$ (denominator degree
  $n = D - 1 \ge m = N - 1$, proper transfer function).
- **C-2 — Non-Zero Leading Denominator**: Leading denominator coefficient $a_{D-1}$
  must be non-zero ($a_{D-1} \neq 0$).
- **C-3 — Capacity Bound**: Numerator and denominator polynomial capacities are
  bounded ($N, D \le 1024$) per `num-types-design.md` C-1.
- **C-4 — `#![no_std]` Environment**: Operates strictly in `#![no_std]` without
  standard library dependencies.

---

### 3. Technical Overview

`TransferFunction<T, N, D, Sn, Sd>` provides a domain-aware rational transfer
function representation parameterized over scalar type `T`,
numerator/denominator dimensions `N: Dim, D: Dim`, and single-column dense
storage backends
`Sn: DenseStorage<T, R = N, C = Const<1>>, Sd: DenseStorage<T, R = D, C = Const<1>>`.

The type operates directly on storage memory without wrapping intermediate
`Polynomial` structs, supporting owning stack instances (`ArrayStorage`) and
borrowed slice views (`StorageView`). It integrates DSP convolution kernels for
system algebra (series, parallel, feedback), Horner evaluation for $O(N+D)$
frequency response sweeps ($H(j\omega)$), Tustin/ZOH discretization, and
canonical controllable/observable realizations into `StateSpace`.

---

### 4. Core Architecture

#### 4.1 Type Signature & Storage Layout

```rust
pub struct TransferFunction<
    T,
    N: Dim,
    D: Dim,
    Sn: DenseStorage<T, R=N, C=Const<1>>,
    Sd: DenseStorage<T, R=D, C=Const<1>>,
> {
    num_storage: Sn,
    den_storage: Sd,
    sample_time: Option<T>, // None = Continuous (s-domain), Some(Ts) = Discrete (z-domain)
}
```

- **Numerator Storage (`Sn`)**: Holds $N$ coefficients for $B(s)$ or $B(z)$.
- **Denominator Storage (`Sd`)**: Holds $D$ coefficients for $A(s)$ or $A(z)$.
- **Sampling Time (`sample_time`)**: `None` specifies a continuous s-domain
  transfer function $H(s)$; `Some(Ts)` specifies a discrete z-domain transfer
  function $H(z)$ with period $T_s$.

#### 4.2 Storage Backends & Zero-Copy Views

```rust
/// Owning transfer function. `N`/`D` are the alias's own const generics.
pub type ArrayTransferFunction<T, const N: usize, const D: usize> =
TransferFunction<T, Const<N>, Const<D>, ArrayStorage<T, N, 1>, ArrayStorage<T, D, 1>>;

/// Zero-copy borrowed read-only view.
pub type TransferFunctionView<'a, T, const N: usize, const D: usize> =
TransferFunction<T, N, D, StorageView<'a, T, N, Const<1>>, StorageView<'a, T, D, Const<1>>>;

/// Zero-copy borrowed mutable view.
pub type TransferFunctionViewMut<'a, T, const N: usize, const D: usize> =
TransferFunction<T, N, D, StorageViewMut<'a, T, N, Const<1>>, StorageViewMut<'a, T, D, Const<1>>>;
```

#### 4.3 Slicing & Memory Access

```rust
impl<T, N: Dim, D: Dim, Sn, Sd> TransferFunction<T, N, D, Sn, Sd>
where
    Sn: ContiguousStorage<T, R=N, C=Const<1>>,
    Sd: ContiguousStorage<T, R=D, C=Const<1>>,
{
    /// Safe contiguous slice view of numerator coefficients.
    pub fn num_slice(&self) -> &[T] {
        self.num_storage.as_slice()
    }

    /// Safe contiguous slice view of denominator coefficients.
    pub fn den_slice(&self) -> &[T] {
        self.den_storage.as_slice()
    }
}
```

#### 4.4 Multiple-Input Multiple-Output (MIMO) Transfer Functions

MIMO systems are represented as a matrix of transfer functions:

```rust
pub type TransferFunctionMatrix<T, const R: usize, const C: usize, const N: usize, const D: usize> =
ArrayMatrix<ArrayTransferFunction<T, N, D>, R, C>;
```

#### 4.5 Error Handling

Following the crate-wide error strategy, fallible constructors return
`Result<T, Error>` via a crate-local `thiserror` enum rather than panicking.
The only runtime-checked invariant at construction is §2.3's Denominator
Validity constraint ($D \ge 1$, non-zero leading coefficient $a_{D-1}$, C-2):

```rust
#[derive(Debug, thiserror::Error)]
pub enum TransferFunctionError {
    #[error("denominator leading coefficient must be non-zero")]
    ZeroLeadingDenominatorCoefficient,
}
```

Runtime constructors that accept caller-supplied coefficients (any
validating variant of `from_coefficients`/`from_storage`) return
`Result<Self, TransferFunctionError>`. No `unwrap()`, `expect()` or `panic!()`
is used outside tests and examples. Near-pole frequency-response evaluation
(§4.7) and partial-fraction ZOH decomposition (§4.9) are *not* treated as
constructor-time errors — a valid `TransferFunction` may still be evaluated at
an ill-conditioned point at call time, which is documented behavior (§4.7)
rather than a distinct error variant, matching how `python-control`'s
`warn_infinity` documents rather than rejects evaluation at a pole.

#### 4.6 Instantiation & Constructors

- **From Storage**:
  `pub const fn from_storage(num_storage: Sn, den_storage: Sd, sample_time: Option<T>) -> Self`
- **Owning Stack Constructor**:
  `pub const fn from_coefficients<const N: usize, const D: usize>(num: [T; N], den: [T; D], sample_time: Option<T>) -> ArrayTransferFunction<T, N, D>`
- **Borrowed views**: `ArrayTransferFunction::view()` / `view_mut()` (FR-6).
  There is no `from_slices(&[T], &[T])` constructor that pairs independent
  `N: Dim`/`D: Dim` with raw slices.

##### 4.6.1 Data-Driven Transfer Function Factories [Proposal (not in evidence)]

To enable dynamic plant identification from time-series and frequency-domain
experimental datasets, dedicated data-driven **Object Factories** estimate
rational transfer function representations without embedding matrix solvers
into `TransferFunction` (Ljung & Chen, 2013; Steiglitz & McBride, 1965; Levy,
1959; Sanathanan & Koerner, 1963; Drmač et al., 2015; Markovsky & Ossareh,
2024; Eckhard, 2026):

- **ARX Time-Domain Estimator (`ArxTransferFunctionEstimator`)**: Produces
  discrete rational transfer function polynomials $G(z) = B(z)/A(z)$ from
  sampled input-output sequences by minimizing equation error in closed form via
  least-squares linear solving (Ljung & Chen, 2013).
- **Levy Frequency Fitter (`LevyTransferFunctionFitter`)**: Fits continuous or
  discrete rational transfer functions $H(j\omega) = B(j\omega)/A(j\omega)$ to
  measured complex frequency response points $(\omega_k, H_k)$ by solving
  linearized real/imaginary least-squares equations (Levy, 1959).
- **Sanathanan–Koerner Iterative Fitter (`SanathananKoernerFitter`)**:
  Iteratively re-weights frequency least-squares equations by
  $|D^{(i-1)}(j\omega_k)|^{-1}$ to converge to optimal complex transfer function
  approximations (Sanathanan & Koerner, 1963; Drmač et al., 2015).

_Detailed standalone design and API signatures for these factories are
specified in `documentation/control-toolboxes/sysid-design.md`._

#### 4.7 Frequency Response Evaluation

Evaluates frequency response $H(s)$ at $s = j\omega$ or $H(z)$
at $z = e^{j\omega T_s}$ using direct Horner evaluation on the numerator and
denominator storage backends:

$$\text{Num}(s) = \text{Horner}(B, s), \quad \text{Den}(s) = \text{Horner}(A, s)$$
$$H(s) = \frac{\text{Num}(s)}{\text{Den}(s)}$$

```rust
impl<T, N: Dim, D: Dim, Sn: DenseStorage<T, R=N, C=Const<1>>, Sd: DenseStorage<T, R=D, C=Const<1>>> TransferFunction<T, N, D, Sn, Sd> {
    pub fn evaluate_complex(&self, s: Complex<T>) -> Complex<T>
    where
        T: Copy + Zero + One + Add<Output=T> + Mul<Output=T>,
    {
        let num_val = horner_eval_storage(&self.num_storage, s);
        let den_val = horner_eval_storage(&self.den_storage, s);
        num_val / den_val
    }
}
```

**Near-Pole Conditioning**: Horner's method is backward-stable — the computed
value is exact for a coefficient set perturbed by a relative error on the order
of machine epsilon (Higham, 2002, Ch. 5). Backward stability does not imply
uniform forward accuracy: the forward error at a given evaluation point $s$ is
goverbed by that point's condition number, which grows as $s$ approaches a
root of the polynomial being evaluated (Higham, 2002). Evaluating
`evaluate_complex` at $s$ close to a pole (a root of the denominator) is
therefore an inherently ill-conditioned operation — division by a
near-zero, noise-dominated `den_val` amplifies the error further. This is
expected behavior common to every reference implementation surveyed
(`python-control`'s `warn_infinity` flag documents rather than suppresses the
same issue at exact poles) and is not treated as a defect requiring
compensation in the initial implementation. Compensated Horner algorithms
(Graillat, Langlois, & Louvet, 2006) exist as a documented, roughly 2x-cost
mitigation if measured near-pole Bode-sweep accuracy proves insufficient in a
future revision.

#### 4.8 System Interconnections

Algebraic interconnections invoke DSP subprograms (such as polynomial
convolution routines) directly on storage elements without creating intermediate
`Polynomial` objects:

##### Series (Cascade) Connection

$$H_1(s) \cdot H_2(s) = \frac{B_1 B_2}{A_1 A_2}$$
Numerator dimension bound: $(N_1 + N_2 - 1)$. Denominator dimension
bound: $(D_1 + D_2 - 1)$.

```rust
impl<T, N1: Dim, D1: Dim, Sn1: DenseStorage<T, R=N1, C=Const<1>>, Sd1: DenseStorage<T, R=D1, C=Const<1>>> TransferFunction<T, N1, D1, Sn1, Sd1> {
    pub fn series<N2: Dim, D2: Dim, Sn2: DenseStorage<T, R=N2, C=Const<1>>, Sd2: DenseStorage<T, R=D2, C=Const<1>>>(
        &self,
        other: &TransferFunction<T, N2, D2, Sn2, Sd2>,
    ) -> TransferFunction<
        T,
        <<N1 as DimAdd<N2>>::Output as DimSub<Const<1>>>::Output,
        <<D1 as DimAdd<D2>>::Output as DimSub<Const<1>>>::Output,
    >
    where
        N1: DimAdd<N2>,
        <N1 as DimAdd<N2>>::Output: DimSub<Const<1>>,
        D1: DimAdd<D2>,
        <D1 as DimAdd<D2>>::Output: DimSub<Const<1>>,
        T: Copy + Zero + Add<Output=T> + Mul<Output=T>,
    {
        // Executes direct DSP convolution on self.num_storage and other.num_storage
        // ...
    }
}
```

##### Parallel Connection

$$H_1(s) + H_2(s) = \frac{B_1 A_2 + B_2 A_1}{A_1 A_2}$$

Numerator dimension bound: $\max(N_1 + D_2 - 1,\ N_2 + D_1 - 1)$ (the two
cross-products summed do not necessarily share a degree). Denominator
dimension bound: $(D_1 + D_2 - 1)$, matching series.

##### Feedback (Closed-Loop) Connection

$$\frac{H_1(s)}{1 + H_1(s) H_2(s)} = \frac{B_1 A_2}{A_1 A_2 + B_1 B_2}$$

Numerator dimension bound: $(N_1 + D_2 - 1)$. Denominator dimension bound:
$\max(D_1 + D_2 - 1,\ N_1 + N_2 - 1)$ — the sum of the two product terms
$A_1 A_2$ and $B_1 B_2$ is bounded by the larger of their two individual
degree bounds, not their sum and is expressed via `DimMax` rather than a
further `DimAdd`:

```rust
impl<T, N1: Dim, D1: Dim, Sn1: DenseStorage<T, R=N1, C=Const<1>>, Sd1: DenseStorage<T, R=D1, C=Const<1>>> TransferFunction<T, N1, D1, Sn1, Sd1> {
    pub fn feedback<N2: Dim, D2: Dim, Sn2: DenseStorage<T, R=N2, C=Const<1>>, Sd2: DenseStorage<T, R=D2, C=Const<1>>>(
        &self,
        other: &TransferFunction<T, N2, D2, Sn2, Sd2>,
    ) -> TransferFunction<
        T,
        <<N1 as DimAdd<D2>>::Output as DimSub<Const<1>>>::Output,
        <<<D1 as DimAdd<D2>>::Output as DimSub<Const<1>>>::Output as DimMax<<<N1 as DimAdd<N2>>::Output as DimSub<Const<1>>>::Output>>::Output,
    >
    where
        N1: DimAdd<D2> + DimAdd<N2>,
        <N1 as DimAdd<D2>>::Output: DimSub<Const<1>>,
        <N1 as DimAdd<N2>>::Output: DimSub<Const<1>>,
        D1: DimAdd<D2>,
        <D1 as DimAdd<D2>>::Output: DimSub<Const<1>>,
        <<D1 as DimAdd<D2>>::Output as DimSub<Const<1>>>::Output: DimMax<<<N1 as DimAdd<N2>>::Output as DimSub<Const<1>>>::Output>,
        T: Copy + Zero + Add<Output=T> + Mul<Output=T>,
    {
        // num = B1 * A2 (DSP convolution); den = A1*A2 + sign * B1*B2
        // ...
    }
}
```

This mirrors the algebra used by reference implementations exactly —
`python-control`'s `feedback()` computes
`num = polymul(num1, den2)` and
`den = polyadd(polymul(den2, den1), -sign * polymul(num2, num1))`
(python-control, `control/xferfcn.py`) — validating the formula above against
the reference source, not just the textbook identity.

##### Non-Minimal Results

Unlike MATLAB or `python-control`, which default to non-minimal returns on
`series`/`parallel`/`feedback` and leave pole-zero cancellation to an explicit
`minreal()` call, the return types above produce exact-dimension arrays
derived from the polynomial product bounds directly. No pole-zero
cancellation is attempted during interconnection arithmetic: exact
cancellation requires root-finding (or polynomial GCD computation), both of
which are numerically ill-conditioned for floating-point coefficients
(Henrici, 1974; Higham, 2002) and must be invoked explicitly through
`minreal()` rather than run silently in arithmetic paths.

#### 4.9 Discretization (Continuous-to-Discrete)

Converts a continuous $H(s)$ to discrete $H(z)$ via Bilinear (Tustin)
transformation with pre-warping or Zero-Order Hold (ZOH).

##### Bilinear (Tustin) Transform

$$s \leftarrow \frac{2}{T_s} \frac{z - 1}{z + 1}$$

With optional pre-warping at critical frequency $\omega_c$:
$$\frac{2}{T_s} \leftarrow \frac{\omega_c}{\tan(\omega_c T_s / 2)}$$

Tustin without pre-warping "introduces a frequency shift that is unacceptable
for many applications" (MathWorks, *Discretize a Compensator*), remedied by
specifying a critical frequency to match exactly under the transform (Ogata,
2010; Franklin et al., 1998).

```rust
impl<T, N: Dim, D: Dim, Sn: DenseStorage<T, R=N, C=Const<1>>, Sd: DenseStorage<T, R=D, C=Const<1>>> TransferFunction<T, N, D, Sn, Sd> {
    pub fn to_discrete_tustin(
        &self,
        sample_time: T,
        prewarp_frequency: Option<T>,
    ) -> ArrayTransferFunction<T, N, D>
    where
        T: Scalar + Div<Output=T>,
        T::Real: Trig,
    {
        // Direct algebraic expansion over numerator and denominator storage
        // ...
    }
}
```

The bound is `T: Scalar + Div` with `T::Real: Trig` rather than `T: Float`:
the pre-warping path needs `tan()`, which `Trig` supplies on the real
projection, and the $\frac{2}{T_s}$ factor needs division, which `Scalar`
deliberately excludes (`num-traits-design.md` FR-2, Alternative 3). Binding
the real projection rather than `Float` keeps the path open to complex
coefficients, since `num-traits-design.md` FR-5 restricts `Float` to
`f32`/`f64`.

##### Zero-Order Hold (ZOH)

ZOH admits two distinct computational paths with materially different
numerical profiles:

- **Transfer-function-direct (chosen)**: $G(z) = (1 - z^{-1})\,
  \mathcal{Z}\left[\mathcal{L}^{-1}\left\{\frac{G(s)}{s}\right\}\right]$,
  via partial-fraction expansion of $G(s)/s$ followed by table-based
  $z$-transform of each term (Franklin et al., 1998). This path never forms a
  state-space realization or a matrix exponential, so it does not inherit the
  companion-form conditioning risk described in §4.10; its own numerical risk is
  partial-fraction decomposition, which is itself ill-conditioned for
  closely-spaced or repeated poles and must be bounded/documented at
  implementation time.
- **State-space-mediated (rejected for this path)**: convert to a canonical
  realization, discretize $A$ via matrix exponential (Moler & Van Loan, 2003),
  convert back. Rejected as the *default* ZOH path specifically because it
  would silently inherit whatever conditioning risk the chosen state-space
  realization carries — material for controllable/observable canonical form
  per §4.10's finding. The state-space-mediated path remains available
  independently through explicit use of §4.10's conversion plus a `StateSpace`
  discretization method, for callers who already need a state-space
  realization for other reasons.

#### 4.10 State-Space Canonical Conversions

Converts a proper continuous or discrete transfer function
$H(s) = \frac{B(s)}{A(s)} = \frac{b_{n} s^n + b_{n-1} s^{n-1} + \dots + b_0}{s^n + a_{n-1} s^{n-1} + \dots + a_0}$ (normalized monic denominator with $a_n = 1$, degree $n = D - 1$, and $b_n = 0$ when strictly proper $N \le n$) directly to a `StateSpace<T, N_X, Const<1>, Const<1>>` system of order $n = D - 1$.

By polynomial division:
$$H(s) = d + \frac{\beta_{n-1} s^{n-1} + \dots + \beta_0}{s^n + a_{n-1} s^{n-1} + \dots + a_0}$$
where direct feedthrough $d = b_n$ (for monic $a_n = 1$; $d = 0$ if strictly proper $N \le n$) and modified output coefficients $\beta_i = b_i - d \cdot a_i = b_i - b_n a_i$ for $i = 0, \dots, n-1$.

In **Controllable Canonical Form**:
$$\mathbf{A} = \begin{bmatrix} 0 & 1 & 0 & \dots & 0 \\ 0 & 0 & 1 & \dots & 0 \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ -a_0 & -a_1 & -a_2 & \dots & -a_{n-1} \end{bmatrix}, \quad \mathbf{B} = \begin{bmatrix} 0 \\ 0 \\ \vdots \\ 1 \end{bmatrix}$$
$$\mathbf{C} = \begin{bmatrix} \beta_0 & \beta_1 & \dots & \beta_{n-1} \end{bmatrix} = \begin{bmatrix} b_0 - b_n a_0 & b_1 - b_n a_1 & \dots & b_{n-1} - b_n a_{n-1} \end{bmatrix}, \quad \mathbf{D} = \begin{bmatrix} d \end{bmatrix} = \begin{bmatrix} b_n \end{bmatrix}$$

In **Observable Canonical Form** (dual realization):
$$\mathbf{A}_o = \mathbf{A}^T = \begin{bmatrix} 0 & 0 & \dots & -a_0 \\ 1 & 0 & \dots & -a_1 \\ 0 & 1 & \dots & -a_2 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \dots & -a_{n-1} \end{bmatrix}, \quad \mathbf{B}_o = \mathbf{C}^T = \begin{bmatrix} b_0 - b_n a_0 \\ b_1 - b_n a_1 \\ \vdots \\ b_{n-1} - b_n a_{n-1} \end{bmatrix}$$
$$\mathbf{C}_o = \mathbf{B}^T = \begin{bmatrix} 0 & 0 & \dots & 1 \end{bmatrix}, \quad \mathbf{D}_o = \begin{bmatrix} b_n \end{bmatrix}$$

**Conditioning Caveat**: Controllable/observable canonical (companion) form is
structurally correct but numerically fragile above low system order. MathWorks
documents that "the transformation to companion form is based on the
controllability matrix, which is almost always numerically singular for
mid-range orders" and marks its own direct companion-form realization command
"Not recommended" in current product documentation (MathWorks, *Canonical
State-Space Realizations*; MathWorks, `canon`). Formal analysis of
companion-form controllability radii confirms this is a structural property of
the form itself, not an implementation artifact (companion-form controllability
radii literature) and recent work characterizes the condition number of the
standard companion-form transformation as growing exponentially with system
dimension, treating numerically reliable computation of it as an open problem
(Yang & Jones, 2026). This design scopes §4.10 to controllable/observable
canonical form for its structural value (explicit characteristic-polynomial
coefficients in $\mathbf{A}$) and low-to-moderate system order; balanced or
modal realization — MathWorks' own recommended numerically-preferred
alternative — is left as future work (see §8 Risks & Open Questions) rather
than implemented in the initial revision.

---

### 5. Alternatives

| Architecture Option                     | Advantages                                                                                                                                                                                                        | Disadvantages                                                                                                                                                                                         | Decision     |
|:----------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------|
| **Wrapping `Polynomial`**               | Reuses existing polynomial methods.                                                                                                                                                                               | Breaks container peer model; adds artificial coupling; forces extra abstraction layers — the same rationale `state-space-design.md` uses to reject wrapping `Matrix` fields directly.                 | **Rejected** |
| **Second-Order-Sections (SOS) Cascade** | Standard embedded-DSP answer to coefficient sensitivity growing with filter order (ARM CMSIS-DSP `BiquadCascadeDF2T`; Rust `biquad` crate); bounds conditioning per-stage rather than across the full polynomial. | Cannot represent an arbitrary rational transfer function, only designed filters reducible to cascaded biquads; would require a structurally different type from this general $N/D$ container.         | **Rejected** |
| **Direct Storage Wrapper (Chosen)**     | Symmetric with `Matrix` and `Polynomial`; zero cost; direct access to `Dim`/DSP/BLAS; supports views and ROM storage.                                                                                             | Requires implementing evaluation and convolution calls against storage directly; flat coefficient representation inherits the coefficient-sensitivity growth SOS is designed to avoid, at high order. | **Selected** |

---

### 6. Verification & Validation

#### 6.1. Objectives

- Demonstrate compile-time verification of numerator and denominator polynomial
  capacities.
- Demonstrate numerical accuracy of frequency response evaluation ($H(j\omega)$
  and Bode magnitude/phase).
- Demonstrate algebraic exactness of series, parallel, and feedback transfer
  function connections.
- Demonstrate numerical correctness of Tustin (bilinear with pre-warping) and
  direct ZOH discretization.
- Demonstrate exact state-space matrix conversions for Controllable and
  Observable Canonical Forms.
- Demonstrate zero dynamic heap allocation in `#![no_std]` execution and
  deterministic real-time performance.

#### 6.2. Methods

| Method                    | Mechanism                                                                                     | Requirements discharged  |
|:--------------------------|:----------------------------------------------------------------------------------------------|:-------------------------|
| Compile-time shape check  | Type-level `Dim` sizing and `compile_fail` doctests                                           | FR-1, C-1, C-3, C-4      |
| Requirements-based test   | `#[test]` unit tests over physical filter benchmarks and singular cases                       | FR-2, FR-3, FR-4, FR-5   |
| Property-based test       | `proptest` suites verifying transfer function commutativity and feedback identities           | FR-3, FR-4               |
| Doctest                   | Runnable rustdoc examples                                                                     | FR-2, FR-6               |
| Back-to-back comparison   | `/cr-prototype numerical-models/transfer-function` and MATLAB `tf` / `python-control` oracles | FR-2, FR-4, FR-5         |
| Resource usage evaluation | `no_alloc` audit, `size_of` assertions, stack analysis                                        | NFR-1, NFR-2, C-2, C-4   |
| On-target execution       | ETS suites under QEMU and Teensy hardware                                                     | NFR-3                    |
| Coverage measurement      | `cargo coverage` reporting statement and branch metrics                                       | FR-1..FR-6, NFR-1..NFR-3 |

#### 6.3. Acceptance Criteria

| Claim                                         | Oracle                                                     | Measure        | Bound                                                                                                                                     | Justification                                                     |
|:----------------------------------------------|:-----------------------------------------------------------|:---------------|:------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------|
| Frequency response $H(j\omega)$ residual      | Analytic rational function                                 | Relative error | $\frac{\|\hat{H}(j\omega) - H_{\text{analytic}}(j\omega)\|}{\|H_{\text{analytic}}(j\omega)\|} \le \gamma_{2(D-1)} \kappa(H(j\omega))$     | Rational Horner evaluation backward stability (Higham, 2002)      |
| Series multiplication                         | Discrete polynomial convolution                            | Absolute error | $\|(N_1 N_2)_k - \sum a_i b_{k-i}\|_\infty \le (N_1+N_2)\epsilon$                                                                         | Discrete convolution arithmetic bound (Oppenheim & Schafer, 2009) |
| Tustin discretization frequency mapping       | $\omega_d = \frac{2}{T_s} \arctan(\frac{\omega_a T_s}{2})$ | Relative error | $\le 5\epsilon$                                                                                                                           | Bilinear mapping identity (Franklin et al., 1998)                 |
| Canonical state-space eigenvalue equivalence  | Roots of denominator polynomial $D(s)$                     | Absolute error | $\|\lambda_i(A_c) - p_i\| \le \mathcal{O}(\epsilon \kappa(D))$                                                                            | Companion matrix spectral equivalence (Kenney & Laub, 1988)       |
| Zero leading denominator validation           | Denominator with zero leading coefficient                  | Exact equality | `Err(TransferFunctionError::ZeroLeadingDenominator)`                                                                                      | Precondition failure contract                                     |
| Strictly improper transfer function rejection | System with $N > D$ in strictly proper contexts            | Exact equality | `Err(TransferFunctionError::ImproperSystem)`                                                                                              | Properness contract                                               |
| Zero-allocation execution                     | Host allocator interception                                | Exact equality | 0 heap allocations                                                                                                                        | NFR-1 `#![no_std]` invariant                                      |

#### 6.4. Traceability

| Requirement                                    | Method                                           | Artifact                                               |
|:-----------------------------------------------|:-------------------------------------------------|:-------------------------------------------------------|
| FR-1 — Rational SISO Transfer Function         | Compile-time shape check                         | `tests/tf_shape_fail.rs` (`compile_fail` doctests)     |
| FR-2 — Frequency Response Evaluation           | Requirements-based test, Back-to-back comparison | `tests/tf_frequency.rs::test_bode_evaluation`          |
| FR-3 — Rational System Algebra                 | Property-based test, Back-to-back comparison     | `tests/tf_algebra.rs::prop_tf_series_parallel`         |
| FR-4 — System Discretization                   | Requirements-based test, Back-to-back comparison | `tests/tf_discretize.rs::test_tustin_prewarped`        |
| FR-5 — State-Space Canonical Realization       | Requirements-based test                          | `tests/tf_state_space.rs::test_canonical_forms`        |
| NFR-1 — Deterministic Fixed-Memory Execution   | Resource usage evaluation                        | `#![no_std]` host allocator audit                      |
| NFR-2 — Real-Time Frequency Sweep Throughput   | Resource usage evaluation                        | `clippy::large_stack_arrays` CI check                  |
| C-1 — Properness Precondition                  | Compile-time shape check                         | Static properness shape assertions                     |
| C-2 — Non-Zero Leading Denominator             | Requirements-based test                          | Zero leading coefficient error assertion               |
| C-3 — Capacity Bound                           | Compile-time shape check                         | Static size bounds checks                              |
| C-4 — `#![no_std]` Environment                 | Resource usage evaluation                        | Compilation under `#![no_std]` target triples          |

#### 6.5. Coverage

- **Target**: $\ge 90\%$ statement coverage, $\ge 85\%$ branch coverage reported
  via `cargo coverage`.
- **Excluded**: Target-specific hardware benchmarking loops and debug display
  formatting (`core::fmt::Debug`).

#### 6.6. Validation

- **Second-Order Low-Pass Filter**: Frequency sweep and Bode plot verification
  for an analog Butterworth low-pass filter in `examples/butterworth_filter.rs`.
- **Digital Notch Filter Execution**: Discretization and step-response
  simulation of a 60 Hz notch filter for biomedical sensor filtering in
  `examples/notch_filter.rs`.

#### 6.7. Not Verified

- Minimal realization reduction (`minreal`) with automatic pole-zero
  cancellation is deferred to future work.
- Frequency evaluation at exact poles where $D(j\omega) = 0$ is mathematically
  undefined and not verified for numerical convergence.

---

### 7. Performance & Resource Considerations

- **Stack Overhead**: Owning transfer functions `ArrayTransferFunction<T, N, D>`
  occupy $(N + D) \times \text{size\_of}(T)$ bytes.
- **Borrowed Views**: `TransferFunctionView<'a, T, N, D>` occupies only 2
  pointer references plus stride/length parameters, incurring zero allocation.
- **Inline Operations**: Storage element accessors monomorphize into direct
  memory loads.

---

### 8. Risks & Open Questions

- **Non-Minimal Realization Handling**: Series/parallel/feedback never cancel
  poles/zeros (§4.8). A `minreal`-equivalent capacity-reducing operation is not
  yet scoped; whether and how to offer one is deferred to a future revision.
- **Partial-Fraction Conditioning for ZOH**: The chosen transfer-function-direct
  ZOH path (§4.9) requires partial-fraction decomposition, which is itself
  ill-conditioned for closely-spaced or repeated poles. The implementation phase
  must bound or document this explicitly rather than assume it away.
- **Canonical Form Scope**: Controllable/observable canonical form (§4.10) is
  numerically fragile above low system order (Kenney & Laub, 1988; Yang & Jones,
  2026). Balanced or modal realization is identified as the
  numerically-preferred alternative used by reference implementations but is not
  implemented in this revision.
- **Compensated Horner Evaluation**: Near-pole frequency-response evaluation (
  §4.7) is documented as inherently ill-conditioned rather than compensated. If
  measured Bode-sweep accuracy proves insufficient once implemented, compensated
  Horner evaluation (Graillat, Langlois, & Louvet, 2006) is the identified
  mitigation path.
- **Analytic Scalar Bounds**: `to_discrete_tustin` (§4.9) binds
  `T: Scalar + Div` with `T::Real: Trig`, following `num-traits-design.md` §4.1.
  Separately, `Convolution<T>` (`src/math/dsp.rs`) is currently declared over
  `T: Float`, which accepts a narrower scalar set than the ring arithmetic
  paths. Widening it to `T: Scalar` is tracked in `polynomial-design.md` §7.

---

### 9. Development Plan

| Task / Feature                              | Description                                                                                                                                                                                                       | Estimated Effort |
|:--------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
| **Phase 1: Storage Wrapper & Constructors** | Base `TransferFunction` struct, storage traits, slice accessors and error type.                                                                                                                                   | 1.0 Day          |
| **Phase 2: Frequency Evaluation**           | Direct Horner evaluation over storage for $H(j\omega)$ and Bode calculations.                                                                                                                                     | 1.0 Day          |
| **Phase 3: Algebra & DSP Convolution**      | Implement series, parallel and feedback connections using direct DSP convolution.                                                                                                                                 | 1.5 Days         |
| **Phase 4: Discretization**                 | Bilinear (Tustin, with pre-warping) transform and transfer-function-direct ZOH, including partial-fraction decomposition (§8's closely-spaced/repeated-pole conditioning risk must be bounded, not assumed away). | 2.5 Days         |
| **Phase 5: State-Space Conversion**         | Controllable and Observable Canonical Form conversions.                                                                                                                                                           | 1.5 Days         |
| **Phase 6: Verification Suite**             | Unit tests, `proptest` suites and cross-validation against two external reference implementations (MATLAB, `python-control`) per `vv-standards.md`.                                                               | 2.0 Days         |

---

### 10. References

1. **Franklin, G. F., Powell, J. D., & Workman, M. L. (1998).** *Digital Control
   of Dynamic Systems* (3rd ed.). Addison-Wesley. — Implementation-oriented
   treatment of discretization and digital filter realization on embedded
   microcontrollers; direct transfer-function-domain ZOH formula.
2. **Moler, C., & Van Loan, C. (2003).** Nineteen Dubious Ways to Compute the
   Exponential of a Matrix, Twenty-Five Years Later. *SIAM Review*, 45(1),
   3–49. — Rationale and conditioning concerns for the state-space-mediated ZOH
   path (§4.9).
3. **python-control developers. (2026).** `control/xferfcn.py`, python-control
   library source. [Online].
   Available: https://github.com/python-control/python-control/blob/main/control/xferfcn.py.
4. **MathWorks. (2026).** *Discretize a Compensator*, MATLAB & Simulink
   documentation. [Online].
   Available: https://www.mathworks.com/help/control/ug/discretize-a-compensator.html.
5. **MathWorks. (2026).** *Canonical State-Space Realizations* and `canon`,
   MATLAB documentation. [Online].
   Available: https://www.mathworks.com/help/control/ug/canonical-state-space-realizations.html.
6. **ARM Ltd. (2025).** *Biquad Cascade IIR Filters Using a Direct Form II
   Transposed Structure*, CMSIS-DSP documentation. [Online].
   Available: https://arm-software.github.io/CMSIS-DSP/main/group__BiquadCascadeDF2T.html.
7. **Oppenheim, A. V., & Schafer, R. W. (2009).** *Discrete-Time Signal
   Processing* (3rd ed.). Pearson. — FIR/IIR filter representation, transfer
   function stability and frequency-response $H(e^{j\omega})$ theory.
8. **Ogata, K. (2010).** *Modern Control Engineering* (5th ed.). Prentice
   Hall. — Continuous transfer functions, bilinear (Tustin) transformation and
   frequency pre-warping derivations.
9. **Henrici, P. (1974).** *Applied and Computational Complex Analysis, Volume
   1*. Wiley. — Complex-arithmetic numerical stability foundations
   for $H(j\omega)$ evaluation.
10. **Higham, N. J. (2002).** *Accuracy and Stability of Numerical Algorithms* (
    2nd ed.). SIAM, Ch. 5. — Horner's method backward-error bound and
    conditioning near polynomial roots (§4.7).
11. **Graillat, S., Langlois, P., & Louvet, N. (2006).** Faithful Polynomial
    Evaluation with Compensated Horner Algorithm. *Proceedings of the 17th IEEE
    Symposium on Computer Arithmetic*. — Compensated Horner as a documented
    mitigation for near-root evaluation error (§4.7).
12. **Kenney, C. S., & Laub, A. J. (1988).** Controllability and stability radii
    for companion form systems. *Mathematics of Control, Signals and Systems*,
    1, 239–256. — Formal treatment of near-uncontrollability and
    near-singularity in high-order companion-form realizations (§4.10).
13. **Yang, S., & Jones, C. N. (2026).** Numerically Reliable Brunovsky
    Transformations. — Exponential condition-number growth of the standard
    companion-form transformation with system dimension (§4.10).
14. **Claessen, K., & Hughes, J. (2000).** QuickCheck: A Lightweight Tool for
    Random Testing of Haskell Programs. *ACM SIGPLAN Notices*, 35(9), 268–279. —
    Property-based testing principles for algebraic invariants.
15. **Rust Project Developers. (2024).** *The Rustonomicon: The Dark Arts of
    Advanced and Unsafe Rust Programming*. — Memory-aliasing rules for
    pointer-backed views.
16. **ISO. (2018).** *ISO 26262-6:2018 Road vehicles — Functional safety — Part
    6: Product development at the software level*.
17. **RTCA / EUROCAE. (2011).** *DO-178C: Software Considerations in Airborne
    Systems and Equipment Certification*.
18. **Ljung, L., & Chen, T. (2013).** System identification - a frequency domain
    approach, or is it a time domain approach? In *2013 9th Asian Control
    Conference (ASCC)*, Istanbul, Turkey. — Time-domain ARX, OE, and PEM
    rational
    transfer function parameterizations.
19. **Steiglitz, K., & McBride, L. E. (1965).** A technique for the
    identification of linear systems. *IEEE Transactions on Automatic Control*,
    10(4), 461–464, doi: 10.1109/TAC.1965.1106097. — Output-error polynomial
    ratio iterative minimization.
20. **Levy, E. C. (1959).** Complex-curve fitting. *IRE Transactions on
    Automatic
    Control*, AC-4(1), 37–43, doi: 10.1109/TAC.1959.1104841. — Linear
    least-squares frequency-domain fitting of rational transfer functions.
21. **Sanathanan, C. K., & Koerner, J. (1963).** Transfer function synthesis as
    a ratio of two complex polynomials. *IEEE Transactions on Automatic
    Control*, 8(1), 56–58, doi: 10.1109/TAC.1963.1105517. — Iterative weighted
    least-squares frequency response fitting.
22. **Drmač, Z., Gugercin, S., & Beattie, C. (2015).** Quadrature-Based Vector
    Fitting for Discretized $\mathcal{H}_2$ Approximation. *SIAM Journal on
    Scientific Computing*, 37(2), A625–A652, doi: 10.1137/140961511. — Vector
    Fitting rational frequency-domain approximation.
23. **Markovsky, I., & Ossareh, H. R. (2024).** Direct data-driven frequency
    response estimation and its application to transfer function fitting.
    *Automatica*, 159, 111351. — Nonparametric ETFE and direct data-driven
    transfer function fitting.
24. **Eckhard, D. (2026).** System identification in Python: The `pysib`
    package.
    *arXiv:2606.26376*. — Polynomial time-domain I/O model structures (ARX,
    ARMAX, OE, Box-Jenkins).

---

### 11. Revision History

| Revision | Date            | Author          | Description                                                                                                                           |
|:---------|:----------------|:----------------|:--------------------------------------------------------------------------------------------------------------------------------------|
| 1.0      | July 26, 2026   | @MitchellDScott | Initial draft: SISO transfer functions with separate numerator and denominator polynomials.                                            |
| 1.1      | August 16, 2026 | @MitchellDScott | Storage parameterization: decoupled polynomial storage into `DenseStorage` traits and added zero-copy views.                           |
| 1.2      | August 25, 2026 | @MitchellDScott | System operations & algebra: added frequency response evaluation, series/parallel/feedback algebra, and bilinear/Tustin discretization. |
| 1.3      | August 25, 2026 | @MitchellDScott | V&V standardization: aligned test oracles with frequency sweep tolerances and algebraic invariants.                                  |
| 1.4      | August 26, 2026 | @MitchellDScott | Storage view retarget: updated references to `StorageView`/`StorageViewMut` and `Const<1>` dimensions.                                |
