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
- **Pole & Zero Extraction**: Computing exact complex poles and zeros across
  arbitrary system dimensions by invoking `Polynomial::roots()`.

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
- **FR-6 — Generic Pole and Zero Extraction**: Computes the complex poles
  ($p \in \mathbb{C}^D$) and zeros ($z \in \mathbb{C}^N$) returning fixed-size worst-case buffers
  `[Complex<T>; D]` and `[Complex<T>; N]` by delegating directly
  to `Polynomial::roots()` on the underlying denominator and numerator polynomial models.

#### 2.2. Non-Functional Requirements

- **NFR-1 — Deterministic Fixed-Memory Execution**: Frequency evaluation and
  rational arithmetic execute entirely within stack-allocated buffers with zero
  heap allocations.
- **NFR-2 — Real-Time Frequency Sweep Throughput**: Frequency response
  calculations maintain linear computational scaling across frequency sweeps
  without heap fragmentation.

#### 2.3. Constraints

- **C-1 — Properness Precondition**: Denominator coefficient capacity $D$ and
  numerator capacity $N$ must satisfy $D \ge 1$ and $D \ge N$ (denominator
  degree $n = D - 1 \ge m = N - 1$, proper transfer function).
- **C-2 — Non-Zero Leading Denominator**: Leading denominator
  coefficient $a_{D-1}$ must be non-zero ($a_{D-1} \neq 0$).
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
- **Borrowed views**: `ArrayTransferFunction::view()` / `view_mut()`.
  There is no `from_slices(&[T], &[T])` constructor that pairs independent
  `N: Dim`/`D: Dim` with raw slices.

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

**Near-Pole Conditioning**: Horner's method is backward-stable. Evaluation
near a pole is ill-conditioned (Higham, 2002, Ch. 5) and is not compensated.

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
2010; Franklin et al., 1998). Clearing $(z+1)^{D-1}$ on both polynomials
fills relative degree $r > 0$ with $(z+1)^r$, so the discrete result is
biproper with capacities `(D, D)`, matching ZOH.

```rust
impl<T, N: Dim, D: Dim, Sn: DenseStorage<T, R=N, C=Const<1>>, Sd: DenseStorage<T, R=D, C=Const<1>>> TransferFunction<T, N, D, Sn, Sd> {
    pub fn to_discrete_tustin(
        &self,
        sample_time: T,
        prewarp_frequency: Option<T>,
    ) -> ArrayTransferFunction<T, D, D>
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

- **State-space-mediated**: convert to
  controllable canonical form (§4.10), discretize via
  `StateSpace::to_discrete_zoh` (Van Loan augmented matrix exponential),
  convert back with the SISO formula $H(z) = C(zI-A)^{-1}B+D$. This path
  inherits the companion-form conditioning caveat of §4.10 and is the public
  `to_discrete_zoh` implementation because the crate has no general
  eigensolver for transfer-function-direct partial fractions.

```rust
impl<T: Float + Copy, const N: usize, const D: usize> ArrayTransferFunction<T, N, D> {
    pub fn to_discrete_zoh(&self, sample_time: T) -> LinAlgResult<Self> { /* ... */ }
}
```

- **Transfer-function-direct (deferred)**: $G(z) = (1 - z^{-1})\,
  \mathcal{Z}\left[\mathcal{L}^{-1}\left\{\frac{G(s)}{s}\right\}\right]$,
  via partial-fraction expansion of $G(s)/s$ followed by table-based
  $z$-transform of each term (Franklin et al., 1998). Deferred to §6.7 / §8
  until a pole solver exists. The state-space-mediated path remains available
  independently through explicit use of §4.10 plus a `StateSpace`
  discretization method.

#### 4.10 State-Space Canonical Conversions

Converts a proper continuous or discrete transfer function
$H(s) = \frac{B(s)}{A(s)} = \frac{b_{n} s^n + b_{n-1} s^{n-1} + \dots + b_0}{s^n + a_{n-1} s^{n-1} + \dots + a_0}$ (
normalized monic denominator with $a_n = 1$, degree $n = D - 1$, and $b_n = 0$
when strictly proper $N \le n$) directly to a
`StateSpace<T, N_X, Const<1>, Const<1>>` system of order $n = D - 1$.

By polynomial division:
$$H(s) = d + \frac{\beta_{n-1} s^{n-1} + \dots + \beta_0}{s^n + a_{n-1} s^{n-1} + \dots + a_0}$$
where direct feedthrough $d = b_n$ (for monic $a_n = 1$; $d = 0$ if strictly
proper $N \le n$) and modified output
coefficients $\beta_i = b_i - d \cdot a_i = b_i - b_n a_i$
for $i = 0, \dots, n-1$.

In **Controllable Canonical Form**:
$$\mathbf{A} = \begin{bmatrix} 0 & 1 & 0 & \dots & 0 \\ 0 & 0 & 1 & \dots & 0 \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ -a_0 & -a_1 & -a_2 & \dots & -a_{n-1} \end{bmatrix}, \quad \mathbf{B} = \begin{bmatrix} 0 \\ 0 \\ \vdots \\ 1 \end{bmatrix}$$
$$\mathbf{C} = \begin{bmatrix} \beta_0 & \beta_1 & \dots & \beta_{n-1} \end{bmatrix} = \begin{bmatrix} b_0 - b_n a_0 & b_1 - b_n a_1 & \dots & b_{n-1} - b_n a_{n-1} \end{bmatrix}, \quad \mathbf{D} = \begin{bmatrix} d \end{bmatrix} = \begin{bmatrix} b_n \end{bmatrix}$$

In **Observable Canonical Form** (dual realization):
$$\mathbf{A}_o = \mathbf{A}^T = \begin{bmatrix} 0 & 0 & \dots & -a_0 \\ 1 & 0 & \dots & -a_1 \\ 0 & 1 & \dots & -a_2 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \dots & -a_{n-1} \end{bmatrix}, \quad \mathbf{B}_o = \mathbf{C}^T = \begin{bmatrix} b_0 - b_n a_0 \\ b_1 - b_n a_1 \\ \vdots \\ b_{n-1} - b_n a_{n-1} \end{bmatrix}$$
$$\mathbf{C}_o = \mathbf{B}^T = \begin{bmatrix} 0 & 0 & \dots & 1 \end{bmatrix}, \quad \mathbf{D}_o = \begin{bmatrix} b_n \end{bmatrix}$$

**Conditioning Caveat**: Controllable/observable canonical form is
numerically fragile above low system order (MathWorks, `canon`; Yang &
Jones, 2026). This revision uses it for its structural value (characteristic
polynomial coefficients explicit in $\mathbf{A}$); balanced or modal
realization is future work (§8).



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

#### 4.11 Pole and Zero Extraction

Computes exact complex poles and zeros by delegating directly to
[`Polynomial::roots`](../numerical-models/polynomial-design.md) on the
denominator and numerator storage memory:

```rust
impl<
    T: Float + Copy,
    const N: usize,
    const D: usize,
    Sn: DenseStorage<T, R = Const<N>, C = Const<1>>,
    Sd: DenseStorage<T, R = Const<D>, C = Const<1>>,
> TransferFunction<T, Const<N>, Const<D>, Sn, Sd>
where
    Const<N>: Dim,
    Const<D>: Dim,
{
    /// Computes the complex poles of the transfer function for denominator capacity D.
    pub fn poles(&self) -> Result<[Complex<T>; D], RootError> {
        let den_storage: StorageView<'_, T, Const<D>, Const<1>> = unsafe {
            StorageView::new_with_strides_unchecked(
                self.den_storage.as_ptr(),
                self.den_storage.r_stride(),
                self.den_storage.c_stride(),
            )
        };
        let den_poly: Polynomial<T, Const<D>, StorageView<'_, T, Const<D>, Const<1>>> =
            Polynomial::from_storage(den_storage);
        den_poly.roots()
    }

    /// Computes the complex zeros of the transfer function for numerator capacity N.
    pub fn zeros(&self) -> Result<[Complex<T>; N], RootError> {
        let num_storage: StorageView<'_, T, Const<N>, Const<1>> = unsafe {
            StorageView::new_with_strides_unchecked(
                self.num_storage.as_ptr(),
                self.num_storage.r_stride(),
                self.num_storage.c_stride(),
            )
        };
        let num_poly: Polynomial<T, Const<N>, StorageView<'_, T, Const<N>, Const<1>>> =
            Polynomial::from_storage(num_storage);
        num_poly.roots()
    }
}
```

---

### 5. Implementation Alternatives

#### 5.1 Evaluated Alternatives

- **Transfer-Function-Direct Partial Fraction Expansion vs State-Space Mediation for ZOH**:
  Direct partial-fraction ZOH requires finding exact complex poles and calculating residue coefficients. Delegating pole finding directly to `Polynomial::roots()` enables closed-form $\mathcal{O}(1)$ pole extraction for second-order systems ($D=3$) while preserving companion-form state-space conversion for higher degrees ($D > 3$).

---

### 6. Verification and Validation

#### 6.1. Principles

The verification approach aligns with [`vv-standards.md`](../vv-standards.md).
Validation compares against NumPy / SciPy / harold reference models.

#### 6.2. Methods

| Method                    | Mechanism                                                                                                                                                    | Requirements discharged  |
|:--------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------|
| Compile-time shape check  | Type-level `Dim` sizing and `compile_fail` doctests                                                                                                          | FR-1, C-1, C-3, C-4      |
| Requirements-based test   | `#[test]` unit tests over physical filter benchmarks and singular cases                                                                                      | FR-2, FR-3, FR-4, FR-5, FR-6 |
| Property-based test       | `proptest` suites verifying transfer function commutativity and feedback identities                                                                          | FR-3, FR-4               |
| Doctest                   | Runnable rustdoc examples                                                                                                                                    | FR-2                     |
| Back-to-back comparison   | `examples/numerical-models-validation/python3/transfer_function_validation.py` vs `src/transfer_function_validation.rs` JSON; [`numerical-models-design.md`](numerical-models-design.md) §5.1 | FR-2, FR-4, FR-5, FR-6   |
| Resource usage evaluation | `no_alloc` audit, `size_of` assertions, stack analysis                                                                                                       | NFR-1, NFR-2, C-2, C-4   |
| On-target execution       | ETS suites under QEMU and Teensy hardware                                                                                                                    | NFR-1                    |
| Coverage measurement      | `cargo coverage` reporting statement and branch metrics                                                                                                      | FR-1..FR-6, NFR-1..NFR-2 |

#### 6.3. Acceptance Criteria

| Claim                                         | Oracle                                                     | Measure        | Bound                                                                                                                                 | Justification                                                     |
|:----------------------------------------------|:-----------------------------------------------------------|:---------------|:--------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------|
| Frequency response $H(j\omega)$ residual      | Analytic rational function                                 | Relative error | $\frac{\|\hat{H}(j\omega) - H_{\text{analytic}}(j\omega)\|}{\|H_{\text{analytic}}(j\omega)\|} \le \gamma_{2(D-1)} \kappa(H(j\omega))$ | Rational Horner evaluation backward stability (Higham, 2002)      |
| Continuous & discretized response             | harold `Transfer` / `frequency_response`                   | Mag / Phase    | Magnitude $\le 10^{-3}$ dB, phase $\le 10^{-2}$ deg, Nyquist locus $\le 10^{-3}$                                                      | Cross-toolbox frequency response (Misra-Patel) & Tustin/ZOH agreement |
| Series multiplication                         | Discrete polynomial convolution                            | Absolute error | $\|(N_1 N_2)_k - \sum a_i b_{k-i}\|_\infty \le (N_1+N_2)\epsilon$                                                                     | Discrete convolution arithmetic bound (Oppenheim & Schafer, 2009) |
| Tustin discretization frequency mapping       | $\omega_d = \frac{2}{T_s} \arctan(\frac{\omega_a T_s}{2})$ | Relative error | $\le 5\epsilon$                                                                                                                       | Bilinear mapping identity (Franklin et al., 1998)                 |
| Canonical state-space eigenvalue equivalence  | Roots of denominator polynomial $D(s)$                     | Absolute error | $\|\lambda_i(A_c) - p_i\| \le \mathcal{O}(\epsilon \kappa(D))$                                                                        | Companion matrix spectral equivalence (Kenney & Laub, 1988)       |
| 2nd-order transfer function poles and zeros   | Analytic quadratic roots via `Polynomial::roots`           | Absolute error | $\|p_i - \hat{p}_i\|_\infty \le \epsilon \omega_n$                                                                                    | Muller/Higham stabilized quadratic formulation                    |
| Higher-order poles and zeros                  | Companion-form Durand-Kerner roots via `Polynomial::roots` | Absolute error | $\|p_i - \hat{p}_i\|_\infty \le 10^{-10}$                                                                                             | Multi-tier generic polynomial root solver                         |
| Zero leading denominator validation           | Denominator with zero leading coefficient                  | Exact equality | `Err(TransferFunctionError::ZeroLeadingDenominator)`                                                                                  | Precondition failure contract                                     |
| Strictly improper transfer function rejection | System with $N > D$ in strictly proper contexts            | Exact equality | `Err(TransferFunctionError::ImproperSystem)`                                                                                          | Properness contract                                               |
| Zero-allocation execution                     | Host allocator interception                                | Exact equality | 0 heap allocations                                                                                                                    | NFR-1 `#![no_std]` invariant                                      |

#### 6.4. Traceability

| Requirement                                           | Method                                           | Artifact                                                                                                                                   |
|:------------------------------------------------------|:-------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------|
| FR-1 — Rational SISO Transfer Function Representation | Compile-time shape check                         | rustdoc `compile_fail` doctests in `src/transfer_function/mod.rs`                                                                          |
| FR-2 — Frequency Response Evaluation                  | Requirements-based test, Back-to-back comparison | `src/transfer_function/tests/transfer_function_tests.rs::test_frequency_response_continuous`                                               |
| FR-3 — Rational System Algebra                        | Property-based test, Back-to-back comparison     | `src/transfer_function/tests/transfer_function_tests.rs::test_transfer_function_series`                                                    |
| FR-4 — System Discretization                          | Requirements-based test, Back-to-back comparison | `src/transfer_function/tests/transfer_function_tests.rs::test_tustin_prewarped`                                                            |
| FR-5 — State-Space Canonical Realization              | Requirements-based test                          | `src/transfer_function/tests/transfer_function_tests.rs::test_controllable_canonical_form`, `test_ccf_eigenvalues_match_denominator_roots` |
| FR-6 — Generic Pole and Zero Extraction               | Requirements-based test, Back-to-back comparison | `src/transfer_function/tests/transfer_function_tests.rs::test_transfer_function_poles_and_zeros`                                            |

| NFR-1 — Deterministic Fixed-Memory Execution | Resource usage evaluation |
`#![no_std]` host allocator audit |
| NFR-2 — Real-Time Frequency Sweep Throughput | Resource usage evaluation |
`clippy::large_stack_arrays` CI check |
| C-1 — Properness Precondition | Compile-time shape check | Static properness
shape assertions |
| C-2 — Non-Zero Leading Denominator | Requirements-based test | Zero leading
coefficient error assertion |
| C-3 — Capacity Bound | Compile-time shape check | Static size bounds checks |
| C-4 — `#![no_std]` Environment | Resource usage evaluation | Compilation under
`#![no_std]` target triples |

#### 6.5. Coverage

- **Target**: $\ge 90\%$ statement coverage, $\ge 85\%$ branch coverage reported
  via `cargo coverage`.
- **Excluded**: Target-specific hardware benchmarking loops and debug display
  formatting (`core::fmt::Debug`).

#### 6.6. Validation

- **Frequency Response, Bode Analysis, & Realization**: Verification of given
  2nd-order transfer function rational frequency evaluation $H(j\omega)$ on
  $\mathrm{logspace}(-2,3,128)$, Bode magnitude/phase, series cascade
  ($H_1 \cdot H_2$), controllable canonical realization, clustered-pole
  $H(s)=1/[(s+1)^4(s+1.01)^4]$, and multi-source cross-validation against
  SciPy and harold oracles in `examples/numerical-models-validation/src/transfer_function_validation.rs`.

#### 6.7. Not Verified

- Minimal realization reduction (`minreal`) with automatic pole-zero
  cancellation is deferred to future work.
- Frequency evaluation at exact poles where $D(j\omega) = 0$ is mathematically
  undefined and not verified for numerical convergence.
- Transfer-function-direct partial-fraction ZOH is not implemented; public
  `to_discrete_zoh` uses controllable canonical form plus Van Loan ZOH (§4.9).
- Controllable-canonical realization at denominator degree $> 32$ is not
  verified against `state-space-design.md` C-2 ($N_x \le 32$). The example
  crate sweeps clustered-pole $H(s)=1/[(s+1)^4(s+1.01)^4]$ on 128 frequencies
  ([`numerical-models-design.md`](numerical-models-design.md) §6.6).

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
  yet scoped; whether and how to offer one is deferred.
- **Partial-Fraction Conditioning for ZOH**: Transfer-function-direct ZOH via
  partial fractions is deferred until a pole solver exists (§4.9, §6.7). Public
  `to_discrete_zoh` uses the state-space-mediated path and inherits §4.10
  companion-form conditioning.
- **Canonical Form Scope**: Controllable/observable canonical form (§4.10) is
  numerically fragile above low system order (Kenney & Laub, 1988; Yang & Jones,
  2026). Balanced or modal realization is identified as the
  numerically-preferred alternative used by reference implementations but is not
  implemented.
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
| **Phase 6: Verification Suite**             | Unit tests, `proptest` suites and cross-validation against two external reference implementations (MATLAB, `python-control`) per [`vv-standards.md`](../vv-standards.md).                                         | 2.0 Days         |

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

---

### 11. Revision History

| Revision | Date            | Author          | Description                                                                                                                             |
|:---------|:----------------|:----------------|:----------------------------------------------------------------------------------------------------------------------------------------|
| 1.0      | July 26, 2026   | @MitchellDScott | Initial draft: SISO transfer functions with separate numerator and denominator polynomials.                                             |
| 1.1      | August 16, 2026 | @MitchellDScott | Storage parameterization: decoupled polynomial storage into `DenseStorage` traits and added zero-copy views.                            |
| 1.2      | August 25, 2026 | @MitchellDScott | System operations & algebra: added frequency response evaluation, series/parallel/feedback algebra, and bilinear/Tustin discretization. |
| 1.3      | August 25, 2026 | @MitchellDScott | V&V standardization: aligned test oracles with frequency sweep tolerances and algebraic invariants.                                     |
| 1.4      | August 26, 2026 | @MitchellDScott | Storage view retarget: updated references to `StorageView`/`StorageViewMut` and `Const<1>` dimensions.                                  |
| 1.5      | August 26, 2026 | @MitchellDScott | Trimmed near-pole and companion-form caveats; crate-wide standards cite `vv-standards.md`.                                              |
| 1.6      | August 28, 2026 | @MitchellDScott | Host-scale V&V: clustered-pole $H(j\omega)$ ($N>50$); realization at degree $>32$ stays in §6.7. Caps unchanged.                        |
| 1.8      | August 28, 2026 | @MitchellDScott | Tustin returns biproper `(D, D)` after clearing $(z+1)^{D-1}$ (matches ZOH).                                                            |
| 1.9      | August 28, 2026 | @MitchellDScott | §6.4 FR-4 `test_tustin_prewarped` and FR-5 CCF eigenvalue match live in `transfer_function_test_suite`.                                 |
| 2.0      | August 30, 2026 | @MitchellDScott | Reverted Butterworth constructor from `transfer_function` module; deferred filter synthesis to future `filters/` crate module.          |
| 2.1      | August 31, 2026 | @MitchellDScott | Added harold multi-source frequency response / discretization cross-validation oracle and updated validation crate paths.                |
| 2.2      | September 1, 2026 | @MitchellDScott | Added FR-6: Generic pole and zero extraction `poles<const ORDER>()` and `zeros<const DEG>()` delegating to `Polynomial::roots()`.        |
| 2.3      | September 1, 2026 | @MitchellDScott | Updated `poles()` and `zeros()` to return worst-case buffers `[Complex<T>; D]` and `[Complex<T>; N]` directly from type bounds without generic parameters. |
