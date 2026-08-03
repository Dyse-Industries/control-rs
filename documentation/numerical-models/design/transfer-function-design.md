# Transfer Function Type (Design Document)

![Date Badge](https://img.shields.io/badge/Date-August_2,_2026-blue)
![Status Badge](https://img.shields.io/badge/Doc%20Status-Draft-orange)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

---

### 1. Introduction

The `TransferFunction` module provides a statically sized representation of
rational transfer functions $H(s) = \frac{B(s)}{A(s)}$ (continuous time)
and $H(z) = \frac{B(z)}{A(z)}$ (discrete time) for linear time-invariant (LTI)
control systems and digital signal processing.

Following the core architecture of `control-rs`, `TransferFunction` is a *
*standalone, type-safe wrapper** built directly on top of generic storage
backends (`Storage<T, N, U1>` and `Storage<T, D, U1>`). It does **not** wrap
`Polynomial` objects under the hood; instead, it operates directly on numerator
and denominator storage memory via lower-level Peano dimension traits (
`num_types.rs`), DSP subprogram traits (such as convolution and polynomial
evaluation), and BLAS kernels. This design preserves symmetry with `Matrix` and
`Polynomial`, ensuring zero-cost abstraction, zero dynamic heap allocation (
`#![no_std]`), and flexible memory ownership (owning arrays, borrowed views, and
Flash ROM tables).

---

### 2. Requirements

#### 2.1 Functional Requirements

##### FR-1: Direct Storage Parameterization

The `TransferFunction` type must accept independent storage backends for the
numerator ($N$ coefficients) and denominator ($D$ coefficients):
`TransferFunction<T, N: Dim, D: Dim, Sn: Storage<T, N, U1>, Sd: Storage<T, D, U1>>`

##### FR-2: Domain Encoding (Continuous vs. Discrete)

The type must encode domain information (continuous $s$-domain vs. discrete $z$
-domain) and hold an optional sampling period $T_s$.

##### FR-3: Direct Interoperability with Math & DSP Traits

All mathematical operations—frequency response
evaluation ($H(j\omega)$ / $H(e^{j\omega T_s})$), system algebra (
cascade/series, parallel, feedback), discretization, and canonical
transformations—must interact directly with Peano dimension traits (`Dim`,
`DimAdd`, `DimSub`), DSP subprogram traits (e.g. `Convolution<T>`, Horner
evaluation), and BLAS kernels without delegating to high-level `Polynomial`
wrappers.

##### FR-4: System Interconnections

The module must provide static capacity computation for algebraic connections:

- **Series (Cascade)**: $H_1(s) \cdot H_2(s) = \frac{B_1 B_2}{A_1 A_2}$
- **Parallel**: $H_1(s) + H_2(s) = \frac{B_1 A_2 + B_2 A_1}{A_1 A_2}$
- **Feedback (Closed-Loop)
  **: $\frac{H_1(s)}{1 + H_1(s) H_2(s)} = \frac{B_1 A_2}{A_1 A_2 + B_1 B_2}$

##### FR-5: Discretization Algorithms

The module must support continuous-to-discrete transformations:

- **Bilinear (Tustin) Transform**: $s \approx \frac{2}{T_s} \frac{z - 1}{z + 1}$
  with optional frequency pre-warping.
- **Zero-Order Hold (ZOH)**: Exact step-invariant discretization.

##### FR-6: State-Space Canonical Conversions

The module must support bidirectional conversions between transfer functions and
state-space systems (`StateSpace<T, NX, NU, NY>`) in Controllable Canonical Form
and Observable Canonical Form.

#### 2.2 Non-Functional Requirements

##### NFR-1: Zero Dynamic Allocation (`#![no_std]`, `no_alloc`)

All operations must execute deterministically without relying on heap
allocation (`Vec`, `Box`).

##### NFR-2: Zero-Cost Abstraction

Storage abstraction and Peano dimension bounds must monomorphize completely,
matching hand-optimized raw array operations.

##### NFR-3: Contiguity-Gated Slice Views

Safe slice accessors (`num_slice()`, `den_slice()`) must be exposed if and only
if the underlying storage backends implement `ContiguousStorage`.

#### 2.3 Constraints

- **Coefficient Ordering**: Ascending order of powers for both numerator and
  denominator, following the same convention as `Polynomial` (see
  [polynomial-design.md §2.3](polynomial-design.md#23-constraints)):
  $$ B(s) = b_0 + b_1 s + \dots + b_{N-1} s^{N-1}, \quad A(s) = a_0 + a_1 s + \dots + a_{D-1} s^{D-1} $$
- **Denominator Validity**: Denominator capacity $D$ must satisfy $D \ge 1$
  and a non-zero leading coefficient ($a_{D-1} \neq 0$).

---

### 3. Technical Overview

`TransferFunction` acts as a domain-aware rational function container over
numerator storage `Sn` and denominator storage `Sd`. Rather than layering
abstractions, `TransferFunction` interacts directly with:

- **`crate::math::num_types`**: Peano arithmetic (`DimAdd`, `DimSub`, `U1`,
  etc.) for compile-time shape verification.
- **`crate::math::subprograms`**: BLAS Level 1/2/3 subprograms (`AXPY`, `SCAL`,
  `GEMV`, `GEMM`) and DSP helpers (`CONV`, Horner's evaluation).
- **`crate::math::storage`**: Storage traits (`Storage`, `StorageMut`,
  `ContiguousStorage`, `ContiguousStorageMut`).

---

### 4. Core Architecture

#### 4.1 Type Signature & Storage Layout

```rust
pub struct TransferFunction<
    T,
    N: Dim,
    D: Dim,
    Sn: Storage<T, N, U1> = ArrayStorage<T, N, U1>,
    Sd: Storage<T, D, U1> = ArrayStorage<T, D, U1>,
> {
    num_storage: Sn,
    den_storage: Sd,
    sample_time: Option<T>, // None = Continuous (s-domain), Some(Ts) = Discrete (z-domain)
    _marker: core::marker::PhantomData<(N, D)>,
}
```

- **Numerator Storage (`Sn`)**: Holds $N$ coefficients for $B(s)$ or $B(z)$.
- **Denominator Storage (`Sd`)**: Holds $D$ coefficients for $A(s)$ or $A(z)$.
- **Sampling Time (`sample_time`)**: `None` specifies a continuous s-domain
  transfer function $H(s)$; `Some(Ts)` specifies a discrete z-domain transfer
  function $H(z)$ with period $T_s$.

#### 4.2 Storage Backends & Zero-Copy Views

```rust
/// Owning transfer function with stack-allocated arrays
pub type ArrayTransferFunction<T, N, D> = TransferFunction<T, N, D, ArrayStorage<T, N, U1>, ArrayStorage<T, D, U1>>;

/// Zero-copy borrowed read-only transfer function view over &[T] slices
pub type TransferFunctionView<'a, T, N, D> = TransferFunction<T, N, D, MatrixView<'a, T, N, U1>, MatrixView<'a, T, D, U1>>;

/// Zero-copy borrowed mutable transfer function view over &mut [T] slices
pub type TransferFunctionViewMut<'a, T, N, D> = TransferFunction<T, N, D, MatrixViewMut<'a, T, N, U1>, MatrixViewMut<'a, T, D, U1>>;
```

#### 4.3 Slicing & Memory Access

```rust
impl<T, N: Dim, D: Dim, Sn, Sd> TransferFunction<T, N, D, Sn, Sd>
where
    Sn: ContiguousStorage<T, N, U1>,
    Sd: ContiguousStorage<T, D, U1>,
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
pub type TransferFunctionMatrix<T, R: Dim, C: Dim, N: Dim, D: Dim, S = ArrayStorage<TransferFunction<T, N, D>, R, C>> =
Matrix<TransferFunction<T, N, D>, R, C, S>;
```

#### 4.5 Error Handling

Following the crate-wide error strategy, fallible constructors return
`Result<T, Error>` via a crate-local `thiserror` enum rather than panicking.
The only runtime-checked invariant at construction is FR-6/§2.3's Denominator
Validity constraint ($D \ge 1$, non-zero leading coefficient $a_{D-1}$):

```rust
#[derive(Debug, thiserror::Error)]
pub enum TransferFunctionError {
    #[error("denominator leading coefficient must be non-zero")]
    ZeroLeadingDenominatorCoefficient,
}
```

Runtime constructors that accept caller-supplied coefficients (`from_slices`,
and any validating variant of `from_coefficients`/`from_storage`) return
`Result<Self, TransferFunctionError>`. No `unwrap()`, `expect()`, or `panic!()`
is used outside tests and examples. Near-pole frequency-response evaluation
(§5.2) and partial-fraction ZOH decomposition (§5.4) are *not* treated as
constructor-time errors — a valid `TransferFunction` may still be evaluated at
an ill-conditioned point at call time, which is documented behavior (§5.2)
rather than a distinct error variant, matching how `python-control`'s
`warn_infinity` documents rather than rejects evaluation at a pole.

---

### 5. API Specification & Operations

#### 5.1 Constructors

- **From Storage**:
  `pub const fn from_storage(num_storage: Sn, den_storage: Sd, sample_time: Option<T>) -> Self`
- **Owning Stack Constructor**:
  `pub const fn from_coefficients(num: [T; N::DIM], den: [T; D::DIM], sample_time: Option<T>) -> ArrayTransferFunction<T, N, D>`
- **Zero-Copy Slice View Constructor**:
  `pub fn from_slices(num: &'a [T], den: &'a [T], sample_time: Option<T>) -> TransferFunctionView<'a, T, N, D>`

#### 5.2 Frequency Response Evaluation

Evaluates frequency response $H(s)$ at $s = j\omega$ or $H(z)$
at $z = e^{j\omega T_s}$ using direct Horner evaluation on the numerator and
denominator storage backends:

$$\text{Num}(s) = \text{Horner}(B, s), \quad \text{Den}(s) = \text{Horner}(A, s)$$
$$H(s) = \frac{\text{Num}(s)}{\text{Den}(s)}$$

```rust
impl<T, N: Dim, D: Dim, Sn: Storage<T, N, U1>, Sd: Storage<T, D, U1>> TransferFunction<T, N, D, Sn, Sd> {
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
governed by that point's condition number, which grows as $s$ approaches a
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

#### 5.3 System Interconnections

Algebraic interconnections invoke DSP subprograms (such as polynomial
convolution routines) directly on storage elements without creating intermediate
`Polynomial` objects:

##### Series (Cascade) Connection

$$H_1(s) \cdot H_2(s) = \frac{B_1 B_2}{A_1 A_2}$$
Numerator dimension bound: $(N_1 + N_2 - 1)$. Denominator dimension
bound: $(D_1 + D_2 - 1)$.

```rust
impl<T, N1: Dim, D1: Dim, Sn1: Storage<T, N1, U1>, Sd1: Storage<T, D1, U1>> TransferFunction<T, N1, D1, Sn1, Sd1> {
    pub fn series<N2: Dim, D2: Dim, Sn2: Storage<T, N2, U1>, Sd2: Storage<T, D2, U1>>(
        &self,
        other: &TransferFunction<T, N2, D2, Sn2, Sd2>,
    ) -> TransferFunction<
        T,
        <<N1 as DimAdd<N2>>::Output as DimSub<U1>>::Output,
        <<D1 as DimAdd<D2>>::Output as DimSub<U1>>::Output,
    >
    where
        N1: DimAdd<N2>,
        <N1 as DimAdd<N2>>::Output: DimSub<U1>,
        D1: DimAdd<D2>,
        <D1 as DimAdd<D2>>::Output: DimSub<U1>,
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
degree bounds, not their sum, and is expressed via `DimMax` rather than a
further `DimAdd`:

```rust
impl<T, N1: Dim, D1: Dim, Sn1: Storage<T, N1, U1>, Sd1: Storage<T, D1, U1>> TransferFunction<T, N1, D1, Sn1, Sd1> {
    pub fn feedback<N2: Dim, D2: Dim, Sn2: Storage<T, N2, U1>, Sd2: Storage<T, D2, U1>>(
        &self,
        other: &TransferFunction<T, N2, D2, Sn2, Sd2>,
    ) -> TransferFunction<
        T,
        <N1 as DimAdd<D2>>::Output,
        <<D1 as DimAdd<D2>>::Output as DimMax<<N1 as DimAdd<N2>>::Output>>::Output,
    >
    where
        N1: DimAdd<D2> + DimAdd<N2>,
        D1: DimAdd<D2>,
        <D1 as DimAdd<D2>>::Output: DimMax<<N1 as DimAdd<N2>>::Output>,
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

None of series, parallel, or feedback perform pole-zero cancellation or degree
reduction after combination. `python-control`'s equivalent operators do not
either — `minreal()` is a separate, explicit, user-invoked operation in the
reference implementation (python-control, `control.minreal`). `TransferFunction`
combination methods therefore always produce a result at the full static
capacity bound derived above; a fixed-capacity type cannot shrink its own
dimension even where the true post-convolution polynomial degree is lower
(e.g. an exact or near-exact pole-zero cancellation introduced by a feedback
loop). A `minreal`-equivalent capacity-reducing operation is left as future
work (see §9 Risks & Open Questions) rather than an implicit side effect of
algebra.

#### 5.4 Discretization (Continuous-to-Discrete)

Converts a continuous $H(s)$ to discrete $H(z)$ via Bilinear (Tustin)
transformation with pre-warping or Zero-Order Hold (ZOH).

##### Bilinear (Tustin) Transform

$$s \approx \frac{2}{T_s} \frac{z - 1}{z + 1}$$

Frequency pre-warping is exposed as a first-class parameter of the Tustin path
rather than an afterthought: MATLAB's own discretization guidance states that
Tustin without pre-warping "introduces a frequency shift that is unacceptable
for many applications" (MathWorks, *Discretize a Compensator*), remedied by
specifying a critical frequency to match exactly under the transform (Ogata,
2010; Franklin et al., 1998).

```rust
impl<T, N: Dim, D: Dim, Sn: Storage<T, N, U1>, Sd: Storage<T, D, U1>> TransferFunction<T, N, D, Sn, Sd> {
    pub fn to_discrete_tustin(
        &self,
        sample_time: T,
        prewarp_frequency: Option<T>,
    ) -> ArrayTransferFunction<T, N, D>
    where
        T: Copy + Float,
    {
        // Direct algebraic expansion over numerator and denominator storage
        // ...
    }
}
```

`T: Float` (`num-traits-design.md`'s current hierarchy — see §9) replaces
a prior `T: Real` bound: the pre-warping path needs `tan()` and the
$\frac{2}{T_s}$ factor needs division, both of which `Float` provides and
the narrower general-arithmetic bounds elsewhere in this document do not.

##### Zero-Order Hold (ZOH)

ZOH admits two distinct computational paths with materially different
numerical profiles:

- **Transfer-function-direct (chosen)**: $G(z) = (1 - z^{-1})\,
  \mathcal{Z}\left[\mathcal{L}^{-1}\left\{\frac{G(s)}{s}\right\}\right]$,
  via partial-fraction expansion of $G(s)/s$ followed by table-based
  $z$-transform of each term (Franklin et al., 1998). This path never forms a
  state-space realization or a matrix exponential, so it does not inherit the
  companion-form conditioning risk described in §5.5; its own numerical risk is
  partial-fraction decomposition, which is itself ill-conditioned for
  closely-spaced or repeated poles and must be bounded/documented at
  implementation time.
- **State-space-mediated (rejected for this path)**: convert to a canonical
  realization, discretize $A$ via matrix exponential (Moler & Van Loan, 2003),
  convert back. Rejected as the *default* ZOH path specifically because it
  would silently inherit whatever conditioning risk the chosen state-space
  realization carries — material for controllable/observable canonical form
  per §5.5's finding. The state-space-mediated path remains available
  independently through explicit use of §5.5's conversion plus a `StateSpace`
  discretization method, for callers who already need a state-space
  realization for other reasons.

#### 5.5 State-Space Canonical Conversions

Converts a strictly proper continuous transfer
function $H(s) = \frac{b_{m} s^m + \dots + b_0}{s^n + a_{n-1} s^{n-1} + \dots + a_0}$ (
where $D = n + 1$) directly to a `StateSpace<T, N_X, U1, U1>` system in
Controllable Canonical Form:

$$\mathbf{A} = \begin{bmatrix} 0 & 1 & 0 & \dots & 0 \\ 0 & 0 & 1 & \dots & 0 \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ -a_0 & -a_1 & -a_2 & \dots & -a_{n-1} \end{bmatrix}, \quad \mathbf{B} = \begin{bmatrix} 0 \\ 0 \\ \vdots \\ 1 \end{bmatrix}$$
$$\mathbf{C} = \begin{bmatrix} b_0 & b_1 & \dots & b_{n-1} \end{bmatrix}, \quad \mathbf{D} = \begin{bmatrix} d_0 \end{bmatrix}$$

**Conditioning Caveat**: Controllable/observable canonical (companion) form is
structurally correct but numerically fragile above low system order. MathWorks
documents that "the transformation to companion form is based on the
controllability matrix, which is almost always numerically singular for
mid-range orders" and marks its own direct companion-form realization command
"Not recommended" in current product documentation (MathWorks, *Canonical
State-Space Realizations*; MathWorks, `canon`). Formal analysis of
companion-form controllability radii confirms this is a structural property of
the form itself, not an implementation artifact (companion-form controllability
radii literature), and recent work characterizes the condition number of the
standard companion-form transformation as growing exponentially with system
dimension, treating numerically reliable computation of it as an open problem
(Yang & Jones, 2026). This design scopes §5.5 to controllable/observable
canonical form for its structural value (explicit characteristic-polynomial
coefficients in $\mathbf{A}$) and low-to-moderate system order; balanced or
modal realization — MathWorks' own recommended numerically-preferred
alternative — is left as future work (see §9 Risks & Open Questions) rather
than implemented in the initial revision.

---

### 6. Alternatives

| Architecture Option                     | Advantages                                                                                                            | Disadvantages                                                                                                                                                                     | Decision     |
|:-----------------------------------------|:----------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------|
| **Wrapping `Polynomial`**               | Reuses existing polynomial methods.                                                                                   | Breaks container peer model; adds artificial coupling; forces extra abstraction layers — the same rationale `state-space-design.md` uses to reject wrapping `Matrix` fields directly. | **Rejected** |
| **Second-Order-Sections (SOS) Cascade** | Standard embedded-DSP answer to coefficient sensitivity growing with filter order (ARM CMSIS-DSP `BiquadCascadeDF2T`; Rust `biquad` crate); bounds conditioning per-stage rather than across the full polynomial. | Cannot represent an arbitrary rational transfer function, only designed filters reducible to cascaded biquads; would require a structurally different type from this general $N/D$ container. | **Rejected** |
| **Direct Storage Wrapper (Chosen)**     | Symmetric with `Matrix` and `Polynomial`; zero cost; direct access to Peano/DSP/BLAS; supports views and ROM storage. | Requires implementing evaluation and convolution calls against storage directly; flat coefficient representation inherits the coefficient-sensitivity growth SOS is designed to avoid, at high order. | **Selected** |

---

### 7. Verification & Validation

1. **Unit Tests**: Test frequency response evaluations ($H(j\omega)$) against
   known analytic
   transfer functions (e.g. 1st order low-pass filter, 2nd order
   mass-spring-damper) using complex arithmetic bounds (Henrici, 1974;
   Oppenheim & Schafer, 2009).
2. **Property-Based Testing**: Validate algebraic identities using `proptest` (
   e.g. $H_1 \cdot H_2 = H_2 \cdot H_1$, feedback stability bounds) following
   QuickCheck methodology (Claessen & Hughes, 2000).
3. **Cross-Validation**: Compare discretization (Tustin) and canonical
   state-space matrices against MATLAB `tf` and Python `control` package
   reference outputs. `python-control` and MATLAB both order coefficients in
   descending powers (highest power first), the opposite of this crate's
   ascending-power convention (§2.3, inherited from `Polynomial`) — the
   cross-validation harness must explicitly reverse coefficient order when
   constructing reference-library inputs and when comparing their outputs,
   rather than assuming a matching layout.
4. **Non-Minimality Regression Tests**: Verify series/parallel/feedback
   produce the full static-capacity result with no implicit pole-zero
   cancellation (§5.3), matching `python-control`'s behavior where `minreal()`
   is a separate, explicit operation.
5. **HIL Verification**: Execute cross-compiled continuous-to-discrete filter
   loops on hardware target runners under real-time safety standards (ISO 26262,
   DO-178C).

---

### 8. Performance & Resource Considerations

- **Stack Overhead**: Owning transfer functions `ArrayTransferFunction<T, N, D>`
  occupy $(N + D) \times \text{size\_of}(T)$ bytes.
- **Borrowed Views**: `TransferFunctionView<'a, T, N, D>` occupies only 2
  pointer references plus stride/length parameters, incurring zero allocation.
- **Inline Operations**: Storage element accessors monomorphize into direct
  memory loads.

---

### 9. Risks & Open Questions

- **Non-Minimal Realization Handling**: Series/parallel/feedback never
  cancel poles/zeros (§5.3). A `minreal`-equivalent capacity-reducing
  operation is not yet scoped; whether and how to offer one is deferred to a
  future revision.
- **Partial-Fraction Conditioning for ZOH**: The chosen transfer-function-direct
  ZOH path (§5.4) requires partial-fraction decomposition, which is itself
  ill-conditioned for closely-spaced or repeated poles. The implementation
  phase must bound or document this explicitly rather than assume it away.
- **Canonical Form Scope**: Controllable/observable canonical form (§5.5) is
  numerically fragile above low system order (Kenney & Laub, 1988; Yang &
  Jones, 2026). Balanced or modal realization is identified as the
  numerically-preferred alternative used by reference implementations but is
  not implemented in this revision.
- **Compensated Horner Evaluation**: Near-pole frequency-response evaluation
  (§5.2) is documented as inherently ill-conditioned rather than compensated.
  If measured Bode-sweep accuracy proves insufficient once implemented,
  compensated Horner evaluation (Graillat, Langlois, & Louvet, 2006) is the
  identified mitigation path.
- **`num-traits-design.md` Dependency Is Provisional**: `to_discrete_tustin`'s
  `T: Float` bound (§5.4) follows `num-traits-design.md`'s current
  hierarchy, which has not yet been through `/cr-research` or
  `/cr-design-doc` and will be revised independently of this document —
  matching the same caveat `matrix-design.md` §7, `state-space-design.md`
  §9, and `tensor-design.md` §7 already carry for the identical dependency.
  Separately, `Convolution<T>` (FR-3, `src/math/dsp.rs`) is shipped code
  still bound on the retired `T: Real`; its migration is tracked in
  `polynomial-design.md` §9, which owns the primary specification of that
  trait's usage.

---

### 10. Development Plan & Roadmap

| Task / Feature                              | Description                                                                        | Estimated Effort |
|:--------------------------------------------|:-----------------------------------------------------------------------------------|:-----------------|
| **Phase 1: Storage Wrapper & Constructors** | Base `TransferFunction` struct, storage traits, slice accessors, and error type.    | 1.0 Day          |
| **Phase 2: Frequency Evaluation**           | Direct Horner evaluation over storage for $H(j\omega)$ and Bode calculations.      | 1.0 Day          |
| **Phase 3: Algebra & DSP Convolution**      | Implement series, parallel, and feedback connections using direct DSP convolution. | 1.5 Days         |
| **Phase 4: Discretization**                 | Bilinear (Tustin, with pre-warping) transform and transfer-function-direct ZOH, including partial-fraction decomposition (§9's closely-spaced/repeated-pole conditioning risk must be bounded, not assumed away). | 2.5 Days         |
| **Phase 5: State-Space Conversion**         | Controllable and Observable Canonical Form conversions.                            | 1.5 Days         |
| **Phase 6: Verification Suite**             | Unit tests, `proptest` suites, and cross-validation against two external reference implementations (MATLAB, `python-control`).       | 2.0 Days          |

---

### 11. References

#### 11.1. Practical

1. **Franklin, G. F., Powell, J. D., & Workman, M. L. (1998).** *Digital Control
   of Dynamic Systems* (3rd ed.). Addison-Wesley. — Implementation-oriented
   treatment of discretization and digital filter realization on embedded
   microcontrollers; direct transfer-function-domain ZOH formula.
2. **Moler, C., & Van Loan, C. (2003).** Nineteen Dubious Ways to Compute the
   Exponential of a Matrix, Twenty-Five Years Later. *SIAM Review*, 45(1),
   3–49. — Rationale and conditioning concerns for the state-space-mediated
   ZOH path (§5.4).
3. **python-control developers.** `control/xferfcn.py`, python-control library
   source. <https://github.com/python-control/python-control/blob/main/control/xferfcn.py>
   — Reference `series`/`parallel`/`feedback` algebra and non-minimality
   behavior (§5.3).
4. **MathWorks.** *Discretize a Compensator*, MATLAB & Simulink documentation.
   <https://www.mathworks.com/help/control/ug/discretize-a-compensator.html>
   — ZOH as `c2d` default; Tustin frequency-shift and pre-warping guidance
   (§5.4).
5. **MathWorks.** *Canonical State-Space Realizations* and `canon`, MATLAB
   documentation.
   <https://www.mathworks.com/help/control/ug/canonical-state-space-realizations.html>
   — Companion-form conditioning caveat and "Not recommended" status (§5.5).
6. **ARM Ltd.** *Biquad Cascade IIR Filters Using a Direct Form II Transposed
   Structure*, CMSIS-DSP documentation.
   <https://arm-software.github.io/CMSIS-DSP/main/group__BiquadCascadeDF2T.html>
   — Second-order-sections cascade convention considered in §6 Alternatives.

#### 11.2. Theoretical

7. **Oppenheim, A. V., & Schafer, R. W. (2009).** *Discrete-Time Signal
   Processing* (3rd ed.). Pearson. — FIR/IIR filter representation, transfer
   function stability, and frequency-response $H(e^{j\omega})$ theory.
8. **Ogata, K. (2010).** *Modern Control Engineering* (5th ed.). Prentice
   Hall. — Continuous transfer functions, bilinear (Tustin) transformation, and
   frequency pre-warping derivations.
9. **Henrici, P. (1974).** *Applied and Computational Complex Analysis, Volume
   1*. Wiley. — Complex-arithmetic numerical stability foundations
   for $H(j\omega)$ evaluation.
10. **Higham, N. J. (2002).** *Accuracy and Stability of Numerical Algorithms*
    (2nd ed.). SIAM, Ch. 5. — Horner's method backward-error bound and
    conditioning near polynomial roots (§5.2).
11. **Graillat, S., Langlois, P., & Louvet, N. (2006).** Faithful Polynomial
    Evaluation with Compensated Horner Algorithm. *Proceedings of the 17th
    IEEE Symposium on Computer Arithmetic*. — Compensated Horner as a
    documented mitigation for near-root evaluation error (§5.2).
12. **Kenney, C. S., & Laub, A. J. (1988).** Controllability and stability
    radii for companion form systems. *Mathematics of Control, Signals and
    Systems*, 1, 239–256. — Formal treatment of near-uncontrollability and
    near-singularity in high-order companion-form realizations (§5.5).
13. **Yang, S., & Jones, C. N. (2026).** Numerically Reliable Brunovsky
    Transformations. — Exponential condition-number growth of the standard
    companion-form transformation with system dimension (§5.5).

#### 11.3. Standards, Safety and Verification

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

---

### 12. Revision History

| Date          | Author          | Description                                                                                                                                     |
|:--------------|:----------------|:------------------------------------------------------------------------------------------------------------------------------------------------|
| July 26, 2026 | @MitchellDScott | Initial draft establishing `TransferFunction` as a standalone peer storage wrapper interacting directly with Peano, DSP, and subprogram traits. |
| July 26, 2026 | @MitchellDScott | Added inline academic citations and 3-tiered references section.                                                                                |
| August 1, 2026 | @MitchellDScott | Deduplicated the coefficient-ordering constraint by cross-referencing `polynomial-design.md`'s canonical statement instead of restating it. |
| August 2, 2026 | @MitchellDScott | Documented near-pole Horner conditioning (§5.2); derived `DimMax`-based capacity bounds and the non-minimal-result guarantee for parallel/feedback (§5.3); split ZOH into a chosen transfer-function-direct path and a rejected state-space-mediated path (§5.4); added a canonical-form conditioning caveat (§5.5), Error Handling (§4.5), SOS cascade alternative (§6), and Risks & Open Questions (§9). |
| August 2, 2026 | @MitchellDScott | Propagated the `num-traits-design.md` pivot to `to_discrete_tustin`'s bound (`T: Real` → `T: Float`, §5.4); added a Risks entry for that dependency's still-pre-research status and for `Convolution<T>`'s shipped, unmigrated `T: Real` bound; revised Phase 4/6 development-plan estimates upward (partial-fraction ZOH and dual-reference-implementation cross-validation were under-scoped). |
