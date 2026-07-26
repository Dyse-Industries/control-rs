# Transfer Function Type (Design Document)

![Date Badge](https://img.shields.io/badge/Date-July_26,_2026-blue)
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

- Coefficients are stored in **ascending order of powers**:
  $$ B(s) = b_0 + b_1 s + \dots + b_{N-1} s^{N-1} $$
  $$ A(s) = a_0 + a_1 s + \dots + a_{D-1} s^{D-1} $$
- Denominator capacity $D$ must satisfy $D \ge 1$ and non-zero leading
  coefficient ($a_{D-1} \neq 0$).

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

##### Feedback (Closed-Loop) Connection

$$\frac{H_1(s)}{1 + H_1(s) H_2(s)} = \frac{B_1 A_2}{A_1 A_2 + B_1 B_2}$$

#### 5.4 Discretization (Continuous-to-Discrete)

Converts a continuous $H(s)$ to discrete $H(z)$ via Bilinear (Tustin)
transformation:
$$s \approx \frac{2}{T_s} \frac{z - 1}{z + 1}$$

```rust
impl<T, N: Dim, D: Dim, Sn: Storage<T, N, U1>, Sd: Storage<T, D, U1>> TransferFunction<T, N, D, Sn, Sd> {
    pub fn to_discrete_tustin(
        &self,
        sample_time: T,
    ) -> ArrayTransferFunction<T, N, D>
    where
        T: Copy + RealField,
    {
        // Direct algebraic expansion over numerator and denominator storage
        // ...
    }
}
```

#### 5.5 State-Space Canonical Conversions

Converts a strictly proper continuous transfer
function $H(s) = \frac{b_{m} s^m + \dots + b_0}{s^n + a_{n-1} s^{n-1} + \dots + a_0}$ (
where $D = n + 1$) directly to a `StateSpace<T, N_X, U1, U1>` system in
Controllable Canonical Form:

$$\mathbf{A} = \begin{bmatrix} 0 & 1 & 0 & \dots & 0 \\ 0 & 0 & 1 & \dots & 0 \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ -a_0 & -a_1 & -a_2 & \dots & -a_{n-1} \end{bmatrix}, \quad \mathbf{B} = \begin{bmatrix} 0 \\ 0 \\ \vdots \\ 1 \end{bmatrix}$$
$$\mathbf{C} = \begin{bmatrix} b_0 & b_1 & \dots & b_{n-1} \end{bmatrix}, \quad \mathbf{D} = \begin{bmatrix} d_0 \end{bmatrix}$$

---

### 6. Alternatives

| Architecture Option                 | Advantages                                                                                                            | Disadvantages                                                                           | Decision     |
|:------------------------------------|:----------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------|:-------------|
| **Wrapping `Polynomial`**           | Reuses existing polynomial methods.                                                                                   | Breaks container peer model; adds artificial coupling; forces extra abstraction layers. | **Rejected** |
| **Direct Storage Wrapper (Chosen)** | Symmetric with `Matrix` and `Polynomial`; zero cost; direct access to Peano/DSP/BLAS; supports views and ROM storage. | Requires implementing evaluation and convolution calls against storage directly.        | **Selected** |

---

### 7. Verification & Validation

1. **Unit Tests**: Test frequency response evaluations against known analytic
   transfer functions (e.g. 1st order low-pass filter, 2nd order
   mass-spring-damper).
2. **Property-Based Testing**: Validate algebraic identities using `proptest` (
   e.g. $H_1 \cdot H_2 = H_2 \cdot H_1$, feedback stability bounds).
3. **Cross-Validation**: Compare discretization (Tustin) and canonical
   state-space matrices against MATLAB `tf` and Python `control` package
   reference outputs.
4. **HIL Verification**: Execute cross-compiled continuous-to-discrete filter
   loops on hardware target runners.

---

### 8. Performance & Resource Considerations

- **Stack Overhead**: Owning transfer functions `ArrayTransferFunction<T, N, D>`
  occupy $(N + D) \times \text{size\_of}(T)$ bytes.
- **Borrowed Views**: `TransferFunctionView<'a, T, N, D>` occupies only 2
  pointer references plus stride/length parameters, incurring zero allocation.
- **Inline Operations**: Storage element accessors monomorphize into direct
  memory loads.

---

### 9. Development Plan & Roadmap

| Task / Feature                              | Description                                                                        | Estimated Effort |
|:--------------------------------------------|:-----------------------------------------------------------------------------------|:-----------------|
| **Phase 1: Storage Wrapper & Constructors** | Base `TransferFunction` struct, storage traits, and slice accessors.               | 1.0 Day          |
| **Phase 2: Frequency Evaluation**           | Direct Horner evaluation over storage for $H(j\omega)$ and Bode calculations.      | 1.0 Day          |
| **Phase 3: Algebra & DSP Convolution**      | Implement series, parallel, and feedback connections using direct DSP convolution. | 1.5 Days         |
| **Phase 4: Discretization**                 | Bilinear (Tustin) transform and ZOH algorithm routines.                            | 1.5 Days         |
| **Phase 5: State-Space Conversion**         | Controllable and Observable Canonical Form conversions.                            | 1.5 Days         |
| **Phase 6: Verification Suite**             | Unit tests, `proptest` suites, and MATLAB cross-validation.                        | 1.0 Day          |

---

### 10. Revision History

| Date          | Author          | Description                                                                                                                                     |
|:--------------|:----------------|:------------------------------------------------------------------------------------------------------------------------------------------------|
| July 26, 2026 | @MitchellDScott | Initial draft establishing `TransferFunction` as a standalone peer storage wrapper interacting directly with Peano, DSP, and subprogram traits. |
