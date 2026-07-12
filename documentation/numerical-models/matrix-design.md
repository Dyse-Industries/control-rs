# Matrix Type & Structural Specializations (Design Document)

![Date Badge](https://img.shields.io/badge/Date-July_12,_2026-blue)
![Status Badge](https://img.shields.io/badge/Doc%20Status-Draft-orange)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

---

### **1. Introduction**

The `Matrix` type in `control-rs` provides a statically verified, 2-dimensional
storage representation for numerical types. To support resource-constrained
real-time platforms, the library is strictly constrained to `#![no_std]`
targets, relying entirely on stack allocations and compile-time dimension
validation.

---

### **2. Motivation & Target Constraints**

#### **2.1 Environmental Limitations**

Bare-metal microcontrollers and real-time operating systems (RTOS) lack a
dynamic memory allocator. Every memory allocation must be deterministic,
stack-allocated, and bounded to prevent stack overflow in highly nested control
loop call stacks.

#### **2.2 Target Applications**

- **High-Frequency Control Loops**: State-space controllers, Kalman filtering,
  PID, and Model Predictive Control (MPC) running at sub-millisecond rates.
- **Guidance, Navigation, and Control (GNC)**: Attitude determination and
  orbital propagation algorithms for aerospace and robotics.
- **Hardware-in-the-Loop (HIL)**: Real-time deterministic plant simulations on
  embedded target nodes.

#### **2.3 Safety & Static Verification**

Traditional numerical libraries validate dimensions at runtime, raising
exceptions or panicking upon mismatch. In safety-critical aerospace and
automotive control loops, a runtime panic can result in total system loss. By
leveraging Rust's const generics and type system, dimension validation is
checked at compile-time, halting compilation if a dimension mismatch occurs.

#### **2.4 In-House Math & Stable Const Generics Rationale (Reuse Trade Study)**

Rather than building on external libraries like `nalgebra`'s `no_std` static-storage mode, `control-rs` implements its own custom math module in-house. This design choice is driven by two critical factors:

1. **Generic `const fn` Support on Stable Rust**: To guarantee deterministic start-up and minimize RAM utilization, static matrices must be placed directly in read-only flash memory. This requires matrix constructors to be `const fn`. Since standard library traits (like `Default`) do not allow calling their methods in `const fn` on stable Rust, `control-rs` provides custom `Zero` and `One` traits in `crate::math::num_traits`. These traits expose associated constants (`T::ZERO` and `T::ONE`) instead of methods, allowing the compiler to resolve generic `const fn` constructors at compile time.
2. **Audit Footprint & Certification**: In safety-critical environments (e.g., ISO 26262, DO-178C), every line of dependency code must be audited and certified. `nalgebra` is a powerful, general-purpose library with a massive API surface and a deep dependency chain. Building on it would drastically expand the codebase's audit surface. In contrast, our in-house math module provides a minimal audit surface, optimized precisely for our safety invariants, and enables unified coordinate mapping and layout conversions across `Matrix`, `Polynomial`, and `Tensor` types.

---

### **3. Core Architecture & Memory Layout**

#### **3.1 Generics Foundation & Sizing**

The core `Matrix` structure is defined as:

```rust
pub struct Matrix<T, R: Dim, C: Dim> {
    data: [[T; R::DIM]; C::DIM],
}
```

Dimensions are bound at the type level using the `Dim` trait and Peano number
representations defined in [num_types.rs](../../src/math/num_types.rs). This
allows performing type-level arithmetic (e.g., dimension addition or
multiplication) to statically verify shape changes during matrix operations.

#### **3.2 Memory Layout & Storage Strategy**

The design utilizes a **column-major array representation** (`[[T; R]; C]`).

- **Cache Locality**: The core operation in state-space control is the
  matrix-vector product: $$ x_{k+1} = A x_k + B u_k $$ Computing $ A x_k $
  requires evaluating linear combinations of the columns of $ A $:$$ A x =
  \sum_{j} x_j A_j $$ Under column-major ordering, each column $ A_j $ is
  contiguous in memory, maximizing CPU cache hit rates.
- **BLAS Interoperability**: Column-major layout matches the standard convention
  of legacy BLAS/LAPACK and embedded DSP libraries (e.g., ARM CMSIS-DSP),
  allowing zero-copy routing to hardware-accelerated kernels.

#### **3.3 Memory Representation & Slicing**

To ensure stable memory layout and compatibility with C-based hardware
libraries:

```rust
#[repr(C)]
pub struct Matrix<T, R: Dim, C: Dim> {
    data: [[T; R::DIM]; C::DIM],
}
```

Contiguous internal memory allows exposing zero-copy flat slice interfaces:

```rust
impl<T, R: Dim, C: Dim> Matrix<T, R, C> {
    pub const fn as_slice(&self) -> &[T] {
        // Safe cast as nested arrays are guaranteed to be laid out contiguously
        unsafe { core::slice::from_raw_parts(self.data.as_ptr() as *const T, R::DIM * C::DIM) }
    }

    pub const fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { core::slice::from_raw_parts_mut(self.data.as_mut_ptr() as *mut T, R::DIM * C::DIM) }
    }
}
```

---

### **4. API Specification**

#### **4.1 Instantiation & Constructors**

- `pub const fn zero() -> Self where T: Zero + Copy`: Instantiates an all-zero matrix using `T::ZERO` as the constant initialization value.
- `pub const fn identity() -> Self where T: Zero + One + Copy`: Instantiates an identity matrix (restricted to square shapes) by initializing elements to `T::ZERO` and filling the main diagonal with `T::ONE` via a const-evaluated loop.
- `pub const fn diagonal(val: [T; D::DIM]) -> Matrix<T, D, D> where T: Zero + Copy`: Constructs a diagonal matrix using the provided array of diagonal values and filling off-diagonal elements with `T::ZERO`.
- `pub fn from_fn<F>(mut f: F) -> Self where F: FnMut(usize, usize) -> T`: Generates a matrix using a coordinate-based mapping function at runtime.

*Implementation Note*: To support generic `const fn` initialization on stable Rust, the scalar type `T` must implement the `Zero` and `One` traits from `crate::math::num_traits`. These traits expose the associated constants `T::ZERO` and `T::ONE`. All static constructors are marked `const fn` to allow placing static matrices directly in read-only flash memory.

#### **4.2 Operator Overloading**

Overloads `Add`, `Sub`, and `Mul` from `core::ops`. Dimension rules are
statically enforced at compile-time:

```rust
impl<T, M: Dim, N: Dim, P: Dim> Mul<Matrix<T, N, P>> for Matrix<T, M, N>
where
    T: Copy + Default + Add<Output=T> + Mul<Output=T>,
{
    type Output = Matrix<T, M, P>;
    // ...
}
```

#### **4.3 Core Operations**

- `pub fn transpose(self) -> Matrix<T, C, R>`: Evaluates transposition.
- `pub fn invert(self) -> Result<Self, LinAlgError>`: Inverts a matrix.
- `pub fn determinant(&self) -> T`: Calculates the determinant.

#### **4.4 Interoperability & Conversions**

##### **4.4.1 Conversion to Polynomial**

A square matrix `Matrix<T, D, D>` converts to its characteristic polynomial
`Polynomial<T, <D as DimAdd<U1>>::Output>`.

- **Type Signature**:
  ```rust
  impl<T, D: Dim> TryFrom<Matrix<T, D, D>> for Polynomial<T, <D as DimAdd<U1>>::Output>
  where
      D: DimAdd<U1>,
      <D as DimAdd<U1>>::Output: Dim,
      T: Field + Copy + From<i32>,
  {
      type Error = ConversionError;
      // ...
  }
  ```
- **Behavior**: Coefficients are computed using the Faddeev-LeVerrier
  algorithm (described
  in [Faddeev-LeVerrier Algorithm](https://arxiv.org/pdf/2008.04247)). This
  algorithm is heap-allocation-free and division-free except for division by
  integer iteration steps, making it highly suitable for embedded targets.
- **Failure Condition**: Returns `ConversionError::DimensionMismatch` if the scalar type cannot perform integer division, if numerical overflow occurs, or if capacity is insufficient.

##### **4.4.2 Conversion to Tensor**

Converts a 2D matrix to a rank-2 `Tensor<T, Layout>`.

- **Type Signature**:
  ```rust
  impl<T, R: Dim, C: Dim, Layout: TensorLayout> TryFrom<Matrix<T, R, C>> for Tensor<T, Layout>
  where
      Layout: TensorLayout<Size = <R as DimMul<C>>::Output>,
  {
      type Error = ConversionError;
      // ...
  }
  ```
- **Behavior**: Maps nested column-major arrays into the flat array
  representation of the `Tensor`.
- **Failure Condition**: Returns `ConversionError::LayoutMismatch` if `Layout::RANK != 2` or if the layout's dimensions do not match $ R \times C $.

---

### **5. Error Handling & State Management**

#### **5.1 Compile-Time Constraints**

Dimension mismatches (e.g., adding matrices of different sizes or multiplying
incompatible dimensions) fail at compile-time. Rust's type checker prevents
compiling invalid math.

#### **5.2 Runtime Error Taxonomy**

To supplement the crate's generic `ArithmeticError`, `control-rs` defines dedicated error enums in the `math` module to represent linear algebra and conversion failures:

```rust
/// Unified linear algebra errors supplementing ArithmeticError.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinAlgError {
    /// The matrix is singular (or near-singular under the given numerical tolerance).
    SingularMatrix,
    /// The matrix operation requires a square shape but a non-square shape was provided.
    NonSquareMatrix,
}

/// Representation and layout conversion errors supplementing ArithmeticError.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConversionError {
    /// Rank or coordinate dimensions do not align between Matrix/Tensor.
    LayoutMismatch,
    /// The polynomial is not monic (leading coefficient is not ONE), preventing companion matrix construction.
    NonMonicPolynomial,
    /// Dimension or capacity overflow/underflow during calculations.
    DimensionMismatch,
}
```

#### **5.3 Runtime Fallbacks**

Dynamic operations that cannot be validated statically use soft failure paths:

- Matrix inversion returns a `Result<Self, LinAlgError>` instead of
  panicking, allowing control loops to handle singular conditions (e.g., falling
  back to a degraded state by returning `Err(LinAlgError::SingularMatrix)`).
- Boundary access returns `Option<&T>` via safe `get` methods.

---

### **6. Testing & Validation Framework**

#### **6.1 Host/Target Test Integration**

Tests run on the host via standard `cargo test` for unit verification and
coverage assessments. Target-specific functionality is validated in a
`#![no_std]` environment using a custom lightweight testing runner run under
QEMU to ensure MCU target compatibility.

#### **6.2 Property-Based Testing**

Uses `proptest` on the host to verify algebraic identities across thousands of
randomized matrices:

- Transpose of product: $ (AB)^T = B^T A^T $
- Distributivity: $ A(B + C) = AB + AC $

To verify the safety and numerical stability of our algorithms, the proptest corpus is explicitly populated with deliberately ill-conditioned matrices (e.g., Hilbert matrices, matrices with high condition numbers, and near-singular matrices with components close to `T::epsilon()`). This ensures that the tolerance-based singularity checks correctly catch numerical instability without crashing.

#### **6.3 Benchmarks and Quality Reporting**

Micro-benchmarks measure target CPU clock cycles using the DWT cycle counter on
ARM Cortex-M cores. The `examples/numerical_methods` binaries monitor code bloat
and flash footprint overhead.

---

### **7. Performance & Resource Considerations**

#### **7.1 Stack Overflow Prevention & Memory Safety**

Embedded systems operate under strict stack memory boundaries, as discussed
in [Levy et al, Building an OS in Rust](https://dl.acm.org/doi/10.1145/2818302.2818306).
Since `Matrix` allocations reside entirely on the stack, the library restricts
maximum dimensions to 32 ($ 32 \times 32 $ elements), ensuring that a single
matrix instance never exceeds 4KB of stack space (for `f32` type).

#### **7.2 Code Bloat & Binary Size Validation**

Code footprint is checked against compiler optimization flags (e.g.,
`-C opt-level=z`). We assert that dead-code elimination removes unused dimension
checks and operations.

#### **7.3 Compiler Optimizations & Hardware Acceleration**

Low-level operations map to subprograms
in [subprograms.rs](../../src/math/subprograms.rs). Loops are structured to
allow auto-vectorization (SIMD) and loop unrolling by LLVM. For target hardware
platforms, operations can be conditionally routed to vendor BLAS
implementations (such as ARM CMSIS-DSP).

---

### **8. Structural Specializations & Extensions**

Specialized matrices are implemented as new-type wrappers around `Matrix` to
enforce mathematical invariants and dispatch optimized routines:

```rust
pub struct UpperTriangular<T, D: Dim>(Matrix<T, D, D>);
pub struct LowerTriangular<T, D: Dim>(Matrix<T, D, D>);
pub struct Symmetric<T, D: Dim>(Matrix<T, D, D>);
```

*Design Decision*: Rather than packing triangular data (which requires complex
non-linear index mapping and prevents flat slicing), we wrap a full square
matrix. This trades memory space for cache friendliness and compatibility with
slice-based BLAS kernels.

#### **8.1 Forward and Backward Substitution Examples**

```rust
/// Solves L * x = b for a lower triangular matrix where L is L::DIM x L::DIM and x, b are L::DIM x 1.
pub fn solve_lower_triangular<T, D: Dim>(
    l: &LowerTriangular<T, D>,
    b: &Matrix<T, D, U1>,
    tolerance: T,
) -> Result<Matrix<T, D, U1>, LinAlgError>
where
    T: Field + Signed + Copy,
{
    let n = D::DIM;
    let mut x = Matrix::<T, D, U1>::zero();
    let l_mat = &l.0;

    for i in 0..n {
        let l_ii = l_mat.data[i][i];
        // Tolerance-based singularity check using the type's Signed abs()
        if l_ii.abs() < tolerance {
            return Err(LinAlgError::SingularMatrix);
        }
        let mut sum = T::ZERO;
        for j in 0..i {
            sum = sum + l_mat.data[j][i] * x.data[0][j]; // column-major: data[col][row]
        }
        x.data[0][i] = (b.data[0][i] - sum) / l_ii;
    }
    Ok(x)
}
```

---

### **9. Kalman Filter State Update Example**

The following example demonstrates the proposed `Matrix` API when computing the
covariance update in a Kalman filter loop:
$ P_{k|k} = (I - K_k H_k) P_{k|k-1} $

```rust
use control_rs::math::matrix::{Matrix, Dim, U1};

pub fn kalman_covariance_update<T, S: Dim, O: Dim>(
    p_pred: &Matrix<T, S, S>,
    k: &Matrix<T, S, O>,
    h: &Matrix<T, O, S>,
) -> Matrix<T, S, S>
where
    T: Ring + Copy,
    S: Dim,
    O: Dim,
    S: DimMul<S>,
    S: DimMul<O>,
    O: DimMul<S>,
{
    // Identity matrix I of state dimension S
    let i = Matrix::<T, S, S>::identity();

    // K * H -> S x S matrix
    let k_h = k * h;

    // I - K * H -> S x S matrix
    let diff = &i - &k_h;

    // (I - K * H) * P_pred -> S x S matrix
    &diff * p_pred
}
```

---

### **10. Development Plan & Roadmap**

| Task / Feature               | Description                                                            | Estimated Effort |
|:-----------------------------|:-----------------------------------------------------------------------|:-----------------|
| **Phase 1: Core Layout**     | Define `Matrix` struct, column-major storage, and slice casting.       | 1.0 Day          |
| **Phase 2: Operators**       | Implement `Add`, `Sub`, `Mul` traits with compile-time checks.         | 1.5 Days         |
| **Phase 3: Solvers**         | Implement LU decomposition, determinants, and matrix inversion.        | 2.0 Days         |
| **Phase 4: Specializations** | Create `UpperTriangular`, `LowerTriangular`, and `Symmetric` wrappers. | 1.0 Day          |
| **Phase 5: Factorizations**  | Implement Cholesky ($ L L^T $) and QR solvers.                         | 2.0 Days         |
| **Phase 6: Verification**    | Set up `proptest` suites and target clock cycle benchmarks.            | 1.5 Days         |
| **Phase 7: Interoperability**| Implement `TryFrom` conversions between `Matrix`, `Polynomial` (Faddeev-LeVerrier), and `Tensor`. Depends on Matrix Phase 1 & 2, Tensor Phase 1 & 3, and Polynomial Phase 1. | 2.0 Days |
