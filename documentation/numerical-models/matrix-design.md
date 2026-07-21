# Matrix Type & Structural Specializations (Design Document)

![Date Badge](https://img.shields.io/badge/Date-July_19,_2026-blue)
![Status Badge](https://img.shields.io/badge/Doc%20Status-Draft-orange)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

---

### **1. Introduction**

This module provides a type-safe wrapper for DSP and subprogram traits to
support the matrix operations required by advanced control and state estimation.
Like `nalgebra`, it utilizes generics and traits to enable compile-time
specializations for various matrix shapes.

---

### **2. Requirements**

#### **2.1. Functional Requirements**

- **Compile-Time Sizing**: Enforce dimensions of arguments at compile time
  using [math::num_types](../../src/math/num_types.rs)
- **Static Constructors**: Provide compile-time evaluated constructors for zero
  matrices, identity matrices, and diagonal matrices.
- **Core Arithmetic**: Implement standard operator overloading for matrix
  addition, subtraction, multiplication, and negation.
- **Matrix Operations**: Provide methods for matrix transposition, determinant
  calculation, and inversion.
- **Specializations**: Support specialized structures (Upper Triangular, Lower
  Triangular, and Symmetric) to dispatch optimized mathematical routines.
- **Coordinate-Based Instantiation**: Expose coordinate-based mapping functions
  to initialize elements at runtime.
- **Type Conversions**: Support conversions between `Matrix`, `Polynomial`, and
  `Tensor` representations (e.g., computing a characteristic polynomial from a
  square matrix, or mapping a 2D matrix to a rank-2 tensor).

#### **2.2. Non-Functional Requirements**

- **Deterministic Execution Bounds**: Matrix operations—especially determinant
  calculation and inversion—must execute within predictable, deterministic
  timeframes.
- **Compile-Time Evaluation Overhead**: The extensive use of `const fn` for
  static constructors and dimension enforcement must not cause unreasonable
  compile-time degradation or binary bloat.
- **Specialization Optimization**: The triangular specializations should run
  in about half the operations of a regular matrix multiplication.

#### **2.3. Constraints**

- **No-Std Environment**: The code must compile and run in `#![no_std]`
  environments without the Rust standard library.
- **No Dynamic Allocation**: The module must not use a heap allocator. All
  memory allocations must be static or stack-based.
- **Memory Footprint**: Limit maximum matrix dimensions to $32 \times 32$
  elements to guarantee that a single matrix instance never exceeds 4KB of stack
  space (when using 32-bit floats).

---

### **3. Technical Overview**

The scope of this module includes the data structures, traits, constructors,
operator implementations, and numerical solvers required for two-dimensional
linear algebra.

---

### **4. Core Architecture**

#### **4.1. Generics Foundation & Sizing**

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

#### **4.2. Memory Layout & Storage Strategy**

The design utilizes a **column-major array representation** (`[[T; R]; C]`).

- **Cache Locality**: Matrix multiplication: Under column-major ordering, each
  column $ A_j $ is contiguous in memory, maximizing CPU cache hit rates.
- **BLAS Interoperability**: Column-major layout matches the standard convention
  of legacy BLAS/LAPACK and embedded DSP libraries (e.g., ARM CMSIS-DSP),
  allowing zero-copy routing to hardware-accelerated kernels.

#### **4.3. Memory Representation & Slicing**

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

#### **4.4. Instantiation & Constructors**

- `pub const fn zero() -> Self where T: Zero + Copy`: Instantiates an all-zero
  matrix using `T::ZERO` as the constant initialization value.
- `pub const fn identity() -> Self where T: Zero + One + Copy`: Instantiates an
  identity matrix (restricted to square shapes) by initializing elements to
  `T::ZERO` and filling the main diagonal with `T::ONE` via a const-evaluated
  loop.

- `pub const fn diagonal(val: [T; D::DIM]) -> Matrix<T, D, D>`:
  Constructs a diagonal matrix using the provided array of diagonal values and
  filling off-diagonal elements with `T::ZERO`.

- `pub fn from_fn<F>(mut f: F) -> Self where F: FnMut(usize, usize) -> T`:
  Generates a matrix using a coordinate-based mapping function at runtime.

*Implementation Note*: To support generic `const fn` initialization on stable
Rust, the scalar type `T` must implement the `Zero` and `One` traits from
`crate::math::num_traits`. These traits expose the associated constants
`T::ZERO` and `T::ONE`. All static constructors are marked `const fn` to allow
placing static matrices directly in read-only flash memory.

#### **4.5. Operator Overloading**

Overloads `Add`, `Sub`, and `Mul` from `core::ops`. Dimension rules are
statically enforced at compile-time. Under the hood, these high-level operator
implementations map directly to specific low-level BLAS subprograms to maximize
performance:

- **Matrix Addition (`Add`) & Subtraction (`Sub`)**: Evaluated element-wise.
  These operations map directly to the BLAS Level 1 **`AXPY`** subprogram (
  `y = a*x + y` trait defined
  in [subprograms.rs](../../src/math/subprograms.rs)), where addition uses
  `a = T::ONE` and subtraction uses `a = -T::ONE`.
- **Matrix Negation (`Neg`)**: Maps to BLAS Level 1 **`AXPY`** with
  `a = -T::ONE` and `y = 0`, or element-wise scaling.
- **Matrix-Matrix Multiplication (`Mul<Matrix>`)**: Statically enforces
  dimension matching (e.g. $(M \times N) \times (N \times P) \to (M \times P)$).
  It maps to the BLAS Level 3 **`GEMM`** subprogram (`C = alpha*A*B + beta*C`
  trait in [subprograms.rs](../../src/math/subprograms.rs)) with
  `alpha = T::ONE` and `beta = T::ZERO`.
- **Matrix-Vector Multiplication (`Mul<Vector>`)**: Maps to the BLAS Level 2 *
  *`GEMV`** subprogram (`y = alpha*A*x + beta*y` trait
  in [subprograms.rs](../../src/math/subprograms.rs)) with `alpha = T::ONE` and
  `beta = T::ZERO`.

```rust
impl<T, M: Dim, N: Dim, P: Dim> Mul<Matrix<T, N, P>> for Matrix<T, M, N>
where
    T: Copy + Default + Add<Output=T> + Mul<Output=T>,
{
    type Output = Matrix<T, M, P>;
    // ...
}
```

#### **4.6. Core Operations**

- `pub fn transpose(self) -> Matrix<T, C, R>`:
  Evaluates transposition. For memory layout compatibility, transposition swaps
  indices from column-major `(col, row)` to `(row, col)`.
- `pub fn invert(self) -> Result<Self, LinAlgError>`:
  Inverts a square matrix.
    - **Symmetric Matrices**: Uses **$LDL^T$ Decomposition** (factorizing the
      matrix into $L D L^T$ where $L$ is unit lower-triangular and $D$ is
      diagonal) followed by forward substitution and backward substitution to
      solve for the inverted columns.
    - **General Square Matrices**: Uses **LU Decomposition with Partial Pivoting
      ** ($P A = L U$, where $P$ is a permutation matrix, $L$ is unit
      lower-triangular, and $U$ is upper-triangular) followed by
      forward/backward substitution against the columns of the identity matrix.
- `pub fn determinant(&self) -> T`:
  Calculates the determinant.
    - **Symmetric Matrices**: Computed from the $LDL^T$ decomposition as the
      product of the diagonal elements of $D$ ($\det(A) = \prod D_{ii}$).
    - **General Square Matrices**: Computed from the LU decomposition
      as $(-1)^p \prod U_{ii}$, where $p$ is the number of row exchanges
      performed during pivoting.

#### **4.7. Interoperability & Conversions**

##### **4.7.1. Conversion to Polynomial**

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
- **Behavior**: Coefficients are computed using a division-free variant of the
  Faddeev-LeVerrier algorithm. This prevents division-by-zero errors and
  subnormal underflow conditions during trace calculations on ill-conditioned
  matrices.
- **Failure Condition**: Returns `ConversionError::DimensionMismatch` if the
  scalar type cannot perform division, if numerical overflow occurs, or if
  capacity is insufficient.

##### **4.7.2. Conversion to Tensor**

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
- **Failure Condition**: Returns `ConversionError::LayoutMismatch` if
  `Layout::RANK != 2` or if the layout's dimensions do not match $ R \times C $.

#### **4.8. Error Handling & State Management**

##### **4.8.1. Compile-Time Constraints**

Dimension mismatches (e.g., adding matrices of different sizes or multiplying
incompatible dimensions) fail at compile-time. Rust's type checker prevents
compiling invalid math.

##### **4.8.2. Runtime Error Taxonomy**

To supplement the crate's generic `ArithmeticError`, `control-rs` defines
dedicated error enums in the `math` module to represent linear algebra and
conversion failures:

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

##### **4.8.3. Runtime Fallbacks**

Dynamic operations that cannot be validated statically use soft failure paths:

- Matrix inversion returns a `Result<Self, LinAlgError>` instead of panicking,
  allowing control loops to handle singular conditions (e.g., falling back to a
  degraded state by returning `Err(LinAlgError::SingularMatrix)`).
- Boundary access returns `Option<&T>` via safe `get` methods.

#### **4.9. Structural Specializations & Extensions**

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

##### **4.9.1. Forward and Backward Substitution Examples**

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

##### **4.9.2. Companion Matrix Root-Finding**

For polynomial root-finding, the coefficients are mapped to a companion matrix
in upper Hessenberg form (strict zeros beneath the first lower subdiagonal).
Instead of using a general $O(N^3)$ QR algorithm, the solver exploits the
unitary-plus-rank-one structure. This reduces storage requirements to $O(N)$ and
computational complexity to $O(N^2)$ flops. Applying a sequence of planar
rotators guarantees normwise backward stability.

##### **4.9.3. Kalman Filter State Update Example**

The following example demonstrates the proposed `Matrix` API when computing the
covariance update in a Kalman filter loop:
$$ P_{k|k} = (I - K_k H_k) P_{k|k-1} $$

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

##### **4.10. Safety Architecture & Unsafe Boundaries**

Because `control-rs` targets high-integrity, safety-critical embedded
systems, memory safety and predictability are paramount. The `Matrix` type acts
as a secure boundary wrapping several underlying `unsafe` operations:

###### **4.10.1. Flat Slice Coercion (`as_slice` and `as_mut_slice`)**

To allow zero-copy interoperability with slice-based algorithms and external DSP
routines, `Matrix` exposes flat slice interfaces:

- **Wrapped Unsafe Functions**: `core::slice::from_raw_parts` and
  `core::slice::from_raw_parts_mut`.
- **Safety Preconditions & Invariants**:
    - **Memory Layout Contiguity**: The `Matrix` struct is annotated with
      `#[repr(C)]`. This guarantees that the nested array representation
      `[[T; R::DIM]; C::DIM]` is laid out contiguously in memory without any
      padding between columns or rows, matching a flat array of
      size $R \times C$.
    - **Lifetime Binding**: The returned slices `&[T]` and `&mut [T]` are bound
      directly to the lifetime of the borrow on the parent `Matrix` instance.
      This prevents use-after-free, dangling references, or slice invalidation.
    - **Bounds Verification**: The number of elements passed to `from_raw_parts`
      is exactly `R::DIM * C::DIM`. Since the dimensions `R` and `C` are
      statically typed constants representing the physical dimensions of the
      array, there is zero risk of buffer overflow or out-of-bounds pointer
      offsets during construction of the slice.

###### **4.10.2. Abstracting Target-Specific DSP / BLAS FFI**

When hardware acceleration (e.g., CMSIS-DSP, ARM NEON, or vendor-specific
DSPLib) is enabled, the underlying BLAS traits dispatch calls to FFI functions.

- **Wrapped Unsafe Functions**: External foreign function interfaces (FFI)
  accepting raw pointers.
- **Safety Preconditions & Invariants**:
    - C-based FFI routines do not perform bounds checking and assume that the
      caller has allocated sufficient, correctly-aligned memory.
    - The `Matrix` type acts as a guard by statically verifying all dimension
      constraints at compile time (using Peano types). It ensures that the
      buffers passed to FFI calls have the precise size expected by the hardware
      kernels, preventing memory corruption or CPU faults.

###### **4.10.3. Prevention of Unsafe Layout Transmutations**

Specialized structures like `Symmetric`, `UpperTriangular`, and
`LowerTriangular` wrap the standard `Matrix` type to dispatch optimized
routines (e.g., substitution solvers).

- **Wrapped Unsafe Functions**: Bypasses `core::mem::transmute` or unsafe
  pointer casts.
- **Safety Preconditions & Invariants**:
    - Instead of transmuting layout representation (which could violate memory
      safety or alignment constraints if representation changes), these
      specialized types are safe new-type wrappers (
      `pub struct Symmetric<T, D>(Matrix<T, D, D>)`).
    - Mathematical properties (such as symmetry or zero-elements) are enforced
      through safe API boundaries (e.g. only exposing getters/setters that
      preserve the structural invariants), ensuring compile-time safety without
      resorting to unsafe casts.

---

### **5. Alternatives**

#### **5.1. External Libraries (e.g., `nalgebra`)**

Using an external library like `nalgebra` in its static-storage, `no_std` mode
was considered. However, `control-rs` implements its own custom math module for
two key reasons:

1. **Generic `const fn` Support on Stable Rust**: Placing static matrices in
   read-only Flash memory requires `const fn` constructors. Traits like
   `Default` cannot be called inside `const fn` on stable Rust. The custom
   `Zero` and `One` traits in `crate::math::num_traits` expose associated
   constants (`T::ZERO`/`T::ONE`), allowing compile-time evaluation.
2. **Audit Footprint & Certification**: Standards such as ISO 26262 and DO-178C
   require auditing and certifying every line of dependency code. `nalgebra`
   contains a massive API surface and a deep dependency chain, significantly
   increasing the audit surface. The custom in-house module limits the audit
   surface to safety-critical invariants.

#### **5.2. Memory Layout Alternatives**

- **Row-Major Layout**: Row-major layouts provide spatial locality when
  accessing rows.
- **Panel-Major Layout (BLASFEO style)**: BLASFEO stores matrices in
  fixed-height panels with column-major layouts inside. This avoids data-packing
  overhead for cache-resident matrices but requires non-contiguous mapping and
  index arithmetic. It prevents exposing zero-copy flat slice APIs (`&[T]`)
  without allocation, which conflicts with safe API requirements.

#### **5.3. Factorization & Inversion Algorithms**

For solving linear systems and matrix inversion, the following factorization
algorithms were analyzed with their trade-offs for embedded deployment:

- **LU Factorization (with Partial Pivoting)**:
    - *Pros*: General-purpose; works on any non-singular square matrix. Pivoting
      prevents division by small values, preserving numeric stability.
    - *Cons*: Pivoting requires row-swapping logic, which complicates loop
      unrolling and SIMD optimization. It has a higher constant factor overhead
      than Cholesky/LDL^T ($O(2N^3/3)$ operations).
- **QR Factorization (via Givens Rotations or Householder Reflections)**:
    - *Pros*: Extremely stable numerically, even for poorly conditioned or
      singular-prone systems.
    - *Cons*: Highly computationally expensive ($O(4N^3/3)$ operations). Givens
      rotations require many square root and trigonometric function calls,
      making it slow on microcontrollers lacking hardware FPU support.
- **Cholesky Factorization ($LL^T$)**:
    - *Pros*: Highly efficient ($O(N^3/3)$ operations, half the operations of
      LU) and exhibits excellent numerical stability for positive-definite
      symmetric matrices.
    - *Cons*: Restricted strictly to symmetric positive-definite matrices.
      Requires calculating square roots for each diagonal element, which
      typically takes many CPU cycles and increases quantization errors in
      fixed-point representations.
- **$LDL^T$ Factorization**:
    - *Pros*: Chosen as the default solver for symmetric matrices. Like
      Cholesky, it requires only $O(N^3/3)$ operations. By decomposing the
      matrix into $L D L^T$ (where $L$ is unit lower-triangular and $D$ is
      diagonal), it completely avoids square root calculations. This preserves
      scaling boundaries in fixed-point formats and optimizes CPU cycle counts.
    - *Cons*: Restricted to symmetric matrices. If the matrix is near-singular
      or indefinite, it may suffer from numerical instability without complex
      block-pivoting algorithms (e.g., Bunch-Kaufman).
- **Normal Equation Solving (Forming $A^T A$)**:
    - *Pros*: Allows solving non-symmetric or rectangular systems ($A x = b$) by
      converting them to a symmetric system ($A^T A x = A^T b$) and applying
      efficient symmetric solvers (Cholesky/LDL^T).
    - *Cons*: Strongly avoided. Forming $A^T A$ squares the condition number of
      the matrix ($\kappa(A^T A) = \kappa(A)^2$), which halves the number of
      valid decimal digits in calculations and leads to severe precision loss.

#### **5.4. Matrix Multiplication Algorithms**

To evaluate $C = A B$, several multiplication approaches were compared:

- **Naive Row-by-Column (Triple Loop, $O(N^3)$)**:
    - *Pros*: Tiny code footprint, no temporary buffer requirements,
      and trivial for the compiler to optimize or auto-vectorize for very small
      dimension limits ($N \le 8$).
    - *Cons*: For larger dimensions (e.g., $N = 32$), this approach suffers from
      high L1 cache miss rates due to non-contiguous memory access in
      column-major matrices.
- **Block-Based (Tiled) Multiplication**:
    - *Pros*: Restructures the triple loop into sub-matrix
      blocks ($k_c \times n_R$) to fit inside the CPU's cache line size,
      drastically reducing memory bus transactions for larger
      matrices ($N > 32$).
    - *Cons*: Adds complex index boundary math and loop nesting, which increases
      target binary size and introduces instruction overhead that outweighs
      cache benefits for small embedded matrices ($N \le 32$).
- **Vectorized SIMD / Hardware BLAS FFI**:
    - *Pros*: Directly utilizes SIMD registers (such as ARM NEON or CMSIS-DSP
      assembly instructions) to perform multiple multiply-accumulate operations
      per cycle.
    - *Cons*: Bypasses safe Rust controls by passing raw pointers to FFI
      functions. It is highly hardware-specific and requires fallback
      implementations for targets lacking SIMD engines.

#### **5.5. Determinant Calculation Algorithms**

For computing $\det(A)$, two primary methods were analyzed:

- **Leibniz Formula / Cofactor Expansion**:
    - *Pros*: Does not require factorization or modifications to the matrix
      data. Highly efficient and division-free for tiny dimensions ($2 \times 2$
      or $3 \times 3$).
    - *Cons*: Factorial complexity ($O(N!)$). Computing the determinant of
      a $32 \times 32$ matrix using cofactor expansion is mathematically
      impossible in real-time.
- **Factorization-Based**:
    - *Pros*: Uses the LU or $LDL^T$ decomposition result. Since the determinant
      of a triangular matrix is the product of its diagonal elements, $\det(A)$
      is computed in $O(N)$ additional operations after factorization.
      Numerically stable and scales to $N=32$.
    - *Cons*: Requires running a full matrix factorization first, which is
      fallible (e.g., singular matrices return zero determinant or error).

---

### **6. Verification & Validation**

For a standard matrix multiplication where $C = A \times B$ with
dimensions $(n \times m)$ and $(m \times k)$, the total execution time $T$ in
seconds can be modeled mathematically
as: $$T \approx \frac{(n \cdot m \cdot k \cdot c_{\text{inner}}) + c_{\text
{overhead}}}{f}$$

**Variable Breakdown**:

- $n, m, k$: The matrix dimensions. The core mathematical operation (a
  Multiply-Accumulate, or MAC) is executed exactly $n \cdot m \cdot k$
  times.
- $f$: The processor clock frequency in Hertz (Hz).
- $c_{\text{inner}}$: The number of clock cycles required to execute
  one iteration of the innermost loop. This includes loading two f32
  values, performing the multiplication and addition, and incrementing
  pointers.
- On an ARM Cortex-M processor with a hardware Floating Point Unit (
  FPU), a highly optimized inner loop typically costs 4 to 8
  cycles.
- If the target architecture lacks a hardware FPU (relying on
  software floating-point emulation), this cost jumps drastically to
  50 to 150 cycles.
- $c_{\text{overhead}}$: The fixed cycle cost of function calls, stack
  setup, and outer loop branching. For larger matrices, this is
  negligible, but for highly constrained
  matrices (e.g., $3 \times 3$), this overhead can represent a
  measurable percentage of the total execution time.

#### **6.1. Verification**

1. **Unit Testing**: Run unit tests on the host system via `cargo test` to
   verify constructors, operators, boundaries, and triangular solvers.
2. **Property-Based Testing**: Use the `proptest` framework to verify
   mathematical identities (e.g., $(AB)^T = B^T A^T$ and
   distributivity $A(B + C) = AB + AC$) across thousands of randomized matrices.
   Deliberately populate the `proptest` corpus with ill-conditioned,
   near-singular, and Hilbert matrices to verify that singularity checks catch
   numerical instability without crashing.
3. **HIL Testing**: Compile and run target-specific test suites on real
   hardware.
4. **Cache Miss Profiling**: Run Cachegrind to monitor L1 instruction/data cache
   misses (`I1mr`/`D1mr`) and last-level cache misses (`LLd`), optimizing tile
   sizes ($k_c \times n_R$) to fit cache limits.
5. **Continuous Integration**: Execute clippy and formatting checks
   automatically in the CI pipeline.

#### **6.2. Validation**

1. **Kalman Filter Covariance Update**: Implement the covariance
   update ($P_{k|k} = (I - K_k H_k) P_{k|k-1}$) as a direct validation scenario.
2. **External Integration**: Validate layout compatibility by passing slice
   views directly to CMSIS-DSP or vendor-specific BLAS libraries.
3. **User Demos**: Provide step-response simulations and closed-loop control
   system examples in the `examples/` directory.

---

### **7. Performance & Resource Considerations**

- **Stack Overflow Prevention**: Limit maximum matrix dimensions
  to $32 \times 32$ elements to ensure a single matrix instance never exceeds
  4KB of stack space (when using 32-bit floats).
- **FPU Register Pressure & Spills**: Modern FPUs (like Cortex-M4/M7) have
  exactly 32 single-precision registers. Code must keep active variables within
  this limit to prevent compiler register spills to RAM.
- **Pipeline Dependencies**: Restructure loops to prevent pipeline stalls from
  data dependencies (e.g., waiting 3-4 cycles for write-back in
  destination-dependent accumulation like `a += b * c`).
- **Compiler Auto-Vectorization & fast-math**:
    - LLVM loop vectorizer fails on loop-carried dependencies where
      iteration $n$ depends on $n-1$.
    - Floating-point non-associativity forbids loop reordering under strict IEEE
      754 compliance. Flag `-ffast-math` overrides this but introduces numerical
      drift.
    - Memory alignment: Ensure arrays are aligned (e.g., `#[repr(align(8))]` or
      `#[repr(align(16))]`) to match vector registers, preventing slow scalar
      fallbacks.
- **Fixed-Point Scaling & Guard Bits**: Q31 multiplication produces Q62
  products. Implement proper bit-shifting and scaling (e.g., guard bits, dynamic
  scaling) to prevent intermediate overflows while maintaining precision.
- **DMA Offloading & Double Buffering**: Offload heavy data movement (e.g., ADC
  sample streams) to DMA. Use true hardware double-buffering (swapping pointers
  via STM32's `M0AR`/`M1AR` registers) instead of simple circular buffering to
  prevent memory tearing and overwriting by the DMA controller when the CPU is
  preempted.

---

### **8. Risks & Open Questions**

- **Const Generics Complexity**: Stabilized const generics are still limited.
  Custom trait bounds (like `DimAdd`, `DimMul`) might increase compile times and
  create verbose error messages.
- **Precision vs. Performance Trade-off**: Deciding whether to utilize
  `-ffast-math` or rely on strict IEEE 754 compliance for float math.
- **Fixed-Point Precision Loss**: Truncation errors in Q31/Q15 accumulator
  scaling might lead to drift in high-frequency loops.

---

### **9. Development Plan**

| Task / Feature               | Description                                                                             | Estimated Effort |
|:-----------------------------|:----------------------------------------------------------------------------------------|:-----------------|
| **Step 1: Core Layout**      | Define `Matrix` struct, column-major storage, and slice casting.                        | 1.0 Day          |
| **Step 2: Operators**        | Implement `Add`, `Sub`, `Mul` traits with compile-time checks.                          | 1.5 Days         |
| **Step 3: Solvers**          | Implement $LDL^T$ decomposition, LU, determinants, and matrix inversion.                | 2.0 Days         |
| **Step 4: Specializations**  | Create `UpperTriangular`, `LowerTriangular`, and `Symmetric` wrappers.                  | 1.0 Day          |
| **Step 5: Factorizations**   | Implement Cholesky and QR solvers.                                                      | 2.0 Days         |
| **Step 6: Verification**     | Set up `proptest` suites, ARM DWT cycle profiling, and Cachegrind setups.               | 1.5 Days         |
| **Step 7: Interoperability** | Implement conversions between `Matrix`, `Polynomial` (Faddeev-LeVerrier), and `Tensor`. | 2.0 Days         |

---

### **10. Revision History**

| Revision | Date          | Author          | Description of Changes                                                         |
|:---------|:--------------|:----------------|:-------------------------------------------------------------------------------|
| 1.0      | July 12, 2026 | @MitchellDScott | Initial draft outlining core concepts, layout, and operations.                 |
| 2.0      | July 19, 2026 | @MitchellDScott | Restructured to new template; added embedded performance/verification details. |
