# Matrix Type & Structural Specializations (Design Document)

![Date Badge](https://img.shields.io/badge/Date-July_11,_2026-blue)
![Status Badge](https://img.shields.io/badge/Doc%20Status-Draft-orange)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

---

### **1. Introduction**

The `Matrix` type in `control-rs` provides a high-performance, statically
verified 2-dimensional storage representation for numerical types. To support
bare-metal targets, the library is strictly constrained to `#![no_std]`
environments, relying entirely on stack memory allocations and compile-time
guarantees.

---

### **2. Motivation and Target Constraints**

#### **2.1 Environmental Limitations**

Bare-metal microcontrollers and real-time operating systems (RTOS) are
restricted by small memory spaces and lack a standard dynamic memory allocator.

#### **2.2 Target Applications**

The library targets resource-constrained, high-reliability control environments:

* **High-frequency control loops:** State-space control, Kalman filtering, PID
  controllers, and Model Predictive Control (MPC) running at sub-millisecond
  rates.
* **GNC Algorithms:** Guidance, Navigation, and Control algorithms for aerospace
  and robotic vehicles.
* **Hardware-in-the-Loop (HIL):** Deterministic real-time simulations running on
  embedded target nodes.

#### **2.3 Safety and Verification**

Traditional numerical libraries validate matrix dimensions at runtime, raising
exceptions or panicking when a mismatch occurs (e.g., multiplying a $3 \times 2$
matrix by a $4 \times 3$ matrix). In safety-critical systems, runtime panics are
unacceptable. By leveraging Rust's type system and const generics, dimension
validation is shifted entirely to compilation, ensuring dimension-mismatch
errors halt compilation.

---

### **3. Core Architecture and Memory Layout**

#### **3.1 Generics Foundation**

To encode dimensions directly into type signatures, the design relies on Rust's
const generics. The primary type definition is:

```rust
pub struct Matrix<T, R: Dim, C: Dim> {
    data: [[T; R::DIM]; C::DIM],
}
```

For advanced dimension validation requiring compile-time mathematical
relationships (like matrix dimensions resulting from type-level addition or
multiplication), the architecture utilizes the `Dim` trait and Peano number
representations defined
in [num_types.rs](../../src/math/num_types.rs).

#### **3.2 Internal Storage Strategy**

**Column-major arrays** (`[[T; R]; C]`).

* **BLAS and LAPACK Compatibility:** Column-major ordering is standard in
  legacy, high-performance linear algebra packages.
* **Cache Locality:** In control loops, matrix-vector products $A x$ are
  computed frequently (e.g., state updates $x_{k+1} = A x_k + B u_k$).
  Column-major ordering enables contiguous cache-friendly access when performing
  linear combinations of columns.
* **Zero-Cost Representation:** Nested arrays directly express column
  contiguity. The compiler resolves offsets at compile time as constant
  expressions.
* **Flat Slice Interoperability:** To support low-level arithmetic kernels, the
  struct exposes:
  ```rust
  impl<T, R: Dim, C: Dim> Matrix<T, R, C> {
      pub const fn as_slice(&self) -> &[T] { /* ... */ }
      pub const fn as_mut_slice(&mut self) -> &mut [T] { /* ... */ }
  }
  ```
  Rust guarantees that nested arrays are laid out contiguously in memory,
  enabling zero-copy pointer casting to flat slices.

---

### **4. API Specification**

#### **4.1 Instantiation**

The API provides constructors for static and runtime initialization:

* **Zero Matrix:** `pub const fn zero() -> Self` (if `T` supports a
  zero-representation).
* **Identity Matrix:** `pub const fn identity() -> Self` (for square matrices).
* **Diagonal Matrix:**
  `pub const fn diagonal(val: [T; D::DIM]) -> Matrix<T, D, D>`.
* **Functional Generation:**
  `pub fn from_fn<F>(mut f: F) -> Self where F: FnMut(usize, usize) -> T`.
* **Compile-Time Safety:** All constructors are marked `const fn` where possible
  to allow instantiating static matrix declarations at compile-time directly in
  read-only memory.

#### **4.2 Operator Overloading**

Arithmetic operators are overloaded by implementing `core::ops` traits:

* **Addition and Subtraction:** `Add` and `Sub` are implemented for matrices of
  matching dimensions.
* **Matrix Multiplication:** Overloaded using `Mul`. The type signature enforces
  dimensions strictly:
  ```rust
  impl<T, M: Dim, N: Dim, P: Dim> Mul<Matrix<T, N, P>> for 
  Matrix<T, M, N>
  where
      T: Copy + Default + core::ops::Add<Output = T> + core::ops::Mul<Output = T>,
  {
      type Output = Matrix<T, M, P>;
      // ...
  }
  ```
  This signature guarantees that multiplying an $M \times N$ matrix by
  an $N \times P$ matrix results in an $M \times P$ matrix at compile time.

#### **4.3 Core Linear Algebra Operations**

* **Transposition:** Returns a transposed matrix type-safely.
  ```rust
  impl<T, R: Dim, C: Dim> Matrix<T, R, C> {
    pub fn transpose(self) -> Matrix<T, C, R> { /* ... */ }
  }
  ```
* **Determinant & Inversion:** Inversion returns a
  `Result<Self, SingularMatrixError>` to handle singular matrices.
  ```rust
  impl<T, R: Dim, C: Dim> Matrix<T, R, C> {
    pub fn determinant(&self) -> T { /* ... */ }
    pub fn invert(self) -> Result<Self, SingularMatrixError> { /* ... */ }
  }
  ```
* **Vector Operations:** Dot and cross products are implemented specifically for
  vector dimensions ($N \times 1$ and $1 \times N$ shapes).

---

### **5. Error Handling & State Management**

#### **5.1 Compile-Time Validation**

Because dimension incompatibilities are verified during compilation, the
generated binary does not contain size-checking branches or panic code for
arithmetic operations.

#### **5.2 Runtime Fallbacks**

Mathematical edge cases (e.g., inversion of a singular matrix or out-of-bounds
indexing) bypass panics by using soft failure paths:

* Out-of-bounds indexing returns `Option<&T>` or `Option<&mut T>` via get
  methods.
* Matrix inversion returns `Result<Matrix<T, D, D>, SingularMatrixError>`,
  allowing the control loop to fall back to a safe state or raise a soft warning
  rather than halting execution.

---

### **6. Testing and Validation Framework**

#### **6.1 Unit Testing and std/no_std Integration**

Host-side validation utilizing the standard test harness (`std::test`) is
acceptable for verifying complex mathematical invariants (and is necessary for
code coverage tools). Target-specific correctness is validated in a `#![no_std]`
environment using a custom lightweight testing runner run under QEMU or target
hardware to ensure target compatibility.

#### **6.2 Property-Based Testing**

Integration with host-side tools like `proptest` automatically generates bounded
matrices to verify algebraic properties, such as:

* $(AB)^T = B^T A^T$ (Transpose of product)
* $A(B + C) = AB + AC$ (Distributivity)
* $(AB)C = A(BC)$ (Associativity)

#### **6.3 Benchmark and Regression Tracking**

Automated micro-benchmarks track cycle counts and stack-depth usage on target
ISAs (e.g., ARM Cortex-M4, RISC-V) using target hardware cycles counter (e.g.,
DWT cycle counter on Cortex-M).

#### **6.4 Examples and Quality Reporting**

The numerical methods examples will allow bloat testing and final binary
size checking.

```rust
/// Solves the normal equations: A^T * A * x = A^T * b
pub fn solve_least_squares<T, M: Dim, N: Dim>(
    a: &Matrix<T, M, N>,
    b: &Matrix<T, M, U1>,
) -> Result<Matrix<T, N, U1>, LinAlgError>
where
    T: Copy + Default + core::ops::Add<Output=T> + core::ops::Sub<Output=T> + core::ops::Mul<Output=T> + core::ops::Div<Output=T> + PartialOrd,
    M: Dim,
    N: Dim,
    N: DimMul<N>,
    N: DimMul<U1>,
    M: DimMul<N>,
    M: DimMul<U1>,
{
    let a_t = a.transpose();
    let a_t_a = &a_t * a; // Resulting in N x N
    let a_t_b = &a_t * b; // Resulting in N x 1

    // Invert the square normal matrix A^T * A
    let a_t_a_inv = a_t_a.invert()?;

    // Solve for x: x = (A^T * A)^-1 * A^T * b
    Ok(&a_t_a_inv * &a_t_b)
}
```

---

### **7. Performance and Resource Considerations**

#### **7.1 Stack Overflow Prevention**

Since matrices are stack-allocated, large dimensions risk stack overflows. The
design mitigates this by restricting maximum dimensions. Under the current
`PeanoTypeNum` implementation, the type aliases
in [num_types.rs](../../src/math/num_types.rs)
restrict dimensions to a maximum of 32 (`U32`), preventing accidental allocation
of oversized matrices on the stack.

#### **7.2 Compiler Optimizations**

Linear algebra arithmetic relies on optimized subprograms in
the [subprograms.rs](../../src/math/subprograms.rs)
module:

* **BLAS-like Kernels:** Arithmetic maps to `axpy`, `scal`, and `gemm`
  functions.
* **SIMD Auto-Vectorization:** Kernels are structured with pointer iterations
  and alignment hints to allow LLVM to unroll loops and apply vectorization.
* **External BLAS Routing:** Flat slices can be routed directly to
  hardware-accelerated vendor libraries (e.g., ARM CMSIS-DSP) on supporting
  architectures.

---

### **8. Future Extensions**

#### **8.1 Specializations**

Structural specializations enforce specific mathematical structures and
invariants at compile-time. Rather than using bare type aliases (which cannot
enforce invariants or support distinct method implementations), the library
defines new-type wrapper structs:

```rust
pub struct UpperTriangular<T, D: Dim>(Matrix<T, D, D>);
pub struct LowerTriangular<T, D: Dim>(Matrix<T, D, D>);
pub struct Symmetric<T, D: Dim>(Matrix<T, D, D>);
```

* **Safety and Performance Dispatch:** Defining these as distinct types allows
  enforcing
  structural properties (such as symmetry or triangularity) at creation time via
  checked
  constructors. This type safety enables dispatching optimized algorithms (like
  Cholesky
  factorization only on a `Symmetric` matrix, or forward/backward substitution
  on triangular
  matrices) without checking invariants at runtime.
* **Memory vs. Performance Trade-off:** Packing triangular matrices requires
  complex index mapping equations (e.g., $i \cdot (i+1)/2 + j$) at runtime,
  which prevents contiguous slicing and degrades cache performance. Wrapping a
  full square matrix trades memory for simplicity and cache performance,
  enabling direct BLAS calls.

#### **8.2 Advanced Factorizations**

Implement stack-allocated decompositions tailored for control systems:

* **LU Decomposition** with partial pivoting.
* **Cholesky Decomposition** ($L L^T$) for covariance matrices (specifically
  operating on the `Symmetric` new-type wrapper).
* **QR Decomposition** using Householder reflections for stable least-squares
  solvers.

---

### **8.3 Development Plan & Roadmap**

| Task / Feature                       | Description                                                                                     | Estimated Effort |
|:-------------------------------------|:------------------------------------------------------------------------------------------------|:-----------------|
| **Phase 1: Core Struct & Layout**    | Define `Matrix` struct, column-major storage, flat slice conversion, and constructors.          | 1.0 Day          |
| **Phase 2: Basic Operations**        | Implement addition, subtraction, transposition, and matrix multiplication with dimension check. | 1.5 Days         |
| **Phase 3: Linear Solvers**          | Implement determinant, inversion, LU decomposition, and least squares solver.                   | 2.0 Days         |
| **Phase 4: Specializations**         | Implement `UpperTriangular`, `LowerTriangular`, and `Symmetric` wrappers.                       | 1.0 Day          |
| **Phase 5: Advanced Factorizations** | Implement Cholesky ($L L^T$) and QR decompositions.                                             | 2.0 Days         |
| **Phase 6: Testing & Benchmarks**    | Implement target hardware cycle benchmarks and `proptest` suites.                               | 1.5 Days         |

*Status: Currently in design phase. Core definitions and type-level signatures
are being aligned with [num_types.rs](../../src/math/num_types.rs).*