# Matrix Type & Structural Specializations (Design Document)

## 1. Context & Objective

The `Matrix` type in `control-rs` provides a high-performance, stack-allocated, compile-time sized representation of mathematical matrices for control systems algorithms (such as state-space propagation, Kalman filtering, and LQR). 

In safety-critical control environments, dynamic heap allocation is prohibited to ensure deterministic execution times and prevent out-of-memory panics. Therefore, `Matrix` must be statically sized and reside entirely on the stack.

---

## 2. Architecture & Design Decisions

### 2.1. Column-Major Memory Layout (`[[T; R]; C]`)
The underlying data structure of a matrix of dimension $R \times C$ is a nested array of shape `[[T; R]; C]`. This represents $C$ columns, where each column contains $R$ row elements.

#### Rationale:
1. **BLAS and LAPACK Compatibility**: Standard linear algebra packages (traditionally written in Fortran) use column-major ordering. By mirroring this layout, we align our memory representation directly with high-performance BLAS/LAPACK subprograms, facilitating easy integration and direct mapping.
2. **CPU Cache Locality for Control Algorithms**: In control systems, matrix-vector products $A x$ are the most common operations (e.g., state updates $x_{k+1} = A x_k + B u_k$). A column-major layout allows us to compute these products as linear combinations of columns, which accesses memory contiguously down each column. This maximizes cache hit rates on modern CPUs and microcontrollers.
3. **Zero-Cost Representation**: Nested arrays `[[T; R]; C]` directly express column contiguity. The compiler can compute element offsets (`col * R + row`) at compile-time as constant expressions, completely eliminating runtime calculation overhead.

---

### 2.2. Standardizing Storage & Removing Complexity
A previous design iteration attempted to use traits (e.g., `MatrixStorage`, `StorageShape`) to abstract over stack storage, heap storage, and structural variants. This was rejected.

#### Rationale:
- **Type Complexity**: The trait-based design introduced a massive number of generic parameters and complex bounds (e.g., `Matrix<T, R, C, S>` where `S: Storage`).
- **Compile-Time Overhead**: Abstracting storage shapes led to highly complex trait resolution paths, significantly increasing compilation times.
- **Diagnostics**: Compiler errors for dimension mismatches or missing trait implementations became extremely difficult to diagnose.
- **Decision**: Standardize on a single, concrete `Matrix<T, const R: usize, const C: usize>` structure that wraps `[[T; R]; C]`.

---

### 2.3. Flat Slice Interoperability (`as_slice` and `as_mut_slice`)
To interface with low-level arithmetic kernels, the matrix exposes:
```rust
pub const fn as_slice(&self) -> &[T];
pub const fn as_mut_slice(&mut self) -> &mut [T];
```

#### Rationale:
- **Contiguity Guarantee**: While `[[T; R]; C]` is logically nested, Rust guarantees that arrays are laid out contiguously in memory. Thus, casting `[[T; R]; C]` to a flat slice of length `R * C` is statically safe and does not copy data.
- **Low-Level Kernels**: This design allows `control-rs` to leverage raw pointer-based mathematical kernels (such as BLAS level 1, 2, and 3 subprograms) by simply passing the flat slice and its dimensions.

---

### 2.4. Hardware-Accelerated Math via BLAS Subprograms
Instead of implementing matrix addition, scaling, and multiplication using manual loops, `control-rs` uses standard BLAS-like subprograms (e.g., `axpy`, `scal`, `gemm`).

#### Rationale:
- **Loop Vectorization**: Low-level subprograms are structured to allow target-specific SIMD auto-vectorization by the compiler.
- **Embedded/SIMD Acceleration**: Flat slices can be routed directly to hardware-accelerated BLAS implementations (such as ARM CMSIS-DSP or Intel MKL) on supporting architectures.
- **Maintainability**: Centralizing arithmetic in optimized subprograms ensures that performance improvements benefit all structures (matrices, vectors, tensors, polynomials) uniformly.

---

## 3. Structural Specializations (Triangular & Symmetric)

Structural specializations enforce specific mathematical invariants at compile-time. They are represented as distinct types wrapping a square `Matrix<T, D, D>`:
```rust
pub struct UpperTriangular<T, const D: usize> { pub(crate) matrix: Matrix<T, D, D> }
pub struct LowerTriangular<T, const D: usize> { pub(crate) matrix: Matrix<T, D, D> }
pub struct Symmetric<T, const D: usize>       { pub(crate) matrix: Matrix<T, D, D> }
```

### Rationale:
1. **Reuse of Memory & Layout**: Wrapping `Matrix<T, D, D>` means we reuse all memory layout optimizations (contiguous column-major slices).
2. **Simplified Operator Signatures**: By using distinct types rather than storage tags, writing specialized algorithms is trivial. For instance, multiplying a lower triangular matrix by a vector can be dispatched to a specialized solver (e.g., forward substitution) at compile-time based on the type.
3. **Square Dimension Guarantees**: By wrapping `Matrix<T, D, D>`, the dimension is controlled by a single const generic `D`. This guarantees the matrix is square at compile-time, eliminating the need for runtime validation checks.

### Trade-offs:
- **Memory Overhead**: An $D \times D$ triangular matrix contains redundant/unused values (elements below/above the diagonal). Storing these wastes approximately half of the memory compared to a packed triangular format.
- **Decision**: We choose to trade memory for simplicity and performance. Packing triangular matrices requires complex index mapping equations (e.g., $i \cdot (i+1)/2 + j$) at runtime, which prevents contiguous slicing and degrades cache performance during BLAS operations.

---

## 4. Invariant Enforcement & Safety

To guarantee the mathematical correctness of specializations, mutation is carefully constrained:

### 4.1. Upper & Lower Triangular
- **Read-Access**: Normal read-access via `.get(r, c)` reads the underlying matrix.
- **Mutation-Access**: `.get_mut(r, c)` returns `None` if the indices lie outside the active triangle (strictly below for `UpperTriangular`, strictly above for `LowerTriangular`). This prevents safe code from corrupting the zero-invariants of the specialization.

### 4.2. Symmetric Matrix
- **Write-Access**: Symmetric matrices do not expose `.get_mut(r, c)` directly, as editing a single index would break the symmetry invariant ($A_{i, j} = A_{j, i}$). Instead, they expose a `.set(row, col, value)` method that updates both mirror elements in the underlying array.