# Matrix Type & Structural Specializations (Design Document)

![Date Badge](https://img.shields.io/badge/Date-July_09,_2026-blue)
![Status Badge](https://img.shields.io/badge/Doc%20Status-WIP-orange)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

## 1. Context & Objective

The `Matrix` type in `control-rs` provides a stack allocated, fixed size
2-Dimensional storage for numerical types tailored for `#[no-std]` environments.

---

## 2. Architecture & Design Decisions

The `Matrix` type relies on generics to provide compile time guarantees
and specify subprogram implementations. The heavy use of generics enables
creating specializations of the matrix without creating a new type.

### 2.1. Column-Major Memory Layout (`[[T; R]; C]`)

The underlying data structure of a matrix of dimension $R \times C$ is a nested
array of shape `[[T; R]; C]`. This represents $C$ columns, where each column
contains $R$ row elements.

1. **BLAS and LAPACK Compatibility**: Standard linear algebra packages (
   traditionally written in Fortran) use column-major ordering.
2. **CPU Cache Locality for Control Algorithms**: In control systems,
   matrix-vector products $A x$ is a very common operations (e.g., state
   updates $x_{k+1} = A x_k + B u_k$).
3. **Zero-Cost Representation**: Nested arrays `[[T; R]; C]` directly express
   column contiguity. The compiler can compute element offsets (`col * R + row`)
   at compile-time as constant expressions.

---

### 2.2. Flat Slice Interoperability (`as_slice` and `as_mut_slice`)

To interface with low-level arithmetic kernels, the matrix exposes:

```rust
impl<T, const R: usize, const C: usize> Matrix<T, R, C> {
    pub const fn as_slice(&self) -> &[T];
    pub const fn as_mut_slice(&mut self) -> &mut [T];
}
```

- **Contiguity Guarantee**: While `[[T; R]; C]` is logically nested, Rust
  guarantees that arrays are laid out contiguously in memory.
  [Type Layout docs](https://doc.rust-lang.org/reference/type-layout.html)
- **Low-Level Kernels**: This design allows `control-rs` to leverage raw
  pointer-based mathematical kernels (such as BLAS level 1, 2, and 3
  subprograms) by simply passing the flat slice and its dimensions.

---

### 2.3. Hardware-Accelerated Math via BLAS Subprograms

Instead of implementing matrix addition, scaling, and multiplication using
manual loops, `control-rs` uses standard BLAS-like subprograms (e.g., `axpy`,
`scal`, `gemm`).

- **Loop Vectorization**: Low-level subprograms are structured to allow
  target-specific SIMD auto-vectorization by the compiler.
- **Embedded/SIMD Acceleration**: Flat slices can be routed directly to
  hardware-accelerated BLAS implementations (such as ARM CMSIS-DSP or Intel MKL)
  on supporting architectures.
- **Maintainability**: Centralizing arithmetic in optimized subprograms ensures
  that performance improvements benefit all structures (matrices, vectors,
  tensors, polynomials) uniformly.

---

## 3. Structural Specializations (Triangular & Symmetric)

Structural specializations enforce specific mathematical invariants at
compile-time. They are represented as distinct types wrapping a square
`Matrix<T, D, D>`:

```rust
pub type UpperTriangular<T, const D: usize> = Matrix<T, D, D>;
pub type LowerTriangular<T, const D: usize> = Matrix<T, D, D>;
pub type Symmetric<T, const D: usize> = Matrix<T, D, D>;
```

1. **Reuse of Memory & Layout**: Wrapping `Matrix<T, D, D>` means we reuse all
   memory layout optimizations (contiguous column-major slices).
2. **Simplified Operator Signatures**: By using distinct types rather than
   storage tags, writing specialized algorithms is trivial.
3. **Square Dimension Guarantees**: By wrapping `Matrix<T, D, D>`, the dimension
   is controlled by a single const generic `D`. This guarantees the matrix is
   square at compile-time, eliminating the need for runtime validation checks.

### Trade-offs:

- **Memory Overhead**: An $D \times D$ triangular matrix contains
  redundant/unused values (elements below/above the diagonal). Storing these
  wastes approximately half of the memory compared to a packed triangular
  format.
- **Decision**: We choose to trade memory for simplicity and performance.
  Packing triangular matrices requires complex index mapping equations (
  e.g., $i \cdot (i+1)/2 + j$) at runtime, which prevents contiguous slicing and
  degrades cache performance during BLAS operations.

---

## 4. Embedded Error Handling Policy

To fit safety-critical runtime paradigms, `control-rs` completely bans
unexpected out-of-bounds `panic!` actions within standard math routines.

* **Index Operations**: Shuns panics by relying on predictable, soft failure
  paths (`Option<&T>` or `Option<&mut T>`).
* **Size Mismatches**: Because dimension mismatches of stack-allocated
  matrices are entirely checked at compile time via mismatched Const Generic
  dimensions no runtime panic handling code is generated or compiled into
  the final binary.