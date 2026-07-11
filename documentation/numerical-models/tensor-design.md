# Tensor Type (Design Document)

## 1. Context & Objective

Multi-dimensional arrays (tensors) are widely used in advanced control
algorithms, such as multi-variable non-linear systems, state-space
representations with spatial dimensions, and multi-channel signal processing.

The `Tensor` implementation in `control-rs` provides a stack-allocated,
compile-time shaped, multi-dimensional array. It is designed to extend linear
algebra operations beyond 2D matrices while enforcing strict memory constraints,
zero-cost abstractions, and safety-critical execution.

---

## 2. Architecture & Design Decisions

### 2.1. Flat Array Storage (`[T; N]`) & `TensorLayout`

Rather than representing an $n$-dimensional tensor using nested arrays (e.g.,
`[[[T; D1]; D2]; D3]`), `Tensor` stores elements in a single flat array
`[T; N]`. The coordinate shapes are governed by the `TensorLayout` trait:

```rust
pub trait TensorLayout {
    const RANK: usize;
    const SIZE: usize;
    type Size: Dim;
    fn dims() -> &'static [usize];
}
```

1. **Simplified Generics**: Representing nested arrays generically in Rust leads
   to type signatures that are nearly impossible to write or constrain. Using a
   flat array `[T; N]` with a separate `Dims` marker type keeps the struct
   definition clean:
   ```rust
   pub struct Tensor<T, const N: usize, Dims: TensorLayout> { ... }
   ```
2. **Flexible Rank Implementations**: We implement `TensorLayout` using macros
   for tuples of compile-time dimension sizes (e.g., `Const<D1>`,
   `(Const<D1>, Const<D2>)`, etc.). This allows the system to easily support 1D,
   2D, 3D, and 4D tensors out-of-the-box.

---

### 2.2. Compile-Time Size Verification via Peano Arithmetic

Because Rust's const generics cannot natively verify
that $N = D_1 \cdot D_2 \cdot \dots \cdot D_k$ at compile-time within struct
definitions, we use Peano type-level arithmetic.

#### Rationale:

  ```rust
  impl<T, const N: usize, Dims> Tensor<T, N, Dims>
where
    Const<N>: Dim,
    Dims: TensorLayout<Size=<Const<N> as Dim>::PeanoTypeNum>
  ```

This trait bound forces the compiler to resolve the product of dimensions at
compile-time and match it against the flat array size $N$. If the dimensions
do not match, the code fails to compile. This ensures absolute compile-time
shape safety without needing runtime assertions or checks.

---

### 2.3. Column-Major Multi-Index Mapping

Mapping multi-dimensional coordinates (e.g. $[i_0, i_1, \dots, i_{k-1}]$) to a
flat index is done using column-major strides (first dimension varies fastest):
$$\text{flat\_index} = i_0 + i_1 \cdot S_1 + i_2 \cdot S_2 + \dots$$
where the strides are $S_0 = 1$ and $S_m = \prod_{j=0}^{m-1} D_j$.

#### Rationale:

- **Consistency**: Column-major mapping aligns directly with the `Matrix`
  implementation ($2$D specialization). This ensures that matrix operations and
  tensor contractions share the same index-resolution math, allowing algorithms
  to transition seamlessly between matrices and tensors.

---

### 2.4. Flattened BLAS Arithmetic Acceleration

Element-wise arithmetic operations (addition, subtraction, and scaling) are
implemented directly on the flat array `[T; N]`.

#### Rationale:

- **Dimension-Agnostic Processing**: Because the data is contiguous, an
  element-wise operation (like adding two tensors of shape $[5, 5, 5]$) is
  mathematically identical to adding two flat vectors of size $125$.
- **BLAS level 1 Dispatch**: By treating tensors as flat slices, we bypass
  coordinate-to-flat-index conversion overhead during arithmetic. We pass the
  slices directly to optimized BLAS subprograms (such as `axpy` and `scal`),
  maximizing performance and loop vectorization regardless of the tensor's rank.
