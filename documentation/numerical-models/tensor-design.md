# Tensor Type (Design Document)

![Date Badge](https://img.shields.io/badge/Date-July_11,_2026-blue)
![Status Badge](https://img.shields.io/badge/Doc%20Status-Draft-orange)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

---

### **1. Introduction**

The `Tensor` implementation in `control-rs` provides a stack-allocated,
compile-time shaped, N-dimensional array. It is designed to extend linear
algebra operations beyond 2D matrices while enforcing strict memory
constraints and zero-cost abstractions. The library targets resource-constrained
embedded environments where dynamic memory allocation is unavailable, ensuring
deterministic memory usage and execution time.

---

### **2. Motivation**

N-dimensional arrays (tensors) are widely used in advanced control algorithms,
such as multi-variable non-linear systems, state-space representations with
spatial dimensions (e.g. discretized heat equations or structural dynamics
grids) and multichannel signal processing.

The rise of edge AI and TinyML requires executing low-power machine learning
models (such as convolutional neural networks or parameter networks) directly on
resource-constrained microcontrollers. These algorithms depend heavily on tensor
structures for weights, activations and multidimensional state trajectories.

Traditional tensor libraries rely on dynamic memory allocators and validate
shapes at runtime, leading to unacceptable runtime overhead and panic risks. By
combining Rust's const generics, layout abstractions, and compile-time dimension
matching, `control-rs` shifts shape mismatch checking entirely to compilation.
This enables safe, high-dimensional arithmetic on bare-metal systems.

---

### **3. Core Architecture and Memory Layout**

#### **3.1 Generics & Size Verification**

Rather than representing an $n$-dimensional tensor using nested arrays (e.g.,
`[[[T; D1]; D2]; D3]`), `Tensor` stores elements in a single flat array
`[T; N]`. Coordinates and dimensions are defined by a type implementing the
`TensorLayout` trait:

```rust
pub trait TensorLayout {
    const RANK: usize;
    type Size: Dim;
    fn dims() -> &'static [usize];
}
```

This separates the type constraints from complex nested arrays:

```rust
pub struct Tensor<T, Layout: TensorLayout> {
    data: [T; Layout::Size::DIM],
    _marker: PhantomData<Layout>,
}
```

To verify that the flat layout size matches the product of the individual
dimensions (e.g., $N = D_1 \cdot D_2 \cdot D_3$), the verification bounds are
placed on the implementation of `TensorLayout` for specific shape types. For
instance, a 3D layout shape uses type-level multiplication bounds to compute the
total size at compile time:

```rust
pub struct MyShape<D1: Dim, D2: Dim, D3: Dim> {
    _marker: PhantomData<(D1, D2, D3)>,
}

impl<D1: Dim, D2: Dim, D3: Dim> TensorLayout for MyShape<D1, D2, D3>
where
    D1: DimMul<D2>,
    <D1 as DimMul<D2>>::Output: DimMul<D3>,
{
    const RANK: usize = 3;
    type Size = <<D1 as DimMul<D2>>::Output as DimMul<D3>>::Output;

    fn dims() -> &'static [usize] {
        &[D1::DIM, D2::DIM, D3::DIM]
    }
}
```

This delegates size verification to layout constructions, guaranteeing that any
`Tensor` instantiated with a valid `TensorLayout` is structurally correct by
design.

#### **3.2 Storage & Column-Major Multi-Index Mapping**

Mapping multidimensional coordinates (e.g. $[i_0, i_1, \dots, i_{k-1}]$) to a
flat index is done using column-major strides (first dimension varies fastest):

$$\text{flat\_index} = i_0 + i_1 \cdot S_1 + i_2 \cdot S_2 + \dots$$

where the strides are $S_0 = 1$ and $S_m = \prod_{j=0}^{m-1} D_j$.

**Rationale:**

1. **Matrix Interoperability**: Column-major mapping aligns directly with the
   `Matrix` library's nested arrays (`[[T; R]; C]`). This ensures that 2D
   tensors can map directly to matrix arithmetic without axis transposition.
2. **Element-wise Efficiency**: By flattening storage into a flat `[T; N]`
   slice, element-wise addition, subtraction, and scaling are
   dimension-agnostic. They are computed sequentially, bypassing
   coordinate-to-index maths entirely.

#### **3.3 Representation**

To ensure standard C layout compatibility and contiguous storage without
padding:

```rust
#[repr(C)]
pub struct Tensor<T, Layout: TensorLayout> {
    data: [T; Layout::Size::DIM],
    _marker: PhantomData<Layout>,
}
```

---

### **4. API Specification**

#### **4.1 Instantiation**

The API provides constructors for static and runtime initialization:

* **Zero Tensor**: `pub const fn zero() -> Self`
* **From Flat Array**: `pub const fn from_raw(data: [T; N]) -> Self`
* **Coordinate Generator**:
  `pub fn from_fn<F>(f: F) -> Self where F: FnMut(&[usize]) -> T`

#### **4.2 Operator Overloading**

Arithmetic operators are overloaded by implementing `core::ops` traits:

* **Element-wise Arithmetic**: `Add`, `Sub`, and scalar `Mul` / `Div` are
  overloaded using the flat array slice, bypassing multi-index computations.

#### **4.3 Core Operations**

To facilitate non-allocating manipulation of tensor subsets, the API introduces
a `TensorView` type for referencing sub-regions without copying, alongside
support for in-place slice mutations and contractions:

```rust
/// Represents a non-allocating view into a subset of a Tensor's data.
pub struct TensorView<'a, T, Layout: TensorLayout> {
    data: &'a [T],
    _marker: PhantomData<Layout>,
}

impl<T, Layout: TensorLayout> Tensor<T, Layout> {
    /// Obtains a read-only TensorView referencing a subset of the tensor's data.
    pub fn as_view<SubLayout>(&self, offset: usize) -> TensorView<'_, T, SubLayout>
    where
        SubLayout: TensorLayout
    { /* ... */ }

    /// Mutates/writes to a subset of the tensor's content in-place.
    pub fn slice_inplace<SubLayout>(&mut self, offset: usize, value: &Tensor<T, SubLayout>)
    where
        SubLayout: TensorLayout
    { /* ... */ }

    /// Contracts the tensor along specified axes with another tensor (Einstein summation).
    /// Computes the contraction directly into a pre-allocated result tensor to avoid stack allocations.
    pub fn contract_into<OtherLayout, ResultLayout>(
        &self,
        other: &Tensor<T, OtherLayout>,
        result: &mut Tensor<T, ResultLayout>,
    )
    where
        OtherLayout: TensorLayout,
        ResultLayout: TensorLayout
    { /* ... */ }

    /// Permutes the axes of the tensor via stride transformations.
    pub fn permute<PermutedLayout>(&self) -> Tensor<T, PermutedLayout>
    where
        PermutedLayout: TensorLayout
    { /* ... */ }
}
```

---

### **5. Error Handling & State Management**

#### **5.1 Compile-Time Validation**

Because dimension incompatibilities are verified during compilation, the
generated binary does not contain size-checking branches or panic code for
arithmetic operations, providing zero runtime overhead.

#### **5.2 Runtime Fallbacks**

For dynamic coordinate retrieval accessed at runtime, the API provides standard
get interfaces that return `Option` wraps:

```rust
impl<T, Layout> Tensor<T, Layout> {
    pub fn get(&self, coords: &[usize]) -> Option<&T> { /* ... */ }
    pub fn get_mut(&mut self, coords: &[usize]) -> Option<&mut T> { /* ... */ }
}
```

To follow standard Rust library conventions, dynamic runtime indexing returns an
`Option<&T>` or `Option<&mut T>` rather than raising a panic, allowing control
loops to handle out-of-bounds cases gracefully.

---

### **6. Testing and Validation Framework**

#### **6.1 Standard Test Harness (CI & Coverage)**

Although the target environment is `#![no_std]`, the test suite is compiled and
run on the host using the standard `std` test harness. This enables tools like
`tarpaulin` to run coverage assessments, ensuring that all index mapping, tensor
contraction, and permutation paths are fully verified.

#### **6.2 Unit Testing in `no_std`**

We run target-specific tests using a minimal custom test harness on QEMU or
microcontrollers to guarantee compatibility with bare-metal target behavior.

#### **6.3 Property-Based Testing**

Integration with `proptest` automatically validates algebraic identities:

* Index mapping invariants: Mapping coordinates to flat index and back is
  identity.
* Double transposition: Permuting axes and reversing them returns the original
  tensor layout.
* Tensor contraction correctness against standard matrix multiplications for 2D
  slices.

---

### **7. Performance and Resource Considerations**

#### **7.1 Examples and Bloat Testing**

To ensure that the tensor type introduces no hidden code bloat or
compiler-injected heap dependencies, we compile mathematical examples located in
`examples/numerical_methods`. These binaries are analyzed with bloat checking
tools to ensure minimal flash footprint and deterministic stack frame sizes on
embedded targets.

#### **7.2 Stack Memory Limits**

Large tensors allocated on the stack risk overflows. To prevent stack overflows
on microcontrollers, the library limits total stack allocation size per
instance. For a 2D Matrix, a maximum dimension of 32 (`U32`) permits up
to $32 \times 32 = 1,024$ elements. For higher-dimensional Tensors, the total
element capacity is capped at 1,024 elements (e.g., $8 \times 8 \times 16$
or $10 \times 10 \times 10$) to maintain a consistent maximum stack footprint (~
4KB for `f32` elements) across all numerical types.

#### **7.3 Compiler Optimizations**

Flat-slice element-wise arithmetic passes slices directly to optimized BLAS
Level 1 routines (`axpy`, `scal`), allowing vectorization. Coordinate
computations are simplified to constant expressions at compile-time where
possible.

---

### **8. Development Plan**

Timeline and effort estimation for the `Tensor` module implementation:

| Task / Feature                         | Description                                                                                                              | Estimated Effort |
|:---------------------------------------|:-------------------------------------------------------------------------------------------------------------------------|:-----------------|
| **Phase 1: Layout & Macros**           | Implement `TensorLayout` trait and macros for shapes up to 4D. Define `Tensor` struct.                                   | 1.5 Days         |
| **Phase 2: Peano Sizing Verification** | Set up Peano bounds in struct signatures to verify compile-time capacity matching.                                       | 1.0 Day          |
| **Phase 3: Multi-Index Mapping**       | Implement column-major stride calculation and coordinate index resolution.                                               | 1.5 Days         |
| **Phase 4: Element-wise Arithmetic**   | Overload operators (`Add`, `Sub`, scalar scaling) using flat-slice operations.                                           | 1.0 Day          |
| **Phase 5: Contractions & Slicing**    | Implement tensor contraction (Einstein sum) and coordinate sub-tensor slicing.                                           | 2.5 Days         |
| **Phase 6: Testing & Bloat Analysis**  | Write property tests for stride correctness, set up host/target testing, and check size via `numerical_methods` example. | 1.5 Days         |
