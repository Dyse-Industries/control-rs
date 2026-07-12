# Tensor Type (Design Document)

![Date Badge](https://img.shields.io/badge/Date-July_12,_2026-blue)
![Status Badge](https://img.shields.io/badge/Doc%20Status-Draft-orange)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

---

### **1. Introduction**

The `Tensor` implementation in `control-rs` provides a stack-allocated,
compile-time shaped, N-dimensional array. It extends linear algebra operations
beyond 2-dimensional boundaries while enforcing strict memory limitations and
zero-cost abstractions on bare-metal systems.

---

### **2. Motivation & Target Constraints**

#### **2.1 Environmental Limitations**

Embedded controllers lack dynamic memory allocation. High-dimensional arrays
must occupy deterministic, contiguous segments of stack memory with compile-time
sized limits.

#### **2.2 Target Applications**

- **Multi-Variable Spatial Grids**: Storing state variables over discretized
  grids (e.g., structural vibrations, thermal distributions).
- **Edge AI & TinyML**: Storing weights, biases, and activation states of neural
  network classifiers directly on microcontrollers, as outlined
  in [TinyML for Ubiquitous Edge AI](https://arxiv.org/pdf/2102.01255).
- **Multi-Channel Signal Processing**: Handling audio, radar, or multi-sensor
  streams.

#### **2.3 Safety & Static Verification**

High-dimensional coordinate indexing is prone to runtime errors. By validating
shapes and layout sizes during compilation, `control-rs` ensures that arithmetic
operations (e.g., tensor contraction, coordinate slicing) never trigger runtime
panic handlers in critical control paths.

---

### **3. Core Architecture & Memory Layout**

#### **3.1 Generics Foundation & Sizing**

To support varying dimensions without nesting arrays, a `Tensor` wraps a single
flat array. Coordinate mapping is delegated to a type implementing the
`TensorLayout` trait:

```rust
pub trait TensorLayout {
    const RANK: usize;
    type Size: Dim;
    fn dims() -> &'static [usize];
}

pub struct Tensor<T, Layout: TensorLayout> {
    data: [T; Layout::Size::DIM],
    _marker: PhantomData<Layout>,
}
```

Layout sizing verification uses type-level bounds:

```rust
pub struct Shape3D<D1: Dim, D2: Dim, D3: Dim> {
    _marker: PhantomData<(D1, D2, D3)>,
}

impl<D1: Dim, D2: Dim, D3: Dim> TensorLayout for Shape3D<D1, D2, D3>
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

This guarantees that shape correctness is verified by the compiler.

#### **3.2 Memory Layout & Storage Strategy**

Multidimensional coordinates are mapped to a flat index using column-major
strides (first dimension varies fastest):
$$ \text{flat\_index} = i_0 + i_1 \cdot S_1 + i_2 \cdot S_2 + \dots $$
where the strides are $ S_0 = 1 $ and $ S_m = \prod_{j=0}^{m-1} D_j $.

- **Matrix Interoperability**: Column-major layout matches the contiguous
  column-major nested layout of the `Matrix` library, allowing 2D tensors to
  interoperate with matrices without transposition overhead.
- **Flat Indexing Efficiency**: Sequentially operating over the flat array
  bypassing multi-index arithmetic minimizes cycles for element-wise operations.

#### **3.3 Memory Representation & Slicing**

To ensure stable memory layout:

```rust
#[repr(C)]
pub struct Tensor<T, Layout: TensorLayout> {
    data: [T; Layout::Size::DIM],
    _marker: PhantomData<Layout>,
}
```

This representation facilitates casting elements to contiguous flat slices (
`&[T]`) for BLAS-like subprogram routing.

---

### **4. API Specification**

#### **4.1 Instantiation & Constructors**

- `pub const fn zero() -> Self where T: Zero + Copy`: Instantiates an all-zero tensor using `T::ZERO`.
- `pub const fn from_raw(data: [T; Layout::Size::DIM]) -> Self`: Directly initializes from a flat array of matching capacity.
- `pub fn from_fn<F>(f: F) -> Self where F: FnMut(&[usize]) -> T`: Generates values using a coordinate mapping function at runtime.

*Implementation Note*: To support generic `const fn` initialization on stable Rust, the scalar type `T` must implement the `Zero` and `One` traits from `crate::math::num_traits`. These traits expose the associated constants `T::ZERO` and `T::ONE`.

#### **4.2 Operator Overloading**

Overloads `Add`, `Sub`, and scalar `Mul` / `Div` traits. These operations
iterate directly over the flat internal array, avoiding multi-index coordinate
mapping.

#### **4.3 Core Operations**

- `TensorView`: Non-allocating subset references.
  ```rust
  pub struct TensorView<'a, T, Layout: TensorLayout> {
      data: &'a [T],
      _marker: PhantomData<Layout>,
  }
  ```
- `slice_inplace`: Writes a sub-tensor in-place into the target layout.
- `contract_into`: Contracts axes with another tensor (Einstein summation) into
  a pre-allocated result tensor.
  ```rust
  pub fn contract_into<OtherLayout, ResultLayout>(
      &self,
      axis_self: usize,
      other: &Tensor<T, OtherLayout>,
      axis_other: usize,
      result: &mut Tensor<T, ResultLayout>,
  )
  ```
- `permute`: Permutes the axes via stride transformations.

#### **4.4 Interoperability & Conversions**

##### **4.4.1 Conversion to Matrix**

A `Tensor<T, Layout>` converts to a `Matrix<T, R, C>` if the layout is 2D and
matches dimensions.

- **Type Signature**:
  ```rust
  impl<T, Layout, R: Dim, C: Dim> TryFrom<Tensor<T, Layout>> for Matrix<T, R, C>
  where
      Layout: TensorLayout<Size = <R as DimMul<C>>::Output>,
  {
      type Error = ConversionError;
      // ...
  }
  ```
- **Behavior**: Copy-maps elements from the flat column-major storage to the
  nested column-major array structure.
- **Failure Condition**: Returns `ConversionError::LayoutMismatch` if `Layout::RANK != 2` or if the layout's dimensions do not match the matrix's rows and columns ($ R \times C $).

##### **4.4.2 Conversion to Polynomial**

A `Tensor<T, Layout>` converts to a `Polynomial<T, N>` if it is a 1D tensor of
matching size.

- **Type Signature**:
  ```rust
  impl<T, Layout, N: Dim> TryFrom<Tensor<T, Layout>> for Polynomial<T, N>
  where
      Layout: TensorLayout<Size = N>,
  {
      type Error = ConversionError;
      // ...
  }
  ```
- **Behavior**: Copies flat tensor data to construct ascending polynomial
  coefficients.
- **Failure Condition**: Returns `ConversionError::LayoutMismatch` if `Layout::RANK != 1` or if the layout's size does not match the polynomial's capacity $N$.

---

### **5. Error Handling & State Management**

#### **5.1 Compile-Time Constraints**

Dimension and rank mismatches in tensor contraction or layout transformations
result in compile-time type errors.

#### **5.2 Runtime Fallbacks**

Dynamic coordinates queried at runtime return `Option<&T>` or `Option<&mut T>`
via `get()` and `get_mut()` rather than raising a panic.

---

### **6. Testing & Validation Framework**

#### **6.1 Host/Target Test Integration**

Tests run on the host via standard `cargo test` for unit verification. Embedded
targets are validated under QEMU to ensure correct cross-compilation alignment.

#### **6.2 Property-Based Testing**

Uses `proptest` to verify:

- Stride index calculations.
- Axis permutation round trips: Permuting axes and reversing them returns the
  original layout.
- Contraction equivalence to standard matrix multiplication for 2D slices.

#### **6.3 Benchmarks and Quality Reporting**

Contraction algorithms are benchmarked on ARM hardware. Binary size checks
verify that unused shape variants are pruned.

---

### **7. Performance & Resource Considerations**

#### **7.1 Stack Overflow Prevention & Memory Safety**

To prevent stack exhaustion, total element capacity across all dimensions is
capped at 1,024 elements (e.g., $ 8 \times 8 \times 16 $ or $ 10 \times 10
\times 10 $). This guarantees that a single tensor never exceeds 4KB of stack
space (for `f32`).

#### **7.2 Code Bloat & Binary Size Validation**

Code footprint is checked against target examples to ensure compiler dead-code
elimination successfully removes unused generic structures.

#### **7.3 Compiler Optimizations & Hardware Acceleration**

Flat slices are passed to optimized BLAS Level 1 and Level 2 routines.
Coordinate computations are simplified to constant expressions at compile-time
where possible.

---

### **8. Structural Specializations & Extensions**

Future extensions include new-type wrappers for sparse tensor representations
and matrix-free operators (which perform tensor operations without instantiating
intermediate storage).

---

### **9. High-Dimensional Array Examples**

#### **9.1 Accessing a 2D View from a 3D Grid**

Extracting a 2D spatial slice from a 3D grid layout using a non-allocating
`TensorView`:

```rust
use control_rs::math::tensor::{Tensor, TensorView, TensorLayout, Shape3D, Shape2D};
use control_rs::math::num_types::{U8, U4, U2};

pub fn extract_spatial_slice<T>(
    grid: &Tensor<T, Shape3D<U8, U8, U4>>, // 8 x 8 x 4 grid
    z_offset: usize, // Select which 2D layer to extract (0..3)
) -> TensorView<'_, T, Shape2D<U8, U8>> // Returns an 8 x 8 view
where
    T: Copy,
{
    // A single 8x8 slice has 64 elements. The offset is z_offset * 64
    let offset = z_offset * 64;

    // Obtain view into the slice
    grid.as_view::<Shape2D<U8, U8>>(offset)
}
```

#### **9.2 Tensor Contraction (Einstein Summation)**

Contracting a 3D state tensor $ X $ (spatial grid over time) with a 2D
transition tensor $ A $ to evaluate the next time step. The contraction along
the spatial index $ j $ is defined as:
$$ Y_{i, k, l} = \sum_{j} A_{i, j} X_{j, k, l} $$

```rust
use control_rs::math::tensor::{Tensor, TensorLayout, Shape3D, Shape2D};
use control_rs::math::num_types::{U4, U2};
use control_rs::math::num_traits::Ring;

pub fn contract_state_transition<T>(
    a: &Tensor<T, Shape2D<U4, U4>>, // 4 x 4 transition matrix (2D tensor)
    x: &Tensor<T, Shape3D<U4, U2, U2>>, // 4 x 2 x 2 state tensor (spatial x grid x time)
    y: &mut Tensor<T, Shape3D<U4, U2, U2>>, // Destination tensor for transition output
)
where
    T: Ring + Copy,
{
    // Contract A and X along axis 1 of A and axis 0 of X:
    // Computes directly into the pre-allocated Y tensor to prevent stack 
    // allocation.
    a.contract_into(1, x, 0, y);
}
```

---

### **10. Development Plan & Roadmap**

| Task / Feature                 | Description                                   | Estimated Effort |
|:-------------------------------|:----------------------------------------------|:-----------------|
| **Phase 1: Layout Traits**     | Define `TensorLayout` and 2D/3D shapes.       | 1.5 Days         |
| **Phase 2: Peano Sizing**      | Add type bounds for layout sizes.             | 1.0 Day          |
| **Phase 3: Coordinate Stride** | Implement column-major index mapping.         | 1.5 Days         |
| **Phase 4: Element Ops**       | Implement addition, subtraction, and scaling. | 1.0 Day          |
| **Phase 5: Contractions**      | Implement tensor contraction and permutation. | 2.5 Days         |
| **Phase 6: Verification**      | Implement `proptest` suites and size audits.  | 1.5 Days         |
| **Phase 7: Interoperability**  | Implement `TryFrom` conversions between `Tensor`, `Matrix`, and `Polynomial`. Depends on Tensor Phase 1, 2, & 3, Matrix Phase 1 & 2, and Polynomial Phase 1. | 2.0 Days |
