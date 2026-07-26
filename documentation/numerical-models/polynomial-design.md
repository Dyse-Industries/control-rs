# Polynomial Type (Design Document)

![Date Badge](https://img.shields.io/badge/Date-July_26,_2026-blue)
![Status Badge](https://img.shields.io/badge/Doc%20Status-Draft-orange)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

---

### **1. Introduction**

The `Polynomial` type in `control-rs` provides a statically sized, single-variable representation of polynomials parameterized over a memory storage backend (`Storage<T, N, U1>`). Designed for classical control engineering, signal processing, trajectory generation, and numerical algebra, this type enables algebraic operations and discretization routines without mandatory heap allocation or forced stack ownership.

---

### **2. Motivation & Target Constraints**

#### **2.1 Environmental Limitations**

Embedded control loop platforms require deterministic memory footprints. Dynamic allocation is prohibited in real-time execution kernels, meaning polynomial capacity must be compile-time bounded or backed by caller-provided borrowed buffers (`&[T]`) to eliminate stack overflow risks on memory-constrained targets.

#### **2.2 Target Applications**

- **Signal Processing & Filter Synthesis**: FIR/IIR digital filter representations and polynomial evaluation.
- **Discretization Algorithms**: Bilinear (Tustin) transform and ZOH mapping of continuous models to discrete equivalents.
- **Trajectory Generation**: Generating smooth motion paths via cubic or quintic splines.
- **Companion Matrices & Root Finding**: Canonical matrix transformations for characteristic polynomial analysis.

#### **2.3 Safety & Static Verification**

Polynomial capacities are bound directly in the type system via Peano numbers (`Dim`). Operations such as polynomial multiplication yield a result whose type-level capacity is statically verified to fit the product degree, catching capacity mismatches at compile time.

---

### **3. Core Architecture & Memory Layout**

#### **3.1 Generics Foundation & Storage Strategy**

The core `Polynomial` structure decouples mathematical dimensions from physical storage using the `Storage<T, R, C>` trait hierarchy (with $R = N$ and $C = U1$):

```rust
pub struct Polynomial<T, N: Dim, S: Storage<T, N, U1> = ArrayStorage<T, N, U1>> {
    storage: S,
    _marker: core::marker::PhantomData<N>,
}
```

Here, `N` represents the capacity (number of coefficients, maximum possible degree is $N - 1$), and `S` defines where the coefficients reside (e.g. stack `ArrayStorage`, borrowed `MatrixView`, or static Flash memory).

#### **3.2 Storage Backends & Zero-Copy Views**

By parameterizing `Polynomial` over `Storage<T, N, U1>`, `control-rs` supports multiple ownership models without duplicating algebraic logic:

```rust
/// Owning polynomial backed by column-major stack array
pub type ArrayPolynomial<T, N> = Polynomial<T, N, ArrayStorage<T, N, U1>>;

/// Zero-copy read-only borrowed polynomial view over &[T]
pub type PolynomialView<'a, T, N> = Polynomial<T, N, MatrixView<'a, T, N, U1>>;

/// Zero-copy mutable borrowed polynomial view over &mut [T]
pub type PolynomialViewMut<'a, T, N> = Polynomial<T, N, MatrixViewMut<'a, T, N, U1>>;
```

#### **3.3 Coefficient Memory Layout**

Coefficients are stored in **ascending order of powers**:
$$ p(x) = c_0 + c_1 x + c_2 x^2 + \dots + c_{N-1} x^{N-1} $$
where index `i` maps to the coefficient of $x^i$.

- **Ascending Power Storage Rationale**:
  - Direct index-to-exponent mapping: element at index `i` corresponds directly to $x^i$.
  - Zero-cost padding: Adding polynomials of differing capacities aligns coefficients naturally without element shifting.

#### **3.4 Memory Representation & Slicing**

Contiguous slice interfaces are safely exposed when the storage backend implements `ContiguousStorage` or `ContiguousStorageMut`:

```rust
impl<T, N: Dim, S> Polynomial<T, N, S>
where
    S: ContiguousStorage<T, N, U1>,
{
    pub fn as_slice(&self) -> &[T] {
        self.storage.as_slice()
    }
}

impl<T, N: Dim, S> Polynomial<T, N, S>
where
    S: ContiguousStorageMut<T, N, U1>,
{
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.storage.as_mut_slice()
    }
}
```

---

### **4. API Specification**

#### **4.1 Instantiation & Constructors**

- `pub const fn constant(val: T) -> Polynomial<T, U1> where T: Copy`: Constructs a degree-0 polynomial containing a single coefficient.
- `pub const fn line(c0: T, c1: T) -> Polynomial<T, U2> where T: Copy`: Constructs a degree-1 polynomial $[c_0, c_1]$ ($c_0 + c_1 x$).
- `pub const fn from_coefficients(data: [T; N::DIM]) -> Polynomial<T, N, ArrayStorage<T, N, U1>>`: Constructs an owning stack polynomial.
- `pub const fn from_storage(storage: S) -> Self`: Constructs a polynomial wrapping a custom storage backend `S`.
- `pub fn from_slice(slice: &'a [T]) -> PolynomialView<'a, T, N>`: Constructs a borrowed zero-copy view over existing memory.
- `pub fn from_fn<F>(f: F) -> Polynomial<T, N, ArrayStorage<T, N, U1>> where F: FnMut(usize) -> T`: Generates coefficients via a mapping closure.

#### **4.2 Operator Overloading**

Overloads standard traits (`Add`, `Sub`, `Neg`). Polynomial multiplication provides two interfaces:

1. `mul_poly`: Static multiplication returning a combined capacity bound:
   ```rust
   impl<T, N: Dim, S1: Storage<T, N, U1>> Polynomial<T, N, S1> {
       pub fn mul_poly<M: Dim, S2: Storage<T, M, U1>>(
           &self,
           other: &Polynomial<T, M, S2>,
       ) -> Polynomial<T, <<N as DimAdd<M>>::Output as DimSub<U1>>::Output, ArrayStorage<T, <<N as DimAdd<M>>::Output as DimSub<U1>>::Output, U1>>
       where
           N: DimAdd<M>,
           <N as DimAdd<M>>::Output: DimSub<U1>,
           <N as DimAdd<M>>::Output: Dim,
           T: Copy + Zero + Add<Output = T> + Mul<Output = T>,
       { /* ... */ }
   }
   ```
2. `mul_with_conv`: Decouples arithmetic from representation by leveraging the `Convolution<T>` trait and underlying hardware-optimized DSP kernels.

#### **4.3 Core Operations**

- **Horner's Method Evaluation**:
  Evaluates $p(x)$ using the recurrence relation $p(x) = c_0 + x(c_1 + x(c_2 + \dots))$ directly via storage element access (`Storage::get_unchecked`). Minimizes floating-point rounding errors ($N-1$ additions and multiplications).
- **Polynomial Division (`div_rem`)**:
  Computes quotient and remainder:
  ```rust
  impl<T, N: Dim, S: Storage<T, N, U1>> Polynomial<T, N, S> {
      pub fn div_rem<M: Dim, Sm: Storage<T, M, U1>, Q: Dim, R: Dim>(
          &self,
          divisor: &Polynomial<T, M, Sm>,
      ) -> Result<(Polynomial<T, Q>, Polynomial<T, R>), DivisionError> { /* ... */ }
  }
  ```
- **Calculus Operations**:
  Analytical derivative and integral methods returning statically resized polynomial bounds.

#### **4.4 Interoperability & Conversions**

##### **4.4.1 Companion Matrix Conversion**

A monic polynomial of degree $n = N - 1$ converts to its $n \times n$ companion matrix in Controllable Canonical Form:

```rust
impl<T, N: Dim, S: Storage<T, N, U1>> TryFrom<Polynomial<T, N, S>>
    for Matrix<T, <N as DimSub<U1>>::Output, <N as DimSub<U1>>::Output>
where
    N: DimSub<U1>,
    <N as DimSub<U1>>::Output: Dim,
    T: Zero + One + Copy + Neg<Output = T> + PartialEq,
{
    type Error = ConversionError;
    // ...
}
```

##### **4.4.2 Tensor Conversion**

Converts flat coefficient data into a 1D `Tensor<T, Layout>`.

---

### **5. Error Handling & State Management**

#### **5.1 Compile-Time Constraints**

Capacity mismatches during polynomial arithmetic are rejected at compile time via Peano type constraints.

#### **5.2 Runtime Error Handling**

- Zero division or degree mismatch returns `Result<..., DivisionError>`.
- Bounds checked element access returns `Option<&T>`.

---

### **6. Testing & Validation Framework**

- **Host/Target Tests**: Unit tests executed on host and qemu targets.
- **Property-Based Testing**: `proptest` validation for commutativity ($P+Q=Q+P$), distributivity ($P(Q+R) = PQ + PR$), and division invariants ($P = QD + R$).

---

### **7. Performance & Resource Considerations**

- **Zero-Cost Abstraction**: Storage abstraction monomorphizes and inlines without vtables or dynamic allocation.
- **Vectorization**: Contiguous storage backends enable compiler SIMD auto-vectorization over coefficient slices.

---

### **8. Revision History**

| Date | Author | Description |
|:---|:---|:---|
| July 12, 2026 | @MitchellDScott | Initial draft with static array layout. |
| July 26, 2026 | @MitchellDScott | Integrated `Storage<T, N, U1>` trait hierarchy to support borrowed zero-copy views and ROM storage. |
