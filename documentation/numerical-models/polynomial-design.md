# Polynomial Type (Design Document)

![Date Badge](https://img.shields.io/badge/Date-July_12,_2026-blue)
![Status Badge](https://img.shields.io/badge/Doc%20Status-Draft-orange)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

---

### **1. Introduction**

The `Polynomial` type in `control-rs` provides a statically sized,
stack-allocated, single-variable representation of polynomials. Designed for
classical frequency-domain control engineering and signal processing, this type
enables algebraic manipulations (cascade, parallel, feedback) and discretization
routines without heap allocation.


---

### **2. Motivation & Target Constraints**

#### **2.1 Environmental Limitations**

Embedded control loop platforms require deterministic memory footprints. A
dynamic allocator cannot be used, meaning polynomials must have compile-time
bounded sizes to avoid runtime stack overflows in real-time execution kernels.

#### **2.2 Target Applications**

- **LTI System Transfer Functions**: Series, parallel, and feedback connections
  of rational functions $ H(s) = B(s)/A(s) $.
- **Discretization Algorithms**: Bilinear (Tustin) transform and ZOH mapping of
  continuous s-domain models to discrete z-domain equivalents.
- **Trajectory Generation**: Generating smooth motion paths via cubic or quintic
  splines.

#### **2.3 Safety & Static Verification**

To prevent runtime sizing bugs in control loops, polynomial capacities are bound
directly in the type system. Operations like polynomial multiplication yield a
result whose type-level capacity is statically verified to fit the product
degree, catching capacity overflows at compile time.

---

### **3. Core Architecture & Memory Layout**

#### **3.1 Generics Foundation & Sizing**

The `Polynomial` structure is declared as:

```rust
pub struct Polynomial<T, N: Dim> {
    data: [T; N::DIM],
}
```

Here, `N` represents the capacity (number of coefficients, meaning the maximum
possible polynomial degree is $ N - 1 $). Dimension sizing is verified using
the type-level Peano traits defined
in [num_types.rs](../../src/math/num_types.rs).

#### **3.2 Memory Layout & Storage Strategy**

Coefficients are stored in a contiguous flat array in **ascending order of
powers**:
$$ p(x) = c_0 + c_1 x + c_2 x^2 + \dots + c_{N-1} x^{N-1} $$
where `data[i]` contains the coefficient for the exponent $ x^i $.

- **Ascending Power Storage Rationale**:
    - Direct index-to-exponent mapping: `data[i]` corresponds exactly to the
      power $ x^i $, avoiding index arithmetic offsets.
    - Zero-cost padding: When adding polynomials of differing sizes, aligning
      coefficients is trivial (coefficients for matching indices are added
      directly) without shifting elements.
- **Alternative Considered (Descending Power Storage)**: Descending order (where
  `data[0]` is the coefficient for $ x^{N-1} $, as used in MATLAB) requires
  index shifting when capacities change, introducing runtime overhead.

#### **3.3 Memory Representation & Slicing**

To ensure zero-cost layout mapping, the struct uses the transparent
representation:

```rust
#[repr(transparent)]
pub struct Polynomial<T, N: Dim> {
    data: [T; N::DIM],
}
```

This layout allows exposing safe flat slice interfaces:

```rust
impl<T, N: Dim> Polynomial<T, N> {
    pub const fn as_slice(&self) -> &[T] {
        &self.data
    }

    pub const fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }
}
```

---

### **4. API Specification**

#### **4.1 Instantiation & Constructors**

- `pub const fn constant(val: T) -> Polynomial<T, U1>`: Constructs a degree-0
  polynomial.
- `pub const fn line(c0: T, c1: T) -> Polynomial<T, U2>`: Constructs a degree-1
  polynomial.
- `pub const fn from_coefficients(data: [T; N::DIM]) -> Self`: Direct array
  initialization.
- `pub fn from_fn<F>(f: F) -> Self where F: FnMut(usize) -> T`: Constructs
  coefficients from a mapping function.

#### **4.2 Operator Overloading**

Overloads standard traits (`Add`, `Sub`). Multiplication features two
interfaces:

1. `mul_poly`: Static multiplication returning a combined capacity bound:
   ```rust
   impl<T, N: Dim> Polynomial<T, N> {
     pub fn mul_poly<M: Dim>(&self, other: &Polynomial<T, M>) -> 
       Polynomial<T,<<N as DimAdd<M>>::Output as DimSub<U1>>::Output>
     where
         N: DimAdd<M>,
         <N as DimAdd<M>>::Output: DimSub<U1> { /* .. */}
   }
   ```
2. `mul_with_conv`: Decouples arithmetic from the representation by accepting an
   implementor of the `Convolution<T>` trait, facilitating the use of
   hardware-optimized DSP convolution libraries.

#### **4.3 Core Operations**

- **Evaluation via Horner's Method**:
  Evaluates the polynomial using the recurrence relation:
  $$ p(x) = c_0 + x(c_1 + x(c_2 + \dots)) $$
  Horner's method minimizes floating-point rounding error
  accumulation and requires only $ N-1 $ additions and multiplications.
  Detailed error bounds are described
  in [Rounding Error of Polynomial Evaluation](https://www.sciencedirect.com/science/article/pii/0771050X79900020).
- **Long Division (`div_rem`)**:
  Calculates the quotient and remainder:
  ```rust
  impl<T, N: Dim> Polynomial<T, N> {
      pub fn div_rem<M: Dim, Q: Dim, R: Dim>(&self, divisor: &Polynomial<T, 
  M>) -> Result<(Polynomial<T, Q>, Polynomial<T, R>), DivisionError> 
    { /* ... */} 
  }
  ```
  [Fast in-place algorithms for polynomial operations: division, evaluation, interpolation](https://dl.acm.org/doi/abs/10.1145/3373207.3404061)
- **Interpolation**: Compute the optimal polynomial for a set of points.
  [Fast in-place algorithms for polynomial operations: division, evaluation, interpolation](https://dl.acm.org/doi/abs/10.1145/3373207.3404061)
- **Calculus Operations**: Analytical derivative and integral functions return
  statically resized polynomial bounds:
  ```rust
  impl<T, N: Dim> Polynomial<T, N> {
    pub fn derivative(&self) -> Polynomial<T, <N as DimSub<U1>>::Output> 
  where N: DimSub<U1> { /* .. */ }
    pub fn integral(&self, constant: T) -> Polynomial<T, <N as 
  DimAdd<U1>>::Output> where N: DimAdd<U1> { /* .. */ }
  }
  ```

#### **4.4 Interoperability & Conversions**

##### **4.4.1 Conversion to Matrix**

A monic polynomial of degree $ n = N - 1 $ converts to its $ n \times n $
companion matrix in Controllable Canonical Form.

- **Type Signature**:
  ```rust
  impl<T, N: Dim> TryFrom<Polynomial<T, N>> for Matrix<T, <N as DimSub<U1>>::Output, <N as DimSub<U1>>::Output>
  where
      N: DimSub<U1>,
      <N as DimSub<U1>>::Output: Dim,
      T: Copy + Default + Neg<Output = T> + From<i32>, { /* .. */ }
  ```
- **Behavior**: Instantiates the companion matrix. The superdiagonal is
  populated with ones, and the bottom row contains the negative polynomial
  coefficients.

##### **4.4.2 Conversion to Tensor**

Converts a polynomial to a 1D `Tensor<T, Layout>`.

- **Type Signature**:
  ```rust
  impl<T, N: Dim, Layout: TensorLayout> TryFrom<Polynomial<T, N>> for Tensor<T, Layout>
  where
      Layout: TensorLayout<Size = N>, { /* .. */ }
  ```
- **Behavior**: Constructs a rank-1 tensor using the flat coefficient data.

---

### **5. Error Handling & State Management**

#### **5.1 Compile-Time Constraints**

Incompatible polynomial additions (different static capacities) or overflow
multiplications fail at compile time, preventing indexing bugs in target
execution loops.

#### **5.2 Runtime Fallbacks**

- Division by zero or capacity overflow during polynomial division returns a
  `Result<..., DivisionError>` instead of panicking.
- Bounds access checks return `Option<&T>` via `get()` methods.

---

### **6. Testing & Validation Framework**

#### **6.1 Host/Target Test Integration**

Host-side testing using standard coverage tools ensures the accuracy of algebra
routines. Cross-compiled QEMU runs verify no-std execution compatibility.

#### **6.2 Property-Based Testing**

Uses `proptest` to check:

- Commutativity: $ P + Q = Q + P $
- Distributivity: $ P \cdot (Q + R) = P \cdot Q + P \cdot R $
- Division consistency: $ P = Q \cdot D + R $

#### **6.3 Benchmarks and Quality Reporting**

Horner evaluation cycle counts are profiled. Target binary sizes are monitored
to prevent compilation bloat.

---

### **7. Performance & Resource Considerations**

#### **7.1 Stack Overflow Prevention & Memory Safety**

To guarantee safety on thin microcontrollers, polynomial array capacities are
capped at $ 128 $ elements. This limits stack allocation overhead while
supporting high-degree models.

#### **7.2 Code Bloat & Binary Size Validation**

Compilation artifacts are monitored via `examples/numerical_methods` to ensure
compiler dead-code elimination successfully removes unused polynomial variants.

#### **7.3 Compiler Optimizations & Hardware Acceleration**

Evaluation iterates over flat slices, allowing compiler auto-vectorization (
SIMD) and loop unrolling to optimize the multiplication-addition steps.

---

### **8. Structural Specializations & Extensions**

Future extensions include new-type wrappers for orthogonal representations (
e.g., Chebyshev polynomials for function approximation) and sparse polynomial
layouts for high-degree models with few coefficients.

---

### **9. Classical Control Design Examples**

#### **9.1 Closed-Loop Transfer Function Synthesis**

Computing the closed-loop system numerator and denominator:
$$ G_{cl}(s) = \frac{G_p(s)}{1 + G_p(s)G_c(s)} $$
where $ G_p(s) = \frac{N_p(s)}{D_p(s)} $ and $ G_c(s) = \frac{N_c(s)}{D_c(
s)} $.
This yields:
$$ G_{cl}(s) = \frac{N_p(s) D_c(s)}{D_p(s) D_c(s) + N_p(s) N_c(s)} $$

```rust
use control_rs::math::polynomial::{Polynomial, Dim};

pub fn compute_closed_loop<T, Np: Dim, Dp: Dim, Nc: Dim, Dc: Dim>(
    num_p: &Polynomial<T, Np>,
    den_p: &Polynomial<T, Dp>,
    num_c: &Polynomial<T, Nc>,
    den_c: &Polynomial<T, Dc>,
) -> (
    Polynomial<T, <<Np as DimAdd<Dc>>::Output as DimSub<U1>>::Output>,
    Polynomial<T, <<Dp as DimAdd<Dc>>::Output as DimSub<U1>>::Output>,
)
where
    T: Copy + Default + Add<Output=T> + Mul<Output=T>,
    Np: DimAdd<Dc>,
    <Np as DimAdd<Dc>>::Output: DimSub<U1>,
    Dp: DimAdd<Dc>,
    <Dp as DimAdd<Dc>>::Output: DimSub<U1>,
    Np: DimAdd<Nc>,
    <Np as DimAdd<Nc>>::Output: DimSub<U1>,
// Verify that denominator product and numerator product have matching capacities for addition
    <<Dp as DimAdd<Dc>>::Output as DimSub<U1>>::Output: Dim,
{
    // Closed-loop numerator: N_p * D_c
    let cl_num = num_p.mul_poly(den_c);

    // Denominator term 1: D_p * D_c
    let den_term1 = den_p.mul_poly(den_c);

    // Denominator term 2: N_p * N_c
    let den_term2 = num_p.mul_poly(num_c);

    // Closed-loop denominator: D_p * D_c + N_p * N_c
    let cl_den = den_term1 + den_term2;

    (cl_num, cl_den)
}
```

#### **9.2 Continuous-to-Discrete Bilinear (Tustin) Transform**

Applying the Tustin transform to discretize a continuous-time denominator. The
Tustin transform maps continuous variables using the approximation:
$$ s \approx \frac{2}{T_s} \frac{z - 1}{z + 1} $$
For a first-order denominator $ A(s) = a_1 s + a_0 $, substituting $ s $
yields the discrete polynomial coefficients described
in [Bilinear/Tustin Transform (S-to-Z)](https://dergipark.org.tr/en/download/article-file/2005762):
$$ A(z) = \left( a_0 + \frac{2 a_1}{T_s} \right) z + \left( a_0 - \frac{2
a_1}{T_s} \right) $$

```rust
use control_rs::math::polynomial::{Polynomial, U2};

pub fn tustin_discretize_first_order<T>(
    den_s: &Polynomial<T, U2>, // a_1 * s + a_0 -> [a_0, a_1]
    t_s: T,
) -> Polynomial<T, U2>
where
    T: Copy + Default + Add<Output=T> + Sub<Output=T> + Mul<Output=T> + Div<Output=T> + From<i32>,
{
    let a_0 = den_s.as_slice()[0];
    let a_1 = den_s.as_slice()[1];

    let two = T::from(2);
    let ratio = (two * a_1) / t_s;

    let z_1 = a_0 + ratio; // Coefficient of z^1
    let z_0 = a_0 - ratio; // Coefficient of z^0

    Polynomial::from_coefficients([z_0, z_1])
}
```

---

### **10. Development Plan & Roadmap**

| Task / Feature             | Description                                                  | Estimated Effort |
|:---------------------------|:-------------------------------------------------------------|:-----------------|
| **Phase 1: Basic Storage** | Ascending storage arrays and constructors.                   | 1.0 Day          |
| **Phase 2: Horner Eval**   | Implement Horner's evaluation method and standard operators. | 1.5 Days         |
| **Phase 3: Calculus**      | Differentiation and integration routines.                    | 1.0 Day          |
| **Phase 4: Div & Conv**    | Polynomial convolution and division.                         | 2.0 Days         |
| **Phase 5: Verification**  | Property tests and binary size audits.                       | 1.5 Days         |
