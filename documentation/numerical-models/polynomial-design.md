# Polynomial Type (Design Document)

![Date Badge](https://img.shields.io/badge/Date-July_11,_2026-blue)
![Status Badge](https://img.shields.io/badge/Doc%20Status-Draft-orange)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

---

### **1. Introduction**

The `Polynomial` type in `control-rs` provides a high-performance,
stack-allocated, single-variable representation of polynomials. Designed
specifically for classical control systems engineering, this type facilitates
frequency-domain analyses, coefficient representations, and trajectory models
without relying on dynamic memory allocations. To support safety-critical
applications, the implementation strictly adheres to`#![no_std]` targets,
ensuring deterministic execution time and memory footprint.

---

### **2. Motivation**

Polynomials are fundamental mathematical structures in classical control
engineering. In continuous-time frequency-domain and discrete-time z-domain
analysis, linear time-invariant (LTI) systems are represented as rational
transfer functions:

$$H(s) = \frac{B(s)}{A(s)} = \frac{b_m s^m + b_{m-1} s^{m-1} + \dots + b_1 s + b_0}{a_n s^n + a_{n-1} s^{n-1} + \dots + a_1 s + a_0}$$

Unlike matrices or vectors, polynomial types are not commonly provided by
general-purpose linear algebra libraries. Therefore, `control-rs` must establish
its own robust, type-safe polynomial engine.

The primary motivation is to represent and manipulate transfer function
equations—such as series cascade connection (polynomial multiplication),
parallel connection (polynomial addition), feedback loops, and discretization
routines—entirely at compile-time and stack-space limits. This guarantees that
classical control models can be evaluated inside sub-millisecond,
safety-critical loops without risking runtime panics or heap-allocation delays.

---

### **3. Core Architecture and Memory Layout**

#### **3.1 Generics Foundation**

To avoid heap allocations while supporting polynomials of varying degrees, the
design models polynomial bounds directly in the type system using Rust's
generics:

```rust
pub struct Polynomial<T, N: Dim> {
    data: [T; N::DIM],
}
```

Here, `N` represents the maximum capacity of the polynomial (the number of
coefficients, equivalent to the maximum degree $N - 1$). This structure enforces
static size limits, allowing the compiler to determine the exact stack space
required at compile-time.

#### **3.2 Internal Storage Strategy**

Coefficients are stored in a contiguous array in **ascending order of powers**:

$$p(x) = c_0 + c_1 x + c_2 x^2 + \dots + c_{N-1} x^{N-1}$$

where `data[i]` corresponds to the coefficient $c_i$ of $x^i$.

1. **Direct Index-to-Exponent Mapping**: Storing coefficients in ascending order
   creates a 1-to-1 mapping between the array index and the exponent of $x$,
   simplifying indexing logic.
2. **Zero-Cost Padding**: When performing operations on polynomials of different
   capacities (e.g. adding a degree-2 polynomial to a degree-5 polynomial),
   lower-order coefficients remain at identical index offsets. Alignment is
   trivial and does not require shifting elements.

*Alternative Considered (Descending Power Storage)*: This is common in computer
algebra systems and MATLAB (e.g. `data[0]` is the coefficient for $x^{N-1}$).
However, descending storage introduces index-mapping overhead ($c_j$
corresponding to power $N - 1 - j$) and complicates operations when changing
capacities, as all coefficients must be shifted.

### **4. API Specification**

#### **4.1 Instantiation**

The API provides constructors for static and runtime initialization:

* **Constant Polynomial**: `pub const fn constant(val: T) -> Polynomial<T, U1>`
* **Linear Polynomial**: `pub const fn line(c0: T, c1: T) -> Polynomial<T, U2>`
* **From Coefficients Array**:
  `pub const fn from_coefficients(data: [T; N::DIM]) -> Self`
* **Functional Generation**:
  `pub fn from_fn<F>(f: F) -> Self where F: FnMut(usize) -> T`

#### **4.2 Operator Overloading**

Arithmetic operators are overloaded by implementing `core::ops` traits:

* **Addition and Subtraction**: `Add` and `Sub` are implemented for matching
  capacities.
* **Multiplication**: Implemented via convolution of the coefficient vectors:
  $$(p \cdot q)_k = \sum_{i} p_i \cdot q_{k-i}$$
  We support two multiplication interfaces:
    1. `mul_poly`: Direct static multiplication returning a product of combined
       Peano capacity bounds:
       ```rust
       impl<T, N:Dim> Polynomial<T, N> {
         pub fn mul_poly<M: Dim>(&self, other: &Polynomial<T, M>) -> 
       Polynomial<T, <<N as DimAdd<M>>::Output as DimSub<U1>>::Output>
         where
           N: DimAdd<M>,
           <N as DimAdd<M>>::Output: DimSub<U1>, { /* ... */ }
       }
       ```
    2. `mul_with_conv`: Accepts a generic implementation of the `Convolution<T>`
       trait (already implemented). This decouples the polynomial representation
       from specific DSP
       algorithms, allowing the user to pass hardware-accelerated DSP
       convolution (e.g., utilizing circular buffers or circular FFT).

#### **4.3 Core Operations**

* **Evaluation via Horner's Method**:
  $$p(x) = c_0 + x(c_1 + x(c_2 + \dots))$$
  Horner's method is mathematically optimal, requiring exactly $N-1$
  multiplications and additions, and offers superior numerical stability.
* **Long Division (`div_rem`)**: Computes the quotient $q(x)$ and
  remainder $r(x)$ such that $p(x) = d(x) \cdot q(x) + r(x)$. Output capacities
  must be specified as type-level dimensions, returning a `Result` to handle
  division-by-zero or capacity mismatch:
  ```rust
  impl<T, N: Dim> Polynomial<T, N> {
    pub fn div_rem<M: Dim, Q: Dim, R: Dim>(
      &self,
      divisor: &Polynomial<T, M>,
    ) -> Result<(Polynomial<T, Q>, Polynomial<T, R>), DivisionError> { /* ... */ }
  }

  ```
* **Differentiation & Integration**: Returns the analytical derivative or
  integral, bounded using type-level Peano math traits:
  ```rust
  impl<T, N: Dim> Polynomial<T, N> {
    pub fn derivative(&self) -> Polynomial<T, <N as DimSub<U1>>::Output>
    where
      N: DimSub<U1> { /* ... */ }

    pub fn integral(&self, constant: T) -> Polynomial<T, <N as DimAdd<U1>>::Output>
    where
      N: DimAdd<U1> { /* ... */ }
  }
  ```

---

### **5. Error Handling & State Management**

#### **5.1 Compile-Time Validation**

Operations that are mathematically constrained (such as the degree change in
multiplication and differentiation) resolve their capacities at compile-time.
Rust's type checker prevents compiling code with mismatched polynomial sizes.

#### **5.2 Runtime Fallbacks**

When division is performed, the divisor's effective degree might be zero (
division by zero) or the allocated quotient/remainder capacities ($Q$ and $R$)
might be mathematically insufficient. Instead of raising a panic, `div_rem`
returns a `Result<(Polynomial<T, Q>, Polynomial<T, R>), DivisionError>`, letting
the caller handle edge cases gracefully without halting the control loop.

---

### **6. Testing and Validation Framework**

#### **6.1 Standard Test Harness (CI & Coverage)**

Although the target environment is `#![no_std]`, the test suite is compiled and
run on the host using the standard `std` test harness. This enables tools like
`tarpaulin` to run coverage assessments, ensuring that all coefficient
arithmetic, division boundaries, and evaluation paths are fully verified.

#### **6.2 Unit Testing in `no_std`**

We run target-specific tests using a minimal custom test harness on QEMU or
microcontrollers (e.g., ARM Cortex-M4) to guarantee compatibility with
bare-metal target behavior.

#### **6.3 Property-Based Testing**

Integration with `proptest` automatically validates algebraic identities:

* Commutativity: $P + Q = Q + P$
* Distributivity: $P \cdot (Q + R) = P \cdot Q + P \cdot R$
* Division verification: $P = Q \cdot D + R$ (evaluating equality across a
  random range of input variables).

---

### **7. Performance and Resource Considerations**

#### **7.1 Examples and Bloat Testing**

To ensure that the polynomial type introduces no hidden code bloat or
compiler-injected heap dependencies, we compile mathematical examples located in
`examples/numerical_methods`. These binaries are analyzed with bloat checking
tools to ensure minimal flash footprint and deterministic stack frame sizes on
embedded targets.

#### **7.2 Stack Memory limits**

Large polynomials allocated on the stack risk overflows. The library enforces
sensible maximum sizes (e.g., restricting polynomial arrays to a maximum of 64
or 128 elements under common settings) to guarantee safety on thin
microcontrollers.

#### **7.3 Compiler Optimizations**

Horner's method is implemented in `BasicSubPrograms::polyeval` using flat-slice
iteration to encourage compiler auto-vectorization and loop unrolling, bypassing
unnecessary index checks.

---

### **8. Development Plan**

Below is the estimated implementation timeline and effort breakdown for the
`Polynomial` type:

| Task / Feature                         | Description                                                                                                                    | Estimated Effort |
|:---------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------|:-----------------|
| **Phase 1: Basic Storage & Types**     | Define `Polynomial` struct, ascending coefficient storage, basic const constructors (`constant`, `line`, `from_coefficients`). | 1.0 Day          |
| **Phase 2: Basic Arithmetic & Horner** | Implement `Add` and `Sub` traits, Horner's evaluation method, and degree utility methods.                                      | 1.5 Days         |
| **Phase 3: Calculus Operations**       | Implement analytical differentiation and integration traits.                                                                   | 1.0 Day          |
| **Phase 4: Multiplication & Division** | Implement CPU-based convolution (`mul_poly`) and long division (`div_rem`) returning `Result`.                                 | 2.0 Days         |
| **Phase 5: Testing & Bloat Analysis**  | Set up `proptest` property suites, std/no_std test runs, and verify binary size in `examples/numerical_methods`.               | 1.5 Days         |

