# Polynomial Type (Design Document)

## 1. Context & Objective

Polynomials are fundamental to classical control systems engineering, primarily
representing transfer functions ($H(s) = \frac{B(s)}{A(s)}$), digital filters,
and system-identification models.

The `Polynomial` model in `control-rs` provides a stack-allocated
representation of single-variable polynomials.

---

## 2. Architecture & Design Decisions

### 2.1. Ascending Power Coefficient Storage (`[T; N]`)

Coefficients are stored in a flat array of size $N$ in ascending order of
powers:
$$p(x) = c_0 + c_1 x + c_2 x^2 + \dots + c_{N-1} x^{N-1}$$
where `data[i]` corresponds directly to the coefficient $c_i$ for $x^i$.

1. **Index-to-Power Mapping**: Storing coefficients in ascending order creates a
   direct 1-to-1 mapping between the array index and the exponent of $x$.
2. **Resizing and Alignment**: When padding a polynomial to a larger capacity (
   e.g., adding higher-order terms) or adding two polynomials of different
   capacities, the lower-order coefficients remain at the same index offsets.

#### Alternatives Considered:

- *Descending Power Storage*: Standard in computer algebra systems (CAS) and
  high-level environments like MATLAB (where the first index is the leading
  coefficient). However, this adds index math complexity (e.g., $c_j$
  corresponds to power $N - 1 - j$) and complicates operations when changing
  capacities.

---

### 2.2. Evaluation via Horner's Method

Polynomial evaluation at a point $x$ is implemented using Horner's method:
$$p(x) = c_0 + x(c_1 + x(c_2 + \dots))$$

#### Rationale:

- **Numerical Stability**: Summing terms in this nested manner is highly
  resistant to precision loss compared to computing powers $x^i$ individually.
- **Complexity**: Horner's method reduces evaluation complexity to exactly $N-1$
  multiplications and $N-1$ additions, which is mathematically optimal.
- **BLAS/Kernel Level 1 Integration**: It is dispatched to a specialized
  subprogram (`BasicSubPrograms::polyeval`) allowing hardware optimization or
  vectorization on target systems.

---

### 2.3. Division Bounds and Compile-Time Safety

Polynomial long division (`div_rem`) computes a quotient $q(x)$ and
remainder $r(x)$ such that:
$$p(x) = d(x) \cdot q(x) + r(x)$$

Because `control-rs` enforces `no_std` and zero heap allocation, `div_rem`
requires the caller to specify the maximum capacity for the output
quotient ($Q$) and remainder ($R$) as const generics:

```rust
pub fn div_rem<const M: usize, const Q: usize, const R: usize>(
    &self,
    divisor: &Polynomial<T, M>,
) -> (Polynomial<T, Q>, Polynomial<T, R>);
```

- **No Stack Overflow**: Under a stack-allocated regime, memory allocations must
  be static.
- **Runtime Sanity Check**: The function checks the mathematical degrees of the
  inputs at runtime and panics if $Q$ or $R$ are insufficient to hold the
  division result.

---

### 2.4. Convolution-Based Multiplication

Multiplication of two polynomials is mathematically equivalent to the discrete
convolution of their coefficient vectors:
$$(p \cdot q)_k = \sum_{i} p_i \cdot q_{k-i}$$

We provide two multiplication interfaces:

1. `mul_poly`: Uses the default static CPU convolution.
2. `mul_with_conv`: Accepts a generic type `C: Convolution<T>`.

- **Decoupling Math from Implementation**: The algebraic representation of a
  polynomial is separated from the DSP algorithm used to multiply them.
- **Hardware Adaptability**: For small polynomials typical in control systems (
  orders $\le 10$), standard CPU loop convolution is highly efficient. For
  large-scale DSP processing, the user can swap in a hardware-accelerated DSP
  convolution (e.g., using circular buffers or FFT processors) by passing a
  custom type implementing the `Convolution` trait.
