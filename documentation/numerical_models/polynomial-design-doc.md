# Polynomial Design Document

**Implementation Order:** 2  
**Estimated Time:** 1.0 days

## 1. Context and Objective

Polynomials serve as the base mathematical representation for transfer function numerators and denominators, and are widely used in curve fitting and system identification.

We define a generic, statically sized, univariate `Polynomial<T, const N: usize>` struct. Since stable Rust does not support arbitrary const expressions in generic return signatures (e.g. `Polynomial<T, {N + M - 1}>`), we leverage the type-level Peano dimension operations (e.g. `DimAdd`, `DimSub`, `U1`, etc.) defined in `math::num_types` to constrain const generic parameters at compile time.

---

## 2. Core Mechanics

### 2.1. Struct Definition and Type Aliases

```rust
//! src/polynomial/mod.rs

use crate::math::num_types::{Dim, DimAdd, DimSub, Const, U1};

/// A generic polynomial stored as a dense array of coefficients.
/// Coefficients are ordered from lowest degree (constant) to highest.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Polynomial<T, const N: usize> {
    /// Coefficients ordered from lowest degree [c0, c1, ..., c_n]
    pub coeffs: [T; N],
}

// Convenient Type Aliases
pub type Constant<T> = Polynomial<T, 1>;
pub type Line<T> = Polynomial<T, 2>;
pub type Quadratic<T> = Polynomial<T, 3>;
pub type Cubic<T> = Polynomial<T, 4>;
pub type Quartic<T> = Polynomial<T, 5>;
pub type Quintic<T> = Polynomial<T, 6>;
```

### 2.2. Type-Safe Polynomial Arithmetic

#### Polynomial Multiplication (Convolution)
For two polynomials of size $N$ and $M$, the product requires exactly $N + M - 1$ coefficients.
```rust
impl<T, const N: usize> Polynomial<T, N>
where
    T: Copy + crate::math::num_traits::Real,
{
    /// Multiplies two polynomials, verifying output size at compile-time.
    pub fn mul<const M: usize, const OUT: usize>(
        &self,
        other: &Polynomial<T, M>,
    ) -> Polynomial<T, OUT>
    where
        Const<N>: Dim,
        Const<M>: Dim,
        Const<OUT>: Dim,
        <Const<N> as Dim>::PeanoTypeNum: DimAdd<<Const<M> as Dim>::PeanoTypeNum>,
        <<Const<N> as Dim>::PeanoTypeNum as DimAdd<<Const<M> as Dim>::PeanoTypeNum>>::Output: DimSub<U1, Output = <Const<OUT> as Dim>::PeanoTypeNum>,
    {
        let mut out_coeffs = [T::ZERO; OUT];
        // Compute convolution using crate::math::dsp::Convolution
        // ...
        Polynomial { coeffs: out_coeffs }
    }
}
```

#### Polynomial Long Division
Dividing a polynomial of size $N$ by a denominator of size $D$ yields a quotient of size $N - D + 1$ and a remainder of size $D - 1$ (for $D > 1$).
```rust
impl<T, const N: usize> Polynomial<T, N>
where
    T: Copy + crate::math::num_traits::Real,
{
    /// Performs polynomial long division, verifying sizes at compile-time.
    pub fn div_rem<const D: usize, const Q: usize, const R: usize>(
        &self,
        denom: &Polynomial<T, D>,
    ) -> (Polynomial<T, Q>, Polynomial<T, R>)
    where
        Const<N>: Dim,
        Const<D>: Dim,
        Const<Q>: Dim,
        Const<R>: Dim,
        <Const<N> as Dim>::PeanoTypeNum: DimSub<<Const<D> as Dim>::PeanoTypeNum, Output = <Const<Q> as Dim>::PeanoTypeNum>,
        <Const<D> as Dim>::PeanoTypeNum: DimSub<U1, Output = <Const<R> as Dim>::PeanoTypeNum>,
    {
        let mut q_coeffs = [T::ZERO; Q];
        let mut r_coeffs = [T::ZERO; R];
        // Perform polynomial synthetic division
        // ...
        (Polynomial { coeffs: q_coeffs }, Polynomial { coeffs: r_coeffs })
    }
}
```

### 2.3. Calculus

```rust
impl<T, const N: usize> Polynomial<T, N>
where
    T: Copy + crate::math::num_traits::Real,
{
    /// Computes the analytical derivative, verifying size at compile-time.
    pub fn derivative<const OUT: usize>(&self) -> Polynomial<T, OUT>
    where
        Const<N>: Dim,
        Const<OUT>: Dim,
        <Const<N> as Dim>::PeanoTypeNum: DimSub<U1, Output = <Const<OUT> as Dim>::PeanoTypeNum>,
    {
        let mut out_coeffs = [T::ZERO; OUT];
        for i in 1..N {
            out_coeffs[i - 1] = T::from_usize(i) * self.coeffs[i];
        }
        Polynomial { coeffs: out_coeffs }
    }

    /// Computes the indefinite integral with constant C, verifying size at compile-time.
    pub fn integral<const OUT: usize>(&self, c: T) -> Polynomial<T, OUT>
    where
        Const<N>: Dim,
        Const<OUT>: Dim,
        <Const<N> as Dim>::PeanoTypeNum: DimAdd<U1, Output = <Const<OUT> as Dim>::PeanoTypeNum>,
    {
        let mut out_coeffs = [T::ZERO; OUT];
        out_coeffs[0] = c;
        for i in 0..N {
            out_coeffs[i + 1] = self.coeffs[i] / T::from_usize(i + 1);
        }
        Polynomial { coeffs: out_coeffs }
    }
}
```

### 2.4. Root Finding

Roots are solved analytically for quadratics ($N=3$). For higher degrees, we construct the companion matrix and find its eigenvalues via QR decomposition.

```rust
impl<T, const N: usize> Polynomial<T, N>
where
    T: Copy + crate::math::num_traits::Real,
{
    /// Finds roots analytically for quadratic polynomials (size = 3).
    pub fn roots_quadratic(&self) -> [crate::math::complex_num::Complex<T>; 2];

    /// Computes polynomial roots using companion matrix eigenvalues.
    pub fn roots<const DEG: usize>(&self) -> [crate::math::complex_num::Complex<T>; DEG]
    where
        Const<N>: Dim,
        Const<DEG>: Dim,
        <Const<N> as Dim>::PeanoTypeNum: DimSub<U1, Output = <Const<DEG> as Dim>::PeanoTypeNum>;
}
```

### 2.5. Curve Fitting

Least-squares polynomial fit finds coefficients minimizing $\|V p - y\|_2$ using SVD. We require the number of points $P \ge \text{DEG} + 1$.

```rust
impl<T> Polynomial<T, 0> {
    /// Fits a polynomial of degree DEG to the given coordinates (x, y).
    pub fn fit<const P: usize, const DEG: usize>(
        x: &[T; P],
        y: &[T; P],
    ) -> Polynomial<T, DEG>
    where
        T: crate::math::num_traits::Real,
        Const<P>: Dim,
        Const<DEG>: Dim,
        <Const<DEG> as Dim>::PeanoTypeNum: DimAdd<U1>,
        <Const<P> as Dim>::PeanoTypeNum: DimSub<<<Const<DEG> as Dim>::PeanoTypeNum as DimAdd<U1>>::Output>,
    {
        // 1. Construct Vandermonde matrix V of size P x DEG
        // 2. Solve V * p = y using SVD pseudo-inverse
        // ...
        Polynomial { coeffs: [T::ZERO; DEG] }
    }
}
```

---

## 3. Usage Example

```rust
use control_rs::polynomial::{Polynomial, Quadratic, Line};

fn main() {
    let p_a = Line::new(2.0, 3.0); // 2x + 3
    let p_b = Line::new(1.0, -1.0); // x - 1

    // Product of degree 1 (Line) and degree 1 (Line) is degree 2 (Quadratic)
    // Enforced at compile-time: size is 2 + 2 - 1 = 3
    let p_c: Quadratic<f32> = p_a.mul::<2, 3>(&p_b);
    assert_eq!(p_c.coeffs, [-3.0, 1.0, 2.0]); // 2x^2 + x - 3
}
```
