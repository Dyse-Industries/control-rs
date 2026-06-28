# Transfer Function Design Document

**Implementation Order:** 3  
**Estimated Time:** 1.0 days

## 1. Context and Objective

Transfer functions model linear, time-invariant (LTI) systems in the frequency (Laplace) domain. They represent the relationship between input and output as a ratio of polynomials:
$$H(s) = \frac{\text{Numerator}(s)}{\text{Denominator}(s)}$$

We implement `StaticTransferFunction<T, const N: usize, const D: usize>` representing transfer functions with statically sized numerator and denominator coefficients. This document details system analysis, stabilization, frequency response calculations, and type-safe scalar arithmetic.

---

## 2. Core Mechanics

### 2.1. Struct Definition

```rust
//! src/transfer_function/mod.rs

use crate::polynomial::Polynomial;

/// A transfer function represented by statically sized arrays for numerator and denominator.
/// Coefficients are stored from lowest degree (constant) to highest.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct StaticTransferFunction<T, const N: usize, const D: usize> {
    pub numerator: Polynomial<T, N>,
    pub denominator: Polynomial<T, D>,
}
```

### 2.2. System Analysis and Standardization

#### Monic Standardization
A transfer function is standardized by dividing all coefficients of both numerator and denominator polynomials by the leading coefficient of the denominator, ensuring the leading denominator coefficient is exactly $1.0$.

```rust
impl<T, const N: usize, const D: usize> StaticTransferFunction<T, N, D>
where
    T: Copy + crate::math::num_traits::Real,
{
    /// Standardizes the transfer function so that the leading denominator coefficient is 1.0.
    pub fn as_monic(&self) -> Self {
        let lead = self.denominator.coeffs[D - 1];
        let mut num_coeffs = self.numerator.coeffs;
        let mut den_coeffs = self.denominator.coeffs;
        for c in num_coeffs.iter_mut() { *c = *c / lead; }
        for c in den_coeffs.iter_mut() { *c = *c / lead; }
        Self {
            numerator: Polynomial { coeffs: num_coeffs },
            denominator: Polynomial { coeffs: den_coeffs },
        }
    }
}
```

#### DC Gain
The DC gain represents the steady-state gain of the system when evaluated at $s = 0$.
$$H(0) = \frac{\text{num}[0]}{\text{den}[0]}$$
If the system has an integrator (pole at origin, so $\text{den}[0] = 0$), the DC gain is infinite.

```rust
impl<T, const N: usize, const D: usize> StaticTransferFunction<T, N, D>
where
    T: Copy + crate::math::num_traits::Real,
{
    /// Computes the DC gain of the system. Returns INF or NAN if den[0] is zero.
    pub fn dc_gain(&self) -> T {
        if self.denominator.coeffs[0].is_zero() {
            T::INF
        } else {
            self.numerator.coeffs[0] / self.denominator.coeffs[0]
        }
    }
}
```

#### System Stability Check (LHP Pole Verification)
A continuous-time system is stable if and only if all poles (roots of the denominator polynomial) are located in the open Left-Half Plane (LHP), meaning their real parts are strictly less than $0$.

```rust
impl<T, const N: usize, const D: usize> StaticTransferFunction<T, N, D>
where
    T: Copy + crate::math::num_traits::Real,
{
    /// Evaluates continuous stability by verifying that all denominator poles lie in LHP.
    pub fn is_stable(&self) -> bool {
        let monic = self.as_monic();
        // 1. Find roots of monic.denominator using QR eigenvalue companion method
        let poles = monic.denominator.roots::<{D - 1}>();
        // 2. Assert all poles have real parts strictly less than zero
        poles.iter().all(|pole| pole.re < T::ZERO)
    }
}
```

### 2.3. Frequency Response (Bode Plot Computations)

To compute the frequency response across a range of angular frequencies $\omega$:
1. Substitute $s = j \omega$ (where $j$ is the imaginary unit).
2. Evaluate the complex value $H(j\omega)$ by running Horner's method on the complex representation.
3. Compute magnitude $|H(j\omega)|$ and phase $\angle H(j\omega) = \text{atan2}(\text{im}, \text{re})$.

```rust
pub struct FrequencyResponse<T, const F: usize> {
    pub frequencies: [T; F],
    pub magnitudes: [T; F],
    pub phases: [T; F],
}

impl<T, const N: usize, const D: usize> StaticTransferFunction<T, N, D>
where
    T: Copy + crate::math::num_traits::Real,
{
    /// Computes magnitude and phase across the specified frequencies.
    pub fn frequency_response<const F: usize>(&self, freqs: &[T; F]) -> FrequencyResponse<T, F> {
        let mut magnitudes = [T::ZERO; F];
        let mut phases = [T::ZERO; F];
        for (i, &w) in freqs.iter().enumerate() {
            let s = crate::math::complex_num::Complex::new(T::ZERO, w);
            let num_val = self.numerator.evaluate_complex(s);
            let den_val = self.denominator.evaluate_complex(s);
            let h = num_val / den_val;
            magnitudes[i] = h.norm();
            phases[i] = h.im.atan2(h.re);
        }
        FrequencyResponse { frequencies: *freqs, magnitudes, phases }
    }
}
```

### 2.4. Type-Safe Arithmetic

When adding a scalar $c$ to a transfer function:
$$H(s) + c = \frac{\text{Numerator}(s) + c \cdot \text{Denominator}(s)}{\text{Denominator}(s)}$$
The resulting numerator size is the maximum of the original numerator size ($N$) and denominator size ($D$). We use the `DimMax` trait to verify this at compile time.

```rust
use crate::math::num_types::{Const, Dim, DimMax};

impl<T, const N: usize, const D: usize> StaticTransferFunction<T, N, D>
where
    T: Copy + crate::math::num_traits::Real,
{
    /// Adds a scalar to the transfer function, verifying the output numerator size at compile time.
    pub fn add_scalar<const OUT_N: usize>(
        &self,
        scalar: T,
    ) -> StaticTransferFunction<T, OUT_N, D>
    where
        Const<N>: Dim,
        Const<D>: Dim,
        Const<OUT_N>: Dim,
        <Const<N> as Dim>::PeanoTypeNum: DimMax<<Const<D> as Dim>::PeanoTypeNum, Output = <Const<OUT_N> as Dim>::PeanoTypeNum>,
    {
        // 1. Scale denominator by scalar: c * den(s)
        // 2. Add to numerator (using zero-padding to length OUT_N)
        // ...
        let out_num = Polynomial { coeffs: [T::ZERO; OUT_N] };
        StaticTransferFunction {
            numerator: out_num,
            denominator: self.denominator,
        }
    }
}
```

---

## 3. Usage Example

```rust
use control_rs::transfer_function::StaticTransferFunction;

fn main() {
    // Continuous integrator model: H(s) = 1 / s
    let num = [1.0];
    let den = [0.0, 1.0];
    let sys = StaticTransferFunction::new(num, den);

    // Compute Bode response at 1 rad/s
    let response = sys.frequency_response(&[1.0]);
    assert!((response.magnitudes[0] - 1.0).abs() < 1e-5);
    assert!((response.phases[0] + std::f32::consts::FRAC_PI_2).abs() < 1e-5);
}
```
