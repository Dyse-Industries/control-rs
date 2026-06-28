# State-Space Design Document

**Implementation Order:** 4  
**Estimated Time:** 1.0 days

## 1. Context and Objective

State-space models describe linear, time-invariant (LTI) systems in the time domain using vector-valued first-order differential equations:
$$\dot{x}(t) = A x(t) + B u(t)$$
$$y(t) = C x(t) + D u(t)$$

We implement `StateSpace<T, const S: usize, const I: usize, const O: usize>` representing state-space models with statically sized arrays for the matrices $A$, $B$, $C$, and $D$. This document details ZOH discretization, model conversion from transfer functions, and eigenvalue-based stability validation.

---

## 2. Core Mechanics

### 2.1. Struct Definition

```rust
//! src/state_space/mod.rs

/// An LTI state-space system represented by statically sized arrays.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct StateSpace<T, const S: usize, const I: usize, const O: usize> {
    /// State transition matrix A (S x S)
    pub a: [[T; S]; S],
    /// Input matrix B (S x I)
    pub b: [[T; I]; S],
    /// Output matrix C (O x S)
    pub c: [[T; S]; O],
    /// Feedthrough matrix D (O x I)
    pub d: [[T; I]; O],
}
```

### 2.2. Zero-Order Hold (ZOH) Discretization

To discretize a continuous state-space model with a sample period $\Delta t$:
$$x_{k+1} = A_d x_k + B_d u_k$$
we use the augmented block matrix exponentiation method, which handles singular $A$ matrices stably.
1. Construct the $(S+I) \times (S+I)$ matrix:
   $$M = \begin{pmatrix} A \Delta t & B \Delta t \\ 0 & 0 \end{pmatrix}$$
2. Compute the matrix exponential $E = e^{M}$ using scaling-and-squaring in `la::expm`:
   $$E = \begin{pmatrix} A_d & B_d \\ 0 & I_I \end{pmatrix}$$
3. Extract the submatrices $A_d = E[0..S, 0..S]$ and $B_d = E[0..S, S..S+I]$.
4. The output matrices remain unchanged: $C_d = C$, $D_d = D$.

```rust
impl<T, const S: usize, const I: usize, const O: usize> StateSpace<T, S, I, O>
where
    T: Copy + crate::math::num_traits::Real,
{
    /// Discretizes the continuous state-space model using Zero-Order Hold (ZOH).
    pub fn discretize(&self, dt: T) -> Self {
        const AUG: usize = S + I;
        let mut m = [T::ZERO; AUG * AUG];

        // Fill M = [A*dt, B*dt; 0, 0]
        for r in 0..S {
            for c in 0..S {
                m[r * AUG + c] = self.a[r][c] * dt;
            }
            for c in 0..I {
                m[r * AUG + S + c] = self.b[r][c] * dt;
            }
        }

        // Compute augmented matrix exponential
        let mut e = [T::ZERO; AUG * AUG];
        crate::math::la::expm::<T, AUG>(&m, &mut e);

        // Extract Ad and Bd
        let mut ad = [[T::ZERO; S]; S];
        let mut bd = [[T::ZERO; I]; S];
        for r in 0..S {
            for c in 0..S {
                ad[r][c] = e[r * AUG + c];
            }
            for c in 0..I {
                bd[r][c] = e[r * AUG + S + c];
            }
        }

        Self {
            a: ad,
            b: bd,
            c: self.c,
            d: self.d,
        }
    }
}
```

### 2.3. Model Conversion: tf2ss (Control Canonical Form)

For a single-input single-output (SISO) transfer function:
$$H(s) = \frac{b_{n-1} s^{n-1} + \dots + b_1 s + b_0}{s^n + a_{n-1} s^{n-1} + \dots + a_1 s + a_0}$$
we construct the **Control Canonical Form (CCF)** state-space model of dimension $S = D-1$ (where $D$ is the denominator polynomial size, and $N \le D$).

```rust
use crate::transfer_function::StaticTransferFunction;

/// Converts a TransferFunction directly to a StateSpace model.
pub fn tf2ss<T, const N: usize, const D: usize>(
    tf: &StaticTransferFunction<T, N, D>,
) -> StateSpace<T, {D - 1}, 1, 1>
where
    T: Copy + crate::math::num_traits::Real,
{
    let monic = tf.as_monic();
    let mut a = [[T::ZERO; D - 1]; D - 1];
    let mut b = [[T::ZERO; 1]; D - 1];
    let mut c = [[T::ZERO; D - 1]; 1];
    let mut d = [[T::ZERO; 1]; 1];

    // CCF: A matrix upper-shifts inputs, bottom row holds negative denominator coefficients
    for i in 0..(D - 2) {
        a[i][i + 1] = T::ONE;
    }
    for j in 0..(D - 1) {
        a[D - 2][j] = T::ZERO - monic.denominator.coeffs[j];
    }

    // CCF: B matrix is [0; ...; 0; 1]
    b[D - 2][0] = T::ONE;

    // CCF: C and D matrices map numerator coefficients
    // If TF is proper but not strictly proper (N = D), D[0][0] is the leading numerator coefficient b_n
    let b_n = if N == D { monic.numerator.coeffs[D - 1] } else { T::ZERO };
    d[0][0] = b_n;

    for j in 0..(D - 1) {
        let b_j = if j < N { monic.numerator.coeffs[j] } else { T::ZERO };
        let a_j = monic.denominator.coeffs[j];
        c[0][j] = b_j - b_n * a_j;
    }

    StateSpace { a, b, c, d }
}
```

### 2.4. Stability Validation

A state-space system is stable if and only if all eigenvalues of its system matrix $A$ lie in the stable region:
- **Continuous System:** Eigenvalues must lie in LHP (real part $< 0$).
- **Discrete System:** Eigenvalues must lie inside the unit circle (magnitude $< 1.0$).

```rust
impl<T, const S: usize, const I: usize, const O: usize> StateSpace<T, S, I, O>
where
    T: Copy + crate::math::num_traits::Real,
{
    /// Evaluates stability by computing the eigenvalues of matrix A.
    pub fn is_stable(&self, discrete: bool) -> bool {
        // Flatten A to a row-major 1D array
        let mut a_flat = [T::ZERO; S * S];
        for r in 0..S {
            for c in 0..S {
                a_flat[r * S + c] = self.a[r][c];
            }
        }

        // Compute eigenvalues
        let eigenvalues = crate::math::la::eigenvalues::<T, S>(&a_flat, 100, None);

        // Check stability condition
        if discrete {
            eigenvalues.iter().all(|ev| ev.norm() < T::ONE)
        } else {
            eigenvalues.iter().all(|ev| ev.re < T::ZERO)
        }
    }
}
```

---

## 3. Usage Example

```rust
use control_rs::state_space::StateSpace;

fn main() {
    // Marginally stable oscillator matrix A
    let a = [[0.0, 1.0], [-1.0, 0.0]];
    let b = [[0.0], [1.0]];
    let c = [[1.0, 0.0]];
    let d = [[0.0]];
    let sys = StateSpace::new(a, b, c, d);

    // Continuous check should be false (poles at s = +/- j, not strictly in LHP)
    assert!(!sys.is_stable(false));
}
```
