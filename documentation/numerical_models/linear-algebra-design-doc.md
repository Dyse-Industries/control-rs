# Linear Algebra Abstraction Design Document

**Implementation Order:** 1  
**Estimated Time:** 1.0 days

## 1. Context and Objective

To support advanced control system computations—such as finding polynomial roots, checking state-space system stability, and fitting curves using least-squares—the library needs core matrix decomposition and linear algebra utilities.

Rather than exposing unsafe or low-level raw BLAS subprograms from `subprograms.rs` directly to the models, we introduce a dedicated `la` (linear algebra) module. This module provides a high-level, safe, and generic interface for:
- **Matrix Multiplication wrappers** over Level 3 BLAS (`GEMM`).
- **QR Eigenvalue Decomposition** (via the QR algorithm) to compute all eigenvalues of a real matrix.
- **Singular Value Decomposition (SVD)** (via the one-sided Jacobi/Hestenes SVD algorithm) to solve least-squares problems.
- **Matrix Exponential** (via scaling-and-squaring + Taylor series) to discretize systems.

---

## 2. Core Mechanics

### 2.1. The la Module Interface

All linear algebra algorithms will be generic over the precision type `T: Real` and will utilize the basic linear algebra subprograms (`level3::GEMM` and `level2::GEMV`) for performance where applicable.

```rust
//! src/math/la.rs

use crate::math::{
    num_traits::Real,
    subprograms::{level3::GEMM, level2::GEMV},
};

/// High-level wrapper for matrix multiplication: C = alpha * A * B + beta * C.
/// Matrices are stored in row-major order.
pub fn mat_mul<T, B, const M: usize, const N: usize, const K: usize>(
    alpha: T,
    a: &[T; M * K],
    b: &[T; K * N],
    beta: T,
    c: &mut [T; M * N],
) where
    T: Real,
    B: GEMM<T>,
{
    B::gemm(alpha, a, b, beta, c, M, N, K);
}
```

### 2.2. QR Eigenvalue Decomposition

The QR algorithm finds the eigenvalues of a square matrix $A$ by iteratively performing QR decompositions:
$$A_{k} = Q_k R_k \implies A_{k+1} = R_k Q_k$$
For real matrices with complex eigenvalues, the matrix converges to a Schur form (quasi-upper triangular, with 1x1 and 2x2 diagonal blocks). To keep the implementation simple, generic, and robust in `no_std` environments, we cast the real companion/system matrix to complex numbers and run the **Complex QR Algorithm** using Householder reflections or Givens rotations.

```rust
use crate::math::complex_num::Complex;

/// Computes all eigenvalues of a square matrix.
/// Returns a vector of complex eigenvalues.
pub fn eigenvalues<T, const N: usize>(
    matrix: &[T; N * N],
    max_iterations: usize,
    tolerance: Option<T>,
) -> [Complex<T>; N]
where
    T: Real;
```

### 2.3. One-Sided Jacobi (Hestenes) SVD

The Hestenes SVD algorithm computes $A = U \Sigma V^T$ for an $M \times N$ matrix by performing Jacobi rotations on columns to make them orthogonal.
- **Configurable Tolerance:** The threshold for convergence is configurable, defaulting to $100 \times \text{epsilon}$.

```rust
/// Computes the Singular Value Decomposition (SVD) of an M x N matrix.
/// 
/// Returns:
/// - `u`: Left singular vectors (M x N orthogonal matrix, stored in-place in A)
/// - `sigma`: Singular values (N-dimensional vector)
/// - `v`: Right singular vectors (N x N orthogonal matrix)
pub fn svd<T, const M: usize, const N: usize>(
    matrix_a: &mut [T; M * N], // Mutated to hold U * Sigma
    matrix_v: &mut [T; N * N], // Output V
    sigma: &mut [T; N],       // Output Singular Values
    max_iterations: usize,
    tolerance: Option<T>,
) -> Result<(), crate::math::ArithmeticError>
where
    T: Real;
```

### 2.4. Matrix Exponential (ZOH Discretization)

We compute $e^{M}$ for a square matrix $M$ using the **scaling and squaring** method:
1. Scale the matrix by $2^{-p}$ such that the matrix norm $\|M / 2^p\| < 0.5$.
2. Compute the Taylor series approximation of $e^{M/2^p}$.
3. Square the result $p$ times.

```rust
/// Computes the matrix exponential of a square matrix.
pub fn expm<T, const N: usize>(
    matrix: &[T; N * N],
    output: &mut [T; N * N],
) where
    T: Real;
```

---

## 3. Usage Example

### Solving Least-Squares Curve Fitting ($A x = b$) using SVD

```rust
fn solve_least_squares<T, const M: usize, const N: usize>(
    a: &[T; M * N],
    b: &[T; M],
    x: &mut [T; N],
) where
    T: Real,
{
    let mut u = *a;
    let mut v = [T::ZERO; N * N];
    let mut sigma = [T::ZERO; N];

    // Compute SVD
    svd::<T, M, N>(&mut u, &mut v, &mut sigma, 100, None).unwrap();

    // Solve for x using pseudo-inverse: x = V * Sigma^+ * U^T * b
    // ...
}
```
