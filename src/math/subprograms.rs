//! # Math Subprograms
//!
//! This module provides a standardized API for common numerical subprograms,
//! abstracting over different hardware backends (e.g., NEON, CMSIS-DSP,
//! MCU-Xpresso DSP Libraries, `DSPLib`). It is organized according to the BLAS
//! (Basic Linear Algebra Subprograms) levels.
//!
//! ## Design
//!
//! - **Traits-Based API**: Functionality is defined through traits, allowing
//!   for generic implementations of algorithms that can be adapted to various
//!   data types and hardware.
//!
//! - **Backend Agnostic**: The traits do not assume a specific backend, making
//!   it possible to switch between implementations via feature flags without
//!   changing the application code.
//!
//! # Example
//!
//! This example demonstrates how to write a function that is agnostic to the hardware backend
//! by using the BLAS traits as generic bounds.
//!
//! ```
//! use control_rs::math::subprograms::level1::AXPY;
//! use core::marker::PhantomData;
//!
//! // A generic controller that uses a math backend B.
//! struct Controller<B> {
//!     _marker: PhantomData<B>,
//! }
//!
//! impl<B: AXPY<f32>> Controller<B> {
//!     fn update(&self, state: &mut [f32], input: &[f32]) {
//!         B::axpy(0.5, input, state);
//!     }
//! }
//!
//! // Mock implementation of a backend
//! struct ArmNeonBackend;
//!
//! impl AXPY<f32> for ArmNeonBackend {
//!     fn axpy(a: f32, x: &[f32], y: &mut [f32]) {
//!         // Real implementation would use NEON intrinsics here
//!         for (xi, yi) in x.iter().zip(y.iter_mut()) {
//!             *yi += a * *xi;
//!         }
//!     }
//! }
//!
//! let mut state = [1.0, 2.0];
//! let input = [4.0, 6.0];
//!
//! let controller = Controller::<ArmNeonBackend> { _marker: PhantomData };
//! controller.update(&mut state, &input);
//!
//! debug_assert_eq!(state, [3.0, 5.0]);
//! ```

#[allow(unused_imports)]
use crate::math::{
    num_traits::{Real, Scalar},
    ops::{Add, Mul},
};

/// Common Digital Signal Processing Operations
pub mod dsp {
    use super::{Real};

    use core::ops::{Neg, Mul, Div};
    use crate::math::num_types::{Const, Dim};
    use crate::math::storage::StaticStorage;

    /// Trait for Fast Fourier Transform (FFT) operations.
    ///
    /// This trait defines the interface for performing FFT and Inverse FFT (IFFT).
    /// Implementations can be backed by hardware accelerators or software libraries.
    ///
    /// # Generic Arguments
    /// * `T` - The numeric type of the elements (must implement `Real`).
    pub trait FFT<T: Real + Neg<Output=T> + Mul<Output=T> + Div<Output=T>> {
        /// Computes the forward Fast Fourier Transform.
        ///
        /// # Arguments
        /// * `input` - The input signal (time domain).
        /// * `output` - The output buffer (frequency domain).
        ///
        /// # Returns
        /// * `()` - Modifies `output` in place.
        ///
        /// # Panics
        /// * Panics if input and output lengths do not match or are not powers of two.
        ///
        /// # Safety
        /// This function does not use `unsafe` code.
        #[allow(clippy::arithmetic_side_effects)]
        fn fft<const N: usize, U: StaticStorage<Const<N>>, F: StaticStorage<Const<N>>>(input: &U, output: &mut F) {
            debug_assert!(N.is_power_of_two(), "Length must be power of two");

            let n_t = T::from_const::<N>();

            for k in 0..=N / 2 {
                let k_t = T::try_from_usize(k);
                let mut sum_re = T::zero();
                let mut sum_im = T::zero();

                for m in 0..N {
                    let m_t = T::try_from_usize(m);
                    let angle = -(T::ONE + T::ONE) * T::PI * k_t.clone() * m_t / n_t.clone();
                    let c = angle.clone().cos();
                    let s = angle.sin();
                    let xm = input[m].clone();

                    sum_re = sum_re + xm.clone() * c;
                    sum_im = sum_im + xm.clone() * s;
                }

                if k == 0 {
                    output[0] = sum_re;
                } else if k == N / 2 {
                    output[N - 1] = sum_re;
                } else {
                    output[2 * k - 1] = sum_re;
                    output[2 * k] = sum_im;
                }
            }
        }

        /// Computes the inverse Fast Fourier Transform.
        ///
        /// # Arguments
        /// * `input` - The input signal (frequency domain).
        /// * `output` - The output buffer (time domain).
        ///
        /// # Returns
        /// * `()` - Modifies `output` in place.
        ///
        /// # Panics
        /// * Panics if input and output lengths do not match or are not powers of two.
        ///
        /// # Safety
        /// This function does not use `unsafe` code.
        #[allow(clippy::arithmetic_side_effects)]
        fn ifft(input: &[T], output: &mut [T]) {
            let n = input.len();
            assert_eq!(n, output.len(), "Input and output lengths must match");
            assert!(n.is_power_of_two(), "Length must be power of two");

            let pi = T::pi();
            let two = T::one() + T::one();

            let n_t = T::try_from(n).unwrap_or(T::zero());

            for m in 0..n {
                let m_t = T::try_from(m).unwrap_or(T::zero());
                let mut sum = T::zero();

                // k = 0 term
                sum = sum + input[0].clone();

                // k = N/2 term
                if m % 2 == 0 {
                    sum = sum + input[n - 1].clone();
                } else {
                    sum = sum - input[n - 1].clone();
                }

                // 0 < k < N/2 terms
                for k in 1..n / 2 {
                    let k_t = T::try_from(k).unwrap_or(T::zero());
                    let angle = two.clone() * pi.clone() * k_t * m_t.clone() / n_t.clone();
                    let c = angle.clone().cos();
                    let s = angle.sin();

                    let re = input[2 * k - 1].clone();
                    let im = input[2 * k].clone();

                    // Add 2 * (re * c - im * s)
                    sum = sum + two.clone() * (re * c - im * s);
                }

                output[m] = sum / n_t.clone();
            }
        }
    }

    /// Trait for Laplace Transform operations.
    ///
    /// This trait defines operations related to the Laplace domain (s-domain),
    /// typically used for continuous-time system analysis.
    pub trait Laplace {
        /// Evaluates the transfer function at a specific complex frequency `s`.
        ///
        /// # Arguments
        /// * `s_real` - The real part of `s`.
        /// * `s_imag` - The imaginary part of `s`.
        ///
        /// # Returns
        /// * `(f32, f32)` - The real and imaginary parts of the response.
        ///
        /// # Safety
        /// This function does not use `unsafe` code.
        fn evaluate(s_real: f32, s_imag: f32) -> (f32, f32);
    }

    /// Trait for Convolution operations.
    ///
    /// This trait defines the interface for performing convolution between two signals.
    ///
    /// # Generic Arguments
    /// * `T` - The numeric type of the elements.
    pub trait Convolution<T: Real> {
        /// Computes the convolution of two signals.
        ///
        /// # Arguments
        /// * `input` - The input signal.
        /// * `kernel` - The convolution kernel (impulse response).
        /// * `output` - The output buffer.
        ///
        /// # Returns
        /// * `()` - Modifies `output` in place.
        ///
        /// # Panics
        /// * Panics if output length is not enough to hold the result.
        ///
        /// # Safety
        /// This function does not use `unsafe` code.
        #[allow(clippy::arithmetic_side_effects)]
        fn convolve(input: &[T], kernel: &[T], output: &mut [T]) {
            let n_in = input.len();
            let n_kern = kernel.len();
            let n_out = output.len();

            assert!(n_out >= n_in + n_kern - 1, "Output buffer too small");

            for i in 0..n_out {
                output[i] = T::zero();
            }

            for n in 0..n_out {
                let mut sum = T::zero();

                let k_min = if n >= n_in { n - n_in + 1 } else { 0 };
                let k_max = if n < n_kern { n } else { n_kern - 1 };

                if k_min <= k_max {
                    for k in k_min..=k_max {
                        sum = sum + kernel[k].clone() * input[n - k].clone();
                    }
                }
                output[n] = sum;
            }
        }
    }

    /// Trait for Continuous-time systems.
    ///
    /// This trait represents systems defined in the continuous time domain.
    pub trait Continuous<R: Real> {
        /// Discretizes the continuous system to a discrete system.
        ///
        /// # Arguments
        /// * `dt` - The sampling interval.
        ///
        /// # Returns
        /// * `()` - This is a placeholder for the discretization logic.
        ///
        /// # Safety
        /// This function does not use `unsafe` code.
        fn discretize<D: Discrete<R>>(&self, dt: f32) -> D;
    }

    /// Trait for Discrete-time systems.
    ///
    /// This trait represents systems defined in the discrete time domain.
    ///
    /// # Generic Arguments
    /// * `T` - The numeric type of the elements.
    pub trait Discrete<T: Real> {
        /// The sampling frequency of the system in Hertz.
        const SAMPLING_FREQUENCY_HZ: T;

        /// Converts the discrete system back to a continuous representation (if possible).
        ///
        /// # Returns
        /// * `()` - This is a placeholder for the reconstruction logic.
        ///
        /// # Safety
        /// This function does not use `unsafe` code.
        fn to_continuous<C: Continuous<T>>(&self) -> C;
    }
}

/// Level 1 BLAS: Vector-Vector Operations
pub mod level1 {
    use super::{Add, Mul, Real, Scalar};

    use core::iter::Sum;

    /// Trait for the AXPY operation: `y = a*x + y`.
    ///
    /// This trait abstracts the scalar-vector multiplication and addition operation.
    ///
    /// # Generic Arguments
    /// * `T` - The numeric type of the elements.
    pub trait AXPY<T: Scalar + Add<Output = T> + Mul<Output = T>> {
        /// Computes `y = a*x + y`.
        ///
        /// This function scales the vector `x` by `a` and adds the result to vector `y`.
        ///
        /// # Generic Arguments
        /// * `T` - The numeric type of the elements.
        ///
        /// # Arguments
        /// * `a` - The scalar scaling factor.
        /// * `x` - The input vector (slice).
        /// * `y` - The input/output vector (mutable slice).
        ///
        /// # Returns
        /// * `()` - This function modifies `y` in place.
        ///
        /// # Errors
        /// This function does not return a `Result`.
        ///
        /// # Panics
        /// * Panics if `x` and `y` have different lengths.
        ///
        /// # Safety
        /// This function does not use `unsafe` code.
        ///
        /// # Example
        /// ```
        /// use control_rs::math::subprograms::level1::AXPY;
        ///
        /// struct CpuAxpy;
        /// impl AXPY<f32> for CpuAxpy {}
        ///
        /// let x = [1.0, 2.0, 3.0];
        /// let mut y = [4.0, 5.0, 6.0];
        /// CpuAxpy::axpy(2.0, &x, &mut y);
        /// debug_assert_eq!(y, [6.0, 9.0, 12.0]);
        /// ```
        #[allow(clippy::arithmetic_side_effects)]
        fn axpy(a: T, x: &[T], y: &mut [T]) {
            debug_assert_eq!(x.len(), y.len());
            for (xi, yi) in x.iter().zip(y.iter_mut()) {
                *yi = yi.clone() + (a.clone() * xi.clone());
            }
        }
    }

    /// Trait for the DOT operation: dot product of two vectors.
    ///
    /// # Generic Arguments
    /// * `T` - The numeric type of the elements.
    pub trait DOT<T: Scalar + Add<Output = T> + Mul<Output = T> + Sum> {
        /// Computes the dot product of two vectors.
        ///
        /// # Generic Arguments
        /// * `T` - The numeric type of the elements.
        ///
        /// # Arguments
        /// * `x` - The first input vector (slice).
        /// * `y` - The second input vector (slice).
        ///
        /// # Returns
        /// * `T` - The dot product of `x` and `y`.
        ///
        /// # Errors
        /// This function does not return a `Result`.
        ///
        /// # Panics
        /// * Panics if `x` and `y` have different lengths.
        ///
        /// # Safety
        /// This function does not use `unsafe` code.
        ///
        /// # Example
        /// ```
        /// use control_rs::math::subprograms::level1::DOT;
        ///
        /// struct CpuDot;
        /// impl DOT<f32> for CpuDot {}
        ///
        /// let x = [1.0, 2.0, 3.0];
        /// let y = [4.0, 5.0, 6.0];
        /// let result = CpuDot::dot(&x, &y);
        /// debug_assert_eq!(result, 32.0);
        /// ```
        #[allow(clippy::arithmetic_side_effects)]
        fn dot(x: &[T], y: &[T]) -> T {
            debug_assert_eq!(x.len(), y.len());
            x.iter()
                .zip(y.iter())
                .map(|(xi, yi)| xi.clone() * yi.clone())
                .sum()
        }
    }

    /// Trait for the NRM2 operation: Euclidean norm of a vector.
    ///
    /// # Generic Arguments
    /// * `T` - The numeric type of the elements.
    pub trait NRM2<T: Real + Mul<Output = T> + Sum> {
        /// Computes the Euclidean norm of a vector.
        ///
        /// # Generic Arguments
        /// * `T` - The numeric type of the elements.
        ///
        /// # Arguments
        /// * `x` - The input vector (slice).
        ///
        /// # Returns
        /// * `T` - The Euclidean norm of `x`.
        ///
        /// # Errors
        /// This function does not return a `Result`.
        ///
        /// # Panics
        /// This function does not panic.
        ///
        /// # Safety
        /// This function does not use `unsafe` code.
        ///
        /// # Example
        /// ```
        /// use control_rs::math::subprograms::level1::NRM2;
        ///
        /// struct CpuNrm2;
        ///
        /// impl NRM2<f32> for CpuNrm2 {
        ///     fn nrm2(x: &[f32]) -> f32 {
        ///         x.iter().map(|xi| xi * xi).sum::<f32>().sqrt()
        ///     }
        /// }
        ///
        /// let x = [3.0, 4.0];
        /// let result = CpuNrm2::nrm2(&x);
        /// debug_assert_eq!(result, 5.0);
        /// ```
        #[allow(clippy::arithmetic_side_effects)]
        fn nrm2(x: &[T]) -> T {
            x.iter().map(|xi| xi.clone() * xi.clone()).sum::<T>().sqrt()
        }
    }

    /// Trait for the IAMAX operation: Index of the element with the maximum absolute value.
    ///
    /// # Generic Arguments
    /// * `T` - The numeric type of the elements.
    pub trait IAMAX<T: Real> {
        /// Finds the index of the element with the maximum absolute value.
        ///
        /// # Generic Arguments
        /// * `T` - The numeric type of the elements.
        ///
        /// # Arguments
        /// * `x` - The input vector (slice).
        ///
        /// # Returns
        /// * `usize` - The index of the element with the maximum absolute value. Returns 0 if the slice is empty.
        ///
        /// # Errors
        /// This function does not return a `Result`.
        ///
        /// # Panics
        /// This function does not panic.
        ///
        /// # Safety
        /// This function does not use `unsafe` code.
        ///
        /// # Example
        /// ```
        /// use control_rs::math::subprograms::level1::IAMAX;
        ///
        /// struct CpuIamax;
        ///
        /// impl IAMAX<f32> for CpuIamax {
        ///     fn iamax(x: &[f32]) -> usize {
        ///         if x.is_empty() { return 0; }
        ///         x.iter()
        ///          .enumerate()
        ///          .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
        ///          .map(|(i, _)| i)
        ///          .unwrap_or(0)
        ///     }
        /// }
        ///
        /// let x = [1.0, -5.0, 3.0];
        /// let result = CpuIamax::iamax(&x);
        /// debug_assert_eq!(result, 1);
        /// ```
        fn iamax(x: &[T]) -> usize {
            x.iter()
                .enumerate()
                .max_by(|x, y| {
                    x.1.clone()
                        .abs()
                        .partial_cmp(&y.1.clone().abs())
                        .unwrap_or(core::cmp::Ordering::Equal)
                })
                .map_or(x.len(), |(i, _)| i)
        }
    }
}

/// Level 2 BLAS: Matrix-Vector Operations
pub mod level2 {
    use super::{Add, Mul, Scalar};

    use core::iter::Sum;

    /// Trait for the GEMV operation: `y = alpha*A*x + beta*y`.
    ///
    /// # Generic Arguments
    /// * `T` - The numeric type of the elements.
    pub trait GEMV<T: Scalar + Add<Output = T> + Mul<Output = T> + Sum> {
        /// Computes `y = alpha*A*x + beta*y`.
        ///
        /// Performs one of the matrix-vector operations:
        /// `y := alpha * A * x + beta * y`
        ///
        /// # Generic Arguments
        /// * `T` - The numeric type of the elements.
        ///
        /// # Arguments
        /// * `alpha` - The scalar alpha.
        /// * `a` - The matrix A (slice). Expected shape: `rows` x `cols` (row-major).
        /// * `x` - The vector x (slice). Expected to be of size `cols`.
        /// * `beta` - The scalar beta.
        /// * `y` - The vector y (mutable slice). Expected to be of size `rows`.
        /// * `rows` - Number of rows in matrix A.
        /// * `cols` - Number of columns in matrix A.
        ///
        /// # Returns
        /// * `()` - This function modifies `y` in place.
        ///
        /// # Errors
        /// This function does not return a `Result`.
        ///
        /// # Panics
        /// * Panics if `a.len() != rows * cols`.
        /// * Panics if `x.len() != cols`.
        /// * Panics if `y.len() != rows`.
        ///
        /// # Safety
        /// This function does not use `unsafe` code.
        ///
        /// # Example
        /// ```
        /// use control_rs::math::subprograms::level2::GEMV;
        ///
        /// struct CpuGemv;
        ///
        /// impl GEMV<f32> for CpuGemv {
        ///     fn gemv(
        ///         alpha: f32,
        ///         a: &[f32],
        ///         x: &[f32],
        ///         beta: f32,
        ///         y: &mut [f32],
        ///         rows: usize,
        ///         cols: usize,
        ///     ) {
        ///         debug_assert_eq!(a.len(), rows * cols);
        ///         debug_assert_eq!(x.len(), cols);
        ///         debug_assert_eq!(y.len(), rows);
        ///
        ///         let y_orig = y.to_vec();
        ///         for i in 0..rows {
        ///             let mut dot = 0.0;
        ///             for j in 0..cols {
        ///                 dot += a[i * cols + j] * x[j];
        ///             }
        ///             y[i] = alpha * dot + beta * y_orig[i];
        ///         }
        ///     }
        /// }
        ///
        /// let a = [1.0, 2.0, 3.0, 4.0]; // 2x2 matrix
        /// let x = [1.0, 1.0];
        /// let mut y = [0.0, 0.0];
        /// // y = 1.0 * A * x + 0.0 * y
        /// // y[0] = 1*1 + 2*1 = 3
        /// // y[1] = 3*1 + 4*1 = 7
        /// CpuGemv::gemv(1.0, &a, &x, 0.0, &mut y, 2, 2);
        /// debug_assert_eq!(y, [3.0, 7.0]);
        /// ```
        #[allow(
            clippy::too_many_arguments,
            clippy::many_single_char_names,
            clippy::arithmetic_side_effects
        )]
        fn gemv(
            alpha: T,
            a: &[T],
            x: &[T],
            beta: T,
            y: &mut [T],
            rows: usize,
            cols: usize,
        ) {
            debug_assert_eq!(a.len(), rows * cols);
            debug_assert_eq!(x.len(), cols);
            debug_assert_eq!(y.len(), rows);

            for (row, yi) in a.chunks_exact(cols).zip(y.iter_mut()) {
                let dot: T = row
                    .iter()
                    .zip(x.iter())
                    .map(|(aij, xj)| aij.clone() * xj.clone())
                    .sum();
                *yi = alpha.clone() * dot.clone() + beta.clone() * yi.clone();
            }
        }
    }
}

/// Level 3 BLAS: Matrix-Matrix Operations
pub mod level3 {
    use super::{Add, Mul, Scalar};

    /// Trait for the GEMM operation: `C = alpha*A*B + beta*C`.
    ///
    /// # Generic Arguments
    /// * `T` - The numeric type of the elements.
    pub trait GEMM<T: Scalar + Add<Output = T> + Mul<Output = T>> {
        /// Computes `C = alpha*A*B + beta*C`.
        ///
        /// Performs the matrix-matrix operation:
        /// `C := alpha * A * B + beta * C`
        ///
        /// # Generic Arguments
        /// * `T` - The numeric type of the elements.
        ///
        /// # Arguments
        /// * `alpha` - The scalar alpha.
        /// * `a` - The matrix A (slice). Expected shape: `m` x `k`.
        /// * `b` - The matrix B (slice). Expected shape: `k` x `n`.
        /// * `beta` - The scalar beta.
        /// * `c` - The matrix C (mutable slice). Expected shape: `m` x `n`.
        /// * `m` - Number of rows in matrix A and C.
        /// * `n` - Number of columns in matrix B and C.
        /// * `k` - Number of columns in matrix A and rows in matrix B.
        ///
        /// # Returns
        /// * `()` - This function modifies `c` in place.
        ///
        /// # Errors
        /// This function does not return a `Result`.
        ///
        /// # Panics
        /// * Panics if `a.len() != m * k`.
        /// * Panics if `b.len() != k * n`.
        /// * Panics if `c.len() != m * n`.
        ///
        /// # Safety
        /// This function does not use `unsafe` code.
        ///
        /// # Example
        /// ```
        /// use control_rs::math::subprograms::level3::GEMM;
        ///
        /// struct CpuGemm;
        ///
        /// impl GEMM<f32> for CpuGemm {
        ///     fn gemm(
        ///         alpha: f32,
        ///         a: &[f32],
        ///         b: &[f32],
        ///         beta: f32,
        ///         c: &mut [f32],
        ///         m: usize,
        ///         n: usize,
        ///         k: usize,
        ///     ) {
        ///         debug_assert_eq!(a.len(), m * k);
        ///         debug_assert_eq!(b.len(), k * n);
        ///         debug_assert_eq!(c.len(), m * n);
        ///
        ///         let c_orig = c.to_vec();
        ///         for i in 0..m {
        ///             for j in 0..n {
        ///                 let mut dot = 0.0;
        ///                 for p in 0..k {
        ///                     dot += a[i * k + p] * b[p * n + j];
        ///                 }
        ///                 c[i * n + j] = alpha * dot + beta * c_orig[i * n + j];
        ///             }
        ///         }
        ///     }
        /// }
        ///
        /// let a = [1.0, 2.0, 3.0, 4.0]; // 2x2
        /// let b = [1.0, 0.0, 0.0, 1.0]; // 2x2 identity
        /// let mut c = [0.0; 4];
        /// // C = 1.0 * A * I + 0.0 * C = A
        /// CpuGemm::gemm(1.0, &a, &b, 0.0, &mut c, 2, 2, 2);
        /// debug_assert_eq!(c, [1.0, 2.0, 3.0, 4.0]);
        /// ```
        #[allow(
            clippy::too_many_arguments,
            clippy::many_single_char_names,
            clippy::arithmetic_side_effects
        )]
        fn gemm(
            alpha: T,
            a: &[T],
            b: &[T],
            beta: T,
            c: &mut [T],
            m: usize,
            n: usize,
            k: usize,
        ) {
            debug_assert_eq!(a.len(), m * k);
            debug_assert_eq!(b.len(), k * n);
            debug_assert_eq!(c.len(), m * n);

            for (c_row, a_row) in c.chunks_exact_mut(n).zip(a.chunks_exact(k)) {
                for val in c_row.iter_mut() {
                    *val = val.clone() * beta.clone();
                }

                for (a_val, b_row) in a_row.iter().zip(b.chunks_exact(n)) {
                    let scalar = alpha.clone() * a_val.clone();
                    for (c_elem, b_elem) in c_row.iter_mut().zip(b_row.iter()) {
                        *c_elem =
                            c_elem.clone() + (scalar.clone() * b_elem.clone());
                    }
                }
            }
        }
    }
}

/// A naive implementation of the BLAS traits for `f32`.
///
/// This implementation uses standard Rust loops and iterators. It does not utilize any
/// hardware acceleration (SIMD, DSP instructions, etc.). It serves as a reference implementation
/// and a fallback for targets without specialized hardware support.
///
/// # Safety
/// This struct and its trait implementations do not use `unsafe` code.
pub struct BasicSubProgramsF32;

/// A naive implementation of the BLAS + DSP traits for `f64`.
///
/// This implementation uses standard Rust loops and iterators. It does not use any
/// hardware acceleration. It serves as a reference implementation and a fallback for targets
/// without specialized hardware support.
///
/// # Safety
/// This struct and its trait implementations do not use `unsafe` code.
pub struct BasicSubProgramsF64;

impl dsp::FFT<f32> for BasicSubProgramsF32 {}

impl level1::AXPY<f32> for BasicSubProgramsF32 {}
impl level1::DOT<f32> for BasicSubProgramsF32 {}
impl level1::NRM2<f32> for BasicSubProgramsF32 {}
impl level1::IAMAX<f32> for BasicSubProgramsF32 {}
impl level2::GEMV<f32> for BasicSubProgramsF32 {}
impl level3::GEMM<f32> for BasicSubProgramsF32 {}

impl dsp::FFT<f64> for BasicSubProgramsF64 {}
impl level1::AXPY<f64> for BasicSubProgramsF64 {}
impl level1::DOT<f64> for BasicSubProgramsF64 {}
impl level1::NRM2<f64> for BasicSubProgramsF64 {}
impl level1::IAMAX<f64> for BasicSubProgramsF64 {}
impl level2::GEMV<f64> for BasicSubProgramsF64 {}
impl level3::GEMM<f64> for BasicSubProgramsF64 {}