//! Common Digital Signal Processing Operations
//!

use crate::math::{num_traits::Real, storage::StaticStorage};

use crate::math::complex_num::Complex;
use core::ops::Neg;

/// Trait for Fast Fourier Transform (FFT) operations.
///
/// This trait defines the interface for performing FFT and Inverse FFT (IFFT).
/// Implementations can be backed by hardware accelerators or software libraries.
///
/// # Generic Arguments
/// * `T` - The numeric type of the elements (must implement `Real`).
pub trait FFT<T: 'static + Real + Neg<Output = T>> {
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
    fn fft<
        const N: usize,
        U: StaticStorage<T>,
        F: StaticStorage<Complex<T>>,
    >(
        input: &U,
        output: &mut F,
    ) {
        debug_assert!(N.is_power_of_two(), "FFT length must be a power of two");

        let in_ptr = input.get_ptr(); // Now points to Complex<T>
        let ptr = output.get_mut_ptr(); // Now points to Complex<T>
        let two_pi = T::PI * T::TWO;

        // 1. Bit-reversal permutation
        // This reorders the input so the butterfly stages can be done in-place.
        let mut j = 0;
        for i in 0..N {
            if i < j {
                unsafe {
                    ptr.add(i)
                        .write(Complex::new(in_ptr.add(j).read(), T::ZERO));
                    ptr.add(j)
                        .write(Complex::new(in_ptr.add(i).read(), T::ZERO));
                }
            }
            let mut m = N >> 1;
            while m >= 1 && j >= m {
                j -= m;
                m >>= 1;
            }
            j += m;
        }

        // 2. Cooley-Tukey Radix-2 Butterfly
        // Processes the data in log2(N) stages.
        let mut stage_len = 1;
        while stage_len < N {
            let step = stage_len << 1;

            // Calculate the angular step for this stage
            let angle = -two_pi.clone() / T::from_usize(step);
            let w_step = Complex::new(angle.clone().cos(), angle.sin());

            for m in (0..N).step_by(step) {
                let mut w = Complex::new(T::ONE, T::ZERO);

                for i in 0..stage_len {
                    let even_idx = m + i;
                    let odd_idx = m + i + stage_len;

                    unsafe {
                        // Butterfly calculation:
                        // Even = Even + W * Odd
                        // Odd  = Even - W * Odd
                        let even = ptr.add(even_idx).read();
                        let odd = ptr.add(odd_idx).read();

                        let twiddled_odd = w.clone() * odd.clone();

                        ptr.add(even_idx)
                            .write(even.clone() + twiddled_odd.clone());
                        ptr.add(odd_idx).write(even - twiddled_odd.clone());
                    }

                    // Update twiddle factor for the next element in the group
                    w = w * w_step.clone();
                }
            }
            stage_len = step;
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
    fn ifft<const N: usize, U: StaticStorage<T>, F: StaticStorage<T>>(
        input: &U,
        output: &mut F,
    ) {
        debug_assert!(N.is_power_of_two(), "Length must be power of two");

        let n_t = T::from_const::<N>();

        let in_ptr = input.get_ptr();
        let out_ptr = output.get_mut_ptr();

        let pi = T::PI;
        let two = T::TWO;

        for m in 0..N {
            let m_t = T::from_usize(m);
            let mut sum = unsafe { in_ptr.add(m).read().clone() };

            // k = N/2 term
            unsafe {
                if m % 2 == 0 {
                    sum = sum + in_ptr.add(N - 1).read().clone();
                } else {
                    sum = sum - in_ptr.add(N - 1).read().clone();
                }
            }

            // 0 < k < N/2 terms
            for k in 1..N / 2 {
                let k_t = T::from_usize(k);
                let angle =
                    two.clone() * pi.clone() * k_t * m_t.clone() / n_t.clone();
                let c = angle.clone().cos();
                let s = angle.sin();

                unsafe {
                    let re = in_ptr.add(2 * k - 1).read().clone();
                    let im = in_ptr.add(2 * k).read().clone();
                    // Add 2 * (re * c - im * s)
                    sum = sum + two.clone() * (re * c - im * s);
                }
            }

            unsafe {
                out_ptr.add(m).write(sum / n_t.clone());
            }
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
    fn convolve_input(_input: &[T], _kernel: &[T], _output: &mut [T]) {
        // unimplemented!("Must implement this before merging DSP traits.")
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
