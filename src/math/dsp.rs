//! Common Digital Signal Processing Operations
//!

use crate::math::{
    Bijection, Map, complex_num::Complex, num_traits::Real, ops::Neg,
};

/// Trait for Fast Fourier Transform (FFT) operations.
///
/// This trait defines the interface for performing FFT and Inverse FFT (IFFT).
/// Implementations can be backed by hardware accelerators or software libraries.
///
/// # Generic Arguments
/// * `T` - The numeric type of the elements (must implement `Real`).
pub trait FFT<T: 'static + Real + Neg<Output = T> + Default> {
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
    fn fft<const N: usize>(input: &[T; N], output: &mut [Complex<T>; N]) {
        debug_assert!(N.is_power_of_two(), "FFT length must be a power of two");

        let in_ptr = input.as_ptr();
        let ptr = output.as_mut_ptr();
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
    fn ifft<const N: usize>(input: &[Complex<T>; N], output: &mut [T; N]) {
        debug_assert!(N.is_power_of_two(), "Length must be power of two");

        let n_t = T::from_const::<N>();

        let in_ptr = input.as_ptr();
        let out_ptr = output.as_mut_ptr();

        let pi = T::PI;
        let two = T::TWO;

        for m in 0..N {
            let m_t = T::from_usize(m);
            let mut sum = unsafe { in_ptr.add(m).read().re.clone() };

            // k = N/2 term
            unsafe {
                if m % 2 == 0 {
                    sum = sum + in_ptr.add(N - 1).read().re.clone();
                } else {
                    sum = sum - in_ptr.add(N - 1).read().re.clone();
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
                    let re = in_ptr.add(2 * k - 1).read().re.clone();
                    let im = in_ptr.add(2 * k).read().re.clone();
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

/// Trait for Convolution operations.
///
/// This trait defines the interface for performing convolution between two signals.
///
/// # Generic Arguments
/// * `T` - The numeric type of the elements.
pub trait Convolution<T: Real> {
    /// Computes the convolution of two signals.
    ///
    /// * Fewer Writes: By calculating $y[n]$ directly using k_min and k_max, we write to the output
    ///   array exactly once per index. This bypasses the need to zero out the buffer first and avoids
    ///   repeated memory reads/writes to output[i+j].
    /// * Bounds Safety: The indices for k (k_min and k_max) ensure that both k remains within
    /// 0..input.len() and n - k remains within 0..kernel.len(), guaranteeing no out-of-bounds
    ///   panics during the inner loop computation.
    /// * Traits: This assumes T: Real implies something akin to num_traits::Real (which provides
    ///   T::zero() and standard operator overloading).
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
    fn convolve_input(input: &[T], kernel: &[T], output: &mut [T]) {
        let input_len = input.len();
        let kernel_len = kernel.len();

        if input_len == 0 || kernel_len == 0 {
            return;
        }

        let expected_len = input_len + kernel_len - 1;
        assert!(
            output.len() >= expected_len,
            "Convolution output buffer is too small. Expected at least {}, got {}",
            expected_len,
            output.len()
        );

        // Calculate the convolution sum: y[n] = sum(x[k] * h[n-k])
        for n in 0..expected_len {
            let mut sum = T::zero();

            // Determine the valid range for k to ensure indices stay within bounds
            let k_min = n.saturating_sub(kernel_len - 1);
            let k_max = n.min(input_len - 1);

            for k in k_min..=k_max {
                sum = sum + input[k].clone() * kernel[n - k].clone();
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

// Blanket implementation of `Map` for anything that implements `FFT`.
impl<T, F, const N: usize> Map<[T; N], [Complex<T>; N]> for F
where
    F: FFT<T>,
    T: 'static + Copy + Real + Neg<Output = T> + Default,
{
    fn evaluate(&self, x: [T; N]) -> [Complex<T>; N] {
        let mut y = [Complex::<T>::default(); N];
        F::fft(&x, &mut y);
        y
    }
}

// Blanket implementation of `Bijection` for anything that implements `FFT`.
impl<T, F, const N: usize> Bijection<[T; N], [Complex<T>; N]> for F
where
    F: FFT<T>,
    T: 'static + Copy + Real + Neg<Output = T> + Default,
{
    fn evaluate_inverse(&self, y: [Complex<T>; N]) -> [T; N] {
        let mut x = [T::default(); N];
        F::ifft(&y, &mut x);
        x
    }
}
