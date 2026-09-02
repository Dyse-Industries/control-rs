//! Common Digital Signal Processing Operations

use crate::math::{
    Bijection, ConversionError, ConversionResult, Map,
    complex_num::Complex,
    num_traits::{Float, Scalar},
    ops::Neg,
    storage,
};

type ComplexArrayMut<T, const N: usize> = [Complex<T>; N];
type ComplexSliceMut<'a, T, const N: usize> = &'a mut [Complex<T>; N];
type RealSlice<'a, T, const N: usize> = &'a [T; N];
type ComplexSlice<'a, T, const N: usize> = &'a [Complex<T>; N];
type RealSliceMut<'a, T, const N: usize> = &'a mut [T; N];

/// Trait for Fast Fourier Transform (FFT) operations.
///
/// This trait defines the interface for performing FFT and Inverse FFT (IFFT).
/// Implementations can be backed by hardware accelerators or software libraries.
///
/// # Generic Arguments
/// * `T` - The numeric type of the elements (must implement `Float`).
///
/// # Contract
/// `N` must be a power of two. Length is a `debug_assert` on every entry
/// point (`fft`, `fft_complex`, `ifft`); it is not a [`Result`] path.
pub trait FFT<T: 'static + Clone + Float + Neg<Output = T> + Default> {
    /// Computes the forward Fast Fourier Transform into `output` (frequency domain).
    /// # Panics
    /// Debug builds panic if `N` is not a power of two.
    fn fft<const N: usize>(
        input: RealSlice<'_, T, N>,
        output: ComplexSliceMut<'_, T, N>,
    ) {
        debug_assert!(N.is_power_of_two(), "FFT length must be a power of two");

        for (i, val) in input.iter().enumerate() {
            if let Some(out) = output.get_mut(i) {
                *out = Complex::new(val.clone(), T::ZERO);
            }
        }

        Self::fft_complex(output);
    }

    /// Computes the forward Fast Fourier Transform on a complex signal, in-place.
    /// # Panics
    /// Debug builds panic if `N` is not a power of two.
    ///
    /// # Safety
    /// Uses pointer reads/writes; indices stay in `0..N` because `N` is a power
    /// of two and the butterfly loops are bounded by `stage_len` and `step`.
    // Case-by-case: Arithmetic side effects are unavoidable for Cooley-Tukey Radix-2 FFT.
    #[allow(clippy::arithmetic_side_effects)]
    fn fft_complex<const N: usize>(data: ComplexSliceMut<'_, T, N>) {
        debug_assert!(N.is_power_of_two(), "FFT length must be a power of two");

        // 1. Bit-reversal permutation
        // This reorders the input so the butterfly stages can be done in-place.
        let mut j = 0;
        for i in 0..N {
            if i < j {
                data.swap(i, j);
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
        let ptr = data.as_mut_ptr();
        let two_pi = T::PI * (T::ONE + T::ONE);
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
                        // `Even = Even + W * Odd`
                        // `Odd  = Even - W * Odd`
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

    /// Computes the inverse Fast Fourier Transform into `output` (time domain).
    ///
    /// # Panics
    /// Debug builds panic if `N` is not a power of two.
    fn ifft<const N: usize>(
        input: ComplexSlice<'_, T, N>,
        output: RealSliceMut<'_, T, N>,
    ) {
        debug_assert!(N.is_power_of_two(), "FFT length must be a power of two");

        let n_t = T::from_const::<N>();
        // # Safety: The input iter has the same number of elements as the output.
        let mut temp_output: ComplexArrayMut<T, N> = unsafe {
            storage::array_from_iterator(input.iter().map(|c| c.clone().conj()))
        };

        Self::fft_complex(&mut temp_output);

        for (i, val) in temp_output.iter().enumerate() {
            if let Some(out) = output.get_mut(i) {
                // Case-by-case: Float division is unavoidable here.
                #[allow(clippy::arithmetic_side_effects)]
                let val_divided = val.clone().conj().re / n_t.clone();
                *out = val_divided;
            }
        }
    }
}

/// Trait for Convolution operations.
///
/// This trait defines the interface for performing convolution between two signals.
///
/// # Generic Arguments
/// * `T` - The numeric type of the elements (must implement `Scalar`).
pub trait Convolution<T: Scalar> {
    /// Computes the convolution of two signals into `output`.
    ///
    /// Writes `input_len + kernel_len - 1` samples. Inner-loop indices stay in
    /// range via `k_min`/`k_max`.
    ///
    /// # Errors
    /// Returns [`ConversionError::DimensionMismatch`] if `output` is shorter than
    /// `input_len + kernel_len - 1`.
    // Performance: Direct indexing is used to bypass bounds checking in performance-critical convolution loops.
    #[allow(clippy::indexing_slicing)]
    // Case-by-case: Arithmetic side effects are unavoidable for convolution math.
    #[allow(clippy::arithmetic_side_effects)]
    fn convolve_input(
        input: &[T],
        kernel: &[T],
        output: &mut [T],
    ) -> ConversionResult<()> {
        let input_len = input.len();
        let kernel_len = kernel.len();

        if input_len == 0 || kernel_len == 0 {
            return Ok(());
        }

        let expected_len = input_len + kernel_len - 1;
        if output.len() < expected_len {
            return Err(ConversionError::DimensionMismatch);
        }

        // Calculate the convolution sum: y[n] = sum(x[k] * h[n-k])
        for n in 0..expected_len {
            let mut sum = T::ZERO;

            // Determine the valid range for k to ensure indices stay within bounds
            let k_min = n.saturating_sub(kernel_len - 1);
            let k_max = n.min(input_len - 1);

            for k in k_min..=k_max {
                sum = sum + input[k].clone() * kernel[n - k].clone();
            }

            output[n] = sum;
        }
        Ok(())
    }
}

/// A continuous-time system that can be sampled at interval `dt`.
pub trait Continuous<T: Float> {
    /// Discrete counterpart produced by [`Continuous::discretize`].
    type Discrete: Discrete<T, Continuous = Self>;

    /// Samples `self` at period `dt`.
    fn discretize(&self, dt: T) -> Self::Discrete;
}

/// A discrete-time system obtained by sampling a [`Continuous`] plant.
pub trait Discrete<T: Float> {
    /// Continuous counterpart recovered by [`Discrete::to_continuous`].
    type Continuous: Continuous<T, Discrete = Self>;

    /// Sampling period used to produce this discrete system.
    fn sampling_period(&self) -> T;

    /// Recovers the continuous plant. Sampling is not invertible; this
    /// returns the plant that was discretized, not a unique reconstruction.
    fn to_continuous(&self) -> Self::Continuous;
}

/// Reference zero-dependency pure-Rust DSP engine implementing [`FFT`] and [`Convolution`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DefaultDsp;

impl<T: 'static + Clone + Float + Neg<Output = T> + Default> FFT<T>
    for DefaultDsp
{
}
impl<T: Scalar> Convolution<T> for DefaultDsp {}

// Blanket implementation of `Map` for anything that implements `FFT`.
impl<T, F, const N: usize> Map<[T; N], [Complex<T>; N]> for F
where
    F: FFT<T>,
    T: 'static + Copy + Float + Neg<Output = T> + Default,
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
    T: 'static + Copy + Float + Neg<Output = T> + Default,
{
    fn evaluate_inverse(&self, y: [Complex<T>; N]) -> [T; N] {
        let mut x = [T::default(); N];
        F::ifft(&y, &mut x);
        x
    }
}

// Sampling interval `dt` maps a continuous plant onto its discrete form.
impl<T: Float, C: Continuous<T>> Map<T, C::Discrete> for C {
    fn evaluate(&self, dt: T) -> C::Discrete {
        self.discretize(dt)
    }
}
