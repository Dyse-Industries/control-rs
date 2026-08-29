//! Digital Signal Processing (DSP) mathematical ETS and unit test suite.
//!
//! `dsp.rs` has no dedicated design doc, so no functional-requirement
//! citations apply here. Length `N` is a power-of-two `debug_assert` on
//! [`FFT`] entry points, not a [`Result`] path.

#[cfg(all(test, debug_assertions))]
mod dsp_debug_contract_tests {
    use crate::math::complex_num::Complex;
    use crate::math::dsp::{DefaultDsp, FFT};

    #[test]
    #[should_panic(expected = "FFT length must be a power of two")]
    fn test_dsp_fft_rejects_non_power_of_two() {
        let input = [1.0f64, 2.0, 3.0];
        let mut output = [Complex::default(); 3];
        DefaultDsp::fft(&input, &mut output);
    }
}

#[cfg_attr(not(test), control_rs_macros::ets_suite)]
pub mod dsp_test_suite {
    #![allow(clippy::arithmetic_side_effects, clippy::indexing_slicing)]

    use crate::math::{
        Bijection, ConversionError, Map,
        complex_num::Complex,
        dsp::{Continuous, Convolution, DefaultDsp, Discrete, FFT},
        num_traits::{Scalar, Trig},
    };

    const FFT_N: usize = 8;
    const FFT_N16: usize = 16;

    struct DummyContinuous;
    struct DummyDiscrete {
        dt: f64,
    }

    impl Continuous<f64> for DummyContinuous {
        type Discrete = DummyDiscrete;

        fn discretize(&self, dt: f64) -> Self::Discrete {
            DummyDiscrete { dt }
        }
    }

    impl Discrete<f64> for DummyDiscrete {
        type Continuous = DummyContinuous;

        fn sampling_period(&self) -> f64 {
            self.dt
        }

        fn to_continuous(&self) -> Self::Continuous {
            DummyContinuous
        }
    }

    #[allow(clippy::cast_precision_loss)]
    fn _fft_tol(n: usize) -> f64 {
        let n_f = n as f64;
        n_f * 8.0 * f64::EPSILON
    }

    fn _assert_close(left: f64, right: f64, tol: f64) {
        assert!(
            (left - right).abs() <= tol,
            "left={left} right={right} tol={tol}"
        );
    }

    fn _fft_impulse_is_ones<const N: usize>() {
        let mut input = [0.0f64; N];
        input[0] = 1.0;
        let mut output = [Complex::default(); N];
        DefaultDsp::fft(&input, &mut output);
        let tol = _fft_tol(N);
        for val in output {
            _assert_close(val.re, 1.0, tol);
            _assert_close(val.im, 0.0, tol);
        }
    }

    fn _parseval_real_fft<const N: usize>(input: &[f64; N]) {
        let mut spectrum = [Complex::default(); N];
        DefaultDsp::fft(input, &mut spectrum);
        let mut time_energy = 0.0;
        for x in input {
            time_energy += x * x;
        }
        let mut freq_energy = 0.0;
        for z in spectrum {
            freq_energy += z.abs2();
        }
        let n_f = {
            #[allow(clippy::cast_precision_loss)]
            let n = N as f64;
            n
        };
        // Unnormalized DFT: ∑|x|² = (1/N) ∑|X|².
        _assert_close(time_energy, freq_energy / n_f, _fft_tol(N) * n_f);
    }

    fn _hermitian_real_fft<const N: usize>(input: &[f64; N]) {
        let mut spectrum = [Complex::default(); N];
        DefaultDsp::fft(input, &mut spectrum);
        let tol = _fft_tol(N);
        for k in 1..(N / 2) {
            let hi = spectrum[k];
            let lo = spectrum[N - k];
            _assert_close(hi.re, lo.re, tol);
            _assert_close(hi.im, -lo.im, tol);
        }
    }

    // --- Convolution Tests ---

    #[cfg_attr(test, test)]
    /// Verifies convolution calculation against the identity (impulse of amplitude 1) kernel.
    fn test_dsp_convolution_identity() {
        let input = [1.0, 2.0, 3.0];
        let kernel = [1.0];
        let mut output = [0.0f64; 3];
        DefaultDsp::convolve_input(&input, &kernel, &mut output).unwrap();
        for (out_val, in_val) in output.iter().zip(input.iter()) {
            _assert_close(*out_val, *in_val, _fft_tol(3));
        }
    }

    #[cfg_attr(test, test)]
    /// Verifies convolution with a shifted/delayed impulse.
    fn test_dsp_convolution_with_impulse() {
        let input = [1.0, 2.0, 3.0];
        let kernel = [0.0, 1.0, 0.0];
        let mut output = [0.0f64; 5];
        DefaultDsp::convolve_input(&input, &kernel, &mut output).unwrap();
        let expected = [0.0, 1.0, 2.0, 3.0, 0.0];
        for (out_val, exp_val) in output.iter().zip(expected.iter()) {
            _assert_close(*out_val, *exp_val, _fft_tol(5));
        }
    }

    #[cfg_attr(test, test)]
    /// Verifies convolution between two boxcar signals.
    fn test_dsp_convolution_boxcar() {
        let input = [1.0, 1.0];
        let kernel = [1.0, 1.0];
        let mut output = [0.0f64; 3];
        DefaultDsp::convolve_input(&input, &kernel, &mut output).unwrap();
        let expected = [1.0, 2.0, 1.0];
        for (out_val, exp_val) in output.iter().zip(expected.iter()) {
            _assert_close(*out_val, *exp_val, _fft_tol(3));
        }
    }

    #[cfg_attr(test, test)]
    fn test_dsp_convolve_dimension_mismatch() {
        let input = [1.0, 2.0];
        let kernel = [1.0, 1.0];
        let mut output = [0.0; 1];
        assert_eq!(
            DefaultDsp::convolve_input(&input, &kernel, &mut output),
            Err(ConversionError::DimensionMismatch)
        );
    }

    #[cfg_attr(test, test)]
    /// Verifies convolution handles empty input or empty kernel arrays gracefully.
    fn test_dsp_convolution_empty() {
        let input = [];
        let kernel = [1.0];
        let mut output = [];
        DefaultDsp::convolve_input(&input, &kernel, &mut output).unwrap();

        let input2 = [1.0];
        let kernel2 = [];
        let mut output2 = [];
        DefaultDsp::convolve_input(&input2, &kernel2, &mut output2).unwrap();
    }

    #[cfg_attr(test, test)]
    /// Leaves `output[expected_len..]` unchanged when the buffer is longer
    /// than `input_len + kernel_len - 1`.
    fn test_dsp_convolution_leftover_output_tail() {
        let input = [1.0, 2.0, 3.0];
        let kernel = [1.0];
        let mut output = [99.0f64; 5];
        DefaultDsp::convolve_input(&input, &kernel, &mut output).unwrap();
        let tol = _fft_tol(5);
        _assert_close(output[0], 1.0, tol);
        _assert_close(output[1], 2.0, tol);
        _assert_close(output[2], 3.0, tol);
        _assert_close(output[3], 99.0, 0.0);
        _assert_close(output[4], 99.0, 0.0);
    }

    #[cfg_attr(test, test)]
    /// Integer [`Scalar`] convolution through [`DefaultDsp`].
    fn test_dsp_convolution_integer_scalar() {
        let input = [1i32, 2, 3];
        let kernel = [1i32, 1];
        let mut output = [0i32; 4];
        DefaultDsp::convolve_input(&input, &kernel, &mut output).unwrap();
        assert_eq!(output, [1, 3, 5, 3]);
    }

    // --- FFT / IFFT Tests ---

    #[cfg_attr(test, test)]
    /// Verifies forward FFT of a time-domain impulse signal results in flat frequency magnitude.
    fn test_dsp_fft_impulse() {
        _fft_impulse_is_ones::<FFT_N>();
    }

    #[cfg_attr(test, test)]
    fn test_dsp_fft_impulse_n1() {
        _fft_impulse_is_ones::<1>();
    }

    #[cfg_attr(test, test)]
    fn test_dsp_fft_impulse_n16() {
        _fft_impulse_is_ones::<FFT_N16>();
    }

    #[cfg_attr(test, test)]
    /// Isolates [`FFT::fft_complex`] on a complex impulse.
    fn test_dsp_fft_complex_impulse() {
        let mut data = [Complex::<f64>::default(); FFT_N];
        data[0] = Complex::new(1.0, 0.0);
        DefaultDsp::fft_complex(&mut data);
        let tol = _fft_tol(FFT_N);
        for val in data {
            _assert_close(val.re, 1.0, tol);
            _assert_close(val.im, 0.0, tol);
        }
    }

    #[cfg_attr(test, test)]
    /// Verifies forward FFT of a DC (constant 1.0) signal concentrated in the zero-frequency bin.
    fn test_dsp_fft_dc() {
        let input = [1.0; FFT_N];
        let mut output = [Complex::default(); FFT_N];

        DefaultDsp::fft(&input, &mut output);

        let n_f = f64::from(u32::try_from(FFT_N).unwrap());
        let tol = _fft_tol(FFT_N);
        _assert_close(output[0].re, n_f, tol);
        _assert_close(output[0].im, 0.0, tol);

        for val in output.iter().take(FFT_N).skip(1) {
            _assert_close(val.re, 0.0, tol);
            _assert_close(val.im, 0.0, tol);
        }
    }

    #[cfg_attr(test, test)]
    /// Cosine at bin `k=1` occupies bins 1 and `N-1` with amplitude `N/2`.
    fn test_dsp_fft_known_bin_sinusoid() {
        let n_f = f64::from(u32::try_from(FFT_N).unwrap());
        let mut input = [0.0f64; FFT_N];
        for (n, val) in input.iter_mut().enumerate() {
            let n_f_idx = f64::from(u32::try_from(n).unwrap());
            *val = Trig::cos(core::f64::consts::TAU * n_f_idx / n_f);
        }
        let mut output = [Complex::default(); FFT_N];
        DefaultDsp::fft(&input, &mut output);
        let tol = _fft_tol(FFT_N) * n_f;
        let half = n_f / 2.0;
        _assert_close(output[1].re, half, tol);
        _assert_close(output[1].im, 0.0, tol);
        _assert_close(output[FFT_N - 1].re, half, tol);
        _assert_close(output[FFT_N - 1].im, 0.0, tol);
        for (k, val) in output.iter().enumerate() {
            if k != 1 && k != FFT_N - 1 {
                _assert_close(val.re, 0.0, tol);
                _assert_close(val.im, 0.0, tol);
            }
        }
    }

    #[cfg_attr(test, test)]
    fn test_dsp_fft_parseval() {
        let mut input = [0.0f64; FFT_N];
        for (i, val) in input.iter_mut().enumerate() {
            *val = Trig::sin(f64::from(u32::try_from(i).unwrap()));
        }
        _parseval_real_fft(&input);
        _parseval_real_fft(&[1.0; FFT_N16]);
    }

    #[cfg_attr(test, test)]
    fn test_dsp_fft_hermitian_symmetry() {
        let mut input = [0.0f64; FFT_N];
        for (i, val) in input.iter_mut().enumerate() {
            *val = Trig::sin(f64::from(u32::try_from(i).unwrap()));
        }
        _hermitian_real_fft(&input);
        let mut input16 = [0.0f64; FFT_N16];
        for (i, val) in input16.iter_mut().enumerate() {
            *val = f64::from(u32::try_from(i).unwrap());
        }
        _hermitian_real_fft(&input16);
    }

    #[cfg_attr(test, test)]
    /// Verifies that running forward FFT and then inverse IFFT returns the original signal.
    fn test_dsp_fft_ifft_roundtrip() {
        let mut input = [0.0f64; FFT_N];
        for (i, val) in input.iter_mut().enumerate() {
            *val = Trig::sin(f64::from(u32::try_from(i).unwrap()));
        }

        let mut fft_output = [Complex::default(); FFT_N];
        DefaultDsp::fft(&input, &mut fft_output);

        let mut ifft_output = [0.0f64; FFT_N];
        DefaultDsp::ifft(&fft_output, &mut ifft_output);

        let tol = _fft_tol(FFT_N) * f64::from(u32::try_from(FFT_N).unwrap());
        for (in_val, out_val) in input.iter().zip(ifft_output.iter()) {
            _assert_close(*in_val, *out_val, tol);
        }
    }

    #[cfg_attr(test, test)]
    /// Verifies implementation of the Map and Bijection traits on [`DefaultDsp`].
    fn test_dsp_fft_map_and_bijection_traits() {
        let time_signal = [1.0, 0.0, 0.0, 0.0];
        let fft = DefaultDsp;
        let freq_signal = fft.evaluate(time_signal);
        let recovered: [f64; 4] = fft.evaluate_inverse(freq_signal);
        _assert_close(recovered[0], 1.0, _fft_tol(4));
    }

    #[cfg_attr(test, test)]
    /// Verifies Continuous/Discrete associated-type round-trip and Map wiring.
    fn test_dsp_continuous_discrete_roundtrip() {
        let plant = DummyContinuous;
        let dt = 0.01_f64;
        let sampled = plant.discretize(dt);
        _assert_close(sampled.sampling_period(), dt, _fft_tol(1));

        let mapped: DummyDiscrete = plant.evaluate(dt);
        _assert_close(mapped.sampling_period(), dt, _fft_tol(1));

        let _ = sampled.to_continuous();
    }
}
