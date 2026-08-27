//! Digital Signal Processing (DSP) mathematical HIL and unit test suite.
//!
//! `dsp.rs` has no dedicated design doc, so no functional-requirement
//! citations apply here.

#[cfg_attr(not(test), control_rs_macros::hil_suite)]
pub mod dsp_test_suite {
    use crate::math::{
        Bijection, Map, complex_num::Complex, dsp::Convolution, dsp::FFT,
        num_traits::Trig,
    };

    const FFT_N: usize = 8;
    const TOLERANCE: f64 = 1e-6;

    struct TestConvolution;
    struct TestFFT;

    impl<T: crate::math::num_traits::Float> Convolution<T> for TestConvolution {}

    impl<
        T: 'static
            + Copy
            + crate::math::num_traits::Float
            + crate::math::ops::Neg<Output = T>
            + Default,
    > FFT<T> for TestFFT
    {
    }

    // --- Convolution Tests ---

    #[cfg_attr(test, test)]
    /// Verifies convolution calculation against the identity (impulse of amplitude 1) kernel.
    fn test_dsp_convolution_identity() {
        let input = [1.0, 2.0, 3.0];
        let kernel = [1.0];
        let mut output = [0.0f64; 3];
        TestConvolution::convolve_input(&input, &kernel, &mut output);
        for (out_val, in_val) in output.iter().zip(input.iter()) {
            assert!((out_val - in_val).abs() < TOLERANCE);
        }
    }

    #[cfg_attr(test, test)]
    /// Verifies convolution with a shifted/delayed impulse.
    fn test_dsp_convolution_with_impulse() {
        let input = [1.0, 2.0, 3.0];
        let kernel = [0.0, 1.0, 0.0];
        let mut output = [0.0f64; 5];
        TestConvolution::convolve_input(&input, &kernel, &mut output);
        let expected = [0.0, 1.0, 2.0, 3.0, 0.0];
        for (out_val, exp_val) in output.iter().zip(expected.iter()) {
            assert!((out_val - exp_val).abs() < TOLERANCE);
        }
    }

    #[cfg_attr(test, test)]
    /// Verifies convolution between two boxcar signals.
    fn test_dsp_convolution_boxcar() {
        let input = [1.0, 1.0];
        let kernel = [1.0, 1.0];
        let mut output = [0.0f64; 3];
        TestConvolution::convolve_input(&input, &kernel, &mut output);
        let expected = [1.0, 2.0, 1.0];
        for (out_val, exp_val) in output.iter().zip(expected.iter()) {
            assert!((out_val - exp_val).abs() < TOLERANCE);
        }
    }

    #[cfg(test)]
    #[test]
    #[should_panic(expected = "Convolution output buffer is too small")]
    fn _test_convolve_buffer_panic() {
        let input = [1.0, 2.0];
        let kernel = [1.0, 1.0];
        let mut output = [0.0; 1]; // Too small! Expected 2 + 2 - 1 = 3
        TestConvolution::convolve_input(&input, &kernel, &mut output);
    }

    #[cfg_attr(test, test)]
    /// Verifies convolution handles empty input or empty kernel arrays gracefully.
    fn test_dsp_convolution_empty() {
        let input = [];
        let kernel = [1.0];
        let mut output = [];
        TestConvolution::convolve_input(&input, &kernel, &mut output);

        let input2 = [1.0];
        let kernel2 = [];
        let mut output2 = [];
        TestConvolution::convolve_input(&input2, &kernel2, &mut output2);
    }

    // --- FFT / IFFT Tests ---

    #[cfg_attr(test, test)]
    /// Verifies forward FFT of a time-domain impulse signal results in flat frequency magnitude.
    fn test_dsp_fft_impulse() {
        let mut input = [0.0f64; FFT_N];
        input[0] = 1.0;
        let mut output = [Complex::default(); FFT_N];

        TestFFT::fft(&input, &mut output);

        for val in output {
            assert!((val.re - 1.0f64).abs() < TOLERANCE);
            assert!(val.im.abs() < TOLERANCE);
        }
    }

    #[cfg_attr(test, test)]
    /// Verifies forward FFT of a DC (constant 1.0) signal concentrated in the zero-frequency bin.
    fn test_dsp_fft_dc() {
        let input = [1.0; FFT_N];
        let mut output = [Complex::default(); FFT_N];

        TestFFT::fft(&input, &mut output);

        assert!(
            (output[0].re - f64::from(u32::try_from(FFT_N).unwrap())).abs()
                < TOLERANCE
        );
        assert!(output[0].im.abs() < TOLERANCE);

        for val in output.iter().take(FFT_N).skip(1) {
            assert!(val.re.abs() < TOLERANCE);
            assert!(val.im.abs() < TOLERANCE);
        }
    }

    #[cfg_attr(test, test)]
    /// Verifies that running forward FFT and then inverse IFFT returns the original signal.
    fn test_dsp_fft_ifft_roundtrip() {
        let mut input = [0.0f64; FFT_N];
        for (i, val) in input.iter_mut().enumerate() {
            *val = Trig::sin(f64::from(u32::try_from(i).unwrap()));
        }

        let mut fft_output = [Complex::default(); FFT_N];
        TestFFT::fft(&input, &mut fft_output);

        let mut ifft_output = [0.0f64; FFT_N];
        TestFFT::ifft(&fft_output, &mut ifft_output);

        for (in_val, out_val) in input.iter().zip(ifft_output.iter()) {
            assert!((in_val - out_val).abs() < TOLERANCE);
        }
    }

    #[cfg_attr(test, test)]
    /// Verifies implementation of the Map and Bijection traits on the FFT runner.
    fn test_dsp_fft_map_and_bijection_traits() {
        let time_signal = [1.0, 0.0, 0.0, 0.0];
        let fft = TestFFT;
        let freq_signal = fft.evaluate(time_signal);
        let recovered: [f64; 4] = fft.evaluate_inverse(freq_signal);
        assert!((recovered[0] - 1.0_f64).abs() < 1e-6);
    }
}
