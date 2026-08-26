//! Digital Signal Processing (DSP) mathematical ETS and unit test suite.
//!
//! `dsp.rs` has no dedicated design doc, so no functional-requirement
//! citations apply here.

#[cfg_attr(not(test), control_rs_macros::ets_suite)]
pub mod dsp_test_suite {
    use crate::math::{
        Bijection, ConversionError, Map,
        complex_num::Complex,
        dsp::{Continuous, Convolution, Discrete, FFT},
        num_traits::Trig,
    };

    const FFT_N: usize = 8;
    const TOLERANCE: f64 = 1e-6;

    struct TestConvolution;
    struct TestFFT;
    struct DummyContinuous;
    struct DummyDiscrete {
        dt: f64,
    }

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

    // --- Convolution Tests ---

    #[cfg_attr(test, test)]
    /// Verifies convolution calculation against the identity (impulse of amplitude 1) kernel.
    fn test_dsp_convolution_identity() {
        let input = [1.0, 2.0, 3.0];
        let kernel = [1.0];
        let mut output = [0.0f64; 3];
        TestConvolution::convolve_input(&input, &kernel, &mut output).unwrap();
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
        TestConvolution::convolve_input(&input, &kernel, &mut output).unwrap();
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
        TestConvolution::convolve_input(&input, &kernel, &mut output).unwrap();
        let expected = [1.0, 2.0, 1.0];
        for (out_val, exp_val) in output.iter().zip(expected.iter()) {
            assert!((out_val - exp_val).abs() < TOLERANCE);
        }
    }

    #[cfg_attr(test, test)]
    fn test_dsp_convolve_dimension_mismatch() {
        let input = [1.0, 2.0];
        let kernel = [1.0, 1.0];
        let mut output = [0.0; 1];
        assert_eq!(
            TestConvolution::convolve_input(&input, &kernel, &mut output),
            Err(ConversionError::DimensionMismatch)
        );
    }

    #[cfg_attr(test, test)]
    /// Verifies convolution handles empty input or empty kernel arrays gracefully.
    fn test_dsp_convolution_empty() {
        let input = [];
        let kernel = [1.0];
        let mut output = [];
        TestConvolution::convolve_input(&input, &kernel, &mut output).unwrap();

        let input2 = [1.0];
        let kernel2 = [];
        let mut output2 = [];
        TestConvolution::convolve_input(&input2, &kernel2, &mut output2)
            .unwrap();
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

    #[cfg_attr(test, test)]
    /// Verifies Continuous/Discrete associated-type round-trip and Map wiring.
    fn test_dsp_continuous_discrete_roundtrip() {
        let plant = DummyContinuous;
        let dt = 0.01_f64;
        let sampled = plant.discretize(dt);
        assert!((sampled.sampling_period() - dt).abs() < TOLERANCE);

        let mapped: DummyDiscrete = plant.evaluate(dt);
        assert!((mapped.sampling_period() - dt).abs() < TOLERANCE);

        let _ = sampled.to_continuous();
    }
}
