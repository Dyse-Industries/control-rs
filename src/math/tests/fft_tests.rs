//! FFT mathematical HIL and unit test suite.
#![allow(clippy::arithmetic_side_effects)]
#![allow(dead_code)]
#![allow(clippy::unwrap_used)]
#![allow(unused_imports)]
#![allow(clippy::module_inception)]

#[cfg_attr(all(not(test), not(feature = "std")), control_rs_macros::hil_suite)]
/// HIL and unit test suite for Fast Fourier Transform.
pub mod fft_tests {
    use crate::math::dsp::FFT;
    use crate::math::num_traits::{Signed, Trig};
    use crate::math::{Bijection, Map, complex_num::Complex};

    const N: usize = 8;
    const TOLERANCE: f64 = 1e-6;

    struct TestFFT;
    impl<
        T: 'static
            + Copy
            + crate::math::num_traits::Real
            + crate::math::ops::Neg<Output = T>
            + Default,
    > FFT<T> for TestFFT
    {
    }

    #[cfg_attr(test, test)]
    fn test_fft_impulse() {
        let mut input = [0.0; N];
        input[0] = 1.0;
        let mut output = [Complex::default(); N];

        TestFFT::fft(&input, &mut output);

        for val in output {
            assert!((val.re - 1.0f64).abs() < TOLERANCE);
            assert!(val.im.abs() < TOLERANCE);
        }
    }

    #[cfg_attr(test, test)]
    fn test_fft_dc() {
        let input = [1.0; N];
        let mut output = [Complex::default(); N];

        TestFFT::fft(&input, &mut output);

        assert!(
            (output[0].re - f64::from(u32::try_from(N).unwrap())).abs()
                < TOLERANCE
        );
        assert!(output[0].im.abs() < TOLERANCE);

        for val in output.iter().take(N).skip(1) {
            assert!(val.re.abs() < TOLERANCE);
            assert!(val.im.abs() < TOLERANCE);
        }
    }

    #[cfg_attr(test, test)]
    fn test_fft_ifft_identity() {
        let mut input = [0.0; N];
        for (i, val) in input.iter_mut().enumerate() {
            *val = f64::from(u32::try_from(i).unwrap()).sin();
        }

        let mut fft_output = [Complex::default(); N];
        TestFFT::fft(&input, &mut fft_output);

        let mut ifft_output = [0.0; N];
        TestFFT::ifft(&fft_output, &mut ifft_output);

        for (in_val, out_val) in input.iter().zip(ifft_output.iter()) {
            assert!((in_val - out_val).abs() < TOLERANCE);
        }
    }

    #[cfg_attr(test, test)]
    fn test_map_and_bijection_traits() {
        let time_signal = [1.0, 0.0, 0.0, 0.0];
        let fft = TestFFT;
        let freq_signal = fft.evaluate(time_signal);
        let recovered: [f64; 4] = fft.evaluate_inverse(freq_signal);
        assert!((recovered[0] - 1.0_f64).abs() < 1e-6);
    }
}
