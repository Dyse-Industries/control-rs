use crate::math::dsp::Convolution;

mod convolution_test_suite {
    use super::*;

    const TOLERANCE: f64 = 1e-6;

    #[test]
    fn test_convolution_identity() {
        let input = [1.0, 2.0, 3.0];
        let kernel = [1.0];
        let mut output = [0.0f64; 3];
        TestConvolution::convolve_input(&input, &kernel, &mut output);
        for (out_val, in_val) in output.iter().zip(input.iter()) {
            assert!((out_val - in_val).abs() < TOLERANCE);
        }
    }

    #[test]
    fn test_convolution_with_impulse() {
        let input = [1.0, 2.0, 3.0];
        let kernel = [0.0, 1.0, 0.0];
        let mut output = [0.0f64; 5];
        TestConvolution::convolve_input(&input, &kernel, &mut output);
        let expected = [0.0, 1.0, 2.0, 3.0, 0.0];
        for (out_val, exp_val) in output.iter().zip(expected.iter()) {
            assert!((out_val - exp_val).abs() < TOLERANCE);
        }
    }

    #[test]
    fn test_convolution_boxcar() {
        let input = [1.0, 1.0];
        let kernel = [1.0, 1.0];
        let mut output = [0.0f64; 3];
        TestConvolution::convolve_input(&input, &kernel, &mut output);
        let expected = [1.0, 2.0, 1.0];
        for (out_val, exp_val) in output.iter().zip(expected.iter()) {
            assert!((out_val - exp_val).abs() < TOLERANCE);
        }
    }

    #[test]
    #[should_panic(expected = "Convolution output buffer is too small")]
    fn test_convolve_buffer_panic() {
        let input = [1.0, 2.0];
        let kernel = [1.0, 1.0];
        let mut output = [0.0; 1]; // Too small! Expected 2 + 2 - 1 = 3
        TestConvolution::convolve_input(&input, &kernel, &mut output);
    }

    #[test]
    fn test_convolution_empty() {
        let input = [];
        let kernel = [1.0];
        let mut output = [];
        TestConvolution::convolve_input(&input, &kernel, &mut output);

        let input2 = [1.0];
        let kernel2 = [];
        let mut output2 = [];
        TestConvolution::convolve_input(&input2, &kernel2, &mut output2);
    }
}

struct TestConvolution;
impl<T: crate::math::num_traits::Real> Convolution<T> for TestConvolution {}
