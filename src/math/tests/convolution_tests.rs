use crate::math::dsp::Convolution;

struct TestConvolution;
impl<T: crate::math::num_traits::Real> Convolution<T> for TestConvolution {}

mod convolution_test_suite {
    use super::*;

    const TOLERANCE: f64 = 1e-6;

    #[test]
    fn test_convolution_identity() {
        let input = [1.0, 2.0, 3.0];
        let kernel = [1.0];
        let mut output = [0.0f64; 3];
        TestConvolution::convolve_input(&input, &kernel, &mut output);
        for i in 0..3 {
            assert!((output[i] - input[i]).abs() < TOLERANCE);
        }
    }

    #[test]
    fn test_convolution_with_impulse() {
        let input = [1.0, 2.0, 3.0];
        let kernel = [0.0, 1.0, 0.0];
        let mut output = [0.0f64; 5];
        TestConvolution::convolve_input(&input, &kernel, &mut output);
        let expected = [0.0, 1.0, 2.0, 3.0, 0.0];
        for i in 0..5 {
            assert!((output[i] - expected[i]).abs() < TOLERANCE);
        }
    }

    #[test]
    fn test_convolution_boxcar() {
        let input = [1.0, 1.0];
        let kernel = [1.0, 1.0];
        let mut output = [0.0f64; 3];
        TestConvolution::convolve_input(&input, &kernel, &mut output);
        let expected = [1.0, 2.0, 1.0];
        for i in 0..3 {
            assert!((output[i] - expected[i]).abs() < TOLERANCE);
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
}
