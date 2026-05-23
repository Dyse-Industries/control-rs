mod test_axioms {
    use crate::{assert_almost_eq, math::complex_num::Complex64};

    // Assuming you have implemented std::ops traits (Add, Mul, etc.)
    #[test]
    fn test_addition_commutativity() {
        let z1 = Complex64::new(1.2, 3.4);
        let z2 = Complex64::new(-5.6, 7.8);
        assert_almost_eq!((z1 + z2).re, (z2 + z1).re);
        assert_almost_eq!((z1 + z2).im, (z2 + z1).im);
    }

    #[test]
    fn test_multiplication_commutativity() {
        let z1 = Complex64::new(1.2, 3.4);
        let z2 = Complex64::new(-5.6, 7.8);
        assert_almost_eq!((z1 * z2).re, (z2 * z1).re);
        assert_almost_eq!((z1 * z2).im, (z2 * z1).im);
    }

    #[test]
    fn test_distributivity() {
        let z1 = Complex64::new(1.0, 2.0);
        let z2 = Complex64::new(3.0, 4.0);
        let z3 = Complex64::new(5.0, 6.0);
        let left = z1 * (z2 + z3);
        let right = (z1 * z2) + (z1 * z3);
        assert_almost_eq!(left.re, right.re);
        assert_almost_eq!(left.im, right.im);
    }

    #[test]
    fn test_identities() {
        let z = Complex64::new(4.2, -1.1);
        let zero = Complex64::new(0.0, 0.0);
        let one = Complex64::new(1.0, 0.0);

        assert_almost_eq!((z + zero).re, z.re);
        assert_almost_eq!((z * one).re, z.re);
        assert_almost_eq!((z + zero).im, z.im);
        assert_almost_eq!((z * one).im, z.im);
    }
}

mod test_basics {
    use crate::assert_almost_eq;
    use crate::math::{complex_num::Complex32, num_traits::Trig};

    #[test]
    fn test_new() {
        let z = Complex32::new(1.0, 2.0);
        assert_almost_eq!(z.re, 1.0_f32);
        assert_almost_eq!(z.im, 2.0_f32);
    }

    #[test]
    fn test_from_real() {
        let z = Complex32::from_real(5.0);
        assert_almost_eq!(z.re, 5.0);
        assert_almost_eq!(z.im, 0.0);
    }

    #[test]
    fn test_from_imag() {
        let z = Complex32::from_imag(5.0);
        assert_almost_eq!(z.re, 0.0);
        assert_almost_eq!(z.im, 5.0);
    }

    #[test]
    fn test_polar_creation() {
        let r = 2.0_f32;
        let theta = f32::PI / 4.0_f32;
        let z = Complex32::from_polar(r, theta);
        let expected = r * (f32::PI / 4.0_f32).cos(); // Both re and im should be this
        assert_almost_eq!(z.re, expected);
        assert_almost_eq!(z.im, expected);
    }
}

mod test_core_math {
    use crate::{
        assert_almost_eq,
        math::{complex_num::Complex64, num_traits::Trig},
    };

    #[test]
    fn test_conjugate() {
        let z = Complex64::new(3.0, 4.0).conj();
        assert_almost_eq!(z.re, 3.0);
        assert_almost_eq!(z.im, -4.0);
    }

    #[test]
    fn test_magnitude() {
        let z = Complex64::new(3.0, 4.0);
        assert_almost_eq!(z.magnitude(), 5.0);
    }

    #[test]
    fn test_argument_phase() {
        let z1 = Complex64::new(1.0, 1.0);
        assert_almost_eq!(z1.arg(), f64::PI / 4.0);

        let z2 = Complex64::new(-1.0, 0.0);
        assert_almost_eq!(z2.arg(), f64::PI);
    }
}

mod test_dsp_patterns {
    use crate::{
        assert_almost_eq,
        math::{
            complex_num::Complex64,
            num_traits::{Exponential, Trig},
        },
    };

    #[test]
    fn test_twiddle_factors() {
        let n = 4.0;

        // k = 1: e^(-i * 2PI * 1 / 4) = e^(-i * PI / 2) = -i
        let arg1 = -2.0 * f64::PI * 1.0 / n;
        let twiddle1 = Complex64::new(0.0, arg1).exp();
        assert_almost_eq!(twiddle1.re, 0.0);
        assert_almost_eq!(twiddle1.im, -1.0);

        // k = 2: e^(-i * 2PI * 2 / 4) = e^(-i * PI) = -1
        let arg2 = -2.0 * f64::PI * 2.0 / n;
        let twiddle2 = Complex64::new(0.0, arg2).exp();
        assert_almost_eq!(twiddle2.re, -1.0);
        assert_almost_eq!(twiddle2.im, 0.0);
    }
}

mod test_ffi_layout {
    use crate::{
        assert_almost_eq,
        math::{complex_num::Complex64, num_traits::Trig},
    };
    use core::mem;

    #[test]
    fn test_size_and_alignment() {
        assert_eq!(size_of::<Complex64>(), 16);
        assert_eq!(align_of::<Complex64>(), 8);
    }

    #[test]
    #[allow(clippy::transmute_undefined_repr)]
    fn test_memory_representation() {
        let z = Complex64::new(f64::PI, -2.71);
        // Transmute strictly requires the memory layout to match exactly
        let array: [f64; 2] = unsafe { mem::transmute(z) };
        assert_almost_eq!(array[0], f64::PI);
        assert_almost_eq!(array[1], -2.71);
    }
}

mod test_limitations {
    use crate::{
        assert_almost_eq,
        math::{
            complex_num::Complex64,
            num_traits::{Radical, Real},
        },
    };

    #[test]
    fn test_nan_propagation() {
        let z1 = Complex64::new(1.0, 2.0);
        let z_nan = Complex64::new(f64::NAN, 0.0);

        let result = z1 + z_nan;
        assert!(result.re.is_nan());
        // Imaginary part might not be NaN depending on your addition implementation,
        // but strictly speaking, the real part must be corrupted.
    }

    #[test]
    fn test_complex_infinity_magnitude() {
        let z = Complex64::new(f64::INF, 5.0);
        assert!(z.magnitude().is_infinite());

        let z2 = Complex64::new(f64::INF, f64::INF);
        assert!(z2.magnitude().is_infinite());
        // Note: A naive sqrt(re*re + im*im) might yield NaN here!
        // You should be using f64::hypot internally to pass this.
    }

    #[test]
    fn test_sqrt_branch_cut() {
        // sqrt(-4) should be 2i, correctly handling the sign of 0.0
        let z = Complex64::new(-4.0, 0.0);
        let result = z.sqrt();
        assert_almost_eq!(result.re, 0.0);
        assert_almost_eq!(result.im, 2.0);
    }
}

mod test_transcendental {
    use crate::{
        assert_almost_eq,
        math::{
            complex_num::Complex32,
            num_traits::{Exponential, Trig},
        },
    };

    #[test]
    fn test_exp() {
        // e^(i * PI) = -1
        let z = Complex32::new(0.0, f32::PI);
        let result = z.exp();
        assert_almost_eq!(result, Complex32::new(-1.0, 0.0));
    }

    #[test]
    fn test_ln() {
        // ln(-1) = i * PI
        let z = Complex32::new(-1.0, 0.0);
        let result = z.ln();
        assert_almost_eq!(result, Complex32::new(0.0, f32::PI));
    }

    #[test]
    fn test_trig_functions() {
        // sin(0) = 0
        let zero = Complex32::new(0.0, 0.0);
        assert_almost_eq!(zero.sin().re, 0.0);
        assert_almost_eq!(zero.sin().im, 0.0);

        // cos(0) = 1
        assert_almost_eq!(zero.cos().re, 1.0);
        assert_almost_eq!(zero.cos().im, 0.0);
    }
}
