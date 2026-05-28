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

    #[test]
    fn test_comparisons() {
        let z1 = crate::math::complex_num::Complex32::new(1.0, 2.0);
        let z2 = crate::math::complex_num::Complex32::new(2.0, 2.0);
        let z3 = crate::math::complex_num::Complex32::new(1.0, 3.0);
        let z4 = crate::math::complex_num::Complex32::new(2.0, 3.0);

        assert!(z1 < z2); // Real part differs
        assert!(z1 < z3); // Imaginary part differs
        assert!(z1 < z4); // Both differ
        assert!(z2 > z1);
        assert!(z3 > z1);
        assert!(z4 > z1);
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

    #[test]
    fn test_polar_conversion() {
        let z = crate::math::complex_num::Complex64::new(3.0, 4.0);
        let (r, theta) = z.to_polar();
        let z_reconstructed =
            crate::math::complex_num::Complex64::from_polar(r, theta);

        assert_almost_eq!(z.re, z_reconstructed.re);
        assert_almost_eq!(z.im, z_reconstructed.im);
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

        let q2 = Complex64::new(-1.0, 1.0);
        assert_almost_eq!(q2.arg(), 3.0 * f64::PI / 4.0);

        let q3 = Complex64::new(-1.0, -1.0);
        assert_almost_eq!(q3.arg(), -3.0 * f64::PI / 4.0);

        let q4 = Complex64::new(1.0, -1.0);
        assert_almost_eq!(q4.arg(), -f64::PI / 4.0);
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
        assert_eq!(mem::size_of::<Complex64>(), 16);
        assert_eq!(mem::align_of::<Complex64>(), 8);
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

        let z_pos_real = Complex64::from_real(4.0);
        let sqrt_pos = z_pos_real.sqrt();
        assert_almost_eq!(sqrt_pos.re, 2.0);
        assert_almost_eq!(sqrt_pos.im, 0.0);

        // sqrt(3 + 4i)
        let z_mixed = Complex64::new(3.0, 4.0);
        let sqrt_mixed = z_mixed.sqrt();
        assert_almost_eq!(sqrt_mixed.re, 2.0);
        assert_almost_eq!(sqrt_mixed.im, 1.0);
    }
}

mod test_transcendental {
    use crate::{
        assert_almost_eq,
        math::{
            complex_num::Complex32,
            complex_num::Complex64,
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

        let log10_real = Complex64::from_real(100.0).log10();
        assert_almost_eq!(log10_real.re, 2.0);
        assert_almost_eq!(log10_real.im, 0.0);
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

        let z = Complex64::new(f64::PI / 4.0, 0.0);
        let tan_z = z.tan();
        assert_almost_eq!(tan_z.re, 1.0);
        assert_almost_eq!(tan_z.im, 0.0);
    }

    #[test]
    fn test_inverse_trigonometry() {
        let z = Complex64::new(0.5, 0.5);

        let asin_z = z.asin();
        let sin_asin = asin_z.sin();
        assert_almost_eq!(sin_asin.re, z.re);
        assert_almost_eq!(sin_asin.im, z.im);

        let acos_z = z.acos();
        let cos_acos = acos_z.cos();
        assert_almost_eq!(cos_acos.re, z.re);
        assert_almost_eq!(cos_acos.im, z.im);

        let atan_z = z.atan();
        let tan_atan = atan_z.tan();
        assert_almost_eq!(tan_atan.re, z.re);
        assert_almost_eq!(tan_atan.im, z.im);
    }
}

mod test_arithmetic {
    use crate::{
        assert_almost_eq,
        math::{
            complex_num::Complex,
            complex_num::Complex64,
            num_traits::{Exponential, Field, One, Radical, Ring, Zero},
            ops::{
                Neg, TryAdd, TryDiv, TryMul, TrySub, WrappingAdd, WrappingMul,
                WrappingSub,
            },
        },
    };

    #[test]
    fn test_basic_arithmetic() {
        let z1 = Complex64::new(1.0, 2.0);
        let z2 = Complex64::new(3.0, 4.0);

        let sum = z1 + z2;
        assert_almost_eq!(sum.re, 4.0);
        assert_almost_eq!(sum.im, 6.0);

        let diff = z1 - z2;
        assert_almost_eq!(diff.re, -2.0);
        assert_almost_eq!(diff.im, -2.0);

        let prod = z1 * z2; // (1+2i)(3+4i) = 3 + 4i + 6i - 8 = -5 + 10i
        assert_almost_eq!(prod.re, -5.0);
        assert_almost_eq!(prod.im, 10.0);

        let div = z1 / z2; // (1+2i)/(3+4i) = (1+2i)(3-4i) / 25 = (3 - 4i + 6i + 8) / 25 = 11/25 + 2/25i
        assert_almost_eq!(div.re, 11.0 / 25.0);
        assert_almost_eq!(div.im, 2.0 / 25.0);
    }

    #[test]
    fn test_wrapping_operations() {
        let z1 = Complex::<u8>::new(250, 10);
        let z2 = Complex::<u8>::new(10, 250);

        let w_add = z1.wrapping_add(&z2);
        assert_eq!(w_add.re, 4); // 260 % 256
        assert_eq!(w_add.im, 4); // 260 % 256

        let w_sub = z1.wrapping_sub(&z2);
        assert_eq!(w_sub.re, 240); // 250 - 10
        assert_eq!(w_sub.im, 16); // 10 - 250 = -240 = 16 (mod 256)

        let z3 = Complex::<u8>::new(16, 16);
        let w_mul = z3.wrapping_mul(&z3);
        // (16+16i)^2 = 256 + 256i + 256i - 256 = 512i = 0i (mod 256)
        assert_eq!(w_mul.re, 0);
        assert_eq!(w_mul.im, 0);
    }

    #[test]
    fn test_try_operations() {
        let z1 = Complex::<u8>::new(200, 10);
        let z2 = Complex::<u8>::new(100, 10);

        assert!(z1.try_add(&z2).is_err()); // Real part 300 > 255

        let z3 = Complex::<u8>::new(100, 10);
        assert!(z3.try_add(&z2).is_ok());
        assert!(z2.try_sub(&z1).is_err()); // Real part 100 - 200 < 0
        assert!(z1.try_sub(&z3).is_ok());
        assert!(z1.try_mul(&z2).is_err());
        assert_eq!(
            Complex::<u8>::ONE.try_div(&Complex::<u8>::ONE).ok(),
            Some(Complex::<u8>::ONE)
        );

        let z_small = Complex::<u8>::new(2, 2);
        assert!(z_small.try_mul(&z_small).is_ok());

        // Division
        let z4 = Complex::<u8>::new(100, 100);
        let z5 = Complex::<u8>::new(2, 2);
        // 100^2 + 100^2 = 20000 > 255, so try_div will error on denominator calc
        assert!(z4.try_div(&z5).is_err());

        assert_eq!(Complex::<i8>::ONE.neg(), Complex::new(-1, 0))
    }

    #[test]
    fn test_fallible_and_traits() {
        let a = Complex::new(5.0, 3.0);
        let b = Complex::new(5.0, 1.0);

        // Covers TrySub
        assert!(a.try_sub(&b).is_ok());

        // Covers PartialOrd equal-real branch
        assert_eq!(a.partial_cmp(&b), Some(core::cmp::Ordering::Greater));

        // Covers Zero, One, Ring constants
        let _zero = Complex::<f64>::ZERO;
        let _one = Complex::<f64>::ONE;
        let _max = Complex::<f64>::MAX;
    }

    #[test]
    fn test_sqrt_edge_cases() {
        // Covers self.is_zero() early return
        let z = Complex::<f64>::zero();
        assert_eq!(z.sqrt(), Complex::zero());

        // Covers self.re < 0 branch
        let neg_re = Complex::new(-4.0, 3.0);
        let _ = neg_re.sqrt();

        // Covers self.re < 0 AND self.im < 0 branch
        let neg_both = Complex::new(-4.0, -3.0);
        let _ = neg_both.sqrt();
    }

    #[test]
    fn test_pow_edge_cases() {
        let zero = Complex::<f64>::zero();
        let one = Complex::<f64>::one();
        let two = Complex::new(2.0, 0.0);

        assert_eq!(zero.pow(zero), one);
        assert_eq!(zero.pow(two), zero);
        assert_eq!(one.pow(two), one);
    }

    #[test]
    fn test_complex_epsilon() {
        let eps = Complex::<f32>::epsilon();
        assert_eq!(eps.re, f32::EPSILON);
        assert_eq!(eps.im, 0.0);
    }
}