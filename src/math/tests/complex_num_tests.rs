//! Complex number mathematical HIL and unit test suite.
#![allow(clippy::arithmetic_side_effects)]

#[cfg_attr(not(test), control_rs_macros::hil_suite)]
/// Edge-case limitations, NaN propagation, infinity, branch cut, pow/sqrt, and transcendental tests.
pub mod complex_num_advanced {
    use crate::{
        assert_almost_eq,
        math::{
            complex_num::{Complex, Complex32, Complex64},
            num_traits::{Exponential, One, Radical, Real, Trig, Zero},
            ops::{TryAdd, TryDiv, TryMul, TrySub},
        },
    };

    #[cfg_attr(test, test)]
    fn test_complex_sqrt_edge_cases_advanced() {
        let z = Complex::<f64>::zero();
        assert_eq!(z.sqrt(), Complex::zero());

        let neg_re = Complex::new(-4.0, 3.0);
        let _ = neg_re.sqrt();

        let neg_both = Complex::new(-4.0, -3.0);
        let _ = neg_both.sqrt();
    }

    #[cfg_attr(test, test)]
    fn test_complex_pow_edge_cases_advanced() {
        let zero = Complex::<f64>::zero();
        let one = Complex::<f64>::one();
        let two = Complex::new(2.0, 0.0);

        assert_eq!(zero.pow(zero), one);
        assert_eq!(zero.pow(two), zero);
        assert_eq!(one.pow(two), one);
    }

    #[cfg_attr(test, test)]
    fn test_complex_conjugate_advanced() {
        let z = Complex64::new(3.0, 4.0).conj();
        assert_almost_eq!(z.re, 3.0);
        assert_almost_eq!(z.im, -4.0);
    }

    #[cfg_attr(test, test)]
    fn test_complex_magnitude_advanced() {
        let z = Complex64::new(3.0, 4.0);
        assert_almost_eq!(z.magnitude(), 5.0);
    }

    #[cfg_attr(test, test)]
    fn test_complex_argument_phase_advanced() {
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

    #[cfg_attr(test, test)]
    fn test_complex_twiddle_factors_advanced() {
        let n = 4.0;

        let arg1 = -2.0 * f64::PI * 1.0 / n;
        let twiddle1 = Complex64::new(0.0, arg1).exp();
        assert_almost_eq!(twiddle1.re, 0.0);
        assert_almost_eq!(twiddle1.im, -1.0);

        let arg2 = -2.0 * f64::PI * 2.0 / n;
        let twiddle2 = Complex64::new(0.0, arg2).exp();
        assert_almost_eq!(twiddle2.re, -1.0);
        assert_almost_eq!(twiddle2.im, 0.0);
    }

    #[cfg_attr(test, test)]
    fn test_complex_nan_propagation_advanced() {
        let z1 = Complex64::new(1.0, 2.0);
        let z_nan = Complex64::new(f64::NAN, 0.0);

        let result = z1 + z_nan;
        assert!(result.re.is_nan());
    }

    #[cfg_attr(test, test)]
    fn test_complex_infinity_magnitude_advanced() {
        let z = Complex64::new(f64::INF, 5.0);
        assert!(z.magnitude().is_infinite());

        let z2 = Complex64::new(f64::INF, f64::INF);
        assert!(z2.magnitude().is_infinite());
    }

    #[cfg_attr(test, test)]
    fn test_complex_sqrt_branch_cut_advanced() {
        let z = Complex64::new(-4.0, 0.0);
        let result = z.sqrt();
        assert_almost_eq!(result.re, 0.0);
        assert_almost_eq!(result.im, 2.0);

        let z_pos_real = Complex64::from_real(4.0);
        let sqrt_pos = z_pos_real.sqrt();
        assert_almost_eq!(sqrt_pos.re, 2.0);
        assert_almost_eq!(sqrt_pos.im, 0.0);

        let z_mixed = Complex64::new(3.0, 4.0);
        let sqrt_mixed = z_mixed.sqrt();
        assert_almost_eq!(sqrt_mixed.re, 2.0);
        assert_almost_eq!(sqrt_mixed.im, 1.0);
    }

    #[cfg_attr(test, test)]
    fn test_complex_division_by_zero_advanced() {
        let z = Complex64::new(1.0, 2.0);
        let zero = Complex64::ZERO;
        let result = z / zero;
        assert!(
            result.re.is_nan(),
            "real part of division by zero should be NaN"
        );
        assert!(
            result.im.is_nan(),
            "imaginary part of division by zero should be NaN"
        );
    }

    #[cfg_attr(test, test)]
    fn test_complex_exp_advanced() {
        let z = Complex32::new(0.0, f32::PI);
        let result = z.exp();
        assert_almost_eq!(result.re, -1.0, 1e-5_f32);
        assert_almost_eq!(result.im, 0.0, 1e-5_f32);
    }

    #[cfg_attr(test, test)]
    fn test_complex_ln_advanced() {
        let z = Complex32::new(-1.0, 0.0);
        let result = z.ln();
        assert_almost_eq!(result.re, 0.0, 1e-5_f32);
        assert_almost_eq!(result.im, f32::PI, 1e-5_f32);

        let log10_real = Complex64::from_real(100.0).log10();
        assert_almost_eq!(log10_real.re, 2.0, 1e-12_f64);
        assert_almost_eq!(log10_real.im, 0.0, 1e-12_f64);
    }

    #[cfg_attr(test, test)]
    fn test_complex_trig_functions_advanced() {
        let zero = Complex32::new(0.0, 0.0);
        assert_almost_eq!(zero.sin().re, 0.0, 1e-5_f32);
        assert_almost_eq!(zero.sin().im, 0.0, 1e-5_f32);

        assert_almost_eq!(zero.cos().re, 1.0, 1e-5_f32);
        assert_almost_eq!(zero.cos().im, 0.0, 1e-5_f32);

        let z = Complex64::new(f64::PI / 4.0, 0.0);
        let tan_z = z.tan();
        assert_almost_eq!(tan_z.re, 1.0, 1e-12_f64);
        assert_almost_eq!(tan_z.im, 0.0, 1e-12_f64);
    }

    #[cfg_attr(test, test)]
    fn test_complex_inverse_trigonometry_advanced() {
        let z = Complex64::new(0.5, 0.5);

        let asin_z = z.asin();
        let sin_asin = asin_z.sin();
        assert_almost_eq!(sin_asin.re, z.re, 1e-12_f64);
        assert_almost_eq!(sin_asin.im, z.im, 1e-12_f64);

        let acos_z = z.acos();
        let cos_acos = acos_z.cos();
        assert_almost_eq!(cos_acos.re, z.re, 1e-12_f64);
        assert_almost_eq!(cos_acos.im, z.im, 1e-12_f64);

        let atan_z = z.atan();
        let tan_atan = atan_z.tan();
        assert_almost_eq!(tan_atan.re, z.re, 1e-12_f64);
        assert_almost_eq!(tan_atan.im, z.im, 1e-12_f64);
    }

    #[cfg_attr(test, test)]
    fn test_complex_uncovered_methods_advanced() {
        use crate::math::num_traits::Signed;
        let z1 = Complex::<f64>::new(-3.0, -4.0);
        let z2 = Complex::<f64>::new(1.0, 2.0);

        let try_add_res = z1.try_add(&z2).unwrap();
        assert_almost_eq!(try_add_res.re, -2.0);
        assert_almost_eq!(try_add_res.im, -2.0);

        let try_sub_res = z1.try_sub(&z2).unwrap();
        assert_almost_eq!(try_sub_res.re, -4.0);
        assert_almost_eq!(try_sub_res.im, -6.0);

        let try_mul_res = z1.try_mul(&z2).unwrap();
        assert_almost_eq!(try_mul_res.re, 5.0);
        assert_almost_eq!(try_mul_res.im, -10.0);

        let try_div_res = z1.try_div(&z2).unwrap();
        assert_almost_eq!(try_div_res.re, -11.0 / 5.0);
        assert_almost_eq!(try_div_res.im, 2.0 / 5.0);

        let abs_res = z1.abs();
        assert_almost_eq!(abs_res.re, 3.0);
        assert_almost_eq!(abs_res.im, 4.0);
    }
}

#[cfg_attr(not(test), control_rs_macros::hil_suite)]
/// Basic creation, polar conversion, axioms, basic arithmetic, and FFI layout tests.
pub mod complex_num_basic {
    use crate::{
        assert_almost_eq,
        math::{
            complex_num::{Complex, Complex32, Complex64},
            num_traits::{Field, One, Ring, Trig, Zero},
            ops::{
                Neg, TryAdd, TryDiv, TryMul, TrySub, WrappingAdd, WrappingMul,
                WrappingSub,
            },
        },
    };
    use core::mem;

    #[cfg_attr(test, test)]
    fn test_complex_new_basic() {
        let z = Complex32::new(1.0, 2.0);
        assert_almost_eq!(z.re, 1.0_f32);
        assert_almost_eq!(z.im, 2.0_f32);
    }

    #[cfg_attr(test, test)]
    fn test_complex_from_real_basic() {
        let z = Complex32::from_real(5.0);
        assert_almost_eq!(z.re, 5.0);
        assert_almost_eq!(z.im, 0.0);
    }

    #[cfg_attr(test, test)]
    fn test_complex_from_imag_basic() {
        let z = Complex32::from_imag(5.0);
        assert_almost_eq!(z.re, 0.0);
        assert_almost_eq!(z.im, 5.0);
    }

    #[cfg_attr(test, test)]
    fn test_complex_polar_creation_basic() {
        let r = 2.0_f32;
        let theta = f32::PI / 4.0_f32;
        let z = Complex32::from_polar(r, theta);
        let expected = r * (f32::PI / 4.0_f32).cos(); // Both re and im should be this
        assert_almost_eq!(z.re, expected);
        assert_almost_eq!(z.im, expected);
    }

    #[cfg_attr(test, test)]
    fn test_complex_polar_conversion_basic() {
        let z = Complex64::new(3.0, 4.0);
        let (r, theta) = z.to_polar();
        let z_reconstructed = Complex64::from_polar(r, theta);

        assert_almost_eq!(z.re, z_reconstructed.re);
        assert_almost_eq!(z.im, z_reconstructed.im);
    }

    #[cfg_attr(test, test)]
    fn test_complex_size_alignment_basic() {
        assert_eq!(size_of::<Complex64>(), 16);
        assert_eq!(align_of::<Complex64>(), 8);
    }

    #[cfg_attr(test, test)]
    fn test_complex_memory_representation_basic() {
        let z = Complex64::new(f64::PI, -2.71);
        let array: [f64; 2] = unsafe { mem::transmute(z) };
        assert_almost_eq!(array[0], f64::PI);
        assert_almost_eq!(array[1], -2.71);
    }

    #[cfg_attr(test, test)]
    fn test_complex_addition_commutativity_basic() {
        let z1 = Complex64::new(1.2, 3.4);
        let z2 = Complex64::new(-5.6, 7.8);
        assert_almost_eq!((z1 + z2).re, (z2 + z1).re);
        assert_almost_eq!((z1 + z2).im, (z2 + z1).im);
    }

    #[cfg_attr(test, test)]
    fn test_complex_multiplication_commutativity_basic() {
        let z1 = Complex64::new(1.2, 3.4);
        let z2 = Complex64::new(-5.6, 7.8);
        assert_almost_eq!((z1 * z2).re, (z2 * z1).re);
        assert_almost_eq!((z1 * z2).im, (z2 * z1).im);
    }

    #[cfg_attr(test, test)]
    fn test_complex_distributivity_basic() {
        let z1 = Complex64::new(1.0, 2.0);
        let z2 = Complex64::new(3.0, 4.0);
        let z3 = Complex64::new(5.0, 6.0);
        let left = z1 * (z2 + z3);
        let right = (z1 * z2) + (z1 * z3);
        assert_almost_eq!(left.re, right.re);
        assert_almost_eq!(left.im, right.im);
    }

    #[cfg_attr(test, test)]
    fn test_complex_identities_basic() {
        let z = Complex64::new(4.2, -1.1);
        let zero = Complex64::new(0.0, 0.0);
        let one = Complex64::new(1.0, 0.0);

        assert_almost_eq!((z + zero).re, z.re);
        assert_almost_eq!((z * one).re, z.re);
        assert_almost_eq!((z + zero).im, z.im);
        assert_almost_eq!((z * one).im, z.im);
    }

    #[cfg_attr(test, test)]
    fn test_complex_comparisons_basic() {
        let z1 = Complex32::new(1.0, 2.0);
        let z2 = Complex32::new(2.0, 2.0);
        let z3 = Complex32::new(1.0, 3.0);
        let z4 = Complex32::new(2.0, 3.0);

        assert!(z1 < z2);
        assert!(z1 < z3);
        assert!(z1 < z4);
        assert!(z2 > z1);
        assert!(z3 > z1);
        assert!(z4 > z1);
    }

    #[cfg_attr(test, test)]
    fn test_complex_arithmetic_basic() {
        let z1 = Complex64::new(1.0, 2.0);
        let z2 = Complex64::new(3.0, 4.0);

        let sum = z1 + z2;
        assert_almost_eq!(sum.re, 4.0);
        assert_almost_eq!(sum.im, 6.0);

        let diff = z1 - z2;
        assert_almost_eq!(diff.re, -2.0);
        assert_almost_eq!(diff.im, -2.0);

        let prod = z1 * z2;
        assert_almost_eq!(prod.re, -5.0);
        assert_almost_eq!(prod.im, 10.0);

        let div = z1 / z2;
        assert_almost_eq!(div.re, 11.0 / 25.0);
        assert_almost_eq!(div.im, 2.0 / 25.0);
    }

    #[cfg_attr(test, test)]
    fn test_complex_wrapping_ops_basic() {
        let z1 = Complex::<u8>::new(250, 10);
        let z2 = Complex::<u8>::new(10, 250);

        let w_add = z1.wrapping_add(&z2);
        assert_eq!(w_add.re, 4);
        assert_eq!(w_add.im, 4);

        let w_sub = z1.wrapping_sub(&z2);
        assert_eq!(w_sub.re, 240);
        assert_eq!(w_sub.im, 16);

        let z3 = Complex::<u8>::new(16, 16);
        let w_mul = z3.wrapping_mul(&z3);
        assert_eq!(w_mul.re, 0);
        assert_eq!(w_mul.im, 0);
    }

    #[cfg_attr(test, test)]
    fn test_complex_try_ops_basic() {
        let z1 = Complex::<u8>::new(200, 10);
        let z2 = Complex::<u8>::new(100, 10);

        assert!(z1.try_add(&z2).is_err());

        let z3 = Complex::<u8>::new(100, 10);
        assert!(z3.try_add(&z2).is_ok());
        assert!(z2.try_sub(&z1).is_err());
        assert!(z1.try_sub(&z3).is_ok());
        assert!(z1.try_mul(&z2).is_err());
        assert_eq!(
            Complex::<u8>::ONE.try_div(&Complex::<u8>::ONE).ok(),
            Some(Complex::<u8>::ONE)
        );

        let z_small = Complex::<u8>::new(2, 2);
        assert!(z_small.try_mul(&z_small).is_ok());

        let z4 = Complex::<u8>::new(100, 100);
        let z5 = Complex::<u8>::new(2, 2);
        assert!(z4.try_div(&z5).is_err());

        assert_eq!(Complex::<i8>::ONE.neg(), Complex::new(-1, 0));
    }

    #[cfg_attr(test, test)]
    fn test_complex_fallible_traits_basic() {
        let a = Complex::new(5.0, 3.0);
        let b = Complex::new(5.0, 1.0);

        assert!(a.try_sub(&b).is_ok());
        assert_eq!(a.partial_cmp(&b), Some(core::cmp::Ordering::Greater));

        let _ = Complex::<f64>::ZERO;
        let _ = Complex::<f64>::ONE;
        let _ = Complex::<f64>::MAX;
    }

    #[cfg_attr(test, test)]
    fn test_complex_epsilon_basic() {
        let eps = Complex::<f32>::epsilon();
        assert_almost_eq!(eps.re, f32::EPSILON);
        assert_almost_eq!(eps.im, 0.0);
    }
}
