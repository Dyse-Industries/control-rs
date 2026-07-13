//! # Numerical Tests
//!
//! These tests cover `[num_traits]` and `[num_types]`.
#![allow(
    unused_imports,
    clippy::arbitrary_source_item_ordering,
    clippy::arithmetic_side_effects
)]

use crate::math::CartesianQuadrant2D;
use crate::math::num_traits::{Field, One, Real, Ring, Scalar, Signed, Zero};

#[cfg_attr(not(test), control_rs_macros::hil_suite)]
/// Basic numerical trait properties including `Scalar`, `Ring`, `CartesianPlane`, identities, and markers.
pub mod num_traits_basic {
    use super::*;

    fn _scalar_property_check<T: Scalar + Zero + One + core::fmt::Debug>() {
        assert!(T::ZERO.is_zero());
        assert!(!T::ONE.is_zero());
        assert!(!T::ZERO.is_one());
        assert!(T::ONE.is_one());
    }

    fn _signed_property_check<T: Signed + Zero + One + core::fmt::Debug>() {
        assert!(T::ONE.is_sign_positive());
        assert!(!T::ONE.is_sign_negative());
        assert!(!T::ZERO.is_sign_positive());
        assert!(!T::ZERO.is_sign_negative());
        assert!(T::ONE.neg().is_sign_negative());
        assert!(!T::ONE.neg().is_sign_positive());
        assert_eq!(T::ONE.abs(), T::ONE);
        assert_eq!(T::ONE.neg().abs(), T::ONE);
        assert_eq!(T::ZERO.abs(), T::ZERO);
        _scalar_property_check::<T>();
    }

    #[cfg_attr(test, test)]
    fn test_num_traits_scalar_properties_basic() {
        _signed_property_check::<f32>();
        _signed_property_check::<f64>();
        _signed_property_check::<i8>();
        _signed_property_check::<i16>();
        _signed_property_check::<i32>();
        _signed_property_check::<i64>();
        _signed_property_check::<i128>();
        _signed_property_check::<isize>();
        _scalar_property_check::<u8>();
        _scalar_property_check::<u16>();
        _scalar_property_check::<u32>();
        _scalar_property_check::<u64>();
        _scalar_property_check::<u128>();
        _scalar_property_check::<usize>();
    }

    fn _check_int_ring<T: Ring + core::fmt::Debug>(a: T, b: T, c: T) {
        assert_eq!(a.clone() + T::zero(), a);
        assert_eq!(a.clone() * T::one(), a);
        assert_eq!(
            (a.clone() + b.clone()) + c.clone(),
            a.clone() + (b.clone() + c.clone())
        );
        let left = a.clone() * (b.clone() + c.clone());
        let right = (a.clone() * b) + (a * c);
        assert_eq!(left, right);
        assert_eq!(T::from_const::<2>(), T::TWO);
        assert_eq!(T::from_usize(2), T::ONE + T::ONE);
    }

    fn _check_float_ring<
        T: Real
            + crate::math::ops::TrySub
            + crate::math::ops::TryMul
            + core::fmt::Debug,
    >(
        a: T,
        b: T,
        c: T,
    ) {
        crate::assert_almost_eq!(a.clone() + T::zero(), a);
        crate::assert_almost_eq!(a.clone() * T::one(), a);
        crate::assert_almost_eq!(
            (a.clone() + b.clone()) + c.clone(),
            a.clone() + (b.clone() + c.clone())
        );
        let left = a.clone() * (b.clone() + c.clone());
        let right = (a.clone() * b) + (a * c);
        crate::assert_almost_eq!(left, right);
        crate::assert_almost_eq!(T::from_const::<2>(), T::TWO);
        crate::assert_almost_eq!(T::from_usize(2), T::ONE + T::ONE);
    }

    #[cfg_attr(test, test)]
    fn test_num_traits_rings_basic() {
        _check_int_ring(3_i8, 4_i8, 5_i8);
        _check_int_ring(0_i16, 10_i16, -5_i16);
        _check_float_ring(2.0f32, 3.0f32, 4.0f32);
        _check_float_ring(2.0f64, 3.0f64, 4.0f64);
    }

    #[cfg_attr(test, test)]
    fn test_num_traits_cartesian_plane_basic() {
        assert_eq!(
            CartesianQuadrant2D::from_coords(&0.0, &0.0),
            CartesianQuadrant2D::Origin
        );
        assert_eq!(
            CartesianQuadrant2D::from_coords(&1.0, &0.0),
            CartesianQuadrant2D::PositiveXAxis
        );
        assert_eq!(
            CartesianQuadrant2D::from_coords(&-1.0, &0.0),
            CartesianQuadrant2D::NegativeXAxis
        );
        assert_eq!(
            CartesianQuadrant2D::from_coords(&0.0, &1.0),
            CartesianQuadrant2D::PositiveYAxis
        );
        assert_eq!(
            CartesianQuadrant2D::from_coords(&0.0, &-1.0),
            CartesianQuadrant2D::NegativeYAxis
        );
        assert_eq!(
            CartesianQuadrant2D::from_coords(&1.0, &1.0),
            CartesianQuadrant2D::Q1
        );
        assert_eq!(
            CartesianQuadrant2D::from_coords(&-1.0, &-1.0),
            CartesianQuadrant2D::Q3
        );
        assert_eq!(
            CartesianQuadrant2D::from_coords(&-1.0, &1.0),
            CartesianQuadrant2D::Q2
        );
        assert_eq!(
            CartesianQuadrant2D::from_coords(&1.0, &-1.0),
            CartesianQuadrant2D::Q4
        );
    }

    #[cfg_attr(test, test)]
    fn test_num_traits_identities_basic() {
        assert!(1.0f32.is_one());
        assert!(!2.0f32.is_one());
        assert!(0.0f32.is_zero());
        assert!(!1.0f32.is_zero());

        assert!(1i32.is_one());
        assert!(!2i32.is_one());
        assert!(0i32.is_zero());
        assert!(!1i32.is_zero());
    }

    #[cfg_attr(test, test)]
    fn test_num_traits_sign_checks_basic() {
        assert!(1.0f32.is_sign_positive());
        assert!(!(-1.0f32).is_sign_positive());
        assert!(0.0f32.is_sign_positive());

        assert!((-1.0f32).is_sign_negative());
        assert!(!1.0f32.is_sign_negative());
        assert!(!0.0f32.is_sign_negative());
    }

    #[cfg_attr(test, test)]
    fn test_num_traits_integer_abs_basic() {
        assert_eq!((i8::MIN + 1_i8).abs(), i8::MAX);
        assert_eq!((i16::MIN + 1_i16).abs(), i16::MAX);
        assert_eq!((i32::MIN + 1_i32).abs(), i32::MAX);
        assert_eq!((i64::MIN + 1_i64).abs(), i64::MAX);
        assert_eq!((i128::MIN + 1_i128).abs(), i128::MAX);
        assert_eq!((isize::MIN + 1).abs(), isize::MAX);

        assert_eq!((-1i8).abs(), 1i8);
        assert_eq!((-1i16).abs(), 1i16);
        assert_eq!((-1i32).abs(), 1i32);
        assert_eq!((-1i64).abs(), 1i64);
        assert_eq!((-1i128).abs(), 1i128);
        assert_eq!((-1isize).abs(), 1isize);
    }

    #[cfg_attr(test, test)]
    fn test_num_traits_cartesian_quadrants_basic() {
        assert_eq!(
            CartesianQuadrant2D::from_coords(&1.0f32, &1.0),
            CartesianQuadrant2D::Q1
        );
        assert_eq!(
            CartesianQuadrant2D::from_coords(&-1.0f32, &1.0),
            CartesianQuadrant2D::Q2
        );
        assert_eq!(
            CartesianQuadrant2D::from_coords(&-1.0f32, &-1.0),
            CartesianQuadrant2D::Q3
        );
        assert_eq!(
            CartesianQuadrant2D::from_coords(&1.0f32, &-1.0),
            CartesianQuadrant2D::Q4
        );

        assert_eq!(
            CartesianQuadrant2D::from_coords(&f32::NAN, &1.0),
            CartesianQuadrant2D::Undefined
        );
    }

    #[cfg_attr(test, test)]
    fn test_num_traits_unsigned_scalar_markers_basic() {
        fn assert_is_unsigned_scalar<
            U: crate::math::num_traits::Unsigned + Scalar,
        >() {
        }
        assert_is_unsigned_scalar::<u8>();
        assert_is_unsigned_scalar::<u16>();
        assert_is_unsigned_scalar::<u32>();
        assert_is_unsigned_scalar::<u64>();
    }

    #[cfg_attr(test, test)]
    fn test_num_traits_axis_coords_compilation_coverage_basic() {
        assert_eq!(
            CartesianQuadrant2D::from_coords(&0.0f32, &1.0f32),
            CartesianQuadrant2D::PositiveYAxis
        );
        assert_eq!(
            CartesianQuadrant2D::from_coords(&0.0f32, &-1.0f32),
            CartesianQuadrant2D::NegativeYAxis
        );
        assert_eq!(
            CartesianQuadrant2D::from_coords(&1.0f32, &0.0f32),
            CartesianQuadrant2D::PositiveXAxis
        );
        assert_eq!(
            CartesianQuadrant2D::from_coords(&-1.0f32, &0.0f32),
            CartesianQuadrant2D::NegativeXAxis
        );

        assert_eq!(
            CartesianQuadrant2D::from_coords(&0.0f64, &1.0f64),
            CartesianQuadrant2D::PositiveYAxis
        );
        assert_eq!(
            CartesianQuadrant2D::from_coords(&0.0f64, &-1.0f64),
            CartesianQuadrant2D::NegativeYAxis
        );
        assert_eq!(
            CartesianQuadrant2D::from_coords(&1.0f64, &0.0f64),
            CartesianQuadrant2D::PositiveXAxis
        );
        assert_eq!(
            CartesianQuadrant2D::from_coords(&-1.0f64, &0.0f64),
            CartesianQuadrant2D::NegativeXAxis
        );
    }
}

#[cfg_attr(not(test), control_rs_macros::hil_suite)]
/// Advanced numerical trait properties including reals, transcendentals, exponentials, hypot, and try operations.
pub mod num_traits_advanced {
    use super::*;
    use crate::math::num_traits::{Exponential, Radical, Trig};

    fn _radical_property_check<
        T: Radical
            + Real
            + Zero
            + One
            + crate::math::ops::TrySub
            + crate::math::ops::TryMul
            + core::fmt::Debug,
    >() {
        crate::assert_almost_eq!(T::sqrt(T::TWO + T::TWO), T::TWO);
    }

    fn _radical_property_panic_check<
        T: Radical
            + Real
            + Zero
            + One
            + crate::math::ops::TrySub
            + crate::math::ops::TryMul
            + core::fmt::Debug,
    >() {
        crate::assert_almost_eq!(T::sqrt(T::ONE.neg()), T::NAN);
    }

    fn _exponential_property_check<
        T: Exponential
            + Real
            + Zero
            + One
            + crate::math::ops::TrySub
            + crate::math::ops::TryMul
            + core::fmt::Debug,
    >() {
        crate::assert_almost_eq!(<T as Exponential>::exp(T::ONE), T::E);
        crate::assert_almost_eq!(<T as Exponential>::ln(T::E), T::ONE);
        crate::assert_almost_eq!(
            <T as Exponential>::log10(T::from_const::<10>()),
            T::ONE
        );
        crate::assert_almost_eq!(
            <T as Exponential>::pow(T::TWO, T::TWO),
            T::TWO + T::TWO
        );
    }

    fn _trig_property_check<
        T: Trig
            + Real
            + Signed
            + Zero
            + One
            + crate::math::ops::TrySub
            + crate::math::ops::TryMul
            + core::fmt::Debug,
    >() {
        crate::assert_almost_eq!(<T as Trig>::cos(T::PI), T::ONE.neg());
        crate::assert_almost_eq!(<T as Trig>::sin(T::PI), T::ZERO);
        crate::assert_almost_eq!(<T as Trig>::tan(T::ZERO), T::ZERO);
        crate::assert_almost_eq!(<T as Trig>::acos(T::ZERO), T::PI / T::TWO);
        crate::assert_almost_eq!(<T as Trig>::asin(T::ZERO), T::ZERO);
        crate::assert_almost_eq!(
            <T as Trig>::atan(T::ONE),
            T::PI / (T::TWO + T::TWO)
        );
        crate::assert_almost_eq!(<T as Real>::atan2(T::ZERO, T::ZERO), T::ZERO);
        crate::assert_almost_eq!(<T as Real>::atan2(T::ZERO, T::ONE), T::ZERO);
        crate::assert_almost_eq!(
            <T as Real>::atan2(T::ZERO, T::ONE.neg()),
            T::PI
        );
        crate::assert_almost_eq!(
            <T as Real>::atan2(T::ONE, T::ONE),
            T::PI / (T::TWO + T::TWO)
        );
        crate::assert_almost_eq!(
            <T as Real>::atan2(T::ONE, T::ZERO),
            T::PI / T::TWO
        );
        crate::assert_almost_eq!(
            <T as Real>::atan2(T::ONE.neg(), T::ZERO),
            T::PI.neg() / T::TWO
        );
    }

    #[allow(clippy::eq_op)]
    fn _real_property_check<
        T: Real
            + Signed
            + Zero
            + One
            + crate::math::ops::TrySub
            + crate::math::ops::TryMul
            + core::fmt::Debug,
    >() {
        assert_ne!(T::NAN, T::NAN);
        assert_eq!(T::INF, T::INF);
        crate::assert_almost_eq!(T::epsilon() / T::TWO, T::ZERO);

        _radical_property_check::<T>();
        _exponential_property_check::<T>();
        _trig_property_check::<T>();
    }

    #[cfg_attr(test, test)]
    fn test_num_traits_real_properties_advanced() {
        _real_property_check::<f32>();
        _real_property_check::<f64>();
    }

    #[cfg(test)]
    #[test]
    #[should_panic(expected = "Input is outside the mathematical domain")]
    fn _test_num_traits_real_f32_panics_advanced() {
        _radical_property_panic_check::<f32>();
    }

    #[cfg(test)]
    #[test]
    #[should_panic(expected = "Input is outside the mathematical domain")]
    fn _test_num_traits_real_f64_panics_advanced() {
        _radical_property_panic_check::<f64>();
    }

    #[cfg_attr(test, test)]
    fn test_num_traits_hypot_advanced() {
        let a = 3.0f32;
        let b = 4.0f32;
        crate::assert_almost_eq!(a.hypot(b), 5.0);
    }

    #[cfg_attr(test, test)]
    fn test_num_traits_atan2_advanced() {
        crate::assert_almost_eq!(0.0f32.atan2(0.0), 0.0);
        crate::assert_almost_eq!(0.0f32.atan2(1.0), 0.0);
        crate::assert_almost_eq!(0.0f32.atan2(-1.0), core::f32::consts::PI);
        crate::assert_almost_eq!(
            1.0f32.atan2(0.0),
            core::f32::consts::PI / 2.0
        );
        crate::assert_almost_eq!(
            -1.0f32.atan2(0.0),
            -core::f32::consts::PI / 2.0
        );
        crate::assert_almost_eq!(
            1.0f32.atan2(1.0),
            core::f32::consts::PI / 4.0
        );
        crate::assert_almost_eq!(
            1.0f32.atan2(-1.0),
            3.0 * core::f32::consts::PI / 4.0
        );
        crate::assert_almost_eq!(
            -1.0f32.atan2(-1.0),
            -3.0 * core::f32::consts::PI / 4.0
        );
        crate::assert_almost_eq!(
            -1.0f32.atan2(1.0),
            -core::f32::consts::PI / 4.0
        );
    }

    #[cfg_attr(test, test)]
    fn test_num_traits_hyperbolic_functions_advanced() {
        crate::assert_almost_eq!(0.0f32.cosh(), 1.0);
        let cosh_val = 1.0f32.cosh();
        let expected_cosh = f32::midpoint(1.0f32.exp(), (-1.0f32).exp());
        assert!((cosh_val - expected_cosh).abs() < 1e-6);

        crate::assert_almost_eq!(0.0f32.sinh(), 0.0);
        let sinh_val_neg = (-1.0f32).sinh();
        let expected_sinh_neg = -((1.0f32.exp() - (-1.0f32).exp()) / 2.0);
        assert!((sinh_val_neg - expected_sinh_neg).abs() < 1e-6);
    }

    #[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
    struct TestReal(f32);

    impl core::ops::Add for TestReal {
        type Output = Self;
        fn add(self, other: Self) -> Self {
            Self(self.0 + other.0)
        }
    }
    impl core::ops::Sub for TestReal {
        type Output = Self;
        fn sub(self, other: Self) -> Self {
            Self(self.0 - other.0)
        }
    }
    impl core::ops::Mul for TestReal {
        type Output = Self;
        fn mul(self, other: Self) -> Self {
            Self(self.0 * other.0)
        }
    }
    impl core::ops::Div for TestReal {
        type Output = Self;
        fn div(self, other: Self) -> Self {
            Self(self.0 / other.0)
        }
    }
    impl core::ops::Neg for TestReal {
        type Output = Self;
        fn neg(self) -> Self {
            Self(-self.0)
        }
    }
    impl Scalar for TestReal {}
    impl One for TestReal {
        const ONE: Self = Self(1.0);
    }
    impl Zero for TestReal {
        const ZERO: Self = Self(0.0);
    }
    impl Ring for TestReal {
        const MAX: Self = Self(f32::MAX);
        const MIN: Self = Self(f32::MIN);
        const MIN_POSITIVE: Self = Self(f32::MIN_POSITIVE);
        const TWO: Self = Self(2.0);
    }
    impl Field for TestReal {
        fn epsilon() -> Self {
            Self(f32::EPSILON)
        }
    }
    impl Signed for TestReal {
        fn abs(self) -> Self {
            Self(self.0.abs())
        }
    }
    impl Radical for TestReal {
        fn sqrt(self) -> Self {
            Self(libm::sqrtf(self.0))
        }
    }
    impl Exponential for TestReal {
        const E: Self = Self(core::f32::consts::E);
        fn exp(self) -> Self {
            Self(libm::expf(self.0))
        }
        fn ln(self) -> Self {
            Self(libm::logf(self.0))
        }
        fn log10(self) -> Self {
            Self(libm::log10f(self.0))
        }
        fn pow(self, n: Self) -> Self {
            Self(libm::powf(self.0, n.0))
        }
    }
    impl Trig for TestReal {
        const PI: Self = Self(core::f32::consts::PI);
        fn acos(self) -> Self {
            Self(libm::acosf(self.0))
        }
        fn asin(self) -> Self {
            Self(libm::asinf(self.0))
        }
        fn atan(self) -> Self {
            Self(libm::atanf(self.0))
        }
        fn cos(self) -> Self {
            Self(libm::cosf(self.0))
        }
        fn sin(self) -> Self {
            Self(libm::sinf(self.0))
        }
        fn tan(self) -> Self {
            Self(libm::tanf(self.0))
        }
    }
    impl Real for TestReal {
        const INF: Self = Self(f32::INFINITY);
        const NAN: Self = Self(f32::NAN);
    }

    #[cfg_attr(test, test)]
    fn test_num_traits_default_atan2_advanced() {
        assert_eq!(TestReal(0.0).atan2(TestReal(0.0)), TestReal(0.0));
        assert_eq!(TestReal(0.0).atan2(TestReal(1.0)), TestReal(0.0));
        assert_eq!(
            TestReal(-1.0).atan2(TestReal(0.0)),
            -TestReal::PI / TestReal::TWO
        );
        assert_eq!(TestReal(0.0).atan2(TestReal(-1.0)), TestReal::PI);
        assert_eq!(
            TestReal(1.0).atan2(TestReal(0.0)),
            TestReal::PI / TestReal::TWO
        );
        assert_eq!(TestReal(1.0).atan2(TestReal(1.0)), TestReal(1.0).atan());
    }

    #[cfg_attr(test, test)]
    fn test_num_traits_complex_try_ops_ordering_advanced() {
        use crate::math::complex_num::Complex;
        use crate::math::ops::{TryAdd, TryDiv, TryMul, TrySub};

        let c1 = Complex::new(4.0f32, 2.0f32);
        let c2 = Complex::new(2.0f32, 1.0f32);

        let sum = c1.try_add(&c2).unwrap();
        assert_eq!(sum, Complex::new(6.0, 3.0));

        let diff = c1.try_sub(&c2).unwrap();
        assert_eq!(diff, Complex::new(2.0, 1.0));

        let prod = c1.try_mul(&c2).unwrap();
        assert_eq!(prod, Complex::new(6.0, 8.0));

        let quot = c1.try_div(&c2).unwrap();
        assert_eq!(quot, Complex::new(2.0, 0.0));

        let c3 = Complex::new(4.0f32, 3.0f32);
        assert!(c3 > c1);
        assert!(c1 < c3);
        assert_eq!(c1.partial_cmp(&c1), Some(core::cmp::Ordering::Equal));
    }
}
