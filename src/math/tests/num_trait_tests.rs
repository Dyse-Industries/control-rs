//! # Numerical Tests
//!
//! These tests cover `[num_traits]` and `[num_types]`.
#![allow(
    unused_imports,
    clippy::arbitrary_source_item_ordering,
    clippy::arithmetic_side_effects
)]

use crate::math::num_traits::{Field, One, Real, Ring, Scalar, Signed, Zero};

mod scalar_tests {
    use crate::math::num_traits::{One, Scalar, Signed, Zero};
    fn scalar_property_check<T: Scalar + Zero + One + core::fmt::Debug>() {
        assert!(T::ZERO.is_zero());
        assert!(!T::ONE.is_zero());
        assert!(!T::ZERO.is_one());
        assert!(T::ONE.is_one());
    }
    #[allow(clippy::arithmetic_side_effects)]
    fn signed_property_check<T: Signed + Zero + One + core::fmt::Debug>() {
        assert!(T::ONE.is_sign_positive());
        assert!(!T::ONE.is_sign_negative());
        assert!(!T::ZERO.is_sign_positive());
        assert!(!T::ZERO.is_sign_negative());
        assert!(T::ONE.neg().is_sign_negative());
        assert!(!T::ONE.neg().is_sign_positive());
        assert_eq!(T::ONE.abs(), T::ONE);
        assert_eq!(T::ONE.neg().abs(), T::ONE);
        assert_eq!(T::ZERO.abs(), T::ZERO);
        scalar_property_check::<T>();
    }
    #[test]
    fn test_scalar_properties() {
        signed_property_check::<f32>();
        signed_property_check::<f64>();
        signed_property_check::<i8>();
        signed_property_check::<i16>();
        signed_property_check::<i32>();
        signed_property_check::<i64>();
        signed_property_check::<i128>();
        signed_property_check::<isize>();
        scalar_property_check::<u8>();
        scalar_property_check::<u16>();
        scalar_property_check::<u32>();
        scalar_property_check::<u64>();
        scalar_property_check::<u128>();
        scalar_property_check::<usize>();
    }
}
mod ring_tests {
    use crate::{
        assert_almost_eq,
        math::{
            num_traits::{Real, Ring},
            ops::{TryMul, TrySub},
        },
    };
    fn check_int_ring<T: Ring + core::fmt::Debug>(a: T, b: T, c: T) {
        // Identity: a + 0 = a
        assert_eq!(a.clone() + T::zero(), a);
        // Identity: a * 1 = a
        assert_eq!(a.clone() * T::one(), a);
        // Associativity: (a + b) + c = a + (b + c)
        assert_eq!(
            (a.clone() + b.clone()) + c.clone(),
            a.clone() + (b.clone() + c.clone())
        );
        // Distributivity: a * (b + c) = a*b + a*c
        let left = a.clone() * (b.clone() + c.clone());
        let right = (a.clone() * b) + (a * c);
        // Note: For floats, exact equality might fail due to precision,
        // but for integers (Ring) it must hold exactly.
        assert_eq!(left, right);
        assert_eq!(T::from_const::<2>(), T::TWO);
        assert_eq!(T::from_usize(2), T::ONE + T::ONE);
    }
    // Real gives access to the assert_almost_eq! macro.
    fn check_float_ring<T: Real + TrySub + TryMul + core::fmt::Debug>(
        a: T,
        b: T,
        c: T,
    ) {
        // Identity: a + 0 = a
        assert_almost_eq!(a.clone() + T::zero(), a);
        // Identity: a * 1 = a
        assert_almost_eq!(a.clone() * T::one(), a);
        // Associativity: (a + b) + c = a + (b + c)
        assert_almost_eq!(
            (a.clone() + b.clone()) + c.clone(),
            a.clone() + (b.clone() + c.clone())
        );
        // Distributivity: a * (b + c) = a*b + a*c
        let left = a.clone() * (b.clone() + c.clone());
        let right = (a.clone() * b) + (a * c);
        // Note: For floats, exact equality might fail due to precision,
        // but for integers (Ring) it must hold exactly.
        assert_almost_eq!(left, right);
        assert_almost_eq!(T::from_const::<2>(), T::TWO);
        assert_almost_eq!(T::from_usize(2), T::ONE + T::ONE);
    }
    #[test]
    fn test_rings() {
        check_int_ring(3_i8, 4_i8, 5_i8);
        check_int_ring(0_i16, 10_i16, -5_i16);
        check_float_ring(2.0f32, 3.0f32, 4.0f32);
        check_float_ring(2.0f64, 3.0f64, 4.0f64);
    }
}

mod cartesian_plane_tests {
    use crate::math::CartesianQuadrant2D;
    #[test]
    fn test_cartesian_plane() {
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
}

mod real_tests {
    use crate::{
        assert_almost_eq,
        math::{
            num_traits::{Exponential, One, Radical, Real, Signed, Trig, Zero},
            ops::{TryMul, TrySub},
        },
    };

    fn radical_property_check<
        T: Radical + Real + Zero + One + TrySub + TryMul + core::fmt::Debug,
    >() {
        assert_almost_eq!(T::sqrt(T::TWO + T::TWO), T::TWO);
    }

    fn radical_property_panic_check<
        T: Radical + Real + Zero + One + TrySub + TryMul + core::fmt::Debug,
    >() {
        assert_almost_eq!(T::sqrt(T::ONE.neg()), T::NAN);
    }

    fn exponential_property_check<
        T: Exponential + Real + Zero + One + TrySub + TryMul + core::fmt::Debug,
    >() {
        assert_almost_eq!(<T as Exponential>::exp(T::ONE), T::E);
        assert_almost_eq!(<T as Exponential>::ln(T::E), T::ONE);
        assert_almost_eq!(
            <T as Exponential>::log10(T::from_const::<10>()),
            T::ONE
        );
        assert_almost_eq!(
            <T as Exponential>::pow(T::TWO, T::TWO),
            T::TWO + T::TWO
        );
    }

    #[allow(clippy::eq_op)]
    fn trig_property_check<
        T: Trig + Real + Signed + Zero + One + TrySub + TryMul + core::fmt::Debug,
    >() {
        assert_almost_eq!(<T as Trig>::cos(T::PI), T::ONE.neg());
        assert_almost_eq!(<T as Trig>::sin(T::PI), T::ZERO);
        assert_almost_eq!(<T as Trig>::tan(T::ZERO), T::ZERO);
        assert_almost_eq!(<T as Trig>::acos(T::ZERO), T::PI / T::TWO);
        assert_almost_eq!(<T as Trig>::asin(T::ZERO), T::ZERO);
        assert_almost_eq!(<T as Trig>::atan(T::ONE), T::PI / (T::TWO + T::TWO));
        assert_almost_eq!(<T as Real>::atan2(T::ZERO, T::ZERO), T::ZERO);
        assert_almost_eq!(<T as Real>::atan2(T::ZERO, T::ONE), T::ZERO);
        assert_almost_eq!(<T as Real>::atan2(T::ZERO, T::ONE.neg()), T::PI);
        assert_almost_eq!(
            <T as Real>::atan2(T::ONE, T::ONE),
            T::PI / (T::TWO + T::TWO)
        );
        assert_almost_eq!(<T as Real>::atan2(T::ONE, T::ZERO), T::PI / T::TWO);
        assert_almost_eq!(
            <T as Real>::atan2(T::ONE.neg(), T::ZERO),
            T::PI.neg() / T::TWO
        );
    }
    #[allow(clippy::eq_op)]
    fn real_property_check<
        T: Real + Signed + Zero + One + TrySub + TryMul + core::fmt::Debug,
    >() {
        assert_ne!(T::NAN, T::NAN);
        assert_eq!(T::INF, T::INF);
        assert_almost_eq!(T::epsilon() / T::TWO, T::ZERO);

        radical_property_check::<T>();
        exponential_property_check::<T>();
        trig_property_check::<T>();
    }
    #[test]
    fn test_real_properties() {
        real_property_check::<f32>();
        real_property_check::<f64>();
    }

    #[test]
    #[should_panic(expected = "Input is outside the mathematical domain")]
    fn test_real_f32_panics() {
        radical_property_panic_check::<f32>();
    }
    #[test]
    #[should_panic(expected = "Input is outside the mathematical domain")]
    fn test_real_f64_panics() {
        radical_property_panic_check::<f64>();
    }
}
mod custom_tests {
    use crate::assert_almost_eq;
    use crate::math::CartesianQuadrant2D;
    use crate::math::num_traits::{
        Exponential, Field, One, Radical, Real, Ring, Scalar, Signed, Trig,
        Unsigned, Zero,
    };

    #[test]
    fn test_identities() {
        assert!(1.0f32.is_one());
        assert!(!2.0f32.is_one());
        assert!(0.0f32.is_zero());
        assert!(!1.0f32.is_zero());

        assert!(1i32.is_one());
        assert!(!2i32.is_one());
        assert!(0i32.is_zero());
        assert!(!1i32.is_zero());
    }

    #[test]
    fn test_sign_checks() {
        assert!(1.0f32.is_sign_positive());
        assert!(!(-1.0f32).is_sign_positive());
        assert!(0.0f32.is_sign_positive());

        assert!((-1.0f32).is_sign_negative());
        assert!(!1.0f32.is_sign_negative());
        assert!(!0.0f32.is_sign_negative());
    }

    #[test]
    fn test_hypot() {
        let a = 3.0f32;
        let b = 4.0f32;
        assert_almost_eq!(a.hypot(b), 5.0);
    }

    #[test]
    fn test_atan2() {
        // Origin
        assert_almost_eq!(0.0f32.atan2(0.0), 0.0);

        // Axis Bounds
        assert_almost_eq!(0.0f32.atan2(1.0), 0.0); // Positive X
        assert_almost_eq!(0.0f32.atan2(-1.0), core::f32::consts::PI); // Negative X
        assert_almost_eq!(1.0f32.atan2(0.0), core::f32::consts::PI / 2.0); // Positive Y
        assert_almost_eq!(-1.0f32.atan2(0.0), -core::f32::consts::PI / 2.0); // Negative Y

        // Standard Quadrants
        assert_almost_eq!(1.0f32.atan2(1.0), core::f32::consts::PI / 4.0); // Q1
        assert_almost_eq!(
            1.0f32.atan2(-1.0),
            3.0 * core::f32::consts::PI / 4.0
        ); // Q2
        assert_almost_eq!(
            -1.0f32.atan2(-1.0),
            -3.0 * core::f32::consts::PI / 4.0
        ); // Q3
        assert_almost_eq!(-1.0f32.atan2(1.0), -core::f32::consts::PI / 4.0); // Q4
    }

    #[test]
    fn test_hyperbolic_functions() {
        // cosh
        assert_almost_eq!(0.0f32.cosh(), 1.0);
        let cosh_val = 1.0f32.cosh();
        let expected_cosh = f32::midpoint(1.0f32.exp(), (-1.0f32).exp());
        assert!((cosh_val - expected_cosh).abs() < 1e-6);

        // sinh
        assert_almost_eq!(0.0f32.sinh(), 0.0);
        let sinh_val_neg = (-1.0f32).sinh();
        let expected_sinh_neg = -((1.0f32.exp() - (-1.0f32).exp()) / 2.0);
        assert!((sinh_val_neg - expected_sinh_neg).abs() < 1e-6);
    }

    #[test]
    fn test_integer_abs() {
        assert_eq!((i8::MIN + 1_i8).abs(), i8::MAX); // Because abs() wraps! wait... standard integer abs returns MIN for MIN.
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

    #[test]
    fn test_cartesian_quadrants() {
        // Standard Quadrants
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

        // Undefined (NaN)
        assert_eq!(
            CartesianQuadrant2D::from_coords(&f32::NAN, &1.0),
            CartesianQuadrant2D::Undefined
        );
    }

    #[test]
    fn test_unsigned_scalar_markers() {
        // A simple compile-time check to ensure the marker traits are applied
        fn assert_is_unsigned_scalar<T: Unsigned + Scalar>() {}

        assert_is_unsigned_scalar::<u8>();
        assert_is_unsigned_scalar::<u16>();
        assert_is_unsigned_scalar::<u32>();
        assert_is_unsigned_scalar::<u64>();
        // If these compile, the lines are covered.
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

    #[test]
    fn test_default_atan2() {
        // 1. Origin
        assert_eq!(TestReal(0.0).atan2(TestReal(0.0)), TestReal(0.0));
        // 2. Positive X
        assert_eq!(TestReal(0.0).atan2(TestReal(1.0)), TestReal(0.0));
        // 3. Negative Y
        assert_eq!(
            TestReal(-1.0).atan2(TestReal(0.0)),
            -TestReal::PI / TestReal::TWO
        );
        // 4. Negative X
        assert_eq!(TestReal(0.0).atan2(TestReal(-1.0)), TestReal::PI);
        // 5. Positive Y
        assert_eq!(
            TestReal(1.0).atan2(TestReal(0.0)),
            TestReal::PI / TestReal::TWO
        );
        // 6. General case (e.g. Q1)
        assert_eq!(TestReal(1.0).atan2(TestReal(1.0)), TestReal(1.0).atan());
    }

    #[test]
    fn test_axis_coords_compilation_coverage() {
        // Test f32 axis combinations explicitly
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

        // Test f64 axis combinations explicitly
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
