//! # Numerical Tests
//!
//! These tests cover `[num_traits]`.
//!
//! ## Functional Requirement Coverage (`num-traits-design.md`)
//!
//! - **FR-1** (granular trait hierarchy: `Zero`/`One`/`AdditiveGroup`/
//!   `Signed`/`Unsigned`/`Radical`/`Exponential`/`Trig`): per-trait axiom
//!   and identity checks, plus the `Unsigned`/`Integer`/`SaturatingInteger`
//!   marker test.
//! - **FR-2** (functional containers `Integer`/`SaturatingInteger`/`Float`):
//!   `test_num_trait_integer_axioms`, `test_num_trait_float_axioms` and
//!   (in `op_tests.rs`) the saturating add/sub/mul suites.
//! - **FR-3** (unified `Scalar`): `test_num_trait_scalar_properties` and
//!   `test_num_trait_scalar_markers`; also exercised transitively via
//!   `Complex<T>: Scalar` in `complex_num_tests.rs`.
//!
//! `CartesianQuadrant2D`, hyperbolic functions and the custom-`atan2`
//! fallback tests exercise implementation details of FR-1's `Trig`/`Float`
//! traits rather than a separately numbered requirement.
#![allow(clippy::arithmetic_side_effects)]

#[cfg_attr(not(test), control_rs_macros::ets_suite)]
pub mod num_trait_test_suite {
    use crate::assert_almost_eq;
    use crate::math::CartesianQuadrant2D;
    use crate::math::complex_num::Complex;
    use crate::math::num_traits::{
        AdditiveGroup, Conjugate, Exponential, Float, Integer, One, Radical,
        SaturatingInteger, Scalar, Signed, Trig, Unsigned, Zero,
    };
    use crate::math::ops::{TryAdd, TryDiv, TryMul, TrySub};

    // --- Custom implementation for TestFloat to verify default atan2 logic ---

    #[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
    struct TestFloat(f32);

    impl core::ops::Add for TestFloat {
        type Output = Self;
        fn add(self, other: Self) -> Self {
            Self(self.0 + other.0)
        }
    }
    impl core::ops::Sub for TestFloat {
        type Output = Self;
        fn sub(self, other: Self) -> Self {
            Self(self.0 - other.0)
        }
    }
    impl core::ops::Mul for TestFloat {
        type Output = Self;
        fn mul(self, other: Self) -> Self {
            Self(self.0 * other.0)
        }
    }
    impl core::ops::Div for TestFloat {
        type Output = Self;
        fn div(self, other: Self) -> Self {
            Self(self.0 / other.0)
        }
    }
    impl core::ops::Neg for TestFloat {
        type Output = Self;
        fn neg(self) -> Self {
            Self(-self.0)
        }
    }
    impl One for TestFloat {
        const ONE: Self = Self(1.0);
    }
    impl Zero for TestFloat {
        const ZERO: Self = Self(0.0);
    }
    impl AdditiveGroup for TestFloat {}
    impl Conjugate for TestFloat {
        fn conj(self) -> Self {
            self
        }
    }
    impl Scalar for TestFloat {
        type Real = Self;
        fn from_real(re: Self::Real) -> Self {
            re
        }
        fn im(&self) -> Self::Real {
            Self::ZERO
        }
        fn re(&self) -> Self::Real {
            *self
        }
    }
    impl Signed for TestFloat {
        fn abs(self) -> Self {
            Self(self.0.abs())
        }
    }
    impl Radical for TestFloat {
        fn sqrt(self) -> Self {
            Self(libm::sqrtf(self.0))
        }
    }
    impl Exponential for TestFloat {
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
    impl Trig for TestFloat {
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
    impl Float for TestFloat {
        fn epsilon() -> Self {
            Self(f32::EPSILON)
        }
    }

    // --- Helper Functions for Tests ---

    fn _scalar_property_check<T: Scalar + Zero + One + core::fmt::Debug>() {
        assert!(T::ZERO.is_zero());
        assert!(!T::ONE.is_zero());
        assert!(!T::ZERO.is_one());
        assert!(T::ONE.is_one());
    }

    fn _signed_property_check<T: Scalar + Signed + core::fmt::Debug>() {
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

    fn _check_additive_group_axioms<T: AdditiveGroup + core::fmt::Debug>(
        a: &T,
    ) {
        // Identity: a + 0 = a
        assert_eq!(a.clone() + T::zero(), *a);
        // Inverse: a - a = 0
        assert_eq!(a.clone() - a.clone(), T::ZERO);
    }

    fn _check_integer_axioms<T: Integer + core::fmt::Debug>(a: T, b: T, c: T) {
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
        assert_eq!(left, right);
        assert_eq!(T::ONE + T::ONE, T::TWO);
    }

    fn _check_float_axioms<T: Float + TrySub + TryMul + core::fmt::Debug>(
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
        assert_almost_eq!(left, right);
        let two = T::ONE + T::ONE;
        assert_almost_eq!(T::from_const::<2>(), two);
        assert_almost_eq!(T::from_usize(2), two);
    }

    fn _radical_property_check<
        T: Radical + Float + TrySub + TryMul + core::fmt::Debug,
    >() {
        let two = T::ONE + T::ONE;
        assert_almost_eq!(T::sqrt(two.clone() + two.clone()), two);
    }

    #[allow(clippy::eq_op)]
    fn _radical_property_panic_check<
        T: Radical + Float + TrySub + TryMul + core::fmt::Debug,
    >() {
        // NaN is produced generically via 0/0, matching the IEEE-754 total
        // division `Float` relies on rather than a dedicated NAN constant.
        let nan = T::ZERO / T::ZERO;
        assert_almost_eq!(T::sqrt(T::ONE.neg()), nan);
    }

    fn _exponential_property_check<
        T: Exponential + Float + TrySub + TryMul + core::fmt::Debug,
    >() {
        let two = T::ONE + T::ONE;
        assert_almost_eq!(<T as Exponential>::exp(T::ONE), T::E);
        assert_almost_eq!(<T as Exponential>::ln(T::E), T::ONE);
        assert_almost_eq!(
            <T as Exponential>::log10(T::from_const::<10>()),
            T::ONE
        );
        assert_almost_eq!(
            <T as Exponential>::pow(two.clone(), two.clone()),
            two.clone() + two
        );
    }

    fn _trig_property_check<
        T: Trig + Float + TrySub + TryMul + core::fmt::Debug,
    >() {
        let two = T::ONE + T::ONE;
        assert_almost_eq!(<T as Trig>::cos(T::PI), T::ONE.neg());
        assert_almost_eq!(<T as Trig>::sin(T::PI), T::ZERO);
        assert_almost_eq!(<T as Trig>::tan(T::ZERO), T::ZERO);
        assert_almost_eq!(<T as Trig>::acos(T::ZERO), T::PI / two.clone());
        assert_almost_eq!(<T as Trig>::asin(T::ZERO), T::ZERO);
        assert_almost_eq!(
            <T as Trig>::atan(T::ONE),
            T::PI / (two.clone() + two.clone())
        );
        assert_almost_eq!(<T as Float>::atan2(T::ZERO, T::ZERO), T::ZERO);
        assert_almost_eq!(<T as Float>::atan2(T::ZERO, T::ONE), T::ZERO);
        assert_almost_eq!(<T as Float>::atan2(T::ZERO, T::ONE.neg()), T::PI);
        assert_almost_eq!(
            <T as Float>::atan2(T::ONE, T::ONE),
            T::PI / (two.clone() + two.clone())
        );
        assert_almost_eq!(
            <T as Float>::atan2(T::ONE, T::ZERO),
            T::PI / two.clone()
        );
        assert_almost_eq!(
            <T as Float>::atan2(T::ONE.neg(), T::ZERO),
            T::PI.neg() / two
        );
    }

    #[allow(clippy::eq_op)]
    fn _float_property_check<T: Float + TrySub + TryMul + core::fmt::Debug>() {
        // NaN/Infinity are produced generically via total IEEE-754 division
        // (0/0, 1/0) rather than dedicated NAN/INF constants (`Float`
        // deliberately carries neither — see `num-traits-design.md`).
        let nan = T::ZERO / T::ZERO;
        let inf = T::ONE / T::ZERO;
        assert_ne!(nan.clone(), nan);
        assert_eq!(inf.clone(), inf);
        let two = T::ONE + T::ONE;
        assert_almost_eq!(T::epsilon() / two, T::ZERO);

        _radical_property_check::<T>();
        _exponential_property_check::<T>();
        _trig_property_check::<T>();
    }

    // --- Test Executables ---

    #[cfg_attr(test, test)]
    /// Verifies properties of the Scalar trait across all supported
    /// primitive types (FR-3 of `num-traits-design.md`).
    fn test_num_trait_scalar_properties() {
        _signed_property_check::<f32>();
        _signed_property_check::<f64>();
        _signed_property_check::<i8>();
        _signed_property_check::<i16>();
        _signed_property_check::<i32>();
        _signed_property_check::<i64>();
        _signed_property_check::<i128>();
        _signed_property_check::<isize>();

        // Unsigned scalars: no Signed methods, but identity and signum
        // properties still hold (the negative branch is unreachable).
        _scalar_property_check::<u8>();
        _scalar_property_check::<u32>();
        _scalar_property_check::<usize>();
        assert_eq!(5_u32.signum(), 1);
        assert_eq!(0_u32.signum(), 0);
        // Qualified: `Ord::clamp` also exists for integer primitives.
        assert_eq!(Scalar::clamp(7_u8, 0, 3), 3);
    }

    #[cfg_attr(test, test)]
    /// Verifies Conjugate and Scalar projection methods across real and complex numbers (FR-3/FR-5 of `num-traits-design.md`).
    fn test_num_trait_conjugate_and_scalar_projections() {
        // Real scalars: conjugation is identity
        assert_eq!(5i32.conj(), 5i32);
        assert_almost_eq!(3.5f64.conj(), 3.5f64);
        assert_eq!(5i32.re(), 5i32);
        assert_eq!(5i32.im(), 0i32);
        assert_eq!(<i32 as Scalar>::from_real(10), 10);
        assert_eq!(3i32.abs2(), 9);

        // Complex scalars: conjugation negates imaginary component
        let z = Complex::new(3.0f64, 4.0f64);
        assert_eq!(z.conj(), Complex::new(3.0, -4.0));
        assert_almost_eq!(z.re(), 3.0);
        assert_almost_eq!(z.im(), 4.0);
        assert_eq!(Complex::<f64>::from_real(7.0), Complex::new(7.0, 0.0));
        assert_almost_eq!(z.abs2(), 25.0);
    }

    #[cfg_attr(test, test)]
    /// Verifies commutative, associative, distributive and identity axioms
    /// of the Integer trait (FR-2 of `num-traits-design.md`).
    fn test_num_trait_integer_axioms() {
        _check_integer_axioms(3_i8, 4_i8, 5_i8);
        _check_integer_axioms(0_i16, 10_i16, 5_i16);
        _check_integer_axioms(3_u8, 4_u8, 5_u8);
    }

    #[cfg_attr(test, test)]
    /// Verifies commutative, associative, distributive and identity axioms
    /// of the Float trait (FR-2 of `num-traits-design.md`).
    fn test_num_trait_float_axioms() {
        _check_float_axioms(2.0f32, 3.0f32, 4.0f32);
        _check_float_axioms(2.0f64, 3.0f64, 4.0f64);
    }

    #[cfg_attr(test, test)]
    /// Verifies the `AdditiveGroup` opt-in (identity and inverse) for signed
    /// integers, floats and `Complex<T>`, none of which are exercised by
    /// the Integer/Float axiom checks above (FR-1 of
    /// `num-traits-design.md`).
    fn test_num_trait_additive_group_axioms() {
        _check_additive_group_axioms(&3_i32);
        _check_additive_group_axioms(&3.0f32);
        _check_additive_group_axioms(&Complex::new(1.0f32, 2.0f32));
    }

    #[cfg_attr(test, test)]
    /// Verifies `CartesianQuadrant2D` coordinate parsing for cardinal directions and quadrants.
    fn test_num_trait_cartesian_plane_quadrants_legacy() {
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
    /// Verifies general Float trait properties including NAN/INF behavior
    /// and epsilons and drives the `Radical`/`Exponential`/`Trig` property
    /// checks (FR-1 + FR-2 of `num-traits-design.md`).
    fn test_num_trait_float_properties() {
        _float_property_check::<f32>();
        _float_property_check::<f64>();
    }

    #[cfg(test)]
    #[test]
    #[should_panic(expected = "Input is outside the mathematical domain")]
    fn _test_float_f32_panics() {
        _radical_property_panic_check::<f32>();
    }

    #[cfg(test)]
    #[test]
    #[should_panic(expected = "Input is outside the mathematical domain")]
    fn _test_float_f64_panics() {
        _radical_property_panic_check::<f64>();
    }

    #[cfg_attr(test, test)]
    /// Verifies standard identities (one/zero) for floating point and
    /// integer types (FR-1 of `num-traits-design.md`).
    fn test_num_trait_identities_zero_one() {
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
    /// Verifies sign-checking methods (positive/negative) for floating
    /// point types (FR-1 of `num-traits-design.md`, `Signed`).
    fn test_num_trait_sign_positive_negative() {
        assert!(1.0f32.is_sign_positive());
        assert!(!(-1.0f32).is_sign_positive());
        assert!(0.0f32.is_sign_positive());

        assert!((-1.0f32).is_sign_negative());
        assert!(!1.0f32.is_sign_negative());
        assert!(!0.0f32.is_sign_negative());
    }

    #[cfg_attr(test, test)]
    /// Verifies hypotenuse (Euclidean distance) calculations.
    fn test_num_trait_hypot_length() {
        let a = 3.0f32;
        let b = 4.0f32;
        assert_almost_eq!(a.hypot(b), 5.0);
    }

    #[cfg_attr(test, test)]
    /// Verifies custom atan2 implementation outputs across boundary angles and quadrants.
    fn test_num_trait_atan2_quadrants() {
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

    #[cfg_attr(test, test)]
    /// Verifies hyperbolic functions (cosh, sinh) for floating point numbers.
    fn test_num_trait_hyperbolic_functions() {
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

    #[cfg_attr(test, test)]
    /// Verifies absolute value behavior of standard integers (including
    /// wrapping boundary checks) (FR-1 of `num-traits-design.md`, `Signed`).
    fn test_num_trait_integer_absolute_value() {
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
    /// Verifies custom Cartesian quadrant mapping logic.
    fn test_num_trait_cartesian_quadrants_custom() {
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

    #[cfg_attr(test, test)]
    /// Statically verifies the compile-time presence of `Unsigned`,
    /// `Integer` and `SaturatingInteger` on every unsigned primitive
    /// (FR-1 + FR-2 of `num-traits-design.md`).
    /// `AdditiveGroup`/`Signed` being withheld from these same types is a
    /// negative compile-time property, verified separately by the
    /// `compile_fail` doctest on `num_traits`'s module documentation
    /// (this suite lives behind `#[cfg(any(test, feature = "ets"))]`,
    /// which `rustdoc` doctest extraction does not set).
    fn test_num_trait_unsigned_integer_markers() {
        fn assert_is_unsigned_integer<
            T: Unsigned + Integer + SaturatingInteger,
        >() {
        }

        assert_is_unsigned_integer::<u8>();
        assert_is_unsigned_integer::<u16>();
        assert_is_unsigned_integer::<u32>();
        assert_is_unsigned_integer::<u64>();
        assert_is_unsigned_integer::<u128>();
        assert_is_unsigned_integer::<usize>();
    }

    #[cfg_attr(test, test)]
    /// Statically verifies the compile-time presence of `Scalar` on every
    /// integer and float primitive, signed and unsigned (FR-3 of
    /// `num-traits-design.md`).
    fn test_num_trait_scalar_markers() {
        fn assert_is_scalar<T: Scalar>() {}

        assert_is_scalar::<i8>();
        assert_is_scalar::<i16>();
        assert_is_scalar::<i32>();
        assert_is_scalar::<i64>();
        assert_is_scalar::<i128>();
        assert_is_scalar::<isize>();
        assert_is_scalar::<u8>();
        assert_is_scalar::<u16>();
        assert_is_scalar::<u32>();
        assert_is_scalar::<u64>();
        assert_is_scalar::<u128>();
        assert_is_scalar::<usize>();
        assert_is_scalar::<f32>();
        assert_is_scalar::<f64>();
    }

    #[cfg_attr(test, test)]
    /// Verifies the default atan2 implementation fallback branches on custom `TestFloat` type.
    fn test_num_trait_default_atan2_fallbacks() {
        let two = TestFloat(2.0);
        // 1. Origin
        assert_eq!(TestFloat(0.0).atan2(TestFloat(0.0)), TestFloat(0.0));
        // 2. Positive X
        assert_eq!(TestFloat(0.0).atan2(TestFloat(1.0)), TestFloat(0.0));
        // 3. Negative Y
        assert_eq!(TestFloat(-1.0).atan2(TestFloat(0.0)), -TestFloat::PI / two);
        // 4. Negative X
        assert_eq!(TestFloat(0.0).atan2(TestFloat(-1.0)), TestFloat::PI);
        // 5. Positive Y
        assert_eq!(TestFloat(1.0).atan2(TestFloat(0.0)), TestFloat::PI / two);
        // 6. Q1
        assert_eq!(TestFloat(1.0).atan2(TestFloat(1.0)), TestFloat(1.0).atan());
        // 7. Q2: atan(y/x) + π
        let q2_atan = (TestFloat(1.0) / TestFloat(-1.0)).atan();
        assert_eq!(
            TestFloat(1.0).atan2(TestFloat(-1.0)),
            q2_atan + TestFloat::PI
        );
        // 8. Q3: atan(y/x) - π
        let q3_atan = (TestFloat(-1.0) / TestFloat(-1.0)).atan();
        assert_eq!(
            TestFloat(-1.0).atan2(TestFloat(-1.0)),
            q3_atan - TestFloat::PI
        );
        // 9. Q4
        assert_eq!(
            TestFloat(-1.0).atan2(TestFloat(1.0)),
            (TestFloat(-1.0) / TestFloat(1.0)).atan()
        );
    }

    #[cfg_attr(test, test)]
    /// Verifies cartesian quadrant parsing compilation coverage on f32 and f64.
    fn test_num_trait_cartesian_plane_axis_coords_coverage() {
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

    #[cfg_attr(test, test)]
    /// Verifies fallible math operations (TryAdd/Sub/Mul/Div) on complex
    /// numbers, transitively exercising `Complex<T>: Scalar` (FR-3 of
    /// `num-traits-design.md`).
    fn test_num_trait_complex_try_ops() {
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
    }
}
