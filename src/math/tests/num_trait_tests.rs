//! # Numerical Tests
//!
//! These tests cover `[num_traits]` and `[num_types]`.
#![allow(
    unused_imports,
    clippy::arbitrary_source_item_ordering,
    clippy::arithmetic_side_effects
)]

use crate::math::ArithmeticError;
use crate::math::num_traits::{Field, One, Real, Ring, Scalar, Signed, Zero};
use crate::{assert_almost_eq, assert_not_almost_eq};

mod scalar_tests {
    use crate::{
        assert_almost_eq,
        math::num_traits::{One, Scalar, Signed, Zero},
    };
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
        // Verify for Integer
        check_int_ring(3_i8, 4_i8, 5_i8);
        check_int_ring(0_i16, 10_i16, -5_i16);
        //
        check_float_ring(2.0f32, 3.0f32, 4.0f32);
        check_float_ring(2.0f64, 3.0f64, 4.0f64);
    }
}

mod real_tests {
    use crate::{
        assert_almost_eq,
        math::{
            num_traits::{One, Real, Signed, Zero},
            ops::{TryMul, TrySub},
        },
    };
    #[allow(clippy::eq_op)]
    fn real_property_check<
        T: Real + Signed + Zero + One + TrySub + TryMul + core::fmt::Debug,
    >() {
        assert_ne!(T::NAN, T::NAN);
        assert_eq!(T::INF, T::INF);
        assert_almost_eq!(T::cos(T::PI), T::ONE.neg());
        assert_almost_eq!(T::epsilon() / T::TWO, T::ZERO);
        assert_almost_eq!(T::exp(T::ONE), T::E);
        assert_almost_eq!(T::ln(T::E), T::ONE);
        assert_almost_eq!(T::log10(T::from_const::<10>()), T::ONE);
        assert_almost_eq!(T::pow(T::TWO, T::TWO), T::TWO + T::TWO);
        assert_almost_eq!(T::sin(T::PI), T::ZERO);
        assert_almost_eq!(T::sqrt(T::TWO + T::TWO), T::TWO);
        assert_almost_eq!(T::tan(T::ZERO), T::ZERO);
    }
    #[test]
    fn test_real_properties() {
        real_property_check::<f32>();
        real_property_check::<f64>();
    }
}
