//! # Numerical Tests
//!
//! These tests cover `[num_traits]` and `[num_types]`.
use crate::math::ArithmeticError;
use crate::math::num_traits::{Field, One, Real, Ring, Scalar, Signed, Zero};
use crate::math::ops::{TryAdd, TryDiv, TryMul, TryNeg, TryRem, TrySub};
use crate::{assert_almost_eq, assert_not_almost_eq};

#[cfg(feature = "std")]
#[test]
fn test_arithmetic_error_display() {
    // Covers src/math/mod.rs: 51-58 (Display impl)
    assert_eq!(
        format!("{}", ArithmeticError::DomainViolation),
        "Input is outside the mathematical domain"
    );
    assert_eq!(
        format!("{}", ArithmeticError::DivisionByZero),
        "Division by zero"
    );
    assert_eq!(
        format!("{}", ArithmeticError::Overflow),
        "Value overflowed representable range"
    );
    assert_eq!(
        format!("{}", ArithmeticError::Underflow),
        "Value underflowed (subnormal)"
    );
    assert_eq!(
        format!("{}", ArithmeticError::Saturation),
        "Value saturated (clamped) at bounds"
    );
    assert_eq!(
        format!("{}", ArithmeticError::PrecisionLoss),
        "Significant precision was lost during operation"
    );
}

#[test]
fn test_scalar_defaults() {
    // Covers src/math/num_traits.rs: default methods for Zero, One, Signed
    let val1 = 0.0f32;
    assert!(val1.is_zero());
    assert!(!1.0f32.is_zero());

    let val2 = 1.0f32;
    assert!(val2.is_one());
    assert!(!0.0f32.is_one());

    let val3 = -1.0f32;
    assert!(val3.is_sign_negative());
    assert!(!val3.is_sign_positive());

    let val4 = 1.0f32;
    assert!(!val4.is_sign_negative());
    assert!(val4.is_sign_positive());
}

#[test]
fn test_real_trait_errors() {
    // Covers src/math/num_traits.rs: impl_real! macro error paths

    // SQRT: Domain error on negative
    assert_eq!(Real::sqrt(-1.0f32), Err(ArithmeticError::DomainViolation));
    assert!(Real::sqrt(4.0f32).is_ok());

    // LOG10: Domain error on <= 0
    assert_eq!(Real::log10(0.0f32), Err(ArithmeticError::DomainViolation));
    assert_eq!(Real::log10(-1.0f32), Err(ArithmeticError::DomainViolation));
    assert!(Real::log10(10.0f32).is_ok());

    // LN: Domain error on <= 0
    assert_eq!(Real::ln(0.0f32), Err(ArithmeticError::DomainViolation));
    assert_eq!(Real::ln(-1.0f32), Err(ArithmeticError::DomainViolation));
    assert!(Real::ln(core::f32::consts::E).is_ok());

    // EXP/POW: Ensure passthrough works (happy path coverage)
    assert_almost_eq!(Real::exp(0.0f32), 1.0);
    assert!(Real::pow(2.0f32, 2.0f32).is_ok());
}

#[test]
fn test_float_ops_edge_cases() {
    // Covers src/math/ops.rs: try_float_impl! error paths

    let nan = f32::NAN;
    let max = f32::MAX;

    // --- TryAdd ---
    // 1. Domain Violation (NaN input)
    assert_eq!(1.0f32.try_add(&nan), Err(ArithmeticError::DomainViolation));
    // 2. Overflow (Result is Infinity)
    assert_eq!(max.try_add(&max), Err(ArithmeticError::Overflow));

    // --- TrySub ---
    // 1. Domain Violation (NaN input)
    assert_eq!(1.0f32.try_sub(&nan), Err(ArithmeticError::DomainViolation));
    // 2. Overflow (Result is Infinity)
    // Note: -MAX - MAX = -INF
    assert_eq!((-max).try_sub(&max), Err(ArithmeticError::Overflow));

    // --- TryMul ---
    // 1. Domain Violation
    assert_eq!(1.0f32.try_mul(&nan), Err(ArithmeticError::DomainViolation));
    // 2. Overflow
    assert_eq!(max.try_mul(&2.0), Err(ArithmeticError::Overflow));

    // --- TryDiv ---
    // 1. Division by Zero
    assert_eq!(1.0f32.try_div(&0.0), Err(ArithmeticError::DivisionByZero));
    // 2. Domain Violation (NaN)
    assert_eq!(nan.try_div(&1.0), Err(ArithmeticError::DomainViolation));
    // 3. Overflow
    // MAX / 0.5 = MAX * 2 = INF
    assert_eq!(max.try_div(&0.5), Err(ArithmeticError::Overflow));

    // --- TryRem ---
    // 1. Division by Zero
    assert_eq!(1.0f32.try_rem(&0.0), Err(ArithmeticError::DivisionByZero));
    // 2. Domain Violation
    assert_eq!(nan.try_rem(&1.0), Err(ArithmeticError::DomainViolation));

    // --- TryNeg ---
    // Happy path (float neg never overflows in this impl)
    assert!(1.0f32.try_neg().is_ok());
}

#[test]
fn test_scalar_properties() {
    // Assumption: Scalar types support Copy, PartialEq, and PartialOrd.
    // We verify this behavior, especially the IEEE 754 compliance for floats.

    // Case 1: Reflexivity holds for standard values
    let a = 10.0f32;
    assert_almost_eq!(a, a);
    assert!(!(a < a));
    assert!(!(a > a));

    // Case 2: Transitivity (a < b and b < c => a < c)
    let b = 20.0f32;
    let c = 30.0f32;
    assert!(a < b && b < c);
    assert!(a < c);

    // Case 3: NaN behavior (The "Partial" in PartialOrd)
    // Assumption: NaN is not equal to itself and is not ordered.
    let nan = f32::NAN;
    assert_not_almost_eq!(nan, nan);
    assert!(!(nan < 0.0));
    assert!(!(nan > 0.0));
    // This confirms our `Scalar` trait doesn't accidentally force total ordering
    // where it shouldn't exist.
}

#[test]
fn test_ring_trait_generic() {
    fn check_ring<T: Ring + PartialEq + core::fmt::Debug + Copy>(
        a: T,
        b: T,
        c: T,
    ) {
        // Identity: a + 0 = a
        assert_eq!(a + T::zero(), a, "Additive identity failed");

        // Identity: a * 1 = a
        assert_eq!(a * T::one(), a, "Multiplicative identity failed");

        // Associativity: (a + b) + c = a + (b + c)
        assert_eq!((a + b) + c, a + (b + c), "Associativity failed");

        // Distributivity: a * (b + c) = a*b + a*c
        let left = a * (b + c);
        let right = (a * b) + (a * c);
        // Note: For floats, exact equality might fail due to precision,
        // but for integers (Ring) it must hold exactly.
        assert_eq!(left, right, "Distributivity failed");
    }

    // Verify for Integer
    check_ring(3, 4, 5);
    check_ring(0, 10, -5);

    // Verify for Float (using small integers to avoid precision issues in equality check)
    check_ring(2.0f32, 3.0f32, 4.0f32);
}
#[test]
fn test_signed_trait_generic() {
    fn check_signed<T: Signed + PartialEq + core::fmt::Debug + Copy>(
        pos: T,
        neg: T,
    ) {
        // Test abs()
        assert_eq!(
            <T as Signed>::abs(neg),
            pos,
            "Absolute value of negative failed"
        );
        assert_eq!(
            <T as Signed>::abs(pos),
            pos,
            "Absolute value of positive failed"
        );

        // Test is_sign_negative()
        assert!(
            <T as Signed>::is_sign_negative(&neg),
            "Negative number check failed"
        );
        assert!(
            !<T as Signed>::is_sign_negative(&pos),
            "Positive number check failed"
        );

        // Test is_sign_positive()
        assert!(
            <T as Signed>::is_sign_positive(&pos),
            "Positive number check failed"
        );
        assert!(
            !<T as Signed>::is_sign_positive(&neg),
            "Negative number check failed"
        );

        // Test Zero behavior
        let zero = T::zero();
        assert_eq!(
            <T as Signed>::abs(zero),
            zero,
            "Absolute value of zero failed"
        );
        // By definition in your trait: zero is neither strictly positive nor negative
        assert!(
            !<T as Signed>::is_sign_negative(&zero),
            "Zero should not be negative"
        );
        assert!(
            !<T as Signed>::is_sign_positive(&zero),
            "Zero should not be positive"
        );
    }
    check_signed(10i8, -10i8);
    check_signed(10i16, -10i16);
    check_signed(10i32, -10i32);
    check_signed(10i64, -10i64);
    check_signed(10i128, -10i128);
    check_signed(10isize, -10isize);
    check_signed(5.5f32, -5.5f32);
    check_signed(5.5f64, -5.5f64);
}

// Verify Copy trait (compile-time check mainly, but runtime proof here)
fn check_copy<T: Scalar>(x: T) -> (T, T) {
    (x, x) // If T wasn't Copy, this would move x twice and fail to compile
}

#[test]
fn test_scalar_trait_generic() {
    fn check_scalar_order<T: Scalar + core::fmt::Debug>(small: T, large: T) {
        // Test PartialOrd through Scalar
        assert!(small < large, "Scalar ordering failed: small < large");
        assert!(large > small, "Scalar ordering failed: large > small");
        assert!(small <= small, "Scalar reflexive <= failed");

        // Test Equality
        assert_eq!(small, small, "Scalar equality failed");
        assert_ne!(small, large, "Scalar inequality failed");
    }

    check_scalar_order(1u8, 255u8);
    check_scalar_order(-100i32, 100i32);
    check_scalar_order(0.0f32, 1.0f32);

    let (a, b) = check_copy(42);
    assert_eq!(a, b);
}

#[test]
fn test_field_trait_generic() {
    fn check_field<T: Field + PartialEq + core::fmt::Debug + Copy>(a: T) {
        // Identity: a + 0 = a
        assert_eq!(a + T::zero(), a, "Additive identity failed");
        // Identity: a * 1 = a
        assert_eq!(a * T::one(), a, "Multiplicative identity failed");

        // Inverse: a * (1/a) = 1 (for a != 0)
        if a != T::zero() {
            assert_eq!(a.div(a), T::one(), "Multiplicative inverse failed");
        }
        // Epsilon is the difference between 1.0 and the next representable floating-point number.
        // It's not necessarily the smallest value, but it's a common property to check.
        assert!(
            T::one() + T::epsilon() > T::one(),
            "Epsilon check failed: 1 + epsilon == 1"
        );
        assert_eq!(
            T::one() + (T::epsilon().div(T::one() + T::one())),
            T::one(),
            "Epsilon check failed: 1 + (epsilon / 2) == 1"
        );
    }

    check_field(3.0f64);
    check_field(f64::MAX);
    check_field(f64::EPSILON); // Smallest positive value
    check_field(f64::one());
    check_field(3.0f32);
    check_field(f32::MAX);
    check_field(f32::EPSILON); // Smallest positive value
    check_field(f32::one());
    check_field(0.0f32);
}
