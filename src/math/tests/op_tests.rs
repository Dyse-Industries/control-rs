#![allow(clippy::arbitrary_source_item_ordering)]

mod test_float_add {
    use crate::math::{
        ArithmeticError,
        num_traits::Real,
        ops::{Neg, TryAdd},
    };

    #[allow(clippy::arithmetic_side_effects, clippy::needless_pass_by_value)]
    fn try_add_float_checks<
        T: Real + TryAdd + Neg<Output = T> + core::fmt::Debug,
    >(
        rhs: T,
        lhs: T,
        expected: T,
    ) {
        assert_eq!(lhs.try_add(&rhs), Ok(expected));
        assert_eq!(T::MIN.try_add(&T::MIN), Ok(T::INF.neg()));
        assert_eq!(
            T::MIN.try_add(&T::ONE),
            Err(ArithmeticError::PrecisionLoss)
        );
        assert_eq!(
            T::ONE.try_add(&T::MAX),
            Err(ArithmeticError::PrecisionLoss)
        );
        assert_eq!(
            T::NAN.try_add(&T::ONE),
            Err(ArithmeticError::DomainViolation)
        );
        assert_eq!(
            T::ZERO.try_add(&(T::MIN_POSITIVE / T::TWO)),
            Err(ArithmeticError::Underflow)
        );
    }
    #[test]
    fn f32_addition() {
        try_add_float_checks(1.0_f32, 2.0_f32, 3.0_f32);
    }
    #[test]
    fn f64_addition() {
        try_add_float_checks(1.0_f64, 2.0_f64, 3.0_f64);
    }
}

mod test_saturating_add {
    use crate::math::num_traits::Ring;
    use crate::math::ops::{Neg, SaturatingAdd};
    #[allow(clippy::arithmetic_side_effects)]
    fn signed_saturating_add_int_checks<
        T: Ring + SaturatingAdd + Neg<Output = T> + core::fmt::Debug,
    >(
        rhs: T,
        lhs: T,
        expected: T,
    ) {
        assert_eq!(T::MIN.saturating_add(&(-T::ONE)), T::MIN);
        unsigned_saturating_add_int_checks(rhs, lhs, expected);
    }
    #[allow(clippy::arithmetic_side_effects, clippy::needless_pass_by_value)]
    fn unsigned_saturating_add_int_checks<
        T: Ring + SaturatingAdd + core::fmt::Debug,
    >(
        rhs: T,
        lhs: T,
        expected: T,
    ) {
        assert_eq!(lhs.saturating_add(&rhs), expected);
        assert_eq!(T::MAX.saturating_add(&T::ONE), T::MAX);
        assert_eq!(T::MAX.saturating_add(&T::MAX), T::MAX);
        assert_eq!(T::MIN.saturating_add(&T::MIN), T::MIN);
    }
    #[test]
    fn signed_integers() {
        signed_saturating_add_int_checks(1_i8, 2_i8, 3_i8);
        signed_saturating_add_int_checks(3_i16, 4_i16, 7_i16);
        signed_saturating_add_int_checks(5_i32, 6_i32, 11_i32);
        signed_saturating_add_int_checks(7_isize, 8_isize, 15_isize);
    }
    #[test]
    fn unsigned_integers() {
        unsigned_saturating_add_int_checks(1_u8, 2_u8, 3_u8);
        unsigned_saturating_add_int_checks(3_u16, 4_u16, 7_u16);
        unsigned_saturating_add_int_checks(5_u32, 6_u32, 11_u32);
        unsigned_saturating_add_int_checks(7_usize, 8_usize, 15_usize);
    }
}

mod test_wrapping_add {
    use crate::math::{
        num_traits::{Ring, Signed},
        ops::WrappingAdd,
    };
    #[allow(clippy::arithmetic_side_effects, clippy::needless_pass_by_value)]
    fn signed_wrapping_add_int_checks<
        T: Ring + Signed + WrappingAdd + core::fmt::Debug,
    >(
        rhs: T,
        lhs: T,
        expected: T,
    ) {
        assert_eq!(T::MIN.wrapping_add(&(T::ONE.neg())), T::MAX);
        assert_eq!(T::MAX.wrapping_add(&T::ONE), T::MIN);
        assert_eq!(T::MAX.wrapping_add(&T::ONE), T::MIN);
        assert_eq!(T::MAX.wrapping_add(&T::MAX), -T::TWO);
        assert_eq!(T::MIN.wrapping_add(&T::MIN), T::ZERO);
        assert_eq!(lhs.wrapping_add(&rhs), expected);
    }
    #[allow(clippy::arithmetic_side_effects, clippy::needless_pass_by_value)]
    fn unsigned_wrapping_add_int_checks<
        T: Ring + WrappingAdd + core::fmt::Debug,
    >(
        rhs: T,
        lhs: T,
        expected: T,
    ) {
        assert_eq!(T::MAX.wrapping_add(&T::ONE), T::MIN);
        assert_eq!(T::MAX.wrapping_add(&T::ONE), T::MIN);
        assert_eq!(T::MAX.wrapping_add(&T::MAX), T::MAX - T::ONE);
        assert_eq!(T::MIN.wrapping_add(&T::MIN), T::ZERO);
        assert_eq!(lhs.wrapping_add(&rhs), expected);
    }
    #[test]
    fn signed_integers() {
        signed_wrapping_add_int_checks(1_i8, 2_i8, 3_i8);
        signed_wrapping_add_int_checks(1_i16, 2_i16, 3_i16);
        signed_wrapping_add_int_checks(1_i32, 2_i32, 3_i32);
        signed_wrapping_add_int_checks(1_isize, 2_isize, 3_isize);
    }
    #[test]
    fn unsigned_integers() {
        unsigned_wrapping_add_int_checks(1_u8, 2_u8, 3_u8);
        unsigned_wrapping_add_int_checks(1_u16, 2_u16, 3_u16);
        unsigned_wrapping_add_int_checks(1_u32, 2_u32, 3_u32);
        unsigned_wrapping_add_int_checks(1_usize, 2_usize, 3_usize);
    }
}

// --- SUBTRACTION TESTS ---

mod test_sub {
    use crate::math::{ArithmeticError, num_traits::Ring, ops::TrySub};

    #[allow(clippy::arithmetic_side_effects, clippy::needless_pass_by_value)]
    fn signed_try_sub_int_checks<
        T: Ring + TrySub + core::fmt::Debug + PartialEq,
    >(
        rhs: T,
        lhs: T,
        expected: T,
    ) {
        assert_eq!(lhs.try_sub(&rhs), Ok(expected));
        assert_eq!(T::MIN.try_sub(&T::ONE), Err(ArithmeticError::Overflow));
    }

    #[allow(clippy::arithmetic_side_effects, clippy::needless_pass_by_value)]
    fn unsigned_try_sub_int_checks<
        T: Ring + TrySub + core::fmt::Debug + PartialEq,
    >(
        rhs: T,
        lhs: T,
        expected: T,
    ) {
        assert_eq!(lhs.try_sub(&rhs), Ok(expected));
        assert_eq!(T::MIN.try_sub(&T::ONE), Err(ArithmeticError::Overflow));
    }

    #[test]
    fn signed_integers() {
        signed_try_sub_int_checks(1_i8, 2_i8, 1_i8);
        signed_try_sub_int_checks(1_i16, 2_i16, 1_i16);
        signed_try_sub_int_checks(1_i32, 2_i32, 1_i32);
        signed_try_sub_int_checks(1_isize, 2_isize, 1_isize);
    }

    #[test]
    fn unsigned_integers() {
        unsigned_try_sub_int_checks(1_u8, 2_u8, 1_u8);
        unsigned_try_sub_int_checks(1_u16, 2_u16, 1_u16);
        unsigned_try_sub_int_checks(1_u32, 2_u32, 1_u32);
        unsigned_try_sub_int_checks(1_usize, 2_usize, 1_usize);
    }
}

mod test_float_sub {
    use crate::math::{ArithmeticError, ops::TrySub};

    #[test]
    fn f32_checks() {
        assert_eq!(2.0_f32.try_sub(&1.0), Ok(1.0));
        assert_eq!(f32::MIN.try_sub(&f32::MAX), Ok(-f32::INFINITY));
        assert_eq!(f32::MAX.try_sub(&f32::MIN), Ok(f32::INFINITY));
        assert_eq!(f32::MAX.try_sub(&f32::MAX), Ok(0.0));
        assert_eq!(
            f32::NAN.try_sub(&1.0),
            Err(ArithmeticError::DomainViolation)
        );
        assert_eq!(
            1e25_f32.try_sub(&(f32::EPSILON / 2.0)),
            Err(ArithmeticError::PrecisionLoss)
        );
        assert_eq!(
            (f32::MIN_POSITIVE / 2.0).try_sub(&1.0),
            Err(ArithmeticError::PrecisionLoss)
        );
        assert_eq!(
            f32::MIN_POSITIVE.try_sub(&(f32::MIN_POSITIVE / 2.0)),
            Err(ArithmeticError::Underflow)
        );
    }

    #[test]
    fn f64_checks() {
        assert_eq!(2.0_f64.try_sub(&1.0), Ok(1.0));
        assert_eq!(f64::MIN.try_sub(&f64::MAX), Ok(-f64::INFINITY));
        assert_eq!(f64::MAX.try_sub(&f64::MIN), Ok(f64::INFINITY));
        assert_eq!(f64::MAX.try_sub(&f64::MAX), Ok(0.0));
        assert_eq!(
            1e250_f64.try_sub(&(f64::EPSILON / 2.0)),
            Err(ArithmeticError::PrecisionLoss)
        );
        assert_eq!(
            (f64::MIN_POSITIVE / 2.0).try_sub(&1.0),
            Err(ArithmeticError::PrecisionLoss)
        );
        assert_eq!(
            f64::MIN_POSITIVE.try_sub(&(f64::MIN_POSITIVE / 2.0)),
            Err(ArithmeticError::Underflow)
        );
    }
}

mod test_saturating_sub {
    use crate::math::{
        num_traits::{Ring, Signed},
        ops::SaturatingSub,
    };

    #[allow(clippy::arithmetic_side_effects, clippy::needless_pass_by_value)]
    fn signed_saturating_sub_int_checks<
        T: Ring
            + Signed
            + SaturatingSub
            + core::ops::Neg<Output = T>
            + core::fmt::Debug,
    >(
        rhs: T,
        lhs: T,
        expected: T,
    ) {
        assert_eq!(T::MAX.saturating_sub(&(-T::ONE)), T::MAX);
        unsigned_saturating_sub_int_checks(rhs, lhs, expected);
    }

    #[allow(clippy::arithmetic_side_effects, clippy::needless_pass_by_value)]
    fn unsigned_saturating_sub_int_checks<
        T: Ring + SaturatingSub + core::fmt::Debug,
    >(
        rhs: T,
        lhs: T,
        expected: T,
    ) {
        assert_eq!(lhs.saturating_sub(&rhs), expected);
        assert_eq!(T::MIN.saturating_sub(&T::ONE), T::MIN);
    }

    #[test]
    fn signed_integers() {
        signed_saturating_sub_int_checks(1_i8, 2_i8, 1_i8);
        signed_saturating_sub_int_checks(1_i16, 2_i16, 1_i16);
        signed_saturating_sub_int_checks(1_i32, 2_i32, 1_i32);
        signed_saturating_sub_int_checks(1_isize, 2_isize, 1_isize);
    }

    #[test]
    fn unsigned_integers() {
        unsigned_saturating_sub_int_checks(1_u8, 2_u8, 1_u8);
        unsigned_saturating_sub_int_checks(1_u16, 2_u16, 1_u16);
        unsigned_saturating_sub_int_checks(1_u32, 2_u32, 1_u32);
        unsigned_saturating_sub_int_checks(1_usize, 2_usize, 1_usize);
    }
}

mod test_wrapping_sub {
    use crate::math::{num_traits::Ring, ops::WrappingSub};

    #[allow(clippy::arithmetic_side_effects)]
    fn signed_wrapping_sub_int_checks<
        T: Ring + WrappingSub + core::ops::Neg<Output = T> + core::fmt::Debug,
    >(
        rhs: T,
        lhs: T,
        expected: T,
    ) {
        assert_eq!(T::MAX.wrapping_sub(&(-T::ONE)), T::MIN);
        unsigned_wrapping_sub_int_checks(rhs, lhs, expected);
    }

    #[allow(clippy::arithmetic_side_effects, clippy::needless_pass_by_value)]
    fn unsigned_wrapping_sub_int_checks<
        T: Ring + WrappingSub + core::fmt::Debug + PartialEq,
    >(
        rhs: T,
        lhs: T,
        expected: T,
    ) {
        assert_eq!(lhs.wrapping_sub(&rhs), expected);
        assert_eq!(T::MIN.wrapping_sub(&T::ONE), T::MAX);
    }

    #[test]
    fn signed_integers() {
        signed_wrapping_sub_int_checks(1_i8, 2_i8, 1_i8);
        signed_wrapping_sub_int_checks(1_i16, 2_i16, 1_i16);
        signed_wrapping_sub_int_checks(1_i32, 2_i32, 1_i32);
        signed_wrapping_sub_int_checks(1_isize, 2_isize, 1_isize);
    }

    #[test]
    fn unsigned_integers() {
        unsigned_wrapping_sub_int_checks(1_u8, 2_u8, 1_u8);
        unsigned_wrapping_sub_int_checks(1_u16, 2_u16, 1_u16);
        unsigned_wrapping_sub_int_checks(1_u32, 2_u32, 1_u32);
        unsigned_wrapping_sub_int_checks(1_usize, 2_usize, 1_usize);
    }
}

// --- MULTIPLICATION TESTS ---

#[allow(clippy::arbitrary_source_item_ordering)]
mod test_int_mul {
    use crate::math::{ArithmeticError, num_traits::Ring, ops::TryMul};

    #[allow(clippy::arithmetic_side_effects)]
    fn signed_try_mul_int_checks<
        T: Ring + TryMul + core::fmt::Debug + PartialEq,
    >(
        rhs: T,
        lhs: T,
        expected: T,
    ) {
        unsigned_try_mul_int_checks(rhs, lhs, expected);
    }

    #[allow(clippy::arithmetic_side_effects, clippy::needless_pass_by_value)]
    fn unsigned_try_mul_int_checks<
        T: Ring + TryMul + core::fmt::Debug + PartialEq,
    >(
        rhs: T,
        lhs: T,
        expected: T,
    ) {
        assert_eq!(lhs.try_mul(&rhs), Ok(expected));
        assert_eq!(T::MAX.try_mul(&T::TWO), Err(ArithmeticError::Overflow));
    }

    #[test]
    fn signed_integers() {
        signed_try_mul_int_checks(3_i8, 2_i8, 6_i8);
        signed_try_mul_int_checks(3_i16, 2_i16, 6_i16);
        signed_try_mul_int_checks(3_i32, 2_i32, 6_i32);
        signed_try_mul_int_checks(3_isize, 2_isize, 6_isize);
    }

    #[test]
    fn unsigned_integers() {
        unsigned_try_mul_int_checks(3_u8, 2_u8, 6_u8);
        unsigned_try_mul_int_checks(3_u16, 2_u16, 6_u16);
        unsigned_try_mul_int_checks(3_u32, 2_u32, 6_u32);
        unsigned_try_mul_int_checks(3_usize, 2_usize, 6_usize);
    }
}

#[allow(clippy::arbitrary_source_item_ordering)]
mod test_float_mul {
    use crate::math::{ArithmeticError, ops::TryMul};

    #[test]
    fn f32_checks() {
        assert_eq!(2.0_f32.try_mul(&3.0), Ok(6.0));
        assert_eq!(f32::MAX.try_mul(&2.0), Err(ArithmeticError::Overflow));
        assert_eq!(
            f32::NAN.try_mul(&1.0),
            Err(ArithmeticError::DomainViolation)
        );
        assert_eq!(f32::MAX.try_mul(&-2.0), Err(ArithmeticError::Overflow));
    }

    #[test]
    fn f64_checks() {
        assert_eq!(2.0_f64.try_mul(&3.0), Ok(6.0));
        assert_eq!(f64::MAX.try_mul(&2.0), Err(ArithmeticError::Overflow));
        assert_eq!(
            f64::NAN.try_mul(&1.0),
            Err(ArithmeticError::DomainViolation)
        );
        assert_eq!(f64::MAX.try_mul(&-2.0), Err(ArithmeticError::Overflow));
    }
}

mod test_saturating_mul {
    use crate::math::{
        num_traits::{Ring, Signed},
        ops::SaturatingMul,
    };

    #[allow(clippy::arithmetic_side_effects)]
    fn signed_saturating_mul_int_checks<
        T: Ring + Signed + SaturatingMul + core::fmt::Debug + PartialEq,
    >(
        rhs: T,
        lhs: T,
        expected: T,
    ) {
        assert_eq!(T::MIN.saturating_mul(&T::TWO), T::MIN);
        unsigned_saturating_mul_int_checks(rhs, lhs, expected);
    }

    #[allow(clippy::arithmetic_side_effects, clippy::needless_pass_by_value)]
    fn unsigned_saturating_mul_int_checks<
        T: Ring + SaturatingMul + core::fmt::Debug + PartialEq,
    >(
        rhs: T,
        lhs: T,
        expected: T,
    ) {
        assert_eq!(lhs.saturating_mul(&rhs), expected);
        assert_eq!(T::MAX.saturating_mul(&T::TWO), T::MAX);
    }

    #[test]
    fn signed_integers() {
        signed_saturating_mul_int_checks(3_i8, 2_i8, 6_i8);
        signed_saturating_mul_int_checks(3_i16, 2_i16, 6_i16);
        signed_saturating_mul_int_checks(3_i32, 2_i32, 6_i32);
        signed_saturating_mul_int_checks(3_isize, 2_isize, 6_isize);
    }

    #[test]
    fn unsigned_integers() {
        unsigned_saturating_mul_int_checks(3_u8, 2_u8, 6_u8);
        unsigned_saturating_mul_int_checks(3_u16, 2_u16, 6_u16);
        unsigned_saturating_mul_int_checks(3_u32, 2_u32, 6_u32);
        unsigned_saturating_mul_int_checks(3_usize, 2_usize, 6_usize);
    }
}

mod test_wrapping_mul {
    use crate::math::{num_traits::Ring, ops::WrappingMul};

    #[allow(clippy::arithmetic_side_effects)]
    fn signed_wrapping_mul_int_checks<
        T: Ring
            + WrappingMul
            + core::ops::Neg<Output = T>
            + core::fmt::Debug
            + PartialEq,
    >(
        rhs: T,
        lhs: T,
        expected: T,
    ) {
        assert_eq!(T::MAX.wrapping_mul(&T::TWO), -T::TWO);
        unsigned_wrapping_mul_int_checks(rhs, lhs, expected);
    }

    #[allow(clippy::arithmetic_side_effects, clippy::needless_pass_by_value)]
    fn unsigned_wrapping_mul_int_checks<
        T: Ring + WrappingMul + core::fmt::Debug + PartialEq,
    >(
        rhs: T,
        lhs: T,
        expected: T,
    ) {
        assert_eq!(lhs.wrapping_mul(&rhs), expected);
    }

    #[test]
    fn signed_integers() {
        signed_wrapping_mul_int_checks(3_i8, 2_i8, 6_i8);
        signed_wrapping_mul_int_checks(3_i16, 2_i16, 6_i16);
        signed_wrapping_mul_int_checks(3_i32, 2_i32, 6_i32);
        signed_wrapping_mul_int_checks(3_isize, 2_isize, 6_isize);
    }

    #[test]
    fn unsigned_integers() {
        unsigned_wrapping_mul_int_checks(3_u8, 2_u8, 6_u8);
        unsigned_wrapping_mul_int_checks(3_u16, 2_u16, 6_u16);
        unsigned_wrapping_mul_int_checks(3_u32, 2_u32, 6_u32);
        unsigned_wrapping_mul_int_checks(3_usize, 2_usize, 6_usize);
    }
}

// --- DIVISION TESTS ---

#[allow(clippy::arbitrary_source_item_ordering)]
mod test_int_div {
    use crate::math::{ArithmeticError, num_traits::Ring, ops::TryDiv};

    #[allow(clippy::arithmetic_side_effects)]
    fn signed_try_div_int_checks<
        T: Ring
            + TryDiv
            + core::ops::Neg<Output = T>
            + core::fmt::Debug
            + PartialEq,
    >(
        rhs: T,
        lhs: T,
        expected: T,
    ) {
        assert_eq!(T::MIN.try_div(&(-T::ONE)), Err(ArithmeticError::Overflow));
        unsigned_try_div_int_checks(rhs, lhs, expected);
    }

    #[allow(clippy::arithmetic_side_effects, clippy::needless_pass_by_value)]
    fn unsigned_try_div_int_checks<
        T: Ring + TryDiv + core::fmt::Debug + PartialEq,
    >(
        rhs: T,
        lhs: T,
        expected: T,
    ) {
        assert_eq!(lhs.try_div(&rhs), Ok(expected));
        let result =
            T::ONE.try_div(&T::ZERO) == Err(ArithmeticError::DivisionByZero);
        assert!(result);
        assert_eq!(T::MAX.try_div(&T::ONE), Ok(T::MAX));
    }

    #[test]
    fn signed_integers() {
        signed_try_div_int_checks(3_i8, 6_i8, 2_i8);
        signed_try_div_int_checks(3_i16, 6_i16, 2_i16);
        signed_try_div_int_checks(3_i32, 6_i32, 2_i32);
        signed_try_div_int_checks(3_isize, 6_isize, 2_isize);
    }

    #[test]
    fn unsigned_integers() {
        unsigned_try_div_int_checks(3_u8, 6_u8, 2_u8);
        unsigned_try_div_int_checks(3_u16, 6_u16, 2_u16);
        unsigned_try_div_int_checks(3_u32, 6_u32, 2_u32);
        unsigned_try_div_int_checks(3_usize, 6_usize, 2_usize);
    }
}

#[allow(clippy::arbitrary_source_item_ordering)]
mod test_float_div {
    use crate::math::{ArithmeticError, ops::TryDiv};

    #[test]
    fn f32_checks() {
        assert_eq!(6.0_f32.try_div(&3.0), Ok(2.0));
        assert_eq!(1.0_f32.try_div(&0.0), Err(ArithmeticError::DivisionByZero));
        assert_eq!(0.0_f32.try_div(&0.0), Err(ArithmeticError::DivisionByZero));
        assert_eq!(
            f32::NAN.try_div(&1.0),
            Err(ArithmeticError::DomainViolation)
        );
        assert_eq!(f32::MAX.try_div(&0.1), Err(ArithmeticError::Overflow));
    }

    #[test]
    fn f64_checks() {
        assert_eq!(6.0_f64.try_div(&3.0), Ok(2.0));
        assert_eq!(1.0_f64.try_div(&0.0), Err(ArithmeticError::DivisionByZero));
        assert_eq!(0.0_f64.try_div(&0.0), Err(ArithmeticError::DivisionByZero));
        assert_eq!(
            f64::NAN.try_div(&1.0),
            Err(ArithmeticError::DomainViolation)
        );
        assert_eq!(f64::MAX.try_div(&0.1), Err(ArithmeticError::Overflow));
    }
}

// --- REMAINDER TESTS ---

#[allow(clippy::arbitrary_source_item_ordering)]
mod test_rem {
    use crate::math::{
        ArithmeticError,
        num_traits::{Ring, Signed},
        ops::TryRem,
    };

    #[allow(clippy::arithmetic_side_effects)]
    fn signed_try_rem_int_checks<
        T: Ring + Signed + TryRem + core::fmt::Debug + PartialEq,
    >(
        rhs: T,
        lhs: T,
        expected: T,
    ) {
        assert_eq!(T::MIN.try_rem(&T::ONE), Ok(T::ZERO));
        assert_eq!(T::ONE.try_rem(&T::MIN), Ok(T::ONE));
        unsigned_try_rem_int_checks(rhs, lhs, expected);
    }

    #[allow(clippy::arithmetic_side_effects, clippy::needless_pass_by_value)]
    fn unsigned_try_rem_int_checks<
        T: Ring + TryRem + core::fmt::Debug + PartialEq,
    >(
        rhs: T,
        lhs: T,
        expected: T,
    ) {
        assert_eq!(lhs.try_rem(&rhs), Ok(expected));
        let result =
            T::ONE.try_rem(&T::ZERO) == Err(ArithmeticError::DivisionByZero);
        assert!(result); // macros are hard on tarpaulin
    }

    #[test]
    fn signed_integers() {
        signed_try_rem_int_checks(3_i8, 7_i8, 1_i8);
        signed_try_rem_int_checks(3_i16, 7_i16, 1_i16);
        signed_try_rem_int_checks(3_i32, 7_i32, 1_i32);
        signed_try_rem_int_checks(3_isize, 7_isize, 1_isize);
    }

    #[test]
    fn unsigned_integers() {
        unsigned_try_rem_int_checks(3_u8, 7_u8, 1_u8);
        unsigned_try_rem_int_checks(3_u16, 7_u16, 1_u16);
        unsigned_try_rem_int_checks(3_u32, 7_u32, 1_u32);
        unsigned_try_rem_int_checks(3_usize, 7_usize, 1_usize);
    }
}

mod test_float_rem {
    use crate::math::{ArithmeticError, ops::TryRem};

    #[test]
    fn f32_checks() {
        assert_eq!(7.0_f32.try_rem(&3.0), Ok(1.0));
        assert_eq!(1.0_f32.try_rem(&0.0), Err(ArithmeticError::DivisionByZero));
        assert_eq!(
            f32::NAN.try_rem(&1.0),
            Err(ArithmeticError::DomainViolation)
        );
        match f32::INFINITY.try_rem(&1.0) {
            Ok(v) => assert!(v.is_nan()),
            Err(e) => assert_eq!(e, ArithmeticError::DomainViolation),
        }
    }

    #[test]
    fn f64_checks() {
        assert_eq!(7.0_f64.try_rem(&3.0), Ok(1.0));
        assert_eq!(1.0_f64.try_rem(&0.0), Err(ArithmeticError::DivisionByZero));
        assert_eq!(
            f64::NAN.try_rem(&1.0),
            Err(ArithmeticError::DomainViolation)
        );
        match f64::INFINITY.try_rem(&1.0) {
            Ok(v) => assert!(v.is_nan()),
            Err(e) => assert_eq!(e, ArithmeticError::DomainViolation),
        }
    }
}

// --- NEGATION TESTS ---

mod test_neg {
    use crate::math::{ArithmeticError, num_traits::Ring, ops::TryNeg};

    #[allow(clippy::arithmetic_side_effects)]
    fn signed_try_neg_int_checks<
        T: Ring + TryNeg + core::fmt::Debug + PartialEq + Copy,
    >(
        val: T,
        expected: T,
    ) {
        assert_eq!(val.try_neg(), Ok(expected));
        assert_eq!(expected.try_neg(), Ok(val));
        assert_eq!(T::MIN.try_neg(), Err(ArithmeticError::Overflow));
    }

    #[test]
    fn signed_integers() {
        signed_try_neg_int_checks(1_i8, -1_i8);
        signed_try_neg_int_checks(1_i16, -1_i16);
        signed_try_neg_int_checks(1_i32, -1_i32);
        signed_try_neg_int_checks(1_isize, -1_isize);
    }

    #[test]
    fn f32_checks() {
        assert_eq!(1.0_f32.try_neg(), Ok(-1.0));
        assert_eq!((-1.0_f32).try_neg(), Ok(1.0));
        assert_eq!(f32::NAN.try_neg(), Err(ArithmeticError::DomainViolation));
        assert_eq!(f32::INFINITY.try_neg(), Ok(-f32::INFINITY));
        assert_eq!(0.0_f32.try_neg(), Ok(0.0));
    }

    #[test]
    fn f64_checks() {
        assert_eq!(1.0_f64.try_neg(), Ok(-1.0));
        assert_eq!((-1.0_f64).try_neg(), Ok(1.0));
        assert_eq!(f64::NAN.try_neg(), Err(ArithmeticError::DomainViolation));
        assert_eq!(f64::INFINITY.try_neg(), Ok(-f64::INFINITY));
        assert_eq!(0.0_f64.try_neg(), Ok(0.0));
    }
}

// --- SHIFT TESTS ---

mod test_shl {
    use crate::math::{ArithmeticError, num_traits::Ring, ops::TryShl};

    #[allow(clippy::arithmetic_side_effects)]
    fn signed_try_shl_int_checks<
        T: Ring + TryShl + core::fmt::Debug + PartialEq,
    >(
        val: T,
        expected: T,
        bits: u32,
    ) {
        unsigned_try_shl_int_checks(val, expected, bits);
    }

    #[allow(clippy::arithmetic_side_effects, clippy::needless_pass_by_value)]
    fn unsigned_try_shl_int_checks<
        T: Ring + TryShl + core::fmt::Debug + PartialEq,
    >(
        val: T,
        expected: T,
        bits: u32,
    ) {
        assert_eq!(val.try_shl(1), Ok(expected));
        assert_eq!(val.try_shl(bits), Err(ArithmeticError::Overflow));
    }

    #[test]
    fn signed_integers() {
        signed_try_shl_int_checks(1_i8, 2_i8, 8);
        signed_try_shl_int_checks(1_i16, 2_i16, 16);
        signed_try_shl_int_checks(1_i32, 2_i32, 32);
        signed_try_shl_int_checks(1_isize, 2_isize, isize::BITS);
    }

    #[test]
    fn unsigned_integers() {
        unsigned_try_shl_int_checks(1_u8, 2_u8, 8);
        unsigned_try_shl_int_checks(1_u16, 2_u16, 16);
        unsigned_try_shl_int_checks(1_u32, 2_u32, 32);
        unsigned_try_shl_int_checks(1_usize, 2_usize, usize::BITS);
    }
}

mod test_shr {
    use crate::math::{ArithmeticError, num_traits::Ring, ops::TryShr};

    #[allow(clippy::arithmetic_side_effects)]
    fn signed_try_shr_int_checks<
        T: Ring + TryShr + core::fmt::Debug + PartialEq,
    >(
        val: T,
        bits: u32,
    ) {
        unsigned_try_shr_int_checks(val, bits);
    }

    #[allow(clippy::arithmetic_side_effects, clippy::needless_pass_by_value)]
    fn unsigned_try_shr_int_checks<
        T: Ring + TryShr + core::fmt::Debug + PartialEq,
    >(
        val: T,
        bits: u32,
    ) {
        assert_eq!(T::ONE.try_shr(1), Ok(T::ZERO));
        if bits > 2 {
            assert_eq!(T::ONE.try_shr(2), Ok(T::ZERO));
        }
        assert_eq!(val.try_shr(bits), Err(ArithmeticError::Overflow));
    }

    #[test]
    fn signed_integers() {
        signed_try_shr_int_checks(16_i8, 8);
        signed_try_shr_int_checks(16_i16, 16);
        signed_try_shr_int_checks(32_i32, 32);
        signed_try_shr_int_checks(64_isize, isize::BITS);
    }

    #[test]
    fn unsigned_integers() {
        unsigned_try_shr_int_checks(16_u8, 8);
        unsigned_try_shr_int_checks(16_u16, 16);
        unsigned_try_shr_int_checks(32_u32, 32);
        unsigned_try_shr_int_checks(64_usize, usize::BITS);
    }
}
