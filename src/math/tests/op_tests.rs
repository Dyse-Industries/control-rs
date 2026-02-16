use crate::math::ArithmeticError;
use crate::math::ops::{
    TryAdd, TryDiv, TryMul, TryNeg, TryRem, TryShl, TryShr, TrySub,
};

mod add {
    use super::*;

    #[test]
    fn i32_add() {
        assert_eq!(1i32.try_add(&1i32), Ok(2i32));
        assert_eq!(i32::MAX.try_add(&1i32), Err(ArithmeticError::Overflow));
    }

    #[test]
    fn f32_add() {
        assert_eq!(1f32.try_add(&1f32), Ok(2f32));
        assert_eq!(f32::MAX.try_add(&f32::MAX), Err(ArithmeticError::Overflow));
        assert_eq!(f32::MIN.try_add(&f32::MIN), Err(ArithmeticError::Overflow));
    }
}

mod sub {
    use super::*;

    #[test]
    fn i32_sub() {
        assert_eq!(2i32.try_sub(&1i32), Ok(1i32));
        assert_eq!(i32::MIN.try_sub(&1i32), Err(ArithmeticError::Overflow));
    }

    #[test]
    fn f32_sub() {
        assert_eq!(2f32.try_sub(&1f32), Ok(1f32));
        assert_eq!(f32::MIN.try_sub(&f32::MAX), Err(ArithmeticError::Overflow));
        assert_eq!(f32::MAX.try_sub(&f32::MIN), Err(ArithmeticError::Overflow));
    }
}

mod mul {
    use super::*;

    #[test]
    fn i32_mul() {
        assert_eq!(2i32.try_mul(&3i32), Ok(6i32));
        assert_eq!(i32::MAX.try_mul(&2i32), Err(ArithmeticError::Overflow));
    }

    #[test]
    fn f32_mul() {
        assert_eq!(2f32.try_mul(&3f32), Ok(6f32));
        assert_eq!(f32::MAX.try_mul(&2f32), Err(ArithmeticError::Overflow));
        assert_eq!(f32::MAX.try_mul(&-2f32), Err(ArithmeticError::Overflow));
    }
}

mod div {
    use super::*;

    #[test]
    fn i32_div() {
        assert_eq!(6i32.try_div(&3i32), Ok(2i32));
        assert_eq!(1i32.try_div(&0i32), Err(ArithmeticError::DivisionByZero));
        assert_eq!(i32::MIN.try_div(&-1i32), Err(ArithmeticError::Overflow));
    }

    #[test]
    fn f32_div() {
        assert_eq!(6f32.try_div(&3f32), Ok(2f32));
        assert_eq!(1f32.try_div(&0f32), Err(ArithmeticError::DivisionByZero));
        assert_eq!(0f32.try_div(&0f32), Err(ArithmeticError::DivisionByZero));
        assert_eq!(f32::MAX.try_div(&0.1f32), Err(ArithmeticError::Overflow));
    }
}

mod rem {
    use super::*;

    #[test]
    fn i32_rem() {
        assert_eq!(7i32.try_rem(&3i32), Ok(1i32));
        assert_eq!(1i32.try_rem(&0i32), Err(ArithmeticError::DivisionByZero));
        assert_eq!(i32::MIN.try_rem(&-1i32), Err(ArithmeticError::Overflow));
    }

    #[test]
    fn f32_rem() {
        assert_eq!(7f32.try_rem(&3f32), Ok(1f32));
        assert_eq!(1f32.try_rem(&0f32), Err(ArithmeticError::DivisionByZero));
        assert!(f32::INFINITY.try_rem(&1.0).unwrap().is_nan());
    }
}

mod neg {
    use super::*;

    #[test]
    fn i32_neg() {
        assert_eq!(1i32.try_neg(), Ok(-1i32));
        assert_eq!(i32::MIN.try_neg(), Err(ArithmeticError::Overflow));
    }

    #[test]
    fn f32_neg() {
        assert_eq!(1f32.try_neg(), Ok(-1f32));
        assert_eq!((-1f32).try_neg(), Ok(1f32));
    }
}

mod shl {
    use super::*;

    #[test]
    fn i32_shl() {
        assert_eq!(1i32.try_shl(1), Ok(2i32));
        assert_eq!(1i32.try_shl(32), Err(ArithmeticError::Overflow));
    }
}

mod shr {
    use super::*;

    #[test]
    fn i32_shr() {
        assert_eq!(2i32.try_shr(1), Ok(1i32));
        assert_eq!(1i32.try_shr(32), Err(ArithmeticError::Overflow));
    }
}
