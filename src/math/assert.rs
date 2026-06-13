//! # Math Assertions.

use crate::math::{
    num_traits::{Field, Signed},
    ops::{TryMul, TrySub},
    ArithmeticResult,
};

/// Asserts that two floating-point numbers are almost equal.
///
/// This macro compares two expressions, `$left` and `$right`, and panics if they are not
/// almost equal. The comparison is done using an absolute difference against the machine
/// epsilon for the floating-point type.
///
/// # Usage
///
/// ```
/// # use control_rs::assert_almost_eq;
/// let a = 0.1_f64 + 0.2_f64;
/// let b = 0.3_f64;
/// assert_almost_eq!(a, b);
/// ```
///
/// ```should_panic
/// # use control_rs::assert_almost_eq;
/// assert_almost_eq!(0.1_f32, 0.2_f32);
/// ```
#[macro_export]
macro_rules! assert_almost_eq {
    ($left:expr, $right:expr) => ({
        let (left_val, right_val) = (&$left, &$right);
        match $crate::math::assert::almost_eq(left_val, right_val) {
             Ok(false) => {
                 panic!("assertion failed: `(left == right)`\n  left: `{:?}`, \n right: `{:?}`", left_val, right_val)
             }
             Err(e) => panic!("assertion failed: {e}")
         }
    });
}

/// Asserts that two floating-point numbers are almost equal.
///
/// This macro compares two expressions, `$left` and `$right`, and panics if they are not
/// almost equal. The comparison is done using an absolute difference against the machine
/// epsilon for the floating-point type.
///
/// # Usage
///
/// ```
/// # use control_rs::assert_almost_eq;
/// let a = 0.1_f64 + 0.2_f64;
/// let b = 0.3_f64;
///
/// // This will not panic
/// assert_almost_eq!(a, b);
/// ```
///
/// ```should_panic
/// # use control_rs::assert_almost_eq;
/// // This will panic because the values are not almost equal.
/// assert_almost_eq!(0.1_f32, 0.2_f32);
/// ```
///
#[macro_export]
macro_rules! assert_not_almost_eq {
    ($left:expr, $right:expr) => ({
         let (left_val, right_val) = (&$left, &$right);
         match $crate::math::assert::almost_eq(left_val, right_val) {
             Ok(true) => {
                 panic!("assertion failed: `(left != right)`\n  left: `{:?}`, \n right: `{:?}`", left_val, right_val)
             }
             _ => {},
         }
    });
}

/// Asserts that two floating-point numbers are almost equal.
///
/// This function compares two numbers `a` and `b` of a type `T` that implements
/// the `Real` trait. It returns `true` if the absolute difference between `a` and `b`
/// is less than the machine epsilon for that type.
///
/// # Generic Arguments
/// - `T`: A type that implements `PartialOrd`, `PartialEq`, `TrySub<Output = T>`, and `Real`.
///
/// # Arguments
/// - `a`: The first value to compare.
/// - `b`: The second value to compare.
///
/// # Returns
/// `true` if `a` and `b` are almost equal, `false` otherwise.
///
/// # Errors
/// Return `ArithmeticError` if the subtraction operation fails.
pub fn almost_eq<T>(a: &T, b: &T) -> ArithmeticResult<bool>
where
    T: TrySub + TryMul + Signed + Field,
{
    if a == b {
        return Ok(true);
    }
    let abs_a = T::abs(a.clone());
    let abs_b = T::abs(b.clone());
    a.try_sub(b).map(|diff| {
        let abs_diff = T::abs(diff);
        let largest = if abs_a > abs_b { abs_a } else { abs_b };
        abs_diff <= T::epsilon().try_mul(&largest).unwrap_or(T::ZERO)
            || abs_diff < T::epsilon()
    })
}

////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use crate::assert_almost_eq;

    #[test]
    fn test_assert_almost_eq_f32() {
        assert_almost_eq!(0.1_f32 + 0.2_f32, 0.3_f32);
    }

    #[test]
    fn test_assert_almost_eq_f64() {
        assert_almost_eq!(0.1_f64 + 0.2_f64, 0.3_f64);
    }
    #[test]
    fn test_assert_not_almost_eq_f64() {
        assert_not_almost_eq!(0.1_f64 + 0.2_f64, 0.4_f64);
    }
    #[test]
    #[should_panic(
        expected = "assertion failed: `(left == right)`\n  left: `0.1`, \n right: `0.2`"
    )]
    fn test_assert_almost_eq_f32_panic() {
        assert_almost_eq!(0.1_f32, 0.2_f32);
    }

    #[test]
    #[should_panic(
        expected = "assertion failed: `(left == right)`\n  left: `0.1`, \n right: `0.2`"
    )]
    fn test_assert_almost_eq_f64_panic() {
        assert_almost_eq!(0.1_f64, 0.2_f64);
    }
    #[test]
    #[should_panic(
        expected = "assertion failed: Input is outside the mathematical domain"
    )]
    fn test_assert_almost_eq_domain_violation() {
        assert_almost_eq!(f32::NAN, f32::NAN);
    }
    #[test]
    #[should_panic(
        expected = "assertion failed: `(left != right)`\n  left: `0.1`, \n right: `0.1`"
    )]
    fn test_assert_not_almost_eq_f64_panic() {
        assert_not_almost_eq!(0.1_f64, 0.1_f64);
    }
    #[test]
    fn test_assert_not_eq_domain_violation() {
        assert_not_almost_eq!(f32::NAN, f32::NAN);
    }

    #[test]
    #[should_panic(
        expected = "assertion failed: Significant precision was lost during operation"
    )]
    fn test_assert_almost_eq_underflow() {
        assert_almost_eq!(10e20_f32, (f32::EPSILON / 2.0));
    }
}