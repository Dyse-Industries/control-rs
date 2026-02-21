//! # Math Assertions.

use super::{ArithmeticError, num_traits::Real, ops::TrySub};

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
macro_rules! assert_almost_eq {
    ($left:expr, $right:expr) => ({
        match (&$left, &$right) {
            (left_val, right_val) => {
                 match $crate::math::assert::almost_eq(*left_val, *right_val) {
                     Ok(is_almost_eq) => {
                         if !is_almost_eq {
                             panic!("assertion failed: `(left == right)`\n  left: `{:?}`, \n right: `{:?}`", *left_val, *right_val)
                         }
                     }
                     Err(e) => panic!("assertion failed: {e}")
                 }
            }
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
        match (&$left, &$right) {
            (left_val, right_val) => {
                 match $crate::math::assert::almost_eq(*left_val, *right_val) {
                     Ok(is_almost_eq) => {
                         if is_almost_eq {
                             panic!("assertion failed: `(left != right)`\n  left: `{:?}`, \n right: `{:?}`", *left_val, *right_val)
                         }
                     }
                     _ => {},
                 }
            }
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
pub fn almost_eq<T>(a: T, b: T) -> Result<bool, ArithmeticError>
where
    T: PartialOrd + PartialEq + TrySub<Output = T> + Real,
{
    if a == b {
        return Ok(true);
    }
    a.try_sub(&b).map(|diff| T::abs(diff) < T::epsilon())
}

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
}
