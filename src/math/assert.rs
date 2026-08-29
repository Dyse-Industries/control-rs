//! # Math Assertions.

use crate::math::{
    ArithmeticResult,
    num_traits::{Float, Signed},
    ops::{TryMul, TrySub},
};

/// Asserts that two floating-point numbers are almost equal.
///
/// This macro compares two expressions, `$left` and `$right` and panics if they are not
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
             Ok(true) => {}
             Err(e) => panic!("assertion failed: {e}")
         }
    });
    ($left:expr, $right:expr, $epsilon:expr) => ({
        let (left_val, right_val) = (&$left, &$right);
        match $crate::math::assert::almost_eq_eps(left_val, right_val, &$epsilon) {
             Ok(false) => {
                 panic!("assertion failed: `(left == right)`\n  left: `{:?}`, \n right: `{:?}` with epsilon `{:?}`", left_val, right_val, &$epsilon)
             }
             Ok(true) => {}
             Err(e) => panic!("assertion failed: {e}")
         }
    });
    ($left:expr, $right:expr, $epsilon:expr, $($arg:tt)+) => ({
        let (left_val, right_val) = (&$left, &$right);
        match $crate::math::assert::almost_eq_eps(left_val, right_val, &$epsilon) {
             Ok(false) => {
                 panic!("assertion failed: `(left == right)`\n  left: `{:?}`, \n right: `{:?}` with epsilon `{:?}`: {}", left_val, right_val, &$epsilon, format_args!($($arg)+))
             }
             Ok(true) => {}
             Err(e) => panic!("assertion failed: {e}: {}", format_args!($($arg)+))
         }
    });
}

/// Asserts that two floating-point numbers are not almost equal.
///
/// Panics if `$left` and `$right` differ by less than the given (or machine)
/// epsilon.
///
/// ```
/// # use control_rs::assert_not_almost_eq;
/// assert_not_almost_eq!(0.1_f64, 0.4_f64);
/// ```
///
/// ```should_panic
/// # use control_rs::assert_not_almost_eq;
/// assert_not_almost_eq!(0.1_f64, 0.1_f64);
/// ```
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
    ($left:expr, $right:expr, $epsilon:expr) => ({
         let (left_val, right_val) = (&$left, &$right);
         match $crate::math::assert::almost_eq_eps(left_val, right_val, &$epsilon) {
             Ok(true) => {
                 panic!("assertion failed: `(left != right)`\n  left: `{:?}`, \n right: `{:?}` with epsilon `{:?}`", left_val, right_val, &$epsilon)
             }
             _ => {},
         }
    });
    ($left:expr, $right:expr, $epsilon:expr, $($arg:tt)+) => ({
         let (left_val, right_val) = (&$left, &$right);
         match $crate::math::assert::almost_eq_eps(left_val, right_val, &$epsilon) {
             Ok(true) => {
                 panic!("assertion failed: `(left != right)`\n  left: `{:?}`, \n right: `{:?}` with epsilon `{:?}`: {}", left_val, right_val, &$epsilon, format_args!($($arg)+))
             }
             _ => {},
         }
    });
}

/// Returns `true` if `a` and `b` differ by less than `T::epsilon()`.
///
/// # Errors
/// Returns [`ArithmeticError`] if the subtraction fails (for example a domain
/// violation on `NaN`).
pub fn almost_eq<T>(a: &T, b: &T) -> ArithmeticResult<bool>
where
    T: TrySub + TryMul + Signed + Float,
{
    almost_eq_eps(a, b, &T::epsilon())
}

/// Returns `true` if `a` and `b` differ by less than `epsilon`.
///
/// # Errors
/// Returns [`ArithmeticError`] if the subtraction fails (for example a domain
/// violation on `NaN`).
pub fn almost_eq_eps<T>(a: &T, b: &T, epsilon: &T) -> ArithmeticResult<bool>
where
    T: TrySub + TryMul + Signed + Float,
{
    if a == b {
        return Ok(true);
    }
    let abs_a = T::abs(a.clone());
    let abs_b = T::abs(b.clone());
    a.try_sub(b).map(|diff| {
        let abs_diff = T::abs(diff);
        let largest = if abs_a > abs_b { abs_a } else { abs_b };
        abs_diff <= epsilon.try_mul(&largest).unwrap_or(T::ZERO)
            || abs_diff < epsilon.clone()
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
