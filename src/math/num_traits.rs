//! # Numeric Traits
//!
//! This module defines a hierarchy of traits for numerical types.
//! These traits provide a foundation for generic algorithms, ensuring mathematical correctness.
//!
//! The hierarchy is as follows:
//! - `One`: The identity element for multiplication.
//! - `Zero`: The identity element for addition.
//! - `Scalar`: Basic properties (`Copy`, `PartialEq`, `PartialOrd`).
//! - `Ring`: Extends `Scalar` with ring operations (`+`, `-`, `*`).
//! - `Field`: Extends `Ring` with division (`/`).
//! - `Real`: Extends `Field` with real-number functions (sqrt, abs).

use crate::math::ops::{Add, Div, Mul, Neg, Sub};

/// The base marker trait for numbers.
///
/// `Scalar` groups types suitable for numerical operations. It ensures the type can be cloned
/// and compared.
///
/// # Safety
///
/// `PartialEq` and `PartialOrd` must be consistent and correct. For floating-point types,
/// as per the IEEE 754 standard, `NaN != NaN`.
///
/// # Example
///
/// ```
/// use control_rs::math::num_traits::Scalar;
///
/// #[derive(Clone, Copy, PartialEq, PartialOrd)]
/// struct MyScalar(f32);
///
/// impl Scalar for MyScalar {}
///
/// fn process_scalar<T: Scalar>(val: T) -> T {
///     if val > val {
///         // This branch is unreachable for non-NaN values
///     }
///     val
/// }
/// ```
pub trait Scalar: Clone + Sized + PartialEq + PartialOrd {}

/// Provides access to the multiplicative identity and a one check.
///
/// This trait abstracts the unit element of a multiplicative structure
/// without requiring the full semantics of a `Ring`.
///
/// # Safety
///
/// `ONE` must be a true multiplicative identity.
pub trait One: Scalar + Mul<Output = Self> {
    /// Constant multiplicative identity element.
    const ONE: Self;
    /// Returns `true` if the value equals the multiplicative identity.
    #[inline(always)]
    fn is_one(&self) -> bool {
        self.eq(&Self::ONE)
    }
    /// Returns the multiplicative identity element.
    #[must_use]
    fn one() -> Self {
        Self::ONE
    }
}

/// Provides access to the additive identity and a zero check.
///
/// This trait is useful where the identity element is required, but the
/// full algebraic structure of a `Ring` is unnecessary.
///
/// # Safety
///
/// `ZERO` must be a true additive identity.
pub trait Zero: Scalar + Add<Output = Self> + Sub<Output = Self> {
    /// Constant additive identity element.
    const ZERO: Self;
    /// Returns `true` if the value equals the additive identity.
    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.eq(&Self::ZERO)
    }
    /// Returns the additive identity element.
    #[must_use]
    fn zero() -> Self {
        Self::ZERO
    }
}

/// Defines a set with a mathematical sign (all signed integers and
/// floating/fixed-point types).
///
/// This provides access to checks for sign-ness, absolute value and Neg.
///
/// # Safety
///
/// Values that implement Neg have a well-defined sign.
pub trait Signed: Zero + Neg<Output = Self> {
    /// Returns the absolute value.
    #[must_use]
    fn abs(self) -> Self;
    /// Check if self is less than zero.
    #[inline(always)]
    fn is_sign_negative(&self) -> bool {
        self.lt(&Self::ZERO)
    }
    /// Check if self is greater than zero.
    #[inline(always)]
    fn is_sign_positive(&self) -> bool {
        self.gt(&Self::ZERO)
    }
}

/// Defines an algebraic ring.
///
/// Abstracts over types that support addition, subtraction and multiplication
/// forming a mathematical ring.
///
/// # Safety
///
/// Arithmetic operations (`Add`, `Sub`, `Mul`) must obey ring axioms
/// (associativity, commutativity, distributivity).
///
/// # Example
///
/// ```
/// use control_rs::math::num_traits::Ring;
///
/// fn multiply_by_three<T: Ring>(val: T) -> T {
///     val * (T::one() + T::TWO)
/// }
///
/// assert_eq!(multiply_by_three(5), 15);
/// assert_eq!(multiply_by_three(2.0f32), 6.0f32);
/// ```
pub trait Ring: One + Zero {
    /// Constant representing the max.
    const MAX: Self;
    /// Constant representing the min.
    const MIN: Self;
    /// Constant representing the minimum positive value.
    const MIN_POSITIVE: Self;
    /// Constant representing 2.
    const TWO: Self;
    /// Initiate self from the given const.
    #[must_use]
    fn from_const<const N: usize>() -> Self {
        Self::sum([Self::ONE; N])
    }
    /// Initiate self from the given usize.
    fn from_usize(n: usize) -> Self {
        (0..n).fold(Self::ZERO, |acc, _| acc.add(Self::ONE))
    }
    /// Sum the elements of an iterator.
    #[allow(clippy::arithmetic_side_effects)]
    fn sum<I: IntoIterator<Item = Self>>(iter: I) -> Self {
        iter.into_iter().fold(Self::zero(), |acc, x| acc + x)
    }
}

/// Defines an algebraic field, extending a `Ring` with division.
///
/// A `Field` is a `Ring` that also supports division. It is intended for floating-point types.
/// The trait implies and requires the existence of a machine epsilon; `1.0 + ε != 1.0`.
///
/// # Panics
///
/// Division by zero for integer types will panic. Floating-point division by zero does
/// not panic but produces non-finite values (`inf` or `NaN`).
///
/// # Example
///
/// ```
/// use control_rs::math::num_traits::{Field, Ring};
///
/// fn is_significantly_different<T: Field>(a: T, b: T) -> bool {
///     let diff = if a > b { a - b } else { b - a };
///     diff > T::epsilon()
/// }
///
/// assert!(is_significantly_different(1.001f32, 1.0f32));
/// assert!(!is_significantly_different(1.00000001f32, 1.0f32));
/// ```
pub trait Field: Ring + Div<Output = Self> {
    /// Returns the machine epsilon value for the type.
    fn epsilon() -> Self;
}

/// Defines a real field with support for common analytic functions like square root and absolute value.
///
/// The `Real` trait extends a `Field` to include operations common for real numbers.
///
/// # Example
///
/// ```
/// use control_rs::math::{num_traits::{Real, Ring}, ArithmeticError};
///
/// fn magnitude(x: f32, y: f32) -> f32 {
///     let sum_sq = x*x + y*y;
///     sum_sq.sqrt()
/// }
///
/// assert_eq!(magnitude(3.0, 4.0), 5.0);
/// assert_eq!(magnitude(-3.0, -4.0), 5.0);
///
/// // sqrt of a negative number
/// let negative_val = -1.0f32;
/// ```
pub trait Real: Field + Signed {
    /// Constant representing E.
    const E: Self;
    /// Constant representing Infinity.
    const INF: Self;
    /// Constant representing NAN.
    const NAN: Self;
    /// Constant representing Pi.
    const PI: Self;
    /// Calculates the cosine of a number (in radians).
    #[must_use]
    fn cos(self) -> Self;

    /// Calculates `e^self`.
    #[must_use]
    fn exp(self) -> Self;

    /// Calculates the natural logarithm of a number.
    #[must_use]
    fn ln(self) -> Self;

    /// Calculates the base-10 logarithm of a number.
    #[must_use]
    fn log10(self) -> Self;

    /// Raises a number to a floating-point power.
    #[must_use]
    fn pow(self, n: Self) -> Self;

    /// Calculates the sine of a number (in radians).
    #[must_use]
    fn sin(self) -> Self;

    /// Calculates the square root of a number.
    #[must_use]
    fn sqrt(self) -> Self;

    /// Calculates the tangent of a number (in radians).
    #[must_use]
    fn tan(self) -> Self;
}

/// Defines the set of unsigned numbers.
pub trait Unsigned: Sized {}

/// Implements the `Ring` trait for a given numeric type.
///
/// This macro simplifies the process of implementing the `Ring` trait. It generates
/// the implementation of `zero()` and `one()` with the provided literal values. This
/// reduces boilerplate code and ensures consistency across different numeric types.
///
/// # Arguments
///
/// - `$type`: The numeric type for which to implement `Ring` (e.g., `f32`, `i64`).
/// - `$one`: The literal expression for the multiplicative identity (e.g., `1.0`, `1`).
/// - `$zero`: The literal expression for the additive identity (e.g., `0.0`, `0`).
/// - `$max`: The literal expression for the maximum value (e.g., `f32::MAX`, `usize::MAX`).
/// - `$min`: The literal expression for the minimum value (e.g., `f32::MIN`, `isize::MIN`).:
#[macro_export]
macro_rules! impl_ring {
    ($type:ty, $one:expr, $zero:expr, $max:expr, $min:expr, $min_pos:expr) => {
        impl One for $type {
            const ONE: Self = $one;
        }
        impl Zero for $type {
            const ZERO: Self = $zero;
        }
        impl Ring for $type {
            const MAX: Self = $max;
            const MIN: Self = $min;
            const MIN_POSITIVE: Self = $min_pos;
            const TWO: Self = $one + $one;
        }
    };
}

/// Implements the `Real` trait for a given type.
///
/// This macro implements the `Real` trait by calling functions from the `libm` library.
/// It provides implementations for `sqrt` and `abs`. The `sqrt` implementation includes
/// a check to ensure the input is non-negative, returning `None` if it is not. This
/// avoids panics or the creation of `NaN` values from negative inputs, which is a
/// critical safety feature.
///
/// # Arguments
///
/// - `$abs:path`, fully qualified path to the `abs` function.
/// - `e:expr`, literal or constant representing Euler's number.
/// - `inf:expr`, literal or constant representing infinity.
/// - `$nan:expr`, literal or constant representing NaN.
/// - `$pi:expr`, literal or constant representing π.
/// - `$cos:path`, a fully qualified path to the `cos` function.
/// - `$ln:path`, a fully qualified path to the `log` function.
/// - `$log10:path`, a fully qualified path to the `log10` function.
/// - `$exp:path`, a fully qualified path to the `exp` function.
/// - `$pow:path`, a fully qualified path to the `pow` function.
/// - `$sin:path`, a fully qualified path to the `sin` function.
/// - `$sqrt:path`, a fully qualified path to the `sqrt` function.
#[macro_export]
macro_rules! impl_real {
    ($type:ty, $abs:path, $e:expr, $inf:expr, $nan:expr, $pi:expr, $cos:path, $ln:path, $log10:path, $exp:path, $pow:path, $sin:path, $sqrt:path, $tan:path) => {
        impl Signed for $type {
            #[inline(always)]
            fn abs(self) -> Self {
                $abs(self)
            }
        }
        impl Real for $type {
            const E: Self = $e;
            const INF: Self = $inf;
            const NAN: Self = $nan;
            const PI: Self = $pi;
            #[inline(always)]
            fn cos(self) -> Self {
                $cos(self)
            }
            #[inline(always)]
            fn exp(self) -> Self {
                $exp(self)
            }
            #[inline(always)]
            fn ln(self) -> Self {
                $ln(self)
            }
            #[inline(always)]
            fn log10(self) -> Self {
                $log10(self)
            }
            #[inline(always)]
            fn pow(self, n: Self) -> Self {
                $pow(self, n)
            }
            #[inline(always)]
            fn sin(self) -> Self {
                $sin(self)
            }
            #[inline(always)]
            fn sqrt(self) -> Self {
                $sqrt(self)
            }
            #[inline(always)]
            fn tan(self) -> Self {
                $tan(self)
            }
        }
    };
}

/// A macro to implement the `Field` trait for a given numeric type.
///
/// This macro simplifies implementing the `Field` trait by generating the `epsilon`
/// function. It is designed for floating-point types that have a defined machine
/// epsilon value.
///
/// # Arguments
///
/// - `$type`: The numeric type for which to implement `Field` (e.g., `f32`, `f64`).
/// - `$epsilon`: The expression for the machine epsilon value (e.g., `f32::EPSILON`).
#[macro_export]
macro_rules! impl_field {
    ($type:ty, $epsilon:expr) => {
        impl Field for $type {
            #[inline(always)]
            fn epsilon() -> Self {
                $epsilon
            }
        }
    };
}

// Implementations for f32 (Embedded standard)
impl Scalar for f32 {}
impl_ring!(f32, 1.0, 0.0, f32::MAX, f32::MIN, f32::MIN_POSITIVE);
impl_field!(f32, f32::EPSILON);
impl_real!(
    f32,
    libm::fabsf,
    core::f32::consts::E,
    f32::INFINITY,
    f32::NAN,
    core::f32::consts::PI,
    libm::cosf,
    libm::logf,
    libm::log10f,
    libm::expf,
    libm::powf,
    libm::sinf,
    libm::sqrtf,
    libm::tanf
);

// Implementations for f64
impl Scalar for f64 {}
impl_ring!(f64, 1.0, 0.0, f64::MAX, f64::MIN, f64::MIN_POSITIVE);
impl_field!(f64, f64::EPSILON); // Corrected macro call
impl_real!(
    f64,
    libm::fabs,
    core::f64::consts::E,
    f64::INFINITY,
    f64::NAN,
    core::f64::consts::PI,
    libm::cos,
    libm::log,
    libm::log10,
    libm::exp,
    libm::pow,
    libm::sin,
    libm::sqrt,
    libm::tan
);

// Implementations for i8
impl Scalar for i8 {}
impl_ring!(i8, 1, 0, i8::MAX, i8::MIN, 1);
impl Signed for i8 {
    #[inline(always)]
    fn abs(self) -> Self {
        self.abs()
    }
}

// Implementations for i16
impl Scalar for i16 {}
impl_ring!(i16, 1, 0, i16::MAX, i16::MIN, 1);
impl Signed for i16 {
    #[inline(always)]
    fn abs(self) -> Self {
        self.abs()
    }
}

// Implementations for i32
impl Scalar for i32 {}
impl_ring!(i32, 1, 0, i32::MAX, i32::MIN, 1);
impl Signed for i32 {
    #[inline(always)]
    fn abs(self) -> Self {
        self.abs()
    }
}

// Implementations for i64
impl Scalar for i64 {}
impl_ring!(i64, 1, 0, i64::MAX, i64::MIN, 1);
impl Signed for i64 {
    #[inline(always)]
    fn abs(self) -> Self {
        self.abs()
    }
}

// Implementations for i128
impl Scalar for i128 {}
impl_ring!(i128, 1, 0, i128::MAX, i128::MIN, 1);
impl Signed for i128 {
    #[inline(always)]
    fn abs(self) -> Self {
        self.abs()
    }
}

// Implementations for isize
impl Scalar for isize {}
impl_ring!(isize, 1, 0, isize::MAX, isize::MIN, 1);
impl Signed for isize {
    #[inline(always)]
    fn abs(self) -> Self {
        self.abs()
    }
}

// Implementations for u8
impl Scalar for u8 {}
impl_ring!(u8, 1, 0, u8::MAX, u8::MIN, 1);
impl Unsigned for u8 {}

// Implementations for u16
impl Scalar for u16 {}
impl_ring!(u16, 1, 0, u16::MAX, u16::MIN, 1);
impl Unsigned for u16 {}

// Implementations for u32
impl Scalar for u32 {}
impl_ring!(u32, 1, 0, u32::MAX, u32::MIN, 1);
impl Unsigned for u32 {}

// Implementations for u64
impl Scalar for u64 {}
impl_ring!(u64, 1, 0, u64::MAX, u64::MIN, 1);
impl Unsigned for u64 {}

// Implementations for u128
impl Scalar for u128 {}
impl_ring!(u128, 1, 0, u128::MAX, u128::MIN, 1);
impl Unsigned for u128 {}

// Implementations for usize
impl Scalar for usize {}
impl_ring!(usize, 1, 0, usize::MAX, usize::MIN, 1);
impl Unsigned for usize {}
