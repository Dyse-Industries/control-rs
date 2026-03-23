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
//! - `Exponential`: Extends `Field` with exponential and root functions.
//! - `Logarithm`: Extends `Field` with logarithmic functions.
//! - `Trig`: Extends `Field` with trigonometric functions.
//! - `Real`: Extends `Field` and aggregates `Signed`, `Exponential`, `Logarithm` and `Trig`.

use crate::math::CartesianQuadrant2D;
use crate::math::ops::{Add, Div, Mul, Neg, Sub};

use core::cmp::Ordering;

/// The base marker trait for numbers.
///
/// `Scalar` groups types suitable for numerical operations. It ensures the type can be cloned
/// and compared.
///
/// # Safety
/// `PartialEq` and `PartialOrd` must be consistent and correct. For floating-point types,
/// as per the IEEE 754 standard, `NaN != NaN`.
///
/// # Example
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
/// Arithmetic operations (`Add`, `Sub`, `Mul`) must obey ring axioms
/// (associativity, commutativity, distributivity).
///
/// # Example
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
/// Division by zero for integer types will panic. Floating-point division by zero does
/// not panic but produces non-finite values (`inf` or `NaN`).
///
/// # Example
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

/// Trait for types that support square root (radical functions).
///
/// # Panics
/// Square root of a negative number must return an imaginary number or
/// a type that indicates the domain was violated. Thus, the domain for
/// this implementation includes all Self.
pub trait Radical: Field {
    /// Computes the hypotenuse of a triangle with the given side lengths.
    #[must_use]
    #[allow(clippy::arithmetic_side_effects)]
    fn hypot(self, rhs: Self) -> Self {
        ((self.clone() * self) + (rhs.clone() * rhs)).sqrt()
    }
    /// Computes the square root of a number.
    #[must_use]
    fn sqrt(self) -> Self;
}

/// Trait for types that support exponential, power and root functions.
///
/// # Panics
/// Exponential of a negative number must return an imaginary number or
/// a type that indicates the domain was violated. Thus, the domain for
/// this implementation includes all Self.
pub trait Exponential: Field {
    /// Constant representing Euler's number.
    const E: Self;
    /// Computes `e^self`.
    #[must_use]
    fn exp(self) -> Self;
    /// Computes the natural logarithm.
    #[must_use]
    fn ln(self) -> Self;
    /// Computes the base-10 logarithm.
    #[must_use]
    fn log10(self) -> Self;
    /// Computes a number raised to a power.
    #[must_use]
    fn pow(self, n: Self) -> Self;
}

/// Trait for types that support trigonometric functions.
///
/// # Panics
/// Trigonometric functions of an invalid number must return a type that
/// indicates the domain was violated. Thus, the domain for these implementations
/// includes all Self.
pub trait Trig: Field {
    /// Constant representing Pi.
    const PI: Self;
    /// Calculates the inverse cosine of a number (in radians).
    #[must_use]
    fn acos(self) -> Self;
    /// Calculates the inverse sine of a number (in radians).
    #[must_use]
    fn asin(self) -> Self;
    /// Calculates the inverse tangent of a number (in radians).
    #[must_use]
    fn atan(self) -> Self;
    /// Calculates the cosine of a number (in radians).
    #[must_use]
    fn cos(self) -> Self;
    /// Calculates the sine of a number (in radians).
    #[must_use]
    fn sin(self) -> Self;
    /// Calculates the tangent of a number (in radians).
    #[must_use]
    fn tan(self) -> Self;
}

/// Defines a real field with support for common analytic functions like square root and absolute value.
///
/// The `Real` trait extends a `Field` to include operations common for real numbers.
///
/// # Example
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
/// // sqrt of a negative number
/// let negative_val = -1.0f32;
/// ```
pub trait Real: Field + Signed + Radical + Exponential + Trig {
    /// Constant representing Infinity.
    const INF: Self;
    /// Constant representing NAN.
    const NAN: Self;
    /// Computes the four-quadrant inverse tangent of `y` and `x` in radians.
    ///
    /// Unlike the standard `atan(y / x)`, this function uses the signs of both
    /// arguments to identify the correct quadrant and handles the case where `x` is zero.
    ///
    /// # Mathematical Definition
    /// The returned value `θ` is in the range `(-π, π]` such that:
    /// - `θ > 0` if `y > 0` (Upper half-plane)
    /// - `θ < 0` if `y < 0` (Lower half-plane)
    /// - `θ = 0` if `y = 0` and `x > 0`
    /// - `θ = π` if `y = 0` and `x < 0`
    ///
    /// # Special Cases
    /// | Input Condition | Output Value |
    /// |--------------------------|--------------------|
    /// | `y = 0, x > 0` | `0` |
    /// | `y = 0, x < 0` | `π` |
    /// | `y > 0, x = 0` | `π / 2` |
    /// | `y < 0, x = 0` | `-π / 2` |
    /// | `y = 0, x = 0` | `0` (Undefined*) |
    ///
    /// *Note: Many implementations return 0 for origin coordinates to avoid NaNs
    /// in real-time control loops.*
    #[must_use]
    #[allow(clippy::arithmetic_side_effects)]
    fn atan2(self, rhs: Self) -> Self {
        match CartesianQuadrant2D::from_coords(&self, &rhs) {
            CartesianQuadrant2D::Origin
            | CartesianQuadrant2D::PositiveXAxis => Self::ZERO,
            CartesianQuadrant2D::NegativeYAxis => -Self::PI / Self::TWO,
            CartesianQuadrant2D::NegativeXAxis => Self::PI,
            CartesianQuadrant2D::PositiveYAxis => Self::PI / Self::TWO,
            _ => Self::atan(self / rhs),
        }
    }
    /// Computes the hyperbolic cosine of the number.
    ///
    /// # Mathematical Definition
    /// The hyperbolic cosine is an even function defined as the average of
    /// the natural exponential function and its inverse:
    /// <pre> cosh(x) = (e^x + e^-x) / 2 </pre>
    ///
    /// # Characteristics
    /// - **Domain**: `(-infty, infty)`
    /// - **Range**: `[1, infty)`
    /// - **Symmetry**: `cosh(-x) = cosh(x)`
    ///
    /// # Overflow Warning
    /// Because this implementation relies on `.exp()`, evaluating this function for
    /// large inputs will result in rapid overflow to infinity (e.g., around `x ~ 89.4` for `f32`).
    #[must_use]
    #[allow(clippy::arithmetic_side_effects)]
    fn cosh(self) -> Self {
        (self.clone().exp() + (Self::ZERO - self).exp()) / Self::TWO
    }
    /// Computes the hyperbolic sine of the number.
    ///
    /// # Mathematical Definition
    /// The hyperbolic sine is an odd function defined as half the difference
    /// between the natural exponential function and its inverse:
    /// $$ \sinh(x) = \frac{e^x - e^{-x}}{2} $$
    ///
    /// # Characteristics
    /// - **Domain**: $(-\infty, \infty)$
    /// - **Range**: $(-\infty, \infty)$
    /// - **Symmetry**: $\sinh(-x) = -\sinh(x)$
    ///
    /// # Numerical Stability Note
    /// This default trait implementation uses the standard algebraic definition.
    /// For values of `self` very close to `0.0`, computing $e^x - e^{-x}$ can
    /// suffer from **catastrophic cancellation**, leading to a loss of significant
    /// digits. If your control loops require high precision near the origin,
    /// consider overriding this default with a Taylor series expansion or an `expm1`
    /// based approach for $|x| < 1$.
    #[must_use]
    #[allow(clippy::arithmetic_side_effects)]
    fn sinh(self) -> Self {
        (self.clone().exp() - (Self::ZERO - self).exp()) / Self::TWO
    }
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
/// - `$type`: The numeric type for which to implement `Ring` (e.g., `f32`, `i64`).
/// - `$one`: The literal expression for the multiplicative identity (e.g., `1.0`, `1`).
/// - `$zero`: The literal expression for the additive identity (e.g., `0.0`, `0`).
/// - `$max`: The literal expression for the maximum value (e.g., `f32::MAX`, `usize::MAX`).
/// - `$min`: The literal expression for the minimum value (e.g., `f32::MIN`, `isize::MIN`).
/// - `$min_pos`: The literal expression for the minimum positive value.
#[macro_export]
macro_rules! impl_ring {
    ($type:ty, $one:expr, $zero:expr, $max:expr, $min:expr, $min_pos:expr) => {
        impl One for $type {
            const ONE: Self = $one;
        }

        ////////////////////////////////////////////////////////////////////////////////

        impl Zero for $type {
            const ZERO: Self = $zero;
        }

        ////////////////////////////////////////////////////////////////////////////////

        impl Ring for $type {
            const MAX: Self = $max;
            const MIN: Self = $min;
            const MIN_POSITIVE: Self = $min_pos;
            const TWO: Self = $one + $one;
        }

        ////////////////////////////////////////////////////////////////////////////////
    };
}

/// Implements the `Exponential` trait for a given type.
///
/// # Arguments
/// - `$type`: The numeric type.
/// - `$e:expr`: Literal or constant representing Euler's number.
/// - `$exp:path`: The `exp` function.
/// - `$pow:path`: The `pow` function.
/// - `$sqrt:path`: The `sqrt` function.
#[macro_export]
macro_rules! impl_exp {
    ($type:ty, $e:expr, $exp:path, $ln:path, $log10:path, $pow:path, $sqrt:path) => {
        impl Radical for $type {
            #[inline(always)]
            fn sqrt(self) -> Self {
                $sqrt(self)
            }
        }

        ////////////////////////////////////////////////////////////////////////////////

        impl Exponential for $type {
            const E: Self = $e;
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
        }

        ////////////////////////////////////////////////////////////////////////////////
    };
}

/// Implements the `Trig` trait for a given type.
///
/// # Arguments
/// - `$type`: The numeric type.
/// - `$pi:expr`: Literal or constant representing π.
/// - `$sin:path`: The `sin` function.
/// - `$cos:path`: The `cos` function.
/// - `$tan:path`: The `tan` function.
#[macro_export]
macro_rules! impl_trig {
    ($type:ty, $pi:expr, $cos:path, $sin:path, $tan:path, $acos:path, $asin:path, $atan:path) => {
        impl Trig for $type {
            const PI: Self = $pi;
            #[inline(always)]
            fn acos(self) -> Self {
                $acos(self)
            }
            #[inline(always)]
            fn asin(self) -> Self {
                $asin(self)
            }
            #[inline(always)]
            fn atan(self) -> Self {
                $atan(self)
            }
            #[inline(always)]
            fn cos(self) -> Self {
                $cos(self)
            }
            #[inline(always)]
            fn sin(self) -> Self {
                $sin(self)
            }
            #[inline(always)]
            fn tan(self) -> Self {
                $tan(self)
            }
        }

        ////////////////////////////////////////////////////////////////////////////////
    };
}

/// Implements the `Real` trait for a given type.
///
/// This macro implements the `Real` trait and the `Signed` trait.
///
/// # Arguments
/// - `$type`: The numeric type.
/// - `$abs:path`: The `abs` function.
/// - `$e:expr`: Literal or constant representing Euler's number.
/// - `$inf:expr`: Literal or constant representing infinity.
/// - `$nan:expr`: Literal or constant representing NaN.
#[macro_export]
macro_rules! impl_real {
    ($type:ty, $abs:path, $inf:expr, $nan:expr, $atan2:path) => {
        impl Signed for $type {
            #[inline(always)]
            fn abs(self) -> Self {
                $abs(self)
            }
        }

        ////////////////////////////////////////////////////////////////////////////////

        impl Real for $type {
            const INF: Self = $inf;
            const NAN: Self = $nan;
            fn atan2(self, rhs: Self) -> Self {
                $atan2(self, rhs)
            }
        }

        ////////////////////////////////////////////////////////////////////////////////
    };
}

/// A macro to implement the `Field` trait for a given numeric type.
///
/// This macro simplifies implementing the `Field` trait by generating the `epsilon`
/// function. It is designed for floating-point types that have a defined machine
/// epsilon value.
///
/// # Arguments
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

        ////////////////////////////////////////////////////////////////////////////////
    };
}

////////////////////////////////////////////////////////////////////////////////
// Cartesian Quadrant Helper
////////////////////////////////////////////////////////////////////////////////

impl CartesianQuadrant2D {
    /// Instantiate a `CartesianQuadrant2D` from 2D coordinate.
    ///
    /// # Generic Arguments
    /// * `T` - Type of the coordinates.
    ///
    /// # Arguments
    /// * `x` - The first coordinate value.
    /// * `y` - The second coordinate value.
    ///
    /// # Returns
    /// * `quadrant` - Variant of the `CartesianQuadrant2D` enum corresponding to the coordinates.
    #[must_use]
    pub fn from_coords<T: Zero + PartialOrd>(x: &T, y: &T) -> Self {
        match (x.partial_cmp(&T::ZERO), y.partial_cmp(&T::ZERO)) {
            (Some(Ordering::Equal), Some(Ordering::Equal)) => Self::Origin,

            (Some(Ordering::Equal), Some(Ordering::Greater)) => {
                Self::PositiveYAxis
            }
            (Some(Ordering::Equal), Some(Ordering::Less)) => {
                Self::NegativeYAxis
            }

            (Some(Ordering::Greater), Some(Ordering::Equal)) => {
                Self::PositiveXAxis
            }
            (Some(Ordering::Less), Some(Ordering::Equal)) => {
                Self::NegativeXAxis
            }

            (Some(Ordering::Greater), Some(Ordering::Greater)) => Self::Q1,
            (Some(Ordering::Less), Some(Ordering::Greater)) => Self::Q2,
            (Some(Ordering::Less), Some(Ordering::Less)) => Self::Q3,
            (Some(Ordering::Greater), Some(Ordering::Less)) => Self::Q4,

            // Catch-all handles Cases where x or y are NaN, returning Undefined
            _ => Self::Undefined,
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Implementations for f32 (Embedded standard)
////////////////////////////////////////////////////////////////////////////////

impl Scalar for f32 {}

////////////////////////////////////////////////////////////////////////////////

impl_ring!(f32, 1.0, 0.0, f32::MAX, f32::MIN, f32::MIN_POSITIVE);

impl_field!(f32, f32::EPSILON);

impl_exp!(
    f32,
    core::f32::consts::E,
    libm::expf,
    libm::logf,
    libm::log10f,
    libm::powf,
    libm::sqrtf
);

impl_trig!(
    f32,
    core::f32::consts::PI,
    libm::cosf,
    libm::sinf,
    libm::tanf,
    libm::acosf,
    libm::asinf,
    libm::atanf
);

impl_real!(f32, libm::fabsf, f32::INFINITY, f32::NAN, libm::atan2f);

////////////////////////////////////////////////////////////////////////////////
// Implementations for f64
////////////////////////////////////////////////////////////////////////////////

impl Scalar for f64 {}

////////////////////////////////////////////////////////////////////////////////

impl_ring!(f64, 1.0, 0.0, f64::MAX, f64::MIN, f64::MIN_POSITIVE);

impl_field!(f64, f64::EPSILON);

impl_exp!(
    f64,
    core::f64::consts::E,
    libm::exp,
    libm::log,
    libm::log10,
    libm::pow,
    libm::sqrt
);

impl_trig!(
    f64,
    core::f64::consts::PI,
    libm::cos,
    libm::sin,
    libm::tan,
    libm::acos,
    libm::asin,
    libm::atan
);

impl_real!(f64, libm::fabs, f64::INFINITY, f64::NAN, libm::atan2);

////////////////////////////////////////////////////////////////////////////////
// Implementations for i8
////////////////////////////////////////////////////////////////////////////////
impl Scalar for i8 {}

////////////////////////////////////////////////////////////////////////////////

impl_ring!(i8, 1, 0, i8::MAX, i8::MIN, 1);

////////////////////////////////////////////////////////////////////////////////

impl Signed for i8 {
    #[inline(always)]
    fn abs(self) -> Self {
        self.abs()
    }
}

////////////////////////////////////////////////////////////////////////////////
// Implementations for i16
////////////////////////////////////////////////////////////////////////////////
impl Scalar for i16 {}

////////////////////////////////////////////////////////////////////////////////

impl_ring!(i16, 1, 0, i16::MAX, i16::MIN, 1);

impl Signed for i16 {
    #[inline(always)]
    fn abs(self) -> Self {
        self.abs()
    }
}

////////////////////////////////////////////////////////////////////////////////
// Implementations for i32
////////////////////////////////////////////////////////////////////////////////
impl Scalar for i32 {}

////////////////////////////////////////////////////////////////////////////////

impl_ring!(i32, 1, 0, i32::MAX, i32::MIN, 1);

impl Signed for i32 {
    #[inline(always)]
    fn abs(self) -> Self {
        self.abs()
    }
}

////////////////////////////////////////////////////////////////////////////////
// Implementations for i64
////////////////////////////////////////////////////////////////////////////////
impl Scalar for i64 {}

////////////////////////////////////////////////////////////////////////////////

impl_ring!(i64, 1, 0, i64::MAX, i64::MIN, 1);

impl Signed for i64 {
    #[inline(always)]
    fn abs(self) -> Self {
        self.abs()
    }
}

////////////////////////////////////////////////////////////////////////////////
// Implementations for i128
////////////////////////////////////////////////////////////////////////////////
impl Scalar for i128 {}

////////////////////////////////////////////////////////////////////////////////

impl_ring!(i128, 1, 0, i128::MAX, i128::MIN, 1);

impl Signed for i128 {
    #[inline(always)]
    fn abs(self) -> Self {
        self.abs()
    }
}

////////////////////////////////////////////////////////////////////////////////
// Implementations for isize
////////////////////////////////////////////////////////////////////////////////
impl Scalar for isize {}

////////////////////////////////////////////////////////////////////////////////

impl_ring!(isize, 1, 0, isize::MAX, isize::MIN, 1);

impl Signed for isize {
    #[inline(always)]
    fn abs(self) -> Self {
        self.abs()
    }
}

////////////////////////////////////////////////////////////////////////////////
// Implementations for u8
////////////////////////////////////////////////////////////////////////////////
impl Scalar for u8 {}

////////////////////////////////////////////////////////////////////////////////

impl_ring!(u8, 1, 0, u8::MAX, u8::MIN, 1);

impl Unsigned for u8 {}

////////////////////////////////////////////////////////////////////////////////
// Implementations for u16
////////////////////////////////////////////////////////////////////////////////
impl Scalar for u16 {}

////////////////////////////////////////////////////////////////////////////////

impl_ring!(u16, 1, 0, u16::MAX, u16::MIN, 1);

impl Unsigned for u16 {}

////////////////////////////////////////////////////////////////////////////////
// Implementations for u32
////////////////////////////////////////////////////////////////////////////////
impl Scalar for u32 {}

////////////////////////////////////////////////////////////////////////////////

impl_ring!(u32, 1, 0, u32::MAX, u32::MIN, 1);

impl Unsigned for u32 {}

////////////////////////////////////////////////////////////////////////////////
// Implementations for u64
////////////////////////////////////////////////////////////////////////////////
impl Scalar for u64 {}

////////////////////////////////////////////////////////////////////////////////

impl_ring!(u64, 1, 0, u64::MAX, u64::MIN, 1);

impl Unsigned for u64 {}

////////////////////////////////////////////////////////////////////////////////
// Implementations for u128
////////////////////////////////////////////////////////////////////////////////
impl Scalar for u128 {}

////////////////////////////////////////////////////////////////////////////////

impl_ring!(u128, 1, 0, u128::MAX, u128::MIN, 1);

impl Unsigned for u128 {}

////////////////////////////////////////////////////////////////////////////////
// Implementations for usize
////////////////////////////////////////////////////////////////////////////////
impl Scalar for usize {}

////////////////////////////////////////////////////////////////////////////////

impl_ring!(usize, 1, 0, usize::MAX, usize::MIN, 1);

impl Unsigned for usize {}
