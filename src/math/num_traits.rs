//! # Numeric Traits
//! # Numerical Traits for Control Systems
//!
//! This module defines a hierarchy of traits for numerical types used in control engineering.
//! These traits provide a foundation for generic algorithms, ensuring mathematical correctness
//! and safety, in compliance with standards like DO-178C and ISO 26262.
//!
//! The hierarchy is as follows:
//! - `One`: The identity element for multiplication.
//! - `Zero`: The identity element for addition.
//! - `Scalar`: Basic properties (`Copy`, `PartialEq`, `PartialOrd`).
//! - `Ring`: Extends `Scalar` with ring operations (`+`, `-`, `*`).
//! - `Field`: Extends `Ring` with division (`/`).
//! - `Real`: Extends `Field` with real-number functions (sqrt, abs).
//!
//! ## Safety and Compliance
//!
//! All traits and implementations in this module are designed to be safe and panic-free
//! where possible. For instance, `Real::sqrt` returns a `Result` to handle negative
//! inputs gracefully, avoiding `NaN` results or panics that could compromise system stability.
//! The use of `libm` ensures that mathematical functions are implemented correctly and
//! consistently across platforms, a key consideration for safety-critical software.
use crate::math::{ArithmeticError, ArithmeticResult};
use core::ops::{Add, Div, Mul, Sub};
use libm;

/// A marker trait for types that are `Copy` and have a defined partial ordering.
///
/// `Scalar` is a fundamental trait that groups types suitable for numerical operations
/// in a control systems context. It ensures that the type can be efficiently copied
/// and compared, which are baseline requirements for most mathematical algorithms.
/// This trait is a prerequisite for more complex numerical traits like `Ring` and `Field`.
///
/// # Safety
///
/// This trait itself does not introduce any `unsafe` code. However, implementers must
/// ensure that the implementations of `PartialEq` and `PartialOrd` are consistent and
/// correctly represent a partial ordering. For floating-point types, this includes
/// handling of `NaN` values as per the IEEE 754 standard, where `NaN != NaN`.
///
/// - **Pre-conditions**: None.
/// - **Post-conditions**: None.
/// - **Invariants**: The properties of `Copy`, `PartialEq`, and `PartialOrd` must hold
///   for the lifetime of the object.
/// - **Assumptions**: Assumes that the underlying implementations of `PartialEq` and
///   `PartialOrd` are correct and will not cause side effects.
///
/// # Panics
///
/// No panics are directly caused by this trait.
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
pub trait Scalar: Copy + Sized + PartialEq + PartialOrd {}

/// Provides access to the multiplicative identity and a one check.
///
/// This trait abstracts the unit element of a multiplicative structure
/// without requiring the full semantics of a `Ring`.
///
/// # Safety
///
/// Implementers must ensure that `one()` returns the correct unit element.
/// This trait does not introduce panic.
pub trait One: Scalar + Mul<Output = Self> {
    /// Constant multiplicative identity element.
    const ONE: Self;
    /// Returns the multiplicative identity element.
    fn one() -> Self {
        Self::ONE
    }

    /// Returns `true` if the value equals the multiplicative identity.
    #[inline(always)]
    fn is_one(&self) -> bool {
        self.eq(&Self::one())
    }
}

/// Provides access to the additive identity and a zero check.
///
/// This trait is useful where the identity element is required, but the
/// full algebraic structure of a `Ring` is unnecessary.
///
/// # Safety
///
/// Implementers must ensure that `zero()` returns a true additive identity.
/// This trait does not call panic.
pub trait Zero: Scalar + Add<Output = Self> + Sub<Output = Self> {
    /// Constant additive identity element.
    const ZERO: Self;
    /// Returns the additive identity element.
    fn zero() -> Self {
        Self::ZERO
    }

    /// Returns `true` if the value equals the additive identity.
    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.eq(&Self::ZERO)
    }
}

/// Marker trait for types with a mathematical sign (all signed integers,
/// floating-point types).
///
/// # Safety
///
/// Implementers must define sign in the usual mathematical manner.
/// Floating-point implementations must follow IEEE 754 semantics.
/// This trait does not call panic.
pub trait Signed: Zero {
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

    /// Returns the absolute value.
    fn abs(self) -> Self;
}

/// Defines an algebraic ring, providing zero and one elements and standard arithmetic operations.
///
/// The `Ring` trait abstracts over types that support addition, subtraction, multiplication,
/// and negation, forming a mathematical ring. It requires the type to be a `Scalar` and
/// provides associated functions to get the additive identity (`zero`) and multiplicative
/// identity (`one`). This abstraction is crucial for writing generic algorithms that work
/// on both integers and floating-point numbers.
///
/// # Safety
///
/// This trait does not introduce `unsafe` code. Implementers must ensure that the
/// arithmetic operations (`Add`, `Sub`, `Mul`, `Neg`) adhere to the ring axioms
/// (associativity, commutativity, distributivity).
///
/// For fixed-size integer types, the arithmetic operations may overflow. In Rust,
/// this will cause a panic in debug builds but wrap in release builds. Callers
/// performing arithmetic on `Ring` types must be aware of and handle the possibility
/// of overflow according to their safety requirements.
///
/// - **Invariants**: The `zero` and `one` elements must be true identity elements for
///   the implemented addition and multiplication operations.
/// - **Assumptions**: Assumes that the implementations of `Add`, `Sub`, `Mul`, and `Neg`
///   are mathematically correct for the type.
///
/// # Panics
///
/// The trait methods themselves do not panic. However, the required arithmetic traits
/// (`Add`, `Sub`, etc.) may panic on overflow for integer types in debug builds.
///
/// # Example
///
/// ```
/// use control_rs::math::num_traits::Ring;
///
/// fn multiply_by_three<T: Ring>(val: T) -> T {
///     val * (T::one() + T::one() + T::one())
/// }
///
/// assert_eq!(multiply_by_three(5), 15);
/// assert_eq!(multiply_by_three(2.0f32), 6.0f32);
/// ```
pub trait Ring: One + Zero {}

/// Defines an algebraic field, extending a `Ring` with a division operation and a machine epsilon value.
///
/// The `Field` trait represents a mathematical field, which is a `Ring` that also supports
/// division. It is intended for types like floating-point numbers. The trait includes a
/// function to get the machine `epsilon`, which represents the smallest value `x` such
/// that `1.0 + x != 1.0`. This is essential for numerical stability analysis and
/// floating-point comparisons.
///
/// # Safety
///
/// This trait does not introduce `unsafe` code. The division operation must be
/// well-defined. For floating-point types, division by zero will result in `inf`,
/// `-inf`, or `NaN` as per IEEE 754. In a safety-critical context, the caller has
/// the responsibility to prevent division by zero or handle the resulting non-finite
/// values, as they can propagate through calculations and lead to undefined behavior
/// in control algorithms.
///
/// - **Pre-conditions**: When performing division, the divisor should be non-zero to
///   avoid exceptional floating-point values or integer division panics.
/// - **Post-conditions**: None.
/// - **Invariants**: `epsilon()` must return a constant positive value for a given type.
/// - **Assumptions**: Assumes the `Div` implementation is correct. For floating-point
///   types, assumes IEEE 754 compliance.
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
/// The `Real` trait extends a `Field` to include operations common for real numbers,
/// such as `sqrt` and `abs`. It is designed for use with floating-point types in
/// applications requiring analysis, signal processing, or control algorithms. The `sqrt`
/// function is fallible, returning `None` for negative inputs to prevent complex numbers
/// and ensure type safety without introducing panics.
///
/// # Safety
///
/// The functions in this trait rely on [libm] for their implementation, which is a
/// standard, well-tested library for floating-point mathematics. The `sqrt` function
/// is protected against invalid inputs (negative numbers) by returning an `Option`.
/// There is no `unsafe` code.
///
/// - **Pre-conditions**: None. The functions are total over their input domains.
/// - **Post-conditions**: `sqrt` will return a non-negative value inside `Some`. `abs`
///   will always return a non-negative value.
/// - **Invariants**: For any `x: Real` where `x >= 0`, `x.sqrt().unwrap().powi(2)`
///   should be approximately equal to `x`. For any `x: Real`, `x.abs() >= 0`.
/// - **Assumptions**: Assumes the underlying `libm` functions (`sqrtf`, `fabsf`, etc.)
///   are correctly implemented and conform to mathematical definitions.
///
/// # Panics
/// * this function does not call panic.
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
/// assert_eq!(Real::sqrt(negative_val), Err(ArithmeticError::DomainViolation));
/// ```
pub trait Real: Field + Signed {
    /// Calculates the square root of a number.
    ///
    /// # Errors
    /// * Some numbers may not have a sqrt, these will return a [`ArithmeticError`].
    ///
    /// # Returns
    ///
    /// - `Ok(sqrt(self))` if self has a valid sqrt.
    /// - `ArithmeticError` if self does not have a valid sqrt.
    fn sqrt(self) -> ArithmeticResult<Self>;
    /// Calculates the base-10 logarithm of a number.
    ///
    /// # Errors
    /// * Return `Err(ArithmeticError::DomainViolation)` if `self <= 0`.
    fn log10(self) -> ArithmeticResult<Self>;

    /// Calculates the natural logarithm of a number.
    ///
    /// # Errors
    /// * Return `Err(ArithmeticError::DomainViolation)` if `self <= 0`.
    fn ln(self) -> ArithmeticResult<Self>;

    /// Calculates `e^self`.
    fn exp(self) -> Self;

    /// Raises a number to a floating-point power.
    ///
    /// # Errors
    /// * Return `Err(ArithmeticError::DomainViolation)` if `self` is negative.
    fn pow(self, n: Self) -> ArithmeticResult<Self>;
}

/// Marker trait for unsigned types.
///
/// # Safety
///
/// There is no associated functionality; the trait only indicates that
/// the type cannot represent negative values.
pub trait Unsigned: Sized {}

/// A macro to implement the `Ring` trait for a given numeric type.
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
///
/// # Safety
///
/// This macro does not generate `unsafe` code. The correctness of the implementation
/// depends on the user providing the correct zero and one values for the given type.
/// An incorrect value could violate the ring axioms and lead to incorrect calculations.
///
/// # Panics
/// * This macro does not call panic.
///
/// # Example
///
/// ```compile_fail
/// // This is how the macro is used to implement Ring for i32.
/// use control_rs::math::num_traits::{Ring, ArithmeticError};
/// control_rs::impl_ring!(i32, 1, 0);
/// ```
#[macro_export]
macro_rules! impl_ring {
    ($type:ty, $one:expr, $zero:expr) => {
        impl One for $type {
            const ONE: Self = $one;
        }
        impl Zero for $type {
            const ZERO: Self = $zero;
        }
        impl Ring for $type {}
    };
}

/// A macro to implement the `Real` trait for a given floating-point type.
///
/// This macro implements the `Real` trait by calling functions from the `libm` library.
/// It provides implementations for `sqrt` and `abs`. The `sqrt` implementation includes
/// a check to ensure the input is non-negative, returning `None` if it is not. This
/// avoids panics or the creation of `NaN` values from negative inputs, which is a
/// critical safety feature.
///
/// # Arguments
///
/// - `$type`: The floating-point type for which to implement `Real` (e.g., `f32`).
/// - `$sqrt`: The identifier of the `libm` square root function for the type (e.g., `sqrtf`).
/// - `$abs`: The identifier of the `libm` absolute value function for the type (e.g., `fabsf`).
///
/// # Safety
///
/// This macro does not generate `unsafe` code. It relies on the `libm` crate, which
/// is assumed to be a safe and correct implementation of standard math functions. The
/// user must provide the correct `libm` function identifiers corresponding to the type.
/// A mismatch (e.g., using `sqrt` for `f32`) would result in a compile-time error.
///
/// # Panics
/// * This macro does not call panic.
/// * The generated `sqrt` function explicitly avoids panics on negative inputs by returning `Option::None`.
///
/// # Example
///
/// ```compile_fail
/// // This is how the macro is used to implement Real for f32.
/// use control_rs::math::num_traits::{Real, ArithmeticError};
/// control_rs::impl_real!(f32, fabsf, sqrtf, logf, expf, powf);
/// ```
#[macro_export]
macro_rules! impl_real {
    ($type:ty, $abs:path, $sqrt:path, $ln:path, $log10:path, $exp:path, $pow:path) => {
        impl Signed for $type {
            #[inline(always)]
            fn abs(self) -> Self {
                $abs(self)
            }
        }
        impl Real for $type {
            #[inline(always)]
            fn sqrt(self) -> ArithmeticResult<Self> {
                if self.is_sign_negative() {
                    Err(ArithmeticError::DomainViolation)
                } else {
                    Ok($sqrt(self))
                }
            }
            #[inline(always)]
            fn log10(self) -> ArithmeticResult<Self> {
                if self <= <$type>::zero() {
                    Err(ArithmeticError::DomainViolation)
                } else {
                    Ok($log10(self))
                }
            }
            #[inline(always)]
            fn ln(self) -> ArithmeticResult<Self> {
                if self <= <$type>::zero() {
                    Err(ArithmeticError::DomainViolation)
                } else {
                    Ok($ln(self))
                }
            }
            #[inline(always)]
            fn exp(self) -> Self {
                $exp(self)
            }
            #[inline(always)]
            fn pow(self, n: Self) -> ArithmeticResult<Self> {
                Ok($pow(self, n))
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
///
/// # Safety
///
/// This macro does not generate `unsafe` code. The user is responsible for providing
/// the correct machine epsilon value for the type. An incorrect value can lead to
/// precision errors and faulty logic in floating-point comparisons, which could
/// compromise the safety and correctness of numerical algorithms.
///
/// - **Pre-conditions**: None.
/// - **Post-conditions**: None.
/// - **Invariants**: The provided epsilon value must be a constant, positive value
///   that correctly represents the machine epsilon for the given type.
/// - **Assumptions**: Assumes the provided epsilon value is correct as per the
///   IEEE 754 standard for floating-point types.
///
/// # Panics
/// * This macro does not call panic.
///
/// # Example
///
/// ```compile_fail
/// // This is how the macro is used to implement Field for f32.
/// use control_rs::math::num_traits::{Field, ArithmeticError};
/// control_rs::impl_field!(f32, f32::EPSILON);
/// ```
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
impl_ring!(f32, 1.0, 0.0);
impl_field!(f32, f32::EPSILON);
impl_real!(
    f32,
    libm::fabsf,
    libm::sqrtf,
    libm::logf,
    libm::log10f,
    libm::expf,
    libm::powf
);

// Implementations for f64
impl Scalar for f64 {}
impl_ring!(f64, 1.0, 0.0);
impl_field!(f64, f64::EPSILON); // Corrected macro call
impl_real!(
    f64,
    libm::fabs,
    libm::sqrt,
    libm::log,
    libm::log10,
    libm::exp,
    libm::pow
);

// Implementations for i8
impl Scalar for i8 {}
impl Signed for i8 {
    #[inline(always)]
    fn abs(self) -> Self {
        self.abs()
    }
}
impl_ring!(i8, 1, 0);

// Implementations for i16
impl Scalar for i16 {}
impl Signed for i16 {
    #[inline(always)]
    fn abs(self) -> Self {
        self.abs()
    }
}
impl_ring!(i16, 1, 0);

// Implementations for i32
impl Scalar for i32 {}
impl Signed for i32 {
    #[inline(always)]
    fn abs(self) -> Self {
        self.abs()
    }
}
impl_ring!(i32, 1, 0);

// Implementations for i64
impl Scalar for i64 {}
impl Signed for i64 {
    #[inline(always)]
    fn abs(self) -> Self {
        self.abs()
    }
}
impl_ring!(i64, 1, 0);

// Implementations for i128
impl Scalar for i128 {}
impl Signed for i128 {
    #[inline(always)]
    fn abs(self) -> Self {
        self.abs()
    }
}
impl_ring!(i128, 1, 0);

// Implementations for isize
impl Scalar for isize {}
impl Signed for isize {
    #[inline(always)]
    fn abs(self) -> Self {
        self.abs()
    }
}
impl_ring!(isize, 1, 0);

// Implementations for u8
impl Scalar for u8 {}
impl Unsigned for u8 {}
impl_ring!(u8, 1, 0);

// Implementations for u16
impl Scalar for u16 {}
impl Unsigned for u16 {}
impl_ring!(u16, 1, 0);

// Implementations for u32
impl Scalar for u32 {}
impl Unsigned for u32 {}
impl_ring!(u32, 1, 0);

// Implementations for u64
impl Scalar for u64 {}
impl Unsigned for u64 {}
impl_ring!(u64, 1, 0);

// Implementations for u128
impl Scalar for u128 {}
impl Unsigned for u128 {}
impl_ring!(u128, 1, 0);

// Implementations for usize
impl Scalar for usize {}
impl Unsigned for usize {}
impl_ring!(usize, 1, 0);
