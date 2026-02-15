//! # Operations
//!
//! This module defines traits for `checked` and `unchecked` arithmetic operations.
//!
//! The primary purpose of this module is to abstract the underlying arithmetic
//! implementations. By using these traits, algorithms can be written generically,
//! allowing the specific numerical backend (e.g., standard floating-point,
//! fixed-point, or hardware-accelerated DSP libraries) to be selected at compile
//! time through Cargo features.
//!
//! ## Design
//!
//! - **Unchecked Operations**: Traits for operations that do not perform overflow
//!   or other runtime checks. These are suitable for contexts where performance is
//!   critical and the inputs are known to be within valid ranges.
//!
//! - **Checked Operations**: Traits for operations that return a `Result`
//!   to indicate whether the computation succeeded. This is essential for
//!   _safety-critical_ applications where numerical stability and correctness must
//!   be guaranteed.
//!
//! By building on top of the traits in this module, the crate can support a wide
//! range of numeric types and implementations while maintaining a clean and
//! consistent API.

// Re-export core arithmetic traits for unchecked operations.
// These are the standard traits like `Add`, `Sub`, `Mul`, `Div`.
pub use core::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

/// An error type for checked arithmetic operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArithmeticError {
    /// A division by zero was attempted.
    DivisionByZero,
    /// An invalid argument was provided to a function.
    ///
    /// For example, taking the square root of a negative number.
    InvalidArgument,
    /// Overflow occurred during the operation.
    Overflow,
    /// Underflow occurred during the operation.
    Underflow,
}

pub type ArithmeticResult<T> = Result<T, ArithmeticError>;

/// Trait for checked addition.
pub trait CheckedAdd<RHS = Self>: Sized + Add<Output = Self> {
    /// Performs checked addition.
    fn checked_add(self, rhs: RHS) -> ArithmeticResult<Self::Output>;
}

/// Trait for checked subtraction.
pub trait CheckedSub<RHS = Self>: Sized + Sub<Output = Self> {
    /// Performs checked subtraction.
    fn checked_sub(self, rhs: RHS) -> ArithmeticResult<Self::Output>;
}

/// Trait for checked multiplication.
pub trait CheckedMul<RHS = Self>: Sized + Mul<Output = Self> {
    /// Performs checked multiplication.
    fn checked_mul(self, rhs: RHS) -> Result<Self::Output, ArithmeticError>;
}

/// Trait for checked division.
pub trait CheckedDiv<RHS = Self>: Sized + Div<Output = Self> {
    /// Performs checked division.
    ///
    /// Returns `Err(ArithmeticError::DivisionByZero)` if `rhs` is zero.
    fn checked_div(self, rhs: RHS) -> Result<Self::Output, ArithmeticError>;
}

/// Trait for checked remainder.
pub trait CheckedRem<RHS = Self>: Sized + Rem<Output = Self> {
    /// Performs checked remainder.
    ///
    /// Returns `Err(ArithmeticError::DivisionByZero)` if `rhs` is zero.
    fn checked_rem(self, rhs: RHS) -> Result<Self::Output, ArithmeticError>;
}

/// Trait for checked negation.
pub trait CheckedNeg: Sized + Neg<Output = Self> {
    /// Performs checked negation.
    ///
    /// Returns `Err(ArithmeticError::Overflow)` if the negation would overflow.
    fn checked_neg(self) -> Result<Self::Output, ArithmeticError>;
}

/// Trait for checked absolute value.
pub trait CheckedAbs: Sized {
    /// The output of the absolute value operation.
    type Output;

    /// Performs checked absolute value.
    ///
    /// Returns `Err(ArithmeticError::Overflow)` if the absolute value would overflow.
    fn checked_abs(self) -> Result<Self::Output, ArithmeticError>;
}

/// Trait for checked left shift.
pub trait CheckedShl<RHS> {
    /// The resulting type after performing the shift left operation.
    type Output;

    /// Performs checked shift left.
    ///
    /// Returns `Err(ArithmeticError::Overflow)` if the shift amount is too large.
    fn checked_shl(self, rhs: RHS) -> Result<Self::Output, ArithmeticError>;
}

/// Trait for checked right shift.
pub trait CheckedShr<RHS> {
    /// The resulting type after performing the shift right operation.
    type Output;

    /// Performs checked shift right.
    ///
    /// Returns `Err(ArithmeticError::Overflow)` if the shift amount is too large.
    fn checked_shr(self, rhs: RHS) -> Result<Self::Output, ArithmeticError>;
}

/// Trait for checked exponentiation.
pub trait CheckedPow<RHS> {
    /// The resulting type after performing the power operation.
    type Output;

    /// Performs checked power.
    ///
    /// Returns `Err(ArithmeticError::Overflow)` if the power operation would overflow.
    fn checked_pow(self, rhs: RHS) -> Result<Self::Output, ArithmeticError>;
}

/// Trait for checked square root.
pub trait CheckedSqrt {
    /// The resulting type after performing the square root operation.
    type Output;

    /// Performs checked square root.
    ///
    /// Returns `Err(ArithmeticError::InvalidArgument)` for negative inputs.
    fn checked_sqrt(self) -> Result<Self::Output, ArithmeticError>;
}

/// Trait for checked cube root.
pub trait CheckedCbrt {
    /// The resulting type after performing the cube root operation.
    type Output;

    /// Performs checked cube root.
    fn checked_cbrt(self) -> Result<Self::Output, ArithmeticError>;
}

/// Trait for checked hypotenuse.
pub trait CheckedHypot<RHS = Self> {
    /// The resulting type after performing the hypotenuse operation.
    type Output;

    /// Performs checked hypotenuse.
    ///
    /// Returns `Err(ArithmeticError::Overflow)` if the hypotenuse operation would overflow.
    fn checked_hypot(self, rhs: RHS) -> Result<Self::Output, ArithmeticError>;
}

/// Trait for checked logarithm.
pub trait CheckedLog<RHS> {
    /// The resulting type after performing the logarithm operation.
    type Output;

    /// Performs checked logarithm.
    ///
    /// Returns `Err(ArithmeticError::InvalidArgument)` if the base is invalid.
    fn checked_log(self, base: RHS) -> Result<Self::Output, ArithmeticError>;
}

/// Trait for checked base-2 logarithm.
pub trait CheckedLog2 {
    /// The resulting type after performing the base-2 logarithm operation.
    type Output;

    /// Performs checked base-2 logarithm.
    ///
    /// Returns `Err(ArithmeticError::InvalidArgument)` if the input is invalid (e.g., non-positive).
    fn checked_log2(self) -> Result<Self::Output, ArithmeticError>;
}

/// Trait for checked base-10 logarithm.
pub trait CheckedLog10 {
    /// The resulting type after performing the base-10 logarithm operation.
    type Output;

    /// Performs checked base-10 logarithm.
    ///
    /// Returns `Err(ArithmeticError::InvalidArgument)` if the input is invalid (e.g., non-positive).
    fn checked_log10(self) -> Result<Self::Output, ArithmeticError>;
}

/// Trait for checked natural logarithm.
pub trait CheckedLn {
    /// The resulting type after performing the natural logarithm operation.
    type Output;

    /// Performs checked natural logarithm.
    ///
    /// Returns `Err(ArithmeticError::InvalidArgument)` if the input is invalid (e.g., non-positive).
    fn checked_ln(self) -> Result<Self::Output, ArithmeticError>;
}

/// Trait for checked exponential.
pub trait CheckedExp {
    /// The resulting type after performing the exponential operation.
    type Output;

    /// Performs checked exponential.
    ///
    /// Returns `Err(ArithmeticError::Overflow)` if the exponential operation would overflow.
    fn checked_exp(self) -> Result<Self::Output, ArithmeticError>;
}
