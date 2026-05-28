//! # Operations
//!
//! This module defines traits for (`try_`), wrapping and saturating
//! arithmetic operations.
#![allow(clippy::arbitrary_source_item_ordering)]

// Re-export core arithmetic traits for unchecked operations.
use crate::math::{ArithmeticError, ArithmeticResult, num_traits::Zero};
pub use core::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Shl,
    ShlAssign, Shr, ShrAssign, Sub, SubAssign,
};

/// Performs addition.
pub trait TryAdd<Rhs = Self>: Sized + Add<Rhs> {
    /// Adds two numbers, returning an error on failure.
    ///
    /// # Arguments
    /// * `v` - The value to add.
    ///
    /// # Returns
    /// * `ArithmeticResult<Self::Output>` - The result of the addition.
    ///
    /// # Errors
    /// - `ArithmeticError::Overflow`: The result exceeded the maximum
    ///   representable range of the type.
    /// - `ArithmeticError::DomainViolation`: The operation is invalid for the
    ///   given inputs (e.g., adding to a `NaN` floating-point number).
    #[allow(clippy::type_complexity)]
    fn try_add(&self, v: &Rhs) -> ArithmeticResult<Self::Output>;
}

/// Performs subtraction.
pub trait TrySub<Rhs = Self>: Sized + Sub<Rhs> {
    /// Subtracts two numbers, returning an error on failure.
    ///
    /// # Arguments
    /// * `v` - The value to subtract.
    ///
    /// # Returns
    /// * `ArithmeticResult<Self::Output>` - The result of the subtraction.
    ///
    /// # Errors
    /// - `ArithmeticError::Overflow`: The result exceeded the maximum
    ///   representable range of the type.
    /// - `ArithmeticError::DomainViolation`: The operation is invalid for the
    ///   given inputs (e.g., subtracting a `NaN` floating-point number).
    #[allow(clippy::type_complexity)]
    fn try_sub(&self, v: &Rhs) -> ArithmeticResult<Self::Output>;
}

/// Performs multiplication.
pub trait TryMul<Rhs = Self>: Sized + Mul<Rhs> {
    /// Multiplies two numbers, returning an error on failure.
    ///
    /// # Arguments
    /// * `v` - The value to multiply by.
    ///
    /// # Returns
    /// * `ArithmeticResult<Self::Output>` - The result of the multiplication.
    ///
    /// # Errors
    /// - `ArithmeticError::Overflow`: The result exceeded the maximum
    ///   representable range of the type.
    /// - `ArithmeticError::DomainViolation`: The operation is invalid for the
    ///   given inputs (e.g., multiplying by a `NaN` floating-point number).
    #[allow(clippy::type_complexity)]
    fn try_mul(&self, v: &Rhs) -> ArithmeticResult<Self::Output>;
}

/// Performs fallible division.
pub trait TryDiv<Rhs = Self>: Sized + Div<Rhs> {
    /// Divides two numbers, returning an error on failure.
    ///
    /// # Arguments
    /// * `v` - The divisor.
    ///
    /// # Returns
    /// * `ArithmeticResult<Self::Output>` - The result of the division.
    ///
    /// # Errors
    /// - `ArithmeticError::DivisionByZero`: The divisor `v` is zero.
    /// - `ArithmeticError::Overflow`: The result exceeded the maximum
    ///   representable range (e.g., `i32::MIN / -1`).
    /// - `ArithmeticError::DomainViolation`: The operation is invalid for the
    ///   given inputs (e.g., dividing by a `NaN` floating-point number).
    #[allow(clippy::type_complexity)]
    fn try_div(&self, v: &Rhs) -> ArithmeticResult<Self::Output>;
}

/// Performs negation.
pub trait TryNeg: Sized + Neg {
    /// Negates a number, returning an error on failure.
    ///
    /// # Returns
    /// * `ArithmeticResult<Self::Output>` - The negated value.
    ///
    /// # Errors
    /// - `ArithmeticError::Overflow`: The result cannot be represented (e.g., `-i32::MIN`).
    #[allow(clippy::type_complexity)]
    fn try_neg(&self) -> ArithmeticResult<Self::Output>;
}

/// Performs remainder.
pub trait TryRem<Rhs = Self>: Sized + Rem<Rhs> {
    /// Finds the remainder of a division, returning an error on failure.
    ///
    /// # Arguments
    /// * `v` - The divisor.
    ///
    /// # Returns
    /// * `ArithmeticResult<Self::Output>` - The remainder.
    ///
    /// # Errors
    ///
    /// - `ArithmeticError::DivisionByZero`: The divisor `v` is zero.
    /// - `ArithmeticError::Overflow`: The operation overflows (e.g., `i32::MIN % -1`).
    /// - `ArithmeticError::DomainViolation`: The operation is invalid for the
    ///   given inputs (e.g., involving a `NaN` floating-point number).
    #[allow(clippy::type_complexity)]
    fn try_rem(&self, v: &Rhs) -> ArithmeticResult<Self::Output>;
}

/// Performs a left shift.
pub trait TryShl<Rhs = u32>: Sized + Shl<Rhs> {
    /// Performs a left shift (`self << rhs`).
    ///
    /// # Arguments
    /// * `rhs` - The number of bits to shift.
    ///
    /// # Returns
    /// * `ArithmeticResult<Self::Output>` - The shifted value.
    ///
    /// # Errors
    /// - `ArithmeticError::Overflow`: The number of bits to shift (`rhs`) is
    ///   greater than or equal to the bit-width of the type, or the result
    ///   of the shift overflows the type.
    #[allow(clippy::type_complexity)]
    fn try_shl(&self, rhs: Rhs) -> ArithmeticResult<Self::Output>;
}

/// Performs a right shift.
pub trait TryShr<Rhs = u32>: Sized + Shr<Rhs> {
    /// Performs a right shift (`self >> rhs`).
    ///
    /// # Arguments
    /// * `rhs` - The number of bits to shift.
    ///
    /// # Returns
    /// * `ArithmeticResult<Self::Output>` - The shifted value.
    ///
    /// # Errors
    ///
    /// - `ArithmeticError::Overflow`: The number of bits to shift (`rhs`) is
    ///   greater than or equal to the bit-width of the type.
    #[allow(clippy::type_complexity)]
    fn try_shr(&self, rhs: Rhs) -> ArithmeticResult<Self::Output>;
}

////////////////////////////////////////////////////////////////////////////////
// Wrapping Traits
////////////////////////////////////////////////////////////////////////////////

/// Performs wrapping addition.
pub trait WrappingAdd: Sized + Add<Self, Output = Self> {
    /// Adds two numbers, wrapping on overflow.
    #[must_use]
    fn wrapping_add(&self, v: &Self) -> Self;
}

/// Performs wrapping subtraction.
pub trait WrappingSub: Sized + Sub<Self, Output = Self> {
    /// Subtracts two numbers, wrapping on overflow.
    #[must_use]
    fn wrapping_sub(&self, v: &Self) -> Self;
}

/// Performs wrapping multiplication.
pub trait WrappingMul: Sized + Mul<Self, Output = Self> {
    /// Multiplies two numbers, wrapping on overflow.
    #[must_use]
    fn wrapping_mul(&self, v: &Self) -> Self;
}

////////////////////////////////////////////////////////////////////////////////
// --- Saturating Traits ---
////////////////////////////////////////////////////////////////////////////////

/// Performs saturating addition.
pub trait SaturatingAdd: Sized + Add<Self, Output = Self> {
    /// Adds two numbers, saturating at the numeric bounds.
    #[must_use]
    fn saturating_add(&self, v: &Self) -> Self;
}

/// Performs saturating subtraction.
pub trait SaturatingSub: Sized + Sub<Self, Output = Self> {
    /// Subtracts two numbers, saturating at the numeric bounds.
    #[must_use]
    fn saturating_sub(&self, v: &Self) -> Self;
}

/// Performs saturating multiplication.
pub trait SaturatingMul: Sized + Mul<Self, Output = Self> {
    /// Multiplies two numbers, saturating at the numeric bounds.
    #[must_use]
    fn saturating_mul(&self, v: &Self) -> Self;
}

////////////////////////////////////////////////////////////////////////////////
// --- Integer Implementations ---
////////////////////////////////////////////////////////////////////////////////

macro_rules! try_impl_int {
    ($trait:ident, $try_method:ident, $checked_method:ident, $($t:ty),+) => {
        $(
            impl $trait for $t {
                #[inline]
                fn $try_method(&self, v: &$t) -> ArithmeticResult<$t> {
                    self.$checked_method(*v).ok_or(ArithmeticError::Overflow)
                }
            }
            ////////////////////////////////////////////////////////////////////////////////
        )+
    };
}

try_impl_int!(
    TryAdd,
    try_add,
    checked_add,
    u8,
    u16,
    u32,
    u64,
    u128,
    usize,
    i8,
    i16,
    i32,
    i64,
    i128,
    isize
);
try_impl_int!(
    TrySub,
    try_sub,
    checked_sub,
    u8,
    u16,
    u32,
    u64,
    u128,
    usize,
    i8,
    i16,
    i32,
    i64,
    i128,
    isize
);
try_impl_int!(
    TryMul,
    try_mul,
    checked_mul,
    u8,
    u16,
    u32,
    u64,
    u128,
    usize,
    i8,
    i16,
    i32,
    i64,
    i128,
    isize
);

macro_rules! try_div_rem_impl {
    ($trait:ident, $try_method:ident, $checked_method:ident, $($t:ty),+) => {
        $(
            impl $trait for $t {
                #[inline]
                fn $try_method(&self, v: &$t) -> ArithmeticResult<$t> {
                    if *v == 0 {
                        return Err(ArithmeticError::DivisionByZero);
                    }
                    self.$checked_method(*v).ok_or(ArithmeticError::Overflow)
                }
            }
            ////////////////////////////////////////////////////////////////////////////////
        )+
    };
}

try_div_rem_impl!(
    TryDiv,
    try_div,
    checked_div,
    u8,
    u16,
    u32,
    u64,
    u128,
    usize,
    i8,
    i16,
    i32,
    i64,
    i128,
    isize
);
try_div_rem_impl!(
    TryRem,
    try_rem,
    checked_rem,
    u8,
    u16,
    u32,
    u64,
    u128,
    usize,
    i8,
    i16,
    i32,
    i64,
    i128,
    isize
);

macro_rules! try_neg_impl {
    ($($t:ty),+) => {
        $(
            impl TryNeg for $t {
                #[inline]
                fn try_neg(&self) -> ArithmeticResult<$t> {
                    self.checked_neg().ok_or(ArithmeticError::Overflow)
                }
            }
            ////////////////////////////////////////////////////////////////////////////////
        )+
    };
}

try_neg_impl!(i8, i16, i32, i64, i128, isize);

macro_rules! try_shift_impl {
    ($trait:ident, $try_method:ident, $checked_method:ident, $($t:ty),+) => {
        $(
            impl $trait<u32> for $t {
                #[inline]
                fn $try_method(&self, rhs: u32) -> ArithmeticResult<$t> {
                    self.$checked_method(rhs).ok_or(ArithmeticError::Overflow)
                }
            }
            ////////////////////////////////////////////////////////////////////////////////
        )+
    };
}

try_shift_impl!(
    TryShr,
    try_shr,
    checked_shr,
    u8,
    u16,
    u32,
    u64,
    u128,
    usize,
    i8,
    i16,
    i32,
    i64,
    i128,
    isize
);

try_shift_impl!(
    TryShl,
    try_shl,
    checked_shl,
    u8,
    u16,
    u32,
    u64,
    u128,
    usize,
    i8,
    i16,
    i32,
    i64,
    i128,
    isize
);

macro_rules! wrapping_impl_int {
    ($trait:ident, $method:ident, $wrapping_method:ident, $($t:ty),+) => {
        $(
            impl $trait for $t {
                #[inline]
                fn $method(&self, v: &$t) -> $t {
                    <$t>::$wrapping_method(*self, *v)
                }
            }
            ////////////////////////////////////////////////////////////////////////////////
        )+
    };
}

wrapping_impl_int!(
    WrappingAdd,
    wrapping_add,
    wrapping_add,
    u8,
    u16,
    u32,
    u64,
    u128,
    usize,
    i8,
    i16,
    i32,
    i64,
    i128,
    isize
);
wrapping_impl_int!(
    WrappingSub,
    wrapping_sub,
    wrapping_sub,
    u8,
    u16,
    u32,
    u64,
    u128,
    usize,
    i8,
    i16,
    i32,
    i64,
    i128,
    isize
);
wrapping_impl_int!(
    WrappingMul,
    wrapping_mul,
    wrapping_mul,
    u8,
    u16,
    u32,
    u64,
    u128,
    usize,
    i8,
    i16,
    i32,
    i64,
    i128,
    isize
);

macro_rules! saturating_impl_int {
    ($trait:ident, $method:ident, $saturating_method:ident, $($t:ty),+) => {
        $(
            impl $trait for $t {
                #[inline]
                fn $method(&self, v: &$t) -> $t {
                    <$t>::$saturating_method(*self, *v)
                }
            }
            ////////////////////////////////////////////////////////////////////////////////
        )+
    };
}

saturating_impl_int!(
    SaturatingAdd,
    saturating_add,
    saturating_add,
    u8,
    u16,
    u32,
    u64,
    u128,
    usize,
    i8,
    i16,
    i32,
    i64,
    i128,
    isize
);
saturating_impl_int!(
    SaturatingSub,
    saturating_sub,
    saturating_sub,
    u8,
    u16,
    u32,
    u64,
    u128,
    usize,
    i8,
    i16,
    i32,
    i64,
    i128,
    isize
);
saturating_impl_int!(
    SaturatingMul,
    saturating_mul,
    saturating_mul,
    u8,
    u16,
    u32,
    u64,
    u128,
    usize,
    i8,
    i16,
    i32,
    i64,
    i128,
    isize
);

////////////////////////////////////////////////////////////////////////////////
// --- Floating Point Implementations ---
////////////////////////////////////////////////////////////////////////////////
macro_rules! try_float_impl {
    ($($t:ty),+) => {
        $(
            impl TryAdd for $t {
                fn try_add(&self, v: &$t) -> ArithmeticResult<$t> {
                    if self.is_nan() || v.is_nan() {
                        return Err(ArithmeticError::DomainViolation);
                    }
                    let result = self.add(v);
                    #[allow(clippy::float_cmp)]
                    if result == *self && *v != <$t>::ZERO {
                        return Err(ArithmeticError::PrecisionLoss);
                    }
                    #[allow(clippy::float_cmp)]
                    if result == *v && *self != <$t>::ZERO {
                        return Err(ArithmeticError::PrecisionLoss);
                    }
                    if result != 0.0 && result.abs() < <$t>::MIN_POSITIVE {
                        return Err(ArithmeticError::Underflow);
                    }
                    return Ok(result);
                }
            }

            ////////////////////////////////////////////////////////////////////////////////

            impl TryDiv for $t {
                fn try_div(&self, v: &$t) -> ArithmeticResult<$t> {
                    if v.is_zero() {
                        return Err(ArithmeticError::DivisionByZero);
                    }
                    if self.is_nan() || v.is_nan() {
                        return Err(ArithmeticError::DomainViolation);
                    }

                    let result = self.div(v);
                    if result.is_infinite() {
                        return Err(ArithmeticError::Overflow);
                    }
                    return Ok(result);
                }
            }

            ////////////////////////////////////////////////////////////////////////////////

            impl TryMul for $t {
                fn try_mul(&self, v: &$t) -> ArithmeticResult<$t> {
                    if self.is_nan() || v.is_nan() {
                        return Err(ArithmeticError::DomainViolation);
                    }
                    let result = self.mul(v);
                    if result.is_infinite() {
                        return Err(ArithmeticError::Overflow);
                    }
                    return Ok(result);
                }
            }

            ////////////////////////////////////////////////////////////////////////////////

            impl TryNeg for $t {
                fn try_neg(&self) -> ArithmeticResult<$t> {
                    if self.is_nan() {
                        return Err(ArithmeticError::DomainViolation);
                    }
                    return Ok(self.neg());
                }
            }

            ////////////////////////////////////////////////////////////////////////////////

            impl TryRem for $t {
                fn try_rem(&self, v: &$t) -> ArithmeticResult<$t> {
                    if v.is_zero() {
                        return Err(ArithmeticError::DivisionByZero);
                    }
                    if self.is_nan() || v.is_nan() {
                        return Err(ArithmeticError::DomainViolation);
                    }
                    let result = self.rem(v);

                    return Ok(result);
                }
            }

            ////////////////////////////////////////////////////////////////////////////////

            impl TrySub for $t {
                fn try_sub(&self, v: &$t) -> ArithmeticResult<$t> {
                    if self.is_nan() || v.is_nan() {
                        return Err(ArithmeticError::DomainViolation);
                    }
                    let result = self.sub(v);
                    #[allow(clippy::float_cmp)]
                    if result == *self && *v != <$t>::ZERO {
                        return Err(ArithmeticError::PrecisionLoss);
                    }
                    #[allow(clippy::float_cmp)]
                    if result == v.neg() && *self != <$t>::ZERO {
                        return Err(ArithmeticError::PrecisionLoss);
                    }
                    if result != <$t>::ZERO && result.abs() < <$t>::MIN_POSITIVE {
                        return Err(ArithmeticError::Underflow);
                    }
                    return Ok(result);
                }
            }

            ////////////////////////////////////////////////////////////////////////////////
        )+
    };
}

try_float_impl!(f32, f64);
