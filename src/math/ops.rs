//! # Operations
//!
//! This module defines traits for fallible (`try_`) and infallible (`unchecked`)
//! arithmetic operations.
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
//!   or other runtime checks. These are the standard `core::ops` traits.
//!
//! - **Fallible Operations**: Traits for operations that return a `Result`
//!   to indicate whether the computation succeeded. This is essential for
//!   _safety-critical_ applications where numerical stability and correctness must
//!   be guaranteed. The `try_` prefix is used to indicate that these operations
//!   can fail.
//!
//! By building on top of the traits in this module, the crate can support a wide
//! range of numeric types and implementations while maintaining a clean and
//! consistent API.
#![allow(clippy::arbitrary_source_item_ordering)]

// Re-export core arithmetic traits for unchecked operations.
use crate::math::{ArithmeticError, ArithmeticResult, num_traits::Zero};
pub use core::ops::{Add, Div, Mul, Neg, Rem, Shl, Shr, Sub};

// --- Fallible Traits ---

/// Performs fallible addition.
pub trait TryAdd: Sized + Add<Self, Output = Self> {
    /// Adds two numbers, returning an error on failure.
    ///
    /// # Arguments
    /// * `v` - The value to add.
    ///
    /// # Returns
    /// * `ArithmeticResult<Self>` - The result of the addition.
    ///
    /// # Errors
    ///
    /// - `ArithmeticError::Overflow`: The result exceeded the maximum
    ///   representable range of the type.
    /// - `ArithmeticError::DomainViolation`: The operation is invalid for the
    ///   given inputs (e.g., adding to a `NaN` floating-point number).
    fn try_add(&self, v: &Self) -> ArithmeticResult<Self>;
}

/// Performs fallible multiplication.
pub trait TryMul: Sized + Mul<Self, Output = Self> {
    /// Multiplies two numbers, returning an error on failure.
    ///
    /// # Arguments
    /// * `v` - The value to multiply by.
    ///
    /// # Returns
    /// * `ArithmeticResult<Self>` - The result of the multiplication.
    ///
    /// # Errors
    ///
    /// - `ArithmeticError::Overflow`: The result exceeded the maximum
    ///   representable range of the type.
    /// - `ArithmeticError::DomainViolation`: The operation is invalid for the
    ///   given inputs (e.g., multiplying by a `NaN` floating-point number).
    fn try_mul(&self, v: &Self) -> ArithmeticResult<Self>;
}

/// Performs fallible subtraction.
pub trait TrySub: Sized + Sub<Self, Output = Self> {
    /// Subtracts two numbers, returning an error on failure.
    ///
    /// # Arguments
    /// * `v` - The value to subtract.
    ///
    /// # Returns
    /// * `ArithmeticResult<Self>` - The result of the subtraction.
    ///
    /// # Errors
    ///
    /// - `ArithmeticError::Overflow`: The result exceeded the maximum
    ///   representable range of the type.
    /// - `ArithmeticError::DomainViolation`: The operation is invalid for the
    ///   given inputs (e.g., subtracting a `NaN` floating-point number).
    fn try_sub(&self, v: &Self) -> ArithmeticResult<Self>;
}

/// Performs fallible division.
pub trait TryDiv: Sized + Div<Self, Output = Self> {
    /// Divides two numbers, returning an error on failure.
    ///
    /// # Arguments
    /// * `v` - The divisor.
    ///
    /// # Returns
    /// * `ArithmeticResult<Self>` - The result of the division.
    ///
    /// # Errors
    ///
    /// - `ArithmeticError::DivisionByZero`: The divisor `v` is zero.
    /// - `ArithmeticError::Overflow`: The result exceeded the maximum
    ///   representable range (e.g., `i32::MIN / -1`).
    /// - `ArithmeticError::DomainViolation`: The operation is invalid for the
    ///   given inputs (e.g., dividing by a `NaN` floating-point number).
    fn try_div(&self, v: &Self) -> ArithmeticResult<Self>;
}

/// Performs fallible negation.
pub trait TryNeg: Sized + Neg<Output = Self> {
    /// Negates a number, returning an error on failure.
    ///
    /// # Returns
    /// * `ArithmeticResult<Self>` - The negated value.
    ///
    /// # Errors
    ///
    /// - `ArithmeticError::Overflow`: The result cannot be represented (e.g., `-i32::MIN`).
    fn try_neg(&self) -> ArithmeticResult<Self>;
}

/// Performs fallible remainder.
pub trait TryRem: Sized + Rem<Self, Output = Self> {
    /// Finds the remainder of a division, returning an error on failure.
    ///
    /// # Arguments
    /// * `v` - The divisor.
    ///
    /// # Returns
    /// * `ArithmeticResult<Self>` - The remainder.
    ///
    /// # Errors
    ///
    /// - `ArithmeticError::DivisionByZero`: The divisor `v` is zero.
    /// - `ArithmeticError::Overflow`: The operation overflows (e.g., `i32::MIN % -1`).
    /// - `ArithmeticError::DomainViolation`: The operation is invalid for the
    ///   given inputs (e.g., involving a `NaN` floating-point number).
    fn try_rem(&self, v: &Self) -> ArithmeticResult<Self>;
}

/// Performs a fallible left shift.
pub trait TryShl: Sized + Shl<u32, Output = Self> {
    /// Performs a fallible left shift (`self << rhs`).
    ///
    /// # Arguments
    /// * `rhs` - The number of bits to shift.
    ///
    /// # Returns
    /// * `ArithmeticResult<Self>` - The shifted value.
    ///
    /// # Errors
    ///
    /// - `ArithmeticError::Overflow`: The number of bits to shift (`rhs`) is
    ///   greater than or equal to the bit-width of the type, or the result
    ///   of the shift overflows the type.
    fn try_shl(&self, rhs: u32) -> ArithmeticResult<Self>;
}

/// Performs a fallible right shift.
pub trait TryShr: Sized + Shr<u32, Output = Self> {
    /// Performs a fallible right shift (`self >> rhs`).
    ///
    /// # Arguments
    /// * `rhs` - The number of bits to shift.
    ///
    /// # Returns
    /// * `ArithmeticResult<Self>` - The shifted value.
    ///
    /// # Errors
    ///
    /// - `ArithmeticError::Overflow`: The number of bits to shift (`rhs`) is
    ///   greater than or equal to the bit-width of the type.
    fn try_shr(&self, rhs: u32) -> ArithmeticResult<Self>;
}

// --- Integer Implementations ---

macro_rules! try_impl_int {
    ($trait:ident, $try_method:ident, $checked_method:ident, $($t:ty),+) => {
        $(
            impl $trait for $t {
                #[inline]
                fn $try_method(&self, v: &$t) -> ArithmeticResult<$t> {
                    self.$checked_method(*v).ok_or(ArithmeticError::Overflow)
                }
            }
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
        )+
    };
}

try_neg_impl!(i8, i16, i32, i64, i128, isize);

macro_rules! try_shift_impl {
    ($trait:ident, $try_method:ident, $checked_method:ident, $($t:ty),+) => {
        $(
            impl $trait for $t {
                #[inline]
                fn $try_method(&self, rhs: u32) -> ArithmeticResult<$t> {
                    self.$checked_method(rhs).ok_or(ArithmeticError::Overflow)
                }
            }
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

// --- Floating Point Implementations ---

macro_rules! try_float_impl {
    ($($t:ty),+) => {
        $(
            impl TryAdd for $t {
                fn try_add(&self, v: &$t) -> ArithmeticResult<$t> {
                    if self.is_nan() || v.is_nan() {
                        return Err(ArithmeticError::DomainViolation);
                    }
                    let result = self.add(v);
                    if result.is_infinite() {
                        return Err(ArithmeticError::Overflow);
                    }
                    return Ok(result);
                }
            }

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

            impl TryNeg for $t {
                fn try_neg(&self) -> ArithmeticResult<$t> {
                    return Ok(self.neg());
                }
            }

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

            impl TrySub for $t {
                fn try_sub(&self, v: &$t) -> ArithmeticResult<$t> {
                    if self.is_nan() || v.is_nan() {
                        return Err(ArithmeticError::DomainViolation);
                    }
                    let result = self.sub(v);
                    if result.is_infinite() {
                        return Err(ArithmeticError::Overflow);
                    }
                    return Ok(result);
                }
            }
        )+
    };
}

try_float_impl!(f32, f64);
