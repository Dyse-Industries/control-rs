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

// Re-export core arithmetic traits for unchecked operations.
pub use core::ops::{Add, Div, Mul, Neg, Rem, Shl, Shr, Sub};
use crate::math::{ArithmeticError, ArithmeticResult, num_traits::Zero};

// --- Fallible Traits ---

/// Performs fallible addition.
pub trait TryAdd: Sized + Add<Self, Output = Self> {
    /// Adds two numbers, returning an error on overflow.
    fn try_add(&self, v: &Self) -> ArithmeticResult<Self>;
}

/// Performs fallible subtraction.
pub trait TrySub: Sized + Sub<Self, Output = Self> {
    /// Subtracts two numbers, returning an error on overflow/underflow.
    fn try_sub(&self, v: &Self) -> ArithmeticResult<Self>;
}

/// Performs fallible multiplication.
pub trait TryMul: Sized + Mul<Self, Output = Self> {
    /// Multiplies two numbers, returning an error on overflow/underflow.
    fn try_mul(&self, v: &Self) -> ArithmeticResult<Self>;
}

/// Performs fallible division.
pub trait TryDiv: Sized + Div<Self, Output = Self> {
    /// Divides two numbers, returning an error on division by zero or overflow.
    fn try_div(&self, v: &Self) -> ArithmeticResult<Self>;
}

/// Performs fallible remainder.
pub trait TryRem: Sized + Rem<Self, Output = Self> {
    /// Finds the remainder of dividing two numbers, returning an error on division by zero or overflow.
    fn try_rem(&self, v: &Self) -> ArithmeticResult<Self>;
}

/// Performs fallible negation.
pub trait TryNeg: Sized + Neg<Output = Self> {
    /// Negates a number, returning an error on overflow.
    fn try_neg(&self) -> ArithmeticResult<Self>;
}

/// Performs a fallible left shift.
pub trait TryShl: Sized + Shl<u32, Output = Self> {
    /// Fallible shift left. Computes `self << rhs`.
    fn try_shl(&self, rhs: u32) -> ArithmeticResult<Self>;
}

/// Performs a fallible right shift.
pub trait TryShr: Sized + Shr<u32, Output = Self> {
    /// Fallible shift right. Computes `self >> rhs`.
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

            impl TryNeg for $t {
                fn try_neg(&self) -> ArithmeticResult<$t> {
                    return Ok(self.neg());
                }
            }
        )+
    };
}

try_float_impl!(f32, f64);
