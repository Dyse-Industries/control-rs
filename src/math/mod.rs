//! # Math
//!
//! Core numerical primitives optimized for bare-metal execution. This module prioritizes
//! fixed-point safety and stack-allocated matrix operations to avoid heap dependency.
//!
//! # References
//! - [Numerical Recipes - The Art of Scientific Computing](https://numerical.recipes/)

pub mod num_traits;
pub mod num_types;
pub mod ops;
mod static_storage;
pub mod subprograms;
#[cfg(test)]
mod tests;

/// A unified error type for arithmetic operations.
///
/// This structure balances high-level control flow (overflow/underflow) with
/// fixed-point specific signals (saturation/precision).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArithmeticError {
    /// The mathematical operation is undefined for the given inputs
    /// (e.g., `sqrt(-1.0)`, `acos(2.0)`).
    DomainViolation,

    /// Attempted to divide by zero.
    DivisionByZero,

    /// The result exceeded the maximum representable range of the type.
    /// In fixed-point arithmetic, this implies a wrapping or undefined result.
    Overflow,

    /// The result is smaller than the smallest representable positive value
    /// (Subnormal/Denormal).
    Underflow,

    /// The value exceeded the range but was clamped to the maximum/minimum
    /// representable value (specific to fixed-point/DSP logic).
    Saturation,

    /// The result could not be represented exactly, resulting in quantization
    /// or rounding errors (e.g., casting `f64` to `u32` where the float has a decimal).
    PrecisionLoss,
}

/// A specialized `Result` type for fallible arithmetic operations.
pub type ArithmeticResult<T> = Result<T, ArithmeticError>;

impl core::fmt::Display for ArithmeticError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::DomainViolation => write!(f, "Input is outside the mathematical domain"),
            Self::DivisionByZero => write!(f, "Division by zero"),
            Self::Overflow => write!(f, "Value overflowed representable range"),
            Self::Underflow => write!(f, "Value underflowed (subnormal)"),
            Self::Saturation => write!(f, "Value saturated (clamped) at bounds"),
            Self::PrecisionLoss => write!(f, "Significant precision was lost during operation"),
        }
    }
}

#[cfg(test)]
mod test {
    use core::fmt::{self, Write};
    use crate::math::ArithmeticError;

    /// A simple helper to capture format output into a stack buffer
    struct TestWriter<'a> {
        buf: &'a mut [u8],
        len: usize,
    }

    impl<'a> TestWriter<'a> {
        fn new(buf: &'a mut [u8]) -> Self {
            Self { buf, len: 0 }
        }

        fn as_str(&self) -> &str {
            core::str::from_utf8(&self.buf[..self.len]).expect("Invalid UTF-8")
        }
    }

    impl<'a> Write for TestWriter<'a> {
        fn write_str(&mut self, s: &str) -> fmt::Result {
            let bytes = s.as_bytes();
            let remaining = self.buf.len() - self.len;

            if bytes.len() > remaining {
                return Err(fmt::Error); // Buffer overflow
            }

            self.buf[self.len..self.len + bytes.len()].copy_from_slice(bytes);
            self.len += bytes.len();
            Ok(())
        }
    }

    #[test]
    fn test_error_display_no_std() {
        {
            let mut buffer = [0u8; 128]; // Stack allocated buffer
            let mut writer = TestWriter::new(&mut buffer);
            // 1. Write the error into the buffer
            let err = ArithmeticError::Overflow;
            write!(writer, "{}", err).expect("Buffer too small");

            // 2. Assert against the string slice
            assert_eq!(writer.as_str(), "Value overflowed representable range");
        }

        {
            let mut buffer = [0u8; 128]; // Stack allocated buffer
            let mut writer = TestWriter::new(&mut buffer);
            // 1. Write the error into the buffer
            let err = ArithmeticError::DomainViolation;
            write!(writer, "{}", err).expect("Buffer too small");

            // 2. Assert against the string slice
            assert_eq!(writer.as_str(), "Input is outside the mathematical domain");
        }
        {
            let mut buffer = [0u8; 128]; // Stack allocated buffer
            let mut writer = TestWriter::new(&mut buffer);
            // 1. Write the error into the buffer
            let err = ArithmeticError::Saturation;
            write!(writer, "{}", err).expect("Buffer too small");

            // 2. Assert against the string slice
            assert_eq!(writer.as_str(), "Value saturated (clamped) at bounds");
        }
        {
            let mut buffer = [0u8; 128]; // Stack allocated buffer
            let mut writer = TestWriter::new(&mut buffer);
            // 1. Write the error into the buffer
            let err = ArithmeticError::PrecisionLoss;
            write!(writer, "{}", err).expect("Buffer too small");

            // 2. Assert against the string slice
            assert_eq!(writer.as_str(), "Significant precision was lost during operation");
        }
        {
            let mut buffer = [0u8; 128]; // Stack allocated buffer
            let mut writer = TestWriter::new(&mut buffer);
            // 1. Write the error into the buffer
            let err = ArithmeticError::PrecisionLoss;
            write!(writer, "{}", err).expect("Buffer too small");

            // 2. Assert against the string slice
            assert_eq!(writer.as_str(), "Significant precision was lost during operation");
        }
        {
            let mut buffer = [0u8; 128]; // Stack allocated buffer
            let mut writer = TestWriter::new(&mut buffer);
            // 1. Write the error into the buffer
            let err = ArithmeticError::DivisionByZero;
            write!(writer, "{}", err).expect("Buffer too small");

            // 2. Assert against the string slice
            assert_eq!(writer.as_str(), "Division by zero");
        }
        {
            let mut buffer = [0u8; 128]; // Stack allocated buffer
            let mut writer = TestWriter::new(&mut buffer);
            // 1. Write the error into the buffer
            let err = ArithmeticError::Overflow;
            write!(writer, "{}", err).expect("Buffer too small");

            // 2. Assert against the string slice
            assert_eq!(writer.as_str(), "Value overflowed representable range");
        }
        {
            let mut buffer = [0u8; 128]; // Stack allocated buffer
            let mut writer = TestWriter::new(&mut buffer);
            // 1. Write the error into the buffer
            let err = ArithmeticError::Underflow;
            write!(writer, "{}", err).expect("Buffer too small");

            // 2. Assert against the string slice
            assert_eq!(writer.as_str(), "Value underflowed (subnormal)");
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ArithmeticError {}