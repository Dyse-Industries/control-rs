//! # Math
//!
//! Core numerical primitives optimized for bare-metal execution.
//!
//! # References
//! - [Numerical Recipes - The Art of Scientific Computing](https://numerical.recipes/)

pub mod assert;

pub mod num_traits;
pub mod num_types;
pub mod ops;
pub mod static_storage;
pub mod subprograms;

#[cfg(test)]
mod tests;

/// A unified error type for arithmetic operations.
///
/// This structure balances high-level control flow (overflow/underflow) with
/// fixed-point specific signals (saturation/precision).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArithmeticError {
    /// Attempted to divide by zero.
    DivisionByZero,

    /// The mathematical operation is undefined for the given inputs
    /// (e.g., `sqrt(-1.0)`, `acos(2.0)`).
    DomainViolation,

    /// The result exceeded the maximum representable range of the type.
    /// In fixed-point arithmetic, this implies a wrapping or undefined result.
    Overflow,

    /// The result could not be represented exactly, resulting in quantization
    /// or rounding errors (e.g., casting `f64` to `u32` where the float has a decimal).
    PrecisionLoss,

    /// The value exceeded the range but was clamped to the maximum/minimum
    /// representable value (specific to fixed-point/DSP logic).
    Saturation,

    /// The result is smaller than the smallest representable positive value
    /// (Subnormal/Denormal).
    Underflow,
}

/// A specialized `Result` type for fallible arithmetic operations.
pub type ArithmeticResult<T> = Result<T, ArithmeticError>;

impl core::error::Error for ArithmeticError {}

impl core::fmt::Display for ArithmeticError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::DomainViolation => {
                write!(f, "Input is outside the mathematical domain")
            }
            Self::DivisionByZero => write!(f, "Division by zero"),
            Self::Overflow => write!(f, "Value overflowed representable range"),
            Self::Underflow => write!(f, "Value underflowed (subnormal)"),
            Self::Saturation => {
                write!(f, "Value saturated (clamped) at bounds")
            }
            Self::PrecisionLoss => {
                write!(f, "Significant precision was lost during operation")
            }
        }
    }
}

#[cfg(test)]
mod test {
    use crate::math::ArithmeticError;
    use core::fmt::{self, Write};

    /// A simple helper to capture format output into a stack buffer
    struct TestWriter<'a> {
        buf: &'a mut [u8],
        len: usize,
    }

    impl<'a> TestWriter<'a> {
        #[allow(clippy::indexing_slicing)]
        fn as_str(&self) -> &str {
            core::str::from_utf8(&self.buf[..self.len]).expect("Invalid UTF-8")
        }
        fn new(buf: &'a mut [u8]) -> Self {
            Self { buf, len: 0 }
        }
    }

    impl Write for TestWriter<'_> {
        #[allow(clippy::indexing_slicing)]
        #[allow(clippy::arithmetic_side_effects)]
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

    /// A helper function to assert the `Display` output of an `ArithmeticError`.
    ///
    /// This encapsulates the buffer and writer creation, making tests for each
    /// error variant clean and concise.
    fn assert_error_display(err: ArithmeticError, expected_msg: &str) {
        let mut buffer = [0u8; 128]; // Stack-allocated buffer
        let mut writer = TestWriter::new(&mut buffer);

        // Write the error's display output into the buffer
        write!(writer, "{err}")
            .expect("Buffer was too small for the error message");

        // Assert that the written string matches the expected message
        assert_eq!(writer.as_str(), expected_msg);
    }

    #[test]
    fn test_display_division_by_zero() {
        assert_error_display(
            ArithmeticError::DivisionByZero,
            "Division by zero",
        );
    }

    #[test]
    fn test_display_domain_violation() {
        assert_error_display(
            ArithmeticError::DomainViolation,
            "Input is outside the mathematical domain",
        );
    }

    #[test]
    fn test_display_overflow() {
        assert_error_display(
            ArithmeticError::Overflow,
            "Value overflowed representable range",
        );
    }

    #[test]
    fn test_display_precision_loss() {
        assert_error_display(
            ArithmeticError::PrecisionLoss,
            "Significant precision was lost during operation",
        );
    }

    #[test]
    fn test_display_saturation() {
        assert_error_display(
            ArithmeticError::Saturation,
            "Value saturated (clamped) at bounds",
        );
    }

    #[test]
    fn test_display_underflow() {
        assert_error_display(
            ArithmeticError::Underflow,
            "Value underflowed (subnormal)",
        );
    }
}