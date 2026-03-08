//! # Math
//!
//! Core numerical primitives optimized for bare-metal execution.
//!
//! ## Usage
//!
//! ```rust
//! use control_rs::math::subprograms::level1::AXPY;
//! use control_rs::math::ArithmeticResult;
//! use core::marker::PhantomData;
//!
//! pub struct Controller<B> {
//!     _marker: PhantomData<B>,
//! }
//!
//! impl<B: AXPY<f32>> Controller<B> {
//!     // the Generic argument N provides a zero-cost safety guarantee.
//!     // (state.size() == input.size())
//!     pub fn update<const N: usize>(
//!         &self,
//!         state: &mut [f32; N],
//!         input: &[f32; N],
//!         gain: f32
//!     ) -> ArithmeticResult<()>
//!     {
//!         B::axpy(gain, input, state);
//!         Ok(())
//!     }
//! }
//! ```
//!
//! By leveraging `DimAdd` and `DimSub`, we can strictly guarantee that the resulting
//! type has the worst case number of coefficients allocated for the output.
//!
//! ```rust
//! use control_rs::math::num_types::{Dim, DimAdd, DimSub, Const, U1};
//! use core::marker::PhantomData;
//!
//! /// A polynomial with a statically known number of coefficients.
//! pub struct StaticPolynomial<C: Dim> {
//!     // The dimension type `C` represents the array length (Degree + 1)
//!     _marker: PhantomData<C>,
//! }
//!
//! impl<C: Dim> StaticPolynomial<C> {
//!     pub fn new() -> Self {
//!         Self { _marker: PhantomData }
//!     }
//! }
//!
//! /// Multiplies two polynomials of length N and M.
//! /// The resulting polynomial requires exactly (N + M) - 1 coefficients.
//! pub fn mul_poly<const N: usize, const M: usize, Sum, Out>(
//!     _a: &StaticPolynomial<Const<N>>,
//!     _b: &StaticPolynomial<Const<M>>,
//! ) -> StaticPolynomial<Out>
//! where
//!     Const<N>: DimAdd<Const<M>, Output = Sum>,
//!     Sum: Dim + DimSub<U1, Output = Out>,
//!     Out: Dim,
//! {
//!     // The compiler enforces that `Out` is exactly N + M - 1.
//!     // Attempting to return a polynomial of any other size fails to compile.
//!     StaticPolynomial::<Out>::new()
//! }
//!
//! // --- Usage ---
//! // Polynomial A: Length 3 (e.g., ax^2 + bx + c)
//! let p_a = StaticPolynomial::<Const<3>>::new();
//!
//! // Polynomial B: Length 2 (e.g., dx + e)
//! let p_b = StaticPolynomial::<Const<2>>::new();
//!
//! // Result C: The compiler infers length 4 (3 + 2-1).
//! // Degree 2 * Degree 1 = Degree 3 (which requires 4 coefficients).
//! let p_c = mul_poly(&p_a, &p_b);
//! ```
//! # References
//! - [Numerical Recipes - The Art of Scientific Computing](https://numerical.recipes/)

pub mod assert;

pub mod num_traits;
pub mod num_types;
pub mod ops;
pub mod storage;
pub mod subprograms;

#[cfg(test)]
mod tests;

/// A unified error type for arithmetic operations.
///
/// This structure balances high-level control flow (overflow/underflow) with
/// fixed-point specific signals (saturation/precision).
///
/// # Safety
/// This enum does not use `unsafe` code.
///
/// # Example
/// ```
/// use control_rs::math::ArithmeticError;
///
/// let err = ArithmeticError::DivisionByZero;
/// assert_eq!(format!("{}", err), "Division by zero");
/// ```
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