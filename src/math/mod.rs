//! # Math
//!
//! Core numerical primitives optimized for bare-metal execution.
//!
//! ## Usage
//!
//! ```rust
//! use control_rs::math::num_types::{Const, Dim};
//! use control_rs::math::storage::{ArrayStorage, DenseStorage, DenseStorageMut};
//! use control_rs::math::subprograms::level1::Axpy;
//! use control_rs::math::ArithmeticResult;
//! use core::marker::PhantomData;
//!
//! pub struct Controller<B> {
//!     _marker: PhantomData<B>,
//! }
//!
//! impl<B> Controller<B> {
//!     pub fn update<const N: usize>(
//!         &self,
//!         state: &mut ArrayStorage<f32, N, 1>,
//!         input: &ArrayStorage<f32, N, 1>,
//!         gain: f32,
//!     ) -> ArithmeticResult<()>
//!     where
//!         Const<N>: Dim,
//!         B: Axpy<f32, ArrayStorage<f32, N, 1>, ArrayStorage<f32, N, 1>>,
//!     {
//!         B::axpy(gain, input, state);
//!         Ok(())
//!     }
//! }
//! ```
//!
//! Leveraging `DimAdd` and `DimSub` strictly guarantees that the resulting
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
//! pub fn mul_poly<N, M, Sum, Out>(
//!     _a: &StaticPolynomial<N>,
//!     _b: &StaticPolynomial<M>,
//! ) -> StaticPolynomial<Out>
//! where
//!     N: Dim + DimAdd<M, Output = Sum>,
//!     M: Dim,
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
pub mod complex_num;
pub mod dsp;
pub mod fixed_num;
pub mod num_traits;
pub mod num_types;
pub mod ops;
pub mod storage;
pub mod subprograms;

#[cfg(any(test, feature = "ets"))]
/// Core mathematical unit tests.
pub mod tests;

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

/// Representation and layout conversion errors, shared across `Matrix`,
/// `Polynomial` and `Tensor` conversions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConversionError {
    /// Dimension or capacity overflow/underflow during calculations.
    DimensionMismatch,

    /// The polynomial is not monic (leading coefficient is not ONE),
    /// preventing companion matrix construction.
    NonMonicPolynomial,
}

/// A specialized `Result` type for fallible representation/layout conversions.
pub type ConversionResult<T> = Result<T, ConversionError>;

/// Indexing, capacity, and structural-invariant failures on storage backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageError {
    /// `nnz` would exceed the compile-time `MAX_NNZ` of a stack sparse leaf.
    CapacityExceeded,

    /// Write to a unit-diagonal slot of `TriangularPackedStorage`.
    ImmutableUnitDiagonal,

    /// Write of a non-real value to a Hermitian diagonal slot.
    InvalidHermitianDiagonal,

    /// Write to an unallocated sparse coordinate, or a compressed buffer
    /// that violates its offset/index contract.
    InvalidStructuralInvariant,

    /// Logical `(r, c)` (or packed `(i, j)`) exceeds the backend's dimensions.
    OutOfBounds,
}

/// A specialized `Result` type for fallible storage backend operations.
pub type StorageResult<T> = Result<T, StorageError>;

/// Unified linear algebra errors, supplementing [`ArithmeticError`] for
/// `Matrix` factorization, inversion and system-solving failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinAlgError {
    /// Jacobi eigensolver (`Syev` / `Heev`) exhausted its iteration bound.
    MaxIterationsReached,

    /// Cholesky (`Potrf` / `Pptrf`) encountered a non-positive pivot.
    NotPositiveDefinite,

    /// LU (`Getrf`) or a triangular solve encountered an exact-zero pivot.
    SingularMatrix,

    /// Caller-provided `tau` / `work` / `ipiv` slice is shorter than required.
    WorkspaceTooSmall,
}

/// A specialized `Result` type for fallible linear algebra operations.
pub type LinAlgResult<T> = Result<T, LinAlgError>;

/// Convenience enum representing the 2D Cartesian Plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CartesianQuadrant2D {
    /// Directly to the left of the origin.
    NegativeXAxis,
    /// Directly below the origin.
    NegativeYAxis,
    /// The center of the plane.
    Origin,
    /// Directly to the right of the origin.
    PositiveXAxis,
    /// Directly above the origin.
    PositiveYAxis,
    /// The first quadrant (+, +)
    Q1,
    /// The second quadrant (-, +)
    Q2,
    /// The third quadrant (-, -)
    Q3,
    /// The fourth quadrant (+, -)
    Q4,
    /// Undefined values do not exist on the plane.
    Undefined,
}

/// A mathematical mapping (or function) $f: X \to Y$.
///
/// This trait defines a deterministic mapping from an input space to an output space.
///
/// # Type Parameters
/// * `Domain` - The mathematical set $X$ of all valid inputs.
/// * `Codomain` - The mathematical set $Y$ into which all outputs fall.
///   (Note: The actual produced outputs form the "Range" or "Image", which is a subset of the Codomain).
///
/// # Example
/// ```
/// use control_rs::math::{Map, StateEquation};
///
/// struct MockStateEquation;
///
/// impl StateEquation<f32, f32, f32> for MockStateEquation {
///     fn dynamics(&self, x: &f32, u: &f32) -> f32 {
///         x + u
///     }
/// }
///
/// let mock_eq = MockStateEquation;
/// let state = 2.0f32;
/// let input = 3.0f32;
///
/// // Call evaluate on the MockStateEquation instance.
/// // The blanket implementation routes the tuple (&state, &input)
/// // to the dynamics method, which calculates state + input.
/// let result = mock_eq.evaluate((&state, &input));
///
/// assert_eq!(result, 5.0f32);
/// ```
pub trait Map<Domain, Codomain> {
    /// Evaluates the mapping $y = f(x)$ for a given input.
    fn evaluate(&self, x: Domain) -> Codomain;
}

/// A bijective mathematical mapping $f: X \to Y$ with a defined inverse $f^{-1}: Y \to X$.
///
/// Because this represents a bijection, every element in the Domain maps to exactly
/// one element in the Codomain and vice versa.
pub trait Bijection<Domain, Codomain>: Map<Domain, Codomain> {
    /// Evaluates the inverse mapping $x = f^{-1}(y)$.
    fn evaluate_inverse(&self, y: Codomain) -> Domain;
}

/// A state-space equation governing a dynamical system.
///
/// Mathematically, this represents a function $f: X \times U \to X$ for discrete-time systems,
/// or $f: X \times U \to T X$ for continuous-time systems.
///
/// # Type Parameters
/// * `State` - The state space $X$ of the system.
/// * `Input` - The control input space $U$.
/// * `Output` - The resulting state derivative or next state.
pub trait StateEquation<State, Input, Output> {
    /// Evaluates the system dynamics for a given state and input.
    fn dynamics(&self, x: &State, u: &Input) -> Output;
}

/// A specialized `Result` type for fallible arithmetic operations.
pub type ArithmeticResult<T> = Result<T, ArithmeticError>;

/// Blanket implementation of `Map` for anything that implements `StateEquation`.
/// The mathematical Domain is the Cartesian product of State and Input (X × U).
///
/// Due to the complexity of the generics, Tarpaulin sees this as uncovered.
impl<'a, T, State, Input, Output> Map<(&'a State, &'a Input), Output> for T
where
    T: StateEquation<State, Input, Output>,
{
    fn evaluate(&self, (x, u): (&'a State, &'a Input)) -> Output {
        T::dynamics(self, x, u)
    }
}

////////////////////////////////////////////////////////////////////////////////

impl core::error::Error for ArithmeticError {}

////////////////////////////////////////////////////////////////////////////////

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

////////////////////////////////////////////////////////////////////////////////

impl core::error::Error for ConversionError {}

////////////////////////////////////////////////////////////////////////////////

impl core::fmt::Display for ConversionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NonMonicPolynomial => {
                write!(f, "Polynomial is not monic")
            }
            Self::DimensionMismatch => {
                write!(f, "Dimension or capacity overflow/underflow")
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

impl core::error::Error for StorageError {}

////////////////////////////////////////////////////////////////////////////////

impl core::fmt::Display for StorageError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::OutOfBounds => write!(f, "Index out of bounds"),
            Self::CapacityExceeded => {
                write!(f, "Maximum sparse capacity exceeded")
            }
            Self::ImmutableUnitDiagonal => {
                write!(f, "Unit diagonal entries cannot be modified")
            }
            Self::InvalidHermitianDiagonal => {
                write!(
                    f,
                    "Hermitian diagonal entries must have zero imaginary part"
                )
            }
            Self::InvalidStructuralInvariant => {
                write!(f, "Storage structural invariant violated")
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

impl core::error::Error for LinAlgError {}

////////////////////////////////////////////////////////////////////////////////

impl core::fmt::Display for LinAlgError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotPositiveDefinite => {
                write!(f, "Matrix is not positive definite")
            }
            Self::SingularMatrix => write!(f, "Matrix is singular"),
            Self::WorkspaceTooSmall => {
                write!(f, "Caller-provided workspace slice is too small")
            }
            Self::MaxIterationsReached => {
                write!(f, "Maximum iterative solver iterations reached")
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod test {
    use crate::math::{
        ArithmeticError, ConversionError, LinAlgError, StorageError,
    };
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

    /// A helper function to assert the `Display` output of a `ConversionError`.
    fn assert_conversion_error_display(
        err: ConversionError,
        expected_msg: &str,
    ) {
        let mut buffer = [0u8; 128]; // Stack-allocated buffer
        let mut writer = TestWriter::new(&mut buffer);

        write!(writer, "{err}")
            .expect("Buffer was too small for the error message");

        assert_eq!(writer.as_str(), expected_msg);
    }

    /// A helper function to assert the `Display` output of a `StorageError`.
    fn assert_storage_error_display(err: StorageError, expected_msg: &str) {
        let mut buffer = [0u8; 128]; // Stack-allocated buffer
        let mut writer = TestWriter::new(&mut buffer);

        write!(writer, "{err}")
            .expect("Buffer was too small for the error message");

        assert_eq!(writer.as_str(), expected_msg);
    }

    /// A helper function to assert the `Display` output of a `LinAlgError`.
    fn assert_lin_alg_error_display(err: LinAlgError, expected_msg: &str) {
        let mut buffer = [0u8; 128]; // Stack-allocated buffer
        let mut writer = TestWriter::new(&mut buffer);

        write!(writer, "{err}")
            .expect("Buffer was too small for the error message");

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

    #[test]
    fn test_display_non_monic_polynomial() {
        assert_conversion_error_display(
            ConversionError::NonMonicPolynomial,
            "Polynomial is not monic",
        );
    }

    #[test]
    fn test_display_dimension_mismatch() {
        assert_conversion_error_display(
            ConversionError::DimensionMismatch,
            "Dimension or capacity overflow/underflow",
        );
    }

    #[test]
    fn test_display_storage_errors() {
        assert_storage_error_display(
            StorageError::OutOfBounds,
            "Index out of bounds",
        );
        assert_storage_error_display(
            StorageError::CapacityExceeded,
            "Maximum sparse capacity exceeded",
        );
        assert_storage_error_display(
            StorageError::ImmutableUnitDiagonal,
            "Unit diagonal entries cannot be modified",
        );
        assert_storage_error_display(
            StorageError::InvalidHermitianDiagonal,
            "Hermitian diagonal entries must have zero imaginary part",
        );
        assert_storage_error_display(
            StorageError::InvalidStructuralInvariant,
            "Storage structural invariant violated",
        );
    }

    #[test]
    fn test_display_lin_alg_errors() {
        assert_lin_alg_error_display(
            LinAlgError::NotPositiveDefinite,
            "Matrix is not positive definite",
        );
        assert_lin_alg_error_display(
            LinAlgError::SingularMatrix,
            "Matrix is singular",
        );
        assert_lin_alg_error_display(
            LinAlgError::WorkspaceTooSmall,
            "Caller-provided workspace slice is too small",
        );
        assert_lin_alg_error_display(
            LinAlgError::MaxIterationsReached,
            "Maximum iterative solver iterations reached",
        );
    }

    #[test]
    fn test_state_equation_blanket_map() {
        use crate::math::{Map, StateEquation};

        struct TestEq;
        impl StateEquation<i32, i32, i32> for TestEq {
            fn dynamics(&self, x: &i32, u: &i32) -> i32 {
                x + u
            }
        }

        let eq = TestEq;
        let x = 2;
        let u = 3;
        assert_eq!(eq.evaluate((&x, &u)), 5);
    }
}
