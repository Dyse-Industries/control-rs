//! # Transfer Function
//!
//! > "A transfer function is a convenient way to represent a linear, time-invariant system in terms
//! > of its input-output relationship. It is obtained by applying a Laplace transform to the
//! > differential equations describing system dynamics, assuming zero initial conditions. In the
//! > absence of these equations, a transfer function can also be estimated from measured
//! > input-output data.
//! >
//! > Transfer functions are frequently used in block diagram representations of systems and are
//! > popular for performing time-domain and frequency-domain analyses and controller design. The
//! > key advantage of transfer functions is that they allow engineers to use simple algebraic
//! > equations instead of complex differential equations for analyzing and designing systems."
//!
//! [MathWorks](https://www.mathworks.com/discovery/transfer-function.html)

#[cfg(any(test, feature = "hil"))]
pub mod test;

/// A trait representing a mathematical transfer function.
///
/// # Generic Arguments
/// * `N` - Numeric type of the numerator coefficients.
/// * `D` - Numeric type of the denominator coefficients.
pub trait TransferFunction<N, D>
where
    N: Copy + Clone,
    D: Copy + Clone,
{
    /// Returns the coefficients of the denominator polynomial, ordered from lowest degree to highest.
    fn denominator(&self) -> &[D];
    /// Returns the coefficients of the numerator polynomial, ordered from lowest degree to highest.
    fn numerator(&self) -> &[N];
}

/// A transfer function represented by statically sized arrays for numerator and denominator.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[allow(clippy::derive_partial_eq_without_eq)]
pub struct StaticTransferFunction<T, const N: usize, const D: usize> {
    /// The coefficients of the denominator polynomial, ordered from lowest degree to highest.
    pub denominator: [T; D],
    /// The coefficients of the numerator polynomial, ordered from lowest degree to highest.
    pub numerator: [T; N],
}

impl<T, const N: usize, const D: usize> StaticTransferFunction<T, N, D> {
    /// Creates a new `StaticTransferFunction` from numerator and denominator arrays.
    pub const fn new(numerator: [T; N], denominator: [T; D]) -> Self {
        Self {
            denominator,
            numerator,
        }
    }
}

impl<T, const N: usize, const D: usize> TransferFunction<T, T>
    for StaticTransferFunction<T, N, D>
where
    T: Copy + Clone,
{
    fn denominator(&self) -> &[T] {
        &self.denominator
    }

    fn numerator(&self) -> &[T] {
        &self.numerator
    }
}
