//! # Polynomial
//!
//! This module contains a base implementation of a generic array polynomial. Many of the methods
//! are not available for the empty polynomial case `N == 0`.
//!
//! # Examples
//!
//! ```rust
//! use control_rs::polynomial::{Polynomial, Constant, Line};
//!
//! let one = Constant::new(1.0);
//! assert_eq!(one.degree(), Some(0));
//! assert_eq!(one.leading_coefficient(), Some(&1.0));
//!
//! let line = Line::new(1.0, 0.0);
//! assert_eq!(line.degree(), Some(1));
//! assert_eq!(line.leading_coefficient(), Some(&1.0));
//! ```
//!
//! # References
//! For an introduction to polynomial functions, see:
//! - [Paul's Online Notes – Polynomials](https://tutorial.math.lamar.edu/Classes/Alg/Polynomials.aspx)
//! - [OpenStax Precalculus – Polynomial Functions](https://openstax.org/books/precalculus/pages/3-introduction-to-polynomial-and-rational-functions)
//!
//! For polynomial evaluation and efficient algorithms like Horner’s method:
//! - [Numerical Recipes – Polynomial Evaluation](https://numerical.recipes/)
use crate::math::ops::{Add, Mul};

#[cfg(any(test, feature = "hil"))]
pub mod test;

/// A trait representing a mathematical polynomial.
///
/// # Generic Arguments
/// * `T` - Numeric type of the coefficients.
pub trait Polynomial<T>
where
    T: Copy + PartialEq + PartialOrd + Add<Output = T> + Mul<Output = T>,
{
    /// Returns a slice of the coefficients of the polynomial, from lowest degree to highest.
    fn coefficients(&self) -> &[T];

    /// Returns a mutable slice of the coefficients of the polynomial.
    fn coefficients_mut(&mut self) -> &mut [T];

    /// Returns the degree of the polynomial, or `None` if the polynomial is empty.
    fn degree(&self) -> Option<usize>;

    /// Evaluates the polynomial at a given point `x` using Horner's method.
    fn evaluate(&self, x: T) -> T;

    /// Returns a reference to the leading coefficient of the polynomial.
    fn leading_coefficient(&self) -> Option<&T>;

    /// Returns a mutable reference to the leading coefficient of the polynomial.
    fn leading_coefficient_mut(&mut self) -> Option<&mut T>;
}

/// A generic polynomial stored as a dense array of coefficients.
/// The coefficients are ordered from lowest degree (constant) to highest.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[allow(clippy::derive_partial_eq_without_eq)]
pub struct DensePolynomial<T, const N: usize> {
    /// The coefficients of the polynomial, ordered from lowest degree (index 0) to highest.
    pub coeffs: [T; N],
}

/// A polynomial of degree 0 (a constant value).
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[allow(clippy::derive_partial_eq_without_eq)]
pub struct Constant<T> {
    /// The constant coefficient value.
    pub coeffs: [T; 1],
}

/// A polynomial of degree 1 (a line `y = m*x + c`).
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[allow(clippy::derive_partial_eq_without_eq)]
pub struct Line<T> {
    /// The line coefficients, ordered as [intercept (c), slope (m)].
    pub coeffs: [T; 2],
}

impl<T, const N: usize> DensePolynomial<T, N> {
    /// Creates a new `DensePolynomial` from an array of coefficients.
    pub const fn new(coeffs: [T; N]) -> Self {
        Self { coeffs }
    }
}

impl<T, const N: usize> Polynomial<T> for DensePolynomial<T, N>
where
    T: Copy + PartialEq + PartialOrd + Add<Output = T> + Mul<Output = T>,
{
    fn coefficients(&self) -> &[T] {
        &self.coeffs
    }

    fn coefficients_mut(&mut self) -> &mut [T] {
        &mut self.coeffs
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn degree(&self) -> Option<usize> {
        if N == 0 { None } else { Some(N - 1) }
    }

    #[allow(clippy::indexing_slicing, clippy::arithmetic_side_effects)]
    fn evaluate(&self, x: T) -> T {
        assert!(N > 0, "Cannot evaluate an empty polynomial");
        // Horner's method
        let mut result = self.coeffs[N - 1];
        let mut i = N - 1;
        while i > 0 {
            i -= 1;
            result = result * x + self.coeffs[i];
        }
        result
    }

    #[allow(clippy::indexing_slicing, clippy::arithmetic_side_effects)]
    fn leading_coefficient(&self) -> Option<&T> {
        if N == 0 {
            None
        } else {
            Some(&self.coeffs[N - 1])
        }
    }

    #[allow(clippy::indexing_slicing, clippy::arithmetic_side_effects)]
    fn leading_coefficient_mut(&mut self) -> Option<&mut T> {
        if N == 0 {
            None
        } else {
            Some(&mut self.coeffs[N - 1])
        }
    }
}

impl<T> Constant<T> {
    /// Creates a new constant polynomial.
    pub const fn new(value: T) -> Self {
        Self { coeffs: [value] }
    }
}

impl<T> Polynomial<T> for Constant<T>
where
    T: Copy + PartialEq + PartialOrd + Add<Output = T> + Mul<Output = T>,
{
    fn coefficients(&self) -> &[T] {
        &self.coeffs
    }

    fn coefficients_mut(&mut self) -> &mut [T] {
        &mut self.coeffs
    }

    fn degree(&self) -> Option<usize> {
        Some(0)
    }

    fn evaluate(&self, _x: T) -> T {
        self.coeffs[0]
    }

    fn leading_coefficient(&self) -> Option<&T> {
        Some(&self.coeffs[0])
    }

    fn leading_coefficient_mut(&mut self) -> Option<&mut T> {
        Some(&mut self.coeffs[0])
    }
}

impl<T> Line<T> {
    /// Creates a new line polynomial with slope `m` and intercept `c`.
    pub const fn new(m: T, c: T) -> Self {
        Self { coeffs: [c, m] }
    }
}

impl<T> Polynomial<T> for Line<T>
where
    T: Copy + PartialEq + PartialOrd + Add<Output = T> + Mul<Output = T>,
{
    fn coefficients(&self) -> &[T] {
        &self.coeffs
    }

    fn coefficients_mut(&mut self) -> &mut [T] {
        &mut self.coeffs
    }

    fn degree(&self) -> Option<usize> {
        Some(1)
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn evaluate(&self, x: T) -> T {
        self.coeffs[1] * x + self.coeffs[0]
    }

    fn leading_coefficient(&self) -> Option<&T> {
        Some(&self.coeffs[1])
    }

    fn leading_coefficient_mut(&mut self) -> Option<&mut T> {
        Some(&mut self.coeffs[1])
    }
}
