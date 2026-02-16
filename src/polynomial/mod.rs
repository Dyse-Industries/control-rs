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

pub trait Polynomial<T>
where
    T: Copy + PartialEq + PartialOrd + Add<Output = T> + Mul<Output = T>,
{
    fn coefficients(&self) -> &[T];
    fn coefficients_mut(&mut self) -> &mut [T];
    fn degree(&self) -> Option<usize>;
    fn evaluate(&self, x: T) -> T;
    fn leading_coefficient(&self) -> Option<&T>;
    fn leading_coefficient_mut(&mut self) -> Option<&mut T>;
}
