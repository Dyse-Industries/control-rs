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
pub use aliases::{Constant, Line};
pub use polynomial::Polynomial;

/// Type aliases for polynomials.
pub mod aliases;

/// Concrete static implementations of polynomials.
///
/// # Clippy Allow explanation
/// We allow `clippy::module_inception` because the user requested that the inner file
/// containing the main polynomial logic be named `polynomial.rs`.
#[allow(clippy::module_inception)]
pub mod polynomial;

#[cfg(any(test, feature = "hil"))]
/// Unit and HIL test suites for polynomials.
pub mod tests;

// /// Divides two polynomials.
// ///
// /// <div class="warning">
// ///
// /// The result is a polynomial with capacity `N`. This may be larger than the degree of the
// /// result polynomial, in which case the higher order coefficients are set to `T::zero()`.
// ///
// /// </div>
// ///
// /// <div class="warning">
// ///
// /// If the divisor is a degenerate polynomial, this will return an array of zeros.
// ///
// /// </div>
// ///
// /// # Generic Arguments
// /// * `T` - Field type of the polynomials.
// /// * `N` - Capacity of the dividend (restricted to `(0,127]`).
// /// * `M` - Capacity of the divisor (restricted to `(0,N]`).
// ///
// /// # Arguments
// /// * `dividend` - The polynomial to be divided.
// /// * `divisor` - The polynomial doing the dividing.
// ///
// /// # Returns
// /// * `quotient` - result of the division.
// ///
// /// # Panics
// /// * There is an unchecked multiplication that may overflow.
// /// * There is an unchecked `add_assign` that may overflow
// /// * There is an unchecked `sub_assign` that may overflow
// ///
// /// # Safety
// /// This function uses `unsafe` code to access elements of the divisor, remainder and quotient.
// /// All the accesses are guaranteed to work based on the generic constants provided.
// ///
// /// # Example
// ///```rust
// /// use control_rs::polynomial::utils::long_division;
// /// let p1: [i32; 2] = [1; 2];
// /// let p2: [i32; 2] = [1; 2];
// /// let expected: [i32; 2] = [1, 0];
// /// assert_eq!(long_division::<_, 2, 2>(p1, &p2), expected, "wrong division result");
// /// ```
// ///
// /// # Algorithm
// /// <pre>
// /// function n / d is
// /// while r ≠ 0 and degree(r) ≥ degree(d) do
// ///     t ← lead(r) / lead(d) // Divide the leading terms
// ///     q ← q + t
// ///     r ← r − t × d
// /// return (q, r)
// ///</pre>
// pub fn long_division<T, const N: usize, const M: usize>(
//     dividend: [T; N],
//     divisor: &[T; M],
// ) -> [T; N]
// where
//     T: Clone + Zero + Div<Output = T> + Mul<Output = T> + AddAssign + SubAssign,
//     Const<N>: DimMax<Const<M>, Output = Const<N>>,
// {
//     let mut quotient: [T; N] = array::from_fn(|_| T::zero());
//     // Find actual degrees
//     let dividend_order = largest_nonzero_index(&dividend);
//     let divisor_order = largest_nonzero_index(divisor);
//
//     // degree of self and rhs exists
//     if let Some(dividend_order) = dividend_order
//         && let Some(divisor_order) = divisor_order
//     {
//         let mut remainder = dividend;
//         // SAFETY: divisor_order is less than the capacity of divisor
//         let leading_divisor = unsafe { divisor.get_unchecked(divisor_order) };
//
//         for i in (divisor_order..=dividend_order).rev() {
//             // SAFETY: index is less than the capacity of dividend, the remainder has the same
//             // capacity
//             let rem_i = unsafe { remainder.get_unchecked(i) };
//             if rem_i.is_zero() {
//                 continue;
//             }
//             // it is guaranteed that `i >= divisor_order` order, this will never panic
//             let q_index = i - divisor_order;
//             // divisor_order is not none so leading divisor is non-zero
//             let term_divisor = rem_i.clone() / leading_divisor.clone();
//             // SAFETY: q_index is less than the capacity of dividend, quotient has the same
//             // capacity
//             unsafe {
//                 *quotient.get_unchecked_mut(q_index) += term_divisor.clone();
//             }
//             for (rem, div) in remainder.iter_mut().skip(q_index).zip(divisor.iter()) {
//                 *rem -= term_divisor.clone() * div.clone();
//             }
//         }
//     }
//     quotient
// }
//
// /// Fits a polynomial to the given data points using the least squares method.
// ///
// /// # Generic Arguments
// /// * `T` - Field type of the data points
// /// * `N` - Capacity of the polynomial coefficients (restricted to `[1,127]`).
// /// * `K` - Number of data points in the X, y sample (restricted to `[1,127]`).
// ///
// /// # Arguments
// /// * `x` - Input data points.
// /// * `y` - Corresponding output values.
// ///
// /// # Returns
// /// * `coefficients` - A degree minor array of coefficients fit to the data.
// ///
// /// # Panics
// /// * The vandermonde matrix cannot be solved
// ///
// /// # Example
// /// ```rust
// /// use control_rs::{polynomial::utils::fit, assert_f64_eq};
// /// let x = [-2.0, -1.0, 0.0, 1.0, 2.0];
// /// let y = [4.0, 1.0, 0.0, 1.0, 4.0];
// /// let coefficients: [f64; 3] = fit::<f64, 3, 5>(&x, &y);
// /// assert_f64_eq!(coefficients[0], 0.0, 2.5e-15);
// /// assert_f64_eq!(coefficients[1], 0.0);
// /// assert_f64_eq!(coefficients[2], 1.0, 9.0e-16);
// /// ```
// /// TODO: Unit tests + docs
// pub fn fit<T: One + RealField, const N: usize, const K: usize>(x: &[T; K], y: &[T; K]) -> [T; N]
// where
//     Const<K>: DimMin<Const<N>>,
//     DimMinimum<Const<K>, Const<N>>: DimSub<U1>,
//     DefaultAllocator: Allocator<DimMinimum<Const<K>, Const<N>>, Const<N>>
//         + Allocator<Const<K>, DimMinimum<Const<K>, Const<N>>>
//         + Allocator<DimMinimum<Const<K>, Const<N>>>
//         + Allocator<DimDiff<DimMinimum<Const<K>, Const<N>>, U1>>,
// {
//     let h = SMatrix::<T, K, 1>::from_row_slice(y);
//     #[allow(clippy::unwrap_used)]
//     let vandermonde = SMatrix::<T, K, N>::from_fn(|i, j| {
//         // (0..degree - j).fold(T::one(), |acc, _| acc * x[i].clone()) // degree major order
//         x[i].clone().powi(j.try_into().unwrap()) // can use RealField trait fn
//     });
//     #[allow(clippy::unwrap_used)]
//     #[allow(clippy::expect_used)]
//     vandermonde
//         .svd(true, true)
//         .solve(&h, T::default_epsilon())
//         .expect("Least squares solution failed")
//         .data
//         .0[0]
//         .clone()
// }
