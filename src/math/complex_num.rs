//! Complex number representation and arithmetic operations.
//!
//! This module provides a generic `Complex` struct and basic arithmetic operations
//! (addition, subtraction, multiplication, division) for complex numbers.

use crate::math::ops::{WrappingAdd, WrappingMul, WrappingSub};
use core::ops::{Add, Div, Mul, Neg, Sub};

/// A complex number consisting of a real and an imaginary part.
#[allow(clippy::arbitrary_source_item_ordering)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Complex<T> {
    /// The real part of the complex number.
    pub re: T,
    /// The imaginary part of the complex number.
    pub im: T,
}

impl<T> Complex<T> {
    /// Returns the conjugate of the complex number.
    ///
    /// The conjugate of `a + bi` is `a - bi`.
    #[must_use]
    pub fn conj(self) -> Self
    where
        T: Neg<Output = T>,
    {
        Self::new(self.re, self.im.neg())
    }

    /// Creates a new complex number from real and imaginary parts.
    ///
    /// # Arguments
    ///
    /// * `re` - The real part.
    /// * `im` - The imaginary part.
    #[must_use]
    pub const fn new(re: T, im: T) -> Self {
        Self { re, im }
    }
}

impl<T: Add<Output = T>> Add for Complex<T> {
    type Output = Self;
    #[allow(clippy::arithmetic_side_effects)]
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl<T: Sub<Output = T>> Sub for Complex<T> {
    type Output = Self;
    #[allow(clippy::arithmetic_side_effects)]
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

impl<T: Clone + Mul<Output = T> + Add<Output = T> + Sub<Output = T>> Mul
    for Complex<T>
{
    type Output = Self;
    #[allow(clippy::arithmetic_side_effects)]
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            re: (self.re.clone() * rhs.re.clone())
                - (self.im.clone() * rhs.im.clone()),
            im: (self.re.clone() * rhs.im.clone()) + (self.im * rhs.re),
        }
    }
}

impl<
    T: Mul<Output = T>
        + Add<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + Copy,
> Div for Complex<T>
{
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects)]
    fn div(self, rhs: Self) -> Self::Output {
        let denominator = (rhs.re * rhs.re) + (rhs.im * rhs.im);
        Self {
            re: ((self.re * rhs.re) + (self.im * rhs.im)) / denominator,
            im: ((self.im * rhs.re) - (self.re * rhs.im)) / denominator,
        }
    }
}

impl<T: WrappingAdd> WrappingAdd for Complex<T> {
    fn wrapping_add(&self, v: &Self) -> Self {
        Self {
            re: self.re.wrapping_add(&v.re),
            im: self.im.wrapping_add(&v.im),
        }
    }
}

impl<T: WrappingSub> WrappingSub for Complex<T> {
    fn wrapping_sub(&self, v: &Self) -> Self {
        Self {
            re: self.re.wrapping_sub(&v.re),
            im: self.im.wrapping_sub(&v.im),
        }
    }
}

impl<T: WrappingAdd + WrappingSub + WrappingMul + Copy> WrappingMul
    for Complex<T>
{
    fn wrapping_mul(&self, v: &Self) -> Self {
        let re = self
            .re
            .wrapping_mul(&v.re)
            .wrapping_sub(&self.im.wrapping_mul(&v.im));
        let im = self
            .re
            .wrapping_mul(&v.im)
            .wrapping_add(&self.im.wrapping_mul(&v.re));
        Self { re, im }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_almost_eq;
    #[test]
    fn test_complex_new() {
        let c = Complex::new(1.0, 2.0);
        assert_almost_eq!(c.re, 1.0);
        assert_almost_eq!(c.im, 2.0);
    }

    #[test]
    fn test_complex_add() {
        let a = Complex::new(1, 2);
        let b = Complex::new(3, 4);
        let c = a + b;
        assert_eq!(c, Complex::new(4, 6));
    }

    #[test]
    fn test_complex_sub() {
        let a = Complex::new(5, 7);
        let b = Complex::new(2, 3);
        let c = a - b;
        assert_eq!(c, Complex::new(3, 4));
    }

    #[test]
    fn test_complex_mul() {
        let a = Complex::new(1, 2);
        let b = Complex::new(3, 4);
        let c = a * b;
        assert_eq!(c, Complex::new(-5, 10));
    }

    #[test]
    fn test_complex_div() {
        let a = Complex::new(1.0, 2.0);
        let b = Complex::new(3.0, 4.0);
        let c = a / b;
        assert_eq!(c, Complex::new(11.0 / 25.0, 2.0 / 25.0));
    }

    #[test]
    fn test_complex_conj() {
        let c = Complex::new(1, 2);
        let conj_c = c.conj();
        assert_eq!(conj_c, Complex::new(1, -2));
    }
}
