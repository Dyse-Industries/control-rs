//! # Polynomial Implementation
//!
//! Concrete static representations and operations on polynomials, using compile-time array sizes.

use super::aliases::{Constant, Line};
use crate::math::{
    dsp::{Convolution, CpuConvolution, convolve_static},
    num_traits::{Real, Ring, Zero},
    num_types::{Const, Dim, DimAdd, DimSub, U1},
    storage::reverse_array,
    subprograms::{BasicSubPrograms, level1::POLYEVAL},
};
use core::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign,
};

// ==========================================
// Types and Structs (PascalCase)
// ==========================================

/// A polynomial with a statically known number of coefficients N.
///
/// Coefficients are stored in ascending order: `data[i]` is the coefficient for $x^i$.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StaticPolynomial<T, const N: usize> {
    pub(crate) data: [T; N],
}

// ==========================================
// Implementations (impls)
// ==========================================

impl<T, const N: usize> StaticPolynomial<T, N> {
    /// Computes the derivative of this polynomial.
    ///
    /// The derivative of a polynomial of length N has length max(1, N - 1).
    ///
    /// # Examples
    /// ```
    /// use control_rs::polynomial::{StaticPolynomial, Polynomial};
    /// let p = StaticPolynomial::from_coefficients([1.0, 2.0, 3.0]); // 1 + 2x + 3x^2
    /// let dp = p.derivative(); // 2 + 6x
    /// assert_eq!(dp.coefficients(), &[2.0, 6.0, 0.0]);
    /// ```
    ///
    /// # Clippy Allow explanation
    /// We allow `clippy::arithmetic_side_effects` here because the loop calculates derivative
    /// indices (`i - 1`) and scales coefficients by multiplying their order.
    #[must_use]
    #[allow(clippy::arithmetic_side_effects)]
    #[inline]
    pub fn derivative(&self) -> Self
    where
        T: Real,
    {
        let mut deriv_data = [T::ZERO; N];
        if N > 1 {
            for (i, (dest, src)) in
                (1..N).zip(deriv_data.iter_mut().zip(self.data.iter().skip(1)))
            {
                *dest = src.clone() * T::from_usize(i);
            }
        }
        Self::from_coefficients(deriv_data)
    }

    /// Create a new polynomial from coefficients in ascending order.
    ///
    /// # Examples
    /// ```
    /// use control_rs::polynomial::StaticPolynomial;
    /// let p = StaticPolynomial::from_coefficients([1.0, 2.0, 3.0]); // 1 + 2x + 3x^2
    /// ```
    #[inline(always)]
    pub const fn from_coefficients(data: [T; N]) -> Self {
        Self { data }
    }

    /// Create a new polynomial from coefficients in descending order.
    ///
    /// # Examples
    /// ```
    /// use control_rs::polynomial::StaticPolynomial;
    /// let p = StaticPolynomial::from_descending([3.0, 2.0, 1.0]); // 3x^2 + 2x + 1
    /// ```
    #[inline(always)]
    pub const fn from_descending(data: [T; N]) -> Self
    where
        T: Copy,
    {
        Self {
            data: reverse_array(data),
        }
    }

    /// Multiplies this polynomial with another, statically verifying the output size.
    #[inline]
    pub fn mul_poly<const M: usize, const OUT: usize, Sum>(
        &self,
        rhs: &StaticPolynomial<T, M>,
    ) -> StaticPolynomial<T, OUT>
    where
        T: Real,
        Const<N>: Dim,
        Const<M>: Dim,
        Const<OUT>: Dim,
        <Const<N> as Dim>::PeanoTypeNum:
            DimAdd<<Const<M> as Dim>::PeanoTypeNum, Output = Sum>,
        Sum: Dim + DimSub<U1, Output = <Const<OUT> as Dim>::PeanoTypeNum>,
    {
        self.mul_with_conv::<CpuConvolution, M, OUT, Sum>(rhs)
    }

    /// Multiplies this polynomial with another using a custom convolution algorithm from the DSP module.
    #[inline]
    pub fn mul_with_conv<C, const M: usize, const OUT: usize, Sum>(
        &self,
        rhs: &StaticPolynomial<T, M>,
    ) -> StaticPolynomial<T, OUT>
    where
        C: Convolution<T>,
        T: Real,
        Const<N>: Dim,
        Const<M>: Dim,
        Const<OUT>: Dim,
        <Const<N> as Dim>::PeanoTypeNum:
            DimAdd<<Const<M> as Dim>::PeanoTypeNum, Output = Sum>,
        Sum: Dim + DimSub<U1, Output = <Const<OUT> as Dim>::PeanoTypeNum>,
    {
        let data =
            convolve_static::<C, T, N, M, OUT, Sum>(&self.data, &rhs.data);
        StaticPolynomial::from_coefficients(data)
    }

    /// Pads the polynomial coefficients to a statically specified size M.
    /// If M is smaller than N, it will truncate the coefficients.
    #[inline]
    pub fn pad<const M: usize>(&self) -> StaticPolynomial<T, M>
    where
        T: Zero + Copy,
    {
        let mut padded_data = [T::ZERO; M];
        let copy_len = if N < M { N } else { M };
        padded_data[..copy_len].copy_from_slice(&self.data[..copy_len]);
        StaticPolynomial::from_coefficients(padded_data)
    }
}

impl<T, const N: usize> super::Polynomial<T> for StaticPolynomial<T, N>
where
    T: Zero + Copy + PartialEq + PartialOrd + Add<Output = T> + Mul<Output = T>,
    Const<N>: Dim,
{
    #[inline(always)]
    fn coefficients(&self) -> &[T] {
        &self.data
    }

    #[inline(always)]
    fn coefficients_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    #[inline]
    fn degree(&self) -> Option<usize> {
        (0..N)
            .rev()
            .find(|&i| self.data.get(i).is_some_and(|val| !val.is_zero()))
    }

    /// Evaluates the polynomial at the given point `x` using Horner's method subprogram.
    #[inline]
    fn evaluate(&self, x: T) -> T {
        BasicSubPrograms::polyeval(&self.data, x)
    }

    #[inline]
    fn leading_coefficient(&self) -> Option<&T> {
        let deg = self.degree()?;
        self.data.get(deg)
    }

    #[inline]
    fn leading_coefficient_mut(&mut self) -> Option<&mut T> {
        let deg = self.degree()?;
        self.data.get_mut(deg)
    }
}

impl<T> Constant<T> {
    /// Create a new constant polynomial.
    #[inline(always)]
    pub const fn new(val: T) -> Self {
        Self { data: [val] }
    }
}

impl<T> Line<T> {
    /// Create a new linear polynomial $a x + b$.
    #[inline(always)]
    pub const fn new(a: T, b: T) -> Self {
        Self { data: [b, a] }
    }
}

impl<T, const N: usize> AddAssign<&Self> for StaticPolynomial<T, N>
where
    T: Ring,
{
    /// Performs element-wise polynomial addition in-place using standard BLAS AXPY subprograms.
    #[inline]
    fn add_assign(&mut self, rhs: &Self) {
        use crate::math::subprograms::level1::AXPY;
        BasicSubPrograms::axpy(T::ONE, &rhs.data, &mut self.data);
    }
}

impl<T, const N: usize> AddAssign<Self> for StaticPolynomial<T, N>
where
    T: Ring,
{
    /// Performs element-wise polynomial addition in-place using standard BLAS AXPY subprograms.
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.add_assign(&rhs);
    }
}

impl<T, const N: usize> SubAssign<&Self> for StaticPolynomial<T, N>
where
    T: Ring + crate::math::ops::Neg<Output = T>,
{
    /// Performs element-wise polynomial subtraction in-place using standard BLAS AXPY subprograms.
    ///
    /// # Clippy Allow explanation
    /// We allow `clippy::arithmetic_side_effects` here because negating `T::ONE` to pass to `axpy`
    /// is a standard algebraic representation of negation/subtraction.
    #[allow(clippy::arithmetic_side_effects)]
    #[inline]
    fn sub_assign(&mut self, rhs: &Self) {
        use crate::math::subprograms::level1::AXPY;
        BasicSubPrograms::axpy(-T::ONE, &rhs.data, &mut self.data);
    }
}

impl<T, const N: usize> SubAssign<Self> for StaticPolynomial<T, N>
where
    T: Ring + crate::math::ops::Neg<Output = T>,
{
    /// Performs element-wise polynomial subtraction in-place using standard BLAS AXPY subprograms.
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.sub_assign(&rhs);
    }
}

impl<T, const N: usize> Add<Self> for StaticPolynomial<T, N>
where
    T: Ring,
{
    type Output = Self;

    /// Adds two polynomials element-wise using BLAS AXPY.
    #[inline]
    fn add(mut self, rhs: Self) -> Self::Output {
        self.add_assign(&rhs);
        self
    }
}

impl<T, const N: usize> Add<&Self> for StaticPolynomial<T, N>
where
    T: Ring,
{
    type Output = Self;

    /// Adds two polynomials element-wise using BLAS AXPY.
    #[inline]
    fn add(mut self, rhs: &Self) -> Self::Output {
        self.add_assign(rhs);
        self
    }
}

impl<T, const N: usize> Sub<Self> for StaticPolynomial<T, N>
where
    T: Ring + crate::math::ops::Neg<Output = T>,
{
    type Output = Self;

    /// Subtracts two polynomials element-wise using BLAS AXPY.
    #[inline]
    fn sub(mut self, rhs: Self) -> Self::Output {
        self.sub_assign(&rhs);
        self
    }
}

impl<T, const N: usize> Sub<&Self> for StaticPolynomial<T, N>
where
    T: Ring + crate::math::ops::Neg<Output = T>,
{
    type Output = Self;

    /// Subtracts two polynomials element-wise using BLAS AXPY.
    #[inline]
    fn sub(mut self, rhs: &Self) -> Self::Output {
        self.sub_assign(rhs);
        self
    }
}

impl<T, const N: usize> Mul<T> for StaticPolynomial<T, N>
where
    T: Ring + Copy,
{
    type Output = Self;

    /// Scales the polynomial by a scalar factor.
    ///
    /// # Clippy Allow explanation
    /// We allow `clippy::arithmetic_side_effects` here because calling the `*=` operator
    /// performs scalar-polynomial multiplication.
    #[allow(clippy::arithmetic_side_effects)]
    #[inline]
    fn mul(mut self, rhs: T) -> Self::Output {
        self *= rhs;
        self
    }
}

impl<T, const N: usize> MulAssign<T> for StaticPolynomial<T, N>
where
    T: Ring + Copy,
{
    /// Scales the polynomial by a scalar factor in-place.
    ///
    /// # Clippy Allow explanation
    /// Allowed at method level because this wraps primitive multiplication where behavior is defined by `T`.
    #[allow(clippy::arithmetic_side_effects)]
    #[inline]
    fn mul_assign(&mut self, rhs: T) {
        for val in &mut self.data {
            *val = *val * rhs;
        }
    }
}

impl<T, const N: usize> Div<T> for StaticPolynomial<T, N>
where
    T: crate::math::num_traits::Field + Copy,
{
    type Output = Self;

    /// Scales the polynomial by a scalar divisor.
    ///
    /// # Clippy Allow explanation
    /// We allow `clippy::arithmetic_side_effects` here because calling the `/=` operator
    /// performs scalar-polynomial division.
    #[allow(clippy::arithmetic_side_effects)]
    #[inline]
    fn div(mut self, rhs: T) -> Self::Output {
        self /= rhs;
        self
    }
}

impl<T, const N: usize> DivAssign<T> for StaticPolynomial<T, N>
where
    T: crate::math::num_traits::Field + Copy,
{
    /// Scales the polynomial by a scalar divisor in-place.
    ///
    /// # Clippy Allow explanation
    /// Allowed at method level because this wraps primitive division where behavior is defined by `T`.
    #[allow(clippy::arithmetic_side_effects)]
    #[inline]
    fn div_assign(&mut self, rhs: T) {
        for val in &mut self.data {
            *val = *val / rhs;
        }
    }
}
