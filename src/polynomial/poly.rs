#![allow(
    clippy::arbitrary_source_item_ordering,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::use_self,
    clippy::manual_find
)]

use crate::math::{
    dsp::{Convolution, CpuConvolution, convolve_static},
    num_traits::{Real, Zero},
    num_types::{Const, Dim, DimAdd, DimSub, U1},
    storage::reverse_array,
};
use core::ops::{Add, AddAssign, Mul, Sub, SubAssign};

/// A polynomial with a statically known number of coefficients N.
///
/// Coefficients are stored in ascending order: `data[i]` is the coefficient for $x^i$.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StaticPolynomial<T, const N: usize> {
    pub(crate) data: [T; N],
}

#[allow(clippy::arithmetic_side_effects)]
impl<T, const N: usize> StaticPolynomial<T, N> {
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

    /// Multiplies this polynomial with another using a custom convolution algorithm.
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
    #[must_use]
    #[inline]
    pub fn derivative(&self) -> Self
    where
        T: Real,
    {
        let mut deriv_data = [T::ZERO; N];
        if N > 1 {
            for i in 1..N {
                deriv_data[i - 1] = self.data[i].clone() * T::from_usize(i);
            }
        }
        Self::from_coefficients(deriv_data)
    }
}

// --- Implementation of the library's Polynomial trait ---

impl<T, const N: usize> super::Polynomial<T> for StaticPolynomial<T, N>
where
    T: Zero + Copy + PartialEq + PartialOrd + Add<Output = T> + Mul<Output = T>,
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
        if N == 0 {
            return None;
        }
        for i in (0..N).rev() {
            if !self.data[i].is_zero() {
                return Some(i);
            }
        }
        None
    }

    #[inline]
    fn evaluate(&self, x: T) -> T {
        if N == 0 {
            return T::ZERO;
        }
        let mut acc = self.data[N - 1];
        let mut i = N - 1;
        while i > 0 {
            i -= 1;
            acc = acc * x + self.data[i];
        }
        acc
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

// --- Wrapper Aliases: Constant and Line ---

/// A constant polynomial: $p(x) = c$.
pub type Constant<T> = StaticPolynomial<T, 1>;

impl<T> StaticPolynomial<T, 1> {
    /// Create a new constant polynomial.
    #[inline(always)]
    pub const fn new(val: T) -> Self {
        Self { data: [val] }
    }
}

/// A linear polynomial: $p(x) = a x + b$.
pub type Line<T> = StaticPolynomial<T, 2>;

impl<T> StaticPolynomial<T, 2> {
    /// Create a new linear polynomial $a x + b$.
    #[inline(always)]
    pub const fn new(a: T, b: T) -> Self {
        Self { data: [b, a] }
    }
}

// --- Arithmetic Operations ---

#[allow(clippy::arithmetic_side_effects)]
impl<T, const N: usize> AddAssign<&Self> for StaticPolynomial<T, N>
where
    T: AddAssign<T> + Copy,
{
    #[inline]
    fn add_assign(&mut self, rhs: &Self) {
        for (d, s) in self.data.iter_mut().zip(rhs.data.iter()) {
            *d += *s;
        }
    }
}

#[allow(clippy::arithmetic_side_effects)]
impl<T, const N: usize> AddAssign<Self> for StaticPolynomial<T, N>
where
    T: AddAssign<T> + Copy,
{
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.add_assign(&rhs);
    }
}

#[allow(clippy::arithmetic_side_effects)]
impl<T, const N: usize> SubAssign<&Self> for StaticPolynomial<T, N>
where
    T: SubAssign<T> + Copy,
{
    #[inline]
    fn sub_assign(&mut self, rhs: &Self) {
        for (d, s) in self.data.iter_mut().zip(rhs.data.iter()) {
            *d -= *s;
        }
    }
}

#[allow(clippy::arithmetic_side_effects)]
impl<T, const N: usize> SubAssign<Self> for StaticPolynomial<T, N>
where
    T: SubAssign<T> + Copy,
{
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.sub_assign(&rhs);
    }
}

#[allow(clippy::arithmetic_side_effects)]
impl<T, const N: usize> Add<Self> for StaticPolynomial<T, N>
where
    T: Add<Output = T> + Copy,
{
    type Output = Self;

    #[inline]
    fn add(mut self, rhs: Self) -> Self::Output {
        for (d, s) in self.data.iter_mut().zip(rhs.data.iter()) {
            *d = *d + *s;
        }
        self
    }
}

#[allow(clippy::arithmetic_side_effects)]
impl<T, const N: usize> Add<&Self> for StaticPolynomial<T, N>
where
    T: Add<Output = T> + Copy,
{
    type Output = Self;

    #[inline]
    fn add(mut self, rhs: &Self) -> Self::Output {
        for (d, s) in self.data.iter_mut().zip(rhs.data.iter()) {
            *d = *d + *s;
        }
        self
    }
}

#[allow(clippy::arithmetic_side_effects)]
impl<T, const N: usize> Sub<Self> for StaticPolynomial<T, N>
where
    T: Sub<Output = T> + Copy,
{
    type Output = Self;

    #[inline]
    fn sub(mut self, rhs: Self) -> Self::Output {
        for (d, s) in self.data.iter_mut().zip(rhs.data.iter()) {
            *d = *d - *s;
        }
        self
    }
}

#[allow(clippy::arithmetic_side_effects)]
impl<T, const N: usize> Sub<&Self> for StaticPolynomial<T, N>
where
    T: Sub<Output = T> + Copy,
{
    type Output = Self;

    #[inline]
    fn sub(mut self, rhs: &Self) -> Self::Output {
        for (d, s) in self.data.iter_mut().zip(rhs.data.iter()) {
            *d = *d - *s;
        }
        self
    }
}
