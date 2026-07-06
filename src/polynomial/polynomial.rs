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
pub struct Polynomial<T, const N: usize> {
    pub(crate) data: [T; N],
}

// ==========================================
// Implementations (impls)
// ==========================================

impl<T, const N: usize> Polynomial<T, N> {
    /// Returns a slice of the polynomial coefficients in ascending order of powers.
    #[inline(always)]
    pub const fn coefficients(&self) -> &[T] {
        &self.data
    }

    /// Returns a mutable slice of the polynomial coefficients.
    #[inline(always)]
    pub const fn coefficients_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Returns the degree of the polynomial, or `None` if the polynomial is zero.
    #[inline]
    pub fn degree(&self) -> Option<usize>
    where
        T: Zero + Copy + PartialEq,
    {
        (0..N)
            .rev()
            .find(|&i| self.data.get(i).is_some_and(|val| !val.is_zero()))
    }

    /// Computes the derivative of this polynomial.
    ///
    /// The derivative of a polynomial of length N has length max(1, N - 1).
    ///
    /// # Examples
    /// ```
    /// use control_rs::polynomial::Polynomial;
    /// let p = Polynomial::from_coefficients([1.0, 2.0, 3.0]); // 1 + 2x + 3x^2
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

    /// Performs polynomial long division, returning the quotient and remainder.
    ///
    /// # Panics
    /// Panics if the divisor is zero (has all zero coefficients) or if the output sizes
    /// `Q` and `R` are insufficient to hold the quotient and remainder.
    #[allow(
        clippy::panic,
        clippy::indexing_slicing,
        clippy::arithmetic_side_effects,
        clippy::type_complexity
    )]
    pub fn div_rem<const M: usize, const Q: usize, const R: usize>(
        &self,
        divisor: &Polynomial<T, M>,
    ) -> (Polynomial<T, Q>, Polynomial<T, R>)
    where
        T: crate::math::num_traits::Field + Copy + PartialEq + PartialOrd,
    {
        let Some(divisor_deg) = divisor.degree() else {
            panic!("Division by zero polynomial");
        };
        let divisor_lead = divisor.data[divisor_deg];

        let mut rem_data = [T::ZERO; N];
        rem_data.copy_from_slice(&self.data);

        let mut quot_data = [T::ZERO; Q];

        let mut rem_deg_opt = (0..N).rev().find(|&i| !rem_data[i].is_zero());

        while let Some(rem_deg) = rem_deg_opt {
            if rem_deg < divisor_deg {
                break;
            }

            let deg_diff = rem_deg - divisor_deg;
            let coeff = rem_data[rem_deg] / divisor_lead;

            assert!(
                deg_diff < Q,
                "Quotient polynomial size Q={} is too small; need at least {}",
                Q,
                deg_diff + 1
            );

            quot_data[deg_diff] = quot_data[deg_diff] + coeff;

            for i in 0..=divisor_deg {
                let target_idx = i + deg_diff;
                if target_idx < N {
                    rem_data[target_idx] =
                        rem_data[target_idx] - coeff * divisor.data[i];
                }
            }

            rem_deg_opt = (0..N).rev().find(|&i| !rem_data[i].is_zero());
        }

        let mut final_rem = [T::ZERO; R];
        if let Some(final_rem_deg) = rem_deg_opt {
            assert!(
                final_rem_deg < R,
                "Remainder polynomial size R={} is too small; need at least {}",
                R,
                final_rem_deg + 1
            );
            final_rem[..=final_rem_deg]
                .copy_from_slice(&rem_data[..=final_rem_deg]);
        }

        (
            Polynomial::from_coefficients(quot_data),
            Polynomial::from_coefficients(final_rem),
        )
    }

    /// Evaluates the polynomial at the given point `x` using Horner's method subprogram.
    #[inline]
    pub fn evaluate(&self, x: T) -> T
    where
        T: Zero + Add<Output = T> + Mul<Output = T> + Copy,
    {
        BasicSubPrograms::polyeval(&self.data, x)
    }

    /// Create a new polynomial from coefficients in ascending order.
    ///
    /// # Examples
    /// ```
    /// use control_rs::polynomial::Polynomial;
    /// let p = Polynomial::from_coefficients([1.0, 2.0, 3.0]); // 1 + 2x + 3x^2
    /// assert_eq!(p.coefficients(), &[1.0, 2.0, 3.0]);
    /// ```
    #[inline(always)]
    pub const fn from_coefficients(data: [T; N]) -> Self {
        Self { data }
    }

    /// Create a new polynomial from coefficients in descending order of powers.
    ///
    /// # Examples
    /// ```
    /// use control_rs::polynomial::Polynomial;
    /// let p = Polynomial::from_descending([3.0, 2.0, 1.0]); // 3x^2 + 2x + 1
    /// assert_eq!(p.coefficients(), &[1.0, 2.0, 3.0]);
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

    /// Returns a reference to the leading coefficient (highest power term).
    #[inline]
    pub fn leading_coefficient(&self) -> Option<&T>
    where
        T: Zero + Copy + PartialEq,
    {
        let deg = self.degree()?;
        self.data.get(deg)
    }

    /// Returns a mutable reference to the leading coefficient.
    #[inline]
    pub fn leading_coefficient_mut(&mut self) -> Option<&mut T>
    where
        T: Zero + Copy + PartialEq,
    {
        let deg = self.degree()?;
        self.data.get_mut(deg)
    }

    /// Multiplies two polynomials, returning a new polynomial of size OUT.
    ///
    /// # Examples
    /// ```
    /// use control_rs::polynomial::Polynomial;
    /// let p1 = Polynomial::from_coefficients([1.0, 2.0]);
    /// let p2 = Polynomial::from_coefficients([3.0, 4.0]);
    /// let p3: Polynomial<f64, 3> = p1.mul_poly(&p2);
    /// assert_eq!(p3.coefficients(), &[3.0, 10.0, 8.0]);
    /// ```
    #[inline]
    pub fn mul_poly<const M: usize, const OUT: usize, Sum>(
        &self,
        rhs: &Polynomial<T, M>,
    ) -> Polynomial<T, OUT>
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
        rhs: &Polynomial<T, M>,
    ) -> Polynomial<T, OUT>
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
        Polynomial::from_coefficients(data)
    }

    /// Pads the polynomial coefficients to a statically specified size M.
    /// If M is smaller than N, it will truncate the coefficients.
    #[inline]
    pub fn pad<const M: usize>(&self) -> Polynomial<T, M>
    where
        T: Zero + Copy,
    {
        let mut padded_data = [T::ZERO; M];
        let copy_len = if N < M { N } else { M };
        padded_data[..copy_len].copy_from_slice(&self.data[..copy_len]);
        Polynomial::from_coefficients(padded_data)
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

impl<T, const N: usize> AddAssign<&Self> for Polynomial<T, N>
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

impl<T, const N: usize> AddAssign<Self> for Polynomial<T, N>
where
    T: Ring,
{
    /// Performs element-wise polynomial addition in-place using standard BLAS AXPY subprograms.
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.add_assign(&rhs);
    }
}

impl<T, const N: usize> SubAssign<&Self> for Polynomial<T, N>
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

impl<T, const N: usize> SubAssign<Self> for Polynomial<T, N>
where
    T: Ring + crate::math::ops::Neg<Output = T>,
{
    /// Performs element-wise polynomial subtraction in-place using standard BLAS AXPY subprograms.
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.sub_assign(&rhs);
    }
}

impl<T, const N: usize> Add<Self> for Polynomial<T, N>
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

impl<T, const N: usize> Add<&Self> for Polynomial<T, N>
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

impl<T, const N: usize> Sub<Self> for Polynomial<T, N>
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

impl<T, const N: usize> Sub<&Self> for Polynomial<T, N>
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

impl<T, const N: usize> Mul<T> for Polynomial<T, N>
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

impl<T, const N: usize> MulAssign<T> for Polynomial<T, N>
where
    T: Ring + Copy,
{
    /// Scales the polynomial by a scalar factor in-place using SCAL subprogram.
    #[inline]
    fn mul_assign(&mut self, rhs: T) {
        use crate::math::subprograms::level1::SCAL;
        BasicSubPrograms::scal(rhs, &mut self.data);
    }
}

impl<T, const N: usize> Div<T> for Polynomial<T, N>
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

impl<T, const N: usize> DivAssign<T> for Polynomial<T, N>
where
    T: crate::math::num_traits::Field + Copy,
{
    /// Scales the polynomial by a scalar divisor in-place using SCAL subprogram with reciprocal.
    #[allow(clippy::arithmetic_side_effects)]
    #[inline]
    fn div_assign(&mut self, rhs: T) {
        use crate::math::subprograms::level1::SCAL;
        BasicSubPrograms::scal(T::one() / rhs, &mut self.data);
    }
}
