//! Complex number representation and arithmetic operations.
//!
//! This module provides a generic `Complex` struct and basic arithmetic operations
//! (addition, subtraction, multiplication, division) for complex numbers.

use crate::math::{
    ArithmeticResult,
    num_traits::{AdditiveGroup, Conjugate, Float, One, Scalar, Zero},
    ops::{
        Add, Div, Mul, Neg, Sub, TryAdd, TryDiv, TryMul, TrySub, WrappingAdd,
        WrappingMul, WrappingSub,
    },
};

/// Tye alias for a complex number using single precision.
pub type Complex32 = Complex<f32>;
/// Tye alias for a complex number using double precision.
pub type Complex64 = Complex<f64>;

/// A complex number consisting of a real and an imaginary part.
#[allow(clippy::arbitrary_source_item_ordering)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(C)]
pub struct Complex<T> {
    /// The real part of the complex number.
    pub re: T,
    /// The imaginary part of the complex number.
    pub im: T,
}

////////////////////////////////////////////////////////////////////////////////

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
    #[must_use]
    pub const fn new(re: T, im: T) -> Self {
        Self { re, im }
    }
}

////////////////////////////////////////////////////////////////////////////////

impl<T: Zero> Complex<T> {
    /// Creates a new complex number from the imaginary part, with the real set to zero.
    #[must_use]
    pub fn from_imag(im: T) -> Self {
        Self::new(T::zero(), im)
    }

    /// Creates a complex number from the real part, with the imaginary part set to zero.
    #[must_use]
    pub fn from_real(re: T) -> Self {
        Self::new(re, T::zero())
    }
}

////////////////////////////////////////////////////////////////////////////////

impl<T: Float> Complex<T> {
    /// Computes the principal Arg of self.
    #[inline]
    pub fn arg(self) -> T {
        self.im.atan2(self.re)
    }

    /// Creates a new complex number from polar coordinates (`r`, `theta`).
    #[must_use]
    #[allow(clippy::arithmetic_side_effects)]
    pub fn from_polar(r: T, theta: T) -> Self {
        Self::new(r.clone() * theta.clone().cos(), r * theta.sin())
    }

    /// Computes the distance from the origin to self.
    #[inline]
    pub fn magnitude(self) -> T {
        self.re.hypot(self.im)
    }

    /// Creates a pair of polar coordinates `(r, theta)` from self.
    #[must_use]
    pub fn to_polar(self) -> (T, T) {
        (self.clone().magnitude(), self.arg())
    }
}

////////////////////////////////////////////////////////////////////////////////

impl<T: Add<Output = T>> Add<Self> for Complex<T> {
    type Output = Self;
    #[allow(clippy::arithmetic_side_effects)]
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

impl<
    T: Mul<Output = T>
        + Add<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + Clone,
> Div for Complex<T>
{
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects)]
    fn div(self, rhs: Self) -> Self::Output {
        let denominator = (rhs.re.clone() * rhs.re.clone())
            + (rhs.im.clone() * rhs.im.clone());
        Self {
            re: ((self.re.clone() * rhs.re.clone())
                + (self.im.clone() * rhs.im.clone()))
                / denominator.clone(),
            im: ((self.im * rhs.re) - (self.re * rhs.im)) / denominator,
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

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

////////////////////////////////////////////////////////////////////////////////

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

////////////////////////////////////////////////////////////////////////////////

impl<T: WrappingAdd> WrappingAdd for Complex<T> {
    fn wrapping_add(&self, v: &Self) -> Self {
        Self {
            re: self.re.wrapping_add(&v.re),
            im: self.im.wrapping_add(&v.im),
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

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

////////////////////////////////////////////////////////////////////////////////

impl<T: WrappingSub> WrappingSub for Complex<T> {
    fn wrapping_sub(&self, v: &Self) -> Self {
        Self {
            re: self.re.wrapping_sub(&v.re),
            im: self.im.wrapping_sub(&v.im),
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

impl<T: Add<T, Output = T> + TryAdd<T>> TryAdd for Complex<T> {
    fn try_add(&self, v: &Self) -> ArithmeticResult<Self::Output> {
        Ok(Self {
            re: self.re.try_add(&v.re)?,
            im: self.im.try_add(&v.im)?,
        })
    }
}

////////////////////////////////////////////////////////////////////////////////

impl<T> TryDiv for Complex<T>
where
    T: Clone
        + Add<T, Output = T>
        + Sub<T, Output = T>
        + Mul<T, Output = T>
        + Div<T, Output = T>
        + TryAdd<T>
        + TrySub<T>
        + TryMul<T>
        + TryDiv<T>,
{
    fn try_div(&self, v: &Self) -> ArithmeticResult<Self::Output> {
        let denominator =
            v.re.try_mul(&v.re)?.try_add(&v.im.try_mul(&v.im)?)?;
        Ok(Self {
            re: self
                .re
                .try_mul(&v.re)?
                .try_add(&self.im.try_mul(&v.im)?)?
                .try_div(&denominator)?,
            im: self
                .im
                .try_mul(&v.re)?
                .try_sub(&self.re.try_mul(&v.im)?)?
                .try_div(&denominator)?,
        })
    }
}

////////////////////////////////////////////////////////////////////////////////

impl<
    T: Clone
        + Add<T, Output = T>
        + Sub<T, Output = T>
        + Mul<T, Output = T>
        + TryAdd<T>
        + TrySub<T>
        + TryMul<T>,
> TryMul for Complex<T>
{
    fn try_mul(&self, v: &Self) -> ArithmeticResult<Self::Output> {
        Ok(Self {
            re: self.re.try_mul(&v.re)?.try_sub(&self.im.try_mul(&v.im)?)?,
            im: self.re.try_mul(&v.im)?.try_add(&self.im.try_mul(&v.re)?)?,
        })
    }
}

////////////////////////////////////////////////////////////////////////////////

impl<T: Sub<T, Output = T> + TrySub<T, Output = T>> TrySub for Complex<T> {
    fn try_sub(&self, v: &Self) -> ArithmeticResult<Self::Output> {
        Ok(Self {
            re: self.re.try_sub(&v.re)?,
            im: self.im.try_sub(&v.im)?,
        })
    }
}

////////////////////////////////////////////////////////////////////////////////
// Numerical traits for complex numbers
////////////////////////////////////////////////////////////////////////////////

impl<T: Neg<Output = T>> Conjugate for Complex<T> {
    #[allow(clippy::arithmetic_side_effects)]
    #[inline(always)]
    fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }
}

////////////////////////////////////////////////////////////////////////////////

impl<T> Scalar for Complex<T>
where
    T: Scalar<Real = T> + Neg<Output = T> + PartialOrd,
{
    type Real = T;

    #[allow(clippy::arithmetic_side_effects)]
    #[inline(always)]
    fn abs2(&self) -> Self::Real {
        self.re.clone() * self.re.clone() + self.im.clone() * self.im.clone()
    }

    #[inline(always)]
    fn from_real(re: Self::Real) -> Self {
        Self::new(re, T::ZERO)
    }

    #[inline(always)]
    fn im(&self) -> Self::Real {
        self.im.clone()
    }

    #[inline(always)]
    fn re(&self) -> Self::Real {
        self.re.clone()
    }
}

////////////////////////////////////////////////////////////////////////////////

impl<T: One + Zero + Sub<Output = T>> One for Complex<T> {
    const ONE: Self = Self {
        re: T::ONE,
        im: T::ZERO,
    };
}

////////////////////////////////////////////////////////////////////////////////

impl<T: Zero> Zero for Complex<T> {
    const ZERO: Self = Self {
        re: T::ZERO,
        im: T::ZERO,
    };
}

////////////////////////////////////////////////////////////////////////////////

impl<T: AdditiveGroup + Neg<Output = T>> AdditiveGroup for Complex<T> {}

////////////////////////////////////////////////////////////////////////////////

impl<T: Neg<Output = T>> Neg for Complex<T> {
    type Output = Self;
    #[allow(clippy::arithmetic_side_effects)]
    fn neg(self) -> Self {
        Self::new(-self.re, -self.im)
    }
}

////////////////////////////////////////////////////////////////////////////////
// Inherent Analytic & Transcendental Operations
////////////////////////////////////////////////////////////////////////////////

impl<T: Float> Complex<T> {
    /// The complex absolute value: `|z|` as a purely real number.
    #[must_use]
    pub fn abs(self) -> T {
        self.magnitude()
    }

    /// Computes the inverse cosine (arccos) of the complex number.
    #[must_use]
    #[allow(clippy::arithmetic_side_effects)]
    pub fn acos(self) -> Self {
        let two = T::ONE + T::ONE;
        Self::from_real(T::PI / two) - self.asin()
    }

    /// Computes the inverse sine (arcsin) of the complex number.
    #[must_use]
    #[allow(clippy::arithmetic_side_effects)]
    pub fn asin(self) -> Self {
        let i = Self::from_imag(T::ONE);
        let one = Self::from_real(T::ONE);
        let iz = i * self.clone();
        let root = (one - (self.clone() * self)).sqrt();
        Self::from_imag(T::ZERO - T::ONE) * (iz + root).ln()
    }

    /// Computes the inverse tangent (arctan) of the complex number.
    #[must_use]
    #[allow(clippy::arithmetic_side_effects)]
    pub fn atan(self) -> Self {
        let two = T::ONE + T::ONE;
        let i = Self::from_imag(T::ONE);
        let half_i = Self::from_imag(T::ONE / two);
        half_i * ((i.clone() + self.clone()) / (i - self)).ln()
    }

    /// Computes the cosine of the complex number.
    #[must_use]
    #[allow(clippy::arithmetic_side_effects)]
    pub fn cos(self) -> Self {
        let (x, y) = (self.re, self.im);
        let (sinh_y, cosh_y) = (y.clone().sinh(), y.cosh());
        Self::new(x.clone().cos() * cosh_y, T::ZERO - (x.sin() * sinh_y))
    }

    /// Returns the machine epsilon for complex numbers of this precision.
    #[must_use]
    pub fn epsilon() -> Self {
        Self::from_real(T::epsilon())
    }

    /// Computes $e^z$ for the complex number.
    #[must_use]
    #[allow(clippy::arithmetic_side_effects)]
    pub fn exp(self) -> Self {
        let exp_re = self.re.exp();
        Self::new(
            exp_re.clone() * self.im.clone().cos(),
            exp_re * self.im.sin(),
        )
    }

    /// Computes the principal natural logarithm $\ln(z)$ of the complex number.
    #[must_use]
    pub fn ln(self) -> Self {
        Self::new(self.clone().magnitude().ln(), self.arg())
    }

    /// Computes the base-10 logarithm $\log_{10}(z)$ of the complex number.
    #[must_use]
    #[allow(clippy::arithmetic_side_effects)]
    pub fn log10(self) -> Self {
        let ln10 = T::from_usize(10).ln();
        self.ln() / Self::from_real(ln10)
    }

    /// Computes $z^n$ for complex $z$ and complex $n$.
    #[must_use]
    #[allow(clippy::arithmetic_side_effects)]
    pub fn pow(self, n: Self) -> Self {
        if self.is_zero() {
            return if n.is_zero() {
                Self::one()
            } else {
                Self::zero()
            };
        }
        (n * self.ln()).exp()
    }

    /// Computes the sine of the complex number.
    #[must_use]
    #[allow(clippy::arithmetic_side_effects)]
    pub fn sin(self) -> Self {
        let (x, y) = (self.re, self.im);
        let sinh_y = y.clone().sinh();
        let cosh_y = y.cosh();
        Self::new(x.clone().sin() * cosh_y, x.cos() * sinh_y)
    }

    /// Computes the principal square root of the complex number.
    #[must_use]
    #[allow(clippy::arithmetic_side_effects)]
    pub fn sqrt(self) -> Self {
        if self.is_zero() {
            return Self::zero();
        }
        let two = T::ONE + T::ONE;
        let r = self.clone().magnitude();
        let re = ((r + self.re.clone().abs()) / two.clone()).sqrt();
        let im = self.im.clone().abs() / (two * re.clone());

        if self.re >= T::ZERO {
            Self::new(re, if self.im >= T::ZERO { im } else { T::ZERO - im })
        } else {
            let sign = if self.im >= T::ZERO {
                T::ONE
            } else {
                T::ZERO - T::ONE
            };
            Self::new(im, sign * re)
        }
    }

    /// Computes the tangent of the complex number.
    #[must_use]
    #[allow(clippy::arithmetic_side_effects)]
    pub fn tan(self) -> Self {
        self.clone().sin() / self.cos()
    }
}
