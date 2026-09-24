//! Complex number representation and arithmetic operations.
//!
//! This module provides a generic `Complex` struct and basic arithmetic operations
//! (addition, subtraction, multiplication, division) for complex numbers.

use crate::math::{
    ArithmeticResult,
    num_traits::{AdditiveGroup, Conjugate, Float, One, Scalar, Zero},
    ops::{
        Add, Div, Mul, Neg, SaturatingAdd, SaturatingDiv, SaturatingMul,
        SaturatingNeg, SaturatingSub, Sub, TryAdd, TryDiv, TryMul, TrySub,
        WrappingAdd, WrappingMul, WrappingSub,
    },
};

/// Type alias for a complex number using single precision.
pub type Complex32 = Complex<f32>;
/// Type alias for a complex number using double precision.
pub type Complex64 = Complex<f64>;

/// A complex number consisting of a real and an imaginary part.
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
    pub fn from_polar(r: &T, theta: &T) -> Self {
        Self::new(
            r.saturating_mul(&theta.clone().cos()),
            r.saturating_mul(&theta.clone().sin()),
        )
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

impl<T: SaturatingAdd> Add<Self> for Complex<T> {
    type Output = Self;
    /// Component-wise sum with the component type's saturating semantics.
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            re: self.re.saturating_add(&rhs.re),
            im: self.im.saturating_add(&rhs.im),
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

impl<T: SaturatingAdd + SaturatingSub + SaturatingMul + SaturatingDiv> Div
    for Complex<T>
{
    type Output = Self;

    /// $(a + bi) / (c + di) = ((ac + bd) - (ad - bc)i) / (c^2 + d^2)$ with the
    /// component type's saturating semantics.
    fn div(self, rhs: Self) -> Self::Output {
        let denominator = rhs
            .re
            .saturating_mul(&rhs.re)
            .saturating_add(&rhs.im.saturating_mul(&rhs.im));
        Self {
            re: self
                .re
                .saturating_mul(&rhs.re)
                .saturating_add(&self.im.saturating_mul(&rhs.im))
                .saturating_div(&denominator),
            im: self
                .im
                .saturating_mul(&rhs.re)
                .saturating_sub(&self.re.saturating_mul(&rhs.im))
                .saturating_div(&denominator),
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

impl<T: SaturatingAdd + SaturatingSub + SaturatingMul> Mul for Complex<T> {
    type Output = Self;
    /// $(a + bi)(c + di) = (ac - bd) + (ad + bc)i$ with the component type's
    /// saturating semantics.
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            re: self
                .re
                .saturating_mul(&rhs.re)
                .saturating_sub(&self.im.saturating_mul(&rhs.im)),
            im: self
                .re
                .saturating_mul(&rhs.im)
                .saturating_add(&self.im.saturating_mul(&rhs.re)),
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

impl<T: SaturatingSub> Sub for Complex<T> {
    type Output = Self;
    /// Component-wise difference with the component type's saturating
    /// semantics.
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            re: self.re.saturating_sub(&rhs.re),
            im: self.im.saturating_sub(&rhs.im),
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

impl<T: SaturatingAdd + Clone> SaturatingAdd for Complex<T> {
    fn saturating_add(&self, v: &Self) -> Self {
        self.clone().add(v.clone())
    }
}

impl<T: SaturatingSub + Clone> SaturatingSub for Complex<T> {
    fn saturating_sub(&self, v: &Self) -> Self {
        self.clone().sub(v.clone())
    }
}

impl<T: SaturatingAdd + SaturatingSub + SaturatingMul + Clone> SaturatingMul
    for Complex<T>
{
    fn saturating_mul(&self, v: &Self) -> Self {
        self.clone().mul(v.clone())
    }
}

impl<T: SaturatingAdd + SaturatingSub + SaturatingMul + SaturatingDiv + Clone>
    SaturatingDiv for Complex<T>
{
    fn saturating_div(&self, v: &Self) -> Self {
        self.clone().div(v.clone())
    }
}

impl<T: SaturatingNeg + Clone> SaturatingNeg for Complex<T> {
    fn saturating_neg(&self) -> Self {
        Self::new(self.re.saturating_neg(), self.im.saturating_neg())
    }
}

////////////////////////////////////////////////////////////////////////////////

impl<T: WrappingAdd + SaturatingAdd> WrappingAdd for Complex<T> {
    fn wrapping_add(&self, v: &Self) -> Self {
        Self {
            re: self.re.wrapping_add(&v.re),
            im: self.im.wrapping_add(&v.im),
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

impl<
    T: WrappingAdd
        + WrappingSub
        + WrappingMul
        + SaturatingAdd
        + SaturatingSub
        + SaturatingMul
        + Copy,
> WrappingMul for Complex<T>
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

impl<T: WrappingSub + SaturatingSub> WrappingSub for Complex<T> {
    fn wrapping_sub(&self, v: &Self) -> Self {
        Self {
            re: self.re.wrapping_sub(&v.re),
            im: self.im.wrapping_sub(&v.im),
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

impl<T: SaturatingAdd + TryAdd<T, Output = T>> TryAdd for Complex<T> {
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
        + SaturatingAdd
        + SaturatingSub
        + SaturatingMul
        + SaturatingDiv
        + TryAdd<T, Output = T>
        + TrySub<T, Output = T>
        + TryMul<T, Output = T>
        + TryDiv<T, Output = T>,
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
        + SaturatingAdd
        + SaturatingSub
        + SaturatingMul
        + TryAdd<T, Output = T>
        + TrySub<T, Output = T>
        + TryMul<T, Output = T>,
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

impl<T: SaturatingSub + TrySub<T, Output = T>> TrySub for Complex<T> {
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

impl<T: SaturatingNeg> Conjugate for Complex<T> {
    #[inline(always)]
    fn conj(self) -> Self {
        let im = self.im.saturating_neg();
        Self::new(self.re, im)
    }
}

////////////////////////////////////////////////////////////////////////////////

impl<T> Scalar for Complex<T>
where
    T: Scalar<Real = T> + SaturatingNeg + PartialOrd,
{
    type Real = T;

    #[inline(always)]
    fn abs2(&self) -> Self::Real {
        self.re
            .clone()
            .saturating_mul(&self.re.clone())
            .saturating_add(&self.im.saturating_mul(&self.im.clone()))
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

impl<T: One + Zero + SaturatingAdd + SaturatingSub + SaturatingMul> One
    for Complex<T>
{
    const ONE: Self = Self {
        re: T::ONE,
        im: T::ZERO,
    };
}

////////////////////////////////////////////////////////////////////////////////

impl<T: Zero + SaturatingAdd> Zero for Complex<T> {
    const ZERO: Self = Self {
        re: T::ZERO,
        im: T::ZERO,
    };
}

////////////////////////////////////////////////////////////////////////////////

impl<T: AdditiveGroup + SaturatingAdd + SaturatingSub + SaturatingNeg>
    AdditiveGroup for Complex<T>
{
}

////////////////////////////////////////////////////////////////////////////////

impl<T: SaturatingNeg> Neg for Complex<T> {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(self.re.saturating_neg(), self.im.saturating_neg())
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
    pub fn acos(self) -> Self {
        let two = T::ONE.saturating_add(&T::ONE);
        Self::from_real(T::PI.saturating_div(&two)).saturating_sub(&self.asin())
    }

    /// Computes the inverse sine (arcsin) of the complex number.
    #[must_use]
    pub fn asin(self) -> Self {
        let i = Self::from_imag(T::ONE);
        let one = Self::from_real(T::ONE);
        let iz = i.saturating_mul(&self);
        let root = one.saturating_sub(&self.saturating_mul(&self)).sqrt();
        Self::from_imag(T::ZERO.saturating_sub(&T::ONE))
            .saturating_mul(&iz.saturating_add(&root).ln())
    }

    /// Computes the inverse tangent (arctan) of the complex number.
    #[must_use]
    pub fn atan(self) -> Self {
        let two = T::ONE.saturating_add(&T::ONE);
        let i = Self::from_imag(T::ONE);
        let half_i = Self::from_imag(T::ONE.saturating_div(&two));
        half_i.saturating_mul(
            &i.clone()
                .saturating_add(&self.clone())
                .saturating_div(&i.saturating_sub(&self))
                .ln(),
        )
    }

    /// Computes the cosine of the complex number.
    #[must_use]
    pub fn cos(self) -> Self {
        let (x, y) = (self.re, self.im);
        let (sinh_y, cosh_y) = (y.clone().sinh(), y.cosh());
        Self::new(
            x.clone().cos().saturating_mul(&cosh_y),
            T::ZERO.saturating_sub(&x.sin().saturating_mul(&sinh_y)),
        )
    }

    /// Returns the machine epsilon for complex numbers of this precision.
    #[must_use]
    pub fn epsilon() -> Self {
        Self::from_real(T::epsilon())
    }

    /// Computes $e^z$ for the complex number.
    #[must_use]
    pub fn exp(self) -> Self {
        let exp_re = self.re.exp();
        Self::new(
            exp_re.saturating_mul(&self.im.clone().cos()),
            exp_re.saturating_mul(&self.im.sin()),
        )
    }

    /// Computes the principal natural logarithm $\ln(z)$ of the complex number.
    #[must_use]
    pub fn ln(self) -> Self {
        Self::new(self.clone().magnitude().ln(), self.arg())
    }

    /// Computes the base-10 logarithm $\log_{10}(z)$ of the complex number.
    #[must_use]
    pub fn log10(self) -> Self {
        let ln10 = T::from_usize(10).ln();
        self.ln().saturating_div(&Self::from_real(ln10))
    }

    /// Computes $z^n$ for complex $z$ and complex $n$.
    #[must_use]
    pub fn pow(self, n: &Self) -> Self {
        if self.is_zero() {
            return if n.is_zero() {
                Self::one()
            } else {
                Self::zero()
            };
        }
        n.saturating_mul(&self.ln()).exp()
    }

    /// Computes the sine of the complex number.
    #[must_use]
    pub fn sin(self) -> Self {
        let (x, y) = (self.re, self.im);
        let sinh_y = y.clone().sinh();
        let cosh_y = y.cosh();
        Self::new(
            x.clone().sin().saturating_mul(&cosh_y),
            x.cos().saturating_mul(&sinh_y),
        )
    }

    /// Computes the principal square root of the complex number.
    #[must_use]
    pub fn sqrt(self) -> Self {
        if self.is_zero() {
            return Self::zero();
        }
        let two = T::ONE.saturating_add(&T::ONE);
        let r = self.clone().magnitude();
        let re = r
            .saturating_add(&self.re.clone().abs())
            .saturating_div(&two)
            .sqrt();
        let im = self
            .im
            .clone()
            .abs()
            .saturating_div(&two.saturating_mul(&re));

        if self.re >= T::ZERO {
            Self::new(
                re,
                if self.im >= T::ZERO {
                    im
                } else {
                    T::ZERO.saturating_sub(&im)
                },
            )
        } else {
            let sign = if self.im >= T::ZERO {
                T::ONE
            } else {
                T::ZERO.saturating_sub(&T::ONE)
            };
            Self::new(im, sign.saturating_mul(&re))
        }
    }

    /// Computes the tangent of the complex number.
    #[must_use]
    pub fn tan(self) -> Self {
        self.clone().sin().saturating_div(&self.cos())
    }
}
