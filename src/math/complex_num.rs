//! Complex number representation and arithmetic operations.
//!
//! This module provides a generic `Complex` struct and basic arithmetic operations
//! (addition, subtraction, multiplication, division) for complex numbers.

use crate::math::{
    ArithmeticResult,
    num_traits::{
        AdditiveGroup, Exponential, Float, One, Radical, Scalar, Signed, Trig,
        Zero,
    },
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

////////////////////////////////////////////////////////////////////////////////

impl<T: Zero> Complex<T> {
    /// Creates a new complex number from the imaginary part, with the real set to zero.
    ///
    /// # Arguments
    /// * `im` - The imaginary component.
    #[must_use]
    pub fn from_imag(im: T) -> Self {
        Self::new(T::zero(), im)
    }

    /// Creates a complex number from the real part, with the imaginary part set to zero.
    ///
    /// # Arguments
    /// * `re` - The real component.
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

    /// Creates a new complex number from polar coordinates.
    ///
    /// # Arguments
    /// * `r` - magnitude of the number.
    /// * `theta` - phase of the number.
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

    /// Creates a pair of polar coordinates from self.
    ///
    /// # Returns
    /// * `r` - magnitude of the number.
    /// * `theta` - phase of the number.
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

impl<T: Scalar> Scalar for Complex<T> {}

////////////////////////////////////////////////////////////////////////////////

impl<T: PartialOrd> PartialOrd for Complex<T> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        match self.re.partial_cmp(&other.re) {
            Some(core::cmp::Ordering::Equal) => self.im.partial_cmp(&other.im),
            ord => ord,
        }
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

impl<T: AdditiveGroup> AdditiveGroup for Complex<T> {}

////////////////////////////////////////////////////////////////////////////////

impl<T: Float> Float for Complex<T> {
    fn epsilon() -> Self {
        Self::from_real(T::epsilon())
    }
}

////////////////////////////////////////////////////////////////////////////////

impl<T: Neg<Output = T>> Neg for Complex<T> {
    type Output = Self;
    #[allow(clippy::arithmetic_side_effects)]
    fn neg(self) -> Self {
        Self::new(-self.re, -self.im)
    }
}

////////////////////////////////////////////////////////////////////////////////

impl<T> Signed for Complex<T>
where
    T: Signed,
{
    fn abs(self) -> Self {
        Self::new(self.re.abs(), self.im.abs())
    }
}

////////////////////////////////////////////////////////////////////////////////

impl<T: Float> Radical for Complex<T> {
    #[allow(clippy::arithmetic_side_effects)]
    fn sqrt(self) -> Self {
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
}

////////////////////////////////////////////////////////////////////////////////

impl<T: Float> Trig for Complex<T> {
    const PI: Self = Self {
        re: T::PI,
        im: T::ZERO,
    };
    #[allow(clippy::arithmetic_side_effects)]
    fn acos(self) -> Self {
        let two = T::ONE + T::ONE;
        Self::from_real(T::PI / two) - self.asin()
    }
    #[allow(clippy::arithmetic_side_effects)]
    fn asin(self) -> Self {
        // asin(z) = -i * ln(iz + sqrt(1 - z^2))
        let i = Self::from_imag(T::ONE);
        let one = Self::from_real(T::ONE);
        let iz = i * self.clone();
        let root = (one - (self.clone() * self)).sqrt();
        Self::from_imag(T::ZERO - T::ONE) * (iz + root).ln()
    }
    #[allow(clippy::arithmetic_side_effects)]
    fn atan(self) -> Self {
        // atan(z) = i/2 * ln((i+z)/(i-z))
        let two = T::ONE + T::ONE;
        let i = Self::from_imag(T::ONE);
        let half_i = Self::from_imag(T::ONE / two);
        half_i * ((i.clone() + self.clone()) / (i - self)).ln()
    }
    #[allow(clippy::arithmetic_side_effects)]
    fn cos(self) -> Self {
        // cos(z) = cos(x)cosh(y) - i sin(x)sinh(y)
        let (x, y) = (self.re, self.im);
        let (sinh_y, cosh_y) = (y.clone().sinh(), y.cosh());
        Self::new(x.clone().cos() * cosh_y, T::ZERO - (x.sin() * sinh_y))
    }
    #[allow(clippy::arithmetic_side_effects)]
    fn sin(self) -> Self {
        // sin(z) = sin(x)cosh(y) + i cos(x)sinh(y)
        let (x, y) = (self.re, self.im);
        let sinh_y = y.clone().sinh();
        let cosh_y = y.cosh();
        Self::new(x.clone().sin() * cosh_y, x.cos() * sinh_y)
    }
    #[allow(clippy::arithmetic_side_effects)]
    fn tan(self) -> Self {
        self.clone().sin() / self.cos()
    }
}

////////////////////////////////////////////////////////////////////////////////

impl<T: Float> Exponential for Complex<T> {
    const E: Self = Self {
        re: T::E,
        im: T::ZERO,
    };

    #[allow(clippy::arithmetic_side_effects)]
    fn exp(self) -> Self {
        let exp_re = self.re.exp();
        Self::new(
            exp_re.clone() * self.im.clone().cos(),
            exp_re * self.im.sin(),
        )
    }

    fn ln(self) -> Self {
        // ln(z) = ln|z| + i arg(z)
        Self::new(self.clone().magnitude().ln(), self.arg())
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn log10(self) -> Self {
        let ln10 = T::from_usize(10).ln();
        self.ln() / Self::from_real(ln10)
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn pow(self, n: Self) -> Self {
        if self.is_zero() {
            return if n.is_zero() {
                Self::one()
            } else {
                Self::zero()
            };
        }
        (n * self.ln()).exp()
    }
}
