//! # Numeric Traits
//!
//! This module defines a hierarchy of traits for numerical types, organized
//! around **hardware behavior** rather than abstract algebra. Each tier
//! reflects a physical ALU/FPU capability, giving generic control algorithms
//! exact compile-time boundaries on overflow behavior (wrapping vs.
//! saturating) instead of an abstract ring/field taxonomy.
//!
//! The hierarchy is as follows:
//! - `Zero` / `One`: Additive and multiplicative identity elements.
//! - `AdditiveGroup`: Opt-in subtraction, granted only where `a - b` is total
//!   and non-panicking (signed integers, floats, `Complex<T>`).
//! - `Integer` / `SaturatingInteger`: Hardware wrap/saturate ALU behavior,
//!   implemented by every integer primitive (signed and unsigned).
//! - `Unsigned`: A `Sized`-only marker distinguishing unsigned primitives.
//! - `Signed`: Sign predicates and absolute value.
//! - `Radical` / `Exponential` / `Trig`: Granular analytic capabilities.
//! - `Float`: The floating-point aggregate
//!   (`Signed + Radical + Exponential + Trig + Div`), the only tier where
//!   division and machine epsilon live.
//! - `Scalar`: The unified target for control-loop arithmetic
//!   (`AdditiveGroup + Signed + Mul`), implemented by signed integers and
//!   floats, deliberately excluding `Div` (integer division is not total).
//!
//! # Compile-Time Marker Boundary
//!
//! `AdditiveGroup` and `Scalar` are withheld from unsigned primitives —
//! `core::ops::Sub` is total for them, but the *semantic* guarantee
//! `AdditiveGroup` makes (`a - b` never underflows) is not:
//!
//! ```compile_fail
//! use control_rs::math::num_traits::Scalar;
//!
//! fn assert_scalar<T: Scalar>() {}
//! assert_scalar::<u32>(); // u32 does not implement Scalar
//! ```

use crate::math::CartesianQuadrant2D;
use crate::math::ops::{
    Add, Div, Mul, Neg, SaturatingAdd, SaturatingMul, SaturatingSub, Sub,
    WrappingAdd, WrappingMul, WrappingSub,
};

////////////////////////////////////////////////////////////////////////////////
// Identity Tier
////////////////////////////////////////////////////////////////////////////////

/// Provides access to the additive identity and a zero check.
///
/// # Safety
/// `ZERO` must be a true additive identity.
///
/// # Example
/// ```
/// use control_rs::math::num_traits::Zero;
///
/// assert!(0i32.is_zero());
/// assert!(!1i32.is_zero());
/// ```
pub trait Zero: Clone + PartialEq + PartialOrd + Add<Output = Self> {
    /// Constant additive identity element.
    const ZERO: Self;
    /// Returns `true` if the value equals the additive identity.
    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.eq(&Self::ZERO)
    }
    /// Returns the additive identity element.
    #[must_use]
    fn zero() -> Self {
        Self::ZERO
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Provides access to the multiplicative identity and a one check.
///
/// # Safety
/// `ONE` must be a true multiplicative identity.
///
/// # Example
/// ```
/// use control_rs::math::num_traits::One;
///
/// assert!(1i32.is_one());
/// assert!(!0i32.is_one());
/// ```
pub trait One: Clone + PartialEq + PartialOrd + Mul<Output = Self> {
    /// Constant multiplicative identity element.
    const ONE: Self;
    /// Returns `true` if the value equals the multiplicative identity.
    #[inline(always)]
    fn is_one(&self) -> bool {
        self.eq(&Self::ONE)
    }
    /// Returns the multiplicative identity element.
    #[must_use]
    fn one() -> Self {
        Self::ONE
    }
}

////////////////////////////////////////////////////////////////////////////////
// Subtraction Tier
////////////////////////////////////////////////////////////////////////////////

/// Opt-in trait binding `Zero` and total, non-panicking subtraction.
///
/// This is the single explicit grant of subtraction for types where
/// `a - b` is total (signed integers, floats, `Complex<T>`). Unsigned
/// primitives never implement it, unlike `core::ops::Sub`, which they do
/// implement but which panics/wraps on underflow.
///
/// # Safety
/// Implementors must guarantee `a - b` never panics and is well-defined for
/// all representable `a`, `b`.
pub trait AdditiveGroup: Zero + Sub<Output = Self> {}

////////////////////////////////////////////////////////////////////////////////
// Hardware Integer Tier
////////////////////////////////////////////////////////////////////////////////

/// Hardware wrapping-ALU integer behavior.
///
/// Implemented by every integer primitive, signed and unsigned.
///
/// # Example
/// ```
/// use control_rs::math::num_traits::Integer;
/// use control_rs::math::ops::WrappingAdd;
///
/// // Explicit trait syntax: `u8` also has an inherent `wrapping_add`
/// // (by-value), which dot-call syntax would prefer over this trait method.
/// assert_eq!(WrappingAdd::wrapping_add(&u8::MAX, &1), 0);
/// ```
pub trait Integer:
    Zero + One + WrappingAdd + WrappingSub + WrappingMul
{
    /// Constant representing the max representable value.
    const MAX: Self;
    /// Constant representing the min representable value.
    const MIN: Self;
    /// Constant representing the minimum positive value.
    const MIN_POSITIVE: Self;
    /// Constant representing 2.
    const TWO: Self;
}

////////////////////////////////////////////////////////////////////////////////

/// Hardware saturating-ALU integer behavior.
///
/// Implemented by every integer primitive, signed and unsigned.
///
/// # Example
/// ```
/// use control_rs::math::num_traits::SaturatingInteger;
/// use control_rs::math::ops::SaturatingAdd;
///
/// // Explicit trait syntax: `u8` also has an inherent `saturating_add`
/// // (by-value), which dot-call syntax would prefer over this trait method.
/// assert_eq!(SaturatingAdd::saturating_add(&u8::MAX, &1), u8::MAX);
/// ```
pub trait SaturatingInteger:
    Zero + One + SaturatingAdd + SaturatingSub + SaturatingMul
{
    /// Constant representing the max representable value.
    const MAX: Self;
    /// Constant representing the min representable value.
    const MIN: Self;
    /// Constant representing the minimum positive value.
    const MIN_POSITIVE: Self;
    /// Constant representing 2.
    const TWO: Self;
}

////////////////////////////////////////////////////////////////////////////////

/// Defines the set of unsigned numbers.
///
/// A `Sized`-only marker distinguishing unsigned primitives from
/// `Scalar`-eligible signed types; unsigned primitives never implement
/// `AdditiveGroup`/`Signed`/`Scalar`.
pub trait Unsigned: Sized {}

////////////////////////////////////////////////////////////////////////////////
// Signed & Analytic Tier
////////////////////////////////////////////////////////////////////////////////

/// Defines a set with a mathematical sign (signed integers and
/// floating/fixed-point types).
///
/// # Safety
/// Values that implement `Neg` have a well-defined sign.
pub trait Signed: AdditiveGroup + Neg<Output = Self> {
    /// Returns the absolute value.
    #[must_use]
    fn abs(self) -> Self;
    /// Check if self is less than zero.
    #[inline(always)]
    fn is_sign_negative(&self) -> bool {
        self.lt(&Self::ZERO)
    }
    /// Check if self is greater than zero.
    #[inline(always)]
    fn is_sign_positive(&self) -> bool {
        self.gt(&Self::ZERO)
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Trait for types that support square root (radical functions).
///
/// # Panics
/// Square root of a negative number must return an imaginary number or
/// a type that indicates the domain was violated. Thus, the domain for
/// this implementation includes all Self.
pub trait Radical:
    Clone + PartialEq + PartialOrd + Add<Output = Self> + Mul<Output = Self>
{
    /// Computes the hypotenuse of a triangle with the given side lengths.
    #[must_use]
    #[allow(clippy::arithmetic_side_effects)]
    fn hypot(self, rhs: Self) -> Self {
        ((self.clone() * self) + (rhs.clone() * rhs)).sqrt()
    }
    /// Computes the square root of a number.
    #[must_use]
    fn sqrt(self) -> Self;
}

////////////////////////////////////////////////////////////////////////////////

/// Trait for types that support exponential, power and root functions.
///
/// # Panics
/// Exponential of a negative number must return an imaginary number or
/// a type that indicates the domain was violated. Thus, the domain for
/// this implementation includes all Self.
pub trait Exponential: Clone + PartialEq + PartialOrd {
    /// Constant representing Euler's number.
    const E: Self;
    /// Computes `e^self`.
    #[must_use]
    fn exp(self) -> Self;
    /// Computes the natural logarithm.
    #[must_use]
    fn ln(self) -> Self;
    /// Computes the base-10 logarithm.
    #[must_use]
    fn log10(self) -> Self;
    /// Computes a number raised to a power.
    #[must_use]
    fn pow(self, n: Self) -> Self;
}

////////////////////////////////////////////////////////////////////////////////

/// Trait for types that support trigonometric functions.
///
/// # Panics
/// Trigonometric functions of an invalid number must return a type that
/// indicates the domain was violated. Thus, the domain for these implementations
/// includes all Self.
pub trait Trig: Clone + PartialEq + PartialOrd {
    /// Constant representing Pi.
    const PI: Self;
    /// Calculates the inverse cosine of a number (in radians).
    #[must_use]
    fn acos(self) -> Self;
    /// Calculates the inverse sine of a number (in radians).
    #[must_use]
    fn asin(self) -> Self;
    /// Calculates the inverse tangent of a number (in radians).
    #[must_use]
    fn atan(self) -> Self;
    /// Calculates the cosine of a number (in radians).
    #[must_use]
    fn cos(self) -> Self;
    /// Calculates the sine of a number (in radians).
    #[must_use]
    fn sin(self) -> Self;
    /// Calculates the tangent of a number (in radians).
    #[must_use]
    fn tan(self) -> Self;
}

////////////////////////////////////////////////////////////////////////////////

/// The floating-point aggregate: `Signed + Radical + Exponential + Trig`
/// plus division and machine epsilon, scoped to floating-point types only.
///
/// The trait implies and requires the existence of a machine epsilon;
/// `1.0 + ε != 1.0`.
///
/// # Panics
/// Floating-point division by zero does not panic but produces non-finite
/// values (`inf` or `NaN`).
///
/// # Example
/// ```
/// use control_rs::math::num_traits::Float;
///
/// fn is_significantly_different<T: Float>(a: T, b: T) -> bool {
///     let diff = if a > b { a - b } else { b - a };
///     diff > T::epsilon()
/// }
///
/// assert!(is_significantly_different(1.001f32, 1.0f32));
/// assert!(!is_significantly_different(1.00000001f32, 1.0f32));
/// ```
pub trait Float:
    Signed + One + Radical + Exponential + Trig + Div<Output = Self>
{
    /// Computes the four-quadrant inverse tangent of `y` and `x` in radians.
    ///
    /// Unlike the standard `atan(y / x)`, this function uses the signs of both
    /// arguments to identify the correct quadrant and handles the case where `x` is zero.
    ///
    /// # Mathematical Definition
    /// The returned value `θ` is in the range `(-π, π]` such that:
    /// - `θ > 0` if `y > 0` (Upper half-plane)
    /// - `θ < 0` if `y < 0` (Lower half-plane)
    /// - `θ = 0` if `y = 0` and `x > 0`
    /// - `θ = π` if `y = 0` and `x < 0`
    ///
    /// # Special Cases
    /// | Input Condition | Output Value |
    /// |--------------------------|--------------------|
    /// | `y = 0, x > 0` | `0` |
    /// | `y = 0, x < 0` | `π` |
    /// | `y > 0, x = 0` | `π / 2` |
    /// | `y < 0, x = 0` | `-π / 2` |
    /// | `y = 0, x = 0` | `0` (Undefined*) |
    ///
    /// *Note: Many implementations return 0 for origin coordinates to avoid NaNs
    /// in real-time control loops.*
    #[must_use]
    #[allow(clippy::arithmetic_side_effects)]
    fn atan2(self, rhs: Self) -> Self {
        let two = Self::ONE + Self::ONE;
        match CartesianQuadrant2D::from_coords(&rhs, &self) {
            CartesianQuadrant2D::Origin
            | CartesianQuadrant2D::PositiveXAxis => Self::ZERO,
            CartesianQuadrant2D::NegativeYAxis => -(Self::PI / two),
            CartesianQuadrant2D::NegativeXAxis => Self::PI,
            CartesianQuadrant2D::PositiveYAxis => Self::PI / two,
            _ => Self::atan(self / rhs),
        }
    }
    /// Computes the hyperbolic cosine of the number.
    ///
    /// # Mathematical Definition
    /// The hyperbolic cosine is an even function defined as the average of
    /// the natural exponential function and its inverse:
    /// <pre> cosh(x) = (e^x + e^-x) / 2 </pre>
    ///
    /// # Characteristics
    /// - **Domain**: `(-infty, infty)`
    /// - **Range**: `[1, infty)`
    /// - **Symmetry**: `cosh(-x) = cosh(x)`
    ///
    /// # Overflow Warning
    /// Because this implementation relies on `.exp()`, evaluating this function for
    /// large inputs results in rapid overflow to infinity (e.g., around `x ~ 89.4` for `f32`).
    #[must_use]
    #[allow(clippy::arithmetic_side_effects)]
    fn cosh(self) -> Self {
        let two = Self::ONE + Self::ONE;
        (self.clone().exp() + (Self::ZERO - self).exp()) / two
    }
    /// Returns the machine epsilon value for the type.
    fn epsilon() -> Self;
    /// Initiate self from the given const.
    #[must_use]
    fn from_const<const N: usize>() -> Self {
        Self::sum([Self::ONE; N])
    }
    /// Initiate self from the given usize.
    #[allow(clippy::arithmetic_side_effects)]
    fn from_usize(n: usize) -> Self {
        (0..n).fold(Self::ZERO, |acc, _| acc.add(Self::ONE))
    }
    /// Computes the hyperbolic sine of the number.
    ///
    /// # Mathematical Definition
    /// The hyperbolic sine is an odd function defined as half the difference
    /// between the natural exponential function and its inverse:
    /// $$ \sinh(x) = \frac{e^x - e^{-x}}{2} $$
    ///
    /// # Characteristics
    /// - **Domain**: $(-\infty, \infty)$
    /// - **Range**: $(-\infty, \infty)$
    /// - **Symmetry**: $\sinh(-x) = -\sinh(x)$
    ///
    /// # Numerical Stability Note
    /// This default trait implementation uses the standard algebraic definition.
    /// For values of `self` very close to `0.0`, computing $e^x - e^{-x}$ can
    /// suffer from **catastrophic cancellation**, leading to a loss of significant
    /// digits. For control loops requiring high precision near the origin,
    /// consider overriding this default with a Taylor series expansion or an `expm1`
    /// based approach for $|x| < 1$.
    #[must_use]
    #[allow(clippy::arithmetic_side_effects)]
    fn sinh(self) -> Self {
        let two = Self::ONE + Self::ONE;
        (self.clone().exp() - (Self::ZERO - self).exp()) / two
    }
    /// Sum the elements of an iterator.
    #[allow(clippy::arithmetic_side_effects)]
    fn sum<I: IntoIterator<Item = Self>>(iter: I) -> Self {
        iter.into_iter().fold(Self::ZERO, |acc, x| acc + x)
    }
}

////////////////////////////////////////////////////////////////////////////////

/// The unified target for control-loop arithmetic.
///
/// `AdditiveGroup + Signed + Mul`, implemented independently by signed
/// integers and floats. Deliberately excludes `Div`: integer division is
/// not total (`/0` panics, `i32::MIN / -1` overflows), so requiring it here
/// would reintroduce the panic surface this hierarchy exists to remove.
/// Division stays on `Float`, where IEEE-754 semantics make it total.
///
/// # Example
/// ```
/// use control_rs::math::num_traits::Scalar;
///
/// fn clamp_to_unit<T: Scalar>(val: T) -> T {
///     val.clamp(-T::ONE, T::ONE)
/// }
///
/// assert_eq!(clamp_to_unit(5), 1);
/// assert_eq!(clamp_to_unit(-5), -1);
/// ```
pub trait Scalar: AdditiveGroup + One + Signed + Mul<Output = Self> {
    /// Restricts a value to a certain interval.
    #[must_use]
    fn clamp(self, min: Self, max: Self) -> Self {
        if self < min {
            min
        } else if self > max {
            max
        } else {
            self
        }
    }
    /// Returns a number that represents the sign of self.
    #[must_use]
    #[allow(clippy::arithmetic_side_effects)]
    fn signum(self) -> Self {
        if self.is_zero() {
            Self::ZERO
        } else if self.is_sign_negative() {
            Self::ZERO - Self::ONE
        } else {
            Self::ONE
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Macro Code Generation
////////////////////////////////////////////////////////////////////////////////

/// Implements `Zero`, `One`, `Integer`, and `SaturatingInteger` for a given
/// integer primitive (signed or unsigned).
///
/// Relies on `ops.rs`'s existing `WrappingAdd`/`WrappingSub`/`WrappingMul`
/// and `SaturatingAdd`/`SaturatingSub`/`SaturatingMul` implementations,
/// already provided for every integer primitive.
///
/// # Arguments
/// - `$type`: The numeric type for which to implement the tier (e.g., `i64`, `u32`).
/// - `$one`: The literal expression for the multiplicative identity.
/// - `$zero`: The literal expression for the additive identity.
/// - `$max`: The literal expression for the maximum value.
/// - `$min`: The literal expression for the minimum value.
/// - `$min_pos`: The literal expression for the minimum positive value.
#[macro_export]
macro_rules! impl_int {
    ($type:ty, $one:expr, $zero:expr, $max:expr, $min:expr, $min_pos:expr) => {
        impl Zero for $type {
            const ZERO: Self = $zero;
        }

        ////////////////////////////////////////////////////////////////////////////////

        impl One for $type {
            const ONE: Self = $one;
        }

        ////////////////////////////////////////////////////////////////////////////////

        impl Integer for $type {
            const MAX: Self = $max;
            const MIN: Self = $min;
            const MIN_POSITIVE: Self = $min_pos;
            const TWO: Self = $one + $one;
        }

        ////////////////////////////////////////////////////////////////////////////////

        impl SaturatingInteger for $type {
            const MAX: Self = $max;
            const MIN: Self = $min;
            const MIN_POSITIVE: Self = $min_pos;
            const TWO: Self = $one + $one;
        }

        ////////////////////////////////////////////////////////////////////////////////
    };
}

/// Implements `AdditiveGroup` and `Signed` for a given type.
///
/// # Arguments
/// - `$type`: The numeric type.
/// - `$abs`: Path to the type's `abs` function (e.g., `i32::abs`, `libm::fabsf`).
#[macro_export]
macro_rules! impl_additive_group {
    ($type:ty, $abs:path) => {
        impl AdditiveGroup for $type {}

        ////////////////////////////////////////////////////////////////////////////////

        impl Signed for $type {
            #[inline(always)]
            fn abs(self) -> Self {
                $abs(self)
            }
        }

        ////////////////////////////////////////////////////////////////////////////////
    };
}

/// Implements the `Scalar` marker trait for a given type.
///
/// # Arguments
/// - `$type`: The numeric type; must already implement `AdditiveGroup + One + Signed`.
#[macro_export]
macro_rules! impl_scalar {
    ($type:ty) => {
        impl Scalar for $type {}

        ////////////////////////////////////////////////////////////////////////////////
    };
}

/// Implements `Zero`, `One`, `Radical`, `Exponential`, `Trig`, and `Float`
/// for a given floating-point type.
///
/// # Arguments
/// - `$type`: The numeric type.
/// - `$one`/`$zero`: Literal identity elements.
/// - `$epsilon`: The machine epsilon value.
/// - `$e`/`$pi`: Constants for Euler's number and Pi.
/// - `$sqrt`/`$exp`/`$ln`/`$log10`/`$pow`: Analytic function paths.
/// - `$cos`/`$sin`/`$tan`/`$acos`/`$asin`/`$atan`: Trigonometric function paths.
/// - `$atan2`: Path to the type's native two-argument arctangent, overriding
///   `Float::atan2`'s quadrant-table default with the hardware/libm form.
#[macro_export]
macro_rules! impl_float {
    (
        $type:ty, $one:expr, $zero:expr, $epsilon:expr, $e:expr, $pi:expr,
        $sqrt:path, $exp:path, $ln:path, $log10:path, $pow:path,
        $cos:path, $sin:path, $tan:path, $acos:path, $asin:path, $atan:path,
        $atan2:path
    ) => {
        impl Zero for $type {
            const ZERO: Self = $zero;
        }

        ////////////////////////////////////////////////////////////////////////////////

        impl One for $type {
            const ONE: Self = $one;
        }

        ////////////////////////////////////////////////////////////////////////////////

        impl Radical for $type {
            #[inline(always)]
            fn sqrt(self) -> Self {
                $sqrt(self)
            }
        }

        ////////////////////////////////////////////////////////////////////////////////

        impl Exponential for $type {
            const E: Self = $e;
            #[inline(always)]
            fn exp(self) -> Self {
                $exp(self)
            }
            #[inline(always)]
            fn ln(self) -> Self {
                $ln(self)
            }
            #[inline(always)]
            fn log10(self) -> Self {
                $log10(self)
            }
            #[inline(always)]
            fn pow(self, n: Self) -> Self {
                $pow(self, n)
            }
        }

        ////////////////////////////////////////////////////////////////////////////////

        impl Trig for $type {
            const PI: Self = $pi;
            #[inline(always)]
            fn acos(self) -> Self {
                $acos(self)
            }
            #[inline(always)]
            fn asin(self) -> Self {
                $asin(self)
            }
            #[inline(always)]
            fn atan(self) -> Self {
                $atan(self)
            }
            #[inline(always)]
            fn cos(self) -> Self {
                $cos(self)
            }
            #[inline(always)]
            fn sin(self) -> Self {
                $sin(self)
            }
            #[inline(always)]
            fn tan(self) -> Self {
                $tan(self)
            }
        }

        ////////////////////////////////////////////////////////////////////////////////

        impl Float for $type {
            #[inline(always)]
            fn atan2(self, rhs: Self) -> Self {
                $atan2(self, rhs)
            }
            #[inline(always)]
            fn epsilon() -> Self {
                $epsilon
            }
        }

        ////////////////////////////////////////////////////////////////////////////////
    };
}

////////////////////////////////////////////////////////////////////////////////
// Cartesian Quadrant Helper
////////////////////////////////////////////////////////////////////////////////

impl CartesianQuadrant2D {
    /// Instantiate a `CartesianQuadrant2D` from 2D coordinate.
    ///
    /// # Generic Arguments
    /// * `T` - Type of the coordinates.
    ///
    /// # Arguments
    /// * `x` - The first coordinate value.
    /// * `y` - The second coordinate value.
    ///
    /// # Returns
    /// * `quadrant` - Variant of the `CartesianQuadrant2D` enum corresponding to the coordinates.
    #[must_use]
    pub fn from_coords<T: Zero>(x: &T, y: &T) -> Self {
        match (x.partial_cmp(&T::ZERO), y.partial_cmp(&T::ZERO)) {
            (
                Some(core::cmp::Ordering::Equal),
                Some(core::cmp::Ordering::Equal),
            ) => Self::Origin,

            (
                Some(core::cmp::Ordering::Equal),
                Some(core::cmp::Ordering::Greater),
            ) => Self::PositiveYAxis,
            (
                Some(core::cmp::Ordering::Equal),
                Some(core::cmp::Ordering::Less),
            ) => Self::NegativeYAxis,

            (
                Some(core::cmp::Ordering::Greater),
                Some(core::cmp::Ordering::Equal),
            ) => Self::PositiveXAxis,
            (
                Some(core::cmp::Ordering::Less),
                Some(core::cmp::Ordering::Equal),
            ) => Self::NegativeXAxis,

            (
                Some(core::cmp::Ordering::Greater),
                Some(core::cmp::Ordering::Greater),
            ) => Self::Q1,
            (
                Some(core::cmp::Ordering::Less),
                Some(core::cmp::Ordering::Greater),
            ) => Self::Q2,
            (
                Some(core::cmp::Ordering::Less),
                Some(core::cmp::Ordering::Less),
            ) => Self::Q3,
            (
                Some(core::cmp::Ordering::Greater),
                Some(core::cmp::Ordering::Less),
            ) => Self::Q4,

            // Catch-all handles Cases where x or y are NaN, returning Undefined
            _ => Self::Undefined,
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Implementations for f32 (Embedded standard)
////////////////////////////////////////////////////////////////////////////////

impl_float!(
    f32,
    1.0,
    0.0,
    f32::EPSILON,
    core::f32::consts::E,
    core::f32::consts::PI,
    libm::sqrtf,
    libm::expf,
    libm::logf,
    libm::log10f,
    libm::powf,
    libm::cosf,
    libm::sinf,
    libm::tanf,
    libm::acosf,
    libm::asinf,
    libm::atanf,
    libm::atan2f
);

impl_additive_group!(f32, libm::fabsf);

impl_scalar!(f32);

////////////////////////////////////////////////////////////////////////////////
// Implementations for f64
////////////////////////////////////////////////////////////////////////////////

impl_float!(
    f64,
    1.0,
    0.0,
    f64::EPSILON,
    core::f64::consts::E,
    core::f64::consts::PI,
    libm::sqrt,
    libm::exp,
    libm::log,
    libm::log10,
    libm::pow,
    libm::cos,
    libm::sin,
    libm::tan,
    libm::acos,
    libm::asin,
    libm::atan,
    libm::atan2
);

impl_additive_group!(f64, libm::fabs);

impl_scalar!(f64);

////////////////////////////////////////////////////////////////////////////////
// Implementations for i8
////////////////////////////////////////////////////////////////////////////////

impl_int!(i8, 1, 0, i8::MAX, i8::MIN, 1);
impl_additive_group!(i8, i8::abs);
impl_scalar!(i8);

////////////////////////////////////////////////////////////////////////////////
// Implementations for i16
////////////////////////////////////////////////////////////////////////////////

impl_int!(i16, 1, 0, i16::MAX, i16::MIN, 1);
impl_additive_group!(i16, i16::abs);
impl_scalar!(i16);

////////////////////////////////////////////////////////////////////////////////
// Implementations for i32
////////////////////////////////////////////////////////////////////////////////

impl_int!(i32, 1, 0, i32::MAX, i32::MIN, 1);
impl_additive_group!(i32, i32::abs);
impl_scalar!(i32);

////////////////////////////////////////////////////////////////////////////////
// Implementations for i64
////////////////////////////////////////////////////////////////////////////////

impl_int!(i64, 1, 0, i64::MAX, i64::MIN, 1);
impl_additive_group!(i64, i64::abs);
impl_scalar!(i64);

////////////////////////////////////////////////////////////////////////////////
// Implementations for i128
////////////////////////////////////////////////////////////////////////////////

impl_int!(i128, 1, 0, i128::MAX, i128::MIN, 1);
impl_additive_group!(i128, i128::abs);
impl_scalar!(i128);

////////////////////////////////////////////////////////////////////////////////
// Implementations for isize
////////////////////////////////////////////////////////////////////////////////

impl_int!(isize, 1, 0, isize::MAX, isize::MIN, 1);
impl_additive_group!(isize, isize::abs);
impl_scalar!(isize);

////////////////////////////////////////////////////////////////////////////////
// Implementations for u8
////////////////////////////////////////////////////////////////////////////////

impl_int!(u8, 1, 0, u8::MAX, u8::MIN, 1);
impl Unsigned for u8 {}

////////////////////////////////////////////////////////////////////////////////
// Implementations for u16
////////////////////////////////////////////////////////////////////////////////

impl_int!(u16, 1, 0, u16::MAX, u16::MIN, 1);
impl Unsigned for u16 {}

////////////////////////////////////////////////////////////////////////////////
// Implementations for u32
////////////////////////////////////////////////////////////////////////////////

impl_int!(u32, 1, 0, u32::MAX, u32::MIN, 1);
impl Unsigned for u32 {}

////////////////////////////////////////////////////////////////////////////////
// Implementations for u64
////////////////////////////////////////////////////////////////////////////////

impl_int!(u64, 1, 0, u64::MAX, u64::MIN, 1);
impl Unsigned for u64 {}

////////////////////////////////////////////////////////////////////////////////
// Implementations for u128
////////////////////////////////////////////////////////////////////////////////

impl_int!(u128, 1, 0, u128::MAX, u128::MIN, 1);
impl Unsigned for u128 {}

////////////////////////////////////////////////////////////////////////////////
// Implementations for usize
////////////////////////////////////////////////////////////////////////////////

impl_int!(usize, 1, 0, usize::MAX, usize::MIN, 1);
impl Unsigned for usize {}
