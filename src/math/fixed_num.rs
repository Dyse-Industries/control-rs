//! # Fixed-Point Scalar Types
//!
//! This module provides deterministic, zero-allocation fixed-point scalar arithmetic
//! in Q-format representation for integer microcontroller units without hardware floating-point
//! units (FPUs).
//!
//! A value with scale `SHIFT` represents $x = \text{raw} \cdot 2^{-\text{SHIFT}}$ with fixed
//! quantization step $\Delta = 2^{-\text{SHIFT}}$.
//!
//! # Compile-Time Representability Gating
//!
//! Numerical traits (`One`, `Scalar`, `SaturatingInteger`) are gated to scales where their
//! mathematical identity constants ($1.0$, $2.0$) are strictly representable.
//!
//! Standard DSP interchange formats (e.g. `Q15`, `Q31`) span $[-1.0, 1.0)$ and cannot represent
//! $1.0$. Consequently, `Q15` implements `Zero` and `Conjugate`, but withholds `One`, `Scalar`,
//! and `SaturatingInteger` as trait bounds:
//!
//! ```compile_fail
//! use control_rs::math::num_traits::One;
//! use control_rs::math::fixed_num::Q15;
//!
//! fn assert_one<T: One>() {}
//! // Q15 cannot represent 1.0; trait bound One is withheld
//! assert_one::<Q15>();
//! ```
//!
//! ```compile_fail
//! use control_rs::math::num_traits::Scalar;
//! use control_rs::math::fixed_num::Fixed;
//!
//! fn assert_scalar<T: Scalar>() {}
//! // Fixed<i16, 15> (Q15) cannot represent 1.0; trait bound Scalar is withheld
//! assert_scalar::<Fixed<i16, 15>>();
//! ```
//!
//! ```compile_fail
//! use control_rs::math::num_traits::SaturatingInteger;
//! use control_rs::math::fixed_num::Fixed;
//!
//! fn assert_sat_int<T: SaturatingInteger>() {}
//! // Fixed<i16, 14> can represent 1.0 but not 2.0; trait bound SaturatingInteger is withheld
//! assert_sat_int::<Fixed<i16, 14>>();
//! ```
//!
//! ```compile_fail
//! use control_rs::math::num_traits::Zero;
//! use control_rs::math::fixed_num::Fixed;
//!
//! fn assert_zero<T: Zero>() {}
//! // Fixed<i16, 17> exceeds 16-bit word length; trait bound Zero is withheld
//! assert_zero::<Fixed<i16, 17>>();
//! ```
//!
//! ```compile_fail
//! use control_rs::math::num_traits::Float;
//! use control_rs::math::fixed_num::Fixed;
//!
//! fn assert_float<T: Float>() {}
//! // Fixed-point scalar types do not implement Float
//! assert_float::<Fixed<i32, 16>>();
//! ```

#![allow(clippy::pedantic)]
#![allow(clippy::arbitrary_source_item_ordering)]
#![allow(clippy::arithmetic_side_effects)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::return_self_not_must_use)]

use crate::math::{
    ArithmeticError, ArithmeticResult,
    num_traits::{
        AdditiveGroup, Conjugate, One, SaturatingInteger, Scalar, Signed,
        Unsigned, Zero,
    },
    num_types::{
        Const, Dim, DimMax, U5, U6, U7, U8, U13, U14, U15, U16, U29, U30, U31,
        U32, U61, U62, U63, U64,
    },
    ops::{
        Add, AddAssign, Mul, MulAssign, Neg, SaturatingAdd, SaturatingMul,
        SaturatingSub, Sub, SubAssign, TryAdd, TryMul, TryNeg, TrySub,
    },
};

mod private {
    pub trait Sealed {}
    impl Sealed for i8 {}
    impl Sealed for i16 {}
    impl Sealed for i32 {}
    impl Sealed for i64 {}
    impl Sealed for u8 {}
    impl Sealed for u16 {}
    impl Sealed for u32 {}
    impl Sealed for u64 {}

    pub trait SealedMarker {}

    pub trait FixedShiftVal<const SHIFT: usize> {
        const ONE_RAW: Self;
        const TWO_RAW: Self;
    }
}

/// Sealed marker trait indicating that $1.0$ is strictly representable at this scale.
pub trait OneRepresentable: private::SealedMarker {}

/// Sealed marker trait indicating that $2.0$ is strictly representable at this scale.
pub trait TwoRepresentable: OneRepresentable {}

#[inline]
#[allow(clippy::arithmetic_side_effects)]
fn scale_to_f64_factor(shift: usize) -> f64 {
    if shift <= 62 {
        (1u64 << (shift as u32)) as f64
    } else if shift == 63 {
        ((1u64 << 62) as f64) * 2.0
    } else {
        ((1u64 << 62) as f64) * 4.0
    }
}

/// Sealed trait parameterizing primitive integer types for fixed-point arithmetic.
pub trait FixedRepr:
    private::Sealed
    + Copy
    + Eq
    + Ord
    + core::fmt::Display
    + core::fmt::Debug
    + Sized
    + 'static
{
    /// Bit width of this primitive integer representation.
    const BITS: u32;

    /// Whether this primitive type is signed.
    const IS_SIGNED: bool;

    /// Type-level bit-width dimension (e.g. `U8`, `U16`, `U32`, `U64`).
    type BitsDim: Dim;

    /// Maximum scale exponent where 1.0 is strictly representable.
    type OneMaxShift: Dim;

    /// Maximum scale exponent where 2.0 is strictly representable.
    type TwoMaxShift: Dim;

    /// Doubled-width intermediate integer type for exact products and rescaling.
    type Wide: Copy + Eq + Ord + Sized + 'static;

    /// Widen this value into doubled-width intermediate format.
    fn widen(self) -> Self::Wide;

    /// Multiply two doubled-width intermediate values.
    fn wide_mul(a: Self::Wide, b: Self::Wide) -> Self::Wide;

    /// Narrow the doubled-width intermediate back to `Self`, saturating at representation bounds.
    fn narrow_saturating(val: Self::Wide) -> Self;

    /// Rescale a doubled-width product (in $2Q$ scale) down to $Q$ scale using convergent rounding (round-ties-to-even).
    fn rescale_product_down(prod: Self::Wide, shift: usize) -> Self;

    /// Rescale from scale $Q$ to scale $R$.
    fn rescale_value(self, q: usize, r: usize) -> Self;

    /// Checked addition.
    fn checked_add_repr(self, rhs: Self) -> Option<Self>;

    /// Checked subtraction.
    fn checked_sub_repr(self, rhs: Self) -> Option<Self>;

    /// Checked multiplication (in $Q$ scale, widening and rescaling).
    fn checked_mul_repr(self, rhs: Self, shift: usize) -> Option<Self>;

    /// Checked negation.
    fn checked_neg_repr(self) -> Option<Self>;

    /// Saturating addition.
    fn saturating_add_repr(self, rhs: Self) -> Self;

    /// Saturating subtraction.
    fn saturating_sub_repr(self, rhs: Self) -> Self;

    /// Saturating negation.
    fn saturating_neg_repr(self) -> Self;

    /// Returns `true` if `self` is zero.
    fn is_zero_repr(self) -> bool;

    /// Minimum representable raw value.
    const MIN_RAW: Self;

    /// Maximum representable raw value.
    const MAX_RAW: Self;

    /// Zero raw value.
    const ZERO_RAW: Self;

    /// One raw value (1 as an integer).
    const ONE_RAW: Self;

    /// Convert from `f64` at the specified scale.
    fn from_f64(val: f64, shift: usize) -> Self;

    /// Convert to `f64` at the specified scale.
    fn to_f64(self, shift: usize) -> f64;
}

macro_rules! impl_signed_repr {
    ($t:ident, $w:ident, $bits:expr, $bits_dim:ident, $one_max:ident, $two_max:ident) => {
        impl<const SHIFT: usize> private::FixedShiftVal<SHIFT> for $t {
            const ONE_RAW: Self = if SHIFT < $bits {
                (1 as $t).wrapping_shl(SHIFT as u32)
            } else {
                0
            };
            const TWO_RAW: Self = if SHIFT + 1 < $bits {
                (1 as $t).wrapping_shl((SHIFT + 1) as u32)
            } else {
                0
            };
        }

        impl FixedRepr for $t {
            const BITS: u32 = $bits;
            const IS_SIGNED: bool = true;
            type BitsDim = $bits_dim;
            type OneMaxShift = $one_max;
            type TwoMaxShift = $two_max;
            type Wide = $w;

            #[inline(always)]
            fn widen(self) -> Self::Wide {
                self as $w
            }

            #[inline(always)]
            #[allow(clippy::arithmetic_side_effects)]
            fn wide_mul(a: Self::Wide, b: Self::Wide) -> Self::Wide {
                a.wrapping_mul(b)
            }

            #[inline]
            fn narrow_saturating(val: Self::Wide) -> Self {
                if val > $t::MAX as $w {
                    $t::MAX
                } else if val < $t::MIN as $w {
                    $t::MIN
                } else {
                    val as $t
                }
            }

            #[inline]
            #[allow(clippy::arithmetic_side_effects)]
            #[allow(clippy::cast_possible_truncation)]
            #[allow(clippy::cast_possible_wrap)]
            fn rescale_product_down(prod: Self::Wide, shift: usize) -> Self {
                if shift == 0 {
                    return Self::narrow_saturating(prod);
                }
                if prod == $w::MIN {
                    return $t::MIN;
                }
                let is_neg = prod < 0;
                let abs_val = prod.unsigned_abs();
                let mask = (1u128.checked_shl(shift as u32).unwrap_or(0))
                    .wrapping_sub(1) as u128;
                let half = 1u128
                    .checked_shl((shift.saturating_sub(1)) as u32)
                    .unwrap_or(0);
                let abs_u128 = abs_val as u128;
                let rem = abs_u128 & mask;
                let truncated = abs_u128 >> (shift as u32);
                let rounded_abs =
                    if rem > half || (rem == half && (truncated & 1) != 0) {
                        truncated.saturating_add(1)
                    } else {
                        truncated
                    };
                let max_abs = $t::MAX as u128;
                let min_abs = ($t::MAX as u128).saturating_add(1);
                if is_neg {
                    if rounded_abs >= min_abs {
                        $t::MIN
                    } else {
                        -(rounded_abs as $t)
                    }
                } else {
                    if rounded_abs >= max_abs {
                        $t::MAX
                    } else {
                        rounded_abs as $t
                    }
                }
            }

            #[inline]
            #[allow(clippy::arithmetic_side_effects)]
            #[allow(clippy::cast_possible_truncation)]
            #[allow(clippy::cast_possible_wrap)]
            fn rescale_value(self, q: usize, r: usize) -> Self {
                if r == q {
                    self
                } else if r > q {
                    let diff = (r - q) as u32;
                    if diff >= $bits {
                        if self > 0 {
                            $t::MAX
                        } else if self < 0 {
                            $t::MIN
                        } else {
                            0
                        }
                    } else {
                        let max_shiftable = $t::MAX >> diff;
                        let min_shiftable = $t::MIN >> diff;
                        if self > max_shiftable {
                            $t::MAX
                        } else if self < min_shiftable {
                            $t::MIN
                        } else {
                            self << diff
                        }
                    }
                } else {
                    let diff = (q - r) as u32;
                    if diff >= $bits {
                        0
                    } else {
                        let is_neg = self < 0;
                        let abs_val = self.unsigned_abs() as u128;
                        let mask = (1u128 << diff) - 1;
                        let half = 1u128 << (diff - 1);
                        let rem = abs_val & mask;
                        let truncated = abs_val >> diff;
                        let rounded_abs = if rem > half
                            || (rem == half && (truncated & 1) != 0)
                        {
                            truncated.saturating_add(1)
                        } else {
                            truncated
                        };
                        let min_abs = ($t::MAX as u128).saturating_add(1);
                        if is_neg {
                            if rounded_abs >= min_abs {
                                $t::MIN
                            } else {
                                -(rounded_abs as $t)
                            }
                        } else {
                            if rounded_abs >= $t::MAX as u128 {
                                $t::MAX
                            } else {
                                rounded_abs as $t
                            }
                        }
                    }
                }
            }

            #[inline(always)]
            fn checked_add_repr(self, rhs: Self) -> Option<Self> {
                self.checked_add(rhs)
            }

            #[inline(always)]
            fn checked_sub_repr(self, rhs: Self) -> Option<Self> {
                self.checked_sub(rhs)
            }

            #[inline]
            #[allow(clippy::arithmetic_side_effects)]
            #[allow(clippy::cast_possible_truncation)]
            #[allow(clippy::cast_possible_wrap)]
            fn checked_mul_repr(self, rhs: Self, shift: usize) -> Option<Self> {
                let w_a = self as $w;
                let w_b = rhs as $w;
                let w_prod = w_a.checked_mul(w_b)?;
                if shift == 0 {
                    if w_prod > $t::MAX as $w || w_prod < $t::MIN as $w {
                        None
                    } else {
                        Some(w_prod as $t)
                    }
                } else {
                    if w_prod == $w::MIN {
                        return None;
                    }
                    let is_neg = w_prod < 0;
                    let abs_val = w_prod.unsigned_abs();
                    let mask = (1u128.checked_shl(shift as u32).unwrap_or(0))
                        .wrapping_sub(1) as u128;
                    let half = 1u128
                        .checked_shl((shift.saturating_sub(1)) as u32)
                        .unwrap_or(0);
                    let abs_u128 = abs_val as u128;
                    let rem = abs_u128 & mask;
                    let truncated = abs_u128 >> (shift as u32);
                    let rounded_abs = if rem > half
                        || (rem == half && (truncated & 1) != 0)
                    {
                        truncated.checked_add(1)?
                    } else {
                        truncated
                    };
                    let max_abs = $t::MAX as u128;
                    let min_abs = ($t::MAX as u128).saturating_add(1);
                    if is_neg {
                        if rounded_abs > min_abs {
                            None
                        } else if rounded_abs == min_abs {
                            Some($t::MIN)
                        } else {
                            Some(-(rounded_abs as $t))
                        }
                    } else {
                        if rounded_abs > max_abs {
                            None
                        } else {
                            Some(rounded_abs as $t)
                        }
                    }
                }
            }

            #[inline(always)]
            fn checked_neg_repr(self) -> Option<Self> {
                self.checked_neg()
            }

            #[inline(always)]
            fn saturating_add_repr(self, rhs: Self) -> Self {
                self.saturating_add(rhs)
            }

            #[inline(always)]
            fn saturating_sub_repr(self, rhs: Self) -> Self {
                self.saturating_sub(rhs)
            }

            #[inline(always)]
            fn saturating_neg_repr(self) -> Self {
                self.saturating_neg()
            }

            #[inline(always)]
            fn is_zero_repr(self) -> bool {
                self == 0
            }

            const MIN_RAW: Self = $t::MIN;
            const MAX_RAW: Self = $t::MAX;
            const ZERO_RAW: Self = 0;
            const ONE_RAW: Self = 1;

            #[inline]
            #[allow(clippy::arithmetic_side_effects)]
            #[allow(clippy::cast_possible_truncation)]
            fn from_f64(val: f64, shift: usize) -> Self {
                let factor = scale_to_f64_factor(shift);
                let scaled = val * factor;
                if scaled >= $t::MAX as f64 {
                    $t::MAX
                } else if scaled <= $t::MIN as f64 {
                    $t::MIN
                } else if scaled >= 0.0 {
                    (scaled + 0.5) as $t
                } else {
                    (scaled - 0.5) as $t
                }
            }

            #[inline]
            #[allow(clippy::arithmetic_side_effects)]
            fn to_f64(self, shift: usize) -> f64 {
                let factor = scale_to_f64_factor(shift);
                (self as f64) / factor
            }
        }
    };
}

impl_signed_repr!(i8, i16, 8, U8, U6, U5);
impl_signed_repr!(i16, i32, 16, U16, U14, U13);
impl_signed_repr!(i32, i64, 32, U32, U30, U29);
impl_signed_repr!(i64, i128, 64, U64, U62, U61);

macro_rules! impl_unsigned_repr {
    ($t:ident, $w:ident, $bits:expr, $bits_dim:ident, $one_max:ident, $two_max:ident) => {
        impl<const SHIFT: usize> private::FixedShiftVal<SHIFT> for $t {
            const ONE_RAW: Self = if SHIFT < $bits {
                (1 as $t).wrapping_shl(SHIFT as u32)
            } else {
                0
            };
            const TWO_RAW: Self = if SHIFT + 1 < $bits {
                (1 as $t).wrapping_shl((SHIFT + 1) as u32)
            } else {
                0
            };
        }

        impl FixedRepr for $t {
            const BITS: u32 = $bits;
            const IS_SIGNED: bool = false;
            type BitsDim = $bits_dim;
            type OneMaxShift = $one_max;
            type TwoMaxShift = $two_max;
            type Wide = $w;

            #[inline(always)]
            fn widen(self) -> Self::Wide {
                self as $w
            }

            #[inline(always)]
            #[allow(clippy::arithmetic_side_effects)]
            fn wide_mul(a: Self::Wide, b: Self::Wide) -> Self::Wide {
                a.wrapping_mul(b)
            }

            #[inline]
            fn narrow_saturating(val: Self::Wide) -> Self {
                if val > $t::MAX as $w {
                    $t::MAX
                } else {
                    val as $t
                }
            }

            #[inline]
            #[allow(clippy::arithmetic_side_effects)]
            #[allow(clippy::cast_possible_truncation)]
            fn rescale_product_down(prod: Self::Wide, shift: usize) -> Self {
                if shift == 0 {
                    return Self::narrow_saturating(prod);
                }
                let mask = (1u128.checked_shl(shift as u32).unwrap_or(0))
                    .wrapping_sub(1) as $w;
                let half = (1 as $w) << (shift.saturating_sub(1));
                let rem = prod & mask;
                let truncated = prod >> (shift as u32);
                let rounded =
                    if rem > half || (rem == half && (truncated & 1) != 0) {
                        truncated.saturating_add(1)
                    } else {
                        truncated
                    };
                Self::narrow_saturating(rounded)
            }

            #[inline]
            #[allow(clippy::arithmetic_side_effects)]
            #[allow(clippy::cast_possible_truncation)]
            fn rescale_value(self, q: usize, r: usize) -> Self {
                if r == q {
                    self
                } else if r > q {
                    let diff = (r - q) as u32;
                    if diff >= $bits {
                        if self > 0 { $t::MAX } else { 0 }
                    } else {
                        let max_shiftable = $t::MAX >> diff;
                        if self > max_shiftable {
                            $t::MAX
                        } else {
                            self << diff
                        }
                    }
                } else {
                    let diff = (q - r) as u32;
                    if diff >= $bits {
                        0
                    } else {
                        let mask = ((1 as $t) << diff) - 1;
                        let half = (1 as $t) << (diff - 1);
                        let rem = self & mask;
                        let truncated = self >> diff;
                        if rem > half || (rem == half && (truncated & 1) != 0) {
                            truncated.saturating_add(1)
                        } else {
                            truncated
                        }
                    }
                }
            }

            #[inline(always)]
            fn checked_add_repr(self, rhs: Self) -> Option<Self> {
                self.checked_add(rhs)
            }

            #[inline(always)]
            fn checked_sub_repr(self, rhs: Self) -> Option<Self> {
                self.checked_sub(rhs)
            }

            #[inline]
            #[allow(clippy::arithmetic_side_effects)]
            #[allow(clippy::cast_possible_truncation)]
            fn checked_mul_repr(self, rhs: Self, shift: usize) -> Option<Self> {
                let w_a = self as $w;
                let w_b = rhs as $w;
                let w_prod = w_a.checked_mul(w_b)?;
                if shift == 0 {
                    if w_prod > $t::MAX as $w {
                        None
                    } else {
                        Some(w_prod as $t)
                    }
                } else {
                    let mask = (1u128.checked_shl(shift as u32).unwrap_or(0))
                        .wrapping_sub(1) as $w;
                    let half = (1 as $w) << (shift.saturating_sub(1));
                    let rem = w_prod & mask;
                    let truncated = w_prod >> (shift as u32);
                    let rounded = if rem > half
                        || (rem == half && (truncated & 1) != 0)
                    {
                        truncated.checked_add(1)?
                    } else {
                        truncated
                    };
                    if rounded > $t::MAX as $w {
                        None
                    } else {
                        Some(rounded as $t)
                    }
                }
            }

            #[inline(always)]
            fn checked_neg_repr(self) -> Option<Self> {
                if self == 0 { Some(0) } else { None }
            }

            #[inline(always)]
            fn saturating_add_repr(self, rhs: Self) -> Self {
                self.saturating_add(rhs)
            }

            #[inline(always)]
            fn saturating_sub_repr(self, rhs: Self) -> Self {
                self.saturating_sub(rhs)
            }

            #[inline(always)]
            fn saturating_neg_repr(self) -> Self {
                0
            }

            #[inline(always)]
            fn is_zero_repr(self) -> bool {
                self == 0
            }

            const MIN_RAW: Self = 0;
            const MAX_RAW: Self = $t::MAX;
            const ZERO_RAW: Self = 0;
            const ONE_RAW: Self = 1;

            #[inline]
            #[allow(clippy::arithmetic_side_effects)]
            #[allow(clippy::cast_possible_truncation)]
            fn from_f64(val: f64, shift: usize) -> Self {
                if val <= 0.0 {
                    return 0;
                }
                let factor = scale_to_f64_factor(shift);
                let scaled = val * factor;
                if scaled >= $t::MAX as f64 {
                    $t::MAX
                } else {
                    (scaled + 0.5) as $t
                }
            }

            #[inline]
            #[allow(clippy::arithmetic_side_effects)]
            fn to_f64(self, shift: usize) -> f64 {
                let factor = scale_to_f64_factor(shift);
                (self as f64) / factor
            }
        }
    };
}

impl_unsigned_repr!(u8, u16, 8, U8, U7, U6);
impl_unsigned_repr!(u16, u32, 16, U16, U15, U14);
impl_unsigned_repr!(u32, u64, 32, U32, U31, U30);
impl_unsigned_repr!(u64, u128, 64, U64, U63, U62);

/// Fixed-point scalar type in Q-format representation.
///
/// Each value represents $x = \text{raw} \cdot 2^{-\text{SHIFT}}$ with fixed quantization step
/// $\Delta = 2^{-\text{SHIFT}}$.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Fixed<Repr, const SHIFT: usize> {
    raw: Repr,
}

/// Canonical type alias for downstream numerical models.
pub type Quantized<Repr, const SHIFT: usize> = Fixed<Repr, SHIFT>;

impl<Repr: FixedRepr, const SHIFT: usize> Fixed<Repr, SHIFT> {
    /// Scale exponent in bits.
    pub const SHIFT: usize = SHIFT;

    /// Bit resolution (quantization step $\Delta = 2^{-\text{SHIFT}}$).
    pub const DELTA: Self = Self { raw: Repr::ONE_RAW };

    /// Minimum positive representable step $\Delta = 2^{-\text{SHIFT}}$.
    pub const MIN_POSITIVE: Self = Self::DELTA;

    /// Minimum representable value.
    pub const MIN: Self = Self { raw: Repr::MIN_RAW };

    /// Maximum representable value.
    pub const MAX: Self = Self { raw: Repr::MAX_RAW };

    /// Additive identity ($0.0$).
    pub const ZERO: Self = Self {
        raw: Repr::ZERO_RAW,
    };

    /// Constructs a fixed-point number from its raw scaled integer representation.
    #[inline(always)]
    #[must_use]
    pub const fn from_bits(raw: Repr) -> Self {
        Self { raw }
    }

    /// Extracts the underlying raw scaled integer representation.
    #[inline(always)]
    #[must_use]
    pub const fn to_bits(self) -> Repr {
        self.raw
    }

    /// Builds a fixed-point value from raw representation (alias for [`Fixed::from_bits`]).
    #[inline(always)]
    #[must_use]
    pub const fn from_raw(raw: Repr) -> Self {
        Self { raw }
    }

    /// Converts a floating-point number into this fixed-point format with saturation.
    #[inline]
    #[must_use]
    pub fn from_num(val: f64) -> Self {
        Self {
            raw: Repr::from_f64(val, SHIFT),
        }
    }

    /// Converts this fixed-point number into a floating-point value.
    #[inline]
    #[must_use]
    pub fn to_num(self) -> f64 {
        Repr::to_f64(self.raw, SHIFT)
    }

    /// Extracts the underlying raw representation.
    #[inline(always)]
    #[must_use]
    pub const fn raw(self) -> Repr {
        self.raw
    }

    /// Quantizes a floating-point scalar into this fixed-point format (alias for [`Fixed::from_num`]).
    #[inline]
    #[must_use]
    pub fn quantize(val: impl Into<f64>) -> Self {
        Self::from_num(val.into())
    }

    /// Dequantizes this fixed-point scalar into a floating-point number (alias for [`Fixed::to_num`]).
    #[inline]
    #[must_use]
    pub fn dequantize(self) -> f64 {
        self.to_num()
    }

    /// Rescales this value to a new scale exponent `R` with convergent rounding.
    #[inline]
    #[must_use]
    pub fn rescale<const R: usize>(self) -> Fixed<Repr, R>
    where
        Const<SHIFT>: Dim,
        Const<R>: Dim,
    {
        Fixed {
            raw: Repr::rescale_value(self.raw, SHIFT, R),
        }
    }
}

impl<Repr: FixedRepr + private::FixedShiftVal<SHIFT>, const SHIFT: usize>
    private::SealedMarker for Fixed<Repr, SHIFT>
where
    Const<SHIFT>: Dim + DimMax<Repr::OneMaxShift, Output = Repr::OneMaxShift>,
{
}

impl<Repr: FixedRepr + private::FixedShiftVal<SHIFT>, const SHIFT: usize>
    OneRepresentable for Fixed<Repr, SHIFT>
where
    Const<SHIFT>: Dim + DimMax<Repr::OneMaxShift, Output = Repr::OneMaxShift>,
{
}

impl<Repr: FixedRepr + private::FixedShiftVal<SHIFT>, const SHIFT: usize>
    TwoRepresentable for Fixed<Repr, SHIFT>
where
    Const<SHIFT>: Dim
        + DimMax<Repr::TwoMaxShift, Output = Repr::TwoMaxShift>
        + DimMax<Repr::OneMaxShift, Output = Repr::OneMaxShift>,
{
}

impl<Repr: FixedRepr + private::FixedShiftVal<SHIFT>, const SHIFT: usize>
    Fixed<Repr, SHIFT>
where
    Self: OneRepresentable,
{
    /// Multiplicative identity ($1.0$), gated to scales where $1.0$ is representable.
    pub const ONE: Self = Self {
        raw: <Repr as private::FixedShiftVal<SHIFT>>::ONE_RAW,
    };
}

impl<Repr: FixedRepr + private::FixedShiftVal<SHIFT>, const SHIFT: usize>
    Fixed<Repr, SHIFT>
where
    Self: TwoRepresentable,
{
    /// Multiplicative constant ($2.0$), gated to scales where $2.0$ is representable.
    pub const TWO: Self = Self {
        raw: <Repr as private::FixedShiftVal<SHIFT>>::TWO_RAW,
    };
}

impl<Repr: FixedRepr + private::FixedShiftVal<SHIFT>, const SHIFT: usize> One
    for Fixed<Repr, SHIFT>
where
    Self: OneRepresentable,
{
    const ONE: Self = Self {
        raw: <Repr as private::FixedShiftVal<SHIFT>>::ONE_RAW,
    };

    #[inline(always)]
    fn is_one(&self) -> bool {
        self.raw == Self::ONE.raw
    }
}

impl<Repr: FixedRepr, const SHIFT: usize> core::fmt::Display
    for Fixed<Repr, SHIFT>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Fixed({}, Q{})", self.raw, SHIFT)
    }
}

impl<Repr: FixedRepr, const SHIFT: usize> Add for Fixed<Repr, SHIFT> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self {
            raw: Repr::saturating_add_repr(self.raw, rhs.raw),
        }
    }
}

impl<Repr: FixedRepr, const SHIFT: usize> AddAssign for Fixed<Repr, SHIFT> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<Repr: FixedRepr, const SHIFT: usize> Sub for Fixed<Repr, SHIFT> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self {
            raw: Repr::saturating_sub_repr(self.raw, rhs.raw),
        }
    }
}

impl<Repr: FixedRepr, const SHIFT: usize> SubAssign for Fixed<Repr, SHIFT> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<Repr: FixedRepr, const SHIFT: usize> Neg for Fixed<Repr, SHIFT> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self {
            raw: Repr::saturating_neg_repr(self.raw),
        }
    }
}

impl<Repr: FixedRepr, const SHIFT: usize> Mul for Fixed<Repr, SHIFT> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        let w_a = self.raw.widen();
        let w_b = rhs.raw.widen();
        let w_prod = Repr::wide_mul(w_a, w_b);
        let raw = Repr::rescale_product_down(w_prod, SHIFT);
        Self { raw }
    }
}

impl<Repr: FixedRepr, const SHIFT: usize> MulAssign for Fixed<Repr, SHIFT> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<Repr: FixedRepr, const SHIFT: usize> TryAdd for Fixed<Repr, SHIFT> {
    #[inline]
    fn try_add(&self, v: &Self) -> ArithmeticResult<Self> {
        Repr::checked_add_repr(self.raw, v.raw)
            .map(|raw| Self { raw })
            .ok_or(ArithmeticError::Overflow)
    }
}

impl<Repr: FixedRepr, const SHIFT: usize> TrySub for Fixed<Repr, SHIFT> {
    #[inline]
    fn try_sub(&self, v: &Self) -> ArithmeticResult<Self> {
        Repr::checked_sub_repr(self.raw, v.raw)
            .map(|raw| Self { raw })
            .ok_or(ArithmeticError::Overflow)
    }
}

impl<Repr: FixedRepr, const SHIFT: usize> TryMul for Fixed<Repr, SHIFT> {
    #[inline]
    fn try_mul(&self, v: &Self) -> ArithmeticResult<Self> {
        Repr::checked_mul_repr(self.raw, v.raw, SHIFT)
            .map(|raw| Self { raw })
            .ok_or(ArithmeticError::Overflow)
    }
}

impl<Repr: FixedRepr, const SHIFT: usize> TryNeg for Fixed<Repr, SHIFT> {
    #[inline]
    fn try_neg(&self) -> ArithmeticResult<Self> {
        Repr::checked_neg_repr(self.raw)
            .map(|raw| Self { raw })
            .ok_or(ArithmeticError::Overflow)
    }
}

impl<Repr: FixedRepr, const SHIFT: usize> SaturatingAdd for Fixed<Repr, SHIFT> {
    #[inline(always)]
    fn saturating_add(&self, v: &Self) -> Self {
        *self + *v
    }
}

impl<Repr: FixedRepr, const SHIFT: usize> SaturatingSub for Fixed<Repr, SHIFT> {
    #[inline(always)]
    fn saturating_sub(&self, v: &Self) -> Self {
        *self - *v
    }
}

impl<Repr: FixedRepr, const SHIFT: usize> SaturatingMul for Fixed<Repr, SHIFT> {
    #[inline(always)]
    fn saturating_mul(&self, v: &Self) -> Self {
        *self * *v
    }
}

impl<Repr: FixedRepr, const SHIFT: usize> Zero for Fixed<Repr, SHIFT>
where
    Const<SHIFT>: Dim + DimMax<Repr::BitsDim, Output = Repr::BitsDim>,
{
    const ZERO: Self = Self::ZERO;

    #[inline(always)]
    fn is_zero(&self) -> bool {
        Repr::is_zero_repr(self.raw)
    }
}

impl<Repr: FixedRepr, const SHIFT: usize> Conjugate for Fixed<Repr, SHIFT> {
    #[inline(always)]
    fn conj(self) -> Self {
        self
    }
}

impl<Repr: FixedRepr + private::FixedShiftVal<SHIFT>, const SHIFT: usize> Scalar
    for Fixed<Repr, SHIFT>
where
    Self: OneRepresentable,
    Const<SHIFT>: Dim + DimMax<Repr::BitsDim, Output = Repr::BitsDim>,
{
    type Real = Self;

    #[inline(always)]
    fn abs2(&self) -> Self::Real {
        *self * *self
    }

    #[inline(always)]
    fn from_real(re: Self::Real) -> Self {
        re
    }

    #[inline(always)]
    fn im(&self) -> Self::Real {
        Self::ZERO
    }

    #[inline(always)]
    fn re(&self) -> Self::Real {
        *self
    }
}

impl<const SHIFT: usize> AdditiveGroup for Fixed<i8, SHIFT> where
    Const<SHIFT>: Dim + DimMax<U8, Output = U8>
{
}
impl<const SHIFT: usize> AdditiveGroup for Fixed<i16, SHIFT> where
    Const<SHIFT>: Dim + DimMax<U16, Output = U16>
{
}
impl<const SHIFT: usize> AdditiveGroup for Fixed<i32, SHIFT> where
    Const<SHIFT>: Dim + DimMax<U32, Output = U32>
{
}
impl<const SHIFT: usize> AdditiveGroup for Fixed<i64, SHIFT> where
    Const<SHIFT>: Dim + DimMax<U64, Output = U64>
{
}

macro_rules! impl_signed_fixed {
    ($($t:ty, $dim:ident),+) => {
        $(
            impl<const SHIFT: usize> Signed for Fixed<$t, SHIFT>
            where
                Const<SHIFT>: Dim + DimMax<$dim, Output = $dim>,
            {
                #[inline(always)]
                fn abs(self) -> Self {
                    Self { raw: self.raw.saturating_abs() }
                }
            }
        )+
    };
}

impl_signed_fixed!(i8, U8, i16, U16, i32, U32, i64, U64);

impl<const SHIFT: usize> Unsigned for Fixed<u8, SHIFT> {}
impl<const SHIFT: usize> Unsigned for Fixed<u16, SHIFT> {}
impl<const SHIFT: usize> Unsigned for Fixed<u32, SHIFT> {}
impl<const SHIFT: usize> Unsigned for Fixed<u64, SHIFT> {}

impl<Repr: FixedRepr + private::FixedShiftVal<SHIFT>, const SHIFT: usize>
    SaturatingInteger for Fixed<Repr, SHIFT>
where
    Self: TwoRepresentable,
    Const<SHIFT>: Dim + DimMax<Repr::BitsDim, Output = Repr::BitsDim>,
{
}

// ============================================================================
// Standard Format Aliases
// ============================================================================

/// Signed Q0.7 fixed-point format (8 bits, 7 fractional bits).
pub type Q7 = Fixed<i8, 7>;

/// Signed Q0.15 fixed-point format (16 bits, 15 fractional bits).
pub type Q15 = Fixed<i16, 15>;

/// Signed Q0.31 fixed-point format (32 bits, 31 fractional bits).
pub type Q31 = Fixed<i32, 31>;

/// Signed Q0.63 fixed-point format (64 bits, 63 fractional bits).
pub type Q63 = Fixed<i64, 63>;

/// Unsigned UQ0.7 fixed-point format (8 bits, 7 fractional bits).
pub type UQ7 = Fixed<u8, 7>;

/// Unsigned UQ0.15 fixed-point format (16 bits, 15 fractional bits).
pub type UQ15 = Fixed<u16, 15>;

/// Unsigned UQ0.31 fixed-point format (32 bits, 31 fractional bits).
pub type UQ31 = Fixed<u32, 31>;

/// Unsigned UQ0.63 fixed-point format (64 bits, 63 fractional bits).
pub type UQ63 = Fixed<u64, 63>;
