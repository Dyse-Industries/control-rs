//! Firmware realizations: direct forms, second-order sections, cascade,
//! factorization, quantization and section stability (FR-8 to FR-10,
//! FR-17 to FR-20).
//!
//! Every realization uses the denominator convention
//! `H(z) = (b_0 + b_1 z^-1 + ... + b_n z^-n) / (1 + a_1 z^-1 + ... + a_n z^-n)`.
//! Sections store `-a_1, -a_2` so each output is one `MulAcc` chain of
//! multiply accumulates that narrows once (C-8). `T` must be [`Signed`]:
//! unsigned `Fixed` implements `saturating_neg` as zero, which would clear
//! the feedback taps. `Df1` keeps two past inputs and two past outputs in
//! `T`; `Df2t` and `DirectForm2T` keep their states in `T::Acc`, so only the
//! output narrows. `update`, `reset` and `set_coefficients` are infallible
//! and run a fixed operation count set by the order or the section count
//! (NFR-4).

use super::{ClassicalError, Succ, TypeNum};
use crate::math::complex_num::Complex;
use crate::math::fixed_num::{Fixed, FixedRepr};
use crate::math::num_traits::{Float, MulAcc, Scalar, Signed};
use crate::math::num_types::{Const, Dim, DimAdd, DimMax};
use crate::math::ops::{
    SaturatingAdd, SaturatingDiv, SaturatingMul, SaturatingSub,
};
use crate::polynomial::ArrayPolynomial;
use crate::transfer_function::ArrayTransferFunction;

#[cfg(kani)]
mod proofs {
    use super::*;

    #[kani::proof]
    pub fn prove_df1_fixed_update_total() {
        type Q = Fixed<i16, 13>;
        let raw: [i16; 10] = kani::any();
        let [b0, b1, b2, a1, a2, u1, u2, y1, y2, u] = raw.map(Q::from_bits);
        let mut s = Df1::from(SectionCoefficients { b0, b1, b2, a1, a2 });
        s.u1 = u1;
        s.u2 = u2;
        s.y1 = y1;
        s.y2 = y2;
        // The property is totality (VC-28.3): Kani checks every arithmetic
        // operation for overflow and every index for bounds, so reaching the
        // end of the harness for all inputs proves `update` cannot panic.
        // The output value is checked exactly by `df1_fixed_single_rounding`.
        let _ = s.update(u);
    }
}

mod sealed {
    /// Seals [`super::Section`] to the crate's section structures.
    pub trait Sealed {}
}

/// `2^52`: from here every `f64` is an integer.
const EXACT_INTEGERS: f64 = 4_503_599_627_370_496.0;

/// `L` section coefficient sets, first section first.
pub type SectionArray<T, const L: usize> = [SectionCoefficients<T>; L];

/// A slice of section coefficient sets.
pub type SectionSlice<T> = [SectionCoefficients<T>];

/// Quantized sections or the first section out of range.
pub type QuantizeResult<Repr, const SHIFT: usize, const L: usize> =
    Result<SectionArray<Fixed<Repr, SHIFT>, L>, ClassicalError>;

/// Factored sections or the reason factorization failed.
pub type SectionsResult<T, const L: usize> =
    Result<SectionArray<T, L>, ClassicalError>;

/// Accumulator states of a realization.
type AccState<T, const N: usize> = [<T as MulAcc>::Acc; N];

/// Coefficients of one section of type `S`.
type CoefficientsOf<S> = SectionCoefficients<<S as Section>::Sample>;

/// Coefficients of `L` sections of type `S`.
type CascadeCoefficients<S, const L: usize> =
    SectionArray<<S as Section>::Sample, L>;

/// Numerator and denominator arrays.
type ArrayPair<T, const D: usize> = ([T; D], [T; D]);

/// Root buffer.
pub(crate) type Roots<T, const D: usize> = [Complex<T>; D];

/// Remaining poles during pairing.
pub(crate) type PoleSlots<T, const D: usize> = [Option<Complex<T>>; D];

/// Remaining zeros during pairing.
pub(crate) type ZeroSlots<T, const D: usize> = [Zero<T>; D];

/// Zeros of a numerator or the reason they cannot be computed.
type ZerosResult<T, const D: usize> = Result<ZeroSlots<T, D>, ClassicalError>;

/// A pole taken from the remaining set.
type MaybePole<T> = Option<Complex<T>>;

/// First-order factor `[c_0, c_1]` in `q = z^-1`.
type Factor<T> = [Complex<T>; 2];

/// The two zeros of a section.
type ZeroPair<T> = [Zero<T>; 2];

/// The two poles of a section; `None` for a first-order section.
type PolePair<T> = [Option<Complex<T>>; 2];

/// Zeros and poles of one section.
type Pairing<T> = (ZeroPair<T>, PolePair<T>);

/// A zero slot during pairing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Zero<T> {
    /// A finite zero at the given location.
    At(Complex<T>),
    /// A zero at infinity: a pure delay factor `z^-1`.
    Infinite,
    /// Consumed or absent.
    Used,
}

/// Coefficients of one first- or second-order section in the
/// `1 + a_1 z^-1 + a_2 z^-2` denominator convention (FR-18).
///
/// A first-order section has `b_2 = a_2 = 0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SectionCoefficients<T> {
    /// `b_0`.
    pub b0: T,
    /// `b_1`.
    pub b1: T,
    /// `b_2`.
    pub b2: T,
    /// `a_1`.
    pub a1: T,
    /// `a_2`.
    pub a2: T,
}

/// Direct Form I second-order section (FR-17).
///
/// Holds `(b_0, b_1, b_2, -a_1, -a_2)` and
/// `(u_k-1, u_k-2, y_k-1, y_k-2)`; each output is one five-term
/// `MulAcc` chain narrowed once. DF1 keeps its state in signal units and
/// overflows internally only if the output does, which makes it the
/// structure for fixed point and for scheduled coefficients.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Df1<T> {
    pub(crate) b0: T,
    pub(crate) b1: T,
    pub(crate) b2: T,
    pub(crate) na1: T,
    pub(crate) na2: T,
    pub(crate) u1: T,
    pub(crate) u2: T,
    pub(crate) y1: T,
    pub(crate) y2: T,
}

/// Transposed Direct Form II second-order section.
///
/// The order-2 case of [`DirectForm2T`]: it holds
/// `(b_0, b_1, b_2, -a_1, -a_2)` and two states in `T::Acc`, the wide
/// dynamic range TDF-II states need; half the state of [`Df1`]. It adds the
/// [`Section`] interface that a [`BiquadCascade`] runs.
#[derive(Clone, Copy)]
pub struct Df2t<T: MulAcc>(pub(crate) DirectForm2T<T, 2>);

/// Transposed Direct Form II realization of order `ORDER` (FR-8).
#[derive(Clone, Copy)]
pub struct DirectForm2T<T: MulAcc, const ORDER: usize> {
    b0: T,
    b: [T; ORDER],
    na: [T; ORDER],
    pub(crate) state: AccState<T, ORDER>,
}

/// Second-order sections of one structure run in series (FR-9).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BiquadCascade<S, const L: usize> {
    sections: [S; L],
}

/// Run-time surface of a section: one sample per `update`, `reset` and
/// coefficient replacement that keeps the state (FR-18).
///
/// Sealed: implemented by [`Df1`] and [`Df2t`].
pub trait Section: sealed::Sealed {
    /// The sample and coefficient type.
    type Sample: Copy;
    /// Processes one input sample and returns the output.
    fn update(&mut self, u: Self::Sample) -> Self::Sample;
    /// Sets every state to zero.
    fn reset(&mut self);
    /// Replaces the coefficients without clearing the state.
    fn set_coefficients(&mut self, c: &CoefficientsOf<Self>);
    /// The coefficients in the `1 + a_1 z^-1 + a_2 z^-2` convention, so a
    /// running section can be checked with [`is_stable`] or copied.
    fn coefficients(&self) -> CoefficientsOf<Self>;
}

impl<T> sealed::Sealed for Df1<T> {}

impl<T: MulAcc> sealed::Sealed for Df2t<T> {}

impl<T: MulAcc + core::fmt::Debug, const ORDER: usize> core::fmt::Debug
    for DirectForm2T<T, ORDER>
where
    T::Acc: core::fmt::Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DirectForm2T")
            .field("b0", &self.b0)
            .field("b", &self.b)
            .field("na", &self.na)
            .field("state", &self.state)
            .finish()
    }
}

impl<T: MulAcc + PartialEq, const ORDER: usize> PartialEq
    for DirectForm2T<T, ORDER>
where
    T::Acc: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.b0 == other.b0
            && self.b == other.b
            && self.na == other.na
            && self.state == other.state
    }
}

impl<T: MulAcc + core::fmt::Debug> core::fmt::Debug for Df2t<T>
where
    T::Acc: core::fmt::Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("Df2t").field(&self.0).finish()
    }
}

impl<T: MulAcc + PartialEq> PartialEq for Df2t<T>
where
    T::Acc: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T: Scalar + Signed + MulAcc> From<SectionCoefficients<T>> for Df1<T> {
    fn from(c: SectionCoefficients<T>) -> Self {
        Self {
            b0: c.b0,
            b1: c.b1,
            b2: c.b2,
            na1: c.a1.saturating_neg(),
            na2: c.a2.saturating_neg(),
            u1: T::ZERO,
            u2: T::ZERO,
            y1: T::ZERO,
            y2: T::ZERO,
        }
    }
}

impl<T: Scalar + Signed + MulAcc> From<SectionCoefficients<T>> for Df2t<T> {
    fn from(c: SectionCoefficients<T>) -> Self {
        Self(DirectForm2T::new(c.b0, [c.b1, c.b2], [c.a1, c.a2]))
    }
}

impl<T: Scalar + Signed + MulAcc> Section for Df1<T> {
    type Sample = T;

    #[inline]
    fn update(&mut self, u: T) -> T {
        let acc = T::mac(T::ZERO.to_acc(), self.b0, u);
        let acc = T::mac(acc, self.b1, self.u1);
        let acc = T::mac(acc, self.b2, self.u2);
        let acc = T::mac(acc, self.na1, self.y1);
        let y = T::from_acc(T::mac(acc, self.na2, self.y2));
        self.u2 = self.u1;
        self.u1 = u;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }

    fn reset(&mut self) {
        self.u1 = T::ZERO;
        self.u2 = T::ZERO;
        self.y1 = T::ZERO;
        self.y2 = T::ZERO;
    }

    fn set_coefficients(&mut self, c: &SectionCoefficients<T>) {
        self.b0 = c.b0;
        self.b1 = c.b1;
        self.b2 = c.b2;
        self.na1 = c.a1.saturating_neg();
        self.na2 = c.a2.saturating_neg();
    }

    fn coefficients(&self) -> SectionCoefficients<T> {
        SectionCoefficients {
            b0: self.b0,
            b1: self.b1,
            b2: self.b2,
            a1: self.na1.saturating_neg(),
            a2: self.na2.saturating_neg(),
        }
    }
}

impl<T: Scalar + Signed + MulAcc> Section for Df2t<T> {
    type Sample = T;

    #[inline]
    fn update(&mut self, u: T) -> T {
        self.0.update(u)
    }

    fn reset(&mut self) {
        self.0.reset();
    }

    fn set_coefficients(&mut self, c: &SectionCoefficients<T>) {
        self.0.set_coefficients(c.b0, [c.b1, c.b2], [c.a1, c.a2]);
    }

    fn coefficients(&self) -> SectionCoefficients<T> {
        let [a1, a2] = self.0.na.map(|v| v.saturating_neg());
        let [b1, b2] = self.0.b;
        SectionCoefficients {
            b0: self.0.b0,
            b1,
            b2,
            a1,
            a2,
        }
    }
}

impl<T: Scalar + Signed + MulAcc, const ORDER: usize> DirectForm2T<T, ORDER> {
    /// Builds a realization from `b_0`, `(b_1, ..., b_n)` and
    /// `(a_1, ..., a_n)` with zero state.
    ///
    /// Stores `-a_i` for the recurrence. `T: Signed` is required so the
    /// negation is not the unsigned `Fixed` no-op that clears feedback.
    #[must_use]
    pub fn new(b0: T, b: [T; ORDER], a: [T; ORDER]) -> Self {
        Self {
            b0,
            b,
            na: a.map(|v| v.saturating_neg()),
            state: [T::ZERO.to_acc(); ORDER],
        }
    }

    /// Processes one input sample and returns the output.
    #[inline]
    pub fn update(&mut self, u: T) -> T {
        let zero = T::ZERO.to_acc();
        let first = self.state.first().copied().unwrap_or(zero);
        let y = T::from_acc(T::mac(first, self.b0, u));
        for i in 0..ORDER {
            let next =
                self.state.get(i.saturating_add(1)).copied().unwrap_or(zero);
            if let (Some(d), Some(&bi), Some(&ai)) =
                (self.state.get_mut(i), self.b.get(i), self.na.get(i))
            {
                *d = T::mac(T::mac(next, bi, u), ai, y);
            }
        }
        y
    }

    /// Sets every state to zero.
    pub fn reset(&mut self) {
        self.state = [T::ZERO.to_acc(); ORDER];
    }

    /// Replaces the coefficients without clearing the state (FR-18).
    pub fn set_coefficients(&mut self, b0: T, b: [T; ORDER], a: [T; ORDER]) {
        self.b0 = b0;
        self.b = b;
        self.na = a.map(|v| v.saturating_neg());
    }
}

impl<T: Float + Copy + MulAcc, const ORDER: usize> DirectForm2T<T, ORDER> {
    /// Builds a realization from a discrete `TransferFunction` divided
    /// through by its leading denominator coefficient (FR-8).
    ///
    /// `ORDER` is the denominator degree `D - 1` and `N <= D`. This
    /// constructor divides coefficients and tests the leading one, so it
    /// needs `T: Float`; [`new`](Self::new) accepts any
    /// `T: Scalar + Signed + MulAcc`, including `Fixed`.
    ///
    /// # Errors
    /// - [`ClassicalError::NotDiscrete`] when `tf` is continuous.
    /// - [`ClassicalError::ZeroLeadingCoefficient`] when the leading
    ///   denominator coefficient has absolute value at most `T::epsilon()`.
    pub fn from_transfer_function<const N: usize, const D: usize>(
        tf: &ArrayTransferFunction<T, N, D>,
    ) -> Result<Self, ClassicalError>
    where
        Const<N>: Dim,
        Const<ORDER>: Dim,
        TypeNum<ORDER>: DimAdd<Const<1>>,
        Const<D>: Dim<TypeNum = Succ<ORDER>>,
        TypeNum<N>: DimMax<TypeNum<D>, Output = TypeNum<D>>,
    {
        if tf.is_continuous() {
            return Err(ClassicalError::NotDiscrete);
        }
        let (num, den) = descending::<T, N, D>(tf);
        let lead = den.first().copied().unwrap_or(T::ONE);
        match lead.abs().partial_cmp(&T::epsilon()) {
            Some(core::cmp::Ordering::Greater) => {}
            _ => return Err(ClassicalError::ZeroLeadingCoefficient),
        }
        let scale = |v: &T| v.saturating_div(&lead);
        let mut b = [T::ZERO; ORDER];
        let mut a = [T::ZERO; ORDER];
        b.iter_mut()
            .zip(num.iter().skip(1))
            .for_each(|(o, v)| *o = scale(v));
        a.iter_mut()
            .zip(den.iter().skip(1))
            .for_each(|(o, v)| *o = scale(v));
        let b0 = num.first().map_or(T::ZERO, scale);
        Ok(Self::new(b0, b, a))
    }
}

impl<S, const L: usize> BiquadCascade<S, L> {
    /// Builds a cascade from its sections, first section first.
    #[must_use]
    pub const fn new(sections: [S; L]) -> Self {
        Self { sections }
    }

    /// Builds a cascade with zero state from section coefficients.
    #[must_use]
    pub fn from_coefficients<T: Copy>(c: &SectionArray<T, L>) -> Self
    where
        S: From<SectionCoefficients<T>>,
    {
        Self {
            sections: c.map(S::from),
        }
    }

    /// The sections in execution order.
    #[must_use]
    pub const fn sections(&self) -> &[S; L] {
        &self.sections
    }
}

impl<S: Section, const L: usize> BiquadCascade<S, L> {
    /// Processes one input sample through every section in order.
    #[inline]
    pub fn update(&mut self, u: S::Sample) -> S::Sample {
        self.sections.iter_mut().fold(u, |x, s| s.update(x))
    }

    /// Sets every section state to zero.
    pub fn reset(&mut self) {
        self.sections.iter_mut().for_each(S::reset);
    }

    /// Replaces every section's coefficients without clearing state
    /// (FR-18).
    pub fn set_coefficients(&mut self, c: &CascadeCoefficients<S, L>) {
        self.sections
            .iter_mut()
            .zip(c)
            .for_each(|(s, ci)| s.set_coefficients(ci));
    }
}

/// Reports whether a first- or second-order section has its poles strictly
/// inside the unit circle: `|a_2| < 1` and `|a_1| < 1 + a_2` (FR-20).
///
/// Exact for `Fixed`; no root finding. Requires [`Signed`] so negative
/// coefficients use a true absolute value rather than the unsigned
/// `saturating_neg` no-op.
#[must_use]
pub fn is_stable<T: Scalar + Signed + Copy>(
    c: &SectionCoefficients<T>,
) -> bool {
    Signed::abs(c.a2) < T::ONE
        && Signed::abs(c.a1) < T::ONE.saturating_add(&c.a2)
}

/// Reports whether every section in `sections` passes [`is_stable`]
/// (FR-20).
#[must_use]
pub fn all_stable<T: Scalar + Signed + Copy>(
    sections: &SectionSlice<T>,
) -> bool {
    sections.iter().all(is_stable)
}

/// Quantizes floating-point section coefficients to `Fixed<Repr, SHIFT>`
/// with round-to-nearest (FR-19).
///
/// `Fixed<Repr, SHIFT>` must be [`Signed`]: unsigned fixed-point cannot
/// hold the negative feedback coefficients that IIR sections require, and
/// its `saturating_neg` would clear them at realization time (C-8).
///
/// # Errors
/// [`ClassicalError::CoefficientRange`] naming the first section with a
/// coefficient that does not round into the representable range; no
/// coefficient is saturated. The comparison is made in raw units, so the
/// `f64` image of `Fixed::MIN` and `Fixed::MAX` is accepted for every
/// `Repr`, including `i64`, where `MAX` has no exact `f64`.
pub fn quantize<Repr: FixedRepr, const SHIFT: usize, const L: usize>(
    sections: &SectionArray<f64, L>,
) -> QuantizeResult<Repr, SHIFT, L>
where
    Const<SHIFT>: Dim + DimMax<Repr::BitsDim, Output = Repr::BitsDim>,
    Fixed<Repr, SHIFT>: Scalar + Signed,
{
    let delta = Fixed::<Repr, SHIFT>::DELTA.to_num();
    // Bounds in raw units (multiples of delta). Scaling by a power of two is
    // exact, so the only inexact bound is `MAX` of a type wider than the f64
    // significand, whose nearest f64 is `MAX + 1` and is accepted: it is the
    // image of `MAX` and saturates to it.
    let max_raw = Fixed::<Repr, SHIFT>::MAX.to_num() / delta;
    let min_raw = Fixed::<Repr, SHIFT>::MIN.to_num() / delta;
    let in_range = |v: f64| {
        let x = v / delta;
        // Above 2^52 every f64 is an integer and half-way points do not
        // exist, so the rounding margin is dropped there.
        if x.abs() >= EXACT_INTEGERS {
            x >= min_raw && x <= max_raw
        } else {
            x >= min_raw - 0.5 && x < max_raw + 0.5
        }
    };
    let mut out = [SectionCoefficients {
        b0: Fixed::ZERO,
        b1: Fixed::ZERO,
        b2: Fixed::ZERO,
        a1: Fixed::ZERO,
        a2: Fixed::ZERO,
    }; L];
    for ((dst, src), section) in out.iter_mut().zip(sections).zip(0usize..) {
        let all = [src.b0, src.b1, src.b2, src.a1, src.a2];
        if !all.iter().all(|v| in_range(*v)) {
            return Err(ClassicalError::CoefficientRange { section });
        }
        let [b0, b1, b2, a1, a2] = all.map(Fixed::from_num);
        *dst = SectionCoefficients { b0, b1, b2, a1, a2 };
    }
    Ok(out)
}

/// The section count `L` that [`to_sections`] needs for a denominator of `d`
/// coefficients: `max(1, ceil((d - 1) / 2))`.
///
/// Order 0 has no poles to pair, and one gain-only section keeps the overall
/// gain that an empty cascade would drop.
#[must_use]
pub const fn section_count(d: usize) -> usize {
    let order = d.saturating_sub(1);
    if order == 0 { 1 } else { order.div_ceil(2) }
}

/// Factors a discrete `TransferFunction` into `L` second-order sections
/// (FR-10).
///
/// Poles and zeros are paired as `SciPy` `zpk2sos` pairs them with
/// `pairing = "nearest"`: sections are formed starting with the pole
/// closest to the unit circle, each paired with its nearest remaining zero
/// and a conjugate or next real pole, and are placed from the last section
/// backward. For odd order the last remaining real pole forms the one
/// first-order section (`b_2 = a_2 = 0`) with its nearest real zero. The
/// overall gain goes into the first section. Zeros at infinity (numerator
/// degree below denominator degree) become pure delays in their section.
/// `L` must equal `max(1, ceil((D - 1) / 2))`, which [`section_count`]
/// computes, so a constant (`D = 1`) still has one section to hold the
/// overall gain. A caller writes it as a const argument:
///
/// ```
/// use control_rs::classical_control::{section_count, to_sections};
/// use control_rs::transfer_function::ArrayTransferFunction;
///
/// // H(z) = 1 / (1 - 0.5 z^-1 + 0.06 z^-2), order 2: one section.
/// let tf = ArrayTransferFunction::<f64, 1, 3>::discrete(
///     [1.0],
///     [0.06, -0.5, 1.0],
///     0.1,
/// );
/// let sections = to_sections::<f64, 1, 3, { section_count(3) }>(&tf)?;
/// assert_eq!(sections.len(), 1);
/// # Ok::<(), control_rs::classical_control::ClassicalError>(())
/// ```
///
/// # Errors
/// - [`ClassicalError::NotDiscrete`]: `tf` is continuous.
/// - [`ClassicalError::SectionCount`]: `L` does not match the order.
/// - [`ClassicalError::Root`]: poles or zeros cannot be computed, or the
///   numerator is zero.
pub fn to_sections<
    T: Float + Copy,
    const N: usize,
    const D: usize,
    const L: usize,
>(
    tf: &ArrayTransferFunction<T, N, D>,
) -> SectionsResult<T, L>
where
    Const<N>: Dim,
    Const<D>: Dim,
    TypeNum<N>: DimMax<TypeNum<D>, Output = TypeNum<D>>,
{
    if tf.is_continuous() {
        return Err(ClassicalError::NotDiscrete);
    }
    let order = D.saturating_sub(1);
    if section_count(D) != L {
        return Err(ClassicalError::SectionCount);
    }
    let mut num = [T::ZERO; D];
    num.iter_mut()
        .zip(tf.num_slice())
        .for_each(|(o, &v)| *o = v);
    let mut poles = [None; D];
    let all_poles = tf.poles()?;
    for (dst, p) in poles.iter_mut().zip(all_poles).take(order) {
        *dst = Some(p);
    }
    let mut zeros = finite_zeros(&num, order)?;
    let lead_num = num
        .iter()
        .rev()
        .find(|v| **v != T::ZERO)
        .copied()
        .unwrap_or(T::ZERO);
    let lead_den = tf.den_slice().last().copied().unwrap_or(T::ONE);
    let mut out = pair_sections::<T, D, L>(&mut poles, &mut zeros);
    if let Some(first) = out.first_mut() {
        let g = lead_num.saturating_div(&lead_den);
        first.b0 = first.b0.saturating_mul(&g);
        first.b1 = first.b1.saturating_mul(&g);
        first.b2 = first.b2.saturating_mul(&g);
    }
    Ok(out)
}

/// Numerator and denominator in descending powers of `z`, padded to `D`.
fn descending<T: Float + Copy, const N: usize, const D: usize>(
    tf: &ArrayTransferFunction<T, N, D>,
) -> ArrayPair<T, D>
where
    Const<N>: Dim,
    Const<D>: Dim,
{
    let mut num = [T::ZERO; D];
    let mut den = [T::ZERO; D];
    num.iter_mut()
        .rev()
        .zip(tf.num_slice())
        .for_each(|(o, &v)| *o = v);
    den.iter_mut()
        .rev()
        .zip(tf.den_slice())
        .for_each(|(o, &v)| *o = v);
    (num, den)
}

/// Zeros of the ascending numerator `num` of a degree-`order` system:
/// exact zeros at the origin, finite zeros from the reversed polynomial and
/// zeros at infinity for each missing leading coefficient.
fn finite_zeros<T: Float + Copy, const D: usize>(
    num: &[T; D],
    order: usize,
) -> ZerosResult<T, D>
where
    Const<D>: Dim,
{
    let top = num.iter().rposition(|v| *v != T::ZERO);
    let low = num.iter().position(|v| *v != T::ZERO);
    let (Some(top), Some(low)) = (top, low) else {
        return Err(ClassicalError::Root(
            crate::polynomial::RootError::ZeroLeadingCoefficient,
        ));
    };
    // R(w) = Σ_j num[j + low] w^(order - j): nonzero leading coefficient.
    let mut rev = [T::ZERO; D];
    for (j, &c) in num.iter().skip(low).enumerate() {
        if let Some(slot) = rev.get_mut(order.saturating_sub(j)) {
            *slot = c;
        }
    }
    let mut w = ArrayPolynomial::<T, D>::from_coefficients(rev).roots()?;
    let finite = top.saturating_sub(low);
    sort_by_magnitude_desc(&mut w, order);
    let mut out = [Zero::Used; D];
    let one = Complex::new(T::ONE, T::ZERO);
    let kinds = w
        .iter()
        .take(finite)
        .map(|wi| Zero::At(one.saturating_div(wi)))
        .chain((0..low).map(|_| Zero::At(Complex::new(T::ZERO, T::ZERO))))
        .chain((0..order.saturating_sub(top)).map(|_| Zero::Infinite));
    out.iter_mut().zip(kinds).for_each(|(o, k)| *o = k);
    Ok(out)
}

/// Sorts the first `n` entries by descending magnitude (insertion sort,
/// initialization only).
pub(crate) fn sort_by_magnitude_desc<T: Float + Copy, const D: usize>(
    roots: &mut Roots<T, D>,
    count: usize,
) {
    let head = roots.get_mut(..count).unwrap_or(&mut []);
    for start in 1..head.len() {
        let mut pos = start;
        while pos > 0 {
            let prev = pos.saturating_sub(1);
            let swap = matches!(
                (head.get(prev), head.get(pos)),
                (Some(lhs), Some(rhs)) if rhs.magnitude() > lhs.magnitude()
            );
            if !swap {
                break;
            }
            head.swap(prev, pos);
            pos = prev;
        }
    }
}

/// Whether `z` is real within `sqrt(eps) max(1, |z|)`.
pub(crate) fn is_real<T: Float + Copy>(z: Complex<T>) -> bool {
    let mag = z.magnitude();
    let scale = if mag > T::ONE { mag } else { T::ONE };
    z.im.abs() <= T::epsilon().sqrt().saturating_mul(&scale)
}

/// Whether a zero slot is real (finite real or at infinity).
fn zero_is_real<T: Float + Copy>(z: Zero<T>) -> bool {
    match z {
        Zero::At(c) => is_real(c),
        Zero::Infinite => true,
        Zero::Used => false,
    }
}

/// Removes and returns the remaining pole minimizing `score` among those
/// passing `keep`.
pub(crate) fn take_pole<T: Float + Copy, const D: usize>(
    poles: &mut PoleSlots<T, D>,
    keep: impl Fn(Complex<T>) -> bool,
    score: impl Fn(Complex<T>) -> T,
) -> MaybePole<T> {
    let mut best = None;
    for (i, p) in poles.iter().enumerate() {
        if let Some(p) = p.filter(|p| keep(*p)) {
            let sc = score(p);
            if best.is_none_or(|(_, b)| sc < b) {
                best = Some((i, sc));
            }
        }
    }
    best.and_then(|(i, _)| poles.get_mut(i))
        .and_then(Option::take)
}

/// Removes and returns the remaining zero nearest `target` among those
/// passing `keep`; zeros at infinity rank after every finite zero.
pub(crate) fn take_zero<T: Float + Copy, const D: usize>(
    zeros: &mut ZeroSlots<T, D>,
    keep: impl Fn(Zero<T>) -> bool,
    target: Complex<T>,
) -> Zero<T> {
    let mut best = None;
    for (i, z) in zeros.iter().enumerate() {
        let sc = match *z {
            Zero::Used => continue,
            _ if !keep(*z) => continue,
            Zero::At(c) => Some(c.saturating_sub(&target).magnitude()),
            Zero::Infinite => None,
        };
        let better = match (best, sc) {
            (None, _) | (Some((_, None)), Some(_)) => true,
            (Some((_, Some(b))), Some(v)) => v < b,
            _ => false,
        };
        if better {
            best = Some((i, sc));
        }
    }
    best.and_then(|(i, _)| zeros.get_mut(i))
        .map_or(Zero::Used, |z| core::mem::replace(z, Zero::Used))
}

/// Factor `(1 - z q)` of a finite zero, `q` of a zero at infinity and `1`
/// of an absent zero, with `q = z^-1`.
fn zero_factor<T: Float + Copy>(z: Zero<T>) -> Factor<T> {
    let zero = Complex::new(T::ZERO, T::ZERO);
    let one = Complex::new(T::ONE, T::ZERO);
    match z {
        Zero::At(c) => [one, zero.saturating_sub(&c)],
        Zero::Infinite => [zero, one],
        Zero::Used => [one, zero],
    }
}

/// Real coefficients of the product of two first-order factors.
fn product<T: Float + Copy>(f: Factor<T>, g: Factor<T>) -> [T; 3] {
    let [f0, f1] = f;
    let [g0, g1] = g;
    [
        f0.saturating_mul(&g0).re,
        f0.saturating_mul(&g1)
            .saturating_add(&f1.saturating_mul(&g0))
            .re,
        f1.saturating_mul(&g1).re,
    ]
}

/// Section with zeros `z` and poles `p` (an absent pole contributes 1).
fn section<T: Float + Copy>(
    z: ZeroPair<T>,
    p: PolePair<T>,
) -> SectionCoefficients<T> {
    let [z1, z2] = z;
    let [b0, b1, b2] = product(zero_factor(z1), zero_factor(z2));
    let first_order = p[1].is_none();
    let [p1, p2] = p.map(|q| zero_factor(q.map_or(Zero::Used, Zero::At)));
    let [_, a1, a2] = product(p1, p2);
    let exact = |v: T, absent: bool| if absent { T::ZERO } else { v };
    SectionCoefficients {
        b0,
        b1,
        b2: exact(b2, matches!(z2, Zero::Used)),
        a1,
        a2: exact(a2, first_order),
    }
}

/// Distance of `p` from the unit circle.
fn circle_distance<T: Float + Copy>(p: Complex<T>) -> T {
    p.magnitude().saturating_sub(&T::ONE).abs()
}

/// Pairs the remaining poles and zeros into `L` sections, filling from the
/// last section backward.
fn pair_sections<T: Float + Copy, const D: usize, const L: usize>(
    poles: &mut PoleSlots<T, D>,
    zeros: &mut ZeroSlots<T, D>,
) -> SectionArray<T, L> {
    let identity = SectionCoefficients {
        b0: T::ONE,
        b1: T::ZERO,
        b2: T::ZERO,
        a1: T::ZERO,
        a2: T::ZERO,
    };
    let mut out = [identity; L];
    for dst in out.iter_mut().rev() {
        let Some(p1) = take_pole(poles, |_| true, circle_distance) else {
            break;
        };
        let (z, p) = pair_one(poles, zeros, p1);
        *dst = section(z, p);
    }
    out
}

/// Chooses the partner pole and both zeros of a section led by `p1`.
fn pair_one<T: Float + Copy, const D: usize>(
    poles: &mut PoleSlots<T, D>,
    zeros: &mut ZeroSlots<T, D>,
    p1: Complex<T>,
) -> Pairing<T> {
    let real_poles_left = poles.iter().flatten().any(|p| is_real(*p));
    if is_real(p1) && !real_poles_left {
        let z1 = take_zero(zeros, zero_is_real, p1);
        return ([z1, Zero::Used], [Some(p1), None]);
    }
    // With exactly one real zero left (finite or at infinity), prefer a
    // complex zero so a later real pole can claim that real zero. Applies
    // for both complex and real `p1`: a real `p1` with other real poles
    // remaining would otherwise take the last real zero and leave `z2`
    // empty, silently dropping the complex zero pair.
    let real_zeros = zeros.iter().filter(|z| zero_is_real(**z)).count();
    let z1 = if real_zeros == 1 {
        take_zero(zeros, |z| matches!(z, Zero::At(c) if !is_real(c)), p1)
    } else {
        take_zero(zeros, |_| true, p1)
    };
    let z1_real = zero_is_real(z1);
    if !is_real(p1) {
        let p2 = take_pole(
            poles,
            |_| true,
            |q| q.saturating_sub(&p1.conj()).magnitude(),
        );
        let z2 = match z1 {
            Zero::At(c) if !z1_real => take_zero(zeros, |_| true, c.conj()),
            _ => take_zero(zeros, zero_is_real, p1),
        };
        return ([z1, z2], [Some(p1), p2]);
    }
    if let (Zero::At(c), false) = (z1, z1_real) {
        let z2 = take_zero(zeros, |_| true, c.conj());
        let p2 =
            take_pole(poles, is_real, |q| q.saturating_sub(&c).magnitude());
        return ([z1, z2], [Some(p1), p2]);
    }
    let p2 = take_pole(poles, is_real, circle_distance);
    let z2 = take_zero(zeros, zero_is_real, p2.unwrap_or(p1));
    ([z1, z2], [Some(p1), p2])
}
