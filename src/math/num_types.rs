//! # Numeric Types (Type-Level Math)
//!
//! Defines dimension types and bounds for matrix and tensor storage using
//! compile-time binary (typenum-style) unsigned arithmetic.
//!
//! Representation depth is O(log N). Named aliases cover `U0..=U1024` plus
//! `U2048`, `U4096`, `U8192`, and `U16384`. Products such as `U128 * U128`
//! resolve even when the result has no name other than those extras.
//!
//! `Const<N>` is a ZST const-generic bridge onto the canonical `UInt`/`UTerm`
//! tree. It has no runtime constructor: a runtime integer cannot be tied to
//! `N` in a way release builds or the type system enforce.
//!
//! Subtraction underflow has no implementation and fails to compile:
//!
//! ```compile_fail
//! use control_rs::math::num_types::{DimSub, U2, U5};
//!
//! let _: <U2 as DimSub<U5>>::Output = Default::default();
//! ```
//!
//! Dimensions outside the supported `C-1` range do not implement [`Dim`] and
//! fail upfront at compile time:
//!
//! ```compile_fail
//! use control_rs::math::num_types::{Const, Dim};
//!
//! fn assert_dim<D: Dim>() {}
//! assert_dim::<Const<1025>>();
//! ```
//!
//! Const values whose product falls outside C-1 do not implement [`Dim`] and fail to compile:
//!
//! ```compile_fail
//! use control_rs::math::num_types::{Const, Dim};
//!
//! fn assert_dim<D: Dim>() {}
//! // 100 * 100 = 10000, which is outside C-1
//! assert_dim::<Const<10000>>();
//! ```
#![allow(clippy::arbitrary_source_item_ordering)]
#![allow(clippy::type_complexity)]
#![allow(clippy::use_self)]
#![allow(clippy::manual_div_ceil)]
#![allow(unused_macro_rules)]

use core::marker::PhantomData;

use private::{
    AddBit, Cmp, Compare, PrivateAnd, PrivateAndOut, PrivateMax, PrivateMaxOut,
    PrivateMin, PrivateMinOut, PrivateOr, PrivateOrOut, PrivateSub,
    PrivateSubOut, PrivateXor, PrivateXorOut, SelectBit, Trim, TrimOut,
};

/// Type-level bit 0. Does not implement [`Dim`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct B0;

/// Type-level bit 1. Does not implement [`Dim`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct B1;

/// Zero in the canonical unsigned encoding. Aliased as [`U0`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct UTerm;

/// Binary unsigned integer: value `(U << 1) | B` with `B` the least
/// significant bit. Leading zeros are not canonical (`UInt<UTerm, B0>` is
/// not a value).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct UInt<U, B>(PhantomData<(U, B)>);

/// A compile-time dimension: a `usize` value and a canonical type-level
/// encoding.
pub trait Dim: Clone + Copy + PartialEq + Eq {
    /// Runtime value of the dimension.
    const USIZE: usize;
    /// Canonical `UTerm` / `UInt` encoding of the dimension.
    type TypeNum: Dim;
}

/// Type-level addition.
pub trait DimAdd<Other> {
    /// Sum, itself a dimension.
    type Output: Dim;
}

/// Type-level subtraction.
pub trait DimSub<Other> {
    /// Difference, itself a dimension.
    type Output: Dim;
}

/// Type-level multiplication.
pub trait DimMul<Other> {
    /// Product, itself a dimension.
    type Output: Dim;
}

/// Type-level maximum.
pub trait DimMax<Other> {
    /// The larger operand.
    type Output: Dim;
}

/// Type-level minimum.
pub trait DimMin<Other> {
    /// The smaller operand.
    type Output: Dim;
}

/// Type-level bitwise AND.
pub trait DimBitAnd<Other> {
    /// Bitwise AND, itself a dimension.
    type Output: Dim;
}

/// Type-level bitwise OR.
pub trait DimBitOr<Other> {
    /// Bitwise OR, itself a dimension.
    type Output: Dim;
}

/// Type-level bitwise XOR.
pub trait DimBitXor<Other> {
    /// Bitwise XOR, itself a dimension.
    type Output: Dim;
}

/// Const-generic front end onto a canonical [`Dim`] encoding.
///
/// `Const<N>` implements [`Dim`] only for `N` in the named-alias set
/// (`0..=1024` and the extra powers of two through `16384`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct Const<const N: usize>;

////////////////////////////////////////////////////////////////////////////////
// Dim
////////////////////////////////////////////////////////////////////////////////

/// A type-level bit: `0` or `1`.
pub trait Bit:
    Clone + Copy + PartialEq + Eq + Default + core::hash::Hash
{
    /// Runtime value of the bit.
    const USIZE: usize;
}

impl Bit for B0 {
    const USIZE: usize = 0;
}

impl Bit for B1 {
    const USIZE: usize = 1;
}

impl Dim for UTerm {
    const USIZE: usize = 0;
    type TypeNum = Self;
}

impl<U: Dim, B: Bit> Dim for UInt<U, B> {
    #[allow(clippy::arithmetic_side_effects)]
    const USIZE: usize = U::USIZE * 2 + B::USIZE;
    type TypeNum = Self;
}

////////////////////////////////////////////////////////////////////////////////
// DimAdd
////////////////////////////////////////////////////////////////////////////////

impl DimAdd<UTerm> for UTerm {
    type Output = UTerm;
}

impl<U, B> DimAdd<UInt<U, B>> for UTerm
where
    UInt<U, B>: Dim,
{
    type Output = UInt<U, B>;
}

impl<U, B> DimAdd<UTerm> for UInt<U, B>
where
    Self: Dim,
{
    type Output = Self;
}

impl<Ul, Ur> DimAdd<UInt<Ur, B0>> for UInt<Ul, B0>
where
    Ul: DimAdd<Ur>,
    <Ul as DimAdd<Ur>>::Output: Dim,
{
    type Output = UInt<<Ul as DimAdd<Ur>>::Output, B0>;
}

impl<Ul, Ur> DimAdd<UInt<Ur, B1>> for UInt<Ul, B0>
where
    Ul: DimAdd<Ur>,
    <Ul as DimAdd<Ur>>::Output: Dim,
{
    type Output = UInt<<Ul as DimAdd<Ur>>::Output, B1>;
}

impl<Ul, Ur> DimAdd<UInt<Ur, B0>> for UInt<Ul, B1>
where
    Ul: DimAdd<Ur>,
    <Ul as DimAdd<Ur>>::Output: Dim,
{
    type Output = UInt<<Ul as DimAdd<Ur>>::Output, B1>;
}

impl<Ul, Ur> DimAdd<UInt<Ur, B1>> for UInt<Ul, B1>
where
    Ul: DimAdd<Ur>,
    <Ul as DimAdd<Ur>>::Output: AddBit<B1>,
    <<Ul as DimAdd<Ur>>::Output as AddBit<B1>>::Output: Dim,
{
    type Output = UInt<<<Ul as DimAdd<Ur>>::Output as AddBit<B1>>::Output, B0>;
}

////////////////////////////////////////////////////////////////////////////////
// DimSub
////////////////////////////////////////////////////////////////////////////////

impl DimSub<UTerm> for UTerm {
    type Output = UTerm;
}

impl<U, B> DimSub<UTerm> for UInt<U, B>
where
    Self: Dim,
{
    type Output = Self;
}

impl<Ul, Bl, Ur, Br> DimSub<UInt<Ur, Br>> for UInt<Ul, Bl>
where
    Self: PrivateSub<UInt<Ur, Br>> + Dim,
    PrivateSubOut<Self, UInt<Ur, Br>>: Trim,
    TrimOut<PrivateSubOut<Self, UInt<Ur, Br>>>: Dim,
{
    type Output = TrimOut<PrivateSubOut<Self, UInt<Ur, Br>>>;
}

////////////////////////////////////////////////////////////////////////////////
// DimMul
////////////////////////////////////////////////////////////////////////////////

impl DimMul<UTerm> for UTerm {
    type Output = UTerm;
}

impl<U, B> DimMul<UInt<U, B>> for UTerm {
    type Output = UTerm;
}

impl<U, B> DimMul<UTerm> for UInt<U, B>
where
    Self: Dim,
{
    type Output = UTerm;
}

impl<Ul, B, Ur> DimMul<UInt<Ur, B>> for UInt<Ul, B0>
where
    Ul: DimMul<UInt<Ur, B>>,
    <Ul as DimMul<UInt<Ur, B>>>::Output: Dim,
{
    type Output = UInt<<Ul as DimMul<UInt<Ur, B>>>::Output, B0>;
}

impl<Ul, B, Ur> DimMul<UInt<Ur, B>> for UInt<Ul, B1>
where
    Ul: DimMul<UInt<Ur, B>>,
    <Ul as DimMul<UInt<Ur, B>>>::Output: Dim,
    UInt<<Ul as DimMul<UInt<Ur, B>>>::Output, B0>: DimAdd<UInt<Ur, B>>,
{
    type Output = <UInt<<Ul as DimMul<UInt<Ur, B>>>::Output, B0> as DimAdd<
        UInt<Ur, B>,
    >>::Output;
}

////////////////////////////////////////////////////////////////////////////////
// DimMax / DimMin
////////////////////////////////////////////////////////////////////////////////

impl DimMax<UTerm> for UTerm {
    type Output = UTerm;
}

impl<U, B> DimMax<UInt<U, B>> for UTerm
where
    UInt<U, B>: Dim,
{
    type Output = UInt<U, B>;
}

impl<U, B> DimMax<UTerm> for UInt<U, B>
where
    Self: Dim,
{
    type Output = Self;
}

impl<Ul, Bl, Ur, Br> DimMax<UInt<Ur, Br>> for UInt<Ul, Bl>
where
    Self: Cmp<UInt<Ur, Br>>
        + PrivateMax<UInt<Ur, Br>, Compare<Self, UInt<Ur, Br>>>
        + Dim,
    PrivateMaxOut<Self, UInt<Ur, Br>, Compare<Self, UInt<Ur, Br>>>: Dim,
{
    type Output =
        PrivateMaxOut<Self, UInt<Ur, Br>, Compare<Self, UInt<Ur, Br>>>;
}

impl DimMin<UTerm> for UTerm {
    type Output = UTerm;
}

impl<U, B> DimMin<UInt<U, B>> for UTerm
where
    UInt<U, B>: Dim,
{
    type Output = UTerm;
}

impl<U, B> DimMin<UTerm> for UInt<U, B>
where
    Self: Dim,
{
    type Output = UTerm;
}

impl<Ul, Bl, Ur, Br> DimMin<UInt<Ur, Br>> for UInt<Ul, Bl>
where
    Self: Cmp<UInt<Ur, Br>>
        + PrivateMin<UInt<Ur, Br>, Compare<Self, UInt<Ur, Br>>>
        + Dim,
    PrivateMinOut<Self, UInt<Ur, Br>, Compare<Self, UInt<Ur, Br>>>: Dim,
{
    type Output =
        PrivateMinOut<Self, UInt<Ur, Br>, Compare<Self, UInt<Ur, Br>>>;
}

////////////////////////////////////////////////////////////////////////////////
// DimBitAnd
////////////////////////////////////////////////////////////////////////////////

impl DimBitAnd<UTerm> for UTerm {
    type Output = UTerm;
}

impl<U, B> DimBitAnd<UInt<U, B>> for UTerm
where
    UInt<U, B>: Dim,
{
    type Output = UTerm;
}

impl<U, B> DimBitAnd<UTerm> for UInt<U, B>
where
    Self: Dim,
{
    type Output = UTerm;
}

impl<Ul, Bl, Ur, Br> DimBitAnd<UInt<Ur, Br>> for UInt<Ul, Bl>
where
    Self: Dim + PrivateAnd<UInt<Ur, Br>>,
    UInt<Ur, Br>: Dim,
    PrivateAndOut<Self, UInt<Ur, Br>>: Trim,
    TrimOut<PrivateAndOut<Self, UInt<Ur, Br>>>: Dim,
{
    type Output = TrimOut<PrivateAndOut<Self, UInt<Ur, Br>>>;
}

////////////////////////////////////////////////////////////////////////////////
// DimBitOr
////////////////////////////////////////////////////////////////////////////////

impl DimBitOr<UTerm> for UTerm {
    type Output = UTerm;
}

impl<U, B> DimBitOr<UInt<U, B>> for UTerm
where
    UInt<U, B>: Dim,
{
    type Output = UInt<U, B>;
}

impl<U, B> DimBitOr<UTerm> for UInt<U, B>
where
    Self: Dim,
{
    type Output = Self;
}

impl<Ul, Bl, Ur, Br> DimBitOr<UInt<Ur, Br>> for UInt<Ul, Bl>
where
    Self: Dim + PrivateOr<UInt<Ur, Br>>,
    UInt<Ur, Br>: Dim,
    PrivateOrOut<Self, UInt<Ur, Br>>: Dim,
{
    type Output = PrivateOrOut<Self, UInt<Ur, Br>>;
}

////////////////////////////////////////////////////////////////////////////////
// DimBitXor
////////////////////////////////////////////////////////////////////////////////

impl DimBitXor<UTerm> for UTerm {
    type Output = UTerm;
}

impl<U, B> DimBitXor<UInt<U, B>> for UTerm
where
    UInt<U, B>: Dim,
{
    type Output = UInt<U, B>;
}

impl<U, B> DimBitXor<UTerm> for UInt<U, B>
where
    Self: Dim,
{
    type Output = Self;
}

impl<Ul, Bl, Ur, Br> DimBitXor<UInt<Ur, Br>> for UInt<Ul, Bl>
where
    Self: Dim + PrivateXor<UInt<Ur, Br>>,
    UInt<Ur, Br>: Dim,
    PrivateXorOut<Self, UInt<Ur, Br>>: Trim,
    TrimOut<PrivateXorOut<Self, UInt<Ur, Br>>>: Dim,
{
    type Output = TrimOut<PrivateXorOut<Self, UInt<Ur, Br>>>;
}

////////////////////////////////////////////////////////////////////////////////
// Const forwarding
////////////////////////////////////////////////////////////////////////////////

macro_rules! forward_const_binop {
    ($trait:ident) => {
        impl<const N: usize> $trait<UTerm> for Const<N>
        where
            Const<N>: Dim,
            <Const<N> as Dim>::TypeNum: $trait<UTerm>,
        {
            type Output = <<Const<N> as Dim>::TypeNum as $trait<UTerm>>::Output;
        }

        impl<const N: usize, U, B> $trait<UInt<U, B>> for Const<N>
        where
            Const<N>: Dim,
            <Const<N> as Dim>::TypeNum: $trait<UInt<U, B>>,
        {
            type Output =
                <<Const<N> as Dim>::TypeNum as $trait<UInt<U, B>>>::Output;
        }

        impl<const N: usize, const M: usize> $trait<Const<M>> for Const<N>
        where
            Const<N>: Dim,
            Const<M>: Dim,
            <Const<N> as Dim>::TypeNum: $trait<<Const<M> as Dim>::TypeNum>,
        {
            type Output = <<Const<N> as Dim>::TypeNum as $trait<
                <Const<M> as Dim>::TypeNum,
            >>::Output;
        }

        impl<const M: usize> $trait<Const<M>> for UTerm
        where
            Const<M>: Dim,
            Self: $trait<<Const<M> as Dim>::TypeNum>,
        {
            type Output = <Self as $trait<<Const<M> as Dim>::TypeNum>>::Output;
        }

        impl<U, B, const M: usize> $trait<Const<M>> for UInt<U, B>
        where
            Const<M>: Dim,
            Self: $trait<<Const<M> as Dim>::TypeNum>,
        {
            type Output = <Self as $trait<<Const<M> as Dim>::TypeNum>>::Output;
        }
    };
}

forward_const_binop!(DimAdd);
forward_const_binop!(DimSub);
forward_const_binop!(DimMul);
forward_const_binop!(DimMax);
forward_const_binop!(DimMin);
forward_const_binop!(DimBitAnd);
forward_const_binop!(DimBitOr);
forward_const_binop!(DimBitXor);

mod private {
    //! Private bit-arithmetic operators for canonical unsigned dimension types.
    //!
    //! Carry, borrow, trim, comparison, and max/min selection stay here so the
    //! public `Dim*` traits only mention dimension types.
    #![allow(clippy::use_self)]

    use super::{B0, B1, UInt, UTerm};

    /// Marker for a type-level comparison result.
    pub struct Less;
    /// Marker for a type-level comparison result.
    pub struct Equal;
    /// Marker for a type-level comparison result.
    pub struct Greater;

    /// Add a single bit, propagating carry.
    pub trait AddBit<B> {
        /// Sum including carry into a possibly wider unsigned type.
        type Output;
    }

    /// Subtract a single bit, propagating borrow.
    pub trait SubBit<B> {
        /// Difference after borrow.
        type Output;
    }

    /// Bitwise subtraction before leading-zero trim.
    pub trait PrivateSub<Rhs> {
        /// Untrimmed difference.
        type Output;
    }

    /// Bitwise AND before leading-zero trim.
    pub trait PrivateAnd<Rhs> {
        /// Untrimmed bitwise AND.
        type Output;
    }

    /// Bitwise OR.
    pub trait PrivateOr<Rhs> {
        /// Bitwise OR.
        type Output;
    }

    /// Bitwise XOR before leading-zero trim.
    pub trait PrivateXor<Rhs> {
        /// Untrimmed bitwise XOR.
        type Output;
    }

    /// Remove leading `UInt<UTerm, B0>` so the result is a canonical `Dim`.
    pub trait Trim {
        /// Canonical unsigned type.
        type Output;
    }

    /// Re-attach an LSB after trimming more-significant bits.
    pub trait AttachBit<B> {
        /// Canonical unsigned type with `B` as the new LSB, or `UTerm` if the
        /// whole value is zero.
        type Output;
    }

    /// Lexicographic comparison of more-significant bits, carrying `SoFar`.
    pub trait PrivateCmp<Rhs, SoFar> {
        /// `Less`, `Equal`, or `Greater`.
        type Output;
    }

    /// Select the maximum given a comparison result.
    pub trait PrivateMax<Rhs, Cmp> {
        /// The larger operand.
        type Output;
    }

    /// Select the minimum given a comparison result.
    pub trait PrivateMin<Rhs, Cmp> {
        /// The smaller operand.
        type Output;
    }

    /// Compare two canonical unsigned types.
    pub trait Cmp<Rhs> {
        /// `Less`, `Equal`, or `Greater`.
        type Output;
    }

    pub type PrivateSubOut<A, Rhs> = <A as PrivateSub<Rhs>>::Output;
    pub type PrivateAndOut<A, Rhs> = <A as PrivateAnd<Rhs>>::Output;
    pub type PrivateOrOut<A, Rhs> = <A as PrivateOr<Rhs>>::Output;
    pub type PrivateXorOut<A, Rhs> = <A as PrivateXor<Rhs>>::Output;
    pub type TrimOut<A> = <A as Trim>::Output;
    pub type PrivateCmpOut<A, Rhs, SoFar> =
        <A as PrivateCmp<Rhs, SoFar>>::Output;
    pub type PrivateMaxOut<A, Rhs, CmpRes> =
        <A as PrivateMax<Rhs, CmpRes>>::Output;
    pub type PrivateMinOut<A, Rhs, CmpRes> =
        <A as PrivateMin<Rhs, CmpRes>>::Output;
    pub type Compare<A, B> = <A as Cmp<B>>::Output;

    ////////////////////////////////////////////////////////////////////////////////
    // AddBit
    ////////////////////////////////////////////////////////////////////////////////

    impl AddBit<B0> for UTerm {
        type Output = UTerm;
    }

    impl AddBit<B1> for UTerm {
        type Output = UInt<UTerm, B1>;
    }

    impl<U, B> AddBit<B0> for UInt<U, B> {
        type Output = Self;
    }

    impl<U> AddBit<B1> for UInt<U, B0> {
        type Output = UInt<U, B1>;
    }

    impl<U> AddBit<B1> for UInt<U, B1>
    where
        U: AddBit<B1>,
    {
        type Output = UInt<U::Output, B0>;
    }

    ////////////////////////////////////////////////////////////////////////////////
    // SubBit
    ////////////////////////////////////////////////////////////////////////////////

    impl SubBit<B0> for UTerm {
        type Output = UTerm;
    }

    impl<U, B> SubBit<B0> for UInt<U, B> {
        type Output = Self;
    }

    impl SubBit<B1> for UInt<UTerm, B1> {
        type Output = UTerm;
    }

    impl<U, B> SubBit<B1> for UInt<UInt<U, B>, B1> {
        type Output = UInt<UInt<U, B>, B0>;
    }

    impl<U> SubBit<B1> for UInt<U, B0>
    where
        U: SubBit<B1>,
    {
        type Output = UInt<U::Output, B1>;
    }

    ////////////////////////////////////////////////////////////////////////////////
    // PrivateSub
    ////////////////////////////////////////////////////////////////////////////////

    impl PrivateSub<UTerm> for UTerm {
        type Output = UTerm;
    }

    impl<U, B> PrivateSub<UTerm> for UInt<U, B> {
        type Output = Self;
    }

    impl<Ul, Ur> PrivateSub<UInt<Ur, B0>> for UInt<Ul, B0>
    where
        Ul: PrivateSub<Ur>,
    {
        type Output = UInt<PrivateSubOut<Ul, Ur>, B0>;
    }

    impl<Ul, Ur> PrivateSub<UInt<Ur, B1>> for UInt<Ul, B0>
    where
        Ul: PrivateSub<Ur>,
        PrivateSubOut<Ul, Ur>: SubBit<B1>,
    {
        type Output = UInt<<PrivateSubOut<Ul, Ur> as SubBit<B1>>::Output, B1>;
    }

    impl<Ul, Ur> PrivateSub<UInt<Ur, B0>> for UInt<Ul, B1>
    where
        Ul: PrivateSub<Ur>,
    {
        type Output = UInt<PrivateSubOut<Ul, Ur>, B1>;
    }

    impl<Ul, Ur> PrivateSub<UInt<Ur, B1>> for UInt<Ul, B1>
    where
        Ul: PrivateSub<Ur>,
    {
        type Output = UInt<PrivateSubOut<Ul, Ur>, B0>;
    }

    ////////////////////////////////////////////////////////////////////////////////
    // PrivateAnd
    ////////////////////////////////////////////////////////////////////////////////

    impl<Rhs> PrivateAnd<Rhs> for UTerm {
        type Output = UTerm;
    }

    impl<U, B> PrivateAnd<UTerm> for UInt<U, B> {
        type Output = UTerm;
    }

    impl<Ul, Ur> PrivateAnd<UInt<Ur, B0>> for UInt<Ul, B0>
    where
        Ul: PrivateAnd<Ur>,
    {
        type Output = UInt<PrivateAndOut<Ul, Ur>, B0>;
    }

    impl<Ul, Ur> PrivateAnd<UInt<Ur, B1>> for UInt<Ul, B0>
    where
        Ul: PrivateAnd<Ur>,
    {
        type Output = UInt<PrivateAndOut<Ul, Ur>, B0>;
    }

    impl<Ul, Ur> PrivateAnd<UInt<Ur, B0>> for UInt<Ul, B1>
    where
        Ul: PrivateAnd<Ur>,
    {
        type Output = UInt<PrivateAndOut<Ul, Ur>, B0>;
    }

    impl<Ul, Ur> PrivateAnd<UInt<Ur, B1>> for UInt<Ul, B1>
    where
        Ul: PrivateAnd<Ur>,
    {
        type Output = UInt<PrivateAndOut<Ul, Ur>, B1>;
    }

    ////////////////////////////////////////////////////////////////////////////////
    // PrivateOr
    ////////////////////////////////////////////////////////////////////////////////

    impl<Rhs> PrivateOr<Rhs> for UTerm {
        type Output = Rhs;
    }

    impl<U, B> PrivateOr<UTerm> for UInt<U, B> {
        type Output = Self;
    }

    impl<Ul, Ur> PrivateOr<UInt<Ur, B0>> for UInt<Ul, B0>
    where
        Ul: PrivateOr<Ur>,
    {
        type Output = UInt<PrivateOrOut<Ul, Ur>, B0>;
    }

    impl<Ul, Ur> PrivateOr<UInt<Ur, B1>> for UInt<Ul, B0>
    where
        Ul: PrivateOr<Ur>,
    {
        type Output = UInt<PrivateOrOut<Ul, Ur>, B1>;
    }

    impl<Ul, Ur> PrivateOr<UInt<Ur, B0>> for UInt<Ul, B1>
    where
        Ul: PrivateOr<Ur>,
    {
        type Output = UInt<PrivateOrOut<Ul, Ur>, B1>;
    }

    impl<Ul, Ur> PrivateOr<UInt<Ur, B1>> for UInt<Ul, B1>
    where
        Ul: PrivateOr<Ur>,
    {
        type Output = UInt<PrivateOrOut<Ul, Ur>, B1>;
    }

    ////////////////////////////////////////////////////////////////////////////////
    // PrivateXor
    ////////////////////////////////////////////////////////////////////////////////

    impl<Rhs> PrivateXor<Rhs> for UTerm {
        type Output = Rhs;
    }

    impl<U, B> PrivateXor<UTerm> for UInt<U, B> {
        type Output = Self;
    }

    impl<Ul, Ur> PrivateXor<UInt<Ur, B0>> for UInt<Ul, B0>
    where
        Ul: PrivateXor<Ur>,
    {
        type Output = UInt<PrivateXorOut<Ul, Ur>, B0>;
    }

    impl<Ul, Ur> PrivateXor<UInt<Ur, B1>> for UInt<Ul, B0>
    where
        Ul: PrivateXor<Ur>,
    {
        type Output = UInt<PrivateXorOut<Ul, Ur>, B1>;
    }

    impl<Ul, Ur> PrivateXor<UInt<Ur, B0>> for UInt<Ul, B1>
    where
        Ul: PrivateXor<Ur>,
    {
        type Output = UInt<PrivateXorOut<Ul, Ur>, B1>;
    }

    impl<Ul, Ur> PrivateXor<UInt<Ur, B1>> for UInt<Ul, B1>
    where
        Ul: PrivateXor<Ur>,
    {
        type Output = UInt<PrivateXorOut<Ul, Ur>, B0>;
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Trim
    ////////////////////////////////////////////////////////////////////////////////

    impl Trim for UTerm {
        type Output = Self;
    }

    impl<U, B> Trim for UInt<U, B>
    where
        U: Trim,
        U::Output: AttachBit<B>,
    {
        type Output = <U::Output as AttachBit<B>>::Output;
    }

    impl AttachBit<B0> for UTerm {
        type Output = UTerm;
    }

    impl AttachBit<B1> for UTerm {
        type Output = UInt<UTerm, B1>;
    }

    impl<U, B> AttachBit<B0> for UInt<U, B> {
        type Output = UInt<Self, B0>;
    }

    impl<U, B> AttachBit<B1> for UInt<U, B> {
        type Output = UInt<Self, B1>;
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Cmp
    ////////////////////////////////////////////////////////////////////////////////

    impl<A, B> Cmp<B> for A
    where
        A: PrivateCmp<B, Equal>,
    {
        type Output = PrivateCmpOut<A, B, Equal>;
    }

    impl<Ul, Ur, SoFar> PrivateCmp<UInt<Ur, B0>, SoFar> for UInt<Ul, B0>
    where
        Ul: PrivateCmp<Ur, SoFar>,
    {
        type Output = PrivateCmpOut<Ul, Ur, SoFar>;
    }

    impl<Ul, Ur, SoFar> PrivateCmp<UInt<Ur, B1>, SoFar> for UInt<Ul, B1>
    where
        Ul: PrivateCmp<Ur, SoFar>,
    {
        type Output = PrivateCmpOut<Ul, Ur, SoFar>;
    }

    impl<Ul, Ur, SoFar> PrivateCmp<UInt<Ur, B1>, SoFar> for UInt<Ul, B0>
    where
        Ul: PrivateCmp<Ur, Less>,
    {
        type Output = PrivateCmpOut<Ul, Ur, Less>;
    }

    impl<Ul, Ur, SoFar> PrivateCmp<UInt<Ur, B0>, SoFar> for UInt<Ul, B1>
    where
        Ul: PrivateCmp<Ur, Greater>,
    {
        type Output = PrivateCmpOut<Ul, Ur, Greater>;
    }

    impl<U, B, SoFar> PrivateCmp<UInt<U, B>, SoFar> for UTerm {
        type Output = Less;
    }

    impl<U, B, SoFar> PrivateCmp<UTerm, SoFar> for UInt<U, B> {
        type Output = Greater;
    }

    impl<SoFar> PrivateCmp<UTerm, SoFar> for UTerm {
        type Output = SoFar;
    }

    ////////////////////////////////////////////////////////////////////////////////
    // PrivateMax / PrivateMin
    ////////////////////////////////////////////////////////////////////////////////

    impl<U, B, Ur> PrivateMax<Ur, Equal> for UInt<U, B> {
        type Output = Self;
    }

    impl<U, B, Ur> PrivateMax<Ur, Less> for UInt<U, B> {
        type Output = Ur;
    }

    impl<U, B, Ur> PrivateMax<Ur, Greater> for UInt<U, B> {
        type Output = Self;
    }

    impl<U, B, Ur> PrivateMin<Ur, Equal> for UInt<U, B> {
        type Output = Self;
    }

    impl<U, B, Ur> PrivateMin<Ur, Less> for UInt<U, B> {
        type Output = Self;
    }

    impl<U, B, Ur> PrivateMin<Ur, Greater> for UInt<U, B> {
        type Output = Ur;
    }

    /// Helper trait to map 0 or 1 to B0 or B1.
    pub trait SelectBit<const B: usize> {
        /// Selected bit type.
        type Output: super::Bit;
    }

    impl SelectBit<0> for () {
        type Output = B0;
    }

    impl SelectBit<1> for () {
        type Output = B1;
    }
}

impl Dim for Const<0> {
    const USIZE: usize = 0;
    type TypeNum = UTerm;
}

/// Type alias for dimension.
pub type U0 = UTerm;

macro_rules! impl_dim_single {
    ($val:expr) => {
        impl Dim for Const<{ $val }> {
            const USIZE: usize = $val;
            type TypeNum = UInt<
                <Const<{ ($val) / 2 }> as Dim>::TypeNum,
                <() as SelectBit<{ ($val) % 2 }>>::Output,
            >;
        }
    };
}

macro_rules! impl_dims {
    ($( $val:expr ),* $(,)?) => {
        $( impl_dim_single!($val); )*
    };
}

macro_rules! impl_dim_block_64 {
    ($base:expr) => {
        impl_dims!(
            $base + 0,
            $base + 1,
            $base + 2,
            $base + 3,
            $base + 4,
            $base + 5,
            $base + 6,
            $base + 7,
            $base + 8,
            $base + 9,
            $base + 10,
            $base + 11,
            $base + 12,
            $base + 13,
            $base + 14,
            $base + 15,
            $base + 16,
            $base + 17,
            $base + 18,
            $base + 19,
            $base + 20,
            $base + 21,
            $base + 22,
            $base + 23,
            $base + 24,
            $base + 25,
            $base + 26,
            $base + 27,
            $base + 28,
            $base + 29,
            $base + 30,
            $base + 31,
            $base + 32,
            $base + 33,
            $base + 34,
            $base + 35,
            $base + 36,
            $base + 37,
            $base + 38,
            $base + 39,
            $base + 40,
            $base + 41,
            $base + 42,
            $base + 43,
            $base + 44,
            $base + 45,
            $base + 46,
            $base + 47,
            $base + 48,
            $base + 49,
            $base + 50,
            $base + 51,
            $base + 52,
            $base + 53,
            $base + 54,
            $base + 55,
            $base + 56,
            $base + 57,
            $base + 58,
            $base + 59,
            $base + 60,
            $base + 61,
            $base + 62,
            $base + 63
        );
    };
}

// 16 blocks of 64 contiguous values cover 1..=1024 without recursion.
impl_dim_block_64!(1);
impl_dim_block_64!(65);
impl_dim_block_64!(129);
impl_dim_block_64!(193);
impl_dim_block_64!(257);
impl_dim_block_64!(321);
impl_dim_block_64!(385);
impl_dim_block_64!(449);
impl_dim_block_64!(513);
impl_dim_block_64!(577);
impl_dim_block_64!(641);
impl_dim_block_64!(705);
impl_dim_block_64!(769);
impl_dim_block_64!(833);
impl_dim_block_64!(897);
impl_dim_block_64!(961);

// Extra powers of two through 16384 (C-1).
impl_dims!(2048, 4096, 8192, 16384);

macro_rules! generate_binary_aliases {
    ($count:expr, ) => {};
    ($count:expr, $current:ident, $($rest:ident,)*) => {
        /// Type alias for dimension.
        pub type $current = <Const<{ $count }> as Dim>::TypeNum;
        generate_binary_aliases!($count + 1, $($rest,)*);
    };
    ($( ($val:expr, $alias:ident) ),* $(,)?) => {
        $(
            /// Type alias for dimension.
            pub type $alias = <Const<{ $val }> as Dim>::TypeNum;
        )*
    };
}

generate_binary_aliases!(
    1, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16,
    U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31,
    U32, U33, U34, U35, U36, U37, U38, U39, U40, U41, U42, U43, U44, U45, U46,
    U47, U48, U49, U50, U51, U52, U53, U54, U55, U56, U57, U58, U59, U60, U61,
    U62, U63, U64, U65, U66, U67, U68, U69, U70, U71, U72, U73, U74, U75, U76,
    U77, U78, U79, U80, U81, U82, U83, U84, U85, U86, U87, U88, U89, U90, U91,
    U92, U93, U94, U95, U96, U97, U98, U99, U100, U101, U102, U103, U104, U105,
    U106, U107, U108, U109, U110, U111, U112, U113, U114, U115, U116, U117,
    U118, U119, U120, U121, U122, U123, U124, U125, U126, U127,
);

generate_binary_aliases!(
    (128, U128),
    (256, U256),
    (512, U512),
    (1024, U1024),
    (2048, U2048),
    (4096, U4096),
    (8192, U8192),
    (16384, U16384),
);
