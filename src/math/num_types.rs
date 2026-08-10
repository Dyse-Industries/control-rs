//! # Numeric Types (Type-Level Math)
//!
//! Defines dimension types and bounds for matrix and tensor storage using
//! purely compile-time type-level arithmetic (Peano axioms).
//!
//! This implementation avoids heavy code generation by forcing the Rust
//! compiler's trait solver to calculate matrix dimensions during compilation.
//!
//! # Ceiling
//!
//! Friendly aliases stop at `U32`, because the unary Peano representation
//! requires trait-solver recursion proportional to dimension size. Beyond
//! it, arithmetic whose Peano depth exceeds the compiler's default
//! trait-solver recursion limit (128) fails to compile with an "overflow
//! evaluating the requirement" error rather than miscompiling silently,
//! e.g. `U32 * U32` (depth 1024):
//!
//! ```compile_fail
//! use control_rs::math::num_types::{DimMul, U32};
//!
//! let _: <U32 as DimMul<U32>>::Output = Default::default();
//! ```
#![allow(clippy::arbitrary_source_item_ordering)]

use core::marker::PhantomData;

/// Defines the base behavior for all dimensions.
pub trait Dim: Clone + Copy + PartialEq + Eq {
    /// Represents the compile-time value of the dimension.
    const DIM: usize;
    /// Type num representation of the dimension.
    type PeanoTypeNum;
}

// ==========================================
// Type-Level Arithmetic Traits
// ==========================================

/// Trait for type-level addition.
pub trait DimAdd<Other> {
    /// The result of the addition.
    type Output: Dim;
}

/// Trait for type-level subtraction.
pub trait DimSub<Other> {
    /// The result of the subtraction.
    type Output: Dim;
}

/// Trait for type-level multiplication.
pub trait DimMul<Other> {
    /// The result of the multiplication.
    type Output: Dim;
}

/// Trait for type-level maximum.
pub trait DimMax<Other> {
    /// The result of the maximum operation.
    type Output: Dim;
}

/// Trait for a type-level minimum.
pub trait DimMin<Other> {
    /// The result of the minimum operation.
    type Output: Dim;
}

// ==========================================
// Base Types: Zero and Successor
// ==========================================

/// Represents Zero (0) in type-level math.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct Z;

/// Represents the Successor of N (i.e., N + 1) in type-level math.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct S<N: Dim>(PhantomData<N>);

/// Hoists literals to the type level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct Const<const N: usize>;

// ==========================================
// Friendly Type Aliases Macro
// ==========================================

/// Type alias for dimension size 0.
pub type U0 = Z;

impl DimAdd<Z> for Const<0> {
    type Output = Z;
}

impl<const N: usize, M: Dim> DimAdd<S<M>> for Const<N>
where
    Self: DimAdd<M>,
{
    type Output = S<<Self as DimAdd<M>>::Output>;
}

impl<const N: usize, const M: usize> DimAdd<Const<M>> for Const<N>
where
    Self: Dim + DimAdd<<Const<M> as Dim>::PeanoTypeNum>,
    Const<M>: Dim,
{
    type Output = <Self as DimAdd<<Const<M> as Dim>::PeanoTypeNum>>::Output;
}

/// Generates friendly `U1`, `U2`, etc. aliases to hide the `S<S<Z>>` complexity.
macro_rules! generate_peano_aliases {
    ($count:expr, $prev:ident, ) => {};
    ($count:expr, $prev:ident, $current:ident, $($rest:ident,)*) => {
        /// Generated Type alias for
        #[doc = stringify!($count)]
        ///.
        pub type $current = S<$prev>;
        impl Dim for Const<{$count}> {
            const DIM: usize = $count;
            type PeanoTypeNum = $current;
        }
        impl DimAdd<Z> for Const<{$count}> {
            type Output = <Const<{$count}> as Dim>::PeanoTypeNum;
        }
        generate_peano_aliases!($count + 1, $current, $($rest,)*);
    };
}

generate_peano_aliases!(
    1, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15,
    U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30,
    U31, U32,
);

impl Dim for Z {
    const DIM: usize = 0;
    type PeanoTypeNum = Self;
}

impl<N: Dim> Dim for S<N> {
    const DIM: usize = N::DIM + 1;
    type PeanoTypeNum = Self;
}

// --- Addition (N + M) ---

// 0 + M = M
impl<M: Dim> DimAdd<M> for Z {
    type Output = M;
}

// (N + 1) + M = (N + M) + 1
impl<N, M> DimAdd<M> for S<N>
where
    N: Dim + DimAdd<M>,
    M: Dim,
    <N as DimAdd<M>>::Output: Dim,
{
    type Output = S<<N as DimAdd<M>>::Output>;
}

// --- Subtraction (N - M) ---

// N - 0 = N
impl<N: Dim> DimSub<Z> for N {
    type Output = N;
}

// (N + 1) - (M + 1) = N - M
impl<N, M> DimSub<S<M>> for S<N>
where
    N: Dim + DimSub<M>,
    M: Dim,
{
    type Output = <N as DimSub<M>>::Output;
}

// --- Multiplication (N * M) ---

// 0 * M = 0
impl<M: Dim> DimMul<M> for Z {
    type Output = Self;
}

// (N + 1) * M = M + (N * M)
impl<N, M> DimMul<M> for S<N>
where
    N: Dim + DimMul<M>,
    M: Dim + DimAdd<<N as DimMul<M>>::Output>,
{
    type Output = <M as DimAdd<<N as DimMul<M>>::Output>>::Output;
}

// --- Maximum ---

// Max(0, M) = M
impl<M: Dim> DimMax<M> for Z {
    type Output = M;
}

// Max(N + 1, 0) = N + 1
impl<N: Dim> DimMax<Z> for S<N> {
    type Output = Self;
}

// Max(N + 1, M + 1) = Max(N, M) + 1
impl<N, M> DimMax<S<M>> for S<N>
where
    N: Dim + DimMax<M>,
    M: Dim,
    <N as DimMax<M>>::Output: Dim,
{
    type Output = S<<N as DimMax<M>>::Output>;
}

// --- Minimum ---

// Min(0, M) = 0
impl<M: Dim> DimMin<M> for Z {
    type Output = Self;
}

// Min(N + 1, 0) = 0
impl<N: Dim> DimMin<Z> for S<N> {
    type Output = Z;
}

// Min(N + 1, M + 1) = Min(N, M) + 1
impl<N, M> DimMin<S<M>> for S<N>
where
    N: Dim + DimMin<M>,
    M: Dim,
    <N as DimMin<M>>::Output: Dim,
{
    type Output = S<<N as DimMin<M>>::Output>;
}
