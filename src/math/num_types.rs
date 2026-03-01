//! # Numeric Types (Type-Level Math Alternative)
//!
//! Defines dimension types and bounds for matrix and tensor storage using
//! purely compile-time type-level arithmetic (Peano axioms).
//!
//! This implementation avoids heavy code generation by forcing the Rust
//! compiler's trait solver to calculate matrix dimensions during compilation.
#![allow(clippy::arbitrary_source_item_ordering)]

use core::marker::PhantomData;

/// Defines the base behavior for dimensions.
pub trait Dim: Clone + Copy + PartialEq + Eq {
    /// Returns the runtime value of the dimension.
    ///
    /// # Returns
    /// * `usize` - The runtime value of the dimension.
    fn value(&self) -> usize;
}

/// Defines a sub-trait exclusively for dimensions known at compile-time.
pub trait DimName: Dim + Default {
    /// Represents the compile-time value of the dimension.
    const VALUE: usize;
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
pub struct S<N: DimName>(PhantomData<N>);

// ==========================================
// Friendly Type Aliases Macro
// ==========================================

/// Type alias for dimension size 0.
pub type U0 = Z;

/// Generates friendly `U1`, `U2`... etc. aliases to hide the `S<S<Z>>` complexity.
macro_rules! generate_peano_aliases {
    ($prev:ident, ) => {};
    ($prev:ident, $current:ident, $($rest:ident,)*) => {
        /// Generated Type alias for a specific dimension size.
        pub type $current = S<$prev>;
        generate_peano_aliases!($current, $($rest,)*);
    };
}

generate_peano_aliases!(
    U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16,
    U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31,
    U32,
);

impl Dim for Z {
    #[inline(always)]
    fn value(&self) -> usize {
        0
    }
}

impl DimName for Z {
    const VALUE: usize = 0;
}

impl<N: DimName> Dim for S<N> {
    #[inline(always)]
    fn value(&self) -> usize {
        Self::VALUE
    }
}

impl<N: DimName> DimName for S<N> {
    const VALUE: usize = N::VALUE + 1;
}

// --- Addition (N + M) ---

// 0 + M = M
impl<M: Dim> DimAdd<M> for Z {
    type Output = M;
}

// (N + 1) + M = (N + M) + 1
impl<N, M> DimAdd<M> for S<N>
where
    N: DimName + DimAdd<M>,
    M: Dim,
    <N as DimAdd<M>>::Output: DimName,
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
    N: DimName + DimSub<M>,
    M: DimName,
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
    N: DimName + DimMul<M>,
    M: DimName + DimAdd<<N as DimMul<M>>::Output>,
{
    type Output = <M as DimAdd<<N as DimMul<M>>::Output>>::Output;
}

// --- Maximum ---

// Max(0, M) = M
impl<M: Dim> DimMax<M> for Z {
    type Output = M;
}

// Max(N + 1, 0) = N + 1
impl<N: DimName> DimMax<Z> for S<N> {
    type Output = Self;
}

// Max(N + 1, M + 1) = Max(N, M) + 1
impl<N, M> DimMax<S<M>> for S<N>
where
    N: DimName + DimMax<M>,
    M: DimName,
    <N as DimMax<M>>::Output: DimName,
{
    type Output = S<<N as DimMax<M>>::Output>;
}

// --- Minimum ---

// Min(0, M) = 0
impl<M: Dim> DimMin<M> for Z {
    type Output = Self;
}

// Min(N + 1, 0) = 0
impl<N: DimName> DimMin<Z> for S<N> {
    type Output = Z;
}

// Min(N + 1, M + 1) = Min(N, M) + 1
impl<N, M> DimMin<S<M>> for S<N>
where
    N: DimName + DimMin<M>,
    M: DimName,
    <N as DimMin<M>>::Output: DimName,
{
    type Output = S<<N as DimMin<M>>::Output>;
}
