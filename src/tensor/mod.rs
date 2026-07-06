//! # Tensor
//!
//! A module providing compile-time shaped, stack-allocated multi-dimensional `Tensor` implementations.

use crate::math::num_types::{Const, Dim, DimMul};
use core::marker::PhantomData;
use core::ops::{AddAssign, DivAssign, MulAssign, SubAssign};

/// Unit and HIL test suites for tensors.
#[cfg(any(test, feature = "hil"))]
pub mod tests;

// ==========================================
// Types and Structs (PascalCase)
// ==========================================

/// Trait to define multi-dimensional tensor shape and layout at compile-time.
pub trait TensorLayout: Clone + Copy + PartialEq + Eq {
    /// Number of dimensions (rank).
    const RANK: usize;

    /// Product of all dimensions (total elements).
    const SIZE: usize;

    /// Type representing the total size as a Dim.
    type Size: Dim;

    /// Returns the dimension sizes as a slice.
    fn dims() -> &'static [usize];
}

/// A multi-dimensional tensor allocated on the stack.
///
/// Internally stored as a flat array of size `N` with dimension types `Dims`.
///
/// # Compile-time size constraint
/// Statically constrained by the trait bound `Dims: TensorLayout<Size = <Const<N> as Dim>::PeanoTypeNum>` on its `impl` blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tensor<T, const N: usize, Dims: TensorLayout> {
    _marker: PhantomData<Dims>,
    pub(crate) data: [T; N],
}

// ==========================================
// Macro for TensorLayout tuple boilerplate
// ==========================================

macro_rules! impl_tensor_layout {
    (
        ($($D:ident),+) => $rank:expr, $size:expr, $size_dim:ty, ($($bounds:tt)*)
    ) => {
        #[allow(clippy::arithmetic_side_effects)]
        impl<$(const $D: usize),+> TensorLayout for ($(Const<$D>),+)
        where
            $($bounds)*
        {
            const RANK: usize = $rank;
            const SIZE: usize = $size;
            type Size = $size_dim;

            #[inline(always)]
            fn dims() -> &'static [usize] {
                &[$($D),+]
            }
        }
    };
}

// ==========================================
// Implementations (impls)
// ==========================================

impl<const D1: usize> TensorLayout for Const<D1>
where
    Self: Dim,
{
    const RANK: usize = 1;
    const SIZE: usize = D1;
    type Size = Self;

    #[inline(always)]
    fn dims() -> &'static [usize] {
        &[D1]
    }
}

impl_tensor_layout!(
    (D1, D2) => 2, D1 * D2,
    <<Const<D1> as Dim>::PeanoTypeNum as DimMul<<Const<D2> as Dim>::PeanoTypeNum>>::Output,
    (
        Const<D1>: Dim,
        Const<D2>: Dim,
        <Const<D1> as Dim>::PeanoTypeNum: DimMul<<Const<D2> as Dim>::PeanoTypeNum>,
        <<Const<D1> as Dim>::PeanoTypeNum as DimMul<<Const<D2> as Dim>::PeanoTypeNum>>::Output: Dim,
    )
);

impl_tensor_layout!(
    (D1, D2, D3) => 3, D1 * D2 * D3,
    <<<Const<D1> as Dim>::PeanoTypeNum as DimMul<<Const<D2> as Dim>::PeanoTypeNum>>::Output as DimMul<<Const<D3> as Dim>::PeanoTypeNum>>::Output,
    (
        Const<D1>: Dim,
        Const<D2>: Dim,
        Const<D3>: Dim,
        <Const<D1> as Dim>::PeanoTypeNum: DimMul<<Const<D2> as Dim>::PeanoTypeNum>,
        <<Const<D1> as Dim>::PeanoTypeNum as DimMul<<Const<D2> as Dim>::PeanoTypeNum>>::Output: Dim + DimMul<<Const<D3> as Dim>::PeanoTypeNum>,
        <<<Const<D1> as Dim>::PeanoTypeNum as DimMul<<Const<D2> as Dim>::PeanoTypeNum>>::Output as DimMul<<Const<D3> as Dim>::PeanoTypeNum>>::Output: Dim,
    )
);

impl_tensor_layout!(
    (D1, D2, D3, D4) => 4, D1 * D2 * D3 * D4,
    <<<<Const<D1> as Dim>::PeanoTypeNum as DimMul<<Const<D2> as Dim>::PeanoTypeNum>>::Output as DimMul<<Const<D3> as Dim>::PeanoTypeNum>>::Output as DimMul<<Const<D4> as Dim>::PeanoTypeNum>>::Output,
    (
        Const<D1>: Dim,
        Const<D2>: Dim,
        Const<D3>: Dim,
        Const<D4>: Dim,
        <Const<D1> as Dim>::PeanoTypeNum: DimMul<<Const<D2> as Dim>::PeanoTypeNum>,
        <<Const<D1> as Dim>::PeanoTypeNum as DimMul<<Const<D2> as Dim>::PeanoTypeNum>>::Output: Dim + DimMul<<Const<D3> as Dim>::PeanoTypeNum>,
        <<<Const<D1> as Dim>::PeanoTypeNum as DimMul<<Const<D2> as Dim>::PeanoTypeNum>>::Output as DimMul<<Const<D3> as Dim>::PeanoTypeNum>>::Output: Dim + DimMul<<Const<D4> as Dim>::PeanoTypeNum>,
        <<<<Const<D1> as Dim>::PeanoTypeNum as DimMul<<Const<D2> as Dim>::PeanoTypeNum>>::Output as DimMul<<Const<D3> as Dim>::PeanoTypeNum>>::Output as DimMul<<Const<D4> as Dim>::PeanoTypeNum>>::Output: Dim,
    )
);

impl<T, const N: usize, Dims> Tensor<T, N, Dims>
where
    Const<N>: Dim,
    Dims: TensorLayout<Size = <Const<N> as Dim>::PeanoTypeNum>,
{
    /// Returns the underlying mutable flat data.
    #[inline(always)]
    pub const fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Returns the underlying flat data.
    #[inline(always)]
    pub const fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Safe read-only element access.
    #[inline]
    pub fn get(&self, coords: &[usize]) -> Option<&T> {
        let idx = self.linear_index(coords)?;
        self.data.get(idx)
    }

    /// Safe mutable element access.
    #[inline]
    pub fn get_mut(&mut self, coords: &[usize]) -> Option<&mut T> {
        let idx = self.linear_index(coords)?;
        self.data.get_mut(idx)
    }

    /// Calculate the flat index from coordinate indices.
    /// Uses column-major ordering (first dimension varies fastest).
    ///
    /// # Clippy Allow explanation
    /// We allow `clippy::arithmetic_side_effects` here because the index computation logic
    /// accumulates strides and offsets which are explicitly checked against shape bounds (`coord >= dim`).
    #[allow(clippy::arithmetic_side_effects)]
    #[inline]
    pub fn linear_index(&self, coords: &[usize]) -> Option<usize> {
        let dims = Dims::dims();
        if coords.len() != dims.len() {
            return None;
        }

        let mut index = 0;
        let mut stride = 1;
        for (&coord, &dim) in coords.iter().zip(dims.iter()) {
            if coord >= dim {
                return None;
            }
            index += coord * stride;
            stride *= dim;
        }
        Some(index)
    }

    /// Create a new tensor from a flat array.
    ///
    /// The size alignment `N == Dims::SIZE` is enforced at compile time
    /// by the trait bound `Dims: TensorLayout<Size = <Const<N> as Dim>::PeanoTypeNum>`.
    #[inline]
    pub const fn new(data: [T; N]) -> Self {
        Self {
            data,
            _marker: PhantomData,
        }
    }

    /// Returns the number of dimensions (rank).
    #[inline(always)]
    pub const fn rank(&self) -> usize {
        Dims::RANK
    }

    /// Returns the dimension sizes as a slice.
    #[inline(always)]
    pub fn shape(&self) -> &'static [usize] {
        Dims::dims()
    }
}

impl<T, const N: usize, Dims> AddAssign<&Self> for Tensor<T, N, Dims>
where
    T: crate::math::num_traits::Ring,
    Const<N>: Dim,
    Dims: TensorLayout<Size = <Const<N> as Dim>::PeanoTypeNum>,
{
    /// Performs element-wise tensor addition in-place using standard BLAS AXPY subprograms.
    #[inline]
    fn add_assign(&mut self, rhs: &Self) {
        use crate::math::subprograms::BasicSubPrograms;
        use crate::math::subprograms::level1::AXPY;
        BasicSubPrograms::axpy(T::ONE, &rhs.data, &mut self.data);
    }
}

impl<T, const N: usize, Dims> AddAssign<Self> for Tensor<T, N, Dims>
where
    T: crate::math::num_traits::Ring,
    Const<N>: Dim,
    Dims: TensorLayout<Size = <Const<N> as Dim>::PeanoTypeNum>,
{
    /// Performs element-wise tensor addition in-place using standard BLAS AXPY subprograms.
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.add_assign(&rhs);
    }
}

impl<T, const N: usize, Dims> SubAssign<&Self> for Tensor<T, N, Dims>
where
    T: crate::math::num_traits::Ring + crate::math::ops::Neg<Output = T>,
    Const<N>: Dim,
    Dims: TensorLayout<Size = <Const<N> as Dim>::PeanoTypeNum>,
{
    /// Performs element-wise tensor subtraction in-place using standard BLAS AXPY subprograms.
    ///
    /// # Clippy Allow explanation
    /// We allow `clippy::arithmetic_side_effects` here because negating `T::ONE` to pass to `axpy`
    /// is a standard algebraic representation of negation/subtraction.
    #[allow(clippy::arithmetic_side_effects)]
    #[inline]
    fn sub_assign(&mut self, rhs: &Self) {
        use crate::math::subprograms::BasicSubPrograms;
        use crate::math::subprograms::level1::AXPY;
        BasicSubPrograms::axpy(-T::ONE, &rhs.data, &mut self.data);
    }
}

impl<T, const N: usize, Dims> SubAssign<Self> for Tensor<T, N, Dims>
where
    T: crate::math::num_traits::Ring + crate::math::ops::Neg<Output = T>,
    Const<N>: Dim,
    Dims: TensorLayout<Size = <Const<N> as Dim>::PeanoTypeNum>,
{
    /// Performs element-wise tensor subtraction in-place using standard BLAS AXPY subprograms.
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.sub_assign(&rhs);
    }
}

impl<T, const N: usize, Dims> MulAssign<T> for Tensor<T, N, Dims>
where
    T: crate::math::num_traits::Ring,
    Const<N>: Dim,
    Dims: TensorLayout<Size = <Const<N> as Dim>::PeanoTypeNum>,
{
    /// Scales all elements of the tensor by a factor using standard BLAS SCAL subprograms.
    #[inline]
    fn mul_assign(&mut self, rhs: T) {
        use crate::math::subprograms::BasicSubPrograms;
        use crate::math::subprograms::level1::SCAL;
        BasicSubPrograms::scal(rhs, &mut self.data);
    }
}

impl<T, const N: usize, Dims> DivAssign<T> for Tensor<T, N, Dims>
where
    T: crate::math::num_traits::Field,
    Const<N>: Dim,
    Dims: TensorLayout<Size = <Const<N> as Dim>::PeanoTypeNum>,
{
    /// Scales all elements of the tensor by a divisor using standard BLAS SCAL subprograms with reciprocal.
    #[allow(clippy::arithmetic_side_effects)]
    #[inline]
    fn div_assign(&mut self, rhs: T) {
        use crate::math::subprograms::BasicSubPrograms;
        use crate::math::subprograms::level1::SCAL;
        BasicSubPrograms::scal(T::one() / rhs, &mut self.data);
    }
}

impl<T, const N: usize, Dims> core::ops::Add<Self> for Tensor<T, N, Dims>
where
    T: crate::math::num_traits::Ring,
    Const<N>: Dim,
    Dims: TensorLayout<Size = <Const<N> as Dim>::PeanoTypeNum>,
{
    type Output = Self;

    #[inline]
    fn add(mut self, rhs: Self) -> Self::Output {
        self.add_assign(&rhs);
        self
    }
}

impl<T, const N: usize, Dims> core::ops::Add<&Self> for Tensor<T, N, Dims>
where
    T: crate::math::num_traits::Ring,
    Const<N>: Dim,
    Dims: TensorLayout<Size = <Const<N> as Dim>::PeanoTypeNum>,
{
    type Output = Self;

    #[inline]
    fn add(mut self, rhs: &Self) -> Self::Output {
        self.add_assign(rhs);
        self
    }
}

impl<T, const N: usize, Dims> core::ops::Sub<Self> for Tensor<T, N, Dims>
where
    T: crate::math::num_traits::Ring + crate::math::ops::Neg<Output = T>,
    Const<N>: Dim,
    Dims: TensorLayout<Size = <Const<N> as Dim>::PeanoTypeNum>,
{
    type Output = Self;

    #[inline]
    fn sub(mut self, rhs: Self) -> Self::Output {
        self.sub_assign(&rhs);
        self
    }
}

impl<T, const N: usize, Dims> core::ops::Sub<&Self> for Tensor<T, N, Dims>
where
    T: crate::math::num_traits::Ring + crate::math::ops::Neg<Output = T>,
    Const<N>: Dim,
    Dims: TensorLayout<Size = <Const<N> as Dim>::PeanoTypeNum>,
{
    type Output = Self;

    #[inline]
    fn sub(mut self, rhs: &Self) -> Self::Output {
        self.sub_assign(rhs);
        self
    }
}

impl<T, const N: usize, Dims> core::ops::Mul<T> for Tensor<T, N, Dims>
where
    T: crate::math::num_traits::Ring,
    Const<N>: Dim,
    Dims: TensorLayout<Size = <Const<N> as Dim>::PeanoTypeNum>,
{
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects)]
    #[inline]
    fn mul(mut self, rhs: T) -> Self::Output {
        self *= rhs;
        self
    }
}

impl<T, const N: usize, Dims> core::ops::Div<T> for Tensor<T, N, Dims>
where
    T: crate::math::num_traits::Field,
    Const<N>: Dim,
    Dims: TensorLayout<Size = <Const<N> as Dim>::PeanoTypeNum>,
{
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects)]
    #[inline]
    fn div(mut self, rhs: T) -> Self::Output {
        self /= rhs;
        self
    }
}
