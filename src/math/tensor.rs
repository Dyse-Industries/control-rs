#![allow(
    clippy::arbitrary_source_item_ordering,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::use_self
)]

use crate::math::num_types::Const;
use core::marker::PhantomData;
use core::ops::{AddAssign, DivAssign, MulAssign, SubAssign};

/// Trait to define multi-dimensional tensor shape and layout at compile-time.
pub trait TensorLayout: Clone + Copy + PartialEq + Eq {
    /// Number of dimensions (rank).
    const RANK: usize;

    /// Product of all dimensions (total elements).
    const SIZE: usize;

    /// Returns the dimension sizes as a slice.
    fn dims() -> &'static [usize];
}

// --- Implementations of TensorLayout for Const and Tuples of Const ---

impl<const D1: usize> TensorLayout for Const<D1> {
    const RANK: usize = 1;
    const SIZE: usize = D1;

    #[inline(always)]
    fn dims() -> &'static [usize] {
        &[D1]
    }
}

impl<const D1: usize, const D2: usize> TensorLayout for (Const<D1>, Const<D2>) {
    const RANK: usize = 2;
    const SIZE: usize = D1 * D2;

    #[inline(always)]
    fn dims() -> &'static [usize] {
        &[D1, D2]
    }
}

impl<const D1: usize, const D2: usize, const D3: usize> TensorLayout
    for (Const<D1>, Const<D2>, Const<D3>)
{
    const RANK: usize = 3;
    const SIZE: usize = D1 * D2 * D3;

    #[inline(always)]
    fn dims() -> &'static [usize] {
        &[D1, D2, D3]
    }
}

impl<const D1: usize, const D2: usize, const D3: usize, const D4: usize>
    TensorLayout for (Const<D1>, Const<D2>, Const<D3>, Const<D4>)
{
    const RANK: usize = 4;
    const SIZE: usize = D1 * D2 * D3 * D4;

    #[inline(always)]
    fn dims() -> &'static [usize] {
        &[D1, D2, D3, D4]
    }
}

/// A multi-dimensional tensor allocated on the stack.
///
/// Internally stored as a flat array of size `N` with dimension types `Dims`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tensor<T, const N: usize, Dims: TensorLayout> {
    pub(crate) data: [T; N],
    _marker: PhantomData<Dims>,
}

#[allow(clippy::arithmetic_side_effects)]
impl<T, const N: usize, Dims: TensorLayout> Tensor<T, N, Dims> {
    /// Create a new tensor from a flat array.
    ///
    /// # Panics
    /// Panics at compile-time/construction time if `N != Dims::SIZE`.
    #[inline]
    pub const fn new(data: [T; N]) -> Self {
        assert!(N == Dims::SIZE, "Tensor size N must match Dims::SIZE");

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

    /// Returns the underlying flat data.
    #[inline(always)]
    pub const fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Returns the underlying mutable flat data.
    #[inline(always)]
    pub const fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Calculate the flat index from coordinate indices.
    /// Uses column-major ordering (first dimension varies fastest).
    #[inline]
    pub fn linear_index(&self, coords: &[usize]) -> Option<usize> {
        let dims = Dims::dims();
        if coords.len() != dims.len() {
            return None;
        }

        let mut index = 0;
        let mut stride = 1;
        for (i, &coord) in coords.iter().enumerate() {
            if coord >= dims[i] {
                return None;
            }
            index += coord * stride;
            stride *= dims[i];
        }
        Some(index)
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
}

// --- Arithmetic Operations ---

#[allow(clippy::arithmetic_side_effects)]
impl<T, const N: usize, Dims: TensorLayout> AddAssign<&Self>
    for Tensor<T, N, Dims>
where
    T: AddAssign<T> + Copy,
{
    #[inline]
    fn add_assign(&mut self, rhs: &Self) {
        for (d, s) in self.data.iter_mut().zip(rhs.data.iter()) {
            *d += *s;
        }
    }
}

#[allow(clippy::arithmetic_side_effects)]
impl<T, const N: usize, Dims: TensorLayout> AddAssign<Self>
    for Tensor<T, N, Dims>
where
    T: AddAssign<T> + Copy,
{
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.add_assign(&rhs);
    }
}

#[allow(clippy::arithmetic_side_effects)]
impl<T, const N: usize, Dims: TensorLayout> SubAssign<&Self>
    for Tensor<T, N, Dims>
where
    T: SubAssign<T> + Copy,
{
    #[inline]
    fn sub_assign(&mut self, rhs: &Self) {
        for (d, s) in self.data.iter_mut().zip(rhs.data.iter()) {
            *d -= *s;
        }
    }
}

#[allow(clippy::arithmetic_side_effects)]
impl<T, const N: usize, Dims: TensorLayout> SubAssign<Self>
    for Tensor<T, N, Dims>
where
    T: SubAssign<T> + Copy,
{
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.sub_assign(&rhs);
    }
}

// --- Scaling Operations ---

#[allow(clippy::arithmetic_side_effects)]
impl<T, const N: usize, Dims: TensorLayout> MulAssign<T> for Tensor<T, N, Dims>
where
    T: MulAssign<T> + Copy,
{
    #[inline]
    fn mul_assign(&mut self, rhs: T) {
        for val in &mut self.data {
            *val *= rhs;
        }
    }
}

#[allow(clippy::arithmetic_side_effects)]
impl<T, const N: usize, Dims: TensorLayout> DivAssign<T> for Tensor<T, N, Dims>
where
    T: DivAssign<T> + Copy,
{
    #[inline]
    fn div_assign(&mut self, rhs: T) {
        for val in &mut self.data {
            *val /= rhs;
        }
    }
}
