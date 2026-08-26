//! # Tensor Module
//!
//! Multi-dimensional tensor representation [`Tensor<T, Layout, B>`] and low-cost embedded inference.
//!
//! Includes:
//! - N-dimensional static shape descriptors: [`Shape1D`], [`Shape2D`], [`Shape3D`], [`Shape4D`].
//! - Column-major strided coordinate indexing and slicing.
//! - Multilinear grid interpolation for flight-control lookup tables.
//! - Fixed-point scalar type [`Quantized<Repr, SHIFT>`] implementing `Scalar`.
//! - Nonlinear activations: [`Relu`] and [`TableActivation`].
//!
//! # Examples
//!
//! ```rust
//! use control_rs::math::num_types::{Const, Dim};
//! use control_rs::tensor::{ArrayTensor, Tensor, Shape2D, Quantized};
//!
//! // 2D gain table lookup
//! let table = ArrayTensor::<f32, 2, 2>::from_raw([[1.0, 2.0], [3.0, 4.0]]);
//! let val = table.interpolate(&[0.5, 0.5]);
//! assert_eq!(val, 2.5);
//! ```
#![allow(
    clippy::arbitrary_source_item_ordering,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::similar_names,
    clippy::needless_range_loop,
    clippy::type_complexity,
    clippy::doc_markdown,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::option_if_let_else,
    clippy::must_use_candidate,
    clippy::many_single_char_names,
    clippy::collapsible_if,
    clippy::use_self,
    clippy::too_many_arguments,
    clippy::missing_const_for_fn,
    clippy::cast_lossless,
    clippy::missing_safety_doc
)]

#[cfg(any(test, feature = "ets"))]
/// Tensor module unit tests.
pub mod tests;

pub use crate::math::fixed_num::Quantized;
use crate::math::num_traits::{Float, Zero};
use crate::math::num_types::{Const, Dim};
use crate::math::storage::{ArrayStorage, RowArrayStorage};
use core::marker::PhantomData;

////////////////////////////////////////////////////////////////////////////////
// Flat Buffer Traits
////////////////////////////////////////////////////////////////////////////////

/// Rank-neutral flat-buffer contract for contiguous memory storage.
pub unsafe trait FlatBuffer<T> {
    /// Number of elements stored in the buffer.
    fn len(&self) -> usize;

    /// Checks if the buffer is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Exposes an immutable slice of elements.
    fn as_slice(&self) -> &[T];

    /// Returns a raw immutable pointer to elements.
    fn as_ptr(&self) -> *const T {
        self.as_slice().as_ptr()
    }
}

/// Mutable flat-buffer contract.
pub unsafe trait FlatBufferMut<T>: FlatBuffer<T> {
    /// Exposes a mutable slice of elements.
    fn as_mut_slice(&mut self) -> &mut [T];

    /// Returns a raw mutable pointer to elements.
    fn as_mut_ptr(&mut self) -> *mut T {
        self.as_mut_slice().as_mut_ptr()
    }
}

unsafe impl<T, const R: usize, const C: usize> FlatBuffer<T>
    for ArrayStorage<T, R, C>
where
    Const<R>: Dim,
    Const<C>: Dim,
{
    fn len(&self) -> usize {
        R * C
    }

    fn as_slice(&self) -> &[T] {
        ArrayStorage::as_slice(self)
    }

    fn as_ptr(&self) -> *const T {
        ArrayStorage::as_slice(self).as_ptr()
    }
}

unsafe impl<T, const R: usize, const C: usize> FlatBufferMut<T>
    for ArrayStorage<T, R, C>
where
    Const<R>: Dim,
    Const<C>: Dim,
{
    fn as_mut_slice(&mut self) -> &mut [T] {
        ArrayStorage::as_mut_slice(self)
    }

    fn as_mut_ptr(&mut self) -> *mut T {
        ArrayStorage::as_mut_slice(self).as_mut_ptr()
    }
}

unsafe impl<T, const R: usize, const C: usize> FlatBuffer<T>
    for RowArrayStorage<T, R, C>
where
    Const<R>: Dim,
    Const<C>: Dim,
{
    fn len(&self) -> usize {
        R * C
    }

    fn as_slice(&self) -> &[T] {
        RowArrayStorage::as_slice(self)
    }

    fn as_ptr(&self) -> *const T {
        RowArrayStorage::as_slice(self).as_ptr()
    }
}

unsafe impl<T, const R: usize, const C: usize> FlatBufferMut<T>
    for RowArrayStorage<T, R, C>
where
    Const<R>: Dim,
    Const<C>: Dim,
{
    fn as_mut_slice(&mut self) -> &mut [T] {
        RowArrayStorage::as_mut_slice(self)
    }

    fn as_mut_ptr(&mut self) -> *mut T {
        RowArrayStorage::as_mut_slice(self).as_mut_ptr()
    }
}

unsafe impl<T, const N: usize> FlatBuffer<T> for [T; N] {
    fn len(&self) -> usize {
        N
    }

    fn as_slice(&self) -> &[T] {
        self.as_ref()
    }
}

unsafe impl<T, const N: usize> FlatBufferMut<T> for [T; N] {
    fn as_mut_slice(&mut self) -> &mut [T] {
        self.as_mut()
    }
}

////////////////////////////////////////////////////////////////////////////////
// Tensor Layout & Shapes
////////////////////////////////////////////////////////////////////////////////

/// Statically sized multi-dimensional tensor layout contract.
pub trait TensorLayout: 'static {
    /// Rank of the tensor (number of dimensions).
    const RANK: usize;

    /// Total number of elements.
    const SIZE: usize;

    /// Evaluates flat 1D column-major offset for a multi-dimensional index coordinate.
    fn flat_offset(indices: &[usize]) -> Option<usize>;

    /// Dimension extents along each axis.
    fn dims(dim_out: &mut [usize]);
}

/// 1D Tensor Shape: `[D0]`.
#[derive(Debug, Clone, Copy)]
pub struct Shape1D<const D0: usize>;

impl<const D0: usize> TensorLayout for Shape1D<D0> {
    const RANK: usize = 1;
    const SIZE: usize = D0;

    fn flat_offset(indices: &[usize]) -> Option<usize> {
        if indices.len() == 1 && indices[0] < D0 {
            Some(indices[0])
        } else {
            None
        }
    }

    fn dims(dim_out: &mut [usize]) {
        if !dim_out.is_empty() {
            dim_out[0] = D0;
        }
    }
}

/// 2D Tensor Shape: `[D0, D1]`.
#[derive(Debug, Clone, Copy)]
pub struct Shape2D<const D0: usize, const D1: usize>;

impl<const D0: usize, const D1: usize> TensorLayout for Shape2D<D0, D1> {
    const RANK: usize = 2;
    const SIZE: usize = D0 * D1;

    fn flat_offset(indices: &[usize]) -> Option<usize> {
        if indices.len() == 2 && indices[0] < D0 && indices[1] < D1 {
            Some(indices[0] + indices[1] * D0)
        } else {
            None
        }
    }

    fn dims(dim_out: &mut [usize]) {
        if dim_out.len() >= 2 {
            dim_out[0] = D0;
            dim_out[1] = D1;
        }
    }
}

/// 3D Tensor Shape: `[D0, D1, D2]`.
#[derive(Debug, Clone, Copy)]
pub struct Shape3D<const D0: usize, const D1: usize, const D2: usize>;

impl<const D0: usize, const D1: usize, const D2: usize> TensorLayout
    for Shape3D<D0, D1, D2>
{
    const RANK: usize = 3;
    const SIZE: usize = D0 * D1 * D2;

    fn flat_offset(indices: &[usize]) -> Option<usize> {
        if indices.len() == 3
            && indices[0] < D0
            && indices[1] < D1
            && indices[2] < D2
        {
            Some(indices[0] + indices[1] * D0 + indices[2] * (D0 * D1))
        } else {
            None
        }
    }

    fn dims(dim_out: &mut [usize]) {
        if dim_out.len() >= 3 {
            dim_out[0] = D0;
            dim_out[1] = D1;
            dim_out[2] = D2;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Tensor Structure
////////////////////////////////////////////////////////////////////////////////

/// Generic compile-time sized tensor container over memory buffer `B`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tensor<T, L: TensorLayout, B: FlatBuffer<T>> {
    buffer: B,
    _marker: PhantomData<(T, L)>,
}

/// Owning 2D stack tensor.
pub type ArrayTensor<T, const R: usize, const C: usize> =
    Tensor<T, Shape2D<R, C>, ArrayStorage<T, R, C>>;

/// Owning 1D stack tensor.
pub type ArrayTensor1D<T, const N: usize> = Tensor<T, Shape1D<N>, [T; N]>;

/// Owning 3D stack tensor.
pub type ArrayTensor3D<
    T,
    const D0: usize,
    const D1: usize,
    const D2: usize,
    const TOTAL: usize,
> = Tensor<T, Shape3D<D0, D1, D2>, [T; TOTAL]>;

impl<T, const R: usize, const C: usize> ArrayTensor<T, R, C>
where
    Const<R>: Dim,
    Const<C>: Dim,
{
    /// Builds an all-zero 2D tensor.
    #[must_use]
    pub const fn zero() -> Self
    where
        T: Zero + Copy,
    {
        Self {
            buffer: ArrayStorage::from_array([[T::ZERO; R]; C]),
            _marker: PhantomData,
        }
    }

    /// Builds a 2D tensor directly from column-major nested arrays `[[T; R]; C]`.
    #[must_use]
    pub const fn from_raw(data: [[T; R]; C]) -> Self
    where
        T: Copy,
    {
        Self {
            buffer: ArrayStorage::from_array(data),
            _marker: PhantomData,
        }
    }
}

impl<T, L: TensorLayout, B: FlatBuffer<T>> Tensor<T, L, B> {
    /// Wraps an existing flat buffer backend.
    pub const fn from_storage(buffer: B) -> Self {
        Self {
            buffer,
            _marker: PhantomData,
        }
    }

    /// Borrows the underlying buffer.
    pub const fn buffer(&self) -> &B {
        &self.buffer
    }

    /// Unwraps the underlying buffer.
    pub fn into_buffer(self) -> B {
        self.buffer
    }

    /// Returns a reference to the element at multi-dimensional coordinate `indices`.
    #[must_use]
    pub fn get(&self, indices: &[usize]) -> Option<&T> {
        let offset = L::flat_offset(indices)?;
        self.buffer.as_slice().get(offset)
    }

    /// Exposes a contiguous slice view of tensor memory.
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        self.buffer.as_slice()
    }
}

impl<T, L: TensorLayout, B: FlatBufferMut<T>> Tensor<T, L, B> {
    /// Returns a mutable reference to the element at multi-dimensional coordinate `indices`.
    pub fn get_mut(&mut self, indices: &[usize]) -> Option<&mut T> {
        let offset = L::flat_offset(indices)?;
        self.buffer.as_mut_slice().get_mut(offset)
    }

    /// Exposes a mutable contiguous slice view of tensor memory.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.buffer.as_mut_slice()
    }
}

////////////////////////////////////////////////////////////////////////////////
// Multilinear Grid Interpolation
////////////////////////////////////////////////////////////////////////////////

impl<T: Float + Copy, L: TensorLayout, B: FlatBuffer<T>> Tensor<T, L, B> {
    /// Evaluates multilinear continuous interpolation at fractional coordinate `coords`.
    ///
    /// Evaluates a weighted sum over the $2^{\text{RANK}}$ surrounding hypercube vertices:
    /// $$\hat{f}(x) = \sum_{v \in \{0, 1\}^{\text{RANK}}} f(x_{\lfloor \rfloor} + v) \prod_{d=0}^{\text{RANK}-1} (v_d (x_d - \lfloor x_d \rfloor) + (1 - v_d)(1 - (x_d - \lfloor x_d \rfloor)))$$
    #[must_use]
    pub fn interpolate(&self, coords: &[T]) -> T {
        let rank = L::RANK;
        let mut dims = [0usize; 8];
        L::dims(&mut dims[..rank]);

        let num_corners = 1usize << rank;
        let mut result = T::ZERO;

        for corner_mask in 0..num_corners {
            let mut corner_indices = [0usize; 8];
            let mut weight = T::ONE;

            for d in 0..rank {
                let coord = coords.get(d).copied().unwrap_or(T::ZERO);
                let dim_max = if dims[d] > 0 { dims[d] - 1 } else { 0 };

                // Clamp coordinate to grid range [0, dim_max]
                let clamped = if coord < T::ZERO {
                    T::ZERO
                } else {
                    let max_t = {
                        let mut m = T::ZERO;
                        for _ in 0..dim_max {
                            m = m + T::ONE;
                        }
                        m
                    };
                    if coord > max_t { max_t } else { coord }
                };

                let base_idx = {
                    // Integer floor of clamped coordinate
                    let mut idx = 0usize;
                    let mut acc = T::ZERO;
                    while idx < dim_max && (acc + T::ONE) <= clamped {
                        acc = acc + T::ONE;
                        idx += 1;
                    }
                    idx
                };

                let frac = clamped - {
                    let mut acc = T::ZERO;
                    for _ in 0..base_idx {
                        acc = acc + T::ONE;
                    }
                    acc
                };

                let is_upper = (corner_mask & (1 << d)) != 0;
                if is_upper {
                    corner_indices[d] = (base_idx + 1).min(dim_max);
                    weight = weight * frac;
                } else {
                    corner_indices[d] = base_idx;
                    weight = weight * (T::ONE - frac);
                }
            }

            if let Some(&val) = self.get(&corner_indices[..rank]) {
                result = result + weight * val;
            }
        }

        result
    }
}

////////////////////////////////////////////////////////////////////////////////
// Activation Functions
////////////////////////////////////////////////////////////////////////////////

/// Activation function contract.
pub trait Activation<T> {
    /// Evaluates the activation function scalar-wise.
    fn apply(&self, x: T) -> T;
}

/// Rectified Linear Unit (ReLU): `max(0, x)`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Relu;

impl<T: Float + Copy> Activation<T> for Relu {
    fn apply(&self, x: T) -> T {
        if x > T::ZERO { x } else { T::ZERO }
    }
}

/// Piecewise-linear table-driven activation function (e.g. sigmoid or tanh).
#[derive(Debug, Clone, Copy)]
pub struct TableActivation<T, const N: usize> {
    /// Ordered breakpoint locations.
    pub breakpoints: [T; N],
    /// Precomputed values at breakpoints.
    pub values: [T; N],
}

impl<T: Float + Copy, const N: usize> Activation<T> for TableActivation<T, N> {
    fn apply(&self, x: T) -> T {
        if N == 0 {
            return T::ZERO;
        }
        if x <= self.breakpoints[0] {
            return self.values[0];
        }
        if x >= self.breakpoints[N - 1] {
            return self.values[N - 1];
        }

        for i in 0..(N - 1) {
            let x0 = self.breakpoints[i];
            let x1 = self.breakpoints[i + 1];
            if x >= x0 && x <= x1 {
                let dx = x1 - x0;
                if dx == T::ZERO {
                    return self.values[i];
                }
                let t = (x - x0) / dx;
                return self.values[i]
                    + t * (self.values[i + 1] - self.values[i]);
            }
        }

        self.values[N - 1]
    }
}
