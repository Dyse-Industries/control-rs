//! # Transfer Function Module
//!
//! Generic, statically sized rational transfer function representation [`TransferFunction`]
//! decoupled from physical memory storage via [`crate::math::storage`].
//!
//! Represents rational transfer functions:
//! $$H(s) = \frac{B(s)}{A(s)} = \frac{b_0 + b_1 s + \dots + b_{N-1} s^{N-1}}{a_0 + a_1 s + \dots + a_{D-1} s^{D-1}}$$
//! in ascending polynomial power order.
//!
//! # Examples
//!
//! ```rust
//! use control_rs::math::num_types::Const;
//! use control_rs::transfer_function::ArrayTransferFunction;
//!
//! // Continuous 1st-order lowpass filter: H(s) = 1 / (1 + s)
//! let lp = ArrayTransferFunction::<f64, 1, 2>::continuous([1.0], [1.0, 1.0]);
//! let h_dc = lp.eval_frequency(0.0);
//! assert_eq!(h_dc.re, 1.0);
//! assert_eq!(h_dc.im, 0.0);
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
    clippy::cast_lossless
)]

#[cfg(any(test, feature = "ets"))]
/// Transfer function module unit tests.
pub mod tests;

use crate::math::complex_num::Complex;
use crate::math::num_traits::{Float, Scalar, Zero};
use crate::math::num_types::{Const, Dim};
use crate::math::storage::{
    ArrayStorage, ContiguousStorage, DenseStorage, Storage, StorageView,
    StorageViewMut,
};
use crate::math::{LinAlgError, LinAlgResult};
use crate::matrix::Owned;
use crate::polynomial::ArrayPolynomial;
use crate::state_space::StateSpace;
use core::marker::PhantomData;

/// Rational transfer function $H(s) = B(s) / A(s)$ over numerator storage `Sn` and denominator storage `Sd`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransferFunction<
    T,
    N: Dim,
    D: Dim,
    Sn: Storage<T, N, Const<1>>,
    Sd: Storage<T, D, Const<1>>,
> {
    num_storage: Sn,
    den_storage: Sd,
    sample_time: Option<T>, // None = Continuous (s-domain), Some(Ts) = Discrete (z-domain)
    _marker: PhantomData<(N, D)>,
}

/// Owning stack-allocated transfer function.
pub type ArrayTransferFunction<T, const N: usize, const D: usize> =
    TransferFunction<
        T,
        Const<N>,
        Const<D>,
        ArrayStorage<T, N, 1>,
        ArrayStorage<T, D, 1>,
    >;

/// Read-only borrowed transfer function view.
pub type TransferFunctionView<'a, T, const N: usize, const D: usize> =
    TransferFunction<
        T,
        Const<N>,
        Const<D>,
        StorageView<'a, T, Const<N>, Const<1>>,
        StorageView<'a, T, Const<D>, Const<1>>,
    >;

/// Mutable borrowed transfer function view.
pub type TransferFunctionViewMut<'a, T, const N: usize, const D: usize> =
    TransferFunction<
        T,
        Const<N>,
        Const<D>,
        StorageViewMut<'a, T, Const<N>, Const<1>>,
        StorageViewMut<'a, T, Const<D>, Const<1>>,
    >;

////////////////////////////////////////////////////////////////////////////////
// Constructors & Accessors
////////////////////////////////////////////////////////////////////////////////

impl<T: Copy, const N: usize, const D: usize> ArrayTransferFunction<T, N, D>
where
    Const<N>: Dim,
    Const<D>: Dim,
{
    /// Builds a continuous-time transfer function from numerator and denominator coefficient arrays (ascending order).
    #[must_use]
    pub const fn continuous(num: [T; N], den: [T; D]) -> Self {
        Self::from_coefficients(num, den, None)
    }

    /// Builds a discrete-time transfer function with sample interval `dt`.
    #[must_use]
    pub const fn discrete(num: [T; N], den: [T; D], dt: T) -> Self {
        Self::from_coefficients(num, den, Some(dt))
    }

    /// Builds a transfer function from coefficient arrays.
    #[must_use]
    pub const fn from_coefficients(
        num: [T; N],
        den: [T; D],
        sample_time: Option<T>,
    ) -> Self {
        let mut num_data = [[num[0]; N]; 1];
        let mut i = 0;
        while i < N {
            num_data[0][i] = num[i];
            i += 1;
        }

        let mut den_data = [[den[0]; D]; 1];
        let mut j = 0;
        while j < D {
            den_data[0][j] = den[j];
            j += 1;
        }

        Self {
            num_storage: ArrayStorage::from_array(num_data),
            den_storage: ArrayStorage::from_array(den_data),
            sample_time,
            _marker: PhantomData,
        }
    }
}

impl<
    T,
    N: Dim,
    D: Dim,
    Sn: Storage<T, N, Const<1>>,
    Sd: Storage<T, D, Const<1>>,
> TransferFunction<T, N, D, Sn, Sd>
{
    /// Wraps custom storage backends.
    pub const fn from_storage(
        num_storage: Sn,
        den_storage: Sd,
        sample_time: Option<T>,
    ) -> Self {
        Self {
            num_storage,
            den_storage,
            sample_time,
            _marker: PhantomData,
        }
    }

    /// Returns the sampling time, or `None` if continuous.
    #[must_use]
    pub const fn sample_time(&self) -> Option<T>
    where
        T: Copy,
    {
        self.sample_time
    }

    /// Checks if continuous-time ($s$-domain).
    #[must_use]
    pub const fn is_continuous(&self) -> bool {
        self.sample_time.is_none()
    }

    /// Checks if discrete-time ($z$-domain).
    #[must_use]
    pub const fn is_discrete(&self) -> bool {
        self.sample_time.is_some()
    }
}

impl<
    T,
    N: Dim,
    D: Dim,
    Sn: Storage<T, N, Const<1>> + ContiguousStorage<T>,
    Sd: Storage<T, D, Const<1>> + ContiguousStorage<T>,
> TransferFunction<T, N, D, Sn, Sd>
{
    /// Exposes a safe slice of numerator coefficients in ascending power order.
    #[must_use]
    pub fn num_slice(&self) -> &[T] {
        self.num_storage.as_slice()
    }

    /// Exposes a safe slice of denominator coefficients in ascending power order.
    #[must_use]
    pub fn den_slice(&self) -> &[T] {
        self.den_storage.as_slice()
    }
}

////////////////////////////////////////////////////////////////////////////////
// Frequency Response Evaluation
////////////////////////////////////////////////////////////////////////////////

impl<
    T: Float + Copy,
    N: Dim,
    D: Dim,
    Sn: Storage<T, N, Const<1>>,
    Sd: Storage<T, D, Const<1>>,
> TransferFunction<T, N, D, Sn, Sd>
{
    /// Evaluates the rational transfer function at complex coordinate `s`:
    /// $$H(s) = \frac{B(s)}{A(s)} = \frac{\text{Horner}(B, s)}{\text{Horner}(A, s)}$$
    #[must_use]
    pub fn evaluate_complex(&self, s: Complex<T>) -> Complex<T> {
        let n_len = self.num_storage.rows();
        let d_len = self.den_storage.rows();

        let mut num_val = if n_len > 0 {
            let c = self
                .num_storage
                .get(n_len - 1, 0)
                .copied()
                .unwrap_or(T::ZERO);
            Complex::new(c, T::ZERO)
        } else {
            Complex::ZERO
        };
        for i in (0..(n_len - 1)).rev() {
            let c = self.num_storage.get(i, 0).copied().unwrap_or(T::ZERO);
            num_val = num_val * s + Complex::new(c, T::ZERO);
        }

        let mut den_val = if d_len > 0 {
            let c = self
                .den_storage
                .get(d_len - 1, 0)
                .copied()
                .unwrap_or(T::ZERO);
            Complex::new(c, T::ZERO)
        } else {
            Complex::ZERO
        };
        for i in (0..(d_len - 1)).rev() {
            let c = self.den_storage.get(i, 0).copied().unwrap_or(T::ZERO);
            den_val = den_val * s + Complex::new(c, T::ZERO);
        }

        num_val / den_val
    }

    /// Evaluates frequency response $H(j\omega)$ (continuous) or $H(e^{j\omega T_s})$ (discrete) at angular frequency `omega`.
    #[must_use]
    pub fn eval_frequency(&self, omega: T) -> Complex<T> {
        match self.sample_time {
            None => {
                // Continuous time: s = j * omega
                let s = Complex::new(T::ZERO, omega);
                self.evaluate_complex(s)
            }
            Some(dt) => {
                // Discrete time: z = e^{j * omega * dt} = cos(omega * dt) + j * sin(omega * dt)
                let theta = omega * dt;
                let z = Complex::new(theta.cos(), theta.sin());
                self.evaluate_complex(z)
            }
        }
    }

    /// Evaluates Bode magnitude $|H(j\omega)|$ and phase $\angle H(j\omega)$ (in radians).
    #[must_use]
    pub fn bode_point(&self, omega: T) -> (T, T) {
        let resp = self.eval_frequency(omega);
        let mag = (resp.re * resp.re + resp.im * resp.im).sqrt();
        let phase = resp.im.atan2(resp.re);
        (mag, phase)
    }
}

////////////////////////////////////////////////////////////////////////////////
// System Interconnections & Algebra
////////////////////////////////////////////////////////////////////////////////

impl<T: Scalar + Copy, const N1: usize, const D1: usize>
    ArrayTransferFunction<T, N1, D1>
where
    Const<N1>: Dim,
    Const<D1>: Dim,
{
    /// Series (cascade) connection: $H_{\text{series}} = H_1 \cdot H_2 = \frac{B_1 B_2}{A_1 A_2}$.
    ///
    /// Capacity: $N_{\text{out}} = N_1 + N_2 - 1$, $D_{\text{out}} = D_1 + D_2 - 1$.
    pub fn series<
        const N2: usize,
        const D2: usize,
        const NOUT: usize,
        const DOUT: usize,
    >(
        &self,
        rhs: &ArrayTransferFunction<T, N2, D2>,
    ) -> ArrayTransferFunction<T, NOUT, DOUT>
    where
        Const<N2>: Dim,
        Const<D2>: Dim,
        Const<NOUT>: Dim,
        Const<DOUT>: Dim,
    {
        let num1 = ArrayPolynomial::<T, N1>::from_storage(self.num_storage);
        let num2 = ArrayPolynomial::<T, N2>::from_storage(rhs.num_storage);
        let num_prod = num1.mul_poly::<N2, NOUT>(&num2);

        let den1 = ArrayPolynomial::<T, D1>::from_storage(self.den_storage);
        let den2 = ArrayPolynomial::<T, D2>::from_storage(rhs.den_storage);
        let den_prod = den1.mul_poly::<D2, DOUT>(&den2);

        ArrayTransferFunction {
            num_storage: num_prod.into_storage(),
            den_storage: den_prod.into_storage(),
            sample_time: self.sample_time,
            _marker: PhantomData,
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Canonical State-Space Realizations
////////////////////////////////////////////////////////////////////////////////

impl<T: Float + Copy, const N: usize, const D: usize>
    ArrayTransferFunction<T, N, D>
where
    Const<N>: Dim,
    Const<D>: Dim,
{
    /// Converts a strictly proper transfer function ($N \le D - 1$) into Controllable Canonical Form.
    ///
    /// System state dimension `ORDER = D - 1`.
    ///
    /// # Errors
    /// Returns [`LinAlgError::SingularMatrix`] if $N \ge D$, $D < 2$, or leading denominator is zero.
    pub fn to_controllable_canonical_form<const ORDER: usize>(
        &self,
    ) -> LinAlgResult<StateSpace<T, ORDER, 1, 1>>
    where
        Const<ORDER>: Dim,
        Const<1>: Dim,
    {
        if ORDER + 1 != D || ORDER == 0 {
            return Err(LinAlgError::SingularMatrix);
        }
        let a_n = self.den_storage.get(ORDER, 0).copied().unwrap_or(T::ZERO);
        if a_n.abs() < T::epsilon() {
            return Err(LinAlgError::SingularMatrix);
        }

        // Normalize polynomial coefficients by a_n
        let mut a_norm = [T::ZERO; ORDER];
        for i in 0..ORDER {
            a_norm[i] =
                self.den_storage.get(i, 0).copied().unwrap_or(T::ZERO) / a_n;
        }

        let mut b_norm = [T::ZERO; ORDER];
        for i in 0..ORDER {
            if i < N {
                b_norm[i] =
                    self.num_storage.get(i, 0).copied().unwrap_or(T::ZERO)
                        / a_n;
            }
        }

        let mut a_mat = Owned::<T, ORDER, ORDER>::zero();
        for i in 1..ORDER {
            if let Some(elem) = a_mat.get_mut(i, i - 1) {
                *elem = T::ONE;
            }
        }
        for i in 0..ORDER {
            if let Some(elem) = a_mat.get_mut(i, ORDER - 1) {
                *elem = T::ZERO - a_norm[i];
            }
        }

        let mut b_mat = Owned::<T, ORDER, 1>::zero();
        if let Some(elem) = b_mat.get_mut(ORDER - 1, 0) {
            *elem = T::ONE;
        }

        let mut c_mat = Owned::<T, 1, ORDER>::zero();
        for i in 0..ORDER {
            if let Some(elem) = c_mat.get_mut(0, i) {
                *elem = b_norm[i];
            }
        }

        let d_mat = Owned::<T, 1, 1>::zero();

        Ok(match self.sample_time {
            None => StateSpace::continuous(a_mat, b_mat, c_mat, d_mat),
            Some(dt) => StateSpace::discrete(a_mat, b_mat, c_mat, d_mat, dt),
        })
    }
}
