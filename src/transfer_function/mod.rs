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
use crate::math::dsp::{Convolution, DefaultDsp};
use crate::math::num_traits::{Float, Scalar, Zero};
use crate::math::num_types::{Const, Dim};
use crate::math::storage::{
    ArrayStorage, ContiguousStorage, DenseStorage, DenseStorageMut, Storage,
    StorageView, StorageViewMut,
};
use crate::math::subprograms::{
    DefaultBlas,
    level1::{Axpy, Scal},
};
use crate::math::{LinAlgError, LinAlgResult};
use crate::matrix::Owned;
use crate::polynomial::ArrayPolynomial;
use crate::state_space::StateSpace;
use core::fmt;
use core::marker::PhantomData;

/// Errors from validating constructors and canonical conversions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferFunctionError {
    /// Denominator leading coefficient is zero.
    ZeroLeadingDenominatorCoefficient,
    /// Numerator degree exceeds denominator degree ($N > D$).
    ImproperSystem,
}

impl fmt::Display for TransferFunctionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroLeadingDenominatorCoefficient => {
                write!(f, "denominator leading coefficient must be non-zero")
            }
            Self::ImproperSystem => {
                write!(f, "transfer function is improper (N > D)")
            }
        }
    }
}

impl core::error::Error for TransferFunctionError {}

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
    ///
    /// Empty numerator (`N == 0`) or denominator (`D == 0`) coefficient arrays
    /// are accepted; the corresponding storage is left empty.
    #[must_use]
    pub const fn from_coefficients(
        num: [T; N],
        den: [T; D],
        sample_time: Option<T>,
    ) -> Self {
        Self {
            num_storage: ArrayStorage::from_column(num),
            den_storage: ArrayStorage::from_column(den),
            sample_time,
            _marker: PhantomData,
        }
    }

    /// Zero-copy strided view of numerator and denominator coefficients.
    #[must_use]
    pub fn view(&self) -> TransferFunctionView<'_, T, N, D> {
        let num_storage = unsafe {
            StorageView::new_with_strides_unchecked(
                self.num_storage.as_ptr(),
                self.num_storage.r_stride(),
                self.num_storage.c_stride(),
            )
        };
        let den_storage = unsafe {
            StorageView::new_with_strides_unchecked(
                self.den_storage.as_ptr(),
                self.den_storage.r_stride(),
                self.den_storage.c_stride(),
            )
        };
        TransferFunction::from_storage(
            num_storage,
            den_storage,
            self.sample_time,
        )
    }

    /// Zero-copy mutable strided view of numerator and denominator coefficients.
    pub fn view_mut(&mut self) -> TransferFunctionViewMut<'_, T, N, D> {
        let sample_time = self.sample_time;
        let num_storage = unsafe {
            StorageViewMut::new_with_strides_unchecked(
                self.num_storage.as_mut_ptr(),
                self.num_storage.r_stride(),
                self.num_storage.c_stride(),
            )
        };
        let den_storage = unsafe {
            StorageViewMut::new_with_strides_unchecked(
                self.den_storage.as_mut_ptr(),
                self.den_storage.r_stride(),
                self.den_storage.c_stride(),
            )
        };
        TransferFunction::from_storage(num_storage, den_storage, sample_time)
    }
}

impl<T: Float + Copy, const N: usize, const D: usize>
    ArrayTransferFunction<T, N, D>
where
    Const<N>: Dim,
    Const<D>: Dim,
{
    /// Validating constructor: non-zero leading denominator and $N \le D$.
    pub fn try_from_coefficients(
        num: [T; N],
        den: [T; D],
        sample_time: Option<T>,
    ) -> Result<Self, TransferFunctionError> {
        if N > D {
            return Err(TransferFunctionError::ImproperSystem);
        }
        let leading = den[D.saturating_sub(1)];
        if leading.abs() < T::epsilon() {
            return Err(
                TransferFunctionError::ZeroLeadingDenominatorCoefficient,
            );
        }
        Ok(Self::from_coefficients(num, den, sample_time))
    }

    /// Validating continuous-time constructor.
    pub fn try_continuous(
        num: [T; N],
        den: [T; D],
    ) -> Result<Self, TransferFunctionError> {
        Self::try_from_coefficients(num, den, None)
    }

    /// Validating discrete-time constructor.
    pub fn try_discrete(
        num: [T; N],
        den: [T; D],
        dt: T,
    ) -> Result<Self, TransferFunctionError> {
        Self::try_from_coefficients(num, den, Some(dt))
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
    ///
    /// Empty numerator or denominator storage evaluates to zero (caller still
    /// divides by the denominator value, so a zero-length denominator yields a
    /// complex NaN/Inf per `T`'s division rules).
    #[must_use]
    pub fn evaluate_complex(&self, s: Complex<T>) -> Complex<T> {
        let n_len = self.num_storage.rows();
        let d_len = self.den_storage.rows();

        // `saturating_sub(1)` avoids `usize` underflow when length is 0
        // (`0usize - 1` wraps to `usize::MAX` and would spin forever).
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
        for i in (0..n_len.saturating_sub(1)).rev() {
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
        for i in (0..d_len.saturating_sub(1)).rev() {
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
    /// Series connection using DSP backend `C`.
    pub fn series_with<
        C,
        const N2: usize,
        const D2: usize,
        const NOUT: usize,
        const DOUT: usize,
    >(
        &self,
        rhs: &ArrayTransferFunction<T, N2, D2>,
    ) -> ArrayTransferFunction<T, NOUT, DOUT>
    where
        C: Convolution<T>,
        Const<N2>: Dim,
        Const<D2>: Dim,
        Const<NOUT>: Dim,
        Const<DOUT>: Dim,
    {
        let num_storage = convolve_poly::<C, T, N1, N2, NOUT>(
            &self.num_storage,
            &rhs.num_storage,
        );
        let den_storage = convolve_poly::<C, T, D1, D2, DOUT>(
            &self.den_storage,
            &rhs.den_storage,
        );

        ArrayTransferFunction::from_storage(
            num_storage,
            den_storage,
            self.sample_time,
        )
    }

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
        self.series_with::<DefaultDsp, N2, D2, NOUT, DOUT>(rhs)
    }

    /// Parallel connection using DSP backend `C` and BLAS backend `B`.
    pub fn parallel_with<
        C,
        B,
        const N2: usize,
        const D2: usize,
        const NOUT: usize,
        const DOUT: usize,
    >(
        &self,
        rhs: &ArrayTransferFunction<T, N2, D2>,
    ) -> ArrayTransferFunction<T, NOUT, DOUT>
    where
        C: Convolution<T>,
        B: Axpy<T, ArrayStorage<T, NOUT, 1>, ArrayStorage<T, NOUT, 1>>,
        Const<N2>: Dim,
        Const<D2>: Dim,
        Const<NOUT>: Dim,
        Const<DOUT>: Dim,
    {
        let mut num = convolve_poly::<C, T, N1, D2, NOUT>(
            &self.num_storage,
            &rhs.den_storage,
        );
        let b2a1 = convolve_poly::<C, T, N2, D1, NOUT>(
            &rhs.num_storage,
            &self.den_storage,
        );
        B::axpy(T::ONE, &b2a1, &mut num);
        let den = convolve_poly::<C, T, D1, D2, DOUT>(
            &self.den_storage,
            &rhs.den_storage,
        );
        ArrayTransferFunction::from_storage(num, den, self.sample_time)
    }

    /// Parallel connection: $H_1 + H_2 = (B_1 A_2 + B_2 A_1) / (A_1 A_2)$.
    ///
    /// `NOUT >= N1+D2-1` and `NOUT >= N2+D1-1`; `DOUT >= D1+D2-1`.
    pub fn parallel<
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
        self.parallel_with::<DefaultDsp, DefaultBlas, N2, D2, NOUT, DOUT>(rhs)
    }

    /// Negative feedback using DSP backend `C` and BLAS backend `B`.
    pub fn feedback_with<
        C,
        B,
        const N2: usize,
        const D2: usize,
        const NOUT: usize,
        const DOUT: usize,
    >(
        &self,
        rhs: &ArrayTransferFunction<T, N2, D2>,
    ) -> ArrayTransferFunction<T, NOUT, DOUT>
    where
        C: Convolution<T>,
        B: Axpy<T, ArrayStorage<T, DOUT, 1>, ArrayStorage<T, DOUT, 1>>,
        Const<N2>: Dim,
        Const<D2>: Dim,
        Const<NOUT>: Dim,
        Const<DOUT>: Dim,
    {
        let num = convolve_poly::<C, T, N1, D2, NOUT>(
            &self.num_storage,
            &rhs.den_storage,
        );
        let mut den = convolve_poly::<C, T, D1, D2, DOUT>(
            &self.den_storage,
            &rhs.den_storage,
        );
        let b1b2 = convolve_poly::<C, T, N1, N2, DOUT>(
            &self.num_storage,
            &rhs.num_storage,
        );
        B::axpy(T::ONE, &b1b2, &mut den);
        ArrayTransferFunction::from_storage(num, den, self.sample_time)
    }

    /// Negative feedback: $H_1 / (1 + H_1 H_2) = (B_1 A_2) / (A_1 A_2 + B_1 B_2)$.
    ///
    /// `NOUT >= N1+D2-1`; `DOUT` is at least the larger of `D1+D2-1` and `N1+N2-1`.
    pub fn feedback<
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
        self.feedback_with::<DefaultDsp, DefaultBlas, N2, D2, NOUT, DOUT>(rhs)
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
    /// Converts a proper transfer function into Controllable Canonical Form using BLAS backend `B`.
    pub fn to_controllable_canonical_form_with<B, const ORDER: usize>(
        &self,
    ) -> Result<StateSpace<T, ORDER, 1, 1>, TransferFunctionError>
    where
        Const<ORDER>: Dim,
        Const<1>: Dim,
        B: Scal<T, ArrayStorage<T, ORDER, 1>>
            + Axpy<T, ArrayStorage<T, ORDER, 1>, ArrayStorage<T, ORDER, 1>>,
    {
        let (a_mat, b_mat, c_mat, d_mat) =
            self.canonical_blocks_with::<B, ORDER>()?;
        Ok(match self.sample_time {
            None => StateSpace::continuous(a_mat, b_mat, c_mat, d_mat),
            Some(dt) => StateSpace::discrete(a_mat, b_mat, c_mat, d_mat, dt),
        })
    }

    /// Converts a proper transfer function ($N \le D$) into Controllable Canonical Form.
    ///
    /// System state dimension must satisfy `ORDER = D - 1`. For a monic
    /// denominator $s^n + a_{n-1}s^{n-1} + \dots + a_0$ the realization is:
    ///
    /// $$
    /// A = \begin{bmatrix}
    /// 0 & 1 & 0 & \dots & 0 \\
    /// 0 & 0 & 1 & \dots & 0 \\
    /// \vdots & \vdots & \vdots & \ddots & \vdots \\
    /// -a_0 & -a_1 & -a_2 & \dots & -a_{n-1}
    /// \end{bmatrix},\quad
    /// B = \begin{bmatrix} 0 \\ \vdots \\ 1 \end{bmatrix}
    /// $$
    /// $$
    /// C = \begin{bmatrix} \beta_0 & \dots & \beta_{n-1} \end{bmatrix},\quad
    /// D = \begin{bmatrix} d \end{bmatrix}
    /// $$
    /// with feedthrough $d = b_n / a_n$ (zero when strictly proper) and
    /// $\beta_i = b_i / a_n - d \cdot a_i$.
    ///
    /// # Errors
    /// Returns [`TransferFunctionError`] if $N > D$, $D < 2$, or the leading
    /// denominator coefficient is zero.
    pub fn to_controllable_canonical_form<const ORDER: usize>(
        &self,
    ) -> Result<StateSpace<T, ORDER, 1, 1>, TransferFunctionError>
    where
        Const<ORDER>: Dim,
        Const<1>: Dim,
    {
        self.to_controllable_canonical_form_with::<DefaultBlas, ORDER>()
    }

    /// Observable canonical form using BLAS backend `B`.
    pub fn to_observable_canonical_form_with<B, const ORDER: usize>(
        &self,
    ) -> Result<StateSpace<T, ORDER, 1, 1>, TransferFunctionError>
    where
        Const<ORDER>: Dim,
        Const<1>: Dim,
        B: Scal<T, ArrayStorage<T, ORDER, 1>>
            + Axpy<T, ArrayStorage<T, ORDER, 1>, ArrayStorage<T, ORDER, 1>>,
    {
        let (a_ccf, b_ccf, c_ccf, d_mat) =
            self.canonical_blocks_with::<B, ORDER>()?;
        let a_mat = a_ccf.transpose();
        let b_mat = c_ccf.transpose();
        let c_mat = b_ccf.transpose();
        Ok(match self.sample_time {
            None => StateSpace::continuous(a_mat, b_mat, c_mat, d_mat),
            Some(dt) => StateSpace::discrete(a_mat, b_mat, c_mat, d_mat, dt),
        })
    }

    /// Observable canonical form (dual of last-row CCF).
    pub fn to_observable_canonical_form<const ORDER: usize>(
        &self,
    ) -> Result<StateSpace<T, ORDER, 1, 1>, TransferFunctionError>
    where
        Const<ORDER>: Dim,
        Const<1>: Dim,
    {
        self.to_observable_canonical_form_with::<DefaultBlas, ORDER>()
    }

    fn canonical_blocks_with<B, const ORDER: usize>(
        &self,
    ) -> Result<
        (
            Owned<T, ORDER, ORDER>,
            Owned<T, ORDER, 1>,
            Owned<T, 1, ORDER>,
            Owned<T, 1, 1>,
        ),
        TransferFunctionError,
    >
    where
        Const<ORDER>: Dim,
        B: Scal<T, ArrayStorage<T, ORDER, 1>>
            + Axpy<T, ArrayStorage<T, ORDER, 1>, ArrayStorage<T, ORDER, 1>>,
    {
        if ORDER + 1 != D || ORDER == 0 {
            return Err(TransferFunctionError::ImproperSystem);
        }
        if N > D {
            return Err(TransferFunctionError::ImproperSystem);
        }
        let a_n = self.den_storage.get(ORDER, 0).copied().unwrap_or(T::ZERO);
        if a_n.abs() < T::epsilon() {
            return Err(
                TransferFunctionError::ZeroLeadingDenominatorCoefficient,
            );
        }

        let mut a_col =
            Self::copy_col_prefix::<ORDER, D>(&self.den_storage, ORDER);
        B::scal(T::ONE / a_n, a_col.storage_mut());

        // Direct feedthrough d = b_n / a_n when deg(num) == deg(den).
        let d = if N == D {
            self.num_storage.get(ORDER, 0).copied().unwrap_or(T::ZERO) / a_n
        } else {
            T::ZERO
        };

        // β = b / a_n - d · a
        let mut beta =
            Self::copy_col_prefix::<ORDER, N>(&self.num_storage, ORDER.min(N));
        B::scal(T::ONE / a_n, beta.storage_mut());
        B::axpy(T::ZERO - d, a_col.storage(), beta.storage_mut());

        // Controllable companion: ones on the superdiagonal, -a on the last row.
        let mut a_mat = Owned::<T, ORDER, ORDER>::zero();
        for i in 0..(ORDER.saturating_sub(1)) {
            if let Some(elem) = a_mat.get_mut(i, i + 1) {
                *elem = T::ONE;
            }
        }
        for i in 0..ORDER {
            if let Some(elem) = a_mat.get_mut(ORDER - 1, i) {
                *elem = T::ZERO - a_col.get(i, 0).copied().unwrap_or(T::ZERO);
            }
        }

        let mut b_mat = Owned::<T, ORDER, 1>::zero();
        if let Some(elem) = b_mat.get_mut(ORDER - 1, 0) {
            *elem = T::ONE;
        }

        let mut c_mat = Owned::<T, 1, ORDER>::zero();
        for i in 0..ORDER {
            if let Some(elem) = c_mat.get_mut(0, i) {
                *elem = beta.get(i, 0).copied().unwrap_or(T::ZERO);
            }
        }

        let mut d_mat = Owned::<T, 1, 1>::zero();
        if let Some(elem) = d_mat.get_mut(0, 0) {
            *elem = d;
        }

        Ok((a_mat, b_mat, c_mat, d_mat))
    }

    fn copy_col_prefix<const ORDER: usize, const SRC: usize>(
        src: &ArrayStorage<T, SRC, 1>,
        count: usize,
    ) -> Owned<T, ORDER, 1>
    where
        Const<ORDER>: Dim,
        Const<SRC>: Dim,
    {
        let mut col = Owned::<T, ORDER, 1>::zero();
        for i in 0..count {
            if let (Some(dst), Some(&v)) = (col.get_mut(i, 0), src.get(i, 0)) {
                *dst = v;
            }
        }
        col
    }

    /// Tustin discretization $s = \frac{2}{T_s}\frac{z-1}{z+1}$ with optional pre-warp.
    ///
    /// Both numerator and denominator are cleared against $(z+1)^{D-1}$ so a
    /// strictly proper continuous plant (relative degree $r > 0$) gains the
    /// required $(z+1)^{r}$ numerator factor. The discrete result is therefore
    /// biproper with coefficient capacities `(D, D)`, matching
    /// [`Self::to_discrete_zoh`].
    #[must_use]
    pub fn to_discrete_tustin(
        &self,
        sample_time: T,
        prewarp_frequency: Option<T>,
    ) -> ArrayTransferFunction<T, D, D> {
        let two = T::ONE + T::ONE;
        let k = match prewarp_frequency {
            None => two / sample_time,
            Some(wc) => wc / (wc * sample_time / two).tan(),
        };
        let ts_eff = two / k;
        let mut num_coeffs = [T::ZERO; D];
        let n_copy = core::cmp::min(N, D);
        for i in 0..n_copy {
            num_coeffs[i] =
                self.num_storage.get(i, 0).copied().unwrap_or(T::ZERO);
        }
        let num = ArrayPolynomial::<T, D>::from_coefficients(num_coeffs)
            .compose_bilinear(ts_eff);
        let den = ArrayPolynomial::<T, D>::from_storage(self.den_storage)
            .compose_bilinear(ts_eff);
        ArrayTransferFunction::from_storage(
            num.into_storage(),
            den.into_storage(),
            Some(sample_time),
        )
    }

    /// ZOH via last-row CCF, Van Loan `StateSpace::to_discrete_zoh`, then SISO TF.
    ///
    /// `ORDER` is the state dimension $D-1$.
    pub fn to_discrete_zoh<const ORDER: usize>(
        &self,
        sample_time: T,
    ) -> LinAlgResult<ArrayTransferFunction<T, D, D>>
    where
        Const<ORDER>: Dim,
    {
        let ss = self
            .to_controllable_canonical_form::<ORDER>()
            .map_err(|_| LinAlgError::SingularMatrix)?;
        let dss = ss.to_discrete_zoh(sample_time);
        Ok(dss.to_transfer_function::<D>())
    }
}

fn convolve_poly<
    C: Convolution<T>,
    T: Scalar + Copy,
    const NA: usize,
    const NB: usize,
    const NO: usize,
>(
    a: &ArrayStorage<T, NA, 1>,
    b: &ArrayStorage<T, NB, 1>,
) -> ArrayStorage<T, NO, 1>
where
    Const<NA>: Dim,
    Const<NB>: Dim,
    Const<NO>: Dim,
{
    let mut out = ArrayStorage::<T, NO, 1>::zero();
    let _ = C::convolve_input(a.as_slice(), b.as_slice(), out.as_mut_slice());
    out
}
