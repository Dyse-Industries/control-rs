//! # State-Space Module
//!
//! Generic, statically-sized Linear Time-Invariant (LTI) state-space model [`StateSpaceCore`]
//! decoupled from physical memory storage via [`crate::math::storage`].
//!
//! Continuous-time dynamics:
//! $$\dot{x}(t) = A x(t) + B u(t), \quad y(t) = C x(t) + D u(t)$$
//!
//! Discrete-time dynamics:
//! $$x[k+1] = A x[k] + B u[k], \quad y[k] = C x[k] + D u[k]$$
//!
//! # Examples
//!
//! ```rust
//! use control_rs::math::num_types::Const;
//! use control_rs::matrix::Owned;
//! use control_rs::state_space::ArrayStateSpace;
//!
//! let a = Owned::<f64, 1, 1>::from_fn(|_, _| 0.5);
//! let b = Owned::<f64, 1, 1>::from_fn(|_, _| 1.0);
//! let c = Owned::<f64, 1, 1>::from_fn(|_, _| 2.0);
//! let d = Owned::<f64, 1, 1>::from_fn(|_, _| 0.0);
//!
//! let sys = ArrayStateSpace::discrete(a, b, c, d, 0.01);
//! let x = Owned::<f64, 1, 1>::zero();
//! let u = Owned::<f64, 1, 1>::from_fn(|_, _| 1.0);
//!
//! let (x_next, y) = sys.step(&x, &u);
//! assert_eq!(y.get(0, 0), Some(&0.0)); // y[0] = C*x[0] = 0
//! assert_eq!(x_next.get(0, 0), Some(&1.0)); // x[1] = A*0 + B*1 = 1
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
    clippy::borrow_as_ptr,
    clippy::ptr_as_ptr
)]

#[cfg(any(test, feature = "ets"))]
/// State-space module unit tests.
pub mod tests;

use crate::math::LinAlgResult;
use crate::math::num_traits::{Float, Scalar};
use crate::math::num_types::{Const, Dim};
use crate::math::storage::{
    ArrayStorage, DenseStorage, DenseStorageMut, StaticStorageView, Storage,
    StorageView, StorageViewMut,
};
use crate::matrix::decomposition::LuDecomposition;
use crate::matrix::{Matrix, MatrixSlice, Owned};
use crate::transfer_function::ArrayTransferFunction;
use core::fmt;
use core::marker::PhantomData;

/// Errors from feedback interconnection and Tustin discretization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateSpaceError {
    /// Feedback loop matrix $(I - \mathrm{sign}\, D_2 D_1)$ is singular.
    SingularLoopMatrix,
    /// Tustin operator $(I - T_s/2 A)$ is singular.
    SingularDiscretizationOperator,
}

impl fmt::Display for StateSpaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SingularLoopMatrix => write!(
                f,
                "feedback loop matrix (I - sign*D2*D1) is singular to working precision"
            ),
            Self::SingularDiscretizationOperator => write!(
                f,
                "Tustin discretization operator (I - Ts/2 * A) is singular to working precision"
            ),
        }
    }
}

impl core::error::Error for StateSpaceError {}

/// Result alias for fallible state-space operations.
pub type StateSpaceResult<T> = Result<T, StateSpaceError>;

/// Statically sized, generic LTI state-space container over 4 storage backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StateSpaceCore<
    T,
    NX: Dim,
    NU: Dim,
    NY: Dim,
    Sa: Storage<T, NX, NX>,
    Sb: Storage<T, NX, NU>,
    Sc: Storage<T, NY, NX>,
    Sd: Storage<T, NY, NU>,
> {
    a_storage: Sa,
    b_storage: Sb,
    c_storage: Sc,
    d_storage: Sd,
    sample_time: Option<T>, // None = Continuous, Some(dt) = Discrete
    _marker: PhantomData<(NX, NU, NY)>,
}

/// Owning stack-allocated state-space model.
pub type StateSpace<T, const NX: usize, const NU: usize, const NY: usize> =
    StateSpaceCore<
        T,
        Const<NX>,
        Const<NU>,
        Const<NY>,
        ArrayStorage<T, NX, NX>,
        ArrayStorage<T, NX, NU>,
        ArrayStorage<T, NY, NX>,
        ArrayStorage<T, NY, NU>,
    >;

/// Alias for standard stack-allocated state-space model.
pub type ArrayStateSpace<T, const NX: usize, const NU: usize, const NY: usize> =
    StateSpace<T, NX, NU, NY>;

/// Read-only borrowed state-space view.
pub type StateSpaceView<
    'a,
    T,
    const NX: usize,
    const NU: usize,
    const NY: usize,
> = StateSpaceCore<
    T,
    Const<NX>,
    Const<NU>,
    Const<NY>,
    StorageView<'a, T, Const<NX>, Const<NX>>,
    StorageView<'a, T, Const<NX>, Const<NU>>,
    StorageView<'a, T, Const<NY>, Const<NX>>,
    StorageView<'a, T, Const<NY>, Const<NU>>,
>;

/// Mutable borrowed state-space view.
pub type StateSpaceViewMut<
    'a,
    T,
    const NX: usize,
    const NU: usize,
    const NY: usize,
> = StateSpaceCore<
    T,
    Const<NX>,
    Const<NU>,
    Const<NY>,
    StorageViewMut<'a, T, Const<NX>, Const<NX>>,
    StorageViewMut<'a, T, Const<NX>, Const<NU>>,
    StorageViewMut<'a, T, Const<NY>, Const<NX>>,
    StorageViewMut<'a, T, Const<NY>, Const<NU>>,
>;

////////////////////////////////////////////////////////////////////////////////
// Constructors & Accessors
////////////////////////////////////////////////////////////////////////////////

impl<
    T,
    NX: Dim,
    NU: Dim,
    NY: Dim,
    Sa: Storage<T, NX, NX>,
    Sb: Storage<T, NX, NU>,
    Sc: Storage<T, NY, NX>,
    Sd: Storage<T, NY, NU>,
> StateSpaceCore<T, NX, NU, NY, Sa, Sb, Sc, Sd>
{
    /// Wraps four storage backends and an optional sample time.
    pub const fn from_storage(
        a_storage: Sa,
        b_storage: Sb,
        c_storage: Sc,
        d_storage: Sd,
        sample_time: Option<T>,
    ) -> Self {
        Self {
            a_storage,
            b_storage,
            c_storage,
            d_storage,
            sample_time,
            _marker: PhantomData,
        }
    }
}

impl<T: Copy, const NX: usize, const NU: usize, const NY: usize>
    StateSpace<T, NX, NU, NY>
where
    Const<NX>: Dim,
    Const<NU>: Dim,
    Const<NY>: Dim,
{
    /// Builds a continuous-time state-space model ($s$-domain).
    pub fn continuous(
        a: Owned<T, NX, NX>,
        b: Owned<T, NX, NU>,
        c: Owned<T, NY, NX>,
        d: Owned<T, NY, NU>,
    ) -> Self {
        Self::from_storage(
            a.into_storage(),
            b.into_storage(),
            c.into_storage(),
            d.into_storage(),
            None,
        )
    }

    /// Builds a discrete-time state-space model ($z$-domain) with sampling interval `dt`.
    pub fn discrete(
        a: Owned<T, NX, NX>,
        b: Owned<T, NX, NU>,
        c: Owned<T, NY, NX>,
        d: Owned<T, NY, NU>,
        dt: T,
    ) -> Self {
        Self::from_storage(
            a.into_storage(),
            b.into_storage(),
            c.into_storage(),
            d.into_storage(),
            Some(dt),
        )
    }

    /// Zero-copy strided view of $A$, $B$, $C$, and $D$.
    #[must_use]
    pub fn view(&self) -> StateSpaceView<'_, T, NX, NU, NY> {
        let a_storage = unsafe {
            StorageView::new_with_strides_unchecked(
                self.a_storage.as_ptr(),
                self.a_storage.r_stride(),
                self.a_storage.c_stride(),
            )
        };
        let b_storage = unsafe {
            StorageView::new_with_strides_unchecked(
                self.b_storage.as_ptr(),
                self.b_storage.r_stride(),
                self.b_storage.c_stride(),
            )
        };
        let c_storage = unsafe {
            StorageView::new_with_strides_unchecked(
                self.c_storage.as_ptr(),
                self.c_storage.r_stride(),
                self.c_storage.c_stride(),
            )
        };
        let d_storage = unsafe {
            StorageView::new_with_strides_unchecked(
                self.d_storage.as_ptr(),
                self.d_storage.r_stride(),
                self.d_storage.c_stride(),
            )
        };
        StateSpaceCore::from_storage(
            a_storage,
            b_storage,
            c_storage,
            d_storage,
            self.sample_time,
        )
    }

    /// Zero-copy mutable strided view of $A$, $B$, $C$, and $D$.
    pub fn view_mut(&mut self) -> StateSpaceViewMut<'_, T, NX, NU, NY> {
        let sample_time = self.sample_time;
        let a_storage = unsafe {
            StorageViewMut::new_with_strides_unchecked(
                self.a_storage.as_mut_ptr(),
                self.a_storage.r_stride(),
                self.a_storage.c_stride(),
            )
        };
        let b_storage = unsafe {
            StorageViewMut::new_with_strides_unchecked(
                self.b_storage.as_mut_ptr(),
                self.b_storage.r_stride(),
                self.b_storage.c_stride(),
            )
        };
        let c_storage = unsafe {
            StorageViewMut::new_with_strides_unchecked(
                self.c_storage.as_mut_ptr(),
                self.c_storage.r_stride(),
                self.c_storage.c_stride(),
            )
        };
        let d_storage = unsafe {
            StorageViewMut::new_with_strides_unchecked(
                self.d_storage.as_mut_ptr(),
                self.d_storage.r_stride(),
                self.d_storage.c_stride(),
            )
        };
        StateSpaceCore::from_storage(
            a_storage,
            b_storage,
            c_storage,
            d_storage,
            sample_time,
        )
    }
}

impl<
    T,
    NX: Dim,
    NU: Dim,
    NY: Dim,
    Sa: Storage<T, NX, NX>,
    Sb: Storage<T, NX, NU>,
    Sc: Storage<T, NY, NX>,
    Sd: Storage<T, NY, NU>,
> StateSpaceCore<T, NX, NU, NY, Sa, Sb, Sc, Sd>
{
    /// Borrows $A$ storage.
    pub const fn a_storage(&self) -> &Sa {
        &self.a_storage
    }

    /// Borrows $B$ storage.
    pub const fn b_storage(&self) -> &Sb {
        &self.b_storage
    }

    /// Borrows $C$ storage.
    pub const fn c_storage(&self) -> &Sc {
        &self.c_storage
    }

    /// Borrows $D$ storage.
    pub const fn d_storage(&self) -> &Sd {
        &self.d_storage
    }
}

impl<T: Copy, const NX: usize, const NU: usize, const NY: usize>
    StateSpace<T, NX, NU, NY>
where
    Const<NX>: Dim,
    Const<NU>: Dim,
    Const<NY>: Dim,
{
    /// Zero-copy [`MatrixSlice`] over $A$.
    #[must_use]
    pub fn a_matrix(&self) -> MatrixSlice<'_, T, Const<NX>, Const<NX>> {
        // SAFETY: `ArrayStorage<T, NX, NX>` length is exactly `NX * NX`.
        Matrix::from_storage(unsafe {
            StaticStorageView::new_unchecked(self.a_storage.as_slice())
        })
    }

    /// Zero-copy [`MatrixSlice`] over $B$.
    #[must_use]
    pub fn b_matrix(&self) -> MatrixSlice<'_, T, Const<NX>, Const<NU>> {
        Matrix::from_storage(unsafe {
            StaticStorageView::new_unchecked(self.b_storage.as_slice())
        })
    }

    /// Zero-copy [`MatrixSlice`] over $C$.
    #[must_use]
    pub fn c_matrix(&self) -> MatrixSlice<'_, T, Const<NY>, Const<NX>> {
        Matrix::from_storage(unsafe {
            StaticStorageView::new_unchecked(self.c_storage.as_slice())
        })
    }

    /// Zero-copy [`MatrixSlice`] over $D$.
    #[must_use]
    pub fn d_matrix(&self) -> MatrixSlice<'_, T, Const<NY>, Const<NU>> {
        Matrix::from_storage(unsafe {
            StaticStorageView::new_unchecked(self.d_storage.as_slice())
        })
    }

    /// Owned copy of $A$ (storage is `Copy` when `T` is).
    #[must_use]
    pub fn a(&self) -> Owned<T, NX, NX>
    where
        T: Copy,
    {
        Owned::from_storage(self.a_storage)
    }

    /// Owned copy of $B$.
    #[must_use]
    pub fn b(&self) -> Owned<T, NX, NU>
    where
        T: Copy,
    {
        Owned::from_storage(self.b_storage)
    }

    /// Owned copy of $C$.
    #[must_use]
    pub fn c(&self) -> Owned<T, NY, NX>
    where
        T: Copy,
    {
        Owned::from_storage(self.c_storage)
    }

    /// Owned copy of $D$.
    #[must_use]
    pub fn d(&self) -> Owned<T, NY, NU>
    where
        T: Copy,
    {
        Owned::from_storage(self.d_storage)
    }

    /// Returns the sampling time $T_s$, or `None` if continuous.
    #[must_use]
    pub const fn sample_time(&self) -> Option<T>
    where
        T: Copy,
    {
        self.sample_time
    }

    /// Checks whether the system is continuous-time.
    #[must_use]
    pub const fn is_continuous(&self) -> bool {
        self.sample_time.is_none()
    }

    /// Checks whether the system is discrete-time.
    #[must_use]
    pub const fn is_discrete(&self) -> bool {
        self.sample_time.is_some()
    }
}

////////////////////////////////////////////////////////////////////////////////
// Time-Domain Simulation
////////////////////////////////////////////////////////////////////////////////

impl<T: Scalar + Copy, const NX: usize, const NU: usize, const NY: usize>
    StateSpace<T, NX, NU, NY>
where
    Const<NX>: Dim,
    Const<NU>: Dim,
    Const<NY>: Dim,
{
    /// Advances discrete state-space dynamics by one sample without mutating `x`:
    ///
    /// $$y[k] = C x[k] + D u[k]$$
    /// $$x[k+1] = A x[k] + B u[k]$$
    #[must_use]
    pub fn step(
        &self,
        x: &Owned<T, NX, 1>,
        u: &Owned<T, NU, 1>,
    ) -> (Owned<T, NX, 1>, Owned<T, NY, 1>) {
        let ax = &self.a_matrix() * x;
        let bu = &self.b_matrix() * u;
        let x_next = &ax + &bu;
        let cx = &self.c_matrix() * x;
        let du = &self.d_matrix() * u;
        let y = &cx + &du;
        (x_next, y)
    }

    /// Computes continuous-time state derivatives and output:
    ///
    /// $$\dot{x}(t) = A x(t) + B u(t)$$
    /// $$y(t) = C x(t) + D u(t)$$
    #[must_use]
    pub fn derivative(
        &self,
        x: &Owned<T, NX, 1>,
        u: &Owned<T, NU, 1>,
    ) -> (Owned<T, NX, 1>, Owned<T, NY, 1>) {
        self.step(x, u)
    }
}

////////////////////////////////////////////////////////////////////////////////
// Interconnections
////////////////////////////////////////////////////////////////////////////////

impl<T: Scalar + Copy, const NX1: usize, const NU: usize, const NY: usize>
    StateSpace<T, NX1, NU, NY>
where
    Const<NX1>: Dim,
    Const<NU>: Dim,
    Const<NY>: Dim,
{
    /// Series (cascade) $G_2 G_1$: output of `self` feeds input of `rhs`.
    #[must_use]
    pub fn series<const NX2: usize, const NZ: usize, const NXOUT: usize>(
        &self,
        rhs: &StateSpace<T, NX2, NY, NZ>,
    ) -> StateSpace<T, NXOUT, NU, NZ>
    where
        Const<NX2>: Dim,
        Const<NZ>: Dim,
        Const<NXOUT>: Dim,
    {
        let a1 = self.a();
        let b1 = self.b();
        let c1 = self.c();
        let d1 = self.d();
        let a2 = rhs.a();
        let b2 = rhs.b();
        let c2 = rhs.c();
        let d2 = rhs.d();
        let b2c1 = &b2 * &c1;
        let b2d1 = &b2 * &d1;
        let d2c1 = &d2 * &c1;
        let d2d1 = &d2 * &d1;

        let mut a = Owned::<T, NXOUT, NXOUT>::zero();
        let mut b = Owned::<T, NXOUT, NU>::zero();
        let mut c = Owned::<T, NZ, NXOUT>::zero();
        a.write_block(0, 0, &a1);
        a.write_block(NX1, 0, &b2c1);
        a.write_block(NX1, NX1, &a2);
        b.write_block(0, 0, &b1);
        b.write_block(NX1, 0, &b2d1);
        c.write_block(0, 0, &d2c1);
        c.write_block(0, NX1, &c2);

        match (self.sample_time, rhs.sample_time) {
            (Some(dt), Some(_)) => StateSpace::discrete(a, b, c, d2d1, dt),
            _ => StateSpace::continuous(a, b, c, d2d1),
        }
    }

    /// Parallel connection $G_1 + G_2$ (identical $N_u$, $N_y$).
    #[must_use]
    pub fn parallel<const NX2: usize, const NXOUT: usize>(
        &self,
        rhs: &StateSpace<T, NX2, NU, NY>,
    ) -> StateSpace<T, NXOUT, NU, NY>
    where
        Const<NX2>: Dim,
        Const<NXOUT>: Dim,
    {
        let a1 = self.a();
        let b1 = self.b();
        let c1 = self.c();
        let d1 = self.d();
        let a2 = rhs.a();
        let b2 = rhs.b();
        let c2 = rhs.c();
        let d2 = rhs.d();

        let mut a = Owned::<T, NXOUT, NXOUT>::zero();
        let mut b = Owned::<T, NXOUT, NU>::zero();
        let mut c = Owned::<T, NY, NXOUT>::zero();
        a.write_block(0, 0, &a1);
        a.write_block(NX1, NX1, &a2);
        b.write_block(0, 0, &b1);
        b.write_block(NX1, 0, &b2);
        c.write_block(0, 0, &c1);
        c.write_block(0, NX1, &c2);
        let d = &d1 + &d2;

        match (self.sample_time, rhs.sample_time) {
            (Some(dt), Some(_)) => StateSpace::discrete(a, b, c, d, dt),
            _ => StateSpace::continuous(a, b, c, d),
        }
    }

    /// Feedback interconnection. `sign = -1` is negative feedback.
    ///
    /// `rhs` maps plant outputs ($N_y$) to plant inputs ($N_u$).
    pub fn feedback<const NX2: usize, const NXOUT: usize>(
        &self,
        rhs: &StateSpace<T, NX2, NY, NU>,
        sign: T,
    ) -> StateSpaceResult<StateSpace<T, NXOUT, NU, NY>>
    where
        T: Float,
        Const<NX2>: Dim,
        Const<NXOUT>: Dim,
    {
        let a1 = self.a();
        let b1 = self.b();
        let c1 = self.c();
        let d1 = self.d();
        let a2 = rhs.a();
        let b2 = rhs.b();
        let c2 = rhs.c();
        let d2 = rhs.d();

        // F = I - sign D2 D1
        let d2d1 = &d2 * &d1;
        let mut f = Owned::<T, NU, NU>::identity();
        for i in 0..NU {
            for j in 0..NU {
                if let (Some(target), Some(&v)) =
                    (f.get_mut(i, j), d2d1.get(i, j))
                {
                    *target = *target - sign * v;
                }
            }
        }
        let lu = LuDecomposition::decompose(f)
            .map_err(|_| StateSpaceError::SingularLoopMatrix)?;
        let f_inv = lu
            .inverse()
            .map_err(|_| StateSpaceError::SingularLoopMatrix)?;

        // u1 = F^{-1} (u + sign D2 C1 x1 + sign C2 x2)
        let b1e = &b1 * &f_inv;
        let d1e = &d1 * &f_inv;
        let b2d1e = &b2 * &d1e;

        let d2c1 = &d2 * &c1;
        let sign_d2c1 = &d2c1 * sign;
        let sign_c2 = &c2 * sign;
        let a1_corr = &b1e * &sign_d2c1;
        let a12 = &b1e * &sign_c2;
        let a21_corr = &b2d1e * &sign_d2c1;
        let a22_corr = &b2d1e * &sign_c2;
        let b2c1 = &b2 * &c1;

        let a11 = &a1 + &a1_corr;
        let a21 = &b2c1 + &a21_corr;
        let a22 = &a2 + &a22_corr;

        let mut a = Owned::<T, NXOUT, NXOUT>::zero();
        let mut b = Owned::<T, NXOUT, NU>::zero();
        let mut c = Owned::<T, NY, NXOUT>::zero();
        a.write_block(0, 0, &a11);
        a.write_block(0, NX1, &a12);
        a.write_block(NX1, 0, &a21);
        a.write_block(NX1, NX1, &a22);
        b.write_block(0, 0, &b1e);
        b.write_block(NX1, 0, &b2d1e);

        let d1e_d2c1 = &d1e * &sign_d2c1;
        let c11 = &c1 + &d1e_d2c1;
        let c12 = &d1e * &sign_c2;
        c.write_block(0, 0, &c11);
        c.write_block(0, NX1, &c12);

        let sys = match (self.sample_time, rhs.sample_time) {
            (Some(dt), Some(_)) => StateSpace::discrete(a, b, c, d1e, dt),
            _ => StateSpace::continuous(a, b, c, d1e),
        };
        Ok(sys)
    }
}

////////////////////////////////////////////////////////////////////////////////
// Discretization & Similarity Transform
////////////////////////////////////////////////////////////////////////////////

impl<T: Float + Copy, const NX: usize, const NU: usize, const NY: usize>
    StateSpace<T, NX, NU, NY>
where
    Const<NX>: Dim,
    Const<NU>: Dim,
    Const<NY>: Dim,
{
    /// Zero-order hold via matrix exponential and series integration.
    #[must_use]
    pub fn to_discrete_zoh(&self, dt: T) -> Self {
        let a = self.a();
        let b = self.b();
        let adt = &a * dt;
        let ad = adt.expm();

        // V = \sum_{k=0}^{19} (A dt)^k / (k+1)! via backward Horner evaluation
        let mut v = Owned::<T, NX, NX>::zero();
        for k in (0..20_usize).rev() {
            let denom = {
                let mut d = T::ZERO;
                for _ in 0..=k {
                    d = d + T::ONE;
                }
                d
            };
            let inv_denom = T::ONE / denom;
            let term = &adt * &v;
            let mut next_v = Owned::<T, NX, NX>::identity();
            for i in 0..NX {
                for j in 0..NX {
                    if let (Some(dst), Some(&src)) =
                        (next_v.get_mut(i, j), term.get(i, j))
                    {
                        *dst = (*dst + src) * inv_denom;
                    }
                }
            }
            v = next_v;
        }

        let bdt = &b * dt;
        let bd = &v * &bdt;
        Self::discrete(ad, bd, self.c(), self.d(), dt)
    }

    /// Bilinear (Tustin) discretization.
    ///
    /// Solves against $M = I - \frac{T_s}{2} A$ and returns
    /// $$
    /// A_d = M^{-1}(I + \tfrac{T_s}{2} A),\quad
    /// B_d = M^{-1} B T_s,\quad
    /// C_d = C M^{-1},\quad
    /// D_d = D + C_d B \tfrac{T_s}{2}.
    /// $$
    ///
    /// # Errors
    ///
    /// Returns [`StateSpaceError::SingularDiscretizationOperator`] when $M$ is
    /// singular to working precision.
    pub fn to_discrete_tustin(&self, dt: T) -> StateSpaceResult<Self> {
        let two = T::ONE + T::ONE;
        let h = dt / two;
        let a = self.a();
        let b = self.b();
        let mut i_minus = Owned::<T, NX, NX>::identity();
        let mut i_plus = Owned::<T, NX, NX>::identity();
        for i in 0..NX {
            for j in 0..NX {
                if let Some(&aij) = a.get(i, j) {
                    if let Some(t) = i_minus.get_mut(i, j) {
                        *t = *t - h * aij;
                    }
                    if let Some(t) = i_plus.get_mut(i, j) {
                        *t = *t + h * aij;
                    }
                }
            }
        }
        let lu = LuDecomposition::decompose(i_minus)
            .map_err(|_| StateSpaceError::SingularDiscretizationOperator)?;
        let m_inv = lu
            .inverse()
            .map_err(|_| StateSpaceError::SingularDiscretizationOperator)?;
        let ad = &m_inv * &i_plus;
        let bd = &m_inv * &(&b * dt);
        let cd = &self.c() * &m_inv;
        let dd = &self.d() + &(&(&cd * &b) * h);
        Ok(Self::discrete(ad, bd, cd, dd, dt))
    }

    /// Performs similarity coordinate transformation $z = T x$:
    ///
    /// $$\tilde{A} = T A T^{-1}, \quad \tilde{B} = T B, \quad \tilde{C} = C T^{-1}, \quad \tilde{D} = D$$
    pub fn similarity_transform(
        &self,
        t: &Owned<T, NX, NX>,
    ) -> LinAlgResult<Self> {
        let lu = LuDecomposition::decompose(*t)?;
        let t_inv = lu.inverse()?;
        let a_tilde = &(t * &self.a()) * &t_inv;
        let b_tilde = t * &self.b();
        let c_tilde = &self.c() * &t_inv;
        Ok(Self::from_storage(
            a_tilde.into_storage(),
            b_tilde.into_storage(),
            c_tilde.into_storage(),
            self.d().into_storage(),
            self.sample_time,
        ))
    }
}

////////////////////////////////////////////////////////////////////////////////
// Controllability, Observability, Transfer Function
////////////////////////////////////////////////////////////////////////////////

impl<T: Scalar + Copy, const NX: usize, const NU: usize, const NY: usize>
    StateSpace<T, NX, NU, NY>
where
    Const<NX>: Dim,
    Const<NU>: Dim,
    Const<NY>: Dim,
{
    /// Controllability matrix $[B, AB, \dots, A^{n-1}B]$.
    #[must_use]
    pub fn controllability_matrix<const NC: usize>(&self) -> Owned<T, NX, NC>
    where
        Const<NC>: Dim,
    {
        let a = self.a();
        let mut block = self.b();
        let mut ctrb = Owned::<T, NX, NC>::zero();
        for k in 0..NX {
            ctrb.write_block(0, k * NU, &block);
            block = &a * &block;
        }
        ctrb
    }

    /// Observability matrix $[C; CA; \dots; CA^{n-1}]$.
    #[must_use]
    pub fn observability_matrix<const NR: usize>(&self) -> Owned<T, NR, NX>
    where
        Const<NR>: Dim,
    {
        let a = self.a();
        let mut block = self.c();
        let mut obsv = Owned::<T, NR, NX>::zero();
        for k in 0..NX {
            obsv.write_block(k * NY, 0, &block);
            block = &block * &a;
        }
        obsv
    }
}

impl<T: Float + Copy, const NX: usize> StateSpace<T, NX, 1, 1>
where
    Const<NX>: Dim,
{
    /// SISO $H(s) = C(sI-A)^{-1}B + D$ via Faddeev–LeVerrier.
    #[must_use]
    pub fn to_transfer_function<const NP: usize>(
        &self,
    ) -> ArrayTransferFunction<T, NP, NP>
    where
        Const<NP>: Dim,
    {
        let a = self.a();
        let mut char_c = [T::ZERO; NX];
        let mut ak = a;
        char_c[0] = T::ZERO - ak.trace();
        for k in 2..=NX {
            let mut tmp = ak;
            add_identity_scaled(&mut tmp, char_c[k - 2]);
            ak = &a * &tmp;
            let kk = {
                let mut s = T::ZERO;
                for _ in 0..k {
                    s = s + T::ONE;
                }
                s
            };
            char_c[k - 1] = (T::ZERO - ak.trace()) / kk;
        }

        let mut den = [T::ZERO; NP];
        if NP > NX {
            den[NX] = T::ONE;
            for i in 0..NX {
                den[i] = char_c[NX - 1 - i];
            }
        }

        let mut bk = Owned::<T, NX, NX>::identity();
        let mut num = [T::ZERO; NP];
        let b = self.b();
        let c = self.c();
        let d = self.d().get(0, 0).copied().unwrap_or(T::ZERO);
        for k in 0..NX {
            let cb = &c * &(&bk * &b);
            let scale = cb.get(0, 0).copied().unwrap_or(T::ZERO);
            if NX - 1 - k < NP {
                num[NX - 1 - k] = num[NX - 1 - k] + scale;
            }
            if k + 1 < NX {
                bk = &a * &bk;
                add_identity_scaled(&mut bk, char_c[k]);
            }
        }
        for i in 0..NP {
            if i < NX + 1 {
                num[i] = num[i] + d * den[i];
            }
        }

        match self.sample_time {
            None => ArrayTransferFunction::continuous(num, den),
            Some(dt) => ArrayTransferFunction::discrete(num, den, dt),
        }
    }
}

fn add_identity_scaled<T: Scalar + Copy, const N: usize>(
    m: &mut Owned<T, N, N>,
    s: T,
) where
    Const<N>: Dim,
{
    for i in 0..N {
        if let Some(v) = m.get_mut(i, i) {
            *v = *v + s;
        }
    }
}
