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
//! let mut x = Owned::<f64, 1, 1>::zero();
//! let u = Owned::<f64, 1, 1>::from_fn(|_, _| 1.0);
//!
//! let y = sys.step(&mut x, &u);
//! assert_eq!(y.get(0, 0), Some(&0.0)); // y[0] = C*x[0] = 0
//! assert_eq!(x.get(0, 0), Some(&1.0)); // x[1] = A*0 + B*1 = 1
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
    ArrayStorage, Storage, StorageView, StorageViewMut,
};
use crate::matrix::Owned;
use crate::matrix::decomposition::LuDecomposition;
use core::marker::PhantomData;

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

impl<T, const NX: usize, const NU: usize, const NY: usize>
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
        Self {
            a_storage: a.into_storage(),
            b_storage: b.into_storage(),
            c_storage: c.into_storage(),
            d_storage: d.into_storage(),
            sample_time: None,
            _marker: PhantomData,
        }
    }

    /// Builds a discrete-time state-space model ($z$-domain) with sampling interval `dt`.
    pub fn discrete(
        a: Owned<T, NX, NX>,
        b: Owned<T, NX, NU>,
        c: Owned<T, NY, NX>,
        d: Owned<T, NY, NU>,
        dt: T,
    ) -> Self {
        Self {
            a_storage: a.into_storage(),
            b_storage: b.into_storage(),
            c_storage: c.into_storage(),
            d_storage: d.into_storage(),
            sample_time: Some(dt),
            _marker: PhantomData,
        }
    }

    /// Exposes matrix $A \in \mathbb{R}^{N_x \times N_x}$.
    #[must_use]
    pub const fn a(&self) -> &Owned<T, NX, NX> {
        unsafe { &*(&raw const self.a_storage).cast::<Owned<T, NX, NX>>() }
    }

    /// Exposes matrix $B \in \mathbb{R}^{N_x \times N_u}$.
    #[must_use]
    pub const fn b(&self) -> &Owned<T, NX, NU> {
        unsafe { &*(&raw const self.b_storage).cast::<Owned<T, NX, NU>>() }
    }

    /// Exposes matrix $C \in \mathbb{R}^{N_y \times N_x}$.
    #[must_use]
    pub const fn c(&self) -> &Owned<T, NY, NX> {
        unsafe { &*(&raw const self.c_storage).cast::<Owned<T, NY, NX>>() }
    }

    /// Exposes feedforward matrix $D \in \mathbb{R}^{N_y \times N_u}$.
    #[must_use]
    pub const fn d(&self) -> &Owned<T, NY, NU> {
        unsafe { &*(&raw const self.d_storage).cast::<Owned<T, NY, NU>>() }
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
    /// Advances discrete state-space dynamics by one sample time step:
    ///
    /// $$y[k] = C x[k] + D u[k]$$
    /// $$x[k+1] = A x[k] + B u[k]$$
    ///
    /// Mutates `x` in place and returns measurement output `y`.
    pub fn step(
        &self,
        x: &mut Owned<T, NX, 1>,
        u: &Owned<T, NU, 1>,
    ) -> Owned<T, NY, 1> {
        let mut y = Owned::<T, NY, 1>::zero();
        let a = self.a();
        let b = self.b();
        let c = self.c();
        let d = self.d();

        // y = C*x + D*u
        let cx = c * &(*x);
        let du = d * u;
        for i in 0..NY {
            if let (Some(target), Some(&v1), Some(&v2)) =
                (y.get_mut(i, 0), cx.get(i, 0), du.get(i, 0))
            {
                *target = v1 + v2;
            }
        }

        // x_next = A*x + B*u
        let ax = a * &(*x);
        let bu = b * u;
        for i in 0..NX {
            if let (Some(target), Some(&v1), Some(&v2)) =
                (x.get_mut(i, 0), ax.get(i, 0), bu.get(i, 0))
            {
                *target = v1 + v2;
            }
        }

        y
    }

    /// Computes continuous-time state derivatives and output:
    ///
    /// $$\dot{x}(t) = A x(t) + B u(t)$$
    /// $$y(t) = C x(t) + D u(t)$$
    pub fn derivative(
        &self,
        x: &Owned<T, NX, 1>,
        u: &Owned<T, NU, 1>,
    ) -> (Owned<T, NX, 1>, Owned<T, NY, 1>) {
        let mut x_dot = Owned::<T, NX, 1>::zero();
        let mut y = Owned::<T, NY, 1>::zero();
        let a = self.a();
        let b = self.b();
        let c = self.c();
        let d = self.d();

        let ax = a * x;
        let bu = b * u;
        for i in 0..NX {
            if let (Some(target), Some(&v1), Some(&v2)) =
                (x_dot.get_mut(i, 0), ax.get(i, 0), bu.get(i, 0))
            {
                *target = v1 + v2;
            }
        }

        let cx = c * x;
        let du = d * u;
        for i in 0..NY {
            if let (Some(target), Some(&v1), Some(&v2)) =
                (y.get_mut(i, 0), cx.get(i, 0), du.get(i, 0))
            {
                *target = v1 + v2;
            }
        }

        (x_dot, y)
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
    /// Discretizes a continuous-time system using Zero-Order Hold (ZOH) with sampling period `dt`.
    ///
    /// Evaluates $A_d = e^{A \cdot dt}$ and $B_d = \int_0^{dt} e^{A \tau} d\tau \cdot B$
    /// via matrix series expansion.
    #[must_use]
    pub fn to_discrete_zoh(&self, dt: T) -> Self {
        // High-precision Taylor series approximation for matrix exponential: e^{A*dt} = I + A*dt + (A*dt)^2/2! + ...
        let a = self.a();
        let b = self.b();

        let mut a_dt = Owned::<T, NX, NX>::zero();
        for i in 0..NX {
            for j in 0..NX {
                if let (Some(target), Some(&v)) =
                    (a_dt.get_mut(i, j), a.get(i, j))
                {
                    *target = v * dt;
                }
            }
        }

        let mut ad = Owned::<T, NX, NX>::identity();
        let mut term = Owned::<T, NX, NX>::identity();
        let mut integral = Owned::<T, NX, NX>::zero();

        // Add I * dt to integral
        for i in 0..NX {
            if let Some(target) = integral.get_mut(i, i) {
                *target = dt;
            }
        }

        let mut factorial = T::ONE;
        for k in 1..=12 {
            factorial = factorial * {
                let mut fk = T::ZERO;
                for _ in 0..k {
                    fk = fk + T::ONE;
                }
                fk
            };
            term = &term * &a_dt;

            // ad += term / k!
            for i in 0..NX {
                for j in 0..NX {
                    if let (Some(target), Some(&v)) =
                        (ad.get_mut(i, j), term.get(i, j))
                    {
                        *target = *target + v / factorial;
                    }
                }
            }

            // integral += (A^k * dt^{k+1}) / (k+1)!
            let next_fact = factorial * {
                let mut fk1 = T::ZERO;
                for _ in 0..=(k) {
                    fk1 = fk1 + T::ONE;
                }
                fk1
            };
            for i in 0..NX {
                for j in 0..NX {
                    if let (Some(target), Some(&v)) =
                        (integral.get_mut(i, j), term.get(i, j))
                    {
                        *target = *target + (v * dt) / next_fact;
                    }
                }
            }
        }

        let bd = &integral * b;
        Self::discrete(ad, bd, *self.c(), *self.d(), dt)
    }

    /// Performs similarity coordinate transformation $z = T x$:
    ///
    /// $$\tilde{A} = T A T^{-1}, \quad \tilde{B} = T B, \quad \tilde{C} = C T^{-1}, \quad \tilde{D} = D$$
    ///
    /// # Errors
    /// Returns [`LinAlgError::SingularMatrix`] if transformation matrix $T$ is singular.
    pub fn similarity_transform(
        &self,
        t: &Owned<T, NX, NX>,
    ) -> LinAlgResult<Self> {
        let lu = LuDecomposition::decompose(*t)?;
        let t_inv = lu.inverse()?;

        let a_tilde = &(t * self.a()) * &t_inv;
        let b_tilde = t * self.b();
        let c_tilde = self.c() * &t_inv;
        let d_tilde = *self.d();

        Ok(Self {
            a_storage: a_tilde.into_storage(),
            b_storage: b_tilde.into_storage(),
            c_storage: c_tilde.into_storage(),
            d_storage: d_tilde.into_storage(),
            sample_time: self.sample_time,
            _marker: PhantomData,
        })
    }
}
