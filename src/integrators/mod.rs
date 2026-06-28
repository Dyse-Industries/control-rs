//! A variety of methods to integrate numerical models

use crate::math::StateEquation;
use crate::math::num_traits::Real;
use crate::math::ops::{Add, Mul};
use crate::math::subprograms::level1::AXPY;

#[cfg(any(test, feature = "hil"))]
pub mod test;

/// Euler numerical integrator.
pub struct Euler;

/// Runge-Kutta 4th Order (RK4) numerical integrator.
pub struct Rk4;

impl Euler {
    /// Integrates the state forward by one step using the Euler method: `x = x + dt * dx`.
    ///
    /// # Generic Arguments
    /// * `T` - Numeric type of the state elements.
    /// * `N` - Dimension of the state vector.
    /// * `Eq` - The dynamics equation, implementing [`StateEquation`].
    /// * `Input` - The control input type.
    /// * `B` - The BLAS subprograms backend implementing [`AXPY`].
    #[allow(clippy::arithmetic_side_effects)]
    pub fn step<T, const N: usize, Eq, Input, B>(
        eq: &Eq,
        x: &mut [T; N],
        u: &Input,
        dt: T,
    ) where
        T: Copy
            + crate::math::num_traits::Scalar
            + Add<Output = T>
            + Mul<Output = T>,
        Eq: StateEquation<[T; N], Input, [T; N]>,
        B: AXPY<T>,
    {
        let dx = eq.dynamics(x, u);
        B::axpy(dt, &dx, x);
    }
}

impl Rk4 {
    /// Integrates the state forward by one step using the 4th-order Runge-Kutta method.
    ///
    /// # Generic Arguments
    /// * `T` - Numeric type of the state elements.
    /// * `N` - Dimension of the state vector.
    /// * `Eq` - The dynamics equation, implementing [`StateEquation`].
    /// * `Input` - The control input type.
    /// * `B` - The BLAS subprograms backend implementing [`AXPY`].
    #[allow(clippy::arithmetic_side_effects)]
    pub fn step<T, const N: usize, Eq, Input, B>(
        eq: &Eq,
        x: &mut [T; N],
        u: &Input,
        dt: T,
    ) where
        T: Real,
        Eq: StateEquation<[T; N], Input, [T; N]>,
        B: AXPY<T>,
    {
        let two = T::TWO;
        let three = T::from_const::<3>();
        let six = T::from_const::<6>();

        let dt_half = dt.clone() / two;

        // k1 = f(x, u)
        let k1 = eq.dynamics(x, u);

        // x_temp = x + 0.5 * dt * k1
        let mut x_temp = x.clone();
        B::axpy(dt_half.clone(), &k1, &mut x_temp);

        // k2 = f(x_temp, u)
        let k2 = eq.dynamics(&x_temp, u);

        // x_temp = x + 0.5 * dt * k2
        x_temp.clone_from(x);
        B::axpy(dt_half, &k2, &mut x_temp);

        // k3 = f(x_temp, u)
        let k3 = eq.dynamics(&x_temp, u);

        // x_temp = x + dt * k3
        x_temp.clone_from(x);
        B::axpy(dt.clone(), &k3, &mut x_temp);

        // k4 = f(x_temp, u)
        let k4 = eq.dynamics(&x_temp, u);

        // x = x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        let w1 = dt.clone() / six.clone();
        let w2 = dt.clone() / three;
        let w4 = dt / six;

        B::axpy(w1, &k1, x);
        B::axpy(w2.clone(), &k2, x);
        B::axpy(w2, &k3, x);
        B::axpy(w4, &k4, x);
    }
}
