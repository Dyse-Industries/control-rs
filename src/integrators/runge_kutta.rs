//! A variety of methods to integrate systems of differential equations.

use crate::math::{
    num_traits::Field,
    ops::{Add, Mul},
    Map,
};

/// Integrate the system for a given time interval
///
/// The time interval is assumed to be small enough the input will be constant for the
/// duration of the integration. For simulations with time-varying input, call this repeatedly
/// in a loop.
///
/// # Arguments
///
/// * `x0` - initial state
/// * `u` - input
/// * `t0` - start time
/// * `tf` - end time
/// * `dt` - length of a step
///
/// # Returns
///
/// * `x` - state at end time
pub fn runge_kutta4<T, X, Dx, Eq>(model: &Eq, x0: X, t0: T, tf: T, dt: T) -> X
where
    X: Copy + Add<Output = X> + Add<Dx, Output = X> + Mul<T, Output = X>,
    Dx: Copy + Add<Output = Dx> + Mul<T, Output = Dx>,
    T: Copy + Field,
    Eq: Map<X, Dx>,
{
    let dt_2 = dt / T::TWO;
    let dt_6 = dt / (T::TWO * (T::ONE + T::TWO));

    let mut t = t0;
    let mut x = x0;

    while t < tf {
        let k1 = model.evaluate(x);
        let k2 = model.evaluate(x + k1 * dt_2);
        let k3 = model.evaluate(x + k2 * dt_2);
        let k4 = model.evaluate(x + k3 * dt);
        x = x + (k1 + k2 * T::TWO + k3 * T::TWO + k4) * dt_6;
        t = t + dt;
    }
    x
}