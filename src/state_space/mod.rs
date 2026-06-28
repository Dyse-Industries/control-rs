//! # State-Space
//!
//! "The idea of **state-space** comes from the state-variable method of describing differential
//! equations. In this method, the differential equations describing a dynamic system are
//! organized as a set of first order differential equations in the vector-valued state of
//! the system. The solution is visualized as a trajectory of this state vector in space.
//! **state-space control design** is the technique in which the control engineer designs a
//! dynamic compensation by working directly with the state-variable description of the system"
//! - 'Feedback Control of Dynamic Systems' by Gene F. Franklin, J. David Powell and Abbas
//!   Emami-Naeini (ch 7.1)

use crate::math::StateEquation;
use crate::math::num_traits::Zero;
use crate::math::ops::{Add, Mul};

#[cfg(any(test, feature = "hil"))]
pub mod test;

/// A linear time-invariant (LTI) state-space system represented by statically sized arrays.
///
/// Mathematically described as:
///   dx/dt = A * x + B * u
///   y     = C * x + D * u
///
/// # Generic Arguments
/// * `T` - Element type (must implement arithmetic traits).
/// * `S` - State vector dimension.
/// * `I` - Input vector dimension.
/// * `O` - Output vector dimension.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[allow(clippy::derive_partial_eq_without_eq, clippy::type_complexity)]
pub struct StateSpace<T, const S: usize, const I: usize, const O: usize> {
    /// State transition matrix A (S x S)
    pub a: [[T; S]; S],
    /// Input matrix B (S x I)
    pub b: [[T; I]; S],
    /// Output matrix C (O x S)
    pub c: [[T; S]; O],
    /// Feedthrough matrix D (O x I)
    pub d: [[T; I]; O],
}

impl<T, const S: usize, const I: usize, const O: usize> StateSpace<T, S, I, O>
where
    T: Copy + Zero + Add<Output = T> + Mul<Output = T>,
{
    /// Creates a new `StateSpace` model.
    #[allow(clippy::type_complexity)]
    pub const fn new(
        a: [[T; S]; S],
        b: [[T; I]; S],
        c: [[T; S]; O],
        d: [[T; I]; O],
    ) -> Self {
        Self { a, b, c, d }
    }

    /// Evaluates the system's output equation: `y = C * x + D * u`.
    #[allow(
        clippy::needless_range_loop,
        clippy::indexing_slicing,
        clippy::arithmetic_side_effects
    )]
    pub fn output(&self, x: &[T; S], u: &[T; I]) -> [T; O] {
        let mut y = [T::ZERO; O];
        for i in 0..O {
            let mut sum_c = T::ZERO;
            for j in 0..S {
                sum_c = sum_c + self.c[i][j] * x[j];
            }
            let mut sum_d = T::ZERO;
            for j in 0..I {
                sum_d = sum_d + self.d[i][j] * u[j];
            }
            y[i] = sum_c + sum_d;
        }
        y
    }
}

impl<T, const S: usize, const I: usize, const O: usize>
    StateEquation<[T; S], [T; I], [T; S]> for StateSpace<T, S, I, O>
where
    T: Copy + Zero + Add<Output = T> + Mul<Output = T>,
{
    #[allow(
        clippy::needless_range_loop,
        clippy::indexing_slicing,
        clippy::arithmetic_side_effects
    )]
    fn dynamics(&self, x: &[T; S], u: &[T; I]) -> [T; S] {
        let mut dx = [T::ZERO; S];
        for i in 0..S {
            let mut sum_a = T::ZERO;
            for j in 0..S {
                sum_a = sum_a + self.a[i][j] * x[j];
            }
            let mut sum_b = T::ZERO;
            for j in 0..I {
                sum_b = sum_b + self.b[i][j] * u[j];
            }
            dx[i] = sum_a + sum_b;
        }
        dx
    }
}
