//! # State-Space Unit and Invariant Tests
#![allow(
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::similar_names,
    clippy::unwrap_used,
    clippy::items_after_statements,
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::approx_constant
)]

#[cfg_attr(not(test), control_rs_macros::ets_suite)]
pub mod state_space_test_suite {
    use crate::assert_almost_eq;
    use crate::matrix::Owned;
    use crate::state_space::ArrayStateSpace;

    #[cfg_attr(test, test)]
    fn test_discrete_simulation_step() {
        // Scalar system: x[k+1] = 0.5 x[k] + 1.0 u[k], y[k] = 2.0 x[k]
        let a = Owned::<f64, 1, 1>::from_fn(|_, _| 0.5);
        let b = Owned::<f64, 1, 1>::from_fn(|_, _| 1.0);
        let c = Owned::<f64, 1, 1>::from_fn(|_, _| 2.0);
        let d = Owned::<f64, 1, 1>::from_fn(|_, _| 0.0);

        let sys = ArrayStateSpace::discrete(a, b, c, d, 0.01);
        let mut x = Owned::<f64, 1, 1>::zero();
        let u = Owned::<f64, 1, 1>::from_fn(|_, _| 1.0);

        // Step 1: x_0 = 0 -> y_0 = 0, x_1 = 1.0
        let y0 = sys.step(&mut x, &u);
        assert_almost_eq!(y0.get(0, 0).copied().unwrap(), 0.0, 1e-12);
        assert_almost_eq!(x.get(0, 0).copied().unwrap(), 1.0, 1e-12);

        // Step 2: x_1 = 1.0 -> y_1 = 2.0, x_2 = 0.5 * 1.0 + 1.0 = 1.5
        let y1 = sys.step(&mut x, &u);
        assert_almost_eq!(y1.get(0, 0).copied().unwrap(), 2.0, 1e-12);
        assert_almost_eq!(x.get(0, 0).copied().unwrap(), 1.5, 1e-12);
    }

    #[cfg_attr(test, test)]
    fn test_continuous_derivative() {
        // Mass-spring system: \dot{x} = [[0, 1], [-k/m, -c/m]] x + [[0], [1/m]] u
        let a = Owned::<f64, 2, 2>::from_array([[0.0, -2.0], [1.0, -1.0]]);
        let b = Owned::<f64, 2, 1>::from_array([[0.0, 1.0]]);
        let c = Owned::<f64, 1, 2>::from_array([[1.0], [0.0]]);
        let d = Owned::<f64, 1, 1>::zero();

        let sys = ArrayStateSpace::continuous(a, b, c, d);
        let x = Owned::<f64, 2, 1>::from_array([[1.0, 0.0]]); // pos = 1, vel = 0
        let u = Owned::<f64, 1, 1>::zero();

        let (x_dot, y) = sys.derivative(&x, &u);
        assert_almost_eq!(x_dot.get(0, 0).copied().unwrap(), 0.0, 1e-12); // vel = 0
        assert_almost_eq!(x_dot.get(1, 0).copied().unwrap(), -2.0, 1e-12); // accel = -k*x = -2
        assert_almost_eq!(y.get(0, 0).copied().unwrap(), 1.0, 1e-12);
    }

    #[cfg_attr(test, test)]
    fn test_zoh_discretization() {
        // Pure integrator: \dot{x} = u, y = x
        let a = Owned::<f64, 1, 1>::zero();
        let b = Owned::<f64, 1, 1>::from_fn(|_, _| 1.0);
        let c = Owned::<f64, 1, 1>::from_fn(|_, _| 1.0);
        let d = Owned::<f64, 1, 1>::zero();

        let sys = ArrayStateSpace::continuous(a, b, c, d);
        let sys_d = sys.to_discrete_zoh(0.1);

        // Ad = e^{0*0.1} = 1.0, Bd = \int_0^{0.1} 1 dt = 0.1
        assert_almost_eq!(sys_d.a().get(0, 0).copied().unwrap(), 1.0, 1e-6);
        assert_almost_eq!(sys_d.b().get(0, 0).copied().unwrap(), 0.1, 1e-6);
    }
}
