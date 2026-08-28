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
    use crate::math::storage::DenseStorage;
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
        let x0 = Owned::<f64, 1, 1>::zero();
        let u = Owned::<f64, 1, 1>::from_fn(|_, _| 1.0);

        // Step 1: x_0 = 0 -> y_0 = 0, x_1 = 1.0
        let (x1, y0) = sys.step(&x0, &u);
        assert_almost_eq!(y0.get(0, 0).copied().unwrap(), 0.0, 1e-12);
        assert_almost_eq!(x1.get(0, 0).copied().unwrap(), 1.0, 1e-12);

        // Step 2: x_1 = 1.0 -> y_1 = 2.0, x_2 = 0.5 * 1.0 + 1.0 = 1.5
        let (x2, y1) = sys.step(&x1, &u);
        assert_almost_eq!(y1.get(0, 0).copied().unwrap(), 2.0, 1e-12);
        assert_almost_eq!(x2.get(0, 0).copied().unwrap(), 1.5, 1e-12);
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

    #[cfg_attr(test, test)]
    fn test_zoh_vs_scalar_exp() {
        let a = Owned::<f64, 1, 1>::from_fn(|_, _| -2.0);
        let b = Owned::<f64, 1, 1>::from_fn(|_, _| 1.0);
        let c = Owned::<f64, 1, 1>::from_fn(|_, _| 1.0);
        let d = Owned::<f64, 1, 1>::zero();
        let sys = ArrayStateSpace::continuous(a, b, c, d);
        let dt = 0.05;
        let sys_d = sys.to_discrete_zoh(dt);
        assert_almost_eq!(
            sys_d.a().get(0, 0).copied().unwrap(),
            0.904_837_418_035_959_5, // exp(-2*0.05)
            1e-10
        );
    }

    #[cfg_attr(test, test)]
    fn test_tustin_scalar() {
        let a = Owned::<f64, 1, 1>::from_fn(|_, _| -2.0);
        let b = Owned::<f64, 1, 1>::from_fn(|_, _| 1.0);
        let c = Owned::<f64, 1, 1>::from_fn(|_, _| 1.0);
        let d = Owned::<f64, 1, 1>::zero();
        let sys = ArrayStateSpace::continuous(a, b, c, d);
        let dt = 0.1;
        let sys_d = sys.to_discrete_tustin(dt).unwrap();
        let h = dt / 2.0;
        let expected_a = (1.0 + h * -2.0) / (1.0 - h * -2.0);
        let m_inv = 1.0 / (1.0 - h * -2.0);
        let expected_b = m_inv * 1.0 * dt;
        let expected_c = 1.0 * m_inv;
        let expected_d = 0.0 + expected_c * 1.0 * h;
        assert_almost_eq!(
            sys_d.a().get(0, 0).copied().unwrap(),
            expected_a,
            1e-12
        );
        assert_almost_eq!(
            sys_d.b().get(0, 0).copied().unwrap(),
            expected_b,
            1e-12
        );
        assert_almost_eq!(
            sys_d.c().get(0, 0).copied().unwrap(),
            expected_c,
            1e-12
        );
        assert_almost_eq!(
            sys_d.d().get(0, 0).copied().unwrap(),
            expected_d,
            1e-12
        );
    }

    #[cfg_attr(test, test)]
    fn test_series_parallel_feedback() {
        let a = Owned::<f64, 1, 1>::from_fn(|_, _| -1.0);
        let b = Owned::<f64, 1, 1>::from_fn(|_, _| 1.0);
        let c = Owned::<f64, 1, 1>::from_fn(|_, _| 1.0);
        let d = Owned::<f64, 1, 1>::zero();
        let g = ArrayStateSpace::continuous(a, b, c, d);
        let h = ArrayStateSpace::continuous(a, b, c, d);

        let ser = g.series::<1, 1, 2>(&h);
        assert_eq!(ser.a().rows(), 2);

        let par = g.parallel::<1, 2>(&h);
        assert_almost_eq!(par.d().get(0, 0).copied().unwrap(), 0.0, 1e-12);

        let cl = g.feedback::<1, 2>(&h, -1.0).unwrap();
        assert_eq!(cl.a().rows(), 2);

        let ident_fb = ArrayStateSpace::continuous(
            Owned::<f64, 1, 1>::zero(),
            Owned::<f64, 1, 1>::zero(),
            Owned::<f64, 1, 1>::zero(),
            Owned::<f64, 1, 1>::from_fn(|_, _| 1.0),
        );
        let _ = g.feedback::<1, 2>(&ident_fb, -1.0).unwrap();
    }

    #[cfg_attr(test, test)]
    fn test_ctrb_obsv_tf() {
        let a = Owned::<f64, 1, 1>::from_fn(|_, _| -2.0);
        let b = Owned::<f64, 1, 1>::from_fn(|_, _| 3.0);
        let c = Owned::<f64, 1, 1>::from_fn(|_, _| 4.0);
        let d = Owned::<f64, 1, 1>::from_fn(|_, _| 1.0);
        let sys = ArrayStateSpace::continuous(a, b, c, d);
        let ctrb = sys.controllability_matrix::<1>();
        assert_almost_eq!(ctrb.get(0, 0).copied().unwrap(), 3.0, 1e-12);
        let obsv = sys.observability_matrix::<1>();
        assert_almost_eq!(obsv.get(0, 0).copied().unwrap(), 4.0, 1e-12);

        let tf = sys.to_transfer_function::<2>();
        // H(s) = 1 + 12/(s+2) = (s+14)/(s+2)
        assert_almost_eq!(tf.den_slice()[0], 2.0, 1e-9);
        assert_almost_eq!(tf.den_slice()[1], 1.0, 1e-9);
        assert_almost_eq!(tf.num_slice()[1], 1.0, 1e-9);
        assert_almost_eq!(tf.num_slice()[0], 14.0, 1e-9);
    }

    #[cfg_attr(test, test)]
    fn test_state_space_display_tustin_and_faddeev() {
        use crate::state_space::StateSpaceError;
        use core::fmt::Write;

        struct StackBuf([u8; 192], usize);
        impl Write for StackBuf {
            fn write_str(&mut self, s: &str) -> core::fmt::Result {
                let rest = self.0.len().saturating_sub(self.1);
                let n = rest.min(s.len());
                self.0[self.1..self.1 + n].copy_from_slice(&s.as_bytes()[..n]);
                self.1 += n;
                Ok(())
            }
        }
        let mut loop_buf = StackBuf([0u8; 192], 0);
        write!(&mut loop_buf, "{}", StateSpaceError::SingularLoopMatrix)
            .unwrap();
        let loop_msg = core::str::from_utf8(&loop_buf.0[..loop_buf.1]).unwrap();
        assert!(loop_msg.contains("singular"));
        let mut tustin_buf = StackBuf([0u8; 192], 0);
        write!(
            &mut tustin_buf,
            "{}",
            StateSpaceError::SingularDiscretizationOperator
        )
        .unwrap();
        let tustin_msg =
            core::str::from_utf8(&tustin_buf.0[..tustin_buf.1]).unwrap();
        assert!(tustin_msg.contains("Tustin"));

        let a = Owned::<f64, 2, 2>::from_fn(|i, j| {
            [[0.0, 1.0], [-2.0, -3.0]][i][j]
        });
        let b = Owned::<f64, 2, 1>::from_fn(|i, _| [0.0, 1.0][i]);
        let c = Owned::<f64, 1, 2>::from_fn(|_, j| [1.0, 0.0][j]);
        let d = Owned::<f64, 1, 1>::zero();
        let sys = ArrayStateSpace::continuous(a, b, c, d);
        assert!(sys.is_continuous());
        assert!(!sys.is_discrete());
        assert!(sys.sample_time().is_none());
        let tf = sys.to_transfer_function::<3>();
        assert_almost_eq!(tf.den_slice()[2], 1.0, 1e-9);

        let disc = ArrayStateSpace::discrete(a, b, c, d, 0.1);
        assert!(disc.is_discrete());
        assert_eq!(disc.sample_time(), Some(0.1));

        let bad_a =
            Owned::<f64, 2, 2>::from_fn(|i, j| [[2.0, 0.0], [0.0, 0.0]][i][j]);
        let bad = ArrayStateSpace::continuous(bad_a, b, c, d);
        assert_eq!(
            bad.to_discrete_tustin(1.0),
            Err(StateSpaceError::SingularDiscretizationOperator)
        );
    }

    #[cfg_attr(test, test)]
    fn test_state_space_view() {
        let a = Owned::<f64, 1, 1>::from_fn(|_, _| 0.5);
        let b = Owned::<f64, 1, 1>::from_fn(|_, _| 1.0);
        let c = Owned::<f64, 1, 1>::from_fn(|_, _| 2.0);
        let d = Owned::<f64, 1, 1>::from_fn(|_, _| 0.0);
        let mut sys = ArrayStateSpace::discrete(a, b, c, d, 0.01);
        {
            let view = sys.view();
            assert_eq!(view.a_storage().get(0, 0), Some(&0.5));
        }
        assert_eq!(sys.a_matrix().get(0, 0), Some(&0.5));
        let x0 = Owned::<f64, 1, 1>::zero();
        let u = Owned::<f64, 1, 1>::from_fn(|_, _| 1.0);
        let (x_next, y) = sys.step(&x0, &u);
        assert_almost_eq!(y.get(0, 0).copied().unwrap(), 0.0, 1e-12);
        assert_almost_eq!(x_next.get(0, 0).copied().unwrap(), 1.0, 1e-12);
        {
            let vm = sys.view_mut();
            if let Some(a00) = vm.a_storage().get(0, 0) {
                assert_eq!(*a00, 0.5);
            }
        }
    }
}
