//! # State-Space Unit and Invariant Tests
#![allow(clippy::unwrap_used)]
// Expected values here are hand-derived reference expressions; the `mul_add`
// form clippy suggests obscures the algebra being asserted and is not a
// performance concern in a test oracle.
#![allow(clippy::suboptimal_flops)]

#[cfg_attr(not(test), control_rs_macros::ets_suite)]
pub mod state_space_test_suite {
    use crate::assert_almost_eq;
    use crate::math::complex_num::Complex;
    #[allow(unused_imports)]
    use crate::math::num_traits::Float;
    use crate::math::storage::DenseStorage;
    use crate::matrix::{ColVector, Owned, RowVector};
    use crate::state_space::{ArrayStateSpace, StateSpaceError};

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: state-space-design#FR-2
    /// Method: Requirements-based test
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
    /// # Verification
    /// Trace: state-space-design#FR-2
    /// Method: Requirements-based test
    fn test_continuous_derivative() {
        // Mass-spring system: \dot{x} = [[0, 1], [-k/m, -c/m]] x + [[0], [1/m]] u
        let a = Owned::<f64, 2, 2>::from_rows([[0.0, 1.0], [-2.0, -1.0]]);
        let b = ColVector::<f64, 2>::from_column([0.0, 1.0]);
        let c = RowVector::<f64, 2>::from_row([1.0, 0.0]);
        let d = Owned::<f64, 1, 1>::zero();

        let sys = ArrayStateSpace::continuous(a, b, c, d);
        let x = ColVector::<f64, 2>::from_column([1.0, 0.0]); // pos = 1, vel = 0
        let u = Owned::<f64, 1, 1>::zero();

        let (x_dot, y) = sys.derivative(&x, &u);
        assert_almost_eq!(x_dot.get(0, 0).copied().unwrap(), 0.0, 1e-12); // vel = 0
        assert_almost_eq!(x_dot.get(1, 0).copied().unwrap(), -2.0, 1e-12); // accel = -k*x = -2
        assert_almost_eq!(y.get(0, 0).copied().unwrap(), 1.0, 1e-12);
    }

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: state-space-design#FR-4
    /// Method: Requirements-based test
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
    /// # Verification
    /// Trace: state-space-design#FR-4
    /// Method: Requirements-based test
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
    /// # Verification
    /// Trace: state-space-design#FR-4
    /// Method: Requirements-based test
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
        let expected_d = (expected_c * 1.0).mul_add(h, 0.0);
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
    /// # Verification
    /// Trace: state-space-design#FR-3
    /// Method: Requirements-based test
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
    /// Algebraic loop $I - D_2 D_1$ singular (`state-space-design.md` §6.3).
    /// # Verification
    /// Trace: state-space-design#FR-3
    /// Method: Requirements-based test
    fn test_feedback_singular_loop_matrix() {
        let a = Owned::<f64, 1, 1>::from_fn(|_, _| -1.0);
        let b = Owned::<f64, 1, 1>::from_fn(|_, _| 1.0);
        let c = Owned::<f64, 1, 1>::from_fn(|_, _| 1.0);
        let d = Owned::<f64, 1, 1>::from_fn(|_, _| 1.0);
        let g = ArrayStateSpace::continuous(a, b, c, d);
        let h = ArrayStateSpace::continuous(a, b, c, d);
        assert_eq!(
            g.feedback::<1, 2>(&h, 1.0),
            Err(StateSpaceError::SingularLoopMatrix)
        );
    }

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: state-space-design#FR-6, state-space-design#FR-7
    /// Method: Requirements-based test
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
    /// # Verification
    /// Trace: state-space-design#FR-4
    /// Method: Requirements-based test
    fn test_state_space_display_tustin_and_faddeev() {
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
    /// # Verification
    /// Trace: state-space-design#FR-1
    /// Method: Requirements-based test
    fn test_state_space_view() {
        let a = Owned::<f64, 1, 1>::from_fn(|_, _| 0.5);
        let b = Owned::<f64, 1, 1>::from_fn(|_, _| 1.0);
        let c = Owned::<f64, 1, 1>::from_fn(|_, _| 2.0);
        let d = Owned::<f64, 1, 1>::from_fn(|_, _| 0.0);
        let mut sys = ArrayStateSpace::discrete(a, b, c, d, 0.01);
        {
            let view = sys.view();
            assert_almost_eq!(*view.a_storage().get(0, 0).unwrap(), 0.5);
        }
        let a_mat = sys.a_matrix();
        assert_almost_eq!(*a_mat.get(0, 0).unwrap(), 0.5);
        let x0 = Owned::<f64, 1, 1>::zero();
        let u = Owned::<f64, 1, 1>::from_fn(|_, _| 1.0);
        let (x_next, y) = sys.step(&x0, &u);
        assert_almost_eq!(y.get(0, 0).copied().unwrap(), 0.0, 1e-12);
        assert_almost_eq!(x_next.get(0, 0).copied().unwrap(), 1.0, 1e-12);
        {
            let vm = sys.view_mut();
            if let Some(a00) = vm.a_storage().get(0, 0) {
                assert_almost_eq!(*a00, 0.5);
            }
        }
    }

    /// Characteristic polynomial coefficients $(-\mathrm{tr}\,A, \det A)$ of a
    /// $2\times 2$ state matrix (roots are the poles).
    fn _charpoly_2(a: &Owned<f64, 2, 2>) -> (f64, f64) {
        let a00 = *a.get(0, 0).unwrap();
        let a01 = *a.get(0, 1).unwrap();
        let a10 = *a.get(1, 0).unwrap();
        let a11 = *a.get(1, 1).unwrap();
        let tr = a00 + a11;
        let det = a00 * a11 - a01 * a10;
        (-tr, det)
    }

    fn _step_y(
        sys: &ArrayStateSpace<f64, 2, 1, 1>,
        x0: &Owned<f64, 2, 1>,
        n: usize,
    ) -> [f64; 8] {
        let u = Owned::<f64, 1, 1>::from_fn(|_, _| 1.0);
        let mut x = *x0;
        let mut y_out = [0.0_f64; 8];
        for k in 0..n.min(8) {
            let (x_next, y) = sys.step(&x, &u);
            y_out[k] = *y.get(0, 0).unwrap();
            x = x_next;
        }
        y_out
    }

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: state-space-design#FR-5
    /// Method: Requirements-based test
    fn test_similarity_transform_poles_and_step() {
        let a = Owned::<f64, 2, 2>::from_rows([[0.0, 1.0], [-4.0, -0.8]]);
        let b = ColVector::<f64, 2>::from_column([0.0, 1.0]);
        let c = RowVector::<f64, 2>::from_row([1.0, 0.0]);
        let d = Owned::<f64, 1, 1>::zero();
        let sys = ArrayStateSpace::continuous(a, b, c, d);
        let sys_d = sys.to_discrete_zoh(0.05);

        let t = Owned::<f64, 2, 2>::from_rows([[1.0, 0.5], [0.25, 1.0]]);
        let sys_t = sys_d.similarity_transform(&t).unwrap();

        let (c0, c1) = _charpoly_2(&sys_d.a());
        let (c0_t, c1_t) = _charpoly_2(&sys_t.a());
        let scale = c0.abs().max(c1.abs()).max(1.0);
        assert!((c0 - c0_t).abs() / scale <= 10.0 * f64::EPSILON);
        assert!((c1 - c1_t).abs() / scale <= 10.0 * f64::EPSILON);

        let x0 = Owned::<f64, 2, 1>::zero();
        let y = _step_y(&sys_d, &x0, 8);
        let y_t = _step_y(&sys_t, &x0, 8);
        for k in 0..8 {
            assert_almost_eq!(y[k], y_t[k], 1e-12);
        }
        let d_t = *sys_t.d().get(0, 0).unwrap();
        let d_d = *sys_d.d().get(0, 0).unwrap();
        assert_almost_eq!(d_t, d_d, 1e-15);
    }

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: state-space-design#FR-4
    /// Method: Requirements-based test
    fn test_zoh_multi_step_matches_coarse_sample() {
        let a = Owned::<f64, 2, 2>::from_rows([[0.0, 1.0], [-4.0, -0.8]]);
        let b = ColVector::<f64, 2>::from_column([0.0, 1.0]);
        let c = RowVector::<f64, 2>::from_row([1.0, 0.0]);
        let d = Owned::<f64, 1, 1>::zero();
        let sys_c = ArrayStateSpace::continuous(a, b, c, d);

        let ts = 0.05_f64;
        let k = 4_usize;
        let sys_fine = sys_c.to_discrete_zoh(ts);
        let sys_coarse = sys_c.to_discrete_zoh(ts * (k as f64));

        let u = Owned::<f64, 1, 1>::from_fn(|_, _| 1.0);
        let mut x_fine = Owned::<f64, 2, 1>::zero();
        for _ in 0..k {
            let (x_next, _) = sys_fine.step(&x_fine, &u);
            x_fine = x_next;
        }
        let (x_coarse, _) = sys_coarse.step(&Owned::<f64, 2, 1>::zero(), &u);
        for i in 0..2 {
            assert_almost_eq!(
                *x_fine.get(i, 0).unwrap(),
                *x_coarse.get(i, 0).unwrap(),
                1e-9
            );
        }
    }

    /// `from_rows`, `continuous` over array literals and `continuous` over
    /// built matrices all produce the same model.
    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: state-space-design#FR-1
    /// Method: Requirements-based test
    fn test_array_constructors_agree() {
        let from_rows = ArrayStateSpace::<f64, 2, 1, 1>::from_rows(
            [[0.0, 1.0], [-2.0, -1.0]],
            [[0.0], [1.0]],
            [[1.0, 0.0]],
            [[0.0]],
            None,
        );
        let from_literals = ArrayStateSpace::<f64, 2, 1, 1>::continuous(
            [[0.0, 1.0], [-2.0, -1.0]],
            [[0.0], [1.0]],
            [[1.0, 0.0]],
            [[0.0]],
        );
        let from_matrices = ArrayStateSpace::<f64, 2, 1, 1>::continuous(
            Owned::<f64, 2, 2>::from_rows([[0.0, 1.0], [-2.0, -1.0]]),
            ColVector::<f64, 2>::from_column([0.0, 1.0]),
            RowVector::<f64, 2>::from_row([1.0, 0.0]),
            Owned::<f64, 1, 1>::zero(),
        );

        assert_eq!(from_rows, from_literals);
        assert_eq!(from_rows, from_matrices);
        assert!(from_rows.is_continuous());
        assert_eq!(from_rows.a().get(1, 0), Some(&-2.0));
        assert_eq!(from_rows.b().get(1, 0), Some(&1.0));

        let discrete = ArrayStateSpace::<f64, 2, 1, 1>::discrete(
            [[0.0, 1.0], [-2.0, -1.0]],
            [[0.0], [1.0]],
            [[1.0, 0.0]],
            [[0.0]],
            0.01,
        );
        assert_eq!(discrete.sample_time(), Some(0.01));
    }

    /// `step` accepts any dense backend, not only the owned column vector.
    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: state-space-design#FR-2
    /// Method: Requirements-based test
    fn test_step_accepts_borrowed_operands() {
        let sys = ArrayStateSpace::<f64, 2, 1, 1>::discrete(
            [[1.0, 0.1], [0.0, 1.0]],
            [[0.0], [0.1]],
            [[1.0, 0.0]],
            [[0.0]],
            0.1,
        );
        let x = ColVector::<f64, 2>::from_column([1.0, 2.0]);
        let u = ColVector::<f64, 1>::from_column([0.5]);

        let (x_owned, y_owned) = sys.step(&x, &u);
        let (x_view, y_view) = sys.step(&x.slice(), &u.slice());

        assert_eq!(x_owned, x_view);
        assert_eq!(y_owned, y_view);
        assert_almost_eq!(*x_owned.get(0, 0).unwrap(), 1.2, 1e-12);
    }

    #[cfg_attr(test, test)]
    fn test_is_continuous_discrete() {
        let sys_c = ArrayStateSpace::<f64, 1, 1, 1>::continuous(
            Owned::zero(),
            Owned::zero(),
            Owned::zero(),
            Owned::zero(),
        );
        assert!(sys_c.is_continuous());
        assert!(!sys_c.is_discrete());

        let sys_d = ArrayStateSpace::<f64, 1, 1, 1>::discrete(
            Owned::zero(),
            Owned::zero(),
            Owned::zero(),
            Owned::zero(),
            0.1,
        );
        assert!(!sys_d.is_continuous());
        assert!(sys_d.is_discrete());
    }

    #[cfg_attr(test, test)]
    fn test_step_simple_integrator() {
        let a = Owned::<f64, 1, 1>::zero();
        let b = Owned::<f64, 1, 1>::from_fn(|_, _| 1.0);
        let c = Owned::<f64, 1, 1>::from_fn(|_, _| 1.0);
        let d = Owned::<f64, 1, 1>::zero();
        let sys = ArrayStateSpace::discrete(a, b, c, d, 1.0);

        let x0 = Owned::<f64, 1, 1>::from_fn(|_, _| 2.0);
        let u = Owned::<f64, 1, 1>::from_fn(|_, _| 3.0);

        let (x1, y0) = sys.step(&x0, &u);
        assert_almost_eq!(x1.get(0, 0).copied().unwrap(), 3.0);
        assert_almost_eq!(y0.get(0, 0).copied().unwrap(), 2.0);
    }

    #[cfg_attr(test, test)]
    fn test_series_first_order() {
        let a1 = Owned::<f64, 1, 1>::from_fn(|_, _| -1.0);
        let b1 = Owned::<f64, 1, 1>::from_fn(|_, _| 2.0);
        let c1 = Owned::<f64, 1, 1>::from_fn(|_, _| 3.0);
        let d1 = Owned::<f64, 1, 1>::from_fn(|_, _| 4.0);
        let sys1 = ArrayStateSpace::continuous(a1, b1, c1, d1);

        let a2 = Owned::<f64, 1, 1>::from_fn(|_, _| -5.0);
        let b2 = Owned::<f64, 1, 1>::from_fn(|_, _| 6.0);
        let c2 = Owned::<f64, 1, 1>::from_fn(|_, _| 7.0);
        let d2 = Owned::<f64, 1, 1>::from_fn(|_, _| 8.0);
        let sys2 = ArrayStateSpace::continuous(a2, b2, c2, d2);

        let ser = sys1.series::<1, 1, 2>(&sys2);

        assert_almost_eq!(ser.a().get(0, 0).copied().unwrap(), -1.0);
        assert_almost_eq!(ser.a().get(0, 1).copied().unwrap(), 0.0);
        assert_almost_eq!(ser.a().get(1, 0).copied().unwrap(), 18.0);
        assert_almost_eq!(ser.a().get(1, 1).copied().unwrap(), -5.0);

        assert_almost_eq!(ser.b().get(0, 0).copied().unwrap(), 2.0);
        assert_almost_eq!(ser.b().get(1, 0).copied().unwrap(), 24.0);

        assert_almost_eq!(ser.c().get(0, 0).copied().unwrap(), 24.0);
        assert_almost_eq!(ser.c().get(0, 1).copied().unwrap(), 7.0);

        assert_almost_eq!(ser.d().get(0, 0).copied().unwrap(), 32.0);
    }

    #[cfg_attr(test, test)]
    fn test_parallel_first_order() {
        let a1 = Owned::<f64, 1, 1>::from_fn(|_, _| -1.0);
        let b1 = Owned::<f64, 1, 1>::from_fn(|_, _| 2.0);
        let c1 = Owned::<f64, 1, 1>::from_fn(|_, _| 3.0);
        let d1 = Owned::<f64, 1, 1>::from_fn(|_, _| 4.0);
        let sys1 = ArrayStateSpace::continuous(a1, b1, c1, d1);

        let a2 = Owned::<f64, 1, 1>::from_fn(|_, _| -5.0);
        let b2 = Owned::<f64, 1, 1>::from_fn(|_, _| 6.0);
        let c2 = Owned::<f64, 1, 1>::from_fn(|_, _| 7.0);
        let d2 = Owned::<f64, 1, 1>::from_fn(|_, _| 8.0);
        let sys2 = ArrayStateSpace::continuous(a2, b2, c2, d2);

        let par = sys1.parallel::<1, 2>(&sys2);

        assert_almost_eq!(par.a().get(0, 0).copied().unwrap(), -1.0);
        assert_almost_eq!(par.a().get(0, 1).copied().unwrap(), 0.0);
        assert_almost_eq!(par.a().get(1, 0).copied().unwrap(), 0.0);
        assert_almost_eq!(par.a().get(1, 1).copied().unwrap(), -5.0);

        assert_almost_eq!(par.b().get(0, 0).copied().unwrap(), 2.0);
        assert_almost_eq!(par.b().get(1, 0).copied().unwrap(), 6.0);

        assert_almost_eq!(par.c().get(0, 0).copied().unwrap(), 3.0);
        assert_almost_eq!(par.c().get(0, 1).copied().unwrap(), 7.0);

        assert_almost_eq!(par.d().get(0, 0).copied().unwrap(), 12.0);
    }

    #[cfg_attr(test, test)]
    fn test_feedback_known_system() {
        let a = Owned::<f64, 1, 1>::from_fn(|_, _| -1.0);
        let b = Owned::<f64, 1, 1>::from_fn(|_, _| 1.0);
        let c = Owned::<f64, 1, 1>::from_fn(|_, _| 1.0);
        let d = Owned::<f64, 1, 1>::from_fn(|_, _| 0.0);
        let g = ArrayStateSpace::continuous(a, b, c, d);

        let a2 = Owned::<f64, 1, 1>::from_fn(|_, _| -2.0);
        let b2 = Owned::<f64, 1, 1>::from_fn(|_, _| 2.0);
        let c2 = Owned::<f64, 1, 1>::from_fn(|_, _| 2.0);
        let d2 = Owned::<f64, 1, 1>::from_fn(|_, _| 0.0);
        let h = ArrayStateSpace::continuous(a2, b2, c2, d2);

        let cl = g.feedback::<1, 2>(&h, -1.0).unwrap();

        assert_almost_eq!(cl.a().get(0, 0).copied().unwrap(), -1.0);
        assert_almost_eq!(cl.a().get(0, 1).copied().unwrap(), -2.0);
        assert_almost_eq!(cl.a().get(1, 0).copied().unwrap(), 2.0);
        assert_almost_eq!(cl.a().get(1, 1).copied().unwrap(), -2.0);
    }

    #[cfg_attr(test, test)]
    fn test_controllability_matrix_verify() {
        let a = Owned::<f64, 2, 2>::from_rows([[1.0, 2.0], [3.0, 4.0]]);
        let b = Owned::<f64, 2, 1>::from_fn(|i, _| [1.0, 0.0][i]);
        let sys = ArrayStateSpace::continuous(
            a,
            b,
            Owned::<f64, 1, 2>::zero(),
            Owned::<f64, 1, 1>::zero(),
        );
        let ctrb = sys.controllability_matrix::<2>();

        assert_almost_eq!(ctrb.get(0, 0).copied().unwrap(), 1.0);
        assert_almost_eq!(ctrb.get(1, 0).copied().unwrap(), 0.0);
        assert_almost_eq!(ctrb.get(0, 1).copied().unwrap(), 1.0);
        assert_almost_eq!(ctrb.get(1, 1).copied().unwrap(), 3.0);
    }

    #[cfg_attr(test, test)]
    fn test_observability_matrix_verify() {
        let a = Owned::<f64, 2, 2>::from_rows([[1.0, 2.0], [3.0, 4.0]]);
        let c = Owned::<f64, 1, 2>::from_fn(|_, j| [0.0, 1.0][j]);
        let sys = ArrayStateSpace::continuous(
            a,
            Owned::<f64, 2, 1>::zero(),
            c,
            Owned::<f64, 1, 1>::zero(),
        );
        let obsv = sys.observability_matrix::<2>();

        assert_almost_eq!(obsv.get(0, 0).copied().unwrap(), 0.0);
        assert_almost_eq!(obsv.get(0, 1).copied().unwrap(), 1.0);
        assert_almost_eq!(obsv.get(1, 0).copied().unwrap(), 3.0);
        assert_almost_eq!(obsv.get(1, 1).copied().unwrap(), 4.0);
    }

    #[cfg_attr(test, test)]
    fn test_to_transfer_function_round_trip() {
        let a = Owned::<f64, 2, 2>::from_rows([[0.0, 1.0], [-2.0, -3.0]]);
        let b = Owned::<f64, 2, 1>::from_fn(|i, _| [0.0, 1.0][i]);
        let c = Owned::<f64, 1, 2>::from_fn(|_, j| [2.0, 1.0][j]);
        let d = Owned::<f64, 1, 1>::zero();
        let sys = ArrayStateSpace::continuous(a, b, c, d);
        let tf = sys.to_transfer_function::<3>();

        assert_almost_eq!(tf.den_slice()[2], 1.0);
        assert_almost_eq!(tf.den_slice()[1], 3.0);
        assert_almost_eq!(tf.den_slice()[0], 2.0);

        assert_almost_eq!(tf.num_slice()[2], 0.0);
        assert_almost_eq!(tf.num_slice()[1], 1.0);
        assert_almost_eq!(tf.num_slice()[0], 2.0);
    }

    /// Negative feedback of two first-order plants assembles the closed-loop
    /// blocks exactly.
    ///
    /// $G = 6/(s+1)$ from $(a,b,c) = (-1, 2, 3)$ and $H = 1/(s+2)$ from
    /// $(-2, 1, 1)$, both strictly proper, so $F = I - \mathrm{sign}\,D_2 D_1$
    /// is the identity and the corrections reduce to the off-diagonal
    /// coupling. The closed loop is
    /// $T = G/(1 + GH) = 6(s+2)/(s^2 + 3s + 8)$, whose denominator is the
    /// characteristic polynomial of the assembled $A$.
    ///
    /// # Verification
    /// Trace: state-space-design#FR-3
    /// Method: Requirements-based test
    #[cfg_attr(test, test)]
    fn test_feedback_first_order_blocks() {
        let g = ArrayStateSpace::continuous(
            Owned::<f64, 1, 1>::from_fn(|_, _| -1.0),
            Owned::<f64, 1, 1>::from_fn(|_, _| 2.0),
            Owned::<f64, 1, 1>::from_fn(|_, _| 3.0),
            Owned::<f64, 1, 1>::zero(),
        );
        let h = ArrayStateSpace::continuous(
            Owned::<f64, 1, 1>::from_fn(|_, _| -2.0),
            Owned::<f64, 1, 1>::from_fn(|_, _| 1.0),
            Owned::<f64, 1, 1>::from_fn(|_, _| 1.0),
            Owned::<f64, 1, 1>::zero(),
        );

        let cl = g.feedback::<1, 2>(&h, -1.0).unwrap();

        // A = [[a1, -b1 c2], [b2 c1, a2]]
        assert_almost_eq!(cl.a().get(0, 0).copied().unwrap(), -1.0, 1e-12);
        assert_almost_eq!(cl.a().get(0, 1).copied().unwrap(), -2.0, 1e-12);
        assert_almost_eq!(cl.a().get(1, 0).copied().unwrap(), 3.0, 1e-12);
        assert_almost_eq!(cl.a().get(1, 1).copied().unwrap(), -2.0, 1e-12);

        // B feeds the forward path only; C reads the forward path only.
        assert_almost_eq!(cl.b().get(0, 0).copied().unwrap(), 2.0, 1e-12);
        assert_almost_eq!(cl.b().get(1, 0).copied().unwrap(), 0.0, 1e-12);
        assert_almost_eq!(cl.c().get(0, 0).copied().unwrap(), 3.0, 1e-12);
        assert_almost_eq!(cl.c().get(0, 1).copied().unwrap(), 0.0, 1e-12);
        assert_almost_eq!(cl.d().get(0, 0).copied().unwrap(), 0.0, 1e-12);

        // trace(A) = -3 and det(A) = 8 are the coefficients of s^2 + 3s + 8.
        let a = cl.a();
        let trace =
            a.get(0, 0).copied().unwrap() + a.get(1, 1).copied().unwrap();
        let det = a.get(0, 0).copied().unwrap() * a.get(1, 1).copied().unwrap()
            - a.get(0, 1).copied().unwrap() * a.get(1, 0).copied().unwrap();
        assert_almost_eq!(trace, -3.0, 1e-12);
        assert_almost_eq!(det, 8.0, 1e-12);

        // Positive feedback flips the coupling sign and moves the poles:
        // s^2 + 3s - 4 has det(A) = -4.
        let cl_pos = g.feedback::<1, 2>(&h, 1.0).unwrap();
        let ap = cl_pos.a();
        let det_pos = ap.get(0, 0).copied().unwrap()
            * ap.get(1, 1).copied().unwrap()
            - ap.get(0, 1).copied().unwrap() * ap.get(1, 0).copied().unwrap();
        assert_almost_eq!(det_pos, -4.0, 1e-12);
    }

    /// With a non-zero $D_1$ the loop matrix $F = I - \mathrm{sign}\,D_2 D_1$
    /// is no longer the identity, so `feedback` must scale the input and
    /// feedthrough paths through $F^{-1}$.
    ///
    /// Static gains $G = 2$, $H = 3$ under negative feedback give
    /// $T = 2/(1 + 6) = 2/7$, which lands entirely in $D$.
    ///
    /// # Verification
    /// Trace: state-space-design#FR-3
    /// Method: Requirements-based test
    #[cfg_attr(test, test)]
    fn test_feedback_static_gains_scale_through_loop_inverse() {
        let g = ArrayStateSpace::continuous(
            Owned::<f64, 1, 1>::zero(),
            Owned::<f64, 1, 1>::zero(),
            Owned::<f64, 1, 1>::zero(),
            Owned::<f64, 1, 1>::from_fn(|_, _| 2.0),
        );
        let h = ArrayStateSpace::continuous(
            Owned::<f64, 1, 1>::zero(),
            Owned::<f64, 1, 1>::zero(),
            Owned::<f64, 1, 1>::zero(),
            Owned::<f64, 1, 1>::from_fn(|_, _| 3.0),
        );

        let cl = g.feedback::<1, 2>(&h, -1.0).unwrap();
        assert_almost_eq!(cl.d().get(0, 0).copied().unwrap(), 2.0 / 7.0, 1e-12);

        // Positive feedback: 2 / (1 - 6) = -0.4.
        let cl_pos = g.feedback::<1, 2>(&h, 1.0).unwrap();
        assert_almost_eq!(cl_pos.d().get(0, 0).copied().unwrap(), -0.4, 1e-12);
    }

    /// The closed loop reproduces the analytic DC gain
    /// $T(0) = C(-A)^{-1}B + D$, which no individual block assertion implies.
    ///
    /// # Verification
    /// Trace: state-space-design#FR-3
    /// Method: Requirements-based test
    #[cfg_attr(test, test)]
    fn test_feedback_dc_gain_matches_closed_loop_formula() {
        let g = ArrayStateSpace::continuous(
            Owned::<f64, 1, 1>::from_fn(|_, _| -1.0),
            Owned::<f64, 1, 1>::from_fn(|_, _| 2.0),
            Owned::<f64, 1, 1>::from_fn(|_, _| 3.0),
            Owned::<f64, 1, 1>::zero(),
        );
        let h = ArrayStateSpace::continuous(
            Owned::<f64, 1, 1>::from_fn(|_, _| -2.0),
            Owned::<f64, 1, 1>::from_fn(|_, _| 1.0),
            Owned::<f64, 1, 1>::from_fn(|_, _| 1.0),
            Owned::<f64, 1, 1>::zero(),
        );
        let cl = g.feedback::<1, 2>(&h, -1.0).unwrap();

        // (-A) x = B solved by the crate's own LU, then y = C x + D.
        let a = cl.a();
        let neg_a: Owned<f64, 2, 2> =
            Owned::from_fn(|i, j| -a.get(i, j).copied().unwrap_or(0.0));
        let lu = crate::matrix::LuDecomposition::decompose(neg_a).unwrap();
        let mut x = cl.b();
        lu.solve_mut(&mut x).unwrap();
        let y = &cl.c() * &x;

        // G(0) = 6, H(0) = 0.5, so T(0) = 6 / (1 + 3) = 1.5.
        assert_almost_eq!(y.get(0, 0).copied().unwrap(), 1.5, 1e-10);
    }

    /// `to_transfer_function` runs Leverrier-Faddeev, so the denominator is
    /// the characteristic polynomial and the numerator is the Markov
    /// expansion. Both are checked against systems whose transfer function is
    /// known in closed form.
    ///
    /// # Verification
    /// Trace: state-space-design#FR-7
    /// Method: Requirements-based test
    #[cfg_attr(test, test)]
    fn test_to_transfer_function_known_systems() {
        // Integrator: 1/s. Ascending num [1, 0], den [0, 1].
        let integ = ArrayStateSpace::continuous(
            Owned::<f64, 1, 1>::zero(),
            Owned::<f64, 1, 1>::from_fn(|_, _| 1.0),
            Owned::<f64, 1, 1>::from_fn(|_, _| 1.0),
            Owned::<f64, 1, 1>::zero(),
        );
        let tf = integ.to_transfer_function::<2>();
        assert_almost_eq!(tf.num_slice()[0], 1.0, 1e-12);
        assert_almost_eq!(tf.num_slice()[1], 0.0, 1e-12);
        assert_almost_eq!(tf.den_slice()[0], 0.0, 1e-12);
        assert_almost_eq!(tf.den_slice()[1], 1.0, 1e-12);

        // Controllable canonical form of 1/(s^2 + 3s + 2).
        let a: Owned<f64, 2, 2> =
            Owned::from_fn(|i, j| [[0.0, 1.0], [-2.0, -3.0]][i][j]);
        let b: Owned<f64, 2, 1> = Owned::from_fn(|i, _| [0.0, 1.0][i]);
        let c: Owned<f64, 1, 2> = Owned::from_fn(|_, j| [1.0, 0.0][j]);
        let sys =
            ArrayStateSpace::continuous(a, b, c, Owned::<f64, 1, 1>::zero());
        let tf2 = sys.to_transfer_function::<3>();
        let den = tf2.den_slice();
        let num = tf2.num_slice();
        assert_almost_eq!(den[0], 2.0, 1e-12);
        assert_almost_eq!(den[1], 3.0, 1e-12);
        assert_almost_eq!(den[2], 1.0, 1e-12);
        assert_almost_eq!(num[0], 1.0, 1e-12);
        assert_almost_eq!(num[1], 0.0, 1e-12);
        assert_almost_eq!(num[2], 0.0, 1e-12);

        // A non-zero D adds D times the denominator, making the transfer
        // function proper rather than strictly proper.
        let sys_d = ArrayStateSpace::continuous(
            a,
            b,
            c,
            Owned::<f64, 1, 1>::from_fn(|_, _| 5.0),
        );
        let tf3 = sys_d.to_transfer_function::<3>();
        let num3 = tf3.num_slice();
        assert_almost_eq!(num3[0], 1.0 + 5.0 * 2.0, 1e-12);
        assert_almost_eq!(num3[1], 5.0 * 3.0, 1e-12);
        assert_almost_eq!(num3[2], 5.0, 1e-12);
    }

    /// Round trip: the transfer function evaluated on the imaginary axis
    /// agrees with $C(j\omega I - A)^{-1}B + D$ computed from the state-space
    /// matrices directly.
    ///
    /// This pins the numerator against the denominator, which the coefficient
    /// assertions alone do not: scaling both by the same factor would pass
    /// them and fail here.
    ///
    /// # Verification
    /// Trace: state-space-design#FR-7
    /// Method: Requirements-based test
    #[cfg_attr(test, test)]
    fn test_to_transfer_function_round_trip_frequency_response() {
        let a: Owned<f64, 2, 2> =
            Owned::from_fn(|i, j| [[0.0, 1.0], [-2.0, -3.0]][i][j]);
        let b: Owned<f64, 2, 1> = Owned::from_fn(|i, _| [0.0, 1.0][i]);
        let c: Owned<f64, 1, 2> = Owned::from_fn(|_, j| [4.0, 1.0][j]);
        let d = Owned::<f64, 1, 1>::from_fn(|_, _| 0.5);
        let sys = ArrayStateSpace::continuous(a, b, c, d);
        let tf = sys.to_transfer_function::<3>();

        for &w in &[0.0_f64, 0.5, 1.0, 3.0] {
            let h = tf.eval_frequency(w);

            // Direct evaluation: (jw I - A)^{-1} B by 2x2 Cramer, in complex
            // arithmetic, then C x + D.
            let m11 = Complex::new(-a.get(0, 0).copied().unwrap(), w);
            let m12 = Complex::new(-a.get(0, 1).copied().unwrap(), 0.0);
            let m21 = Complex::new(-a.get(1, 0).copied().unwrap(), 0.0);
            let m22 = Complex::new(-a.get(1, 1).copied().unwrap(), w);
            let det = m11 * m22 - m12 * m21;
            let b0 = Complex::new(b.get(0, 0).copied().unwrap(), 0.0);
            let b1 = Complex::new(b.get(1, 0).copied().unwrap(), 0.0);
            let x0 = (m22 * b0 - m12 * b1) / det;
            let x1 = (m11 * b1 - m21 * b0) / det;
            let want = Complex::new(c.get(0, 0).copied().unwrap(), 0.0) * x0
                + Complex::new(c.get(0, 1).copied().unwrap(), 0.0) * x1
                + Complex::new(0.5, 0.0);

            assert_almost_eq!(h.re, want.re, 1e-9);
            assert_almost_eq!(h.im, want.im, 1e-9);
        }
    }
}

#[cfg(test)]
mod state_space_property_tests {
    #[allow(unused_imports)]
    use crate::math::num_traits::Float;
    use crate::matrix::{ColVector, Owned, RowVector};
    use crate::state_space::ArrayStateSpace;
    use proptest::prelude::*;

    fn owned2(vals: &[f64]) -> Owned<f64, 2, 2> {
        Owned::from_fn(|i, j| vals[j * 2 + i])
    }

    fn charpoly_2(a: &Owned<f64, 2, 2>) -> (f64, f64) {
        let a00 = *a.get(0, 0).unwrap();
        let a01 = *a.get(0, 1).unwrap();
        let a10 = *a.get(1, 0).unwrap();
        let a11 = *a.get(1, 1).unwrap();
        (-(a00 + a11), a00 * a11 - a01 * a10)
    }

    proptest! {
        /// Random invertible $T$: $\tilde{A}=TAT^{-1}$ shares the characteristic
        /// polynomial of $A$, and the forced step $y$ is unchanged.
        #[test]
        /// # Verification
        /// Trace: state-space-design#FR-5
        /// Method: Property-based test
        fn prop_similarity_random_t(
            t_vals in proptest::collection::vec(-5.0..5.0_f64, 4),
        ) {
            let t = owned2(&t_vals);
            let det = (*t.get(0, 0).unwrap()).mul_add(
                *t.get(1, 1).unwrap(),
                -(*t.get(0, 1).unwrap() * *t.get(1, 0).unwrap()),
            );
            if !det.is_finite() || det.abs() < 0.05 {
                return Ok(());
            }
            let a = Owned::<f64, 2, 2>::from_rows([[0.0, 1.0], [-4.0, -0.8]]);
            let b = ColVector::<f64, 2>::from_column([0.0, 1.0]);
            let c = RowVector::<f64, 2>::from_row([1.0, 0.0]);
            let d = Owned::<f64, 1, 1>::zero();
            let sys = ArrayStateSpace::discrete(a, b, c, d, 0.05);
            let Ok(sys_t) = sys.similarity_transform(&t) else {
                return Ok(());
            };
            let (p0, p1) = charpoly_2(&sys.a());
            let (q0, q1) = charpoly_2(&sys_t.a());
            let scale = p0.abs().max(p1.abs()).max(1.0);
            prop_assert!((p0 - q0).abs() / scale <= 1e-9);
            prop_assert!((p1 - q1).abs() / scale <= 1e-9);

            let u = Owned::<f64, 1, 1>::from_fn(|_, _| 1.0);
            let mut x = Owned::<f64, 2, 1>::zero();
            let mut z = Owned::<f64, 2, 1>::zero();
            for _ in 0..6 {
                let (xn, y) = sys.step(&x, &u);
                let (zn, yt) = sys_t.step(&z, &u);
                prop_assert!((y.get(0, 0).unwrap() - yt.get(0, 0).unwrap()).abs() < 1e-9);
                x = xn;
                z = zn;
            }
        }
    }
}
