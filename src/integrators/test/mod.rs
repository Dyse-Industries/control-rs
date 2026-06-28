//! Integrators HIL and unit test suite.
#![allow(clippy::arithmetic_side_effects, clippy::float_cmp)]

#[cfg_attr(not(test), control_rs_macros::hil_suite)]
/// Integrators integration tests.
pub mod test_integrators {
    use crate::integrators::{Euler, Rk4};
    use crate::math::StateEquation;
    use crate::math::subprograms::BasicSubProgramsF32;

    struct TestModel;

    impl StateEquation<[f32; 1], f32, [f32; 1]> for TestModel {
        fn dynamics(&self, x: &[f32; 1], u: &f32) -> [f32; 1] {
            // dx/dt = -x + u
            [-x[0] + *u]
        }
    }

    #[cfg_attr(test, test)]
    /// Verifies Euler integrator steps forward correctly.
    fn test_euler_integrator() {
        let model = TestModel;
        let mut x = [1.0];
        let u = 2.0;
        let dt = 0.1;

        // dx = -1.0 + 2.0 = 1.0
        // x_new = 1.0 + 0.1 * 1.0 = 1.1
        Euler::step::<f32, 1, _, _, BasicSubProgramsF32>(
            &model, &mut x, &u, dt,
        );
        assert!((x[0] - 1.1).abs() < 1e-6);
    }

    #[cfg_attr(test, test)]
    /// Verifies RK4 integrator steps forward correctly.
    fn test_rk4_integrator() {
        let model = TestModel;
        let mut x = [1.0];
        let u = 2.0;
        let dt = 0.1;

        // Analytical solution for dx/dt = -x + 2, x(0) = 1:
        // x(t) = 2 - e^(-t)
        // x(0.1) = 2 - e^(-0.1) = 2 - 0.904837418 = 1.095162582
        Rk4::step::<f32, 1, _, _, BasicSubProgramsF32>(&model, &mut x, &u, dt);
        assert!((x[0] - 1.095_162_6).abs() < 1e-6);
    }
}
