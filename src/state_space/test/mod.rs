//! State-space HIL and unit test suite.
#![allow(
    clippy::arithmetic_side_effects,
    clippy::many_single_char_names,
    clippy::float_cmp
)]

#[cfg_attr(not(test), control_rs_macros::hil_suite)]
/// State-space dynamic model tests.
pub mod test_state_space {
    use crate::math::StateEquation;
    use crate::state_space::StateSpace;

    #[cfg_attr(test, test)]
    /// Verifies state-space dynamics evaluation.
    fn test_state_space_dynamics() {
        // A simple 2D system:
        // dx/dt = A * x + B * u
        let a = [[1.0, 2.0], [3.0, 4.0]];
        let b = [[5.0], [6.0]];
        let c = [[0.0, 0.0]];
        let d = [[0.0]];
        let sys = StateSpace::new(a, b, c, d);

        let x = [1.0, 2.0];
        let u = [3.0];

        // dx[0] = 1*1 + 2*2 + 5*3 = 5 + 15 = 20
        // dx[1] = 3*1 + 4*2 + 6*3 = 11 + 18 = 29
        let dx = sys.dynamics(&x, &u);
        assert_eq!(dx[0], 20.0);
        assert_eq!(dx[1], 29.0);
    }

    #[cfg_attr(test, test)]
    /// Verifies state-space output evaluation.
    fn test_state_space_output() {
        let a = [[0.0, 0.0], [0.0, 0.0]];
        let b = [[0.0], [0.0]];
        let c = [[1.0, 2.0], [3.0, 4.0]];
        let d = [[5.0], [6.0]];
        let sys = StateSpace::new(a, b, c, d);

        let x = [1.0, 2.0];
        let u = [3.0];

        // y[0] = 1*1 + 2*2 + 5*3 = 5 + 15 = 20
        // y[1] = 3*1 + 4*2 + 6*3 = 11 + 18 = 29
        let y = sys.output(&x, &u);
        assert_eq!(y[0], 20.0);
        assert_eq!(y[1], 29.0);
    }
}
