//! Emitter for the `state_space` validation suite.
//!
//! Writes `results/state_space.rust.h5`; the validate runner then executes the
//! oracle command and compares the containers.

fn main() {
    control_rs_validation::state_space::run();
}
