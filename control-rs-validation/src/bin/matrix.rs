//! Emitter for the `matrix` validation suite.
//!
//! Writes `results/matrix.rust.h5`; the validate runner then executes the
//! oracle command and compares the containers.

fn main() {
    control_rs_validation::matrix::run();
}
