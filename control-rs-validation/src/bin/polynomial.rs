//! Emitter for the `polynomial` validation suite.
//!
//! Writes `results/polynomial.rust.h5`; the validate runner then executes the
//! oracle command and compares the containers.

fn main() {
    control_rs_validation::polynomial::run();
}
