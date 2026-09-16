//! Emitter for the `transfer_function` validation suite.
//!
//! Writes `results/transfer_function.rust.h5`; the validate runner then executes the
//! oracle command and compares the containers.

fn main() {
    control_rs_validation::transfer_function::run();
}
