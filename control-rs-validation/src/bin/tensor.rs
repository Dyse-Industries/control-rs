//! Emitter for the `tensor` validation suite.
//!
//! Writes `results/tensor.rust.h5`; the validate runner then executes the
//! oracle command and compares the containers.

fn main() {
    control_rs_validation::tensor::run();
}
