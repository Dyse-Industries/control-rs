//! Emitter for the `buck_converter` validation suite.
//!
//! Writes `results/buck_converter.rust.h5`; the validate runner then executes the
//! oracle command and compares the containers.

fn main() {
    control_rs_validation::buck_converter::run();
}
