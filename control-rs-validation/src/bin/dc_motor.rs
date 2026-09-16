//! Emitter for the `dc_motor` validation suite.
//!
//! Writes `results/dc_motor.rust.h5`; the validate runner then executes the
//! oracle command and compares the containers.

fn main() {
    control_rs_validation::dc_motor::run();
}
