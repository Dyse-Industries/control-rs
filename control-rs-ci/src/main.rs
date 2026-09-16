//! Continuous integration test runner and quality-gate orchestrator for `control-rs`.

fn main() {
    control_rs_ci::runner::run();
}
