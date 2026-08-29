//! Native JSON generator for state_space.
fn main() {
    control_rs_numerical_model_examples::suite::run_cli(
        "state_space",
        control_rs_numerical_model_examples::state_space::run,
    );
}
