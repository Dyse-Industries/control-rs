//! Native JSON generator for transfer_function.
fn main() {
    control_rs_numerical_model_examples::suite::run_cli(
        "transfer_function",
        control_rs_numerical_model_examples::transfer_function::run,
    );
}
