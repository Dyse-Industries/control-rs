//! Native JSON generator for row-major matrix (`source: rust-row`).
fn main() {
    control_rs_numerical_model_examples::suite::run_cli(
        "matrix",
        control_rs_numerical_model_examples::matrix::run_row,
    );
}
