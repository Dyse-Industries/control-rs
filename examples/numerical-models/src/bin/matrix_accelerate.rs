//! Native JSON generator for Accelerate GEMM (`source: rust-accelerate`).
fn main() {
    control_rs_numerical_model_examples::suite::run_cli(
        "matrix",
        control_rs_numerical_model_examples::matrix::run_accelerate,
    );
}
