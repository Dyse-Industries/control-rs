//! Write `results/transfer_function/native.json`.

use control_rs_numerical_model_examples::{emit, transfer_function};

fn main() {
    emit::write_artifact(
        "results/transfer_function/native.json",
        &transfer_function::native_artifact(),
    );
}
