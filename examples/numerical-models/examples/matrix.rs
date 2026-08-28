//! Write `results/matrix/native.json`.

use control_rs_numerical_model_examples::{emit, matrix};

fn main() {
    emit::write_artifact("results/matrix/native.json", &matrix::native_artifact());
}
