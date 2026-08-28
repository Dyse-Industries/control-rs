//! Write `results/tensor/native.json`.

use control_rs_numerical_model_examples::{emit, tensor};

fn main() {
    emit::write_artifact("results/tensor/native.json", &tensor::native_artifact());
}
