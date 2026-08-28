//! Write `results/polynomial/native.json`.

use control_rs_numerical_model_examples::{emit, polynomial};

fn main() {
    emit::write_artifact(
        "results/polynomial/native.json",
        &polynomial::native_artifact(),
    );
}
