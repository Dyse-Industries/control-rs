//! Write `results/state_space/native.json`.

use control_rs_numerical_model_examples::{emit, state_space};

fn main() {
    emit::write_artifact(
        "results/state_space/native.json",
        &state_space::native_artifact(),
    );
}
