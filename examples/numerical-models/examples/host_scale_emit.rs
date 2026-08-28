//! Emit host-scale native JSON artifacts under `results/<slug>/`.

use control_rs_numerical_model_examples::host_scale_emit;

fn main() {
    host_scale_emit::emit_all();
}
