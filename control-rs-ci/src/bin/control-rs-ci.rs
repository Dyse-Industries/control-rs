//! Main CLI coordinator for `control-rs-ci`.

use control_rs_ci::run_cli;

fn main() {
    run_cli("cargo ci");
}
