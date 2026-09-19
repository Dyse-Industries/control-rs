//! Main CLI coordinator for `control-rs-ci`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::missing_docs_in_private_items,
    clippy::uninlined_format_args
)]

use control_rs_ci::run_cli;

fn main() {
    run_cli("cargo ci");
}
