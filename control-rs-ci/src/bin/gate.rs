//! Targeted quality gate runner CLI (`cargo gate`).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::missing_docs_in_private_items,
    clippy::uninlined_format_args,
    clippy::too_many_lines,
    clippy::collapsible_if,
    clippy::items_after_statements
)]

use control_rs_ci::run_cli;

fn main() {
    run_cli("cargo gate");
}
