//! Argument parsing, artifact loading, and report emission for the standalone CLI binaries.
//!
//! Each submodule exposes a `main_impl(args) -> i32` that the matching
//! `bin/*.rs` shell wraps in `std::process::exit`, so the parsing and
//! report-building logic is testable without a subprocess or a real
//! `env::args()`.

pub mod report;
pub mod trace;
