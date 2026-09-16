//! Continuous integration, verification, and quality gate library for `control-rs`.

pub use report::{CiOptions, CiSkip, GateVerdict};
pub use validate::{H5Container, ToleranceTable, compare_h5_files};

pub mod cli;
pub mod gates;
pub mod quality_gate;
pub mod report;
pub mod runner;
pub mod target_matrix;
pub mod trace;
pub mod validate;
