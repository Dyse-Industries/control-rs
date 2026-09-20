//! `control-rs-compare` — Cross-Compare Harness & HDF5 Verification Comparison System.
//!
//! Provides typed HDF5 numerical comparison, unified `compare.toml` configuration loading,
//! multi-language variant process orchestration, and structured verification reporting.

#![deny(missing_docs)]
#![allow(
    clippy::arbitrary_source_item_ordering,
    clippy::arithmetic_side_effects,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::collapsible_if,
    clippy::doc_markdown,
    clippy::indexing_slicing,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::module_name_repetitions,
    clippy::multiple_crate_versions,
    clippy::must_use_candidate,
    clippy::nursery,
    clippy::similar_names,
    clippy::suboptimal_flops,
    clippy::too_many_lines,
    clippy::type_complexity,
    clippy::uninlined_format_args
)]

pub mod compare;
pub mod config;
pub mod error;
pub mod report;
pub mod runner;

pub use crate::compare::{
    ComparatorOptions, ToleranceSpec, compare_dataset, compare_float_arrays,
    run_comparison,
};
pub use crate::config::{
    CompareConfigFile, CompareGeneralConfig, MasterPlan, OracleConfigFile,
    OracleGeneralConfig, SuiteConfig, VariantConfig,
};
pub use crate::error::HarnessError;
pub use crate::report::ValidationReport;
pub use crate::runner::{RunnerOptions, execute_master_plan};
