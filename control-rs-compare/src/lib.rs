//! `control-rs-compare` — Cross-Compare Harness & HDF5 Verification Comparison System.
//!
//! Provides typed HDF5 numerical comparison, unified `compare.toml` configuration loading,
//! multi-language variant process orchestration, and structured control-rs-verification reporting.

#![deny(missing_docs)]

pub use crate::compare::{
    ComparatorOptions, SignalTolerancePolicy, ToleranceSpec, compare_dataset,
    compare_float_arrays, discover_datasets, resolve_signal_tolerances,
    run_comparison,
};
pub use crate::config::{
    CompareConfigFile, CompareGeneralConfig, MasterPlan, OracleConfigFile,
    OracleGeneralConfig, SignalToleranceConfig, SuiteConfig,
    ToleranceMethodConfig, ToleranceTable, VariantConfig,
};
pub use crate::error::HarnessError;
pub use crate::report::ValidationReport;
pub use crate::runner::{RunnerOptions, execute_master_plan};

pub mod compare;
pub mod config;
pub mod error;
pub mod report;
pub mod runner;
