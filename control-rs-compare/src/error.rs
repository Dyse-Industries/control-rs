//! Error types for the oracle harness and comparison engine.

use std::path::PathBuf;
use thiserror::Error;

/// Core error classification for the oracle control-rs-verification harness.
#[derive(Debug, Error)]
pub enum HarnessError {
    /// Configuration file reading, parsing, or resolution error.
    #[error("Configuration error in '{path}': {message}")]
    Config {
        /// File path where the configuration error occurred.
        path: PathBuf,
        /// Description of the configuration error.
        message: String,
    },

    /// HDF5 container read, traversal, or format error.
    #[error("HDF5 error in '{path}': {message}")]
    Hdf5 {
        /// File path of the HDF5 container.
        path: PathBuf,
        /// Description of the HDF5 error.
        message: String,
    },

    /// Variant execution or process spawn failure.
    #[error(
        "Execution failure for variant '{variant}' in suite '{suite}': {message}"
    )]
    Execution {
        /// Name of the suite.
        suite: String,
        /// Name of the execution variant.
        variant: String,
        /// Description of the failure.
        message: String,
    },

    /// Execution timeout exceeded.
    #[error(
        "Execution timeout exceeded ({timeout_secs:.1}s) for variant '{variant}' in suite '{suite}'"
    )]
    Timeout {
        /// Name of the suite.
        suite: String,
        /// Name of the execution variant.
        variant: String,
        /// Configured timeout in seconds.
        timeout_secs: f64,
    },

    /// Discrepancy or tolerance breach during dataset comparison.
    #[error(
        "Tolerance discrepancy in suite '{suite}', signal '{signal}': {details}"
    )]
    Discrepancy {
        /// Name of the suite.
        suite: String,
        /// Signal or dataset path.
        signal: String,
        /// Details of the discrepancy.
        details: String,
    },

    /// Standard I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization or deserialization error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// TOML deserialization error.
    #[error("TOML error: {0}")]
    Toml(#[from] toml::de::Error),
}
