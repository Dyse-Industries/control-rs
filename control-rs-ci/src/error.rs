//! Error types for the `control-rs-ci` quality gating and reporting infrastructure.

use std::path::PathBuf;
use thiserror::Error;

/// Result of a gate, configuration or reporting operation.
pub type GateResult<T> = Result<T, GateError>;

/// Core error type representing failures encountered during gate execution and reporting.
#[derive(Debug, Error)]
pub enum GateError {
    /// Failed to read or parse gate configuration file.
    #[error("Configuration error in {path:?}: {message}")]
    Config {
        /// Path to the configuration file.
        path: PathBuf,
        /// Detail error message.
        message: String,
    },

    /// An I/O error occurred while reading or writing artifacts.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization or deserialization failure.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Process execution exceeded configured wall-clock timeout bound.
    #[error(
        "Process '{gate}' exceeded timeout limit of {timeout_secs:.2} seconds"
    )]
    Timeout {
        /// Name of the gate that timed out.
        gate: String,
        /// Configured timeout in seconds.
        timeout_secs: f64,
    },

    /// Failed to spawn process or execute command.
    #[error("Failed to spawn process for '{gate}': {message}")]
    Spawn {
        /// Name of the gate.
        gate: String,
        /// Detail message.
        message: String,
    },

    /// Git command or repository inspection error.
    #[error("Git error: {0}")]
    Git(String),
}
