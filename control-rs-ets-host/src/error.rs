//! Error types for host-side ETS communication and execution.

use std::fmt;
use thiserror::Error;

/// Contextual error source description implementing [`std::error::Error`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErrorSource(pub String);

impl fmt::Display for ErrorSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for ErrorSource {}

impl From<String> for ErrorSource {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for ErrorSource {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// Host-side ETS communication and execution error.
#[derive(Debug, Error)]
pub enum HostError {
    /// Failed to compile target binary.
    #[error("failed to build target '{target}': {source}")]
    Build {
        /// Target triple or identifier.
        target: String,
        /// Failure message or underlying error description.
        source: ErrorSource,
    },

    /// Serial port failed to open after multiple attempts.
    #[error(
        "failed to open serial port '{port}' after {attempts} attempts: {source}"
    )]
    SerialOpen {
        /// Serial port path.
        port: String,
        /// Number of connection attempts made.
        attempts: u32,
        /// Underlying error description.
        source: ErrorSource,
    },

    /// Serial port handle could not be cloned for reader thread.
    #[error("failed to clone serial port: {source}")]
    SerialClone {
        /// Underlying error description.
        source: ErrorSource,
    },

    /// Failed to spawn the target subprocess.
    #[error("failed to spawn subprocess: {source}")]
    Spawn {
        /// Underlying error description.
        source: ErrorSource,
    },

    /// Transport-level I/O error on an established link.
    #[error("transport error: {source}")]
    Transport {
        /// Underlying error description.
        source: ErrorSource,
    },

    /// Protocol version mismatch between target and host.
    #[error(
        "protocol mismatch: host expected {host}, target reported {target}"
    )]
    ProtocolMismatch {
        /// Host expected protocol version.
        host: u32,
        /// Target reported protocol version.
        target: u32,
    },

    /// Target failed to complete discovery within the allowed duration.
    #[error("target discovery timed out")]
    Discovery,
}
