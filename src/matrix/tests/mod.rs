#![allow(missing_docs)]
#![allow(clippy::used_underscore_items)]

pub mod matrix_tests;

/// Grouped re-exports of all ETS suite descriptors to force link them in example binaries.
#[cfg(not(test))]
pub mod suites {
    // matrix
    pub use super::matrix_tests::matrix_test_suite::SUITE_DESCRIPTOR_PTR as matrix;
}
