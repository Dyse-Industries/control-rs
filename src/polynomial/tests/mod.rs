#![allow(missing_docs)]
#![allow(clippy::used_underscore_items)]

pub mod polynomial_tests;

#[cfg(not(test))]
pub mod suites {
    pub use super::polynomial_tests::polynomial_test_suite::SUITE_DESCRIPTOR_PTR as polynomial;
}
