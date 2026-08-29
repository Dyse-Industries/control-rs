#![allow(missing_docs)]
#![allow(clippy::used_underscore_items)]

pub mod transfer_function_tests;

#[cfg(not(test))]
pub mod suites {
    pub use super::transfer_function_tests::transfer_function_test_suite::SUITE_DESCRIPTOR_PTR as transfer_function;
}
