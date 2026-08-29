#![allow(missing_docs)]
#![allow(clippy::used_underscore_items)]

pub mod tensor_tests;

#[cfg(not(test))]
pub mod suites {
    pub use super::tensor_tests::tensor_test_suite::SUITE_DESCRIPTOR_PTR as tensor;
}
