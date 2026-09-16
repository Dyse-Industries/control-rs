#![cfg_attr(feature = "ets", allow(clippy::unwrap_used))]
#![allow(missing_docs)]

pub mod tensor_tests;

#[cfg(not(test))]
pub mod suites {
    pub use super::tensor_tests::tensor_test_suite::SUITE_DESCRIPTOR_PTR as tensor;
}
