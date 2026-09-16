#![cfg_attr(feature = "ets", allow(clippy::unwrap_used))]
#![allow(missing_docs)]

pub mod transfer_function_tests;

#[cfg(not(test))]
pub mod suites {
    pub use super::transfer_function_tests::transfer_function_test_suite::SUITE_DESCRIPTOR_PTR as transfer_function;
}
