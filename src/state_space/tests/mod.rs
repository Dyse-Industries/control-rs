#![allow(missing_docs)]
#![allow(clippy::used_underscore_items)]

pub mod state_space_tests;

#[cfg(not(test))]
pub mod suites {
    pub use super::state_space_tests::state_space_test_suite::SUITE_DESCRIPTOR_PTR as state_space;
}
