#![cfg_attr(feature = "ets", allow(clippy::unwrap_used))]
#![allow(
    clippy::cast_precision_loss,
    clippy::items_after_statements,
    clippy::type_complexity,
    clippy::used_underscore_items,
    missing_docs
)]

pub mod compensators_tests;
pub mod margins_tests;
pub mod pid_tests;
pub mod realization_tests;
pub mod root_locus_tests;
pub mod routh_tests;
pub mod step_info_tests;

#[cfg(not(test))]
pub mod suites {
    pub use super::compensators_tests::compensators_test_suite::SUITE_DESCRIPTOR_PTR as compensators;
    pub use super::margins_tests::margins_test_suite::SUITE_DESCRIPTOR_PTR as margins;
    pub use super::pid_tests::pid_test_suite::SUITE_DESCRIPTOR_PTR as pid;
    pub use super::realization_tests::firmware_test_suite::SUITE_DESCRIPTOR_PTR as firmware;
    pub use super::root_locus_tests::root_locus_test_suite::SUITE_DESCRIPTOR_PTR as root_locus;
    pub use super::routh_tests::routh_test_suite::SUITE_DESCRIPTOR_PTR as routh;
    pub use super::step_info_tests::step_info_test_suite::SUITE_DESCRIPTOR_PTR as step_info;
}
