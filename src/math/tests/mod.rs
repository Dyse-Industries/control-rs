#![allow(missing_docs)]
#![allow(clippy::used_underscore_items)]

/// HIL test suite for complex number mathematical operations.
pub mod complex_num_tests;
pub mod dsp_tests;
pub mod fixed_num_tests;
pub mod num_trait_tests;
pub mod num_type_tests;
pub mod op_tests;
pub mod storage_tests;
pub mod subprogram_tests;

/// Grouped re-exports of all HIL suite descriptors to force link them in example binaries.
#[cfg(not(test))]
pub mod suites {
    // complex_num
    pub use super::complex_num_tests::complex_num_test_suite::SUITE_DESCRIPTOR_PTR as complex_num;

    // dsp
    pub use super::dsp_tests::dsp_test_suite::SUITE_DESCRIPTOR_PTR as dsp;

    // fixed_num
    pub use super::fixed_num_tests::fixed_num_test_suite::SUITE_DESCRIPTOR_PTR as fixed_num;

    // num_traits
    pub use super::num_trait_tests::num_trait_test_suite::SUITE_DESCRIPTOR_PTR as num_traits;

    // num_types
    pub use super::num_type_tests::num_type_test_suite::SUITE_DESCRIPTOR_PTR as num_types;

    // ops
    pub use super::op_tests::op_test_suite::SUITE_DESCRIPTOR_PTR as ops;

    // storage
    pub use super::storage_tests::storage_test_suite::SUITE_DESCRIPTOR_PTR as storage;

    // subprograms
    pub use super::subprogram_tests::subprogram_test_suite::SUITE_DESCRIPTOR_PTR as subprograms;
}
