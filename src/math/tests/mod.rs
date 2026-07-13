#![allow(missing_docs)]
#![allow(clippy::used_underscore_items)]

/// HIL test suite for complex number mathematical operations.
pub mod complex_num_tests;
pub mod dsp_tests;
pub mod num_trait_tests;
pub mod num_type_tests;
pub mod op_tests;
pub mod storage_tests;
pub mod subprogram_tests;

/// Grouped re-exports of all HIL suite descriptors to force link them in example binaries.
#[cfg(not(test))]
pub mod suites {
    // complex_num_tests
    pub use super::complex_num_tests::{
        complex_num_advanced::SUITE_DESCRIPTOR_PTR as complex_num_advanced,
        complex_num_basic::SUITE_DESCRIPTOR_PTR as complex_num_basic,
    };

    // dsp_tests
    pub use super::dsp_tests::dsp_advanced::SUITE_DESCRIPTOR_PTR as dsp_advanced;

    // num_trait_tests
    pub use super::num_trait_tests::{
        num_traits_advanced::SUITE_DESCRIPTOR_PTR as num_traits_advanced,
        num_traits_basic::SUITE_DESCRIPTOR_PTR as num_traits_basic,
    };

    // num_type_tests
    pub use super::num_type_tests::num_type_basic::SUITE_DESCRIPTOR_PTR as num_type_basic;

    // op_tests
    pub use super::op_tests::{
        ops_advanced::SUITE_DESCRIPTOR_PTR as ops_advanced,
        ops_basic::SUITE_DESCRIPTOR_PTR as ops_basic,
    };

    // storage_tests
    pub use super::storage_tests::storage_basic::SUITE_DESCRIPTOR_PTR as storage_basic;

    // subprogram_tests
    pub use super::subprogram_tests::{
        subprograms_advanced::SUITE_DESCRIPTOR_PTR as subprograms_advanced,
        subprograms_basic::SUITE_DESCRIPTOR_PTR as subprograms_basic,
    };
}
