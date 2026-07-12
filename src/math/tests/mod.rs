#![allow(missing_docs)]
#![allow(clippy::used_underscore_items)]

/// HIL test suite for complex number mathematical operations.
pub mod complex_num_tests;
pub mod convolution_tests;
pub mod fft_tests;
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
        test_arithmetic::SUITE_DESCRIPTOR_PTR as complex_num_arithmetic,
        test_axioms::SUITE_DESCRIPTOR_PTR as complex_num_axioms,
        test_basics::SUITE_DESCRIPTOR_PTR as complex_num_basics,
        test_core_math::SUITE_DESCRIPTOR_PTR as complex_num_core_math,
        test_dsp_patterns::SUITE_DESCRIPTOR_PTR as complex_num_dsp_patterns,
        test_ffi_layout::SUITE_DESCRIPTOR_PTR as complex_num_ffi_layout,
        test_limitations::SUITE_DESCRIPTOR_PTR as complex_num_limitations,
        test_transcendental::SUITE_DESCRIPTOR_PTR as complex_num_transcendental,
    };

    // convolution_tests
    pub use super::convolution_tests::convolution_test_suite::SUITE_DESCRIPTOR_PTR as convolution;

    // fft_tests
    pub use super::fft_tests::fft_test_suite::SUITE_DESCRIPTOR_PTR as fft;

    // num_trait_tests
    pub use super::num_trait_tests::{
        cartesian_plane_tests::SUITE_DESCRIPTOR_PTR as num_trait_cartesian_plane,
        custom_tests::SUITE_DESCRIPTOR_PTR as num_trait_custom,
        real_tests::SUITE_DESCRIPTOR_PTR as num_trait_real,
        ring_tests::SUITE_DESCRIPTOR_PTR as num_trait_ring,
        scalar_tests::SUITE_DESCRIPTOR_PTR as num_trait_scalar,
    };

    // num_type_tests
    pub use super::num_type_tests::num_type_test_suite::SUITE_DESCRIPTOR_PTR as num_type;

    // op_tests
    pub use super::op_tests::{
        test_float_add::SUITE_DESCRIPTOR_PTR as op_float_add,
        test_float_div::SUITE_DESCRIPTOR_PTR as op_float_div,
        test_float_mul::SUITE_DESCRIPTOR_PTR as op_float_mul,
        test_float_rem::SUITE_DESCRIPTOR_PTR as op_float_rem,
        test_float_sub::SUITE_DESCRIPTOR_PTR as op_float_sub,
        test_int_div::SUITE_DESCRIPTOR_PTR as op_int_div,
        test_int_mul::SUITE_DESCRIPTOR_PTR as op_int_mul,
        test_neg::SUITE_DESCRIPTOR_PTR as op_neg,
        test_rem::SUITE_DESCRIPTOR_PTR as op_rem,
        test_saturating_add::SUITE_DESCRIPTOR_PTR as op_saturating_add,
        test_saturating_mul::SUITE_DESCRIPTOR_PTR as op_saturating_mul,
        test_saturating_sub::SUITE_DESCRIPTOR_PTR as op_saturating_sub,
        test_shl::SUITE_DESCRIPTOR_PTR as op_shl,
        test_shr::SUITE_DESCRIPTOR_PTR as op_shr,
        test_sub::SUITE_DESCRIPTOR_PTR as op_sub,
        test_wrapping_add::SUITE_DESCRIPTOR_PTR as op_wrapping_add,
        test_wrapping_mul::SUITE_DESCRIPTOR_PTR as op_wrapping_mul,
        test_wrapping_sub::SUITE_DESCRIPTOR_PTR as op_wrapping_sub,
    };

    // storage_tests
    pub use super::storage_tests::storage_test_suite::SUITE_DESCRIPTOR_PTR as storage;

    // subprogram_tests
    pub use super::subprogram_tests::{
        fuzzing::SUITE_DESCRIPTOR_PTR as subprogram_fuzzing,
        level1::SUITE_DESCRIPTOR_PTR as subprogram_level1,
        level2::SUITE_DESCRIPTOR_PTR as subprogram_level2,
        level3::SUITE_DESCRIPTOR_PTR as subprogram_level3,
    };
}
