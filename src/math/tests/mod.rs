#![allow(clippy::used_underscore_items)]

/// Unit test suite for complex number mathematical operations.
pub mod complex_num_tests;
pub mod dsp_tests;
pub mod fixed_num_tests;
pub mod num_trait_tests;
pub mod num_type_tests;
pub mod op_tests;
#[cfg(not(kani))]
pub mod storage_layout_tests;
pub mod storage_sparse_tests;
pub mod storage_tests;
#[cfg(not(kani))]
pub mod subprogram_kernel_tests;
pub mod subprogram_tests;
