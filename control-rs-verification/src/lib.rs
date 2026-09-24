//! Verification harness library for control-rs numerical models.
//!
//! Emits `.rust.h5` containers across all 5 numerical domains for
//! cross-control-rs-verification against `NumPy`/`SciPy` reference oracles.

pub use h5_writer::{H5Writer, results_dir};

pub mod h5_writer;
pub mod matrix;
pub mod numeric;
pub mod polynomial;
pub mod state_space;
pub mod tensor;
pub mod transfer_function;
