//! Verification harness library for control-rs numerical models.
//!
//! Emits `.rust.h5` containers across all 5 numerical domains for
//! cross-control-rs-verification against NumPy/SciPy reference oracles.

#![allow(
    clippy::arbitrary_source_item_ordering,
    clippy::arithmetic_side_effects,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::expect_used,
    clippy::imprecise_flops,
    clippy::indexing_slicing,
    clippy::items_after_statements,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::module_name_repetitions,
    clippy::must_use_candidate,
    clippy::needless_range_loop,
    clippy::nursery,
    clippy::panic,
    clippy::pedantic,
    clippy::redundant_closure_for_method_calls,
    clippy::similar_names,
    clippy::suboptimal_flops,
    clippy::too_many_lines,
    clippy::type_complexity,
    clippy::uninlined_format_args,
    clippy::unwrap_used
)]

pub mod h5_writer;
pub mod matrix;
pub mod polynomial;
pub mod state_space;
pub mod tensor;
pub mod transfer_function;

pub use h5_writer::{H5Writer, results_dir};
