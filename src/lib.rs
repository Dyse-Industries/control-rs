#![doc = include_str!("../README.md")]
#![cfg_attr(not(feature = "std"), no_std)]
// Clippy docs: https://doc.rust-lang.org/clippy/usage.html
#![deny(
    unused,
    clippy::all,
    clippy::todo,
    clippy::panic,
    clippy::cargo,
    clippy::style,
    clippy::nursery,
    clippy::pedantic,
    clippy::suspicious,
    clippy::complexity,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::unimplemented,
    clippy::big_endian_bytes,
    clippy::indexing_slicing,
    clippy::shadow_unrelated,
    clippy::large_stack_arrays,
    clippy::empty_structs_with_brackets
)]
#![warn(
    missing_docs,
    rust_2018_idioms,
    clippy::complexity,
    clippy::arithmetic_side_effects,
    clippy::arbitrary_source_item_ordering
)]
#![allow(clippy::inline_always)]

pub mod classical_tools;
pub mod integrators;
pub mod math;
pub mod modern_tools;
pub mod nonlinear_tools;
// pub mod polynomial;
pub mod robust_tools;
// pub mod state_space;
// mod transfer_function;

#[cfg(feature = "hil")]
/// Hardware-in-the-loop (HIL) testing and benchmarking tools.
pub mod hil {
    pub use control_rs_hil::{comms, server, settings, time};
    pub use control_rs_macros::{hil_setup, hil_suite};
}