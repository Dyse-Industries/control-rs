//! # Control-rs
#![cfg_attr(not(feature = "std"), no_std)]
#![recursion_limit = "256"]

pub mod classical_tools;
pub mod integrators;
pub mod math;
pub mod matrix;
pub mod modern_tools;
pub mod nonlinear_tools;
pub mod polynomial;
/// Plant-model types for the common import path.
pub mod prelude;
pub mod robust_tools;
pub mod state_space;
pub mod tensor;
pub mod transfer_function;
