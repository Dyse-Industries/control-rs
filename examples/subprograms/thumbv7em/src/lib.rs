//! Cortex-M CMSIS-DSP subprogram backend (`CmsisDspBlas`).
//!
//! Copy this package into firmware rather than depending on it from `control-rs`.
//! Operator instructions: `README.md` in this directory.
#![no_std]
#![allow(unsafe_op_in_unsafe_fn)]

pub mod cmsis;
pub use cmsis::CmsisDspBlas;
