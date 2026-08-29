//! RISC-V NMSIS-DSP subprogram backend (`NmsisDspBlas`).
//!
//! Copy this package into firmware rather than depending on it from `control-rs`.
//! Operator instructions: `README.md` in this directory.
#![no_std]
#![allow(unsafe_op_in_unsafe_fn)]

pub mod nmsis;
pub use nmsis::NmsisDspBlas;
