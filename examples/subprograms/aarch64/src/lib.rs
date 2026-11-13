//! AArch64 hardware-accelerated subprograms example crate.
//!
//! Provides [`NeonBlas`] using ARM NEON vector intrinsics, and conditionally
//! [`AccelerateBlas`] using Apple vecLib CBLAS via `-framework Accelerate`.
//!
//! Copy this package into firmware rather than depending on it from `control-rs`.
//! Operator instructions: `README.md` in this directory.

pub mod neon;
pub use neon::NeonBlas;

pub mod accelerate;
pub use accelerate::AccelerateBlas;
