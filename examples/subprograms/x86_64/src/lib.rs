//! x86_64 hardware-accelerated subprograms example crate.
//!
//! Provides [`Avx2Blas`] using AVX2+FMA vector intrinsics, and conditionally
//! [`CblasBlas`] using Netlib CBLAS via `-lcblas -lblas`.
//!
//! Copy this package into firmware rather than depending on it from `control-rs`.
//! Operator instructions: `README.md` in this directory.

#![allow(unsafe_op_in_unsafe_fn)]

pub mod avx2;
pub use avx2::Avx2Blas;

pub mod cblas;
pub use cblas::CblasBlas;
