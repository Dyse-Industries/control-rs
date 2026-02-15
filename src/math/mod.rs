//! # Math
//!
//! Core numerical primitives optimized for bare-metal execution. This module prioritizes
//! fixed-point safety and stack-allocated matrix operations to avoid heap dependency.
//!
//! # References
//! - [Numerical Recipes - The Art of Scientific Computing](https://numerical.recipes/)

pub mod num_traits;
pub mod num_types;
pub mod ops;
mod static_storage;
pub mod subprograms;
