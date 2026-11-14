//! Classical control unit tests.

pub use crate::classical_control::*;
pub use crate::math::complex_num::Complex;
pub use crate::math::fixed_num::{Fixed, FixedRepr};
pub use crate::math::num_traits::{
    Float, MulAcc, Radical, Scalar, Signed, Trig, Zero,
};
pub use crate::math::num_types::{Const, Dim, DimMax};
pub use crate::math::ops::{SaturatingAdd, SaturatingDiv, SaturatingMul};
pub use crate::polynomial::ArrayPolynomial;
pub use crate::transfer_function::ArrayTransferFunction;

pub mod compensator_tests;
pub mod locus_tests;
pub mod margins_tests;
pub mod nyquist_tests;
pub mod pid_tests;
pub mod realization_tests;
pub mod response_tests;
pub mod routh_tests;
pub mod step_tests;
