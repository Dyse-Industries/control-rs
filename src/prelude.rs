//! # Prelude
//!
//! The model types and dimension bridge needed to define a plant.
//!
//! Importing the prelude covers the common case: building a matrix, a vector,
//! a polynomial, a transfer function and a state-space model. Storage backends
//! stay out. Reaching for one is the signal that a model-level constructor is
//! missing.
//!
//! # Example
//!
//! ```
//! use control_rs::prelude::*;
//!
//! let sys = ArrayStateSpace::<f64, 2, 1, 1>::continuous(
//!     [[0.0, 1.0], [-2.0, -1.0]],
//!     [[0.0], [1.0]],
//!     [[1.0, 0.0]],
//!     [[0.0]],
//! );
//! let x = ColVector::<f64, 2>::from_column([1.0, 0.0]);
//! let u = ColVector::<f64, 1>::from_column([0.0]);
//! let (x_dot, _y) = sys.derivative(&x, &u);
//! assert_eq!(x_dot.get(1, 0), Some(&-2.0));
//! ```

pub use crate::math::num_types::{Const, Dim};
pub use crate::matrix::{
    ColVector, DiagonalMatrix, Matrix, Owned, RowOwned, RowVector,
};
pub use crate::polynomial::{ArrayPolynomial, Polynomial};
pub use crate::state_space::{ArrayStateSpace, StateSpace};
pub use crate::tensor::{ArrayTensor, Tensor};
pub use crate::transfer_function::{ArrayTransferFunction, TransferFunction};
