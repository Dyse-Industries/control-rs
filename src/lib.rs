//! # control-rs
//!
//! `no_std` numerical modeling and control library for embedded and host
//! targets. Model dimensions are const generics, so shape errors are compile
//! errors and no model allocates.
//!
//! # Modules
//!
//! - [`math`]: numeric traits and types, fixed-point scalars, storage
//!   backends, DSP and the BLAS/LAPACK subprogram traits.
//! - [`matrix`]: dense and structured matrices, factorizations, solvers and
//!   the matrix exponential.
//! - [`polynomial`]: evaluation, calculus, division, splines and root finding.
//! - [`tensor`]: N-D tables, interpolation, contraction and quantized
//!   activations.
//! - [`transfer_function`]: rational SISO models, frequency response,
//!   interconnection and discretization.
//! - [`state_space`]: continuous and discrete LTI models, simulation,
//!   interconnection and discretization.
//! - [`classical_tools`], [`modern_tools`], [`robust_tools`],
//!   [`nonlinear_tools`], [`integrators`]: reserved for the control toolboxes.
//!
//! # Usage
//!
//! ```
//! use control_rs::matrix::Owned;
//! use control_rs::state_space::ArrayStateSpace;
//!
//! let a = Owned::<f64, 2, 2>::from_row_arrays([[0.0, 1.0], [-4.0, -0.8]]);
//! let b = Owned::<f64, 2, 1>::from_column([0.0, 1.0]);
//! let c = Owned::<f64, 1, 2>::from_row([1.0, 0.0]);
//! let d = Owned::<f64, 1, 1>::scalar(0.0);
//!
//! let sys = ArrayStateSpace::continuous(a, b, c, d).to_discrete_zoh(0.05);
//! let x = Owned::<f64, 2, 1>::zero();
//! let u = Owned::<f64, 1, 1>::scalar(1.0);
//! let (_x_next, _y) = sys.step(&x, &u);
//! ```
//!
//! # Features
//!
//! | Feature | Effect |
//! |:--|:--|
//! | `std` | Links `std`; the crate is `no_std` without it. |
//! | `ets` | Compiles the test suites for on-target execution through `control-rs-ets`. |
//!
//! # Limitations
//!
//! - The control toolbox modules are placeholders with no public API.
//! - Design documents for each module are in `documentation/` in the
//!   repository.
#![cfg_attr(not(feature = "std"), no_std)]
#![recursion_limit = "256"]

pub mod classical_tools;
pub mod integrators;
pub mod math;
pub mod matrix;
pub mod modern_tools;
pub mod nonlinear_tools;
pub mod polynomial;
pub mod robust_tools;
pub mod state_space;
pub mod tensor;
pub mod transfer_function;
