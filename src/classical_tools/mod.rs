//! # Classical Tools
//!
//! SISO control system analysis, compensator synthesis, and real-time
//! firmware execution topologies:
//!
//! - Routh-Hurwitz closed-loop characteristic polynomial stability and
//!   right-half-plane pole counts ([`routh::stability`]).
//! - Root-locus closed-loop pole trajectories over parameterized gain
//!   sweeps ([`root_locus::sweep`]).
//! - Gain margin, phase margin, delay margin, and crossover frequency
//!   extraction from rational transfer functions
//!   ([`margins::stability_margins`]).
//! - Discrete-time firmware realizations of transfer functions and
//!   polynomials: Transposed Direct Form II
//!   ([`realization::DirectForm2T`]) and cascaded Second-Order Sections
//!   ([`realization::Biquad`] / [`realization::BiquadCascade`]).
//! - Continuous and discrete lead, lag, and lead-lag compensator synthesis
//!   emitting [`crate::transfer_function::TransferFunction`] primitives
//!   ([`compensators::lead`], [`compensators::lag`],
//!   [`compensators::lead_lag`]).
//! - Discrete PID control on scalar signals with a continuous-time transfer
//!   function model for analysis ([`pid::Pid`]).

pub mod compensators;
pub mod margins;
pub mod pid;
pub mod realization;
pub mod root_locus;
pub mod routh;
pub mod step_info;

#[cfg(any(test, feature = "ets"))]
/// Classical tools module unit tests.
pub mod tests;
