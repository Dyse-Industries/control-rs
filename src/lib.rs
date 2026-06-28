//! # control-rs
//!
//! `control-rs` is a `no_std` Rust library for numerical modeling, control
//! synthesis, and real-time execution. It targets autonomous systems and bare-metal
//! embedded platforms.
//!
//! ## Core Concepts
//!
//! The library is designed around standard control system representations and algorithms:
//! - **Mathematical Primitives**: Real and field traits, linear algebra subprograms (BLAS), and DSP tools.
//! - **Polynomials**: Statically sized and dynamic polynomials with Horner's evaluation.
//! - **Transfer Functions**: Laplace domain models for classical control design.
//! - **State-Space**: Statically sized linear state-space models.
//! - **Nonlinear Systems**: Custom dynamic models defined by implementing the [`StateEquation`](crate::math::StateEquation) trait.
//! - **Integrators**: Numerical integration solvers (e.g., [`Euler`], [`Rk4`]) to simulate continuous dynamics.
//!
//! ## Usage: Quadcopter Simulation & Control
//!
//! The following example demonstrates how to define a 1D quadcopter altitude control system using
//! a linear state-space model, design a Proportional-Derivative (PD) controller, and simulate the closed-loop
//! system using the Euler integrator.
//!
//! ```rust
//! use control_rs::state_space::StateSpace;
//! use control_rs::integrators::Euler;
//! use control_rs::math::subprograms::BasicSubProgramsF32;
//!
//! // 1. Define the Quadcopter Dynamics (1D Altitude Lift)
//! // State: x = [z (altitude), v (velocity)]
//! // Input: u = [net thrust]
//! // Equations: dz = v, dv = u / mass
//! let mass = 1.0f32;
//! let a = [[0.0, 1.0],
//!          [0.0, 0.0]];
//! let b = [[0.0],
//!          [1.0 / mass]];
//! let c = [[1.0, 0.0]];
//! let d = [[0.0]];
//!
//! let quadcopter = StateSpace::new(a, b, c, d);
//!
//! // 2. Simulation setup
//! let mut x = [0.0, 0.0]; // Initial state [z, v]
//! let r = 10.0;           // Target altitude (setpoint)
//! let dt = 0.01;          // Time step
//!
//! // PD Controller gains
//! let kp = 4.0;
//! let kd = 2.0;
//!
//! // 3. Closed-loop simulation loop
//! for _ in 0..500 {
//!     // PD Control Law: u = Kp * error - Kd * velocity
//!     let u = [kp * (r - x[0]) - kd * x[1]];
//!
//!     // Propagate the system state forward in time
//!     Euler::step::<f32, 2, _, _, BasicSubProgramsF32>(&quadcopter, &mut x, &u, dt);
//! }
//!
//! // The altitude should have converged close to the setpoint
//! assert!((x[0] - r).abs() < 0.1);
//! ```
//!
//! ## Usage: Custom Nonlinear System
//!
//! For systems with nonlinear dynamics (e.g., aerodynamic drag), you can implement [`StateEquation`](crate::math::StateEquation) directly.
//!
//! ```rust
//! use control_rs::math::StateEquation;
//! use control_rs::integrators::Rk4;
//! use control_rs::math::subprograms::BasicSubProgramsF32;
//!
//! struct NonlinearQuadcopter {
//!     mass: f32,
//!     gravity: f32,
//!     drag_coeff: f32,
//! }
//!
//! impl StateEquation<[f32; 2], f32, [f32; 2]> for NonlinearQuadcopter {
//!     fn dynamics(&self, x: &[f32; 2], u: &f32) -> [f32; 2] {
//!         let _z = x[0];
//!         let v = x[1];
//!         let thrust = *u;
//!
//!         let dz = v;
//!         let abs_v = if v < 0.0 { -v } else { v };
//!         // dv = (Thrust / mass) - gravity - (drag * v^2 / mass)
//!         let dv = (thrust / self.mass) - self.gravity - (self.drag_coeff * v * abs_v) / self.mass;
//!         [dz, dv]
//!     }
//! }
//!
//! let quadcopter = NonlinearQuadcopter {
//!     mass: 1.0,
//!     gravity: 9.81,
//!     drag_coeff: 0.1,
//! };
//!
//! let mut x = [0.0, 0.0];
//! let u = 15.0; // Constant thrust greater than gravity (9.81 N)
//! let dt = 0.01;
//!
//! // Integrate using Runge-Kutta 4th Order
//! Rk4::step::<f32, 2, _, _, BasicSubProgramsF32>(&quadcopter, &mut x, &u, dt);
//!
//! assert!(x[0] > 0.0);
//! ```
//!
//! ## Features
//!
//! - `std`: (Optional) Enables the standard library for testing and host-side execution.
//! - `hil`: (Optional) Enables Hardware-in-the-Loop simulation target drivers and macros.
//!
//! ## Limitations
//!
//! - **Fixed-Point support**: While designed for `no_std`, many high-level tools assume floating-point precision.
//! - **Memory allocation**: All matrix/vector sizes must be statically defined via const generics to avoid dynamic memory allocation.

#![cfg_attr(not(feature = "std"), no_std)]

pub mod classical_tools;
pub mod integrators;
pub mod math;
pub mod modern_tools;
pub mod nonlinear_tools;
pub mod polynomial;
pub mod robust_tools;
pub mod state_space;
pub mod transfer_function;
