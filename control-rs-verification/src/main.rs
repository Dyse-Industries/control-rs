//! Verification runner for control-rs numerical models.
//!
//! Emits `.rust.h5` containers across all 5 numerical domains for
//! cross-control-rs-verification against SciPy reference oracles.

#![allow(clippy::doc_markdown, clippy::too_many_lines)]

use std::env;
use std::process::ExitCode;

use control_rs_verification::{
    matrix, polynomial, results_dir, state_space, tensor, transfer_function,
};

fn main() -> ExitCode {
    let rdir = results_dir();
    let target = env::args().nth(1);

    match target.as_deref() {
        Some("matrix") => {
            if let Err(e) = matrix::emit_container(&rdir.join("matrix.rust.h5"))
            {
                eprintln!("Matrix emission error: {e}");
                return ExitCode::FAILURE;
            }
        }
        Some("polynomial") => {
            if let Err(e) =
                polynomial::emit_container(&rdir.join("polynomial.rust.h5"))
            {
                eprintln!("Polynomial emission error: {e}");
                return ExitCode::FAILURE;
            }
        }
        Some("state_space") => {
            if let Err(e) =
                state_space::emit_container(&rdir.join("state_space.rust.h5"))
            {
                eprintln!("State space emission error: {e}");
                return ExitCode::FAILURE;
            }
        }
        Some("transfer_function") => {
            if let Err(e) = transfer_function::emit_container(
                &rdir.join("transfer_function.rust.h5"),
            ) {
                eprintln!("Transfer function emission error: {e}");
                return ExitCode::FAILURE;
            }
        }
        Some("tensor") => {
            if let Err(e) = tensor::emit_container(&rdir.join("tensor.rust.h5"))
            {
                eprintln!("Tensor emission error: {e}");
                return ExitCode::FAILURE;
            }
        }
        None | Some("all") => {
            if let Err(e) = matrix::emit_container(&rdir.join("matrix.rust.h5"))
            {
                eprintln!("Matrix emission error: {e}");
                return ExitCode::FAILURE;
            }
            if let Err(e) =
                polynomial::emit_container(&rdir.join("polynomial.rust.h5"))
            {
                eprintln!("Polynomial emission error: {e}");
                return ExitCode::FAILURE;
            }
            if let Err(e) =
                state_space::emit_container(&rdir.join("state_space.rust.h5"))
            {
                eprintln!("State space emission error: {e}");
                return ExitCode::FAILURE;
            }
            if let Err(e) = transfer_function::emit_container(
                &rdir.join("transfer_function.rust.h5"),
            ) {
                eprintln!("Transfer function emission error: {e}");
                return ExitCode::FAILURE;
            }
            if let Err(e) = tensor::emit_container(&rdir.join("tensor.rust.h5"))
            {
                eprintln!("Tensor emission error: {e}");
                return ExitCode::FAILURE;
            }
        }
        Some(other) => {
            eprintln!(
                "Unknown domain '{other}'. Choose from: matrix, polynomial, state_space, transfer_function, tensor, all"
            );
            return ExitCode::FAILURE;
        }
    }

    ExitCode::SUCCESS
}
