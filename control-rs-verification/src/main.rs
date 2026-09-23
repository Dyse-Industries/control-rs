//! Verification runner for control-rs numerical models.
//!
//! Emits `.rust.h5` containers across all 5 numerical domains for
//! cross-control-rs-verification against `SciPy` reference oracles.

use std::env;
use std::path::Path;
use std::process::ExitCode;

use control_rs_verification::numeric::KernelResult;
use control_rs_verification::{
    matrix, polynomial, results_dir, state_space, tensor, transfer_function,
};

/// Every domain in emission order.
const DOMAINS: [Domain; 5] = [
    Domain::new("matrix", "Matrix", matrix::emit_container),
    Domain::new("polynomial", "Polynomial", polynomial::emit_container),
    Domain::new("state_space", "State space", state_space::emit_container),
    Domain::new(
        "transfer_function",
        "Transfer function",
        transfer_function::emit_container,
    ),
    Domain::new("tensor", "Tensor", tensor::emit_container),
];

/// Writes one domain's container to the given path.
type Emitter = fn(&Path) -> KernelResult<()>;

/// One numerical domain the runner can emit.
struct Domain {
    /// CLI selector and container file stem.
    name: &'static str,
    /// Label used in error messages.
    label: &'static str,
    /// Container writer.
    emit: Emitter,
}

impl Domain {
    const fn new(
        name: &'static str,
        label: &'static str,
        emit: Emitter,
    ) -> Self {
        Self { name, label, emit }
    }
}

fn main() -> ExitCode {
    let rdir = results_dir();
    let target = env::args().nth(1);

    let selected: Vec<_> = match target.as_deref() {
        None | Some("all") => DOMAINS.iter().collect(),
        Some(name) => DOMAINS.iter().filter(|d| d.name == name).collect(),
    };
    if selected.is_empty() {
        eprintln!(
            "Unknown domain '{}'. Choose from: matrix, polynomial, state_space, transfer_function, tensor, all",
            target.unwrap_or_default()
        );
        return ExitCode::FAILURE;
    }

    for domain in selected {
        let path = rdir.join(format!("{}.rust.h5", domain.name));
        if let Err(e) = (domain.emit)(&path) {
            eprintln!("{} emission error: {e}", domain.label);
            return ExitCode::FAILURE;
        }
    }

    ExitCode::SUCCESS
}
