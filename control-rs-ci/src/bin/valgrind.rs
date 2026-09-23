//! Multi-example Valgrind Memcheck runner for control-rs.
//!
//! Iterates through all workspace domain examples, executing Valgrind Memcheck
//! to verify zero memory leaks and zero invalid reads/writes across numerical models,
//! fixed-point filters, and DSP algorithms.
//!
//! Handles platform degradation gracefully: on environments without native Valgrind
//! (such as macOS Apple Silicon), it logs a diagnostic message and exits 0. On supported
//! platforms (for example, Linux CI runners), it enforces strict memory safety across all targets.

#![allow(
    clippy::arithmetic_side_effects,
    clippy::too_many_lines,
    clippy::uninlined_format_args
)]

use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};

const KNOWN_EXAMPLES: &[&str] = &[
    "dc_motor",
    "buck_converter",
    "fixed_point_math",
    "dsp_spectral_analysis",
];

type ExampleRunResult = Result<(ExitStatus, String), String>;
type FailureRecord = (String, String);

fn find_workspace_root() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if let Some(parent) = manifest_dir.parent()
        && parent.join("Cargo.toml").exists()
    {
        return parent.to_path_buf();
    }
    manifest_dir
}

fn is_valgrind_available() -> bool {
    Command::new("valgrind")
        .arg("--version")
        .output()
        .is_ok_and(|out| out.status.success())
}

fn build_examples(root: &Path) -> Result<(), String> {
    let status = Command::new("cargo")
        .current_dir(root)
        .args(["build", "--examples"])
        .status()
        .map_err(|e| format!("Failed to spawn cargo build --examples: {e}"))?;

    if status.success() {
        Ok(())
    } else {
        Err(format!(
            "cargo build --examples failed with exit code {:?}",
            status.code()
        ))
    }
}

fn run_valgrind_on_example(
    root: &Path,
    example_name: &str,
) -> ExampleRunResult {
    let binary_path = root.join("target/debug/examples").join(example_name);
    if !binary_path.exists() {
        return Err(format!(
            "Example binary not found: {}",
            binary_path.display()
        ));
    }

    let output = Command::new("valgrind")
        .current_dir(root)
        .args([
            "--leak-check=full",
            "--show-leak-kinds=all",
            "--error-exitcode=1",
            binary_path.to_str().unwrap_or(example_name),
        ])
        .output()
        .map_err(|e| {
            format!("Failed to spawn valgrind on {example_name}: {e}")
        })?;

    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    Ok((output.status, stderr))
}

fn main() {
    println!("=== control-rs Valgrind Memcheck Multi-Example Runner ===");

    if !is_valgrind_available() {
        println!(
            "Notice: 'valgrind' is not installed or supported natively on this host (e.g., macOS Apple Silicon).\n\
             Skipping memory checks (degraded). Valgrind is enforced on Linux CI environments."
        );
        std::process::exit(78);
    }

    let root = find_workspace_root();
    println!("Workspace root: {}", root.display());

    println!("Building all workspace examples...");
    if let Err(e) = build_examples(&root) {
        eprintln!("Error compiling examples: {e}");
        std::process::exit(1);
    }

    let mut failed_examples: Vec<FailureRecord> = Vec::new();
    let mut passed_count = 0;

    for &example in KNOWN_EXAMPLES {
        println!("Running Valgrind Memcheck on example '{example}'...");
        match run_valgrind_on_example(&root, example) {
            Ok((status, stderr)) => {
                if status.success() {
                    println!("  [PASS] '{example}' clean (0 leaks, 0 errors)");
                    passed_count += 1;
                } else {
                    eprintln!(
                        "  [FAIL] '{example}' reported memory errors or leaks (exit code {:?})",
                        status.code()
                    );
                    failed_examples.push((example.to_string(), stderr));
                }
            }
            Err(e) => {
                eprintln!("  [ERROR] '{example}' execution error: {e}");
                failed_examples.push((example.to_string(), e));
            }
        }
    }

    println!();
    if failed_examples.is_empty() {
        println!(
            "Valgrind Memcheck passed cleanly across all {passed_count} examples (0 leaks, 0 errors)."
        );
        std::process::exit(0);
    } else {
        eprintln!(
            "Valgrind Memcheck FAILED on {}/{} examples:",
            failed_examples.len(),
            KNOWN_EXAMPLES.len()
        );
        for (name, log) in &failed_examples {
            eprintln!("--- Example '{name}' Failure Log ---");
            eprintln!("{log}");
        }
        std::process::exit(1);
    }
}
