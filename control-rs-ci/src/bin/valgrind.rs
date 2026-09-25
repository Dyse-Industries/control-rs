//! Multi-example Valgrind Memcheck runner for control-rs.
//!
//! Iterates through all workspace domain examples, executing Valgrind Memcheck
//! to verify zero memory leaks and zero invalid reads/writes across numerical models,
//! fixed-point filters, and DSP algorithms.
//!
//! Handles platform degradation gracefully: on environments without native Valgrind
//! (such as macOS Apple Silicon), it logs a diagnostic message and exits 0. On supported
//! platforms (for example, Linux CI runners), it enforces strict memory safety across all targets.

use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};

use control_rs_ci::ui;

const KNOWN_EXAMPLES: &[&str] = &[
    "dc_motor",
    "buck_converter",
    "fixed_point_math",
    "dsp_spectral_analysis",
];

type ExampleRunResult = Result<(ExitStatus, String), String>;
type FailureRecord = (String, String);
type CheckResult = (usize, Vec<FailureRecord>);

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

/// Resolves the Cargo target directory from `CARGO_TARGET_DIR`, relative to
/// `root` when not absolute, defaulting to `<root>/target`.
fn target_dir(root: &Path) -> PathBuf {
    std::env::var_os("CARGO_TARGET_DIR")
        .map_or_else(|| root.join("target"), |dir| root.join(dir))
}

fn run_valgrind_on_example(
    root: &Path,
    example_name: &str,
) -> ExampleRunResult {
    let binary_path =
        target_dir(root).join("debug/examples").join(example_name);
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

/// Runs Valgrind on every known example, returning pass count and failures.
fn check_examples(root: &Path) -> CheckResult {
    let mut failed: Vec<FailureRecord> = Vec::new();
    let mut passed = 0_usize;

    for &example in KNOWN_EXAMPLES {
        ui::status("Running", format!("Valgrind Memcheck on '{example}'"));
        match run_valgrind_on_example(root, example) {
            Ok((status, stderr)) => {
                if status.success() {
                    ui::status(
                        "Passed",
                        format!("'{example}' clean (0 leaks, 0 errors)"),
                    );
                    passed = passed.saturating_add(1);
                } else {
                    ui::failure(
                        "Failed",
                        format!(
                            "'{example}' reported memory errors or leaks \
                             (exit code {:?})",
                            status.code()
                        ),
                    );
                    failed.push((example.to_string(), stderr));
                }
            }
            Err(e) => {
                ui::failure(
                    "Error",
                    format!("'{example}' execution error: {e}"),
                );
                failed.push((example.to_string(), e));
            }
        }
    }

    (passed, failed)
}

/// Prints the final verdict and exits with the appropriate code.
fn report(passed: usize, failed: &[FailureRecord]) -> ! {
    if failed.is_empty() {
        ui::status(
            "Finished",
            format!(
                "Valgrind Memcheck passed across all \
                 {passed} examples (0 leaks, 0 errors)"
            ),
        );
        std::process::exit(0);
    }
    ui::failure(
        "Failed",
        format!(
            "Valgrind Memcheck on {}/{} examples",
            failed.len(),
            KNOWN_EXAMPLES.len()
        ),
    );
    for (name, log) in failed {
        ui::error(format!("'{name}' failure log:\n{log}"));
    }
    std::process::exit(1);
}

fn main() {
    if !is_valgrind_available() {
        ui::warn_diag(
            "'valgrind' is not installed or supported natively on this host \
             (e.g., macOS Apple Silicon). \
             Skipping memory checks (degraded). \
             Valgrind is enforced on Linux CI environments.",
        );
        std::process::exit(78);
    }

    let root = find_workspace_root();
    ui::status_info("Workspace", root.display());

    ui::status("Building", "all workspace examples");
    if let Err(e) = build_examples(&root) {
        ui::error(format!("compiling examples: {e}"));
        std::process::exit(1);
    }

    let (passed, failed) = check_examples(&root);
    report(passed, &failed);
}
