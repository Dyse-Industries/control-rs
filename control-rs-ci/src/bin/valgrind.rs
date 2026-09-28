//! Multi-example Valgrind Memcheck runner (`cargo valgrind`).
//!
//! Builds the examples of the package in the current directory and runs
//! Valgrind Memcheck on each example named by `--example`, failing on any
//! leak or invalid access.
//!
//! ```text
//! cargo valgrind --example dc_motor --example buck_converter
//! ```
//!
//! Fails closed: without a `valgrind` executable (for example on macOS) it
//! exits 1, so a host without Valgrind skips the gate explicitly with
//! `cargo ci --skip valgrind`. No example named is a usage error (exit 2).

use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};

use control_rs_ci::cli::{USAGE_EXIT, arg_spelling};
use control_rs_ci::ui;

/// Example names from the command line.
type ExampleNames = Vec<String>;
type ExampleRunResult = Result<(ExitStatus, String), String>;
type FailureRecord = (String, String);
type CheckResult = (usize, Vec<FailureRecord>);

fn print_usage() {
    ui::init_color();
    let h = ui::HELP_HEADER;
    let f = ui::HELP_FLAG;
    let a = ui::HELP_ARG;
    anstream::println!(
        "{h}Usage:{h:#} {f}cargo valgrind{f:#} {f}--example{f:#} {a}<name>{a:#} [{f}--example{f:#} {a}<name>{a:#}...]\n\n\
         {h}Options:{h:#}\n  \
           {f}--example{f:#} {a}<name>{a:#}    Example target to run under Valgrind (repeatable)\n  \
           {f}-h{f:#}, {f}--help{f:#}           Print help information"
    );
}

/// Example names from `--example <name>` pairs; any other argument is a
/// usage error.
fn parse_examples(args: &[String]) -> Result<ExampleNames, String> {
    use lexopt::prelude::*;
    let mut parser = lexopt::Parser::from_iter(
        std::iter::once(String::new()).chain(args.iter().cloned()),
    );
    let mut examples = Vec::new();
    while let Some(arg) = parser.next().map_err(|e| e.to_string())? {
        match arg {
            Long("example") => {
                let name = parser
                    .value()
                    .map_err(|_| "--example requires a name".to_string())?
                    .string()
                    .map_err(|e| format!("Invalid UTF-8: {e:?}"))?;
                examples.push(name);
            }
            Short('h') | Long("help") => {
                print_usage();
                std::process::exit(0);
            }
            _ => {
                return Err(format!(
                    "unknown argument '{}'",
                    arg_spelling(&arg)
                ));
            }
        }
    }
    if examples.is_empty() {
        return Err("no example named; nothing would be checked".to_string());
    }
    Ok(examples)
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

/// Runs Valgrind on every named example, returning pass count and failures.
fn check_examples(root: &Path, examples: &[String]) -> CheckResult {
    let mut failed: Vec<FailureRecord> = Vec::new();
    let mut passed = 0_usize;

    for example in examples {
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
                    failed.push((example.clone(), stderr));
                }
            }
            Err(e) => {
                ui::failure(
                    "Error",
                    format!("'{example}' execution error: {e}"),
                );
                failed.push((example.clone(), e));
            }
        }
    }

    (passed, failed)
}

/// Prints the final verdict and exits with the appropriate code.
fn report(passed: usize, failed: &[FailureRecord], total: usize) -> ! {
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
        format!("Valgrind Memcheck on {}/{} examples", failed.len(), total),
    );
    for (name, log) in failed {
        ui::error(format!("'{name}' failure log:\n{log}"));
    }
    std::process::exit(1);
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let examples = parse_examples(&args).unwrap_or_else(|e| {
        ui::error(format!("{e}; usage: cargo valgrind --example <name>..."));
        std::process::exit(USAGE_EXIT);
    });
    if !is_valgrind_available() {
        ui::error(
            "'valgrind' is not installed or not supported on this host; \
             skip the gate explicitly with `cargo ci --skip valgrind`",
        );
        std::process::exit(1);
    }

    let root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    ui::status_info("Workspace", root.display());

    ui::status("Building", "workspace examples");
    if let Err(e) = build_examples(&root) {
        ui::error(format!("compiling examples: {e}"));
        std::process::exit(1);
    }

    let (passed, failed) = check_examples(&root, &examples);
    report(passed, &failed, examples.len());
}

#[cfg(test)]
mod tests {
    use super::parse_examples;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(ToString::to_string).collect()
    }

    #[test]
    fn examples_come_from_repeated_flags() {
        assert_eq!(
            parse_examples(&args(&["--example", "a", "--example", "b"])),
            Ok(vec!["a".to_string(), "b".to_string()])
        );
    }

    #[test]
    fn empty_or_malformed_arguments_are_rejected() {
        assert!(parse_examples(&[]).is_err());
        assert!(parse_examples(&args(&["--example"])).is_err());
        assert!(parse_examples(&args(&["dc_motor"])).is_err());
    }
}
