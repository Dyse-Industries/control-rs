//! Clippy suppression audit (`cargo run -p control-rs-ci --bin allow-audit`).
//!
//! Fails when the workspace suppresses a clippy lint that the baseline does
//! not list. `--write` regenerates the baseline from the current tree, which
//! is only legitimate after suppressions were removed.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use control_rs_ci::allow_audit::{
    AuditDiff, Suppressions, collect, diff, parse_baseline, render_baseline,
};
use control_rs_ci::ui;

/// Command-line options.
struct AuditArgs {
    /// Workspace root to scan.
    root: PathBuf,
    /// Baseline file of permitted legacy suppressions.
    baseline: PathBuf,
    /// Regenerate the baseline instead of checking against it.
    write: bool,
}

/// Parses `--root <dir>`, `--baseline <file>` and `--write`.
fn parse_args() -> Result<AuditArgs, String> {
    let mut parsed = AuditArgs {
        root: PathBuf::from("."),
        baseline: PathBuf::from("clippy-allow-baseline.txt"),
        write: false,
    };
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--root" => {
                parsed.root = args
                    .next()
                    .map(PathBuf::from)
                    .ok_or("--root requires a directory")?;
            }
            "--baseline" => {
                parsed.baseline = args
                    .next()
                    .map(PathBuf::from)
                    .ok_or("--baseline requires a file")?;
            }
            "--write" => parsed.write = true,
            other => return Err(format!("Unknown argument: {other}")),
        }
    }
    Ok(parsed)
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            ui::error(e);
            return ExitCode::from(2);
        }
    };

    let found = match collect(&args.root) {
        Ok(found) => found,
        Err(e) => {
            ui::error(format!("Failed to scan {}: {e}", args.root.display()));
            return ExitCode::from(2);
        }
    };

    if args.write {
        return write_baseline(&args, &found);
    }

    let baseline = match fs::read_to_string(&args.baseline) {
        Ok(text) => parse_baseline(&text),
        Err(e) => {
            ui::error(format!(
                "Failed to read {}: {e}",
                args.baseline.display()
            ));
            return ExitCode::from(2);
        }
    };

    report(&diff(&found, &baseline), found.len())
}

/// Writes the current suppressions as the new baseline.
fn write_baseline(args: &AuditArgs, found: &Suppressions) -> ExitCode {
    match fs::write(&args.baseline, render_baseline(found)) {
        Ok(()) => {
            ui::status(
                "Writing",
                format!(
                    "{} ({} suppressions)",
                    args.baseline.display(),
                    found.len()
                ),
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            ui::error(format!(
                "Failed to write {}: {e}",
                args.baseline.display()
            ));
            ExitCode::from(2)
        }
    }
}

/// Prints removed and new suppressions; fails on any new one.
fn report(result: &AuditDiff, total: usize) -> ExitCode {
    for entry in &result.stale {
        ui::status_info("Removed", entry);
    }
    if !result.stale.is_empty() {
        ui::warn_diag(format!(
            "{} baseline entries are gone; shrink the baseline with --write",
            result.stale.len()
        ));
    }
    if result.added.is_empty() {
        ui::status(
            "Finished",
            format!("{total} legacy suppressions, none new"),
        );
        return ExitCode::SUCCESS;
    }
    for entry in &result.added {
        ui::failure("New", entry);
    }
    ui::error(format!(
        "{} new clippy suppressions; fix the code instead of allowing the lint",
        result.added.len()
    ));
    ExitCode::FAILURE
}
