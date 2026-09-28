//! Clippy suppression audit (`cargo run -p control-rs-ci --bin allow-audit`).
//!
//! Fails when the workspace has more suppression sites of a clippy lint in a
//! file than the baseline allows, and when the baseline lists more sites than
//! remain (a stale entry that `--write` must shrink). With `--base-ref <ref>`,
//! it also fails when the baseline grew relative to the baseline at `<ref>`,
//! so `--write` cannot legitimize a new suppression. `--write` regenerates the
//! baseline from the current tree.

use std::fs;
use std::path::PathBuf;
use std::process::{Command, ExitCode};

use control_rs_ci::allow_audit::{
    AuditDiff, Suppressions, collect, diff, growth, parse_baseline,
    render_baseline,
};
use control_rs_ci::cli::arg_spelling;
use control_rs_ci::ui;

/// Parsed options, `None` for `--help`, or a usage error.
type ParsedAudit = Result<Option<AuditArgs>, String>;

/// Command-line options.
struct AuditArgs {
    /// Workspace root to scan.
    root: PathBuf,
    /// Baseline file of permitted legacy suppressions.
    baseline: PathBuf,
    /// Regenerate the baseline instead of checking against it.
    write: bool,
    /// Git ref whose baseline the current baseline must not exceed.
    base_ref: Option<String>,
}

fn print_usage() {
    ui::init_color();
    let h = ui::HELP_HEADER;
    let f = ui::HELP_FLAG;
    let a = ui::HELP_ARG;
    anstream::println!(
        "{h}Usage:{h:#} {f}allow-audit{f:#} {a}[OPTIONS]{a:#}\n\n\
         {h}Options:{h:#}\n  \
           {f}--root{f:#} {a}<dir>{a:#}           Workspace root to scan [default: .]\n  \
           {f}--baseline{f:#} {a}<file>{a:#}      Baseline file [default: .cargo/clippy-allow-baseline.txt]\n  \
           {f}--write{f:#}                Regenerate the baseline instead of checking\n  \
           {f}--base-ref{f:#} {a}<ref>{a:#}       Git ref whose baseline current must not exceed\n  \
           {f}-h{f:#}, {f}--help{f:#}             Print help information"
    );
}

/// Parses `--root <dir>`, `--baseline <file>`, `--base-ref <ref>` and
/// `--write` (program name first); `Ok(None)` means `--help`.
fn parse_args(args: impl IntoIterator<Item = String>) -> ParsedAudit {
    use lexopt::prelude::*;
    let mut parser = lexopt::Parser::from_iter(args);
    let mut parsed = AuditArgs {
        root: PathBuf::from("."),
        baseline: PathBuf::from(".cargo/clippy-allow-baseline.txt"),
        write: false,
        base_ref: None,
    };
    while let Some(arg) = parser.next().map_err(|e| e.to_string())? {
        match arg {
            Long("root") => {
                let val = parser
                    .value()
                    .map_err(|_| "--root requires a directory".to_string())?;
                parsed.root = PathBuf::from(val);
            }
            Long("baseline") => {
                let val = parser
                    .value()
                    .map_err(|_| "--baseline requires a file".to_string())?;
                parsed.baseline = PathBuf::from(val);
            }
            Long("write") => parsed.write = true,
            Long("base-ref") => {
                let val = parser
                    .value()
                    .map_err(|_| "--base-ref requires a git ref".to_string())?
                    .string()
                    .map_err(|e| format!("Invalid UTF-8: {e:?}"))?;
                parsed.base_ref = Some(val);
            }
            Short('h') | Long("help") => return Ok(None),
            _ => {
                return Err(format!(
                    "Unknown argument: {}",
                    arg_spelling(&arg)
                ));
            }
        }
    }
    Ok(Some(parsed))
}

fn main() -> ExitCode {
    let args = match parse_args(std::env::args()) {
        Ok(Some(a)) => a,
        Ok(None) => {
            print_usage();
            return ExitCode::SUCCESS;
        }
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

    let mut failed = report(&diff(&found, &baseline), found.len());
    if let Some(base_ref) = &args.base_ref {
        match baseline_at(&args, base_ref) {
            Ok(base) => {
                failed |= report_growth(&growth(&baseline, &base), base_ref);
            }
            Err(e) => {
                ui::error(e);
                return ExitCode::from(2);
            }
        }
    }
    if failed {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

/// The baseline file as committed at `base_ref`, read with `git show`.
fn baseline_at(
    args: &AuditArgs,
    base_ref: &str,
) -> Result<Suppressions, String> {
    let path = args.baseline.to_string_lossy().replace('\\', "/");
    let path = path.strip_prefix("./").unwrap_or(&path);
    let out = Command::new("git")
        .current_dir(&args.root)
        .args(["show", &format!("{base_ref}:{path}")])
        .output()
        .map_err(|e| format!("Failed to run git show: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "Cannot read {path} at {base_ref}: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(parse_baseline(&String::from_utf8_lossy(&out.stdout)))
}

/// Prints baseline entries that grew relative to `base_ref`; returns whether
/// any did.
fn report_growth(grown: &[String], base_ref: &str) -> bool {
    for entry in grown {
        ui::failure("Grown", entry);
    }
    if grown.is_empty() {
        return false;
    }
    ui::error(format!(
        "{} baseline entries exceed {base_ref}; the baseline may only shrink",
        grown.len()
    ));
    true
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

/// Prints new and stale suppressions; returns whether either exists.
fn report(result: &AuditDiff, total: usize) -> bool {
    for entry in &result.stale {
        ui::failure("Stale", entry);
    }
    if !result.stale.is_empty() {
        ui::error(format!(
            "{} baseline entries list more sites than remain; shrink the baseline with --write",
            result.stale.len()
        ));
    }
    for entry in &result.added {
        ui::failure("New", entry);
    }
    if !result.added.is_empty() {
        ui::error(format!(
            "{} new clippy suppressions; fix the code instead of allowing the lint",
            result.added.len()
        ));
    }
    let failed = !result.added.is_empty() || !result.stale.is_empty();
    if !failed {
        ui::status(
            "Finished",
            format!("{total} legacy suppressions, none new, none stale"),
        );
    }
    failed
}

#[cfg(test)]
mod tests {
    use super::{ParsedAudit, parse_args};

    fn parse(list: &[&str]) -> ParsedAudit {
        parse_args(
            std::iter::once("allow-audit")
                .chain(list.iter().copied())
                .map(ToString::to_string),
        )
    }

    #[test]
    fn options_parse_with_defaults() {
        let parsed = parse(&[]).unwrap().unwrap();
        assert_eq!(parsed.root.to_str(), Some("."));
        assert!(!parsed.write && parsed.base_ref.is_none());
        let parsed = parse(&[
            "--root=r",
            "--baseline",
            "b.txt",
            "--write",
            "--base-ref",
            "origin/main",
        ])
        .unwrap()
        .unwrap();
        assert_eq!(parsed.root.to_str(), Some("r"));
        assert_eq!(parsed.baseline.to_str(), Some("b.txt"));
        assert!(parsed.write);
        assert_eq!(parsed.base_ref.as_deref(), Some("origin/main"));
    }

    #[test]
    fn help_and_usage_errors() {
        assert!(parse(&["-h"]).unwrap().is_none());
        assert_eq!(
            parse(&["--bogus"]).err().unwrap(),
            "Unknown argument: --bogus"
        );
        assert_eq!(parse(&["x"]).err().unwrap(), "Unknown argument: x");
        assert_eq!(
            parse(&["--base-ref"]).err().unwrap(),
            "--base-ref requires a git ref"
        );
    }
}
