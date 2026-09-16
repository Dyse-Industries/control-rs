//! Logic for the standalone requirement traceability tool (`cargo trace` /
//! `trace`). `bin/trace.rs` is a thin shell over [`main_impl`].

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use crate::{gates, report};

type ParseArgsResult = Result<Option<TraceArgs>, String>;

struct TraceArgs {
    repo_root: PathBuf,
    test_output_path: Option<PathBuf>,
    ets_json_path: Option<PathBuf>,
    out_dir: PathBuf,
}

impl Default for TraceArgs {
    fn default() -> Self {
        Self {
            repo_root: PathBuf::from("."),
            test_output_path: None,
            ets_json_path: None,
            out_dir: PathBuf::from("."),
        }
    }
}

fn print_help() {
    println!(
        "Usage: cargo trace [OPTIONS]\n       trace [OPTIONS]\n\n\
        Options:\n  \
          --repo-root <PATH>     Repository root directory [default: .]\n  \
          --test-output <PATH>   Path to captured cargo test output file\n  \
          --ets-json <PATH>      Path to captured ETS results JSON file\n  \
          --out-dir <DIR>        Directory for trace report output [default: .]\n  \
          --json                 Accepted; JSON is always written to trace-report.json\n  \
          -h, --help             Print help"
    );
}

fn handle_flag(
    arg: &str,
    iter: &mut impl Iterator<Item = String>,
    args: &mut TraceArgs,
) -> Result<(), String> {
    match arg {
        "--repo-root" => {
            if let Some(val) = iter.next() {
                args.repo_root = PathBuf::from(val);
            }
        }
        "--test-output" => {
            args.test_output_path = iter.next().map(PathBuf::from);
        }
        "--ets-json" => {
            args.ets_json_path = iter.next().map(PathBuf::from);
        }
        "--out-dir" => {
            if let Some(val) = iter.next() {
                args.out_dir = PathBuf::from(val);
            }
        }
        "--json" | "--" => {}
        other => return Err(format!("Unknown argument: {other}")),
    }
    Ok(())
}

fn parse_args(args: &[String]) -> ParseArgsResult {
    let mut parsed = TraceArgs::default();
    let mut raw_args = args.iter().skip(1).cloned().peekable();
    if raw_args.peek().map(String::as_str) == Some("trace") {
        raw_args.next();
    }
    while let Some(arg) = raw_args.next() {
        if matches!(arg.as_str(), "-h" | "--help") {
            print_help();
            return Ok(None);
        }
        handle_flag(&arg, &mut raw_args, &mut parsed)?;
    }
    Ok(Some(parsed))
}

fn write_reports(
    out_dir: &std::path::Path,
    summary: &crate::trace::TraceMatrixSummary,
    brief: &str,
) {
    if let Err(e) = fs::create_dir_all(out_dir) {
        report::error(format!(
            "Failed to create output directory '{}': {e}",
            out_dir.display()
        ));
    }

    let json_path = out_dir.join("trace-report.json");
    if let Ok(json) = serde_json::to_string_pretty(summary) {
        let _ = fs::write(&json_path, json);
        report::status(
            "Saved",
            format!("JSON trace report to {}", json_path.display()),
        );
    }

    let md_path = out_dir.join("trace-report.md");
    let _ = fs::write(&md_path, brief);
}

/// Parses `args`, runs the requirement-traceability analysis, writes its
/// reports, and returns the process exit code. Never exits the process
/// itself, so it can be driven from a test.
#[must_use]
pub fn main_impl(args: &[String]) -> i32 {
    report::init_color();
    let start_time = Instant::now();
    let cli_args = match parse_args(args) {
        Ok(Some(cli_args)) => cli_args,
        Ok(None) => return 0,
        Err(e) => {
            report::error(e);
            print_help();
            return 1;
        }
    };

    let test_output = cli_args
        .test_output_path
        .and_then(|p| fs::read_to_string(p).ok());
    let ets_json = cli_args
        .ets_json_path
        .and_then(|p| fs::read_to_string(p).ok());

    let (passed, summary, brief) = gates::run_traceability(
        &cli_args.repo_root,
        test_output.as_deref(),
        ets_json.as_deref(),
    );
    write_reports(&cli_args.out_dir, &summary, &brief);

    let elapsed = report::format_elapsed(start_time.elapsed());
    if passed {
        report::status(
            "Finished",
            format!("traceability check passed in {elapsed}"),
        );
        0
    } else {
        report::error(format!(
            "traceability check failed: {} missing, {} unresolved in {elapsed}",
            summary.approved_missing_count, summary.approved_unresolved_count
        ));
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_args_help_returns_none() {
        assert!(
            parse_args(&["trace".to_string(), "--help".to_string()])
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn test_parse_args_unknown_flag_errors() {
        assert!(
            parse_args(&["trace".to_string(), "--bogus".to_string()]).is_err()
        );
    }

    #[test]
    fn test_parse_args_json_is_accepted() {
        let parsed = parse_args(&[
            "trace".to_string(),
            "--json".to_string(),
            "--out-dir".to_string(),
            "target/ci".to_string(),
        ])
        .unwrap()
        .unwrap();
        assert_eq!(parsed.out_dir, PathBuf::from("target/ci"));
    }

    #[test]
    fn test_parse_args_defaults() {
        let parsed = parse_args(&["trace".to_string()]).unwrap().unwrap();
        assert_eq!(parsed.repo_root, PathBuf::from("."));
        assert!(parsed.test_output_path.is_none());
        assert!(parsed.ets_json_path.is_none());
        assert_eq!(parsed.out_dir, PathBuf::from("."));
    }

    #[test]
    fn test_parse_args_overrides() {
        let parsed = parse_args(&[
            "trace".to_string(),
            "--repo-root".to_string(),
            "/repo".to_string(),
            "--test-output".to_string(),
            "test.log".to_string(),
            "--ets-json".to_string(),
            "ets.json".to_string(),
            "--out-dir".to_string(),
            "target/ci".to_string(),
        ])
        .unwrap()
        .unwrap();
        assert_eq!(parsed.repo_root, PathBuf::from("/repo"));
        assert_eq!(parsed.test_output_path, Some(PathBuf::from("test.log")));
        assert_eq!(parsed.ets_json_path, Some(PathBuf::from("ets.json")));
        assert_eq!(parsed.out_dir, PathBuf::from("target/ci"));
    }

    #[test]
    fn test_main_impl_help_returns_zero() {
        assert_eq!(main_impl(&["trace".to_string(), "--help".to_string()]), 0);
    }

    #[test]
    fn test_main_impl_unknown_flag_returns_one() {
        assert_eq!(main_impl(&["trace".to_string(), "--nope".to_string()]), 1);
    }

    #[test]
    fn test_main_impl_runs_against_empty_repo_root_and_writes_reports() {
        let temp_dir =
            std::env::temp_dir().join("control_rs_ci_test_trace_cli_main_impl");
        let _ = fs::remove_dir_all(&temp_dir);
        fs::create_dir_all(&temp_dir).unwrap();

        let code = main_impl(&[
            "trace".to_string(),
            "--repo-root".to_string(),
            temp_dir.to_string_lossy().to_string(),
            "--out-dir".to_string(),
            temp_dir.to_string_lossy().to_string(),
        ]);
        // No requirement documents in an empty fixture: the gate reports
        // silence, which `gates::run_traceability` treats as a failure.
        assert_eq!(code, 1);
        assert!(temp_dir.join("trace-report.json").exists());
        assert!(temp_dir.join("trace-report.md").exists());

        let _ = fs::remove_dir_all(&temp_dir);
    }
}
