//! Standalone multi-example cross-comparison and oracle verification tool (`cargo compare` / `compare`).

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::exit;
use std::time::Instant;

use control_rs_ci::gates;
use control_rs_ci::report;
use control_rs_ci::validate::{self, ExampleTargetConfig};

struct CompareArgs {
    config_path: Option<PathBuf>,
    name: Option<String>,
    timeout_secs: Option<u64>,
    out_dir: PathBuf,
    repo_root: PathBuf,
}

impl Default for CompareArgs {
    fn default() -> Self {
        Self {
            config_path: None,
            name: None,
            timeout_secs: None,
            out_dir: PathBuf::from("."),
            repo_root: PathBuf::from("."),
        }
    }
}

fn print_help() {
    println!(
        "Usage: cargo compare [OPTIONS]\n       compare [OPTIONS]\n\n\
        Options:\n  \
          --config <PATH>        Path to validate.toml [default: validate.toml]\n  \
          --name <NAME>          Run only the named suite\n  \
          --timeout <SECS>       Per-command execution timeout in seconds\n  \
          --out-dir <DIR>        Directory for report output [default: .]\n  \
          --repo-root <PATH>     Repository root directory [default: .]\n  \
          -h, --help             Print help"
    );
}

fn handle_flag(
    arg: &str,
    iter: &mut impl Iterator<Item = String>,
    args: &mut CompareArgs,
) {
    match arg {
        "-h" | "--help" => {
            print_help();
            exit(0);
        }
        "--config" => {
            args.config_path = iter.next().map(PathBuf::from);
        }
        "--name" => args.name = iter.next(),
        "--timeout" => {
            if let Some(Ok(t)) = iter.next().map(|v| v.parse::<u64>()) {
                args.timeout_secs = Some(t);
            }
        }
        "--out-dir" => {
            if let Some(val) = iter.next() {
                args.out_dir = PathBuf::from(val);
            }
        }
        "--repo-root" => {
            if let Some(val) = iter.next() {
                args.repo_root = PathBuf::from(val);
            }
        }
        "--" => {}
        other => {
            report::error(format!("Unknown argument: {other}"));
            print_help();
            exit(1);
        }
    }
}

fn parse_args() -> CompareArgs {
    let mut args = CompareArgs::default();
    let mut raw_args = env::args().skip(1).peekable();
    if raw_args.peek().map(String::as_str) == Some("compare") {
        raw_args.next();
    }
    while let Some(arg) = raw_args.next() {
        handle_flag(&arg, &mut raw_args, &mut args);
    }
    args
}

fn resolve_suite_dirs(
    config_path: &Path,
    mut suites: Vec<ExampleTargetConfig>,
) -> Vec<ExampleTargetConfig> {
    let config_dir = config_path.parent().unwrap_or_else(|| Path::new("."));
    for target in &mut suites {
        let raw = Path::new(&target.manifest_path);
        if raw.is_relative() {
            target.manifest_path =
                config_dir.join(raw).to_string_lossy().to_string();
        }
    }
    suites
}

fn main() {
    report::init_color();
    let start_time = Instant::now();
    let args = parse_args();

    let config_path = args
        .config_path
        .clone()
        .unwrap_or_else(|| PathBuf::from("validate.toml"));
    let Some(cfg) = validate::load_validate_config(&config_path) else {
        report::error(format!(
            "no usable config at {}; declare `bin` and `oracle` in validate.toml",
            config_path.display()
        ));
        exit(1);
    };

    let timeout = args.timeout_secs.unwrap_or(cfg.validate.timeout_secs);
    let mut targets = resolve_suite_dirs(&config_path, cfg.suites);
    if let Some(name) = args.name.as_deref() {
        targets.retain(|s| s.name == name);
    }
    if targets.is_empty() {
        report::error(
            "no suites configured; declare `bin` and `oracle` in validate.toml",
        );
        exit(1);
    }

    let (passed, summary, brief) =
        gates::run_cross_comparison(&targets, &args.repo_root, timeout);

    if let Err(e) = fs::create_dir_all(&args.out_dir) {
        report::error(format!(
            "Failed to create output directory '{}': {e}",
            args.out_dir.display()
        ));
    }

    let json_path = args.out_dir.join("cross-val-report.json");
    if let Ok(json) = serde_json::to_string_pretty(&summary) {
        let _ = fs::write(&json_path, json);
        report::status(
            "Saved",
            format!("cross-val report to {}", json_path.display()),
        );
    }

    let md_path = args.out_dir.join("cross-val-report.md");
    let _ = fs::write(&md_path, &brief);

    let elapsed = report::format_elapsed(start_time.elapsed());
    if passed {
        report::status(
            "Finished",
            format!("cross-validation passed in {elapsed}"),
        );
        exit(0);
    } else {
        report::error(format!(
            "cross-validation failed: {} failed examples in {elapsed}",
            summary.failed_examples
        ));
        exit(1);
    }
}
