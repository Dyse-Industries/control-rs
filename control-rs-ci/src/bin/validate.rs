//! Standalone multi-example validation and oracle verification tool (`cargo validate` / `validate`).

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::exit;
use std::time::Instant;

use control_rs_ci::report;
use control_rs_ci::validate::{self, ExampleTargetConfig, ValidateConfigFile};

struct ValidateArgs {
    config_path: Option<PathBuf>,
    manifest_path: String,
    examples: Vec<String>,
    timeout_secs: Option<u64>,
    out_dir: Option<PathBuf>,
    repo_root: PathBuf,
    emit_json: bool,
    emit_markdown: bool,
    strict: Option<bool>,
    quiet: bool,
}

/// Resolved validation plan: targets, timeout, output directory and coverage flag.
type ValidationPlan = (Vec<ExampleTargetConfig>, u64, PathBuf, bool);

impl Default for ValidateArgs {
    fn default() -> Self {
        Self {
            config_path: None,
            manifest_path: ".".to_string(),
            examples: Vec::new(),
            timeout_secs: None,
            out_dir: None,
            repo_root: PathBuf::from("."),
            emit_json: true,
            emit_markdown: true,
            strict: None,
            quiet: false,
        }
    }
}

fn print_help() {
    report::init_color();
    let h = report::HELP_HEADER;
    let f = report::HELP_FLAG;
    let a = report::HELP_ARG;
    anstream::println!(
        "{h}Usage:{h:#} {f}cargo validate{f:#} {a}[OPTIONS]{a:#}\n       {f}validate{f:#} {a}[OPTIONS]{a:#}\n\n\
        {h}Options:{h:#}\n  \
          {f}--config{f:#} {a}<PATH>{a:#}              Path to configuration file [default: validate.toml]\n  \
          {f}--manifest-path{f:#} {a}<PATH>{a:#}        Path to Cargo.toml or crate directory [default: .]\n  \
          {f}--example{f:#}, {f}--name{f:#} {a}<NAME>{a:#}      Specific example to validate (repeatable)\n  \
          {f}--timeout{f:#} {a}<SECS>{a:#}              Execution timeout in seconds [default: 90]\n  \
          {f}--out-dir{f:#} {a}<DIR>{a:#}               Directory for report output [default: .]\n  \
          {f}--repo-root{f:#} {a}<PATH>{a:#}            Repository root directory [default: .]\n  \
          {f}--json{f:#}                        Emit validate-report.json [default: true]\n  \
          {f}--markdown{f:#}                    Emit validate-report.md [default: true]\n  \
          {f}--strict{f:#}                      Exit non-zero on any tolerance breach [default: true]\n  \
          {f}--bypass-gate{f:#}                 Do not exit non-zero on tolerance breaches\n  \
          {f}--quiet{f:#}                       Capture subprocess output without streaming it to the terminal\n  \
          {f}-h{f:#}, {f}--help{f:#}                    Print help"
    );
}

fn handle_flag(
    arg: &str,
    iter: &mut impl Iterator<Item = String>,
    args: &mut ValidateArgs,
) {
    match arg {
        "-h" | "--help" => {
            print_help();
            exit(0);
        }
        "--config" => {
            args.config_path = iter.next().map(PathBuf::from);
        }
        "--manifest-path" | "--path" => {
            if let Some(val) = iter.next() {
                args.manifest_path.clone_from(&val);
            }
        }
        "--example" | "--name" => {
            if let Some(val) = iter.next() {
                args.examples.push(val);
            }
        }
        "--timeout" => {
            if let Some(Ok(t)) = iter.next().map(|v| v.parse::<u64>()) {
                args.timeout_secs = Some(t);
            }
        }
        "--out-dir" => {
            if let Some(val) = iter.next() {
                args.out_dir = Some(PathBuf::from(val));
            }
        }
        "--repo-root" => {
            if let Some(val) = iter.next() {
                args.repo_root = PathBuf::from(val);
            }
        }
        "--json" => args.emit_json = true,
        "--no-json" => args.emit_json = false,
        "--markdown" => args.emit_markdown = true,
        "--no-markdown" => args.emit_markdown = false,
        "--strict" => args.strict = Some(true),
        "--no-strict" | "--bypass-gate" => args.strict = Some(false),
        "--quiet" => args.quiet = true,
        "--" => {}
        other => {
            report::error(format!("Unknown argument: {other}"));
            print_help();
            exit(1);
        }
    }
}

fn parse_args() -> ValidateArgs {
    let mut args = ValidateArgs::default();
    let mut raw_args = env::args().skip(1).peekable();
    if raw_args.peek().map(String::as_str) == Some("validate") {
        raw_args.next();
    }
    while let Some(arg) = raw_args.next() {
        handle_flag(&arg, &mut raw_args, &mut args);
    }
    args
}

fn resolve_config_file(args: &ValidateArgs) -> Option<ValidateConfigFile> {
    let candidate = args.config_path.clone().unwrap_or_else(|| {
        let default_path = PathBuf::from("validate.toml");
        if default_path.exists() {
            default_path
        } else {
            PathBuf::from("ci.toml")
        }
    });

    if candidate.exists()
        && let Ok(content) = fs::read_to_string(&candidate)
        && let Ok(cfg) = toml::from_str::<ValidateConfigFile>(&content)
    {
        return Some(cfg);
    }
    None
}

/// Resolve each `--example <name>` against the suites declared in
/// `validate.toml`, falling back to treating the argument as a path.
///
/// The gate assumes no directory layout for the crates it validates
/// (ci-design.md C-9), so "examples/" appears nowhere here.
fn selected_suites(
    args: &ValidateArgs,
    config: Option<&ValidateConfigFile>,
    timeout: u64,
) -> Vec<ExampleTargetConfig> {
    let declared = config.map(|c| c.suites.as_slice()).unwrap_or_default();
    args.examples
        .iter()
        .map(|name| {
            if let Some(suite) = declared
                .iter()
                .find(|s| s.name == *name || s.manifest_path == *name)
            {
                let mut selected = suite.clone();
                selected.timeout_secs = selected.timeout_secs.or(Some(timeout));
                return selected;
            }
            ExampleTargetConfig {
                name: name.clone(),
                manifest_path: if Path::new(name).is_dir() {
                    name.clone()
                } else {
                    args.manifest_path.clone()
                },
                bin: String::new(),
                oracle: String::new(),
                commands: Vec::new(),
                timeout_secs: Some(timeout),
            }
        })
        .collect()
}

fn build_targets(
    args: &ValidateArgs,
    config: Option<&ValidateConfigFile>,
) -> ValidationPlan {
    let default_timeout = config.map_or(90, |c| c.validate.timeout_secs);
    let timeout = args.timeout_secs.unwrap_or(default_timeout);

    let default_out_dir = config.map_or_else(
        || PathBuf::from("target/ci"),
        |c| PathBuf::from(&c.validate.out_dir),
    );
    let out_dir = args.out_dir.clone().unwrap_or(default_out_dir);

    let default_strict = config.is_none_or(|c| c.validate.strict);
    let strict = args.strict.unwrap_or(default_strict);

    if !args.examples.is_empty() {
        (
            selected_suites(args, config, timeout),
            timeout,
            out_dir,
            strict,
        )
    } else if let Some(cfg) = config
        && !cfg.suites.is_empty()
    {
        (cfg.suites.clone(), timeout, out_dir, strict)
    } else {
        let manifest_p = Path::new(&args.manifest_path);
        let target_name = manifest_p.file_name().map_or_else(
            || "example".to_string(),
            |s| s.to_string_lossy().to_string(),
        );
        let targets = vec![ExampleTargetConfig {
            name: target_name,
            manifest_path: args.manifest_path.clone(),
            bin: String::new(),
            oracle: String::new(),
            commands: Vec::new(),
            timeout_secs: Some(timeout),
        }];
        (targets, timeout, out_dir, strict)
    }
}

fn emit_reports(
    args: &ValidateArgs,
    out_dir: &Path,
    summary: &control_rs_ci::validate::CrossComparisonSummary,
    brief: &str,
) {
    if let Err(e) = fs::create_dir_all(out_dir) {
        report::error(format!(
            "Failed to create output directory '{}': {e}",
            out_dir.display()
        ));
    }

    if args.emit_json {
        let json_path = out_dir.join("validate-report.json");
        if let Ok(json) = serde_json::to_string_pretty(&summary) {
            let _ = fs::write(&json_path, json);
            report::status(
                "Saved",
                format!("validation report to {}", json_path.display()),
            );
        }
        // Also write legacy cross-val-report.json for backward compatibility
        let legacy_json = out_dir.join("cross-val-report.json");
        if let Ok(json) = serde_json::to_string_pretty(&summary) {
            let _ = fs::write(&legacy_json, json);
        }
    }

    if args.emit_markdown {
        let md_path = out_dir.join("validate-report.md");
        let _ = fs::write(&md_path, brief);
        let legacy_md = out_dir.join("cross-val-report.md");
        let _ = fs::write(&legacy_md, brief);
    }
}

fn main() {
    report::init_color();
    let start_time = Instant::now();
    let args = parse_args();
    control_rs_ci::gates::set_capture_only(args.quiet);

    let config = resolve_config_file(&args);
    let (targets, timeout_secs, out_dir, strict) =
        build_targets(&args, config.as_ref());

    report::status("Running", "`cargo validate`");
    let summary =
        validate::run_cross_comparison(&targets, &args.repo_root, timeout_secs);
    let passed = summary.failed_examples == 0;
    let brief = validate::render_cross_comparison_brief(&summary);

    emit_reports(&args, &out_dir, &summary, &brief);

    let elapsed = report::format_elapsed(start_time.elapsed());
    if passed || !strict {
        report::status("Finished", format!("validation passed in {elapsed}"));
        exit(0);
    } else {
        report::error(format!(
            "validation failed: {} failed suites in {elapsed}",
            summary.failed_examples
        ));
        exit(1);
    }
}
