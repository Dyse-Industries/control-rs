//! Report aggregator CLI (`cargo report`).

use std::env;
use std::path::{Path, PathBuf};
use std::process::exit;

use control_rs_ci::GateError;
use control_rs_ci::cli::{USAGE_EXIT, arg_spelling};
use control_rs_ci::config::GateConfig;
use control_rs_ci::report::ReportAggregator;
use control_rs_ci::ui;

/// Parsed options, `None` for `--help`, or a usage error.
type ParsedReport = Result<Option<ReportArgs>, String>;

/// Options accepted by `cargo report`.
struct ReportArgs {
    /// `--config` override.
    config_path: Option<PathBuf>,
    /// Clean artifacts instead of reporting.
    clean: bool,
}

fn print_usage(binary_name: &str) {
    ui::init_color();
    let h = ui::HELP_HEADER;
    let f = ui::HELP_FLAG;
    let a = ui::HELP_ARG;
    anstream::println!(
        "{h}Usage:{h:#} {f}{binary_name}{f:#} {a}[OPTIONS]{a:#}\n\n\
         {h}Options:{h:#}\n  \
           {f}-c{f:#}, {f}--config{f:#} {a}<path>{a:#}    Path to gate.toml (default: .cargo/gate.toml)\n  \
           {f}-X{f:#}, {f}--clean{f:#}            Clean CI artifacts and reports\n  \
           {f}-h{f:#}, {f}--help{f:#}             Print help information\n\n\
         {h}Examples:{h:#}\n  \
           {f}{binary_name}{f:#}\n  \
           {f}{binary_name}{f:#} {f}--clean{f:#}\n  \
           {f}{binary_name}{f:#} {f}--config{f:#} {a}.cargo/gate.toml{a:#}"
    );
}

/// Parses the command line (program name first); `Ok(None)` means `--help`.
fn parse_report_args(args: &[String]) -> ParsedReport {
    use lexopt::prelude::*;
    let mut parsed = ReportArgs {
        config_path: None,
        clean: false,
    };
    let mut parser = lexopt::Parser::from_iter(args);
    while let Some(arg) = parser.next().map_err(|e| e.to_string())? {
        match arg {
            Short('h') | Long("help") => return Ok(None),
            Short('X') | Long("clean") => parsed.clean = true,
            Value(ref val) if val == "clean" => parsed.clean = true,
            Short('c') | Long("config") => {
                let val = parser
                    .value()
                    .map_err(|_| "--config requires a path".to_string())?;
                parsed.config_path = Some(PathBuf::from(val));
            }
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

/// Removes CI artifacts and exits with the outcome.
fn clean_and_exit(workspace_root: &Path, config_path: &Path) -> ! {
    ui::init_color();
    match control_rs_ci::clean_artifacts(workspace_root, config_path) {
        Ok(out_dir) => {
            ui::status(
                "Cleaned",
                format!("Removed CI artifacts in {}", out_dir.display()),
            );
            exit(0);
        }
        Err(e) => {
            ui::error(format!("Failed to clean CI artifacts: {e}"));
            exit(if matches!(e, GateError::Config { .. }) {
                USAGE_EXIT
            } else {
                1
            });
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let parsed = match parse_report_args(&args) {
        Ok(Some(parsed)) => parsed,
        Ok(None) => {
            print_usage("cargo report");
            exit(0);
        }
        Err(e) => {
            ui::error(e);
            print_usage("cargo report");
            exit(USAGE_EXIT);
        }
    };

    let workspace_root =
        env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let config_path = parsed
        .config_path
        .unwrap_or_else(|| workspace_root.join(".cargo/gate.toml"));

    if parsed.clean {
        clean_and_exit(&workspace_root, &config_path);
    }

    let config = match GateConfig::load_from_path(&config_path) {
        Ok(c) => c,
        Err(e) => {
            ui::error(format!("Failed to load gate.toml: {e}"));
            exit(USAGE_EXIT);
        }
    };

    let artifacts_dir = workspace_root.join(&config.runner.out_dir);
    let aggregator = ReportAggregator::new(artifacts_dir, workspace_root);

    match aggregator.write_report(&config, None) {
        Ok(report) => {
            ui::status("Writing", format!("{}", report.path.display()));
            if report.pass {
                ui::status("Finished", "All fail-closed quality gates passed");
                exit(0);
            }
            ui::error("One or more fail-closed quality gates failed");
            exit(1);
        }
        Err(e) => {
            ui::error(format!("Failed to generate report: {e}"));
            exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::parse_report_args;

    fn args(list: &[&str]) -> Vec<String> {
        std::iter::once("report")
            .chain(list.iter().copied())
            .map(ToString::to_string)
            .collect()
    }

    #[test]
    fn config_spellings_resolve() {
        for form in [
            &["-c", "g.toml"][..],
            &["-c=g.toml"],
            &["--config", "g.toml"],
            &["--config=g.toml"],
        ] {
            let parsed = parse_report_args(&args(form)).unwrap().unwrap();
            let path = parsed.config_path.unwrap();
            assert_eq!(path.to_str(), Some("g.toml"), "{form:?}");
        }
    }

    #[test]
    fn clean_has_three_spellings() {
        for form in ["-X", "--clean", "clean"] {
            let parsed = parse_report_args(&args(&[form])).unwrap().unwrap();
            assert!(parsed.clean, "{form}");
        }
        assert!(!parse_report_args(&args(&[])).unwrap().unwrap().clean);
    }

    #[test]
    fn help_and_usage_errors() {
        assert!(parse_report_args(&args(&["--help"])).unwrap().is_none());
        assert_eq!(
            parse_report_args(&args(&["--bogus"])).err().unwrap(),
            "Unknown argument: --bogus"
        );
        assert_eq!(
            parse_report_args(&args(&["tidy"])).err().unwrap(),
            "Unknown argument: tidy"
        );
        assert!(parse_report_args(&args(&["--config"])).is_err());
    }
}
