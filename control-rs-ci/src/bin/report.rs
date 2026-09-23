//! Report aggregator CLI (`cargo report`).

use std::env;
use std::path::{Path, PathBuf};
use std::process::exit;

use control_rs_ci::config::GateConfig;
use control_rs_ci::report::ReportAggregator;
use control_rs_ci::ui;

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
           {f}-c{f:#}, {f}--config{f:#} {a}<path>{a:#}    Path to gate.toml (default: workspace gate.toml)\n  \
           {f}-X{f:#}, {f}--clean{f:#}            Clean CI artifacts and reports\n  \
           {f}-h{f:#}, {f}--help{f:#}             Print help information\n\n\
         {h}Examples:{h:#}\n  \
           {f}{binary_name}{f:#}\n  \
           {f}{binary_name}{f:#} {f}--clean{f:#}\n  \
           {f}{binary_name}{f:#} {f}--config{f:#} {a}gate.toml{a:#}"
    );
}

/// Parses the command line; `--help` and unknown arguments exit.
fn parse_report_args(args: &[String]) -> ReportArgs {
    let mut parsed = ReportArgs {
        config_path: None,
        clean: false,
    };
    let mut rest = args.iter().skip(1);
    while let Some(arg) = rest.next().map(String::as_str) {
        if arg == "-h" || arg == "--help" {
            print_usage("cargo report");
            exit(0);
        } else if arg == "-X" || arg == "--clean" || arg == "clean" {
            parsed.clean = true;
        } else if arg == "-c" || arg == "--config" {
            if let Some(val) = rest.next() {
                parsed.config_path = Some(PathBuf::from(val));
            }
        } else if let Some(val) = arg
            .strip_prefix("--config=")
            .or_else(|| arg.strip_prefix("-c="))
        {
            parsed.config_path = Some(PathBuf::from(val));
        } else {
            ui::error(format!("Unknown argument: {arg}"));
            print_usage("cargo report");
            exit(1);
        }
    }
    parsed
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
            exit(1);
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let parsed = parse_report_args(&args);

    let workspace_root =
        env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let config_path = parsed
        .config_path
        .unwrap_or_else(|| workspace_root.join("gate.toml"));

    if parsed.clean {
        clean_and_exit(&workspace_root, &config_path);
    }

    let config = match GateConfig::load_from_path(&config_path) {
        Ok(c) => c,
        Err(e) => {
            ui::error(format!("Failed to load gate.toml: {e}"));
            exit(1);
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
