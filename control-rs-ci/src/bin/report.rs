//! Report aggregator CLI (`cargo report`).

use std::env;
use std::path::PathBuf;
use std::process::exit;

use control_rs_ci::config::GateConfig;
use control_rs_ci::report::ReportAggregator;
use control_rs_ci::ui;

fn print_usage(binary_name: &str) {
    ui::init_color();
    let h = ui::HELP_HEADER;
    let f = ui::HELP_FLAG;
    let a = ui::HELP_ARG;
    anstream::println!(
        "{h}Usage:{h:#} {f}{binary_name}{f:#} {a}[OPTIONS]{a:#}\n\n\
         {h}Options:{h:#}\n  \
           {f}-c{f:#}, {f}--config{f:#} {a}<path>{a:#}    Path to gate.toml (default: workspace gate.toml)\n  \
           {f}-h{f:#}, {f}--help{f:#}             Print help information\n\n\
         {h}Examples:{h:#}\n  \
           {f}{binary_name}{f:#}\n  \
           {f}{binary_name}{f:#} {f}--config{f:#} {a}gate.toml{a:#}"
    );
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut config_path_override = None;
    let mut i = 1;
    while i < args.len() {
        let arg = match args.get(i) {
            Some(a) => a.as_str(),
            None => break,
        };
        if arg == "-h" || arg == "--help" {
            print_usage("cargo report");
            exit(0);
        } else if arg == "-c" || arg == "--config" {
            i = i.saturating_add(1);
            if let Some(val) = args.get(i) {
                config_path_override = Some(PathBuf::from(val));
            }
        } else if let Some(val) = arg
            .strip_prefix("--config=")
            .or_else(|| arg.strip_prefix("-c="))
        {
            config_path_override = Some(PathBuf::from(val));
        } else {
            ui::error(format!("Unknown argument: {arg}"));
            print_usage("cargo report");
            exit(1);
        }
        i = i.saturating_add(1);
    }

    let workspace_root =
        env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let config_path = config_path_override
        .unwrap_or_else(|| workspace_root.join("gate.toml"));

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
        Ok((is_pass, report_path)) => {
            ui::status("Writing", format!("{}", report_path.display()));
            if is_pass {
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
