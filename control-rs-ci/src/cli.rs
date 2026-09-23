//! Command-line argument parsing and execution for CI and gate runners.

use std::path::PathBuf;
use std::process::exit;

use crate::config::GateConfig;
use crate::gate::build_all_gates;
use crate::run_pipeline;
use crate::ui;

/// Options parsed from command line arguments.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct CliOptions {
    /// Whitelist of execution groups to run.
    pub groups: Vec<String>,
    /// Whitelist of gates to run.
    pub only_gates: Vec<String>,
    /// Blacklist of gates to skip.
    pub skip_gates: Vec<String>,
    /// Gate name to stop execution after.
    pub up_to_gate: Option<String>,
    /// Custom path to gate.toml.
    pub config_path: Option<PathBuf>,
    /// Clean previous CI artifacts and reports.
    pub clean: bool,
    /// Run all quality gates.
    pub run_all: bool,
    /// Echo each gate's output to stderr as it runs.
    pub verbose: bool,
}

/// Formats the help and usage string using cargo-style terminal colors.
#[must_use]
pub fn render_usage(binary_name: &str) -> String {
    let h = ui::HELP_HEADER;
    let f = ui::HELP_FLAG;
    let a = ui::HELP_ARG;
    format!(
        "{h}Usage:{h:#} {f}{binary_name}{f:#} {a}[OPTIONS]{a:#} {a}[GATES]...{a:#}\n\n\
         {h}Options:{h:#}\n  \
           {f}-g{f:#}, {f}--group{f:#} {a}<group>{a:#}    Run all quality gates in the specified group(s)\n  \
           {f}-o{f:#}, {f}--only{f:#} {a}<gate>{a:#}      Run only the specified gate(s) (comma-separated or repeated)\n  \
           {f}-s{f:#}, {f}--skip{f:#} {a}<gate>{a:#}      Skip the specified gate(s)\n  \
           {f}-u{f:#}, {f}--up-to{f:#} {a}<gate>{a:#}     Run gates up to and including the specified gate\n  \
           {f}-c{f:#}, {f}--config{f:#} {a}<path>{a:#}    Path to gate.toml (default: workspace gate.toml)\n  \
           {f}-X{f:#}, {f}--clean{f:#}            Clean previous CI artifacts and reports\n  \
           {f}-a{f:#}, {f}--all{f:#}              Run all registered quality gates\n  \
           {f}-v{f:#}, {f}--verbose{f:#}          Echo each gate's output, prefixed with its group and name\n  \
           {f}-l{f:#}, {f}--list{f:#}             List all registered quality gates\n  \
           {f}-h{f:#}, {f}--help{f:#}             Print help information\n\n\
         {h}Examples:{h:#}\n  \
           {f}{binary_name}{f:#} {a}clean{a:#}\n  \
           {f}{binary_name}{f:#} {f}--clean{f:#}\n  \
           {f}{binary_name}{f:#} {f}--group{f:#} {a}lint{a:#}\n  \
           {f}{binary_name}{f:#} {f}--only{f:#} {a}fmt,clippy{a:#}\n  \
           {f}{binary_name}{f:#} {f}--verbose{f:#} {f}--group{f:#} {a}verify{a:#}\n  \
           {f}{binary_name}{f:#} {a}fmt{a:#}\n  \
           {f}{binary_name}{f:#} {f}--up-to{f:#} {a}test{a:#}"
    )
}

/// Prints cargo-styled usage instructions to stdout.
pub fn print_usage(binary_name: &str) {
    ui::init_color();
    anstream::println!("{}", render_usage(binary_name));
}

/// Parses CLI arguments into `CliOptions`.
#[must_use]
pub fn parse_args(args: &[String], binary_name: &str) -> CliOptions {
    let mut options = CliOptions::default();
    let mut i = 1;
    while i < args.len() {
        let arg = match args.get(i) {
            Some(a) => a.as_str(),
            None => break,
        };

        if arg == "-h" || arg == "--help" {
            print_usage(binary_name);
            exit(0);
        } else if arg == "-l" || arg == "--list" {
            let workspace_root =
                std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            let config_path = options
                .config_path
                .clone()
                .unwrap_or_else(|| workspace_root.join("gate.toml"));
            let config =
                GateConfig::load_from_path(&config_path).unwrap_or_default();
            let all_gates = build_all_gates(&config).unwrap_or_default();
            ui::init_color();
            let h = ui::HELP_HEADER;
            let f = ui::HELP_FLAG;
            anstream::println!("{h}Registered Quality Gates:{h:#}");
            for g in all_gates {
                anstream::println!(
                    "  - {f}{:<12}{f:#} : {}",
                    g.name(),
                    g.description().unwrap_or("-")
                );
            }
            exit(0);
        } else if arg == "-X" || arg == "--clean" {
            options.clean = true;
        } else if arg == "-v" || arg == "--verbose" {
            options.verbose = true;
        } else if arg == "-a" || arg == "--all" {
            options.run_all = true;
        } else if arg == "-g" || arg == "--group" {
            i = i.saturating_add(1);
            while i < args.len() {
                let val = match args.get(i) {
                    Some(a) if !a.starts_with('-') => a.as_str(),
                    _ => {
                        i = i.saturating_sub(1);
                        break;
                    }
                };
                for part in val.split(',') {
                    let trimmed = part.trim();
                    if !trimmed.is_empty() {
                        options.groups.push(trimmed.to_string());
                    }
                }
                i = i.saturating_add(1);
            }
        } else if let Some(val) = arg
            .strip_prefix("--group=")
            .or_else(|| arg.strip_prefix("-g="))
        {
            for part in val.split(',') {
                let trimmed = part.trim();
                if !trimmed.is_empty() {
                    options.groups.push(trimmed.to_string());
                }
            }
        } else if arg == "-o" || arg == "--only" {
            i = i.saturating_add(1);
            while i < args.len() {
                let val = match args.get(i) {
                    Some(a) if !a.starts_with('-') => a.as_str(),
                    _ => {
                        i = i.saturating_sub(1);
                        break;
                    }
                };
                for part in val.split(',') {
                    let trimmed = part.trim();
                    if !trimmed.is_empty() {
                        options.only_gates.push(trimmed.to_string());
                    }
                }
                i = i.saturating_add(1);
            }
        } else if let Some(val) = arg
            .strip_prefix("--only=")
            .or_else(|| arg.strip_prefix("-o="))
        {
            for part in val.split(',') {
                let trimmed = part.trim();
                if !trimmed.is_empty() {
                    options.only_gates.push(trimmed.to_string());
                }
            }
        } else if arg == "-s" || arg == "--skip" {
            i = i.saturating_add(1);
            while i < args.len() {
                let val = match args.get(i) {
                    Some(a) if !a.starts_with('-') => a.as_str(),
                    _ => {
                        i = i.saturating_sub(1);
                        break;
                    }
                };
                for part in val.split(',') {
                    let trimmed = part.trim();
                    if !trimmed.is_empty() {
                        options.skip_gates.push(trimmed.to_string());
                    }
                }
                i = i.saturating_add(1);
            }
        } else if let Some(val) = arg
            .strip_prefix("--skip=")
            .or_else(|| arg.strip_prefix("-s="))
        {
            for part in val.split(',') {
                let trimmed = part.trim();
                if !trimmed.is_empty() {
                    options.skip_gates.push(trimmed.to_string());
                }
            }
        } else if arg == "-u" || arg == "--up-to" {
            i = i.saturating_add(1);
            if let Some(val) = args.get(i) {
                options.up_to_gate = Some(val.clone());
            }
        } else if let Some(val) = arg
            .strip_prefix("--up-to=")
            .or_else(|| arg.strip_prefix("-u="))
        {
            options.up_to_gate = Some(val.to_string());
        } else if arg == "-c" || arg == "--config" {
            i = i.saturating_add(1);
            if let Some(val) = args.get(i) {
                options.config_path = Some(PathBuf::from(val));
            }
        } else if let Some(val) = arg
            .strip_prefix("--config=")
            .or_else(|| arg.strip_prefix("-c="))
        {
            options.config_path = Some(PathBuf::from(val));
        } else if !arg.starts_with('-') {
            for part in arg.split(',') {
                let trimmed = part.trim();
                if !trimmed.is_empty() {
                    if trimmed == "clean" {
                        options.clean = true;
                    } else if trimmed == "all" {
                        options.run_all = true;
                    } else {
                        options.only_gates.push(trimmed.to_string());
                    }
                }
            }
        } else {
            ui::error(format!("Unknown argument: {arg}"));
            print_usage(binary_name);
            exit(1);
        }

        i = i.saturating_add(1);
    }
    options
}

/// Runs the CLI application using parsed options.
pub fn run_cli(binary_name: &str) {
    let args: Vec<String> = std::env::args().collect();
    let mut options = parse_args(&args, binary_name);
    let workspace_root =
        std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let config_path = options
        .config_path
        .clone()
        .unwrap_or_else(|| workspace_root.join("gate.toml"));

    let is_clean_only = options.clean
        && !options.run_all
        && options.groups.is_empty()
        && options.only_gates.is_empty()
        && options.up_to_gate.is_none();

    if is_clean_only {
        ui::init_color();
        match crate::clean_artifacts(&workspace_root, &config_path) {
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

    let config = match GateConfig::load_from_path(&config_path) {
        Ok(cfg) => cfg,
        Err(e) => {
            ui::error(format!("Failed to load configuration: {e}"));
            exit(1);
        }
    };

    // Expand groups specified via --group / -g into options.only_gates
    for group_name in &options.groups {
        if group_name == "exclusive" {
            for gate in &config.execution.exclusive_gates {
                if !options.only_gates.contains(gate) {
                    options.only_gates.push(gate.clone());
                }
            }
        } else if let Some(members) = config.execution.groups.get(group_name) {
            for gate in members {
                if !options.only_gates.contains(gate) {
                    options.only_gates.push(gate.clone());
                }
            }
        } else {
            ui::error(format!("Unknown execution group: '{group_name}'"));
            exit(1);
        }
    }

    // Also expand any positional arguments that match group names
    let mut expanded_gates = Vec::new();
    for gate_or_group in &options.only_gates {
        if gate_or_group == "exclusive" {
            for gate in &config.execution.exclusive_gates {
                if !expanded_gates.contains(gate) {
                    expanded_gates.push(gate.clone());
                }
            }
        } else if let Some(members) = config.execution.groups.get(gate_or_group)
        {
            for gate in members {
                if !expanded_gates.contains(gate) {
                    expanded_gates.push(gate.clone());
                }
            }
        } else if !expanded_gates.contains(gate_or_group) {
            expanded_gates.push(gate_or_group.clone());
        }
    }
    options.only_gates = expanded_gates;

    let only_ref = if options.run_all || options.only_gates.is_empty() {
        None
    } else {
        Some(options.only_gates.as_slice())
    };
    let skip_ref = if options.skip_gates.is_empty() {
        None
    } else {
        Some(options.skip_gates.as_slice())
    };

    match run_pipeline(
        &workspace_root,
        &config_path,
        only_ref,
        skip_ref,
        options.up_to_gate.as_deref(),
        options.clean,
        options.verbose,
    ) {
        Ok(true) => exit(0),
        Ok(false) => exit(1),
        Err(e) => {
            ui::error(format!("Fatal error executing CI pipeline: {e}"));
            exit(1);
        }
    }
}
