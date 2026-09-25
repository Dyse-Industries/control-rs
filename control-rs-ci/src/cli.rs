//! Command-line argument parsing and execution for CI and gate runners.

use std::path::{Path, PathBuf};
use std::process::exit;

use crate::config::GateConfig;
use crate::gate::build_all_gates;
use crate::ui;
use crate::{GateFilter, PipelineOptions, run_pipeline};

/// The only binary that accepts `-- <args>` passthrough (FR-14).
pub const PASSTHROUGH_BINARY: &str = "cargo gate";

/// Arguments after `--`, or `None` when no `--` was given.
pub type Passthrough = Option<Vec<String>>;

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
    /// Arguments after `--`, appended to the one selected gate (FR-14).
    pub passthrough: Passthrough,
}

/// Formats the help and usage string using cargo-style terminal colors.
#[must_use]
pub fn render_usage(binary_name: &str) -> String {
    let h = ui::HELP_HEADER;
    let f = ui::HELP_FLAG;
    let a = ui::HELP_ARG;
    let (usage_tail, example_tail) = if binary_name == PASSTHROUGH_BINARY {
        (
            format!(" {a}[-- <ARGS>...]{a:#}"),
            format!(
                "\n  {f}{binary_name}{f:#} {f}--only{f:#} {a}mutants -- --jobs 8{a:#}"
            ),
        )
    } else {
        (String::new(), String::new())
    };
    format!(
        "{h}Usage:{h:#} {f}{binary_name}{f:#} {a}[OPTIONS]{a:#} {a}[GATES]...{a:#}{usage_tail}\n\n\
         {h}Options:{h:#}\n  \
           {f}-g{f:#}, {f}--group{f:#} {a}<group>{a:#}    Run all quality gates in the specified group(s)\n  \
           {f}-o{f:#}, {f}--only{f:#} {a}<gate>{a:#}      Run only the specified gate(s) (comma-separated or repeated)\n  \
           {f}-s{f:#}, {f}--skip{f:#} {a}<gate>{a:#}      Skip the specified gate(s)\n  \
           {f}-u{f:#}, {f}--up-to{f:#} {a}<gate>{a:#}     Run gates up to and including the specified gate\n  \
           {f}-c{f:#}, {f}--config{f:#} {a}<path>{a:#}    Path to gate.toml (default: .cargo/gate.toml)\n  \
           {f}-X{f:#}, {f}--clean{f:#}            Clean previous CI artifacts and reports\n  \
           {f}-a{f:#}, {f}--all{f:#}              Run every enabled gate, including default = false gates\n  \
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
           {f}{binary_name}{f:#} {f}--up-to{f:#} {a}test{a:#}{example_tail}"
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
        i = parse_one(&mut options, args, i, binary_name);
    }
    options
}

/// Consumes the argument at `i` (and its values) and returns the index of
/// the next unconsumed argument. `--help`, `--list` and unknown flags exit.
fn parse_one(
    options: &mut CliOptions,
    args: &[String],
    i: usize,
    binary_name: &str,
) -> usize {
    let Some(arg) = args.get(i).map(String::as_str) else {
        return args.len();
    };
    let next = i.saturating_add(1);
    match arg {
        "--" => {
            options.passthrough = Some(
                args.get(next..).map(<[String]>::to_vec).unwrap_or_default(),
            );
            return args.len();
        }
        "-h" | "--help" => {
            print_usage(binary_name);
            exit(0);
        }
        "-l" | "--list" => {
            list_gates(options);
            exit(0);
        }
        "-X" | "--clean" => options.clean = true,
        "-v" | "--verbose" => options.verbose = true,
        "-a" | "--all" => options.run_all = true,
        "-g" | "--group" => {
            return take_values(args, next, &mut options.groups);
        }
        "-o" | "--only" => {
            return take_values(args, next, &mut options.only_gates);
        }
        "-s" | "--skip" => {
            return take_values(args, next, &mut options.skip_gates);
        }
        "-u" | "--up-to" => {
            if let Some(val) = args.get(next) {
                options.up_to_gate = Some(val.clone());
            }
            return next.saturating_add(1);
        }
        "-c" | "--config" => {
            if let Some(val) = args.get(next) {
                options.config_path = Some(PathBuf::from(val));
            }
            return next.saturating_add(1);
        }
        _ => parse_inline(options, arg, binary_name),
    }
    next
}

/// Handles `--flag=value` forms, positional gate names and unknown flags.
fn parse_inline(options: &mut CliOptions, arg: &str, binary_name: &str) {
    if let Some(val) = inline_value(arg, "--group=", "-g=") {
        push_list(&mut options.groups, val);
    } else if let Some(val) = inline_value(arg, "--only=", "-o=") {
        push_list(&mut options.only_gates, val);
    } else if let Some(val) = inline_value(arg, "--skip=", "-s=") {
        push_list(&mut options.skip_gates, val);
    } else if let Some(val) = inline_value(arg, "--up-to=", "-u=") {
        options.up_to_gate = Some(val.to_string());
    } else if let Some(val) = inline_value(arg, "--config=", "-c=") {
        options.config_path = Some(PathBuf::from(val));
    } else if !arg.starts_with('-') {
        for part in arg.split(',').map(str::trim).filter(|p| !p.is_empty()) {
            match part {
                "clean" => options.clean = true,
                "all" => options.run_all = true,
                gate => options.only_gates.push(gate.to_string()),
            }
        }
    } else {
        ui::error(format!("Unknown argument: {arg}"));
        print_usage(binary_name);
        exit(1);
    }
}

/// Value of `arg` after the `long` or `short` `=`-terminated prefix.
fn inline_value<'a>(arg: &'a str, long: &str, short: &str) -> Option<&'a str> {
    arg.strip_prefix(long).or_else(|| arg.strip_prefix(short))
}

/// Appends the non-empty, trimmed comma-separated entries of `val`.
fn push_list(dest: &mut Vec<String>, val: &str) {
    dest.extend(
        val.split(',')
            .map(str::trim)
            .filter(|p| !p.is_empty())
            .map(str::to_string),
    );
}

/// Collects the values following a multi-value flag, starting at `start`,
/// up to the next argument that begins with `-`. Returns the index of that
/// argument (or the end).
fn take_values(args: &[String], start: usize, dest: &mut Vec<String>) -> usize {
    let mut i = start;
    while let Some(val) = args.get(i).filter(|a| !a.starts_with('-')) {
        push_list(dest, val);
        i = i.saturating_add(1);
    }
    i
}

/// Prints every registered gate with its description.
fn list_gates(options: &CliOptions) {
    let config_path = resolve_config_path(options);
    let config = GateConfig::load_from_path(&config_path).unwrap_or_default();
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
}

/// Current directory, or `.` when it cannot be read.
fn workspace_root() -> PathBuf {
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

/// `--config` when given, otherwise `.cargo/gate.toml` in the workspace root.
fn resolve_config_path(options: &CliOptions) -> PathBuf {
    options
        .config_path
        .clone()
        .unwrap_or_else(|| workspace_root().join(".cargo/gate.toml"))
}

/// Runs the CLI application using parsed options.
pub fn run_cli(binary_name: &str) {
    let args: Vec<String> = std::env::args().collect();
    let mut options = parse_args(&args, binary_name);
    let workspace_root = workspace_root();
    let config_path = resolve_config_path(&options);

    let is_clean_only = options.clean
        && !options.run_all
        && options.groups.is_empty()
        && options.only_gates.is_empty()
        && options.up_to_gate.is_none();
    if is_clean_only {
        clean_and_exit(&workspace_root, &config_path);
    }

    let config = GateConfig::load_from_path(&config_path).unwrap_or_else(|e| {
        ui::error(format!("Failed to load configuration: {e}"));
        exit(1);
    });

    // Expand groups specified via --group / -g into options.only_gates
    for group_name in &options.groups {
        let Some(members) = group_members(&config, group_name) else {
            ui::error(format!("Unknown execution group: '{group_name}'"));
            exit(1);
        };
        push_unique(&mut options.only_gates, members);
    }

    // Also expand any positional arguments that match group names
    let mut expanded_gates = Vec::new();
    for gate_or_group in &options.only_gates {
        match group_members(&config, gate_or_group) {
            Some(members) => push_unique(&mut expanded_gates, members),
            None => {
                push_unique(
                    &mut expanded_gates,
                    std::slice::from_ref(gate_or_group),
                );
            }
        }
    }
    options.only_gates = expanded_gates;

    if let Err(e) = check_passthrough(binary_name, &options, &config) {
        ui::error(e);
        exit(2);
    }

    let pipeline = PipelineOptions {
        only_gates: (!options.run_all && !options.only_gates.is_empty())
            .then_some(options.only_gates.as_slice()),
        skip_gates: (!options.skip_gates.is_empty())
            .then_some(options.skip_gates.as_slice()),
        up_to_gate: options.up_to_gate.as_deref(),
        all: options.run_all,
        clean: options.clean,
        verbose: options.verbose,
        extra_args: options.passthrough.as_deref().unwrap_or_default(),
    };
    match run_pipeline(&workspace_root, &config_path, &pipeline) {
        Ok(true) => exit(0),
        Ok(false) => exit(1),
        Err(e) => {
            ui::error(format!("Fatal error executing CI pipeline: {e}"));
            exit(1);
        }
    }
}

/// Checks the FR-14 passthrough rules once groups are expanded: `--` is
/// accepted by [`PASSTHROUGH_BINARY`] only, and the selection before it must
/// resolve to exactly one configured gate.
///
/// # Errors
/// Returns the diagnostic to print when a rule is violated.
pub fn check_passthrough(
    binary_name: &str,
    options: &CliOptions,
    config: &GateConfig,
) -> Result<(), String> {
    if options.passthrough.is_none() {
        return Ok(());
    }
    if binary_name != PASSTHROUGH_BINARY {
        return Err(format!(
            "`--` is accepted by `{PASSTHROUGH_BINARY}` only; `{binary_name}` runs many gates"
        ));
    }
    let single = match options.only_gates.as_slice() {
        [gate] if !options.run_all && options.up_to_gate.is_none() => gate,
        _ => {
            return Err(
                "`--` requires exactly one selected gate (`--only <gate>` or a positional name)"
                    .to_string(),
            );
        }
    };
    if options.skip_gates.contains(single) {
        return Err(format!("`--` target '{single}' is also skipped"));
    }
    if config.gate_def(single).is_none() {
        return Err(format!("unknown gate '{single}'"));
    }
    Ok(())
}

/// Removes CI artifacts and exits with the outcome.
fn clean_and_exit(workspace_root: &Path, config_path: &Path) -> ! {
    ui::init_color();
    match crate::clean_artifacts(workspace_root, config_path) {
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

/// Gates of the named group (`exclusive` included), or `None` if no such
/// group exists.
fn group_members<'a>(config: &'a GateConfig, name: &str) -> GateFilter<'a> {
    if name == "exclusive" {
        Some(&config.execution.exclusive_gates)
    } else {
        config.execution.groups.get(name).map(Vec::as_slice)
    }
}

/// Appends each of `names` not already in `dest`.
fn push_unique(dest: &mut Vec<String>, names: &[String]) {
    for name in names {
        if !dest.contains(name) {
            dest.push(name.clone());
        }
    }
}
