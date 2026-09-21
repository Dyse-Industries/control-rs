//! CLI entrypoint for the `compare` runner and comparator (`cargo compare`).

#![allow(
    clippy::arbitrary_source_item_ordering,
    clippy::arithmetic_side_effects,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cmp_owned,
    clippy::collapsible_if,
    clippy::doc_markdown,
    clippy::indexing_slicing,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::module_name_repetitions,
    clippy::multiple_crate_versions,
    clippy::must_use_candidate,
    clippy::nursery,
    clippy::similar_names,
    clippy::struct_excessive_bools,
    clippy::too_many_lines,
    clippy::type_complexity,
    clippy::uninlined_format_args
)]

use std::collections::BTreeSet;
use std::env;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Duration;

use control_rs_compare::compare::{ComparatorOptions, run_comparison};
use control_rs_compare::config::CompareConfigFile;
use control_rs_compare::runner::{RunnerOptions, execute_master_plan};

struct CliArgs {
    config_path: PathBuf,
    results_dir: PathBuf,
    run_filter: Option<Vec<String>>,
    skip_run: bool,
    compare_filter: Option<Vec<String>>,
    skip_compare: bool,
    oracle_override: Option<String>,
    signals: Option<Vec<String>>,
    timeout_secs: Option<u64>,
    threads: Option<usize>,
    strict: bool,
    bypass_gate: bool,
    quiet: bool,
}

impl Default for CliArgs {
    fn default() -> Self {
        let default_config = if Path::new("compare.toml").exists() {
            PathBuf::from("compare.toml")
        } else if Path::new("oracle.toml").exists() {
            PathBuf::from("oracle.toml")
        } else {
            PathBuf::from("compare.toml")
        };

        Self {
            config_path: default_config,
            results_dir: PathBuf::from("results"),
            run_filter: None,
            skip_run: false,
            compare_filter: None,
            skip_compare: false,
            oracle_override: None,
            signals: None,
            timeout_secs: None,
            threads: None,
            strict: true,
            bypass_gate: false,
            quiet: false,
        }
    }
}

fn parse_args() -> Result<CliArgs, String> {
    let mut args = CliArgs::default();
    let mut iter = env::args().skip(1);

    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--config" | "-c" => {
                let val = iter.next().ok_or_else(|| {
                    "--config requires a path argument".to_string()
                })?;
                args.config_path = PathBuf::from(val);
            }
            "--results-dir" | "--out-dir" | "-o" => {
                let val = iter.next().ok_or_else(|| {
                    "--results-dir requires a directory argument".to_string()
                })?;
                args.results_dir = PathBuf::from(val);
            }
            "--run" => {
                let val = iter.next().ok_or_else(|| {
                    "--run requires a target suite or 'all'".to_string()
                })?;
                if val == "none" || val == "false" {
                    args.skip_run = true;
                } else if val != "all" {
                    args.run_filter =
                        Some(val.split(',').map(String::from).collect());
                }
            }
            "--skip-run" | "--no-run" => {
                args.skip_run = true;
            }
            "--compare" => {
                let val = iter.next().ok_or_else(|| {
                    "--compare requires a target suite, 'all', or 'false'"
                        .to_string()
                })?;
                if val == "none" || val == "false" {
                    args.skip_compare = true;
                } else if val != "all" && val != "true" {
                    args.compare_filter =
                        Some(val.split(',').map(String::from).collect());
                }
            }
            "--skip-compare" | "--no-compare" => {
                args.skip_compare = true;
            }
            "--oracle" => {
                let val = iter.next().ok_or_else(|| {
                    "--oracle requires a variant name".to_string()
                })?;
                args.oracle_override = Some(val);
            }
            "--signals" | "--signal" => {
                let val = iter.next().ok_or_else(|| {
                    "--signals requires comma-separated signal names"
                        .to_string()
                })?;
                let sigs: Vec<String> = val
                    .split(',')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(String::from)
                    .collect();
                if let Some(existing) = &mut args.signals {
                    existing.extend(sigs);
                } else {
                    args.signals = Some(sigs);
                }
            }
            "--timeout" => {
                let val = iter.next().ok_or_else(|| {
                    "--timeout requires a number of seconds".to_string()
                })?;
                let secs: u64 = val
                    .parse()
                    .map_err(|_| "Invalid timeout integer".to_string())?;
                args.timeout_secs = Some(secs);
            }
            "--threads" | "-j" => {
                let val = iter.next().ok_or_else(|| {
                    "--threads requires a thread count integer".to_string()
                })?;
                let n: usize = val
                    .parse()
                    .map_err(|_| "Invalid threads integer".to_string())?;
                args.threads = Some(n);
            }
            "--strict" => {
                args.strict = true;
            }
            "--bypass-gate" => {
                args.bypass_gate = true;
            }
            "--quiet" | "-q" => {
                args.quiet = true;
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => {
                return Err(format!("Unrecognized CLI argument '{other}'"));
            }
        }
    }

    Ok(args)
}

fn print_help() {
    println!("compare — Cross-Compare Harness & HDF5 Verification System");
    println!();
    println!("USAGE:");
    println!("    cargo compare [OPTIONS]");
    println!("    compare [OPTIONS]");
    println!();
    println!("OPTIONS:");
    println!(
        "    -c, --config <FILE>          Path to compare.toml (default: compare.toml)"
    );
    println!(
        "    -o, --results-dir <DIR>      Output results directory (default: results)"
    );
    println!(
        "        --run <SUITES>           Suites to run ('all', 'none', or 's1,s2')"
    );
    println!(
        "        --skip-run               Skip variant execution, run comparison only"
    );
    println!(
        "        --compare <SUITES>       Suites to compare ('all', 'none', or 's1,s2')"
    );
    println!(
        "        --skip-compare           Skip comparison, run variants only"
    );
    println!(
        "        --oracle <VARIANT>       Override true oracle variant (default: scipy)"
    );
    println!(
        "        --signals <SIGNALS>      Explicit signals to compare ('s1,s2', overrides discovery)"
    );
    println!("        --timeout <SECS>         Timeout in seconds per variant");
    println!(
        "    -j, --threads <N>            Worker threads for chunked dataset comparison"
    );
    println!(
        "        --strict                 Fail-closed exit status on discrepancies"
    );
    println!(
        "        --bypass-gate            Generate reports without non-zero exit"
    );
    println!("    -q, --quiet                  Suppress streaming output");
    println!("    -h, --help                   Print help information");
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("Error: {e}");
            return ExitCode::from(2);
        }
    };

    // 1. Load configuration if present
    let config_file = if args.config_path.exists() {
        match CompareConfigFile::load_from_file(&args.config_path) {
            Ok(cfg) => Some(cfg),
            Err(e) => {
                eprintln!("Error loading config: {e}");
                return ExitCode::from(1);
            }
        }
    } else {
        None
    };

    let master_plan = config_file
        .as_ref()
        .and_then(|cfg| cfg.resolve_master_plan(&args.config_path).ok());

    let results_dir = if let Some(plan) = &master_plan {
        if args.results_dir == std::path::Path::new("results") {
            PathBuf::from(&plan.general.out_dir)
        } else {
            args.results_dir.clone()
        }
    } else {
        args.results_dir.clone()
    };

    let timeout_secs = args
        .timeout_secs
        .or_else(|| master_plan.as_ref().map(|p| p.general.timeout_secs))
        .unwrap_or(120);

    // 2. Phase A: Variant Execution (unless skipped)
    if !args.skip_run {
        if let Some(plan) = &master_plan {
            let runner_opts = RunnerOptions {
                workspace_root: PathBuf::from("."),
                out_dir: results_dir.clone(),
                timeout: Duration::from_secs(timeout_secs),
                quiet: args.quiet,
            };

            let run_filter = args.run_filter.as_deref().unwrap_or(&[]);
            if let Err(e) = execute_master_plan(plan, &runner_opts, run_filter)
            {
                eprintln!("Execution error: {e}");
                if !args.bypass_gate {
                    return ExitCode::from(1);
                }
            }
        } else if !args.quiet {
            println!(
                "No master plan loaded from '{}', skipping execution phase",
                args.config_path.display()
            );
        }
    }

    // 3. Phase B: Multi-Method Comparison (unless skipped)
    if !args.skip_compare {
        let suite_filter_set: Option<BTreeSet<String>> =
            args.compare_filter.map(|list| list.into_iter().collect());

        let num_threads = args
            .threads
            .or_else(|| master_plan.as_ref().and_then(|p| p.general.threads));

        let comparator_opts = ComparatorOptions {
            results_dir: results_dir.clone(),
            suite_filter: suite_filter_set,
            oracle_override: args.oracle_override,
            signals: args.signals,
            strict: args.strict,
            quiet: args.quiet,
            num_threads,
        };

        match run_comparison(master_plan.as_ref(), &comparator_opts) {
            Ok(report) => {
                if let Err(e) = report.save_reports(&results_dir) {
                    eprintln!("Error saving validation reports: {e}");
                    return ExitCode::from(1);
                }

                if !args.quiet {
                    println!();
                    println!("{}", report.render_markdown());
                }

                if report.summary.verdict == "Fail"
                    && args.strict
                    && !args.bypass_gate
                {
                    return ExitCode::from(1);
                }
            }
            Err(e) => {
                eprintln!("Comparison error: {e}");
                if !args.bypass_gate {
                    return ExitCode::from(1);
                }
            }
        }
    }

    ExitCode::SUCCESS
}
