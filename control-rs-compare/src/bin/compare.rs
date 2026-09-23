//! CLI entrypoint for the `compare` runner and comparator (`cargo compare`).

use std::env;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Duration;

use control_rs_compare::compare::{
    ComparatorOptions, SignalNames, SuiteNames, run_comparison,
};
use control_rs_compare::config::{CompareConfigFile, MasterPlan};
use control_rs_compare::runner::{RunnerOptions, execute_master_plan};

struct CliArgs {
    config_path: PathBuf,
    results_dir: PathBuf,
    run: Selection,
    compare: Selection,
    oracle_override: Option<String>,
    signals: Option<SignalNames>,
    timeout_secs: Option<u64>,
    threads: Option<usize>,
    strict: bool,
    bypass_gate: bool,
    quiet: bool,
}

/// Suite names in command-line order.
type SuiteList = Vec<String>;

/// Which suites a phase (variant execution or comparison) covers.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Selection {
    /// Every configured suite.
    All,
    /// Only the named suites.
    Only(SuiteList),
    /// The phase is skipped. Once selected it is never overridden.
    Skip,
}

impl Selection {
    /// Replaces the selection unless the phase was already skipped.
    fn set(&mut self, next: Self) {
        if *self != Self::Skip {
            *self = next;
        }
    }

    /// Parses a `--run`/`--compare` value: `skip_words` skip the phase,
    /// `all_words` select every suite, anything else is a suite list.
    fn parse(val: &str, skip_words: &[&str], all_words: &[&str]) -> Self {
        if skip_words.contains(&val) {
            Self::Skip
        } else if all_words.contains(&val) {
            Self::All
        } else {
            Self::Only(val.split(',').map(String::from).collect())
        }
    }

    /// Suite names when the selection is a list.
    const fn names(&self) -> Option<&SuiteList> {
        match self {
            Self::Only(names) => Some(names),
            Self::All | Self::Skip => None,
        }
    }
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
            run: Selection::All,
            compare: Selection::All,
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
            "--skip-run" | "--no-run" => args.run = Selection::Skip,
            "--skip-compare" | "--no-compare" => args.compare = Selection::Skip,
            "--strict" => args.strict = true,
            "--bypass-gate" => args.bypass_gate = true,
            "--quiet" | "-q" => args.quiet = true,
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            flag => apply_valued_flag(&mut args, flag, &mut iter)?,
        }
    }

    Ok(args)
}

/// Applies a flag that takes a value, reading the value from `iter`.
fn apply_valued_flag(
    args: &mut CliArgs,
    flag: &str,
    iter: &mut impl Iterator<Item = String>,
) -> Result<(), String> {
    let mut value = |what: &str| {
        iter.next().ok_or_else(|| format!("{flag} requires {what}"))
    };
    match flag {
        "--config" | "-c" => {
            args.config_path = PathBuf::from(value("a path argument")?);
        }
        "--results-dir" | "--out-dir" | "-o" => {
            args.results_dir = PathBuf::from(value("a directory argument")?);
        }
        "--run" => {
            let val = value("a target suite or 'all'")?;
            args.run
                .set(Selection::parse(&val, &["none", "false"], &["all"]));
        }
        "--compare" => {
            let val = value("a target suite, 'all', or 'false'")?;
            args.compare.set(Selection::parse(
                &val,
                &["none", "false"],
                &["all", "true"],
            ));
        }
        "--oracle" => args.oracle_override = Some(value("a variant name")?),
        "--signals" | "--signal" => {
            let val = value("comma-separated signal names")?;
            args.signals.get_or_insert_with(Vec::new).extend(
                val.split(',')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(String::from),
            );
        }
        "--timeout" => {
            let val = value("a number of seconds")?;
            args.timeout_secs = Some(
                val.parse()
                    .map_err(|_| "Invalid timeout integer".to_string())?,
            );
        }
        "--threads" | "-j" => {
            let val = value("a thread count integer")?;
            args.threads = Some(
                val.parse()
                    .map_err(|_| "Invalid threads integer".to_string())?,
            );
        }
        other => return Err(format!("Unrecognized CLI argument '{other}'")),
    }
    Ok(())
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

    let results_dir = match &master_plan {
        Some(plan) if args.results_dir == Path::new("results") => {
            PathBuf::from(&plan.general.out_dir)
        }
        _ => args.results_dir.clone(),
    };

    // 2. Phase A: Variant Execution (unless skipped)
    if args.run != Selection::Skip
        && let Some(code) =
            run_variants(&args, master_plan.as_ref(), &results_dir)
    {
        return code;
    }

    // 3. Phase B: Multi-Method Comparison (unless skipped)
    if args.compare != Selection::Skip
        && let Some(code) =
            compare_results(args, master_plan.as_ref(), &results_dir)
    {
        return code;
    }

    ExitCode::SUCCESS
}

/// Executes every selected variant. Returns an exit code when the run must
/// stop here.
fn run_variants(
    args: &CliArgs,
    master_plan: Option<&MasterPlan>,
    results_dir: &Path,
) -> Option<ExitCode> {
    let Some(plan) = master_plan else {
        if !args.quiet {
            println!(
                "No master plan loaded from '{}', skipping execution phase",
                args.config_path.display()
            );
        }
        return None;
    };

    let timeout_secs = args.timeout_secs.unwrap_or(plan.general.timeout_secs);
    let runner_opts = RunnerOptions {
        workspace_root: PathBuf::from("."),
        out_dir: results_dir.to_path_buf(),
        timeout: Duration::from_secs(timeout_secs),
        quiet: args.quiet,
    };

    let run_filter = args.run.names().map_or(&[][..], Vec::as_slice);
    if let Err(e) = execute_master_plan(plan, &runner_opts, run_filter) {
        eprintln!("Execution error: {e}");
        if !args.bypass_gate {
            return Some(ExitCode::from(1));
        }
    }
    None
}

/// Compares the result containers and writes the reports. Returns an exit
/// code when the comparison fails the gate.
fn compare_results(
    args: CliArgs,
    master_plan: Option<&MasterPlan>,
    results_dir: &Path,
) -> Option<ExitCode> {
    let suite_filter: Option<SuiteNames> = args
        .compare
        .names()
        .map(|list| list.iter().cloned().collect());

    let num_threads = args
        .threads
        .or_else(|| master_plan.and_then(|p| p.general.threads));

    let comparator_opts = ComparatorOptions {
        results_dir: results_dir.to_path_buf(),
        suite_filter,
        oracle_override: args.oracle_override,
        signals: args.signals,
        strict: args.strict,
        quiet: args.quiet,
        num_threads,
    };

    match run_comparison(master_plan, &comparator_opts) {
        Ok(report) => {
            if let Err(e) = report.save_reports(results_dir) {
                eprintln!("Error saving validation reports: {e}");
                return Some(ExitCode::from(1));
            }

            if !args.quiet {
                println!();
                println!("{}", report.render_markdown());
            }

            (report.summary.verdict == "Fail"
                && args.strict
                && !args.bypass_gate)
                .then(|| ExitCode::from(1))
        }
        Err(e) => {
            eprintln!("Comparison error: {e}");
            (!args.bypass_gate).then(|| ExitCode::from(1))
        }
    }
}
