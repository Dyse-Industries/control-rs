//! Criterion performance regression evaluator and benchmark budget harness.
//!
//! Runs `cargo bench`, echoes Criterion's output and parses it. Each
//! benchmark's `time:` point estimate is checked against a real-time latency
//! budget (for example, 10 µs jitter for flight control loops). Criterion
//! compares every benchmark with its `base/` sample and prints a verdict; a
//! `regressed` verdict fails the gate. A benchmark without a baseline is only
//! checked against its budget.
//!
//! Budgets come from a TOML file (`--budgets`, default
//! `.cargo/regression.toml`). A key names a benchmark identifier exactly, or a
//! prefix when it ends in `*`; the exact key wins, then the longest prefix. A
//! benchmark that matches no key fails the gate. The harness runs in the
//! current directory, which the gate runner sets to the workspace root.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// Benchmarks in the order Criterion printed them.
type Results = Vec<BenchmarkResult>;

/// Criterion's verdict on the change against the baseline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Verdict {
    Regressed,
    Improved,
    NoChange,
    WithinNoise,
}

/// One benchmark as printed by Criterion.
#[derive(Debug, Clone)]
struct BenchmarkResult {
    /// Benchmark identifier (for example, `jitter/hilbert_32x32_solve`).
    id: String,
    /// `time:` point estimate in nanoseconds.
    time_ns: f64,
    /// `change:` point estimate in %, when a baseline exists.
    change_pct: Option<f64>,
    /// Criterion's verdict, when a baseline exists.
    verdict: Option<Verdict>,
}

/// Command-line options.
#[derive(Debug, Clone)]
struct CliOptions {
    /// `--bench <NAME>`: run one bench target.
    bench_target: Option<String>,
    /// `--all`: run every bench target.
    run_all: bool,
    /// `--budgets <FILE>`: budget table.
    budgets: PathBuf,
}

/// Latency budgets in nanoseconds, keyed by benchmark identifier or by a
/// prefix ending in `*`.
#[derive(Debug, Clone, Default, PartialEq)]
struct Budgets {
    /// Exact identifiers.
    exact: BTreeMap<String, f64>,
    /// Prefixes (the key without its trailing `*`).
    prefixes: BTreeMap<String, f64>,
}

/// On-disk form: `[budgets]` maps keys to durations such as `"10 us"`.
#[derive(Debug, serde::Deserialize)]
struct BudgetFile {
    /// Key to duration string.
    budgets: BTreeMap<String, String>,
}

impl Budgets {
    /// Parses a budget file's contents.
    fn parse(text: &str) -> Result<Self, String> {
        let file: BudgetFile =
            toml::from_str(text).map_err(|e| format!("budget file: {e}"))?;
        let mut budgets = Self::default();
        for (key, value) in file.budgets {
            let ns = parse_duration_ns(&value).ok_or_else(|| {
                format!("budget '{key}': unparseable duration '{value}'")
            })?;
            match key.strip_suffix('*') {
                Some(prefix) => budgets.prefixes.insert(prefix.to_string(), ns),
                None => budgets.exact.insert(key, ns),
            };
        }
        if budgets.exact.is_empty() && budgets.prefixes.is_empty() {
            return Err("budget file declares no budgets".to_string());
        }
        Ok(budgets)
    }

    /// Budget for `id`: the exact key, else the longest matching prefix.
    fn for_id(&self, id: &str) -> Option<f64> {
        self.exact.get(id).copied().or_else(|| {
            self.prefixes
                .iter()
                .filter(|(prefix, _)| id.starts_with(prefix.as_str()))
                .max_by_key(|(prefix, _)| prefix.len())
                .map(|(_, ns)| *ns)
        })
    }
}

impl Verdict {
    /// Maps one of Criterion's verdict lines to its verdict.
    fn from_line(line: &str) -> Option<Self> {
        match line {
            "Performance has regressed." => Some(Self::Regressed),
            "Performance has improved." => Some(Self::Improved),
            "No change in performance detected." => Some(Self::NoChange),
            "Change within noise threshold." => Some(Self::WithinNoise),
            _ => None,
        }
    }

    /// Matrix label.
    const fn label(self) -> &'static str {
        match self {
            Self::Regressed => "regressed",
            Self::Improved => "improved",
            Self::NoChange => "no change",
            Self::WithinNoise => "within noise",
        }
    }
}

/// Parses `"<number> <unit>"` (`ns`, `us`, `µs`, `ms`, `s`) into nanoseconds.
fn parse_duration_ns(text: &str) -> Option<f64> {
    let text = text.trim();
    let split = text
        .find(|c: char| !(c.is_ascii_digit() || c == '.'))
        .unwrap_or(text.len());
    let (number, unit) = text.split_at(split);
    let value: f64 = number.parse().ok()?;
    let scale = match unit.trim() {
        "ns" => 1.0,
        "us" | "µs" => 1e3,
        "ms" => 1e6,
        "s" => 1e9,
        _ => return None,
    };
    (value > 0.0).then_some(value * scale)
}

fn parse_cli_args() -> Result<CliOptions, String> {
    let mut args = std::env::args().skip(1);
    let mut bench_target = None;
    let mut run_all = false;
    let mut budgets = PathBuf::from(".cargo/regression.toml");

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--all" => {
                run_all = true;
            }
            "--bench" => {
                if let Some(target) = args.next() {
                    bench_target = Some(target);
                } else {
                    return Err("Missing argument for --bench".to_string());
                }
            }
            "--budgets" => {
                budgets = args
                    .next()
                    .map(PathBuf::from)
                    .ok_or("Missing argument for --budgets")?;
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            unknown => {
                return Err(format!("Unknown option: {unknown}"));
            }
        }
    }

    Ok(CliOptions {
        bench_target,
        run_all,
        budgets,
    })
}

fn print_help() {
    println!(
        "Usage: cargo regression [OPTIONS]\n\n\
         Options:\n  \
           --bench <NAME>        Run only the specified benchmark target (e.g. jitter, scaling)\n  \
           --all                 Run all workspace benchmark targets\n  \
           --budgets <FILE>      Budget table [default: .cargo/regression.toml]\n  \
           -h, --help            Print help information"
    );
}

fn format_duration(ns: f64) -> String {
    if ns < 1_000.0 {
        format!("{ns:.2} ns")
    } else if ns < 1_000_000.0 {
        format!("{:.2} µs", ns / 1_000.0)
    } else if ns < 1_000_000_000.0 {
        format!("{:.2} ms", ns / 1_000_000.0)
    } else {
        format!("{:.2} s", ns / 1_000_000_000.0)
    }
}

/// Runs `cargo bench`, echoing its standard output, and returns that output.
fn run_benchmarks(root: &Path, opts: &CliOptions) -> Result<String, String> {
    let mut cmd = Command::new("cargo");
    cmd.current_dir(root);
    cmd.arg("bench");
    cmd.stdout(Stdio::piped());

    if opts.run_all {
        println!("Executing all Criterion benchmarks (cargo bench)...");
    } else if let Some(ref target) = opts.bench_target {
        println!(
            "Executing Criterion benchmark target '{target}' (cargo bench --bench {target})..."
        );
        cmd.args(["--bench", target]);
    } else {
        println!(
            "Executing default Criterion benchmark target 'jitter' (cargo bench --bench jitter)..."
        );
        cmd.args(["--bench", "jitter"]);
    }

    let mut child = cmd
        .spawn()
        .map_err(|e| format!("Failed to spawn cargo bench: {e}"))?;

    let mut output = String::new();
    if let Some(stdout) = child.stdout.take() {
        for line in BufReader::new(stdout).lines() {
            let line = line.map_err(|e| {
                format!("Failed to read cargo bench output: {e}")
            })?;
            println!("{line}");
            output.push_str(&line);
            output.push('\n');
        }
    }

    let status = child
        .wait()
        .map_err(|e| format!("Failed to wait for cargo bench: {e}"))?;

    if status.success() {
        println!("Benchmark execution completed successfully.\n");
        Ok(output)
    } else {
        Err(format!(
            "cargo bench failed with exit code {:?}",
            status.code()
        ))
    }
}

/// Parses Criterion's standard output into one result per benchmark.
///
/// Criterion prints an identifier longer than 23 characters on its own line,
/// followed by an indented `time:` line. Fails on a `time:` or `change:` line
/// that does not parse, and when the output holds no benchmark.
fn parse_criterion_output(output: &str) -> Result<Results, String> {
    let mut results = Results::new();
    let mut previous = "";

    for line in output.lines() {
        let trimmed = line.trim();
        if let Some((head, interval)) = line.split_once("time:") {
            let id = if head.trim().is_empty() {
                previous.trim()
            } else {
                head.trim()
            };
            let time_ns = parse_time_ns(interval)
                .filter(|_| !id.is_empty())
                .ok_or_else(|| format!("Unparseable Criterion line: {line}"))?;
            results.push(BenchmarkResult {
                id: id.to_string(),
                time_ns,
                change_pct: None,
                verdict: None,
            });
        } else if let Some(result) = results.last_mut() {
            if let Some(interval) = trimmed.strip_prefix("change:") {
                result.change_pct =
                    Some(parse_change_pct(interval).ok_or_else(|| {
                        format!("Unparseable Criterion line: {line}")
                    })?);
            } else if let Some(verdict) = Verdict::from_line(trimmed) {
                result.verdict = Some(verdict);
            }
        }
        previous = line;
    }

    if results.is_empty() {
        return Err("No Criterion benchmark results found".to_string());
    }
    Ok(results)
}

/// Parses the point estimate of a `time:` interval
/// (`[1.2345 µs 1.2400 µs 1.2500 µs]`) into nanoseconds.
fn parse_time_ns(interval: &str) -> Option<f64> {
    let inner = interval.trim().strip_prefix('[')?.split(']').next()?;
    let mut tokens = inner.split_whitespace().skip(2);
    let value: f64 = tokens.next()?.parse().ok()?;
    let scale = match tokens.next()? {
        "ps" => 1e-3,
        "ns" => 1.0,
        "µs" => 1e3,
        "ms" => 1e6,
        "s" => 1e9,
        _ => return None,
    };
    Some(value * scale)
}

/// Parses the point estimate of a `change:` interval
/// (`[-1.2345% +0.5000% +2.1000%]`) in %.
fn parse_change_pct(interval: &str) -> Option<f64> {
    let inner = interval.trim().strip_prefix('[')?.split(']').next()?;
    inner
        .split_whitespace()
        .nth(1)?
        .strip_suffix('%')?
        .parse()
        .ok()
}

/// Returns why a benchmark fails the gate; empty when it passes.
fn failures(result: &BenchmarkResult, budgets: &Budgets) -> Vec<String> {
    let mut reasons = Vec::new();
    match budgets.for_id(&result.id) {
        Some(budget_ns) if result.time_ns > budget_ns => {
            reasons.push(format!(
                "Latency {} exceeded cycle budget of {}",
                format_duration(result.time_ns),
                format_duration(budget_ns)
            ));
        }
        Some(_) => {}
        None => reasons.push("No budget registered".to_string()),
    }
    if result.verdict == Some(Verdict::Regressed) {
        reasons.push("Criterion reports a performance regression".to_string());
    }
    reasons
}

/// Prints the performance and regression matrix.
fn print_matrix(results: &[BenchmarkResult], budgets: &Budgets) {
    println!("--- Benchmark Performance & Regression Matrix ---");
    println!(
        "{:<42} {:>14} {:>10} {:>12} {:>14} {:>8}",
        "Benchmark Identifier",
        "Time",
        "Change",
        "Criterion",
        "Cycle Budget",
        "Status"
    );
    println!("{}", "-".repeat(105));

    for result in results {
        let change = result
            .change_pct
            .map_or_else(|| "-".to_string(), |c| format!("{c:+.2}%"));
        let verdict = result.verdict.map_or("no baseline", Verdict::label);
        let status = if failures(result, budgets).is_empty() {
            "PASS"
        } else {
            "FAIL"
        };

        println!(
            "{:<42} {:>14} {:>10} {:>12} {:>14} {:>8}",
            result.id,
            format_duration(result.time_ns),
            change,
            verdict,
            budgets
                .for_id(&result.id)
                .map_or_else(|| "none".to_string(), format_duration),
            status
        );
    }

    println!("{}", "-".repeat(105));
}

fn main() {
    println!("=== control-rs Performance Regression & Budget Harness ===");

    let opts = match parse_cli_args() {
        Ok(o) => o,
        Err(e) => {
            eprintln!("Error: {e}\n");
            print_help();
            std::process::exit(1);
        }
    };

    let budgets = match std::fs::read_to_string(&opts.budgets)
        .map_err(|e| format!("{}: {e}", opts.budgets.display()))
        .and_then(|text| Budgets::parse(&text))
    {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    };
    let root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    println!("Workspace root: {}", root.display());

    let results = match run_benchmarks(&root, &opts)
        .and_then(|output| parse_criterion_output(&output))
    {
        Ok(results) => results,
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    };

    print_matrix(&results, &budgets);
    let failed: Vec<&BenchmarkResult> = results
        .iter()
        .filter(|result| !failures(result, &budgets).is_empty())
        .collect();

    if !failed.is_empty() {
        eprintln!(
            "\nPerformance verification FAILED: {}/{} benchmark(s) breached latency budgets or regressed.",
            failed.len(),
            results.len()
        );
        for result in failed {
            eprintln!(
                "  - '{}': {}",
                result.id,
                failures(result, &budgets).join("; ")
            );
        }
        std::process::exit(1);
    }

    println!(
        "\nPerformance verification PASSED: All {} benchmark(s) satisfied latency budgets without regressions.",
        results.len()
    );
    std::process::exit(0);
}

#[cfg(test)]
mod tests {
    use super::{
        Budgets, Results, Verdict, failures, format_duration,
        parse_criterion_output, parse_duration_ns,
    };

    const BUDGETS: &str = r#"
[budgets]
"jitter/hilbert_32x32_solve" = "100 us"
"jitter/*" = "50 us"
"state_space_scaling/zoh_dim/128" = "100 ms"
"state_space_scaling/*" = "50 ms"
"#;

    /// `cargo bench` output: a benchmark without a baseline, then one
    /// benchmark per Criterion verdict. Identifiers longer than 23 characters
    /// sit on their own line.
    const OUTPUT: &str = "
running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

jitter/short            time:   [812.34 ps 815.67 ps 819.01 ps]
Found 3 outliers among 100 measurements (3.00%)
  3 (3.00%) high mild
jitter/hilbert_32x32_solve
                        time:   [41.123 µs 41.456 µs 41.789 µs]
                        change: [+20.123% +21.456% +22.789%] (p = 0.00 < 0.05)
                        Performance has regressed.
jitter/fast_path        time:   [98.765 ns 99.012 ns 99.345 ns]
                        change: [-30.123% -29.456% -28.789%] (p = 0.00 < 0.05)
                        Performance has improved.
state_space_scaling/zoh_dim/32
                        time:   [1.2340 ms 1.2345 ms 1.2350 ms]
                        change: [-1.2345% +0.1234% +1.5678%] (p = 0.45 > 0.05)
                        No change in performance detected.
state_space_scaling/zoh_dim/128
                        time:   [1.2340 s 1.2345 s 1.2350 s]
                        change: [+1.0123% +2.3456% +3.6789%] (p = 0.01 < 0.05)
                        Change within noise threshold.
";

    fn budgets() -> Budgets {
        Budgets::parse(BUDGETS).unwrap()
    }

    const fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-9 * b.abs()
    }

    fn parsed() -> Results {
        parse_criterion_output(OUTPUT).unwrap()
    }

    #[test]
    fn parses_identifiers_verdicts_and_changes() {
        let results = parsed();
        let ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
        assert_eq!(
            ids,
            [
                "jitter/short",
                "jitter/hilbert_32x32_solve",
                "jitter/fast_path",
                "state_space_scaling/zoh_dim/32",
                "state_space_scaling/zoh_dim/128",
            ]
        );
        assert_eq!(
            results.iter().map(|r| r.verdict).collect::<Vec<_>>(),
            [
                None,
                Some(Verdict::Regressed),
                Some(Verdict::Improved),
                Some(Verdict::NoChange),
                Some(Verdict::WithinNoise),
            ]
        );
        let changes = [
            None,
            Some(21.456),
            Some(-29.456),
            Some(0.1234),
            Some(2.3456),
        ];
        for (result, want) in results.iter().zip(changes) {
            match (result.change_pct, want) {
                (None, None) => {}
                (Some(got), Some(want)) => assert!(close(got, want)),
                other => panic!("{}: {other:?}", result.id),
            }
        }
    }

    #[test]
    fn converts_time_units_to_nanoseconds() {
        let expected = [815.67e-3, 41.456e3, 99.012, 1.2345e6, 1.2345e9];
        for (result, want) in parsed().iter().zip(expected) {
            assert!(close(result.time_ns, want), "{}: {}", result.id, want);
        }
    }

    #[test]
    fn rejects_output_without_benchmarks() {
        assert!(parse_criterion_output("").is_err());
        assert!(parse_criterion_output("running 0 tests\n").is_err());
    }

    #[test]
    fn rejects_unparseable_lines() {
        assert!(parse_criterion_output("a/b time: [1.0 hours]\n").is_err());
        let bad_change = "a/b time: [1.0 ns 2.0 ns 3.0 ns]\n change: [?]\n";
        assert!(parse_criterion_output(bad_change).is_err());
        assert!(parse_criterion_output("  time: [1 ns 2 ns 3 ns]\n").is_err());
    }

    #[test]
    fn fails_on_regression_or_budget_only() {
        let b = budgets();
        let reasons: Vec<usize> =
            parsed().iter().map(|r| failures(r, &b).len()).collect();
        // Regressed fails; the 1.23 s benchmark exceeds its 100 ms budget.
        assert_eq!(reasons, [0, 1, 0, 0, 1]);
    }

    #[test]
    fn resolves_exact_before_longest_prefix() {
        let b = budgets();
        assert_eq!(b.for_id("jitter/hilbert_32x32_solve"), Some(100e3));
        assert_eq!(b.for_id("jitter/other"), Some(50e3));
        assert_eq!(b.for_id("state_space_scaling/zoh_dim/128"), Some(100e6));
        assert_eq!(b.for_id("state_space_scaling/zoh_dim/32"), Some(50e6));
        assert_eq!(b.for_id("tensor/unknown"), None);
    }

    #[test]
    fn unregistered_benchmark_fails() {
        let b = Budgets::parse("[budgets]\n\"jitter/*\" = \"1 s\"\n").unwrap();
        let unknown = parse_criterion_output(
            "other/bench             time:   [1.0 ns 2.0 ns 3.0 ns]\n",
        )
        .unwrap();
        assert_eq!(
            failures(unknown.first().unwrap(), &b),
            ["No budget registered"]
        );
    }

    #[test]
    fn rejects_bad_budget_files() {
        assert!(Budgets::parse("[budgets]\n").is_err());
        assert!(Budgets::parse("[budgets]\n\"a\" = \"10 hours\"\n").is_err());
        assert!(Budgets::parse("[budgets]\n\"a\" = \"0 ns\"\n").is_err());
        assert!(Budgets::parse("budgets = 3").is_err());
    }

    #[test]
    fn parses_duration_units() {
        assert_eq!(parse_duration_ns("10 us"), Some(10e3));
        assert_eq!(parse_duration_ns("10µs"), Some(10e3));
        assert_eq!(parse_duration_ns("1.5 ms"), Some(1.5e6));
        assert_eq!(parse_duration_ns("2 s"), Some(2e9));
        assert_eq!(parse_duration_ns("ns"), None);
    }

    #[test]
    fn formats_seconds() {
        assert_eq!(format_duration(2.5e9), "2.50 s");
    }
}
