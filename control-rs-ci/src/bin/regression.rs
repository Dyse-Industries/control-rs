//! Criterion performance regression evaluator and benchmark budget harness.
//!
//! Runs `cargo bench`, echoes Criterion's output and parses it. Each
//! benchmark's `time:` point estimate is checked against a real-time latency
//! budget (for example, 10 µs jitter for flight control loops). Criterion
//! compares every benchmark with its `base/` sample and prints a verdict; a
//! `regressed` verdict fails the gate. A benchmark without a baseline is only
//! checked against its budget.

#![allow(missing_docs)]

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

#[derive(Debug, Clone)]
struct CliOptions {
    bench_target: Option<String>,
    run_all: bool,
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

fn parse_cli_args() -> Result<CliOptions, String> {
    let mut args = std::env::args().skip(1);
    let mut bench_target = None;
    let mut run_all = false;

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
    })
}

fn print_help() {
    println!(
        "Usage: cargo regression [OPTIONS]\n\n\
         Options:\n  \
           --bench <NAME>        Run only the specified benchmark target (e.g. jitter, scaling)\n  \
           --all                 Run all workspace benchmark targets\n  \
           -h, --help            Print help information"
    );
}

fn find_workspace_root() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if let Some(parent) = manifest_dir.parent()
        && parent.join("Cargo.toml").exists()
    {
        return parent.to_path_buf();
    }
    manifest_dir
}

fn budget_for_benchmark(id: &str) -> f64 {
    match id {
        "jitter/state_space_step_response"
        | "jitter/ekf_8x8_covariance_update" => 10_000.0, // 10 µs
        "jitter/hilbert_32x32_solve" => 100_000.0, // 100 µs
        s if s.starts_with("jitter/") => 50_000.0, // 50 µs
        s if s.starts_with("state_space_scaling/zoh_dim/128") => 100_000_000.0, // 100 ms
        s if s.starts_with("state_space_scaling/") => 50_000_000.0, // 50 ms
        s if s.starts_with("matrix_inversion_scaling/") => 10_000_000.0, // 10 ms
        s if s.starts_with("tensor_contraction_scaling/") => 10_000_000.0, // 10 ms
        s if s.starts_with("polynomial_evaluation_scaling/") => 1_000_000.0, // 1 ms
        _ => 100_000_000.0, // 100 ms default budget
    }
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
fn failures(result: &BenchmarkResult) -> Vec<String> {
    let budget_ns = budget_for_benchmark(&result.id);
    let mut reasons = Vec::new();
    if result.time_ns > budget_ns {
        reasons.push(format!(
            "Latency {} exceeded cycle budget of {}",
            format_duration(result.time_ns),
            format_duration(budget_ns)
        ));
    }
    if result.verdict == Some(Verdict::Regressed) {
        reasons.push("Criterion reports a performance regression".to_string());
    }
    reasons
}

/// Prints the performance and regression matrix.
fn print_matrix(results: &[BenchmarkResult]) {
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
        let status = if failures(result).is_empty() {
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
            format_duration(budget_for_benchmark(&result.id)),
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

    let root = find_workspace_root();
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

    print_matrix(&results);
    let failed: Vec<&BenchmarkResult> = results
        .iter()
        .filter(|result| !failures(result).is_empty())
        .collect();

    if !failed.is_empty() {
        eprintln!(
            "\nPerformance verification FAILED: {}/{} benchmark(s) breached latency budgets or regressed.",
            failed.len(),
            results.len()
        );
        for result in failed {
            eprintln!("  - '{}': {}", result.id, failures(result).join("; "));
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
        Results, Verdict, failures, format_duration, parse_criterion_output,
    };

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
        let reasons: Vec<usize> =
            parsed().iter().map(|r| failures(r).len()).collect();
        // Regressed fails; the 1.23 s benchmark exceeds its 100 ms budget.
        assert_eq!(reasons, [0, 1, 0, 0, 1]);
    }

    #[test]
    fn formats_seconds() {
        assert_eq!(format_duration(2.5e9), "2.50 s");
    }
}
