//! Continuous integration test runner and quality-gate orchestrator for `control-rs`.

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fmt;
use std::fs;
use std::iter::Peekable;
use std::path::{Path, PathBuf};
use std::process::exit;
use std::time::Instant;

use control_rs_ets_host::error::HostError;
use control_rs_ets_host::runner::{EtsRunResult, TestOutcome};
use control_rs_ets_host::target::{SubprocessTarget, Target};

use crate::quality_gate::{
    CargoArgvGate, CoverageGate, EtsGate, HostToolGate, TraceGate,
    ValidateGate, run_gate,
};
use crate::report::{CiSkip, GateVerdict};
use crate::{gates, report, validate};

pub use report::CiOptions;

type EtsAssemble = (bool, Vec<TestOutcome>, Vec<TargetEtsOwned>);
type GateEntry = (Gate, GateMode);
type HostRunMap = BTreeMap<Gate, HostToolRun>;
pub(crate) type EtsTargetResult = Result<EtsRunResult, HostError>;
pub(crate) type MatrixEntry = (String, EtsTargetResult);
/// A `cargo <subcommand>` invocation: display label plus argv.
pub(crate) type CargoInvocation<'a> = (&'a str, &'a [&'a str]);
type ParseArgsResult = Result<Option<CiConfig>, String>;
type TargetEtsOwned = (String, Result<Vec<TestOutcome>, String>);
type TargetEtsRef<'a> = (&'a str, Result<Vec<TestOutcome>, String>);

/// Enumeration of all pipeline quality gates in execution order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Gate {
    /// Cargo clean gate (`cargo clean`).
    Clean = 1,
    /// Format gate (`cargo fmt`).
    Fmt = 2,
    /// Clippy linter gate (`cargo clippy`).
    Clippy = 3,
    /// Cargo check gate (`cargo check`).
    Check = 4,
    /// Cargo build gate (`cargo build`).
    Build = 5,
    /// Unit test gate (`cargo test`).
    Test = 6,
    /// Coverage gate (`cargo tarpaulin`).
    Coverage = 7,
    /// Mutation testing gate (`cargo mutants`).
    Mutants = 8,
    /// Miri UB detection gate (`cargo miri test`).
    Miri = 9,
    /// Valgrind memory gate (`valgrind cargo test`).
    Valgrind = 10,
    /// Coverage-guided fuzzing gate (`cargo fuzz`).
    Fuzz = 11,
    /// Lockbud deadlock / atomicity gate (`cargo lockbud`).
    Lockbud = 12,
    /// Supply-chain license/ban/advisory gate (`cargo deny`).
    Deny = 13,
    /// Advisory audit gate (`cargo audit`).
    Audit = 14,
    /// Kani model-checking gate (`cargo kani`).
    Kani = 15,
    /// Requirement traceability gate.
    Trace = 16,
    /// Target execution matrix gate (ETS).
    Ets = 17,
    /// Multi-example validation gate.
    Validate = 18,
}

/// Per-gate policy in `gate.toml`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateMode {
    /// Do not run the gate.
    Skip,
    /// Run the gate; a failure is reported and does not fail the pipeline.
    Warn,
    /// Run the gate; a failure fails the pipeline.
    Fail,
}

/// Parsed structure of `gate.toml` / `ci.toml`.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct CiConfigFile {
    #[serde(alias = "runner", default)]
    ci: CiGeneralConfig,
    #[serde(default)]
    gates: CiGatesConfig,
    #[serde(default)]
    ets: EtsSectionConfig,
    #[serde(alias = "suites", default)]
    examples: Vec<ExampleCrateConfig>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct CiGeneralConfig {
    #[serde(default = "default_title")]
    title: String,
    #[serde(default = "default_out_dir")]
    out_dir: String,
    #[serde(default = "default_timeout")]
    timeout_secs: u64,
}

/// Ordered `[gates]` table: key order is the pipeline, omitted gates are skip.
#[derive(Debug, Clone, Default)]
struct CiGatesConfig {
    entries: Vec<GateEntry>,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct EtsSectionConfig {
    #[serde(default)]
    targets: Vec<EtsTargetItem>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct EtsTargetItem {
    name: Option<String>,
    #[serde(alias = "path", default = "default_manifest_path")]
    manifest_path: String,
    target: String,
    bin: String,
    #[serde(default)]
    args: Vec<String>,
}

type ExampleCrateConfig = validate::ExampleTargetConfig;

/// CLI boolean flags packed to stay under `struct_excessive_bools`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct CliFlags(u8);

/// Parsed CLI argument options.
#[derive(Debug, Clone, Default)]
struct CliArgs {
    manifest_path: Option<String>,
    config_path: Option<String>,
    out_dir: Option<String>,
    title: Option<String>,
    timeout_secs: Option<u64>,
    flags: CliFlags,
    filter_target: Option<String>,
    filter_bin: Option<String>,
    serial_port: Option<String>,
    skipped_gates: Vec<Gate>,
    only_gates: Vec<Gate>,
    up_to_gate: Option<Gate>,
}

/// Active execution configuration merging `gate.toml` / `ci.toml` and CLI flags.
#[derive(Debug, Clone)]
pub struct CiConfig {
    file: CiConfigFile,
    cli: CliArgs,
    config_file_path: PathBuf,
    pub(crate) repo_root: PathBuf,
}

pub(crate) struct GateCounters {
    pub(crate) ci_success: bool,
    pub(crate) passed_gates: usize,
    pub(crate) failed_gate_names: Vec<&'static str>,
    pub(crate) warned_gate_names: Vec<&'static str>,
    pub(crate) skipped_gates: usize,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct HostToolRun {
    pub(crate) verdict: GateVerdict,
    pub(crate) output: String,
    pub(crate) details: String,
    pub(crate) time: f32,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct StandardGateOutput {
    pub(crate) verdict: GateVerdict,
    pub(crate) output: String,
    pub(crate) time: f32,
}

pub(crate) struct PipelineState {
    pub(crate) counters: GateCounters,
    pub(crate) clean: StandardGateOutput,
    pub(crate) fmt: StandardGateOutput,
    pub(crate) clippy: StandardGateOutput,
    pub(crate) check: StandardGateOutput,
    pub(crate) build: StandardGateOutput,
    pub(crate) test_cmd: StandardGateOutput,
    pub(crate) coverage: StandardGateOutput,
    pub(crate) tarp_summary: gates::TarpaulinSummary,
    pub(crate) tarp_output: String,
    pub(crate) host_runs: HostRunMap,
    pub(crate) matrix_results: Vec<MatrixEntry>,
    pub(crate) ets_time: f32,
    pub(crate) trace: StandardGateOutput,
    pub(crate) trace_summary: Option<crate::trace::TraceMatrixSummary>,
    pub(crate) cross_val: StandardGateOutput,
    pub(crate) cross_val_summary:
        Option<crate::validate::CrossComparisonSummary>,
}

pub(crate) struct PipelineCtx<'a> {
    pub(crate) config: &'a CiConfig,
    pub(crate) out_path: &'a Path,
    pub(crate) workspace_dir: &'a Path,
    pub(crate) timeout_secs: u64,
    pub(crate) targets_to_run: &'a [Target],
    pub(crate) example_targets: &'a [validate::ExampleTargetConfig],
    pub(crate) state: &'a mut PipelineState,
}

struct ReportWrite<'a> {
    title: &'a str,
    out_path: &'a Path,
    skipped_standard: &'a [&'a str],
    ci_options: CiOptions,
}

struct Finalize<'a> {
    config: &'a CiConfig,
    title: &'a str,
    out_path: &'a Path,
    pipeline: &'a [GateEntry],
    start_time: Instant,
}

struct SummaryArgs<'a> {
    ci_success: bool,
    passed_gates: usize,
    failed_gate_names: &'a [&'a str],
    warned_gate_names: &'a [&'a str],
    skipped_gates: usize,
    elapsed: &'a str,
}

impl std::str::FromStr for Gate {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "clean" => Ok(Self::Clean),
            "fmt" | "format" => Ok(Self::Fmt),
            "clippy" | "lint" => Ok(Self::Clippy),
            "check" => Ok(Self::Check),
            "build" => Ok(Self::Build),
            "test" | "tests" => Ok(Self::Test),
            "cov" | "coverage" | "tarpaulin" => Ok(Self::Coverage),
            "mutants" | "mutant" | "mutation" => Ok(Self::Mutants),
            "miri" => Ok(Self::Miri),
            "valgrind" => Ok(Self::Valgrind),
            "fuzz" | "fuzzing" => Ok(Self::Fuzz),
            "lockbud" => Ok(Self::Lockbud),
            "deny" => Ok(Self::Deny),
            "audit" => Ok(Self::Audit),
            "kani" => Ok(Self::Kani),
            "trace" | "traceability" => Ok(Self::Trace),
            "ets" | "matrix" | "target" | "qemu" => Ok(Self::Ets),
            "validate" | "examples" | "example" | "compare" => {
                Ok(Self::Validate)
            }
            other => Err(format!("Unknown gate name: '{other}'")),
        }
    }
}

impl Gate {
    /// Canonical gate name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Clean => "clean",
            Self::Fmt => "fmt",
            Self::Clippy => "clippy",
            Self::Check => "check",
            Self::Build => "build",
            Self::Test => "test",
            Self::Coverage => "coverage",
            Self::Mutants => "mutants",
            Self::Miri => "miri",
            Self::Valgrind => "valgrind",
            Self::Fuzz => "fuzz",
            Self::Lockbud => "lockbud",
            Self::Deny => "deny",
            Self::Audit => "audit",
            Self::Kani => "kani",
            Self::Trace => "trace",
            Self::Ets => "ets",
            Self::Validate => "validate",
        }
    }

    /// Command printed on Running / Skipped status lines.
    #[must_use]
    pub const fn command_label(self) -> &'static str {
        match self {
            Self::Clean => "`cargo clean`",
            Self::Fmt => "`cargo fmt --all -- --check`",
            Self::Clippy => {
                "`cargo clippy --workspace --all-targets --all-features -- -D warnings`"
            }
            Self::Check => "`cargo check --workspace`",
            Self::Build => "`cargo build --workspace`",
            Self::Test => "`cargo test --workspace`",
            Self::Coverage => {
                "`cargo tarpaulin --workspace --out Html --out Json`"
            }
            Self::Mutants => "`cargo mutants`",
            Self::Miri => "`cargo miri test -p control-rs --lib`",
            Self::Valgrind => "`valgrind cargo test -p control-rs --lib`",
            Self::Fuzz => "`cargo fuzz`",
            Self::Lockbud => "`cargo lockbud -k all`",
            Self::Deny => "`cargo deny check`",
            Self::Audit => "`cargo audit`",
            Self::Kani => "`cargo kani`",
            Self::Trace => "`cargo trace`",
            Self::Ets => "`cargo qemu`",
            Self::Validate => "`cargo validate`",
        }
    }

    /// Report detail explaining why a gate did not run.
    #[must_use]
    pub fn skip_reason(self) -> String {
        format!("skipped (`--skip {}`)", self.name())
    }

    /// Default monolithic execution order (`ets` before `trace`).
    pub const ALL: [Self; 18] = [
        Self::Clean,
        Self::Fmt,
        Self::Clippy,
        Self::Check,
        Self::Build,
        Self::Test,
        Self::Coverage,
        Self::Mutants,
        Self::Miri,
        Self::Valgrind,
        Self::Fuzz,
        Self::Lockbud,
        Self::Deny,
        Self::Audit,
        Self::Kani,
        Self::Ets,
        Self::Trace,
        Self::Validate,
    ];
}

impl GateMode {
    /// True when the gate is invoked.
    #[must_use]
    pub const fn is_run(self) -> bool {
        !matches!(self, Self::Skip)
    }
}

impl fmt::Display for GateMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Skip => "skip",
            Self::Warn => "warn",
            Self::Fail => "fail",
        })
    }
}

impl serde::Serialize for GateMode {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> serde::Deserialize<'de> for GateMode {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Self, D::Error> {
        struct Visitor;
        impl serde::de::Visitor<'_> for Visitor {
            type Value = GateMode;

            fn expecting(
                &self,
                formatter: &mut fmt::Formatter<'_>,
            ) -> fmt::Result {
                formatter
                    .write_str("true/false or \"skip\", \"warn\", or \"fail\"")
            }

            fn visit_bool<E: serde::de::Error>(
                self,
                v: bool,
            ) -> Result<Self::Value, E> {
                Ok(if v { GateMode::Fail } else { GateMode::Skip })
            }

            fn visit_str<E: serde::de::Error>(
                self,
                v: &str,
            ) -> Result<Self::Value, E> {
                parse_gate_mode(v).ok_or_else(|| {
                    E::unknown_variant(
                        v,
                        &["skip", "warn", "fail", "true", "false"],
                    )
                })
            }
        }
        deserializer.deserialize_any(Visitor)
    }
}

impl Default for CiGeneralConfig {
    fn default() -> Self {
        Self {
            title: default_title(),
            out_dir: default_out_dir(),
            timeout_secs: default_timeout(),
        }
    }
}

impl serde::Serialize for CiGatesConfig {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(Some(self.entries.len()))?;
        for (gate, mode) in &self.entries {
            map.serialize_entry(gate.name(), mode)?;
        }
        map.end()
    }
}

impl<'de> serde::Deserialize<'de> for CiGatesConfig {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Self, D::Error> {
        struct GatesVisitor;
        impl<'de> serde::de::Visitor<'de> for GatesVisitor {
            type Value = CiGatesConfig;

            fn expecting(
                &self,
                formatter: &mut fmt::Formatter<'_>,
            ) -> fmt::Result {
                formatter.write_str(
                    "a table of gate name = skip|warn|fail (or true/false)",
                )
            }

            fn visit_map<M: serde::de::MapAccess<'de>>(
                self,
                mut map: M,
            ) -> Result<Self::Value, M::Error> {
                let mut entries = Vec::new();
                let mut seen = BTreeSet::new();
                while let Some((key, mode)) =
                    map.next_entry::<String, GateMode>()?
                {
                    let gate: Gate =
                        key.parse().map_err(serde::de::Error::custom)?;
                    if !seen.insert(gate) {
                        return Err(serde::de::Error::custom(format!(
                            "duplicate gate '{key}'"
                        )));
                    }
                    entries.push((gate, mode));
                }
                Ok(CiGatesConfig { entries })
            }
        }
        deserializer.deserialize_map(GatesVisitor)
    }
}

impl CliFlags {
    const FMT: Self = Self(1 << 0);
    const STRICT: Self = Self(1 << 1);
    const VERBOSE: Self = Self(1 << 2);
    const SERIAL: Self = Self(1 << 3);
    const QUIET: Self = Self(1 << 4);

    const fn contains(self, flag: Self) -> bool {
        self.0 & flag.0 != 0
    }

    const fn set(&mut self, flag: Self) {
        self.0 |= flag.0;
    }
}

impl CiConfig {
    /// Pipeline in `[gates]` key order, after CLI overlays.
    fn pipeline(&self) -> Vec<GateEntry> {
        let mut entries = self.file.gates.entries.clone();
        if self.cli.only_gates.is_empty() {
            self.apply_up_to_and_skips(&mut entries);
        } else {
            entries = self.select_only_gates(&entries);
        }
        if self.cli.flags.contains(CliFlags::STRICT) {
            for (_, mode) in &mut entries {
                if *mode == GateMode::Warn {
                    *mode = GateMode::Fail;
                }
            }
        }
        entries
    }

    fn apply_up_to_and_skips(&self, entries: &mut Vec<GateEntry>) {
        if let Some(max_gate) = self.cli.up_to_gate
            && let Some(idx) = entries.iter().position(|(g, _)| *g == max_gate)
        {
            entries.truncate(idx.saturating_add(1));
        }
        for skipped in &self.cli.skipped_gates {
            for (gate, mode) in entries.iter_mut() {
                if gate == skipped {
                    *mode = GateMode::Skip;
                }
            }
        }
    }

    fn select_only_gates(&self, entries: &[GateEntry]) -> Vec<GateEntry> {
        let mut selected = Vec::new();
        for (gate, mode) in entries {
            if self.cli.only_gates.contains(gate) {
                let mode = if *mode == GateMode::Skip {
                    GateMode::Fail
                } else {
                    *mode
                };
                selected.push((*gate, mode));
            }
        }
        for gate in &self.cli.only_gates {
            if !selected.iter().any(|(g, _)| g == gate) {
                selected.push((*gate, GateMode::Fail));
            }
        }
        selected
    }

    /// Effective mode for `gate` (omitted from `[gates]` is skip).
    #[must_use]
    pub fn gate_mode(&self, gate: Gate) -> GateMode {
        self.pipeline()
            .into_iter()
            .find(|(g, _)| *g == gate)
            .map_or(GateMode::Skip, |(_, m)| m)
    }

    /// Determine whether a specific gate is invoked for this execution.
    #[must_use]
    pub fn is_gate_enabled(&self, gate: Gate) -> bool {
        self.gate_mode(gate).is_run()
    }
}

impl Default for GateCounters {
    fn default() -> Self {
        Self {
            ci_success: true,
            passed_gates: 0,
            failed_gate_names: Vec::new(),
            warned_gate_names: Vec::new(),
            skipped_gates: 0,
        }
    }
}

impl Default for PipelineState {
    fn default() -> Self {
        Self {
            counters: GateCounters::default(),
            clean: StandardGateOutput::default(),
            fmt: StandardGateOutput::default(),
            clippy: StandardGateOutput::default(),
            check: StandardGateOutput::default(),
            build: StandardGateOutput::default(),
            test_cmd: StandardGateOutput::default(),
            coverage: StandardGateOutput::default(),
            tarp_summary: gates::TarpaulinSummary::default(),
            tarp_output: String::new(),
            host_runs: BTreeMap::new(),
            matrix_results: Vec::new(),
            ets_time: 0.0,
            trace: StandardGateOutput::default(),
            trace_summary: None,
            cross_val: StandardGateOutput::default(),
            cross_val_summary: None,
        }
    }
}

fn parse_gate_mode(raw: &str) -> Option<GateMode> {
    match raw.to_ascii_lowercase().as_str() {
        "skip" | "off" | "false" => Some(GateMode::Skip),
        "warn" | "warning" | "advisory" => Some(GateMode::Warn),
        "fail" | "required" | "error" | "true" => Some(GateMode::Fail),
        _ => None,
    }
}

fn default_title() -> String {
    "control-rs".to_string()
}

fn default_out_dir() -> String {
    "target/ci".to_string()
}

fn default_manifest_path() -> String {
    ".".to_string()
}

const fn default_timeout() -> u64 {
    90
}

/// Locates repository root by traversing upward until `.git` or `documentation/` exists.
fn find_repo_root(start: &Path) -> PathBuf {
    let mut curr = if start.is_file() {
        start
            .parent()
            .map_or_else(|| PathBuf::from("."), Path::to_path_buf)
    } else {
        start.to_path_buf()
    };
    if let Ok(abs) = curr.canonicalize() {
        curr = abs;
    }
    loop {
        if curr.join(".git").exists()
            || (curr.join("documentation").exists()
                && curr.join("Cargo.toml").exists())
        {
            return curr;
        }
        if !curr.pop() {
            break;
        }
    }
    PathBuf::from(".")
}

/// Load `gate.toml` (or a `--config` path) if present, else defaults.
fn load_ci_config_file(path: &Path) -> Result<CiConfigFile, String> {
    if !path.exists() {
        report::status(
            "Defaults",
            format!("config '{}' not found; using defaults", path.display()),
        );
        return Ok(CiConfigFile::default());
    }
    let content = fs::read_to_string(path).map_err(|e| {
        format!("Failed to read config file '{}': {e}", path.display())
    })?;
    let config = toml::from_str(&content).map_err(|e| {
        format!("Failed to parse config file '{}': {e}", path.display())
    })?;
    report::status("Reading", format!("'{}'", path.display()));
    Ok(config)
}

fn take_value<'a, I: Iterator<Item = &'a String>>(
    iter: &mut Peekable<I>,
    flag: &str,
) -> Result<String, String> {
    iter.next()
        .ok_or_else(|| format!("Missing value for {flag}"))
        .cloned()
}

fn parse_gate_csv(val: &str, dest: &mut Vec<Gate>) -> Result<(), String> {
    for part in val.split(',') {
        let trimmed = part.trim();
        if !trimmed.is_empty() {
            let gate = trimmed.parse::<Gate>()?;
            if !dest.contains(&gate) {
                dest.push(gate);
            }
        }
    }
    Ok(())
}

fn absorb_gate_tokens<'a, I: Iterator<Item = &'a String>>(
    iter: &mut Peekable<I>,
    dest: &mut Vec<Gate>,
) {
    while let Some(peeked) = iter.peek() {
        if peeked.starts_with('-') {
            break;
        }
        if let Ok(gate) = peeked.parse::<Gate>() {
            iter.next();
            if !dest.contains(&gate) {
                dest.push(gate);
            }
        } else {
            break;
        }
    }
}

fn no_flag_gate(arg: &str) -> Option<Gate> {
    Some(match arg {
        "--no-fmt" => Gate::Fmt,
        "--no-clippy" => Gate::Clippy,
        "--no-check" => Gate::Check,
        "--no-build" => Gate::Build,
        "--no-clean" => Gate::Clean,
        "--no-test" => Gate::Test,
        "--no-cov" | "--no-coverage" => Gate::Coverage,
        "--no-mutants" => Gate::Mutants,
        "--no-miri" => Gate::Miri,
        "--no-valgrind" => Gate::Valgrind,
        "--no-fuzz" => Gate::Fuzz,
        "--no-lockbud" => Gate::Lockbud,
        "--no-deny" => Gate::Deny,
        "--no-audit" => Gate::Audit,
        "--no-kani" => Gate::Kani,
        "--no-ets" => Gate::Ets,
        "--no-trace" => Gate::Trace,
        "--no-validate" | "--no-examples" => Gate::Validate,
        _ => return None,
    })
}

fn subcommand_gate(arg: &str) -> Option<Gate> {
    Some(match arg {
        "ets" => Gate::Ets,
        "validate" | "example" | "examples" => Gate::Validate,
        "mutants" => Gate::Mutants,
        "miri" => Gate::Miri,
        "valgrind" => Gate::Valgrind,
        "fuzz" => Gate::Fuzz,
        "lockbud" => Gate::Lockbud,
        "deny" => Gate::Deny,
        "audit" => Gate::Audit,
        "kani" => Gate::Kani,
        _ => return None,
    })
}

fn handle_value_flags<'a, I: Iterator<Item = &'a String>>(
    arg: &str,
    iter: &mut Peekable<I>,
    cli: &mut CliArgs,
) -> Result<bool, String> {
    match arg {
        "--config" => cli.config_path = Some(take_value(iter, "--config")?),
        "--manifest-path" | "--path" => {
            cli.manifest_path = Some(take_value(iter, "--manifest-path")?);
        }
        "--out-dir" => cli.out_dir = Some(take_value(iter, "--out-dir")?),
        "--title" => cli.title = Some(take_value(iter, "--title")?),
        "--timeout" => {
            let val_str = take_value(iter, "--timeout")?;
            cli.timeout_secs = Some(val_str.parse().map_err(|e| {
                format!("Invalid timeout value '{val_str}': {e}")
            })?);
        }
        "--up-to" => {
            cli.up_to_gate =
                Some(take_value(iter, "--up-to")?.parse::<Gate>()?);
        }
        "-t" | "--target" => {
            cli.filter_target = Some(take_value(iter, "--target")?);
        }
        "-b" | "--bin" => cli.filter_bin = Some(take_value(iter, "--bin")?),
        "--port" => {
            cli.serial_port = Some(take_value(iter, "--port")?);
            cli.flags.set(CliFlags::SERIAL);
        }
        _ => return Ok(false),
    }
    Ok(true)
}

fn handle_list_flags<'a, I: Iterator<Item = &'a String>>(
    arg: &str,
    iter: &mut Peekable<I>,
    cli: &mut CliArgs,
) -> Result<bool, String> {
    match arg {
        "--only" => {
            parse_gate_csv(&take_value(iter, "--only")?, &mut cli.only_gates)?;
            absorb_gate_tokens(iter, &mut cli.only_gates);
        }
        "--skip" => {
            parse_gate_csv(
                &take_value(iter, "--skip")?,
                &mut cli.skipped_gates,
            )?;
            absorb_gate_tokens(iter, &mut cli.skipped_gates);
        }
        _ => return Ok(false),
    }
    Ok(true)
}

fn handle_switch_flags(arg: &str, cli: &mut CliArgs) -> bool {
    match arg {
        "--fmt" => cli.flags.set(CliFlags::FMT),
        "--quiet" => cli.flags.set(CliFlags::QUIET),
        "--strict" => cli.flags.set(CliFlags::STRICT),
        "--serial" => cli.flags.set(CliFlags::SERIAL),
        "--verbose" | "-v" => cli.flags.set(CliFlags::VERBOSE),
        _ => return false,
    }
    true
}

fn resolve_config_path(cli: &CliArgs) -> PathBuf {
    if let Some(ref cp) = cli.config_path {
        return PathBuf::from(cp);
    }
    let dir = cli.manifest_path.as_deref().map_or_else(
        || Path::new("."),
        |mp| {
            let p = Path::new(mp);
            if p.is_file() || p.file_name().is_some_and(|n| n == "Cargo.toml") {
                p.parent().unwrap_or_else(|| Path::new("."))
            } else {
                p
            }
        },
    );
    let candidates = [
        dir.join("gate.toml"),
        dir.join("ci.toml"),
        dir.join("control-rs.toml"),
        dir.join("control-rs-ci.toml"),
    ];
    candidates
        .into_iter()
        .find(|c| c.exists())
        .unwrap_or_else(|| dir.join("gate.toml"))
}

/// Parses CLI arguments into [`CiConfig`].
fn parse_args(args: &[String]) -> ParseArgsResult {
    let mut cli = CliArgs::default();
    let mut iter = args.iter().skip(1).peekable();
    while let Some(arg) = iter.next() {
        let arg = arg.as_str();
        if matches!(arg, "-h" | "--help") {
            print_help();
            return Ok(None);
        }
        if handle_value_flags(arg, &mut iter, &mut cli)?
            || handle_list_flags(arg, &mut iter, &mut cli)?
            || handle_switch_flags(arg, &mut cli)
        {
            continue;
        }
        if let Some(gate) = no_flag_gate(arg) {
            cli.skipped_gates.push(gate);
            continue;
        }
        if matches!(arg, "ci" | "gate" | "--") {
            continue;
        }
        if let Some(gate) = subcommand_gate(arg) {
            if !cli.only_gates.contains(&gate) {
                cli.only_gates.push(gate);
            }
            continue;
        }
        return Err(format!("Unknown argument: {arg}"));
    }
    let config_file_path = resolve_config_path(&cli);
    let repo_root = find_repo_root(&config_file_path);
    let file_config = load_ci_config_file(&config_file_path)?;
    if cli.only_gates.is_empty()
        && let Some(up_to) = cli.up_to_gate
        && !file_config.gates.entries.iter().any(|(g, _)| *g == up_to)
    {
        return Err(format!(
            "--up-to {} is not listed in [gates]",
            up_to.name()
        ));
    }
    Ok(Some(CiConfig {
        file: file_config,
        cli,
        config_file_path,
        repo_root,
    }))
}

fn print_help() {
    report::init_color();
    let h = report::HELP_HEADER;
    let f = report::HELP_FLAG;
    let a = report::HELP_ARG;
    anstream::println!(
        "{h}Usage:{h:#} {f}ci{f:#} {a}[SUBCOMMAND]{a:#} {a}[OPTIONS]{a:#}\n       {f}gate{f:#} {a}[SUBCOMMAND]{a:#} {a}[OPTIONS]{a:#}\n\n\
        {h}Subcommands:{h:#}\n  \
          {f}ci{f:#}                         Run workspace continuous integration pipeline (default)\n  \
          {f}gate{f:#}                       Run workspace quality gate pipeline\n  \
          {f}ets{f:#}                        Run Embedded Test Server (ETS) target matrix\n  \
          {f}validate{f:#}                   Run multi-example validation suites\n  \
          {f}mutants{f:#}                    Run cargo-mutants against library unit tests\n  \
          {f}miri{f:#}                       Run Miri undefined-behavior tests\n  \
          {f}valgrind{f:#}                   Run unit tests under Valgrind\n  \
          {f}fuzz{f:#}                       Run time-bounded cargo-fuzz campaigns\n  \
          {f}lockbud{f:#}                    Run lockbud deadlock / atomicity analysis\n  \
          {f}deny{f:#}                       Run cargo-deny supply-chain checks\n  \
          {f}audit{f:#}                      Run cargo-audit advisory scan\n  \
          {f}kani{f:#}                       Run Kani model-checking proofs\n\n\
        {h}Options:{h:#}\n      \
              {f}--config{f:#} {a}<PATH>{a:#}        Path to configuration file [default: gate.toml]\n      \
              {f}--manifest-path{f:#} {a}<PATH>{a:#} Path to target Cargo.toml or crate directory\n      \
              {f}--out-dir{f:#} {a}<DIR>{a:#}        Directory for report output\n      \
              {f}--title{f:#} {a}<TITLE>{a:#}        Report title\n      \
              {f}--timeout{f:#} {a}<SECS>{a:#}       Per-target wall-clock bound\n      \
              {f}--fmt{f:#}                  Apply formatting instead of checking\n      \
              {f}--quiet{f:#}                Capture gate output without streaming it to the terminal\n      \
              {f}--verbose{f:#}, {f}-v{f:#}          Print the reason for an implicitly skipped gate (e.g. ets with no matching targets)\n\n\
        {h}Gate Controls:{h:#}\n      \
              {f}--up-to{f:#} {a}<GATE>{a:#}         Run gates from start up to specified gate (e.g. check, test, trace)\n      \
              {f}--only{f:#} {a}<GATE>...{a:#}       Run only specified gate(s); omitted or skip-listed names run as fail\n      \
              {f}--skip{f:#} {a}<GATE>...{a:#}       Skip specified gate(s), repeatable / comma-separated (e.g. clean, fmt, cov)\n\n\
        {h}Target & Item Filters:{h:#}\n  \
          {f}-t{f:#}, {f}--target{f:#} {a}<TRIPLE>{a:#}      Filter targets to matching cross-compilation triple\n  \
          {f}-b{f:#}, {f}--bin{f:#} {a}<BIN>{a:#}            Filter targets to matching binary name\n      \
              {f}--strict{f:#}               Enforce strict zero-panic / budget violation gating\n      \
              {f}--serial{f:#}               Verify a physical target over serial\n      \
              {f}--port{f:#} {a}<PORT>{a:#}          Serial device path [default: /dev/ttyACM0]\n  \
          {f}-h{f:#}, {f}--help{f:#}                 Print help"
    );
}

fn write_host_tool_report(out_dir: &Path, summary: &gates::HostToolSummary) {
    let path = out_dir.join(format!("{}-report.json", summary.tool));
    if let Ok(json) = serde_json::to_string_pretty(summary) {
        let _ = fs::write(path, json);
    }
}

fn host_tool_row<'a>(
    name: &'a str,
    command: &'a str,
    run: &'a HostToolRun,
) -> report::HostToolRow<'a> {
    report::HostToolRow {
        name,
        command,
        verdict: run.verdict,
        details: &run.details,
        output: &run.output,
        time: run.time,
    }
}

fn store_standard(
    out: &mut StandardGateOutput,
    result: &crate::quality_gate::GateRunResult,
) {
    out.verdict = result.verdict;
    out.output.clone_from(&result.output);
    out.time = result.time;
}

fn store_host_run(
    host_runs: &mut HostRunMap,
    gate: Gate,
    result: &crate::quality_gate::GateRunResult,
) {
    let run = host_runs.entry(gate).or_default();
    run.verdict = result.verdict;
    run.output.clone_from(&result.output);
    run.details.clone_from(&result.details);
    run.time = result.time;
}

/// Runs a `cargo <subcommand>` gate and stores its result into the
/// `StandardGateOutput` field matching `gate`. Covers clean, fmt, clippy,
/// check, build, and test; `gate` must be one of those variants.
fn run_standard_cargo_gate(
    gate: Gate,
    invocation: CargoInvocation<'_>,
    mode: GateMode,
    ctx: &mut PipelineCtx<'_>,
) {
    let (label, argv) = invocation;
    let job = CargoArgvGate {
        label,
        argv,
        envs: &[],
    };
    let result = run_gate(gate, mode, &job, ctx);
    let out = match gate {
        Gate::Clean => &mut ctx.state.clean,
        Gate::Fmt => &mut ctx.state.fmt,
        Gate::Clippy => &mut ctx.state.clippy,
        Gate::Check => &mut ctx.state.check,
        Gate::Build => &mut ctx.state.build,
        Gate::Test => &mut ctx.state.test_cmd,
        _ => return,
    };
    store_standard(out, &result);
}

/// Runs a host-tool gate (custom result parsing for Mutants; plain
/// `cargo <subcommand>` for the rest) and persists both the in-memory
/// `HostToolRun` and its `<gate>-report.json` artifact.
fn run_special_host_gate(
    gate: Gate,
    mode: GateMode,
    ctx: &mut PipelineCtx<'_>,
) {
    let job = HostToolGate { gate };
    let mut result = run_gate(gate, mode, &job, ctx);
    if mode == GateMode::Skip {
        result.details = gate.skip_reason();
    }
    let verdict = result.verdict;
    let details = result.details.clone();
    store_host_run(&mut ctx.state.host_runs, gate, &result);
    write_host_tool_report(
        ctx.out_path,
        &gates::HostToolSummary {
            tool: gate.name().to_string(),
            success: verdict == GateVerdict::Pass
                || verdict == GateVerdict::Skip,
            skipped: verdict == GateVerdict::Skip,
            details,
        },
    );
}

fn run_coverage_gate(mode: GateMode, ctx: &mut PipelineCtx<'_>) {
    let job = CoverageGate;
    let result = run_gate(Gate::Coverage, mode, &job, ctx);
    store_standard(&mut ctx.state.coverage, &result);
}

/// Runs the ETS target matrix. A gate mode of `Skip` behaves as usual; an
/// enabled gate with no matching targets is treated as an implicit skip
/// (only reported when `--verbose` is set) rather than a failure, since an
/// empty target/bin filter is a configuration choice, not a broken build.
fn run_ets_gate(mode: GateMode, ctx: &mut PipelineCtx<'_>) {
    if mode != GateMode::Skip && ctx.targets_to_run.is_empty() {
        if ctx.config.cli.flags.contains(CliFlags::VERBOSE) {
            report::status(
                "Skipped",
                "ets: no targets configured or matching filters",
            );
        }
        ctx.state.counters.skipped_gates =
            ctx.state.counters.skipped_gates.saturating_add(1);
        return;
    }
    let job = EtsGate;
    // EtsGate computes the real pass/fail verdict itself (see its
    // execute()), so run_gate's single counter update is authoritative --
    // no second correction pass is needed once this returns.
    let _ = run_gate(Gate::Ets, mode, &job, ctx);
}

fn run_trace_gate(mode: GateMode, ctx: &mut PipelineCtx<'_>) {
    let job = TraceGate;
    let result = run_gate(Gate::Trace, mode, &job, ctx);
    store_standard(&mut ctx.state.trace, &result);
}

fn run_validate_gate(mode: GateMode, ctx: &mut PipelineCtx<'_>) {
    let job = ValidateGate;
    let result = run_gate(Gate::Validate, mode, &job, ctx);
    let verdict = result.verdict;
    store_standard(&mut ctx.state.cross_val, &result);
    if verdict == GateVerdict::Fail && ctx.example_targets.is_empty() {
        report::error("validation failed: no suites configured in toml");
    }
}

const fn fmt_invocation(ctx: &PipelineCtx<'_>) -> CargoInvocation<'static> {
    if ctx.config.cli.flags.contains(CliFlags::FMT) {
        ("`cargo fmt --all`", &["fmt", "--all"])
    } else {
        (
            Gate::Fmt.command_label(),
            &["fmt", "--all", "--", "--check"],
        )
    }
}

/// Resolves the `cargo <subcommand>` invocation for a standard gate (clean,
/// fmt, clippy, check, build, test). `gate` must be one of those variants.
const fn standard_invocation(
    gate: Gate,
    ctx: &PipelineCtx<'_>,
) -> CargoInvocation<'static> {
    match gate {
        Gate::Clean => (Gate::Clean.command_label(), &["clean"]),
        Gate::Fmt => fmt_invocation(ctx),
        Gate::Clippy => (
            Gate::Clippy.command_label(),
            &[
                "clippy",
                "--workspace",
                "--all-targets",
                "--all-features",
                "--",
                "-D",
                "warnings",
            ],
        ),
        Gate::Check => (Gate::Check.command_label(), &["check", "--workspace"]),
        Gate::Build => (Gate::Build.command_label(), &["build", "--workspace"]),
        Gate::Test => (Gate::Test.command_label(), &["test", "--workspace"]),
        _ => (gate.command_label(), &[]),
    }
}

fn run_one_gate(gate: Gate, mode: GateMode, ctx: &mut PipelineCtx<'_>) {
    match gate {
        Gate::Clean
        | Gate::Fmt
        | Gate::Clippy
        | Gate::Check
        | Gate::Build
        | Gate::Test => {
            let invocation = standard_invocation(gate, ctx);
            run_standard_cargo_gate(gate, invocation, mode, ctx);
        }
        Gate::Mutants
        | Gate::Miri
        | Gate::Valgrind
        | Gate::Fuzz
        | Gate::Lockbud
        | Gate::Deny
        | Gate::Audit
        | Gate::Kani => {
            run_special_host_gate(gate, mode, ctx);
        }
        Gate::Coverage => run_coverage_gate(mode, ctx),
        Gate::Ets => run_ets_gate(mode, ctx),
        Gate::Trace => run_trace_gate(mode, ctx),
        Gate::Validate => run_validate_gate(mode, ctx),
    }
}

/// True when a single ETS target's run completed and every test passed.
pub(crate) fn ets_target_passed(res: &EtsTargetResult) -> bool {
    res.as_ref().is_ok_and(|run_result| {
        run_result.completion
            == control_rs_ets_host::runner::Completion::Drained
            && !run_result.results.is_empty()
            && run_result.results.iter().all(|t| {
                matches!(t.state, control_rs_ets::comms::TestState::Passed)
            })
    })
}

/// True when every configured ETS target passed (the real, post-hoc verdict
/// used by both `EtsGate::execute`'s counter update and the report tables).
pub(crate) fn ets_all_passed(matrix_results: &[MatrixEntry]) -> bool {
    matrix_results.iter().all(|(_, res)| ets_target_passed(res))
}

fn assemble_ets_results(state: &PipelineState) -> EtsAssemble {
    let all_ets_passed = ets_all_passed(&state.matrix_results);
    let mut combined_ets_tests = Vec::new();
    let mut target_ets_results = Vec::new();
    for (name, res) in &state.matrix_results {
        match res {
            Ok(run_result) => {
                for t in &run_result.results {
                    let mut tagged = t.clone();
                    tagged.suite_name =
                        format!("{} ({})", tagged.suite_name, name);
                    combined_ets_tests.push(tagged);
                }
                target_ets_results
                    .push((name.clone(), Ok(run_result.results.clone())));
            }
            Err(e) => {
                target_ets_results.push((name.clone(), Err(e.to_string())));
            }
        }
    }
    (all_ets_passed, combined_ets_tests, target_ets_results)
}

fn ci_skip_flags(config: &CiConfig) -> CiSkip {
    let mut skip = CiSkip::default();
    if !config.is_gate_enabled(Gate::Coverage) {
        skip.set(CiSkip::COV);
    }
    if !config.is_gate_enabled(Gate::Trace) {
        skip.set(CiSkip::TRACE);
    }
    if !config.is_gate_enabled(Gate::Validate) {
        skip.set(CiSkip::EXAMPLES);
    }
    if !config.is_gate_enabled(Gate::Ets) {
        skip.set(CiSkip::ETS);
    }
    skip
}

fn write_json_reports(
    out_path: &Path,
    state: &PipelineState,
    combined_ets_tests: &[TestOutcome],
) {
    if let Some(summary) = state.trace_summary.as_ref() {
        let json_path = out_path.join("trace-report.json");
        if let Ok(json) = serde_json::to_string_pretty(summary) {
            let _ = fs::write(&json_path, &json);
        }
        let md_path = out_path.join("trace-report.md");
        let brief = crate::trace::render_trace_brief(summary);
        let _ = fs::write(&md_path, &brief);
        report::status(
            "Saved",
            format!("trace report to {}", json_path.display()),
        );
    }
    if let Some(summary) = state.cross_val_summary.as_ref() {
        let json_path = out_path.join("validate-report.json");
        let legacy_json_path = out_path.join("cross-val-report.json");
        if let Ok(json) = serde_json::to_string_pretty(summary) {
            let _ = fs::write(&json_path, &json);
            let _ = fs::write(&legacy_json_path, &json);
            report::status(
                "Saved",
                format!("validation report to {}", json_path.display()),
            );
        }
    }
    let json_path = out_path.join("ets-results.json");
    if combined_ets_tests.is_empty() {
        let _ = fs::write(&json_path, "[]");
        report::status(
            "Saved",
            format!("empty ETS results to {}", json_path.display()),
        );
    } else if let Ok(json) = serde_json::to_string_pretty(combined_ets_tests) {
        let _ = fs::write(&json_path, &json);
        report::status(
            "Saved",
            format!("ETS results to {}", json_path.display()),
        );
    }
}

fn skipped_host_run(gate: Gate) -> HostToolRun {
    HostToolRun {
        verdict: GateVerdict::Skip,
        output: String::new(),
        details: gate.skip_reason(),
        time: 0.0,
    }
}

fn take_host(host_runs: &mut HostRunMap, gate: Gate) -> HostToolRun {
    host_runs
        .remove(&gate)
        .unwrap_or_else(|| skipped_host_run(gate))
}

fn collect_ets_targets(config: &CiConfig) -> Vec<Target> {
    let mut targets_to_run = Vec::new();
    for ets_item in &config.file.ets.targets {
        if let Some(ref ft) = config.cli.filter_target
            && &ets_item.target != ft
        {
            continue;
        }
        if let Some(ref fb) = config.cli.filter_bin
            && &ets_item.bin != fb
        {
            continue;
        }

        let path_str = config
            .cli
            .manifest_path
            .as_ref()
            .unwrap_or(&ets_item.manifest_path);
        let mut sub = SubprocessTarget::new(path_str)
            .with_target(&ets_item.target)
            .with_bin(&ets_item.bin);
        if let Some(ref name) = ets_item.name {
            sub = sub.with_name(name);
        }
        for arg in &ets_item.args {
            sub = sub.with_arg(arg);
        }
        targets_to_run.push(Target::Subprocess(sub));
    }

    if config.cli.flags.contains(CliFlags::SERIAL) {
        let port = config
            .cli
            .serial_port
            .clone()
            .unwrap_or_else(|| "/dev/ttyACM0".to_string());
        targets_to_run.push(Target::Serial {
            port,
            baud: 115_200,
        });
    }
    targets_to_run
}

fn collect_example_targets(
    config: &CiConfig,
    config_dir: &Path,
) -> Vec<validate::ExampleTargetConfig> {
    let mut configured_suites = config.file.examples.clone();
    if configured_suites.is_empty() {
        let val_path = config_dir.join("validate.toml");
        if let Some(val_cfg) = validate::load_validate_config(&val_path) {
            configured_suites = val_cfg.suites;
        } else {
            let root_val = config.repo_root.join("validate.toml");
            if let Some(val_cfg) = validate::load_validate_config(&root_val) {
                configured_suites = val_cfg.suites;
            }
        }
    }

    configured_suites
        .into_iter()
        .map(|mut target| {
            let raw_p = Path::new(&target.manifest_path);
            let resolved = if raw_p.is_relative() {
                config_dir.join(raw_p)
            } else {
                raw_p.to_path_buf()
            };
            target.manifest_path = resolved.to_string_lossy().to_string();
            target
        })
        .collect()
}

/// Main entry point for the CI / quality-gate runner.
pub fn run() {
    // SAFETY: Called before spawning threads.
    unsafe {
        env::set_var("RUST_BACKTRACE", "full");
    }
    let args: Vec<String> = env::args().collect();
    exit(execute(&args));
}

/// Parses `args`, runs the configured pipeline, and returns the process exit
/// code (0 on success, 1 on a parse error or a failed gate). Never exits the
/// process itself, so the whole pipeline can be driven from a test.
fn execute(args: &[String]) -> i32 {
    let start_time = Instant::now();
    report::init_color();
    let config = match parse_args(args) {
        Ok(Some(cfg)) => cfg,
        Ok(None) => return 0,
        Err(e) => {
            report::error(e);
            return 1;
        }
    };
    gates::set_capture_only(config.cli.flags.contains(CliFlags::QUIET));
    finish_run(&config, start_time)
}

fn finish_run(config: &CiConfig, start_time: Instant) -> i32 {
    let title = config
        .cli
        .title
        .as_ref()
        .unwrap_or(&config.file.ci.title)
        .clone();
    let out_dir = config
        .cli
        .out_dir
        .as_ref()
        .unwrap_or(&config.file.ci.out_dir)
        .clone();
    let timeout_secs = config
        .cli
        .timeout_secs
        .unwrap_or(config.file.ci.timeout_secs);
    let out_path = Path::new(&out_dir);
    let _ = fs::create_dir_all(out_path);
    let workspace_dir = config
        .config_file_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let config_dir = config
        .config_file_path
        .parent()
        .unwrap_or_else(|| Path::new("."));
    let targets_to_run = collect_ets_targets(config);
    let example_targets = collect_example_targets(config, config_dir);
    let mut state = PipelineState::default();
    let pipeline = config.pipeline();
    {
        let mut ctx = PipelineCtx {
            config,
            out_path,
            workspace_dir,
            timeout_secs,
            targets_to_run: &targets_to_run,
            example_targets: &example_targets,
            state: &mut state,
        };
        for &(gate, mode) in &pipeline {
            run_one_gate(gate, mode, &mut ctx);
        }
    }
    finalize_report(
        &Finalize {
            config,
            title: &title,
            out_path,
            pipeline: &pipeline,
            start_time,
        },
        state,
    )
}

fn skipped_standard_gates(pipeline: &[GateEntry]) -> Vec<&'static str> {
    [
        Gate::Clean,
        Gate::Fmt,
        Gate::Clippy,
        Gate::Check,
        Gate::Build,
        Gate::Test,
    ]
    .into_iter()
    .filter(|g| !pipeline.iter().any(|(pg, m)| pg == g && m.is_run()))
    .map(Gate::name)
    .collect()
}

fn format_summary_message(args: &SummaryArgs<'_>) -> String {
    let failed_gates = args.failed_gate_names.len();
    let warned_gates = args.warned_gate_names.len();
    let warn_detail = if warned_gates == 0 {
        format!("{warned_gates} warned")
    } else {
        format!(
            "{warned_gates} warned ({})",
            args.warned_gate_names.join(", ")
        )
    };
    if args.ci_success {
        format!(
            "{} gates passed, {failed_gates} failed, {warn_detail}, {} skipped in {}",
            args.passed_gates, args.skipped_gates, args.elapsed
        )
    } else {
        let fail_detail = if args.failed_gate_names.is_empty() {
            format!("{failed_gates} failed")
        } else {
            format!(
                "{failed_gates} failed ({})",
                args.failed_gate_names.join(", ")
            )
        };
        format!(
            "{} gates passed, {fail_detail}, {warn_detail}, {} skipped in {}",
            args.passed_gates, args.skipped_gates, args.elapsed
        )
    }
}

fn finalize_report(args: &Finalize<'_>, mut state: PipelineState) -> i32 {
    // EtsGate::execute already applies the real pass/fail verdict to
    // state.counters via run_gate's single counting path -- no second
    // correction pass here.
    let (_, combined_ets_tests, target_ets_results) =
        assemble_ets_results(&state);
    let target_ets_refs: Vec<TargetEtsRef<'_>> = target_ets_results
        .iter()
        .map(|(name, res)| (name.as_str(), res.clone()))
        .collect();
    let skipped_standard = skipped_standard_gates(args.pipeline);
    let ci_options = CiOptions {
        fmt: args.config.cli.flags.contains(CliFlags::FMT),
        skip: ci_skip_flags(args.config),
    };
    write_ci_report(
        &ReportWrite {
            title: args.title,
            out_path: args.out_path,
            skipped_standard: &skipped_standard,
            ci_options,
        },
        &mut state,
        &target_ets_refs,
    );
    write_json_reports(args.out_path, &state, &combined_ets_tests);
    let elapsed = report::format_elapsed(args.start_time.elapsed());
    let summary_msg = format_summary_message(&SummaryArgs {
        ci_success: state.counters.ci_success,
        passed_gates: state.counters.passed_gates,
        failed_gate_names: &state.counters.failed_gate_names,
        warned_gate_names: &state.counters.warned_gate_names,
        skipped_gates: state.counters.skipped_gates,
        elapsed: &elapsed,
    });
    if state.counters.ci_success {
        report::status("Finished", summary_msg);
        return 0;
    }
    report::error(summary_msg);
    1
}

fn write_ci_report(
    args: &ReportWrite<'_>,
    state: &mut PipelineState,
    target_ets_refs: &[TargetEtsRef<'_>],
) {
    let mutants_run = take_host(&mut state.host_runs, Gate::Mutants);
    let miri_run = take_host(&mut state.host_runs, Gate::Miri);
    let valgrind_run = take_host(&mut state.host_runs, Gate::Valgrind);
    let fuzz_run = take_host(&mut state.host_runs, Gate::Fuzz);
    let lockbud_run = take_host(&mut state.host_runs, Gate::Lockbud);
    let deny_run = take_host(&mut state.host_runs, Gate::Deny);
    let audit_run = take_host(&mut state.host_runs, Gate::Audit);
    let kani_run = take_host(&mut state.host_runs, Gate::Kani);
    let host_tool_rows = [
        host_tool_row("Mutants", "`cargo mutants`", &mutants_run),
        host_tool_row(
            "Miri",
            "`cargo miri test -p control-rs --lib`",
            &miri_run,
        ),
        host_tool_row(
            "Valgrind",
            "`valgrind cargo test -p control-rs --lib`",
            &valgrind_run,
        ),
        host_tool_row("Fuzz", "`cargo fuzz`", &fuzz_run),
        host_tool_row("Lockbud", "`cargo lockbud -k all`", &lockbud_run),
        host_tool_row("Deny", "`cargo deny check`", &deny_run),
        host_tool_row("Audit", "`cargo audit`", &audit_run),
        host_tool_row("Kani", "`cargo kani`", &kani_run),
    ];
    let report_content = report::build_report(&report_params(
        args,
        state,
        &host_tool_rows,
        target_ets_refs,
    ));
    save_ci_markdown(args.out_path, &report_content);
}

fn report_params<'a>(
    args: &'a ReportWrite<'a>,
    state: &'a PipelineState,
    host_tools: &'a [report::HostToolRow<'a>],
    target_ets_results: &'a [TargetEtsRef<'a>],
) -> report::CiReportParams<'a> {
    report::CiReportParams {
        title: args.title,
        options: args.ci_options,
        fmt: state.fmt.verdict,
        fmt_output: &state.fmt.output,
        fmt_time: state.fmt.time,
        clippy: state.clippy.verdict,
        clippy_output: &state.clippy.output,
        clippy_time: state.clippy.time,
        check: state.check.verdict,
        check_output: &state.check.output,
        check_time: state.check.time,
        build: state.build.verdict,
        build_output: &state.build.output,
        build_time: state.build.time,
        clean: state.clean.verdict,
        clean_output: &state.clean.output,
        clean_time: state.clean.time,
        test_cmd: state.test_cmd.verdict,
        test_cmd_output: &state.test_cmd.output,
        test_cmd_time: state.test_cmd.time,
        coverage: state.coverage.verdict,
        tarp_summary: &state.tarp_summary,
        tarp_output: &state.tarp_output,
        test_time: state.coverage.time,
        host_tools,
        warned_gates: &state.counters.warned_gate_names,
        skipped_standard: args.skipped_standard,
        target_ets_results,
        ets_time: state.ets_time,
        trace: state.trace.verdict,
        trace_summary: state.trace_summary.as_ref(),
        trace_time: state.trace.time,
        cross_val: state.cross_val.verdict,
        cross_val_summary: state.cross_val_summary.as_ref(),
        cross_val_time: state.cross_val.time,
    }
}

fn save_ci_markdown(out_path: &Path, report_content: &str) {
    if let Err(e) = fs::create_dir_all(out_path) {
        report::error(format!(
            "Failed to create output directory '{}': {e}",
            out_path.display()
        ));
    }
    let report_path = out_path.join("ci-report.md");
    if let Err(e) =
        report::save_report(&report_path.to_string_lossy(), report_content)
    {
        report::error(format!(
            "Failed to write ci-report.md to '{}': {e}",
            report_path.display()
        ));
    } else {
        report::status("Saved", format!("report to {}", report_path.display()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quality_gate;

    struct MockGate(bool);

    /// A gate whose `execute()` increments a shared counter, so a test can
    /// assert `run_gate` applies pipeline counters exactly once per actual
    /// execution -- the invariant the ETS double-count regression violated.
    struct CountingGate<'a> {
        passed: bool,
        calls: &'a std::cell::Cell<u32>,
    }

    impl quality_gate::QualityGate for MockGate {
        fn execute(
            &self,
            _ctx: &mut PipelineCtx<'_>,
        ) -> quality_gate::GateOutcome {
            quality_gate::GateOutcome {
                passed: self.0,
                output: String::new(),
                details: String::new(),
            }
        }
    }

    impl quality_gate::QualityGate for CountingGate<'_> {
        fn execute(
            &self,
            _ctx: &mut PipelineCtx<'_>,
        ) -> quality_gate::GateOutcome {
            self.calls.set(self.calls.get().saturating_add(1));
            quality_gate::GateOutcome {
                passed: self.passed,
                output: String::new(),
                details: String::new(),
            }
        }
    }

    fn cfg_from(toml: &str, cli: CliArgs) -> CiConfig {
        CiConfig {
            file: toml::from_str(toml).unwrap(),
            cli,
            config_file_path: PathBuf::from("gate.toml"),
            repo_root: PathBuf::from("."),
        }
    }

    #[test]
    fn test_gate_from_str() {
        assert_eq!("clean".parse::<Gate>(), Ok(Gate::Clean));
        assert_eq!("fmt".parse::<Gate>(), Ok(Gate::Fmt));
        assert_eq!("clippy".parse::<Gate>(), Ok(Gate::Clippy));
        assert_eq!("check".parse::<Gate>(), Ok(Gate::Check));
        assert_eq!("build".parse::<Gate>(), Ok(Gate::Build));
        assert_eq!("test".parse::<Gate>(), Ok(Gate::Test));
        assert_eq!("cov".parse::<Gate>(), Ok(Gate::Coverage));
        assert_eq!("mutants".parse::<Gate>(), Ok(Gate::Mutants));
        assert_eq!("miri".parse::<Gate>(), Ok(Gate::Miri));
        assert_eq!("valgrind".parse::<Gate>(), Ok(Gate::Valgrind));
        assert_eq!("fuzz".parse::<Gate>(), Ok(Gate::Fuzz));
        assert_eq!("lockbud".parse::<Gate>(), Ok(Gate::Lockbud));
        assert_eq!("deny".parse::<Gate>(), Ok(Gate::Deny));
        assert_eq!("audit".parse::<Gate>(), Ok(Gate::Audit));
        assert_eq!("kani".parse::<Gate>(), Ok(Gate::Kani));
        assert_eq!("trace".parse::<Gate>(), Ok(Gate::Trace));
        assert_eq!("ets".parse::<Gate>(), Ok(Gate::Ets));
        assert_eq!("validate".parse::<Gate>(), Ok(Gate::Validate));
        assert!("unknown".parse::<Gate>().is_err());
    }

    #[test]
    fn test_unspecified_gates_are_skipped() {
        let cfg = cfg_from("", CliArgs::default());
        for gate in Gate::ALL {
            assert_eq!(cfg.gate_mode(gate), GateMode::Skip);
            assert!(!cfg.is_gate_enabled(gate));
        }
    }

    #[test]
    fn test_gates_table_order_is_pipeline() {
        let cfg = cfg_from(
            r#"
            [gates]
            test = "fail"
            fmt = "warn"
            clippy = true
            "#,
            CliArgs::default(),
        );
        assert_eq!(
            cfg.pipeline(),
            vec![
                (Gate::Test, GateMode::Fail),
                (Gate::Fmt, GateMode::Warn),
                (Gate::Clippy, GateMode::Fail),
            ]
        );
        assert!(!cfg.is_gate_enabled(Gate::Clean));
        assert!(cfg.is_gate_enabled(Gate::Fmt));
        assert_eq!(cfg.gate_mode(Gate::Fmt), GateMode::Warn);
    }

    #[test]
    fn test_unknown_gate_name_is_rejected() {
        let res: Result<CiConfigFile, _> = toml::from_str(
            r#"
            [gates]
            analyze = "fail"
            "#,
        );
        assert!(res.is_err());
    }

    #[test]
    fn test_format_summary_message() {
        let msg = format_summary_message(&SummaryArgs {
            ci_success: true,
            passed_gates: 8,
            failed_gate_names: &[],
            warned_gate_names: &[],
            skipped_gates: 4,
            elapsed: "1.23s",
        });
        assert_eq!(
            msg,
            "8 gates passed, 0 failed, 0 warned, 4 skipped in 1.23s"
        );
        let msg = format_summary_message(&SummaryArgs {
            ci_success: false,
            passed_gates: 7,
            failed_gate_names: &["clippy"],
            warned_gate_names: &[],
            skipped_gates: 4,
            elapsed: "1.23s",
        });
        assert_eq!(
            msg,
            "7 gates passed, 1 failed (clippy), 0 warned, 4 skipped in 1.23s"
        );
        let msg = format_summary_message(&SummaryArgs {
            ci_success: false,
            passed_gates: 6,
            failed_gate_names: &["clean", "clippy"],
            warned_gate_names: &["mutants"],
            skipped_gates: 4,
            elapsed: "1.23s",
        });
        assert_eq!(
            msg,
            "6 gates passed, 2 failed (clean, clippy), 1 warned (mutants), 4 skipped in 1.23s"
        );
    }

    #[test]
    fn test_parse_args_up_to() {
        let cli = CliArgs {
            up_to_gate: Some(Gate::Check),
            ..CliArgs::default()
        };
        let cfg = cfg_from(
            r#"
            [gates]
            clean = "fail"
            fmt = "fail"
            clippy = "fail"
            check = "fail"
            build = "fail"
            test = "fail"
            "#,
            cli,
        );
        assert!(cfg.is_gate_enabled(Gate::Clean));
        assert!(cfg.is_gate_enabled(Gate::Fmt));
        assert!(cfg.is_gate_enabled(Gate::Clippy));
        assert!(cfg.is_gate_enabled(Gate::Check));
        assert!(!cfg.is_gate_enabled(Gate::Build));
        assert!(!cfg.is_gate_enabled(Gate::Test));
        assert!(!cfg.is_gate_enabled(Gate::Coverage));
        assert!(!cfg.is_gate_enabled(Gate::Ets));
    }

    #[test]
    fn test_parse_args_only() {
        let args =
            vec!["ci".to_string(), "--only".to_string(), "ets".to_string()];
        let cfg = parse_args(&args).unwrap().unwrap();
        assert!(!cfg.is_gate_enabled(Gate::Fmt));
        assert!(!cfg.is_gate_enabled(Gate::Clippy));
        assert!(!cfg.is_gate_enabled(Gate::Check));
        assert!(cfg.is_gate_enabled(Gate::Ets));
        assert!(!cfg.is_gate_enabled(Gate::Trace));
    }

    #[test]
    fn test_parse_args_subcommand_ets() {
        let args = vec!["ci".to_string(), "ets".to_string()];
        let cfg = parse_args(&args).unwrap().unwrap();
        assert!(cfg.is_gate_enabled(Gate::Ets));
        assert!(!cfg.is_gate_enabled(Gate::Fmt));
    }

    #[test]
    fn test_parse_args_skip() {
        let cli = CliArgs {
            skipped_gates: vec![Gate::Fmt, Gate::Coverage],
            ..CliArgs::default()
        };
        let cfg = cfg_from(
            r#"
            [gates]
            fmt = "fail"
            clippy = "fail"
            coverage = "fail"
            "#,
            cli,
        );
        assert!(!cfg.is_gate_enabled(Gate::Fmt));
        assert!(cfg.is_gate_enabled(Gate::Clippy));
        assert!(!cfg.is_gate_enabled(Gate::Coverage));
    }

    #[test]
    fn test_parse_args_skip_multi_gate_comma_and_space() {
        let cli = CliArgs {
            skipped_gates: vec![Gate::Fmt, Gate::Clippy, Gate::Clean],
            ..CliArgs::default()
        };
        let cfg = cfg_from(
            r#"
            [gates]
            fmt = "fail"
            clippy = "fail"
            clean = "fail"
            check = "fail"
            "#,
            cli,
        );
        assert!(!cfg.is_gate_enabled(Gate::Fmt));
        assert!(!cfg.is_gate_enabled(Gate::Clippy));
        assert!(!cfg.is_gate_enabled(Gate::Clean));
        assert!(cfg.is_gate_enabled(Gate::Check));
    }

    #[test]
    fn test_parse_args_only_multi_gate_comma_and_space() {
        // Comma-separated
        let args = vec![
            "ci".to_string(),
            "--only".to_string(),
            "check,test".to_string(),
        ];
        let cfg = parse_args(&args).unwrap().unwrap();
        assert!(!cfg.is_gate_enabled(Gate::Fmt));
        assert!(cfg.is_gate_enabled(Gate::Check));
        assert!(cfg.is_gate_enabled(Gate::Test));
        assert!(!cfg.is_gate_enabled(Gate::Coverage));

        // Space-separated
        let args = vec![
            "ci".to_string(),
            "--only".to_string(),
            "ets".to_string(),
            "trace".to_string(),
        ];
        let cfg = parse_args(&args).unwrap().unwrap();
        assert!(!cfg.is_gate_enabled(Gate::Test));
        assert!(cfg.is_gate_enabled(Gate::Ets));
        assert!(cfg.is_gate_enabled(Gate::Trace));
    }

    #[test]
    fn test_parse_args_quiet() {
        let args = vec!["ci".to_string(), "--quiet".to_string()];
        let cfg = parse_args(&args).unwrap().unwrap();
        assert!(cfg.cli.flags.contains(CliFlags::QUIET));

        let args = vec!["ci".to_string()];
        let cfg = parse_args(&args).unwrap().unwrap();
        assert!(!cfg.cli.flags.contains(CliFlags::QUIET));
    }

    #[test]
    fn test_parse_args_manifest_path() {
        let args = vec![
            "ci".to_string(),
            "--manifest-path".to_string(),
            "examples/qemu/Cargo.toml".to_string(),
            "--only".to_string(),
            "ets".to_string(),
        ];
        let cfg = parse_args(&args).unwrap().unwrap();
        assert_eq!(
            cfg.cli.manifest_path.as_deref(),
            Some("examples/qemu/Cargo.toml")
        );
        assert!(cfg.is_gate_enabled(Gate::Ets));
    }

    #[test]
    fn test_parse_args_manifest_path_loads_workspace_config() {
        let toml_str = r#"
            [runner]
            title = "control-rs-qemu"

            [gates]
            test = false
            ets = true

            [[ets.targets]]
            name = "t1"
            target = "thumbv7em"
            bin = "b1"

            [[ets.targets]]
            name = "t2"
            target = "thumbv7em"
            bin = "b2"

            [[ets.targets]]
            name = "t3"
            target = "riscv32"
            bin = "b3"

            [[ets.targets]]
            name = "t4"
            target = "riscv64"
            bin = "b4"
        "#;
        let file: CiConfigFile = toml::from_str(toml_str).unwrap();
        let cfg = CiConfig {
            file,
            cli: CliArgs::default(),
            config_file_path: PathBuf::from("gate.toml"),
            repo_root: PathBuf::from("."),
        };
        assert!(cfg.is_gate_enabled(Gate::Ets));
        assert!(!cfg.is_gate_enabled(Gate::Test));
        assert_eq!(cfg.file.ets.targets.len(), 4);
    }

    #[test]
    fn test_parse_args_buck_converter_config() {
        let toml_str = r#"
            [runner]
            title = "control-rs-buck-converter"

            [gates]
            validate = true
            trace = false

            [[suites]]
            name = "buck-converter"
            bin = "validate"
            oracle = "scipy"
            commands = [["python3", "python3/buck_converter_oracle.py"]]
        "#;
        let file: CiConfigFile = toml::from_str(toml_str).unwrap();
        let cfg = CiConfig {
            file,
            cli: CliArgs::default(),
            config_file_path: PathBuf::from("gate.toml"),
            repo_root: PathBuf::from("."),
        };
        assert!(cfg.is_gate_enabled(Gate::Validate));
        assert!(!cfg.is_gate_enabled(Gate::Trace));
        assert_eq!(cfg.file.examples.len(), 1);
        assert_eq!(cfg.file.examples.first().unwrap().name, "buck-converter");
        assert_eq!(cfg.file.examples.first().unwrap().manifest_path, ".");
        assert_eq!(cfg.file.examples.first().unwrap().bin, "validate");
        assert_eq!(cfg.file.examples.first().unwrap().oracle, "scipy");
        assert_eq!(
            cfg.file.examples.first().unwrap().commands,
            vec![vec![
                "python3".to_string(),
                "python3/buck_converter_oracle.py".to_string()
            ]]
        );
    }

    #[test]
    fn test_config_denies_unknown_fields() {
        let bad_toml = r#"
            [runner]
            title = "test"
            unknown_key = 123
        "#;
        let res: Result<CiConfigFile, _> = toml::from_str(bad_toml);
        assert!(res.is_err());

        let bad_example = r#"
            [[suites]]
            name = "test"
            invalid_prop = "value"
        "#;
        let res: Result<CiConfigFile, _> = toml::from_str(bad_example);
        assert!(res.is_err());

        // Suites require `bin` and `oracle`. A `bin`-only table is incomplete.
        let cargo_bin = r#"
            [[suites]]
            name = "test"
            bin = "validate"
        "#;
        let res: Result<CiConfigFile, _> = toml::from_str(cargo_bin);
        assert!(res.is_err());

        let unknown_container = r#"
            [[suites]]
            name = "test"
            bin = "validate"
            oracle = "scipy"
            container = "results/x.h5"
        "#;
        let res: Result<CiConfigFile, _> = toml::from_str(unknown_container);
        assert!(res.is_err());
    }

    #[test]
    fn test_parse_args_dc_motor_config() {
        let toml_str = r#"
            [runner]
            title = "control-rs-dc-motor"

            [gates]
            validate = true
            trace = false

            [[suites]]
            name = "dc-motor"
            bin = "validate"
            oracle = "scipy"
            commands = [["python3", "python3/dc_motor_oracle.py"]]
        "#;
        let file: CiConfigFile = toml::from_str(toml_str).unwrap();
        let cfg = CiConfig {
            file,
            cli: CliArgs::default(),
            config_file_path: PathBuf::from("gate.toml"),
            repo_root: PathBuf::from("."),
        };
        assert!(cfg.is_gate_enabled(Gate::Validate));
        assert!(!cfg.is_gate_enabled(Gate::Trace));
        assert_eq!(cfg.file.examples.len(), 1);
        assert_eq!(cfg.file.examples.first().unwrap().name, "dc-motor");
        assert_eq!(cfg.file.examples.first().unwrap().manifest_path, ".");
        assert_eq!(cfg.file.examples.first().unwrap().bin, "validate");
        assert_eq!(cfg.file.examples.first().unwrap().oracle, "scipy");
        assert_eq!(
            cfg.file.examples.first().unwrap().commands,
            vec![vec![
                "python3".to_string(),
                "python3/dc_motor_oracle.py".to_string()
            ]]
        );
    }

    #[test]
    fn test_parse_args_numerical_models_config() {
        let toml_str = r#"
            [runner]
            title = "control-rs-numerical-models"

            [gates]
            validate = true
            trace = false

            [[suites]]
            name = "matrix"
            bin = "matrix"
            oracle = "scipy"
            commands = [["python3", "python3/matrix_oracle.py"]]
        "#;
        let file: CiConfigFile = toml::from_str(toml_str).unwrap();
        let cfg = CiConfig {
            file,
            cli: CliArgs::default(),
            config_file_path: PathBuf::from("gate.toml"),
            repo_root: PathBuf::from("."),
        };
        assert!(cfg.is_gate_enabled(Gate::Validate));
        assert!(!cfg.is_gate_enabled(Gate::Trace));
        assert_eq!(cfg.file.examples.len(), 1);
        assert_eq!(cfg.file.examples.first().unwrap().name, "matrix");
        assert_eq!(cfg.file.examples.first().unwrap().manifest_path, ".");
        assert_eq!(cfg.file.examples.first().unwrap().bin, "matrix");
        assert_eq!(cfg.file.examples.first().unwrap().oracle, "scipy");
        assert_eq!(
            cfg.file.examples.first().unwrap().commands,
            vec![vec![
                "python3".to_string(),
                "python3/matrix_oracle.py".to_string()
            ]]
        );
    }

    fn test_ctx<'a>(
        config: &'a CiConfig,
        state: &'a mut PipelineState,
    ) -> PipelineCtx<'a> {
        PipelineCtx {
            config,
            out_path: Path::new("."),
            workspace_dir: Path::new("."),
            timeout_secs: 1,
            targets_to_run: &[],
            example_targets: &[],
            state,
        }
    }

    #[test]
    fn test_run_gate_pass() {
        let cfg = cfg_from("", CliArgs::default());
        let mut state = PipelineState::default();
        let mut ctx = test_ctx(&cfg, &mut state);
        let result = quality_gate::run_gate(
            Gate::Clean,
            GateMode::Fail,
            &MockGate(true),
            &mut ctx,
        );
        assert_eq!(result.verdict, GateVerdict::Pass);
        assert!(ctx.state.counters.ci_success);
        assert_eq!(ctx.state.counters.passed_gates, 1);
        assert!(ctx.state.counters.failed_gate_names.is_empty());
        assert_eq!(ctx.state.counters.skipped_gates, 0);
    }

    #[test]
    fn test_run_gate_fail() {
        let cfg = cfg_from("", CliArgs::default());
        let mut state = PipelineState::default();
        let mut ctx = test_ctx(&cfg, &mut state);
        let result = quality_gate::run_gate(
            Gate::Clippy,
            GateMode::Fail,
            &MockGate(false),
            &mut ctx,
        );
        assert_eq!(result.verdict, GateVerdict::Fail);
        assert!(!ctx.state.counters.ci_success);
        assert_eq!(ctx.state.counters.failed_gate_names, vec!["clippy"]);
    }

    #[test]
    fn test_run_gate_warn() {
        let cfg = cfg_from("", CliArgs::default());
        let mut state = PipelineState::default();
        let mut ctx = test_ctx(&cfg, &mut state);
        let result = quality_gate::run_gate(
            Gate::Mutants,
            GateMode::Warn,
            &MockGate(false),
            &mut ctx,
        );
        assert_eq!(result.verdict, GateVerdict::Warn);
        assert!(ctx.state.counters.ci_success);
        assert_eq!(ctx.state.counters.warned_gate_names, vec!["mutants"]);
    }

    #[test]
    fn test_run_gate_skip() {
        let cfg = cfg_from("", CliArgs::default());
        let mut state = PipelineState::default();
        let mut ctx = test_ctx(&cfg, &mut state);
        let result = quality_gate::run_gate(
            Gate::Fmt,
            GateMode::Skip,
            &MockGate(true),
            &mut ctx,
        );
        assert_eq!(result.verdict, GateVerdict::Skip);
        assert!(result.output.is_empty());
        assert_eq!(ctx.state.counters.skipped_gates, 1);
        assert_eq!(ctx.state.counters.passed_gates, 0);
    }

    #[test]
    fn test_run_gate_counts_once_per_execute_call() {
        let calls = std::cell::Cell::new(0u32);
        let cfg = cfg_from("", CliArgs::default());
        let mut state = PipelineState::default();
        let mut ctx = test_ctx(&cfg, &mut state);
        let job = CountingGate {
            passed: true,
            calls: &calls,
        };
        for _ in 0..3 {
            let _ = quality_gate::run_gate(
                Gate::Ets,
                GateMode::Fail,
                &job,
                &mut ctx,
            );
        }
        assert_eq!(calls.get(), 3);
        assert_eq!(ctx.state.counters.passed_gates, 3);
        assert!(ctx.state.counters.failed_gate_names.is_empty());
    }

    #[test]
    fn test_execute_help_returns_zero_without_running_pipeline() {
        assert_eq!(execute(&["ci".to_string(), "--help".to_string()]), 0);
        assert_eq!(execute(&["ci".to_string(), "-h".to_string()]), 0);
    }

    #[test]
    fn test_execute_unknown_argument_returns_one() {
        assert_eq!(
            execute(&["ci".to_string(), "--not-a-real-flag".to_string()]),
            1
        );
    }

    #[test]
    fn test_gate_name_and_command_label_round_trip_all_variants() {
        for gate in Gate::ALL {
            let name = gate.name();
            assert_eq!(name.parse::<Gate>(), Ok(gate));
            assert!(!gate.command_label().is_empty());
            assert!(gate.command_label().starts_with('`'));
            assert!(gate.command_label().ends_with('`'));
        }
    }

    #[test]
    fn test_skipped_standard_gates_lists_omitted_only() {
        let pipeline =
            vec![(Gate::Clean, GateMode::Fail), (Gate::Test, GateMode::Skip)];
        let skipped = skipped_standard_gates(&pipeline);
        assert!(!skipped.contains(&"clean"));
        assert!(skipped.contains(&"test"));
        // Never listed in `pipeline` at all: also counts as omitted.
        assert!(skipped.contains(&"fmt"));
        assert!(skipped.contains(&"clippy"));
        assert!(skipped.contains(&"check"));
        assert!(skipped.contains(&"build"));
    }

    #[test]
    fn test_collect_ets_targets_filters_by_target_and_bin() {
        let cfg = cfg_from(
            r#"
            [[ets.targets]]
            name = "arm"
            target = "thumbv7em"
            bin = "b1"

            [[ets.targets]]
            name = "riscv"
            target = "riscv32"
            bin = "b2"
            "#,
            CliArgs::default(),
        );
        assert_eq!(collect_ets_targets(&cfg).len(), 2);

        let cfg_filtered = cfg_from(
            r#"
            [[ets.targets]]
            name = "arm"
            target = "thumbv7em"
            bin = "b1"

            [[ets.targets]]
            name = "riscv"
            target = "riscv32"
            bin = "b2"
            "#,
            CliArgs {
                filter_target: Some("riscv32".to_string()),
                ..CliArgs::default()
            },
        );
        assert_eq!(collect_ets_targets(&cfg_filtered).len(), 1);

        let cfg_filtered_bin = cfg_from(
            r#"
            [[ets.targets]]
            name = "arm"
            target = "thumbv7em"
            bin = "b1"

            [[ets.targets]]
            name = "riscv"
            target = "riscv32"
            bin = "b2"
            "#,
            CliArgs {
                filter_bin: Some("b1".to_string()),
                ..CliArgs::default()
            },
        );
        assert_eq!(collect_ets_targets(&cfg_filtered_bin).len(), 1);
    }

    #[test]
    fn test_collect_ets_targets_serial_flag_adds_serial_target() {
        let cfg = cfg_from(
            "",
            CliArgs {
                flags: {
                    let mut f = CliFlags::default();
                    f.set(CliFlags::SERIAL);
                    f
                },
                serial_port: Some("/dev/ttyFAKE".to_string()),
                ..CliArgs::default()
            },
        );
        let targets = collect_ets_targets(&cfg);
        assert_eq!(targets.len(), 1);
        assert!(matches!(
            targets.first(),
            Some(Target::Serial { port, baud: 115_200 }) if port == "/dev/ttyFAKE"
        ));
    }

    #[test]
    fn test_collect_example_targets_resolves_relative_manifest_path() {
        let cfg = cfg_from(
            r#"
            [[suites]]
            name = "matrix"
            bin = "matrix"
            oracle = "scipy"
            commands = [["python3", "python3/matrix_oracle.py"]]
            "#,
            CliArgs::default(),
        );
        let config_dir = Path::new("/repo/some/dir");
        let targets = collect_example_targets(&cfg, config_dir);
        assert_eq!(targets.len(), 1);
        assert_eq!(targets.first().unwrap().manifest_path, "/repo/some/dir/.");
    }

    #[test]
    fn test_ci_skip_flags_reflects_disabled_gates() {
        let cfg = cfg_from(
            r#"
            [gates]
            coverage = "fail"
            "#,
            CliArgs::default(),
        );
        let skip = ci_skip_flags(&cfg);
        assert!(!skip.contains(CiSkip::COV));
        assert!(skip.contains(CiSkip::TRACE));
        assert!(skip.contains(CiSkip::EXAMPLES));
        assert!(skip.contains(CiSkip::ETS));
    }

    fn passing_ets_run() -> EtsRunResult {
        EtsRunResult {
            completion: control_rs_ets_host::runner::Completion::Drained,
            results: vec![TestOutcome {
                suite_name: "suite".to_string(),
                test_name: "t1".to_string(),
                state: control_rs_ets::comms::TestState::Passed,
                cycles: None,
                time_us: None,
                stack_peak: None,
            }],
            console: String::new(),
            resets: 0,
        }
    }

    fn failing_ets_run() -> EtsRunResult {
        EtsRunResult {
            completion: control_rs_ets_host::runner::Completion::Drained,
            results: vec![TestOutcome {
                suite_name: "suite".to_string(),
                test_name: "t1".to_string(),
                state: control_rs_ets::comms::TestState::Failed,
                cycles: None,
                time_us: None,
                stack_peak: None,
            }],
            console: String::new(),
            resets: 0,
        }
    }

    #[test]
    fn test_assemble_ets_results_empty() {
        let state = PipelineState::default();
        let (all_passed, tests, refs) = assemble_ets_results(&state);
        assert!(all_passed);
        assert!(tests.is_empty());
        assert!(refs.is_empty());
    }

    #[test]
    fn test_assemble_ets_results_all_passed() {
        let mut state = PipelineState::default();
        state
            .matrix_results
            .push(("target-a".to_string(), Ok(passing_ets_run())));
        let (all_passed, tests, refs) = assemble_ets_results(&state);
        assert!(all_passed);
        assert_eq!(tests.len(), 1);
        assert_eq!(refs.len(), 1);
    }

    #[test]
    fn test_assemble_ets_results_failed_test_flips_all_passed() {
        let mut state = PipelineState::default();
        state
            .matrix_results
            .push(("target-a".to_string(), Ok(failing_ets_run())));
        let (all_passed, tests, _refs) = assemble_ets_results(&state);
        assert!(!all_passed);
        assert_eq!(tests.len(), 1);
    }

    #[test]
    fn test_assemble_ets_results_host_error_flips_all_passed() {
        let mut state = PipelineState::default();
        state.matrix_results.push((
            "target-a".to_string(),
            Err(HostError::Build {
                target: "thumbv7em".to_string(),
                source: control_rs_ets_host::error::ErrorSource(
                    "linker failed".to_string(),
                ),
            }),
        ));
        let (all_passed, tests, refs) = assemble_ets_results(&state);
        assert!(!all_passed);
        assert!(tests.is_empty());
        assert_eq!(refs.len(), 1);
        assert!(refs.first().unwrap().1.is_err());
    }

    #[test]
    fn test_write_json_reports_writes_only_populated_summaries() {
        let temp_dir =
            std::env::temp_dir().join("control_rs_ci_test_write_json_reports");
        let _ = fs::remove_dir_all(&temp_dir);
        fs::create_dir_all(&temp_dir).unwrap();

        write_json_reports(&temp_dir, &PipelineState::default(), &[]);
        assert!(!temp_dir.join("trace-report.json").exists());
        assert!(!temp_dir.join("validate-report.json").exists());
        assert!(temp_dir.join("ets-results.json").exists());

        let state = PipelineState {
            trace_summary: Some(crate::trace::TraceMatrixSummary::default()),
            ..PipelineState::default()
        };
        write_json_reports(&temp_dir, &state, &[]);
        assert!(temp_dir.join("trace-report.json").exists());
        assert!(temp_dir.join("trace-report.md").exists());

        let combined = vec![TestOutcome {
            suite_name: "suite".to_string(),
            test_name: "t1".to_string(),
            state: control_rs_ets::comms::TestState::Passed,
            cycles: None,
            time_us: None,
            stack_peak: None,
        }];
        write_json_reports(&temp_dir, &PipelineState::default(), &combined);
        assert!(temp_dir.join("ets-results.json").exists());

        let _ = fs::remove_dir_all(&temp_dir);
    }

    fn finalize_in(
        temp_name: &str,
        config: &CiConfig,
        state: PipelineState,
        pipeline: &[GateEntry],
    ) -> (i32, PathBuf) {
        let temp_dir = std::env::temp_dir().join(temp_name);
        let _ = fs::remove_dir_all(&temp_dir);
        fs::create_dir_all(&temp_dir).unwrap();
        let code = finalize_report(
            &Finalize {
                config,
                title: "control-rs-test",
                out_path: &temp_dir,
                pipeline,
                start_time: Instant::now(),
            },
            state,
        );
        (code, temp_dir)
    }

    #[test]
    fn test_finalize_report_success_writes_ci_report() {
        let config = cfg_from("", CliArgs::default());
        let (code, temp_dir) = finalize_in(
            "control_rs_ci_test_finalize_success",
            &config,
            PipelineState::default(),
            &[],
        );
        assert_eq!(code, 0);
        assert!(temp_dir.join("ci-report.md").exists());
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_finalize_report_failed_gate_returns_one() {
        let config = cfg_from("", CliArgs::default());
        let mut state = PipelineState::default();
        state.counters.ci_success = false;
        state.counters.failed_gate_names.push("clippy");
        let (code, temp_dir) = finalize_in(
            "control_rs_ci_test_finalize_failure",
            &config,
            state,
            &[],
        );
        assert_eq!(code, 1);
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_run_ets_gate_skip_mode_counts_as_skipped() {
        let config = cfg_from("", CliArgs::default());
        let mut state = PipelineState::default();
        let mut ctx = PipelineCtx {
            config: &config,
            out_path: Path::new("."),
            workspace_dir: Path::new("."),
            timeout_secs: 1,
            targets_to_run: &[],
            example_targets: &[],
            state: &mut state,
        };
        run_one_gate(Gate::Ets, GateMode::Skip, &mut ctx);
        assert_eq!(ctx.state.counters.skipped_gates, 1);
        assert!(ctx.state.counters.ci_success);
    }

    #[test]
    fn test_run_ets_gate_no_targets_is_implicit_skip_not_failure() {
        let config = cfg_from("", CliArgs::default());
        let mut state = PipelineState::default();
        let mut ctx = PipelineCtx {
            config: &config,
            out_path: Path::new("."),
            workspace_dir: Path::new("."),
            timeout_secs: 1,
            targets_to_run: &[],
            example_targets: &[],
            state: &mut state,
        };
        run_one_gate(Gate::Ets, GateMode::Fail, &mut ctx);
        assert!(ctx.state.counters.ci_success);
        assert!(ctx.state.counters.failed_gate_names.is_empty());
        assert_eq!(ctx.state.counters.skipped_gates, 1);
    }

    #[test]
    fn test_run_validate_gate_skip_mode_counts_as_skipped() {
        let config = cfg_from("", CliArgs::default());
        let mut state = PipelineState::default();
        let mut ctx = PipelineCtx {
            config: &config,
            out_path: Path::new("."),
            workspace_dir: Path::new("."),
            timeout_secs: 1,
            targets_to_run: &[],
            example_targets: &[],
            state: &mut state,
        };
        run_one_gate(Gate::Validate, GateMode::Skip, &mut ctx);
        assert_eq!(ctx.state.cross_val.verdict, GateVerdict::Skip);
        assert_eq!(ctx.state.counters.skipped_gates, 1);
    }

    #[test]
    fn test_run_validate_gate_no_suites_fail_mode_fails_pipeline() {
        let config = cfg_from("", CliArgs::default());
        let mut state = PipelineState::default();
        let mut ctx = PipelineCtx {
            config: &config,
            out_path: Path::new("."),
            workspace_dir: Path::new("."),
            timeout_secs: 1,
            targets_to_run: &[],
            example_targets: &[],
            state: &mut state,
        };
        run_one_gate(Gate::Validate, GateMode::Fail, &mut ctx);
        assert!(!ctx.state.counters.ci_success);
        assert_eq!(ctx.state.cross_val.verdict, GateVerdict::Fail);
        assert_eq!(ctx.state.counters.failed_gate_names, vec!["validate"]);
    }

    #[test]
    fn test_run_validate_gate_no_suites_warn_mode_does_not_fail_pipeline() {
        let config = cfg_from("", CliArgs::default());
        let mut state = PipelineState::default();
        let mut ctx = PipelineCtx {
            config: &config,
            out_path: Path::new("."),
            workspace_dir: Path::new("."),
            timeout_secs: 1,
            targets_to_run: &[],
            example_targets: &[],
            state: &mut state,
        };
        run_one_gate(Gate::Validate, GateMode::Warn, &mut ctx);
        assert!(ctx.state.counters.ci_success);
        assert_eq!(ctx.state.cross_val.verdict, GateVerdict::Warn);
    }
}
