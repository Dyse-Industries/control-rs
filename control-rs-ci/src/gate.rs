//! Generic quality gate model, execution engine, and outcome data types.

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::config::{
    DEFAULT_TIMEOUT_SECS, GateConfig, GateDefinition, GatePolicy,
};
use crate::error::{GateError, GateResult};

/// Prefix for the machine-readable outcome trailer comment in `<gate>.log`.
pub const LOG_TRAILER_PREFIX: &str = "# control-rs-ci-outcome: ";

/// Version of the `GateOutcome` record, incremented on every incompatible
/// change (NFR-2). `report` rejects records of any other version.
pub const OUTCOME_SCHEMA: u32 = 1;

/// Upper bound on the wait for output pumps after the gate process exits.
///
/// A descendant that outlives the gate can hold its pipe open; the pump is then
/// detached rather than blocking the pipeline.
const PUMP_DRAIN_TIMEOUT: Duration = Duration::from_secs(2);

/// Suffix of the per-gate outcome artifact.
pub const RESULT_SUFFIX: &str = ".result.json";

/// Fingerprint capturing inputs that determine gate output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputFingerprint {
    /// Schema version for the fingerprint.
    pub schema: u32,
    /// Hash of the gate command, `args`, `env`, and `cwd`.
    pub config_hash: String,
    /// Content hash of all files under the gate's triggered inputs.
    pub content_hash: String,
    /// Git commit SHA at the time of log generation, if in a git repository.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub head_commit: Option<String>,
    /// Git tree hash of the triggered paths, if clean.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tree_hash: Option<String>,
}

/// Structured outcome envelope written to the tail of `<gate>.log`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LogEnvelope {
    /// Schema version; [`OUTCOME_SCHEMA`].
    pub schema: u32,
    /// Gate slug.
    pub gate: String,
    /// High-level verdict.
    pub verdict: Verdict,
    /// Process exit code.
    pub exit_code: Option<i32>,
    /// Wall-clock execution duration in seconds.
    pub duration_secs: f64,
    /// Outcome summary.
    pub summary: Option<String>,
    /// Recorded input fingerprint.
    pub fingerprint: InputFingerprint,
}

/// A gate shared between the scheduler and its group threads.
pub type SharedGate = Arc<Gate>;

/// Shared, ordered list of gates.
pub type GateList = Vec<SharedGate>;

/// Log file shared by a gate's stdout and stderr pumps.
type LogSink = Arc<Mutex<File>>;

/// Threads copying a gate's output into its log.
type Pumps = Vec<thread::JoinHandle<()>>;

/// A gate process that ran to completion.
struct Finished {
    /// Exit status of the process.
    status: ExitStatus,
    /// Wall-clock duration in seconds.
    duration_secs: f64,
}

/// Outcome verdict classification for a quality gate execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Verdict {
    /// Gate succeeded cleanly without warnings or errors.
    Pass,
    /// Gate completed with non-blocking warnings or degraded capability.
    Warn,
    /// Gate failed or violated quality bounds.
    Fail,
    /// Gate execution was skipped by configuration or user flag.
    Skipped,
}

/// Standardized metadata record written to `<gate>.result.json` after execution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GateOutcome {
    /// Record version; [`OUTCOME_SCHEMA`] when written by this crate.
    pub schema: u32,
    /// Unique identifier / slug of the gate (for example, "fmt", "valgrind").
    pub gate: String,
    /// High-level verdict outcome.
    pub verdict: Verdict,
    /// Process exit status code, if applicable.
    pub exit_code: Option<i32>,
    /// Total wall-clock execution duration in seconds.
    pub duration_secs: f64,
    /// Concise human-readable summary of the gate result.
    pub summary: Option<String>,
    /// Relative filename of the captured log output (for example, "fmt.log").
    pub log_file: String,
    /// Optional input fingerprint for up-to-date detection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fingerprint: Option<InputFingerprint>,
}

/// Execution context passed to quality gate runners.
#[derive(Debug, Clone)]
pub struct GateContext {
    /// Absolute path to the workspace root directory.
    pub workspace_root: PathBuf,
    /// Output directory for logs and JSON outcome artifacts.
    pub out_dir: PathBuf,
    /// Default wall-clock execution timeout.
    pub default_timeout: Duration,
}

/// The single, concrete quality gate type used for all gates.
#[derive(Debug, Clone)]
pub struct Gate {
    /// Gate identifier (for example, `"clean"`, `"fmt"`, `"clippy"`).
    pub name: String,
    /// Base command string (for example, `"cargo clean"`, `"cargo fmt"`, `"vale"`).
    pub command: String,
    /// Additional arguments passed to the command.
    pub args: Vec<String>,
    /// Optional description for status displays and reports.
    pub description: Option<String>,
    /// Environment variable overrides.
    pub env: HashMap<String, String>,
    /// Execution mode / policy for this gate.
    pub mode: GatePolicy,
    /// Wall-clock bound for this gate.
    pub timeout: Duration,
    /// Exit codes reported as `Verdict::Skipped`.
    pub skip_exit_codes: Vec<i32>,
    /// Whether an unfiltered run selects this gate.
    pub default: bool,
    /// Working directory relative to the workspace root.
    pub cwd: Option<PathBuf>,
    /// Path prefixes or files whose change runs this gate (FR-22).
    pub diff_trigger: Vec<String>,
}

impl GateOutcome {
    /// Writes this gate outcome to `<out_dir>/<gate>.result.json`.
    ///
    /// # Errors
    /// Returns `GateError::Io` or `GateError::Json` on failure.
    pub fn save_to_dir(&self, out_dir: &Path) -> GateResult<PathBuf> {
        std::fs::create_dir_all(out_dir)?;
        let result_path = out_dir.join(format!("{}{RESULT_SUFFIX}", self.gate));
        let file = File::create(&result_path)?;
        serde_json::to_writer_pretty(file, self)?;
        Ok(result_path)
    }

    /// Loads a gate outcome record from a JSON file.
    ///
    /// # Errors
    /// Returns `GateError::Io` or `GateError::Json` on failure.
    pub fn load_from_file(path: &Path) -> GateResult<Self> {
        let file = File::open(path)?;
        let outcome = serde_json::from_reader(file)?;
        Ok(outcome)
    }
}

impl GateContext {
    /// Returns the target log path for the given gate name.
    #[must_use]
    pub fn log_path(&self, gate_name: &str) -> PathBuf {
        self.out_dir.join(format!("{gate_name}.log"))
    }
}

impl Gate {
    /// Constructs a new `Gate` with no description and no environment
    /// overrides; see [`Gate::with_description`] and [`Gate::with_env`].
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        command: impl Into<String>,
        args: Vec<String>,
    ) -> Self {
        Self {
            name: name.into(),
            command: command.into(),
            args,
            description: None,
            env: HashMap::new(),
            mode: GatePolicy::Fail,
            timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECS),
            skip_exit_codes: Vec::new(),
            default: true,
            cwd: None,
            diff_trigger: Vec::new(),
        }
    }

    /// Sets the description shown in status lines and reports.
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Sets the environment variable overrides.
    #[must_use]
    pub fn with_env(mut self, env: HashMap<String, String>) -> Self {
        self.env = env;
        self
    }

    /// Constructs a `Gate` from a parsed `GateDefinition`; a definition
    /// without `timeout_secs` gets `default_timeout_secs`.
    #[must_use]
    pub fn from_definition(
        name: impl Into<String>,
        def: &GateDefinition,
        default_timeout_secs: u64,
    ) -> Self {
        Self {
            name: name.into(),
            command: def.command.clone(),
            args: def.args.clone(),
            description: def.description.clone(),
            env: def.env.clone(),
            mode: def.mode(),
            timeout: Duration::from_secs(
                def.timeout_secs.unwrap_or(default_timeout_secs),
            ),
            skip_exit_codes: def.skip_exit_codes.clone(),
            default: def.default,
            cwd: def.cwd.clone(),
            diff_trigger: def.diff_trigger.clone(),
        }
    }

    /// Returns the name of this gate.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the execution policy / mode for this gate.
    #[must_use]
    pub const fn mode(&self) -> GatePolicy {
        self.mode
    }

    /// Returns the optional description of this gate.
    #[must_use]
    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    /// Records and saves the outcome of a gate disabled by `mode = "skip"`
    /// that was selected explicitly. The gate does not execute.
    ///
    /// # Errors
    /// Returns `GateError` if the outcome artifact cannot be written.
    pub fn record_disabled(
        &self,
        ctx: &GateContext,
    ) -> GateResult<GateOutcome> {
        self.record_skipped(
            ctx,
            &format!("{} disabled by mode = \"skip\"; not executed", self.name),
        )
    }

    /// Records and saves a `Skipped` outcome carrying `summary`. The gate
    /// does not execute.
    ///
    /// # Errors
    /// Returns `GateError` if the outcome artifact cannot be written.
    pub fn record_skipped(
        &self,
        ctx: &GateContext,
        summary: &str,
    ) -> GateResult<GateOutcome> {
        let outcome = GateOutcome {
            schema: OUTCOME_SCHEMA,
            gate: self.name.clone(),
            verdict: Verdict::Skipped,
            exit_code: None,
            duration_secs: 0.0,
            summary: Some(summary.to_string()),
            log_file: format!("{}.log", self.name),
            fingerprint: None,
        };
        let _ = outcome.save_to_dir(&ctx.out_dir)?;
        Ok(outcome)
    }

    /// Formatted command line string for display.
    #[must_use]
    pub fn command_display(&self) -> String {
        if self.args.is_empty() {
            format!("`{}`", self.command)
        } else {
            format!("`{} {}`", self.command, self.args.join(" "))
        }
    }

    /// Executes the gate command, captures logs, bounds execution by timeout,
    /// evaluates verdict and writes `<gate>.result.json`.
    ///
    /// # Errors
    /// Returns `GateError` only when artifact file creation or I/O fails.
    pub fn execute(&self, ctx: &GateContext) -> GateResult<GateOutcome> {
        self.execute_with_echo(ctx, None)
    }

    /// Executes the gate as [`Gate::execute`] does. When `echo` is
    /// `Some(prefix)`, each line of the gate's stdout and stderr is also
    /// written to stderr behind `prefix` as it arrives.
    ///
    /// The log file receives the same lines in both modes. With echo enabled,
    /// stdout and stderr are read through separate pipes, so their relative
    /// order in the log is preserved per stream but not across streams.
    ///
    /// # Errors
    /// Returns `GateError` only when artifact file creation or I/O fails.
    pub fn execute_with_echo(
        &self,
        ctx: &GateContext,
        echo: Option<&str>,
    ) -> GateResult<GateOutcome> {
        let log_path = ctx.log_path(&self.name);
        if let Some(parent) = log_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let fingerprint = self.current_fingerprint(&ctx.workspace_root);
        let mut cmd = self.command_for(&ctx.workspace_root);
        let mut outcome = match self.spawn_and_log(&mut cmd, &log_path, echo) {
            Ok(finished) => self.outcome_from_exit(&finished),
            Err(GateError::Timeout { timeout_secs, .. }) => {
                let _ = append_log(
                    &log_path,
                    &format!(
                        "\ncontrol-rs-ci: gate '{}' timed out after {:.1}s\n",
                        self.name, timeout_secs
                    ),
                );
                self.aborted_outcome(
                    timeout_secs,
                    format!(
                        "{} timed out after {:.1}s",
                        self.name, timeout_secs
                    ),
                )
            }
            Err(e) => {
                let err_msg = e.to_string();
                let _ = append_log(
                    &log_path,
                    &format!(
                        "control-rs-ci: gate '{}' failed to execute: {}\n",
                        self.name, err_msg
                    ),
                );
                self.aborted_outcome(
                    0.0,
                    format!("{} execution error: {}", self.name, err_msg),
                )
            }
        };

        outcome.fingerprint = Some(fingerprint);
        let _ = outcome.save_to_dir(&ctx.out_dir)?;
        Ok(outcome)
    }

    /// Computes the configuration hash of this gate's definition.
    #[must_use]
    pub fn config_hash(&self) -> String {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.command.hash(&mut hasher);
        self.args.hash(&mut hasher);
        let sorted_env: BTreeMap<_, _> = self.env.iter().collect();
        for (k, v) in &sorted_env {
            k.hash(&mut hasher);
            v.hash(&mut hasher);
        }
        self.cwd.hash(&mut hasher);
        format!("{:016x}", hasher.finish())
    }

    /// Computes the current input fingerprint for this gate against `root`.
    #[must_use]
    pub fn current_fingerprint(&self, root: &Path) -> InputFingerprint {
        let config_hash = self.config_hash();
        let audited =
            crate::diff_trigger::audit_inputs(root, &self.diff_trigger);

        InputFingerprint {
            schema: OUTCOME_SCHEMA,
            config_hash,
            content_hash: audited.content_hash,
            head_commit: audited.head,
            tree_hash: audited.tree_hash,
        }
    }

    /// Appends the structured outcome trailer envelope to the gate's log file.
    ///
    /// # Errors
    ///
    /// Returns [`GateError::Io`] if appending to the log file fails, or
    /// [`GateError::Json`] if serializing the outcome envelope fails.
    pub fn append_outcome_trailer(
        &self,
        log_path: &Path,
        outcome: &GateOutcome,
        fingerprint: &InputFingerprint,
    ) -> GateResult<()> {
        let envelope = LogEnvelope {
            schema: outcome.schema,
            gate: outcome.gate.clone(),
            verdict: outcome.verdict,
            exit_code: outcome.exit_code,
            duration_secs: outcome.duration_secs,
            summary: outcome.summary.clone(),
            fingerprint: fingerprint.clone(),
        };
        let serialized = serde_json::to_string(&envelope)?;
        let trailer_line = format!("\n{LOG_TRAILER_PREFIX}{serialized}\n");
        append_log(log_path, &trailer_line)
    }

    /// Attempts to read and validate the outcome envelope from an existing log.
    ///
    /// Scans the trailing bytes of `log_path` for [`LOG_TRAILER_PREFIX`].
    ///
    /// # Errors
    ///
    /// Returns [`GateError::Io`] if opening, seeking, or reading the log file fails,
    /// or if no valid outcome trailer is found. Returns [`GateError::Json`] if
    /// the trailer JSON is malformed.
    pub fn read_outcome_from_log(
        &self,
        log_path: &Path,
    ) -> GateResult<LogEnvelope> {
        if !log_path.is_file() {
            return Err(GateError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "log file missing",
            )));
        }
        let mut file = File::open(log_path)?;
        let metadata = file.metadata()?;
        let len = metadata.len();
        if len == 0 {
            return Err(GateError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "log file is empty",
            )));
        }

        let to_read = 8192.min(len);
        let offset = i64::try_from(to_read).map_err(|_| {
            GateError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "log offset overflow",
            ))
        })?;
        let negative_offset = offset.checked_neg().ok_or_else(|| {
            GateError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "log offset negation overflow",
            ))
        })?;
        file.seek(SeekFrom::End(negative_offset))?;
        let mut buffer = String::new();
        file.read_to_string(&mut buffer).map_err(GateError::Io)?;

        for line in buffer.lines().rev() {
            let line = line.trim();
            if let Some(json_str) = line.strip_prefix(LOG_TRAILER_PREFIX) {
                let envelope: LogEnvelope = serde_json::from_str(json_str)?;
                if envelope.schema == OUTCOME_SCHEMA
                    && envelope.gate == self.name
                {
                    return Ok(envelope);
                }
            }
        }

        Err(GateError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "no valid outcome trailer found in log",
        )))
    }

    /// Evaluates whether this gate is up-to-date and returns its cached outcome if valid.
    #[must_use]
    pub fn try_read_up_to_date_outcome(
        &self,
        ctx: &GateContext,
    ) -> Option<GateOutcome> {
        let log_path = ctx.log_path(&self.name);
        if !log_path.is_file() {
            return None;
        }

        // 1. Try reading from <gate>.result.json first
        let result_path =
            ctx.out_dir.join(format!("{}{RESULT_SUFFIX}", self.name));
        let outcome =
            if let Ok(loaded) = GateOutcome::load_from_file(&result_path) {
                loaded
            } else if let Ok(envelope) = self.read_outcome_from_log(&log_path) {
                GateOutcome {
                    schema: envelope.schema,
                    gate: self.name.clone(),
                    verdict: envelope.verdict,
                    exit_code: envelope.exit_code,
                    duration_secs: envelope.duration_secs,
                    summary: envelope.summary,
                    log_file: format!("{}.log", self.name),
                    fingerprint: Some(envelope.fingerprint),
                }
            } else {
                return None;
            };

        // Only passing verdicts can be reused; failures must reproduce.
        if outcome.verdict != Verdict::Pass {
            return None;
        }

        let recorded_fp = outcome.fingerprint.as_ref()?;
        let current_fp = self.current_fingerprint(&ctx.workspace_root);
        if current_fp.config_hash != recorded_fp.config_hash {
            return None;
        }

        // Verify inputs are not newer than the log
        let is_fresh = crate::diff_trigger::is_log_fresh(
            &ctx.workspace_root,
            &self.diff_trigger,
            &log_path,
            recorded_fp,
        );

        if !is_fresh {
            return None;
        }

        let _ = outcome.save_to_dir(&ctx.out_dir);
        Some(outcome)
    }

    /// Builds the process for this gate, splitting a compound command string
    /// (for example, `"cargo clean"`) into program and leading arguments.
    fn command_for(&self, workspace_root: &Path) -> Command {
        let mut words = self.command.split_whitespace();
        let program = words.next().unwrap_or(&self.command);
        let mut cmd = Command::new(program);
        cmd.current_dir(self.cwd.as_ref().map_or_else(
            || workspace_root.to_path_buf(),
            |d| workspace_root.join(d),
        ));
        cmd.args(words);
        cmd.args(&self.args);
        cmd.envs(&self.env);
        cmd.env("CARGO_TERM_COLOR", crate::ui::cargo_color_env());
        cmd
    }

    /// Verdict for a failure that is not a skip: `Warn` under a warn
    /// policy, otherwise `Fail`.
    fn failure_verdict(&self) -> Verdict {
        if self.mode == GatePolicy::Warn {
            Verdict::Warn
        } else {
            Verdict::Fail
        }
    }

    /// Outcome of a process that exited on its own.
    fn outcome_from_exit(&self, finished: &Finished) -> GateOutcome {
        let status = finished.status;
        let exit_code = status.code();
        let verdict =
            if exit_code.is_some_and(|c| self.skip_exit_codes.contains(&c)) {
                Verdict::Skipped
            } else if status.success() {
                Verdict::Pass
            } else {
                self.failure_verdict()
            };

        let summary = if verdict == Verdict::Skipped {
            format!(
                "{} skipped with exit code {}",
                self.name,
                exit_code.unwrap_or(0)
            )
        } else if status.success() {
            format!("{} succeeded cleanly", self.name)
        } else {
            format!(
                "{} failed with exit code {}",
                self.name,
                exit_code.unwrap_or(-1)
            )
        };

        GateOutcome {
            schema: OUTCOME_SCHEMA,
            gate: self.name.clone(),
            verdict,
            exit_code,
            duration_secs: finished.duration_secs,
            summary: Some(summary),
            log_file: format!("{}.log", self.name),
            fingerprint: None,
        }
    }

    /// Outcome of a gate that timed out or could not be run.
    fn aborted_outcome(
        &self,
        duration_secs: f64,
        summary: String,
    ) -> GateOutcome {
        GateOutcome {
            schema: OUTCOME_SCHEMA,
            gate: self.name.clone(),
            verdict: self.failure_verdict(),
            exit_code: None,
            duration_secs,
            summary: Some(summary),
            log_file: format!("{}.log", self.name),
            fingerprint: None,
        }
    }

    fn spawn_and_log(
        &self,
        cmd: &mut Command,
        log_path: &Path,
        echo: Option<&str>,
    ) -> GateResult<Finished> {
        let log_file = File::create(log_path)?;
        let sink = if echo.is_some() {
            cmd.stdout(Stdio::piped());
            cmd.stderr(Stdio::piped());
            Some(Arc::new(Mutex::new(log_file)))
        } else {
            let err_file = log_file.try_clone()?;
            cmd.stdout(Stdio::from(log_file));
            cmd.stderr(Stdio::from(err_file));
            None
        };

        let start = Instant::now();
        let mut child = cmd.spawn().map_err(|e| GateError::Spawn {
            gate: self.name.clone(),
            message: e.to_string(),
        })?;

        let mut pumps = Pumps::new();
        if let (Some(prefix), Some(sink)) = (echo, sink) {
            if let Some(out) = child.stdout.take() {
                pumps.push(spawn_pump(
                    out,
                    Arc::clone(&sink),
                    prefix.to_owned(),
                ));
            }
            if let Some(err) = child.stderr.take() {
                pumps.push(spawn_pump(err, sink, prefix.to_owned()));
            }
        }

        let pid = child.id();
        let poll_interval = Duration::from_millis(50);
        loop {
            match child.try_wait() {
                Ok(Some(status)) => {
                    let duration_secs = start.elapsed().as_secs_f64();
                    drain_pumps(pumps);
                    return Ok(Finished {
                        status,
                        duration_secs,
                    });
                }
                Ok(None) => {
                    if start.elapsed() > self.timeout {
                        terminate_tree(&mut child, pid);
                        return Err(GateError::Timeout {
                            gate: self.name.clone(),
                            timeout_secs: self.timeout.as_secs_f64(),
                        });
                    }
                    thread::sleep(poll_interval);
                }
                Err(e) => {
                    terminate_tree(&mut child, pid);
                    return Err(GateError::Spawn {
                        gate: self.name.clone(),
                        message: e.to_string(),
                    });
                }
            }
        }
    }
}

fn append_log(path: &Path, text: &str) -> Result<(), GateError> {
    use std::fs::OpenOptions;
    use std::io::Write;
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    file.write_all(text.as_bytes())?;
    Ok(())
}

/// Copies `reader` line by line into the shared log and echoes each line to
/// stderr behind `prefix`.
fn spawn_pump<R: Read + Send + 'static>(
    reader: R,
    sink: LogSink,
    prefix: String,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let mut reader = BufReader::new(reader);
        let mut line = Vec::new();
        loop {
            line.clear();
            match reader.read_until(b'\n', &mut line) {
                Ok(0) | Err(_) => break,
                Ok(_) => {
                    if let Ok(mut log) = sink.lock() {
                        let _ = log.write_all(&line);
                    }
                    let text = String::from_utf8_lossy(&line);
                    crate::ui::gate_output(
                        &prefix,
                        text.trim_end_matches(['\n', '\r']),
                    );
                }
            }
        }
    })
}

/// Joins pumps that finish within `PUMP_DRAIN_TIMEOUT` and detaches the rest.
fn drain_pumps(pumps: Pumps) {
    let now = Instant::now();
    let deadline = now.checked_add(PUMP_DRAIN_TIMEOUT).unwrap_or(now);
    for pump in pumps {
        while !pump.is_finished() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(10));
        }
        if pump.is_finished() {
            let _ = pump.join();
        }
    }
}

/// Kills the gate process and every descendant (FR-16).
///
/// Walks the tree top-down: each process is stopped before its children are
/// listed, so no process in the tree can fork after it has been visited. Every
/// stopped process is then killed. Gates stay in the runner's process group so
/// a terminal interrupt still reaches them; this walk is what bounds their
/// descendants on timeout. Without `pgrep`, only the gate process is killed.
#[cfg(unix)]
fn terminate_tree(child: &mut std::process::Child, pid: u32) {
    let tree = stop_tree(pid);
    for id in &tree {
        signal(*id, "-KILL");
    }
    let _ = child.kill();
    let _ = child.wait();
}

/// Stops `root` and all of its descendants, returning their process ids in
/// visit order.
#[cfg(unix)]
fn stop_tree(root: u32) -> Vec<u32> {
    let mut tree = vec![root];
    let mut next = 0;
    while let Some(&id) = tree.get(next) {
        signal(id, "-STOP");
        for child in children_of(id) {
            if !tree.contains(&child) {
                tree.push(child);
            }
        }
        next = next.saturating_add(1);
    }
    tree
}

/// Direct children of `pid`, as listed by `pgrep -P`.
#[cfg(unix)]
fn children_of(pid: u32) -> Vec<u32> {
    Command::new("pgrep")
        .args(["-P", &pid.to_string()])
        .stderr(Stdio::null())
        .output()
        .map(|out| {
            String::from_utf8_lossy(&out.stdout)
                .lines()
                .filter_map(|l| l.trim().parse().ok())
                .collect()
        })
        .unwrap_or_default()
}

/// Sends `sig` (for example `-KILL`) to one process through `kill`.
#[cfg(unix)]
fn signal(pid: u32, sig: &str) {
    let _ = Command::new("kill")
        .args([sig, &pid.to_string()])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
}

#[cfg(not(unix))]
fn terminate_child_only(child: &mut std::process::Child, _pid: u32) {
    let _ = child.kill();
    let _ = child.wait();
}

#[cfg(not(unix))]
use terminate_child_only as terminate_tree;

/// Instantiates every configured gate in [`GateConfig::pipeline_order`].
#[must_use]
pub fn build_all_gates(config: &GateConfig) -> GateList {
    let default_timeout = config.runner.timeout_secs;
    config
        .pipeline_order()
        .into_iter()
        .filter_map(|name| {
            config.gate_def(name).map(|def| {
                Arc::new(Gate::from_definition(name, def, default_timeout))
            })
        })
        .collect()
}
