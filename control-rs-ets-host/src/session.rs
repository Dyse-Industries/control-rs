//! The single command and telemetry session with a target.
//!
//! # Description
//!
//! A [`Session`] owns a [`HostLink`], is the only writer on it and runs one executable at a
//! time. [`Session::open`] checks the wire revision and discovers the target into a
//! [`Catalog`]. Cases run through [`Session::run_case`] and [`Session::run_cases`]; lifecycle
//! tasks run through [`Session::run_task`] (`task_run`). Settings are read and written by name or
//! index over the session, from target-reported information alone.
//!
//! # Core Concepts
//!
//! - **Blocking**: Every call polls the link until its reply arrives or a deadline passes. A
//!   frontend that must stay responsive runs the session on a worker thread.
//! - **Device state**: The session tracks idle, running an item and resetting. Silence or a
//!   closed link during an active run, with no report, is an unreported fault.
//! - **Recovery**: After a target panic or an unresponsive run the session sends `TryReset`,
//!   resets the link, rechecks the revision and rediscovers. Collected results survive.
//!
//! # Usage
//!
//! ```no_run
//! use control_rs_ets_host::session::Session;
//! use control_rs_ets_host::target::Target;
//! use control_rs_ets_host::ETSBridge;
//!
//! let bridge = ETSBridge::new(Target::qemu_arm(), false)?;
//! let mut session = Session::open(bridge)?;
//! let ids = session.catalog().case_ids();
//! for id in ids {
//!     let outcome = session.run_case(id)?;
//!     println!("{id:?}: {:?}", outcome.status);
//! }
//! # Ok::<(), control_rs_ets_host::HostError>(())
//! ```
//!
//! # Limitations
//!
//! - **Single target**: One session drives one target; nothing establishes that two sessions in
//!   one process are independent.

use std::thread;
use std::time::{Duration, Instant};

use control_rs_ets::wire::{
    Command, ExecMode, PROTOCOL_VERSION, Param, ParamKey, Verdict,
};

use crate::bridge::{OwnedTelemetry, TestState};
use crate::catalog::{Catalog, ExecEntry, ItemId, SettingEntry, TargetDetails};
use crate::error::HostError;
use crate::link::{HostLink, LinkEvent, Polled, discover};

/// Pause between polls of an idle link.
const POLL_PAUSE: Duration = Duration::from_millis(1);

/// Result of a session operation.
pub type SessionError = HostError;

/// Result of a session call that returns a value.
pub type SessionResult<T> = Result<T, SessionError>;

/// What a session tracks about the target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceState {
    /// Nothing is running.
    Idle,
    /// The target is being reset and rediscovered.
    Resetting,
    /// The item is running.
    Running(ItemId),
}

/// How a case ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CaseStatus {
    /// The target reported a verdict.
    Done(Verdict),
    /// The target reported a panic or exception.
    Faulted {
        /// Source file of the panic.
        file: String,
        /// Source line of the panic.
        line: u32,
        /// Panic message.
        message: String,
    },
    /// The target went silent without a report.
    UnreportedFault,
}

/// Statistics a target reports for a finished run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Stats {
    /// CPU cycles.
    pub cycles: u64,
    /// Largest release jitter in nanoseconds.
    pub jitter_max_ns: u32,
    /// Releases that arrived before the previous step finished.
    pub missed_releases: u32,
    /// Peak stack bytes.
    pub stack_peak: u32,
    /// Longest step in nanoseconds.
    pub step_max_ns: u32,
    /// Shortest step in nanoseconds.
    pub step_min_ns: u32,
    /// Steps executed.
    pub steps: u64,
    /// Run duration in nanoseconds.
    pub time_ns: u64,
}

/// The result of one case.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CaseOutcome {
    /// The case.
    pub id: ItemId,
    /// Message that came with the verdict, if any.
    pub message: Option<String>,
    /// Statistics, when the target reported them.
    pub stats: Option<Stats>,
    /// How the case ended.
    pub status: CaseStatus,
}

/// Outcomes collected by a session; they survive recovery.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResultStore {
    /// Case outcomes in run order.
    pub cases: Vec<CaseOutcome>,
    /// Target resets performed.
    pub resets: u32,
}

/// Receives each case outcome as it completes.
pub trait CaseObserver {
    /// Called after each case.
    fn on_outcome(&mut self, _outcome: &CaseOutcome) {}
}

/// Deadlines of a session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionOptions {
    /// How long one case may run before the target counts as silent.
    pub case_timeout: Duration,
    /// How long discovery may take.
    pub discovery_timeout: Duration,
    /// How long a setting request waits for its echo.
    pub reply_timeout: Duration,
    /// How long a stopping run may take before the target is reset.
    pub stop_timeout: Duration,
}

/// The session with one target.
pub struct Session<L: HostLink<Error = HostError>> {
    pub(crate) catalog: Catalog,
    pub(crate) console: String,
    pub(crate) device: DeviceState,
    pub(crate) link: L,
    pub(crate) log: Vec<String>,
    pub(crate) opts: SessionOptions,
    pub(crate) details: TargetDetails,
    pub(crate) results: ResultStore,
}

/// What a wait for the next event produced.
pub(crate) enum Wait {
    /// The link is gone.
    Closed,
    /// A telemetry message arrived.
    Event(Box<OwnedTelemetry>),
    /// The deadline passed.
    Timeout,
}

impl Default for SessionOptions {
    fn default() -> Self {
        Self {
            case_timeout: Duration::from_secs(30),
            discovery_timeout: Duration::from_secs(10),
            reply_timeout: Duration::from_secs(2),
            stop_timeout: Duration::from_secs(2),
        }
    }
}

impl CaseObserver for () {}

impl<L: HostLink<Error = HostError>> Session<L> {
    /// Opens a session: checks the wire revision and discovers the target.
    ///
    /// # Errors
    /// Returns [`HostError::ProtocolMismatch`] when the target speaks another revision,
    /// [`HostError::Discovery`] when discovery does not finish, or a transport error.
    pub fn open(link: L) -> SessionResult<Self> {
        Self::open_with(link, SessionOptions::default())
    }

    /// Opens a session with explicit deadlines.
    ///
    /// # Errors
    /// See [`Session::open`].
    pub fn open_with(link: L, opts: SessionOptions) -> SessionResult<Self> {
        let mut session = Self::new(link, opts);
        session.discover()?;
        Ok(session)
    }

    /// Creates a session without discovering the target.
    ///
    /// A frontend calls [`Session::discover`] until it succeeds, for a target that is still
    /// booting.
    #[must_use]
    pub const fn new(link: L, opts: SessionOptions) -> Self {
        Self {
            catalog: Catalog { suites: Vec::new() },
            console: String::new(),
            device: DeviceState::Idle,
            link,
            log: Vec::new(),
            opts,
            details: TargetDetails {
                board_id: 0,
                core_clock_hz: 0,
                fpu_flags: 0,
                protocol_version: 0,
                release_source: false,
            },
            results: ResultStore {
                cases: Vec::new(),
                resets: 0,
            },
        }
    }

    /// Whether the link reports that the target process or device exited.
    ///
    /// # Errors
    /// Returns the link error if the status cannot be queried.
    pub fn exited(&mut self) -> SessionResult<bool> {
        self.link.exited()
    }

    /// Describes the link.
    #[must_use]
    pub fn link_description(&self) -> &str {
        self.link.describe()
    }

    /// What `TargetInfo` reported at the last discovery.
    #[must_use]
    pub const fn details(&self) -> TargetDetails {
        self.details
    }

    /// Takes the log lines (target logs and console text) received since the last call.
    pub fn take_log(&mut self) -> Vec<String> {
        std::mem::take(&mut self.log)
    }

    /// What discovery reported.
    #[must_use]
    pub const fn catalog(&self) -> &Catalog {
        &self.catalog
    }

    /// Console text the target printed outside the protocol.
    #[must_use]
    pub fn console(&self) -> &str {
        &self.console
    }

    /// What the session believes the target is doing.
    #[must_use]
    pub const fn device_state(&self) -> DeviceState {
        self.device
    }

    /// Closes the link.
    pub fn close(&mut self) {
        self.link.close();
    }

    /// Outcomes collected so far.
    #[must_use]
    pub const fn results(&self) -> &ResultStore {
        &self.results
    }

    /// Reads one setting.
    ///
    /// # Errors
    /// Returns a transport error, or [`HostError::Refused`] when the target does not answer.
    pub fn get_setting(
        &mut self,
        suite: u16,
        key: ParamKey<'_>,
    ) -> SessionResult<SettingEntry> {
        self.param_request(suite, key, None)
    }

    /// Writes one setting by name and returns the value the target stored.
    ///
    /// The target answers with the stored value whether or not the write succeeded, so the
    /// caller compares the returned value with the requested one.
    ///
    /// # Errors
    /// Returns a transport error, or [`HostError::Refused`] when the target does not answer.
    pub fn set_setting(
        &mut self,
        suite: u16,
        name: &str,
        value: Param,
    ) -> SessionResult<SettingEntry> {
        self.param_request(suite, ParamKey::Name(name), Some(value))
    }

    /// Runs one case to its verdict.
    ///
    /// A target panic or silence is recorded in the outcome and the target is recovered; only a
    /// transport failure is an error.
    ///
    /// # Errors
    /// Returns [`HostError::Refused`] for an unknown id or a task, or a transport error.
    pub fn run_case(&mut self, id: ItemId) -> SessionResult<CaseOutcome> {
        let allowed = self
            .catalog
            .exec(id)
            .is_some_and(|e| !e.is_task() && e.allows(ExecMode::Atomic));
        if !allowed {
            return Err(refused("not a case of this target"));
        }
        self.link.send(&Command::RunExecutable {
            suite: id.0,
            item: id.1,
            mode: ExecMode::Atomic,
            max_steps: 0,
            period_ns: 0,
        })?;
        self.device = DeviceState::Running(id);
        let outcome = self.collect_case(id)?;
        self.results.cases.push(outcome.clone());
        Ok(outcome)
    }

    /// Runs `ids` in order and never starts a task.
    ///
    /// A target panic is recorded against its case and the sequence continues with the next id.
    ///
    /// # Errors
    /// Returns a transport error.
    pub fn run_cases<O: CaseObserver>(
        &mut self,
        ids: impl IntoIterator<Item = ItemId>,
        observer: &mut O,
    ) -> SessionResult<()> {
        for id in ids {
            if self.catalog.exec(id).is_some_and(ExecEntry::is_task) {
                continue;
            }
            let outcome = self.run_case(id)?;
            observer.on_outcome(&outcome);
        }
        Ok(())
    }

    /// Resets the target and rediscovers it.
    ///
    /// # Errors
    /// Returns a transport error or a discovery error.
    pub fn recover(&mut self) -> SessionResult<()> {
        self.device = DeviceState::Resetting;
        let _ = self.link.send(&Command::TryReset);
        self.link.reset()?;
        self.results.resets = self.results.resets.saturating_add(1);
        self.discover()
    }

    /// Checks the wire revision and discovers the target, filling the catalog.
    ///
    /// # Errors
    /// Returns [`HostError::ProtocolMismatch`] when the target speaks another revision,
    /// [`HostError::Discovery`] when discovery does not finish in time, or a transport error.
    pub fn discover(&mut self) -> SessionResult<()> {
        self.catalog = Catalog::default();
        discover(&mut self.link)?;
        let until = deadline(self.opts.discovery_timeout);
        let mut revision = None;
        loop {
            match self.wait_event(until) {
                Wait::Closed => return Err(closed()),
                Wait::Timeout => return Err(HostError::Discovery),
                Wait::Event(event) => match *event {
                    OwnedTelemetry::TargetInfo {
                        protocol_version,
                        board_id,
                        core_clock_hz,
                        fpu_flags,
                        release_source,
                    } => {
                        if protocol_version != PROTOCOL_VERSION {
                            return Err(HostError::ProtocolMismatch {
                                host: u32::from(PROTOCOL_VERSION),
                                target: u32::from(protocol_version),
                            });
                        }
                        self.details = TargetDetails {
                            board_id,
                            core_clock_hz,
                            fpu_flags,
                            protocol_version,
                            release_source,
                        };
                        revision = Some(protocol_version);
                    }
                    OwnedTelemetry::DiscoveryComplete => break,
                    other => self.catalog.apply(&other),
                },
            }
        }
        if revision.is_none() {
            return Err(HostError::Discovery);
        }
        for (suite, index) in self.catalog.missing_settings() {
            let _ = self.get_setting(suite, ParamKey::Index(index))?;
        }
        // Settings refill by Param index; executable holes have no refetch.
        // Skipping them would let case_ids omit registered cases (false green).
        if !self.catalog.missing_executables().is_empty() {
            return Err(HostError::Discovery);
        }
        self.device = DeviceState::Idle;
        Ok(())
    }

    /// Waits for the outcome of the case `id`, recovering the target after a fault.
    fn collect_case(&mut self, id: ItemId) -> SessionResult<CaseOutcome> {
        let until = deadline(self.opts.case_timeout);
        let mut outcome = CaseOutcome {
            id,
            message: None,
            stats: None,
            status: CaseStatus::UnreportedFault,
        };
        let mut verdict = None;
        loop {
            match self.wait_event(until) {
                Wait::Closed => return Err(closed()),
                Wait::Timeout => {
                    // Wire settle is Done(verdict) then RunStats. A timeout after
                    // Done must keep the verdict and leave Idle; only true silence
                    // is UnreportedFault + recover.
                    if let Some(v) = verdict {
                        outcome.status = CaseStatus::Done(v);
                        self.device = DeviceState::Idle;
                        return Ok(outcome);
                    }
                    outcome.status = CaseStatus::UnreportedFault;
                    let _ = self.recover();
                    return Ok(outcome);
                }
                Wait::Event(event) => match *event {
                    OwnedTelemetry::TestStateChange {
                        suite_id,
                        test_id,
                        verdict: Some(v),
                        message,
                        ..
                    } if (suite_id, test_id) == id => {
                        verdict = Some(v);
                        outcome.message = message;
                    }
                    OwnedTelemetry::RunStats {
                        suite_id, test_id, ..
                    } if (suite_id, test_id) == id => {
                        outcome.stats = stats_of(&event);
                        if let Some(v) = verdict {
                            outcome.status = CaseStatus::Done(v);
                            self.device = DeviceState::Idle;
                            return Ok(outcome);
                        }
                    }
                    OwnedTelemetry::TargetPanic {
                        file,
                        line,
                        message,
                        ..
                    } => {
                        outcome.status = CaseStatus::Faulted {
                            file,
                            line,
                            message,
                        };
                        // Keep Faulted even when rediscovery fails after reset.
                        let _ = self.recover();
                        return Ok(outcome);
                    }
                    _ => {}
                },
            }
        }
    }

    /// Sends a `Param` request and waits for the echoed value.
    fn param_request(
        &mut self,
        suite: u16,
        key: ParamKey<'_>,
        value: Option<Param>,
    ) -> SessionResult<SettingEntry> {
        self.link.send(&Command::Param { suite, key, value })?;
        let until = deadline(self.opts.reply_timeout);
        loop {
            match self.wait_event(until) {
                Wait::Closed => return Err(closed()),
                Wait::Timeout => {
                    return Err(refused("the target did not answer"));
                }
                Wait::Event(event) => {
                    let found = self.note_setting(suite, &event);
                    if let Some(entry) = found {
                        return Ok(entry);
                    }
                }
            }
        }
    }

    /// Records a setting echo for `suite` in the catalog and returns the entry.
    pub(crate) fn note_setting(
        &mut self,
        suite: u16,
        event: &OwnedTelemetry,
    ) -> Option<SettingEntry> {
        let OwnedTelemetry::SettingInfo {
            suite_id,
            setting_id,
            ..
        } = *event
        else {
            return None;
        };
        if suite_id != suite {
            return None;
        }
        self.catalog.apply(event);
        self.catalog
            .suites
            .get(usize::from(suite))?
            .settings
            .get(usize::from(setting_id))?
            .clone()
    }

    /// Records one log line in the console and the log queue.
    fn note_log(&mut self, line: String) {
        self.console.push_str(&line);
        self.console.push('\n');
        self.log.push(line);
    }

    /// Waits for the next telemetry message until `until`.
    pub(crate) fn wait_event(&mut self, until: Instant) -> Wait {
        loop {
            match self.link.poll() {
                Polled::Event(LinkEvent::Telemetry(OwnedTelemetry::Log {
                    level,
                    target,
                    message,
                })) => {
                    let tag = match level {
                        0 => "ERROR",
                        1 => "WARN",
                        _ => "INFO",
                    };
                    self.note_log(format!("[{tag}] {target}: {message}"));
                }
                Polled::Event(LinkEvent::Telemetry(t)) => {
                    return Wait::Event(Box::new(t));
                }
                Polled::Event(LinkEvent::Console(text)) => {
                    self.note_log(text);
                }
                Polled::Idle => {
                    if Instant::now() >= until {
                        return Wait::Timeout;
                    }
                    thread::sleep(POLL_PAUSE);
                }
                Polled::Closed => return Wait::Closed,
            }
        }
    }
}

/// The host-side view of a wire `ItemState`.
#[must_use]
pub(crate) const fn state_of(
    state: TestState,
    verdict: Option<Verdict>,
) -> control_rs_ets::wire::ItemState {
    use control_rs_ets::wire::ItemState;
    match (state, verdict) {
        (_, Some(v)) => ItemState::Done(v),
        (TestState::Running, None) => ItemState::Running,
        (TestState::Pending, None) => ItemState::Idle,
        _ => ItemState::Faulted,
    }
}

/// The statistics carried by a `RunStats` event.
pub(crate) const fn stats_of(event: &OwnedTelemetry) -> Option<Stats> {
    match *event {
        OwnedTelemetry::RunStats {
            cycles,
            jitter_max_ns,
            missed_releases,
            stack_peak,
            step_max_ns,
            step_min_ns,
            steps,
            time_ns,
            ..
        } => Some(Stats {
            cycles,
            jitter_max_ns,
            missed_releases,
            stack_peak,
            step_max_ns,
            step_min_ns,
            steps,
            time_ns,
        }),
        _ => None,
    }
}

/// The instant `after` from now.
pub(crate) fn deadline(after: Duration) -> Instant {
    let now = Instant::now();
    now.checked_add(after).unwrap_or(now)
}

/// A request the host refuses.
pub(crate) fn refused(reason: &str) -> HostError {
    HostError::Refused {
        reason: reason.to_string(),
    }
}

/// The link ended while a reply was due.
pub(crate) fn closed() -> HostError {
    HostError::Transport {
        source: "the link closed while waiting for the target".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{Fake, ScriptLink, quick};

    /// Collects the ids of finished cases.
    struct Collect(Vec<ItemId>);

    impl CaseObserver for Collect {
        fn on_outcome(&mut self, outcome: &CaseOutcome) {
            self.0.push(outcome.id);
        }
    }

    fn open(fake: Fake) -> Session<ScriptLink> {
        Session::open_with(fake.link(), quick()).unwrap()
    }

    #[test]
    fn open_poll_close() {
        let mut s = open(Fake::default());
        assert_eq!(s.device_state(), DeviceState::Idle);
        assert_eq!(s.console(), "");
        s.close();
        assert!(s.link.exited().unwrap());
        assert_eq!(s.link.describe(), "script");
    }

    #[test]
    fn target_logs_and_console_text_are_queued() {
        let mut link = Fake::default().link();
        link.preload.push(LinkEvent::Console("boot".to_string()));
        link.preload.push(LinkEvent::Telemetry(OwnedTelemetry::Log {
            level: 1,
            target: "pid".to_string(),
            message: "clamped".to_string(),
        }));
        let mut s = Session::open_with(link, quick()).unwrap();
        assert_eq!(s.take_log(), ["boot", "[WARN] pid: clamped"]);
        assert_eq!(s.take_log(), Vec::<String>::new());
        assert!(s.console().contains("clamped"));
        assert_eq!(s.link_description(), "script");
        assert!(!s.details().release_source);
        assert_eq!(s.details().board_id, 7);
        assert!(!s.exited().unwrap());
    }

    #[test]
    fn discovery_and_run_queue() {
        let mut s = open(Fake::default());
        assert_eq!(s.catalog().case_ids(), [(0, 0), (0, 1)]);
        let mut seen = Collect(Vec::new());
        s.run_cases([(0, 0)], &mut seen).unwrap();
        assert_eq!(seen.0, [(0, 0)]);
        let outcome = s.results().cases.first().unwrap();
        assert_eq!(outcome.status, CaseStatus::Done(Verdict::Pass));
        assert_eq!(outcome.stats.map(|st| st.cycles), Some(10));
        assert!(s.run_case((0, 9)).is_err());
    }

    #[test]
    fn run_cases_excludes_tasks() {
        let mut s = open(Fake::default());
        let mut seen = Collect(Vec::new());
        let all: Vec<ItemId> = s.catalog().exec_ids().collect();
        assert!(all.contains(&(0, 2)));
        s.run_cases(all, &mut seen).unwrap();
        assert_eq!(seen.0, [(0, 0), (0, 1)]);
        assert!(s.run_case((0, 2)).is_err());
    }

    #[test]
    fn panic_restart_keeps_results() {
        let mut s = open(Fake {
            panic_item: Some(1),
            ..Fake::default()
        });
        let mut seen = Collect(Vec::new());
        s.run_cases([(0, 0), (0, 1), (0, 0)], &mut seen).unwrap();
        let cases = &s.results().cases;
        assert_eq!(cases.len(), 3);
        assert_eq!(
            cases.get(1).unwrap().status,
            CaseStatus::Faulted {
                file: "f.rs".to_string(),
                line: 3,
                message: "boom".to_string(),
            }
        );
        assert_eq!(
            cases.get(2).unwrap().status,
            CaseStatus::Done(Verdict::Pass)
        );
        assert_eq!(s.results().resets, 1);
        assert_eq!(s.link.resets, 1);
    }

    #[test]
    fn wire_revision_mismatch() {
        let link = Fake {
            version: PROTOCOL_VERSION.wrapping_add(1),
            ..Fake::default()
        }
        .link();
        let err = Session::open_with(link, quick()).err().unwrap();
        assert!(matches!(err, HostError::ProtocolMismatch { .. }));
    }

    #[test]
    fn setting_query_and_update() {
        let mut s = open(Fake::default());
        let by_name = s.get_setting(0, ParamKey::Name("gain")).unwrap();
        let by_index = s.get_setting(0, ParamKey::Index(0)).unwrap();
        assert_eq!(by_name, by_index);
        let echoed = s.set_setting(0, "gain", Param::U8(5)).unwrap();
        assert_eq!(echoed.value, Param::U8(5));
        let refused = s.set_setting(0, "gain", Param::U8(200)).unwrap();
        assert_eq!(refused.value, Param::U8(5));
        assert!(s.get_setting(0, ParamKey::Name("none")).is_err());
    }

    #[test]
    fn device_state_tracking() {
        let mut s = open(Fake::default());
        assert_eq!(s.device_state(), DeviceState::Idle);
        s.run_case((0, 0)).unwrap();
        assert_eq!(s.device_state(), DeviceState::Idle);
        s.link.mute = true;
        s.link.closed = true;
        assert!(matches!(
            s.run_case((0, 1)),
            Err(HostError::Transport { .. })
        ));
        assert_eq!(s.device_state(), DeviceState::Running((0, 1)));
    }

    #[test]
    fn silent_reset_during_run_is_unreported_fault() {
        let mut s = open(Fake {
            hang_on: Some(0),
            ..Fake::default()
        });
        let outcome = s.run_case((0, 0)).unwrap();
        assert_eq!(outcome.status, CaseStatus::UnreportedFault);
        assert_eq!(s.results().resets, 1);
        assert_eq!(s.device_state(), DeviceState::Idle);
    }

    #[test]
    fn discovery_and_setting_edit_without_elf() {
        let mut s = open(Fake {
            drop_setting: Some(1),
            ..Fake::default()
        });
        let settings: Vec<_> = s
            .catalog()
            .suites
            .first()
            .unwrap()
            .settings
            .iter()
            .flatten()
            .map(|e| e.name.as_str())
            .collect();
        assert_eq!(settings, ["gain", "bias"], "the dropped entry is refilled");
        let echoed = s.set_setting(0, "bias", Param::U8(4)).unwrap();
        assert_eq!(echoed.name, "bias");
        assert_eq!(s.catalog().missing_settings(), Vec::<ItemId>::new());
    }

    #[test]
    fn a_target_without_discovery_is_an_error() {
        let mut quiet = Fake::default().link();
        quiet.mute = true;
        assert!(matches!(
            Session::open_with(quiet, quick()).err(),
            Some(HostError::Discovery)
        ));
        let mut gone = Fake::default().link();
        gone.mute = true;
        gone.closed = true;
        gone.preload.push(LinkEvent::Console("boot".to_string()));
        assert!(matches!(
            Session::open_with(gone, quick()).err(),
            Some(HostError::Transport { .. })
        ));
        assert_eq!(open(Fake::default()).console(), "");
    }

    #[test]
    fn dropped_executable_fails_discovery() {
        let err = Session::open_with(
            Fake {
                drop_exec: Some(1),
                ..Fake::default()
            }
            .link(),
            quick(),
        )
        .err();
        assert!(matches!(err, Some(HostError::Discovery)));
    }

    #[test]
    fn done_without_stats_keeps_verdict() {
        let mut s = open(Fake {
            omit_stats: Some((0, Verdict::Fail)),
            ..Fake::default()
        });
        let outcome = s.run_case((0, 0)).unwrap();
        assert_eq!(outcome.status, CaseStatus::Done(Verdict::Fail));
        assert!(outcome.stats.is_none());
        assert_eq!(s.device_state(), DeviceState::Idle);
    }
}
