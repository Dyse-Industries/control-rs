//! Lifecycle task runs with a host-side plant.
//!
//! # Description
//!
//! [`Session::run_task`] starts a lifecycle task and closes the loop: it hands every output
//! sample `k` to a [`TaskSim`], which returns input `k + 1`, and delivers every step and the
//! final record to a [`RunObserver`]. A [`RunControl`] carries a stop request and queued
//! setting writes from another thread.
//!
//! # Core Concepts
//!
//! - **`TaskSim`**: Maps task output `k` to task input `k + 1`. A simulation with
//!   `DRIVES_INPUT = false` only listens, for a task that needs no input.
//! - **Escalation**: A stop request or the elapsed duration sends `StopNow`. If the run does not
//!   end within `stop_timeout`, the session resets the target through the link and records the
//!   run as [`RunStatus::Unresponsive`].
//! - **Silence**: A target that sends nothing for `silence` while a run is active is an
//!   unreported fault.
//! - **Release-paced runs**: `Triggered` is offered only when the target reports a release
//!   source. Missed releases and release jitter appear in the record.
//!
//! # Usage
//!
//! ```
//! use control_rs_ets_host::task_run::{MemoryObserver, RunControl, TaskOptions};
//!
//! let control = RunControl::default();
//! assert!(!control.stop_requested());
//! control.stop();
//! assert!(control.stop_requested());
//! control.clear();
//! assert!(!control.stop_requested());
//! let _ = (MemoryObserver::default(), TaskOptions::default());
//! ```
//!
//! # Limitations
//!
//! - **Blocking**: The call returns when the run ends; stop and setting writes from another
//!   thread go through [`RunControl`].

use std::fmt::Write as _;
use std::io::Write;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use control_rs_ets::exec::Packet;
use control_rs_ets::wire::{
    Command, ExecMode, ItemState, MAX_MESSAGE_SIZE, Param, ParamKey, Verdict,
};

use crate::bridge::{OwnedTelemetry, PacketDecl};
use crate::catalog::ItemId;
use crate::error::HostError;
use crate::link::HostLink;
use crate::session::{
    Session, SessionResult, Wait, closed, deadline, refused, state_of, stats_of,
};

/// Interval between heartbeats while a task runs.
const HEARTBEAT: Duration = Duration::from_millis(100);

/// Longest wait for one event inside the run loop.
const SLICE: Duration = Duration::from_millis(5);

/// A queued setting write: suite, name, value.
type QueuedWrite = (u16, String, Param);

/// The queue of setting writes.
type Queue = Mutex<Vec<QueuedWrite>>;

/// States a run went through, with their messages.
type States = Vec<(ItemState, Option<String>)>;

/// Output samples with their step.
type Samples = Vec<(u64, Vec<u8>)>;

/// Maps task output to the next task input.
pub trait TaskSim {
    /// Packet type the task consumes.
    type Input: Packet;
    /// Packet type the task produces.
    type Output: Packet;
    /// Whether this simulation produces the task's input.
    const DRIVES_INPUT: bool = true;

    /// Input `0`.
    fn initial(&mut self) -> Self::Input;

    /// Input `k + 1` from output `k`.
    fn advance(&mut self, k: u64, y: &Self::Output) -> Self::Input;
}

/// Receives the steps and the final record of a run.
pub trait RunObserver {
    /// Called with the bytes of output sample `k`.
    fn on_step(&mut self, _k: u64, _y: &[u8]) {}

    /// Called when the run changes state.
    fn on_state(&mut self, _state: ItemState, _message: Option<&str>) {}

    /// Called once with the final record.
    fn on_end(&mut self, _record: &TaskRecord) {}

    /// Called with each log line the target or its console produced during the run.
    fn on_log(&mut self, _line: &str) {}
}

/// Stop and setting requests from another thread.
#[derive(Debug, Default)]
pub struct RunControl {
    settings: Queue,
    stop: AtomicBool,
}

/// How a task run is configured.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TaskOptions {
    /// Stop the run after this long (`None` runs until the step bound or a stop request).
    pub duration: Option<Duration>,
    /// Step bound (`0` is unbounded).
    pub max_steps: u64,
    /// Mode of the run.
    pub mode: ExecMode,
    /// Release period in nanoseconds for a `Periodic` run.
    pub period_ns: u32,
    /// Silence that counts as an unreported fault (`None` waits forever).
    pub silence: Option<Duration>,
}

/// How a run ended, besides the verdict.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum RunStatus {
    /// The target reported the end of the run.
    #[default]
    Completed,
    /// The target reported a panic or exception.
    Faulted,
    /// The target ignored `StopNow` and was reset.
    Unresponsive,
    /// The target went silent without a report.
    UnreportedFault,
}

/// The record of a task run.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TaskRecord {
    /// Largest release jitter in nanoseconds.
    pub jitter_max_ns: u32,
    /// Releases that arrived before the previous step finished.
    pub missed_releases: u32,
    /// Output samples missing from the sequence.
    pub seq_gaps: u64,
    /// How the run ended.
    pub status: RunStatus,
    /// Longest step in nanoseconds.
    pub step_max_ns: u32,
    /// Shortest step in nanoseconds.
    pub step_min_ns: u32,
    /// Steps the target executed.
    pub steps: u64,
    /// Whether teardown succeeded, when it was reported.
    pub teardown_ok: Option<bool>,
    /// Verdict of the final state message.
    pub verdict: Option<Verdict>,
    /// Problems found before the run, such as a packet mismatch.
    pub warnings: Vec<String>,
    /// Wall-clock time of the run.
    pub wall_time: Duration,
}

/// Keeps every step, state and the record in memory.
#[derive(Debug, Default)]
pub struct MemoryObserver {
    /// The final record, once the run ended.
    pub record: Option<TaskRecord>,
    /// States in order.
    pub states: States,
    /// Output samples in order.
    pub steps: Samples,
}

/// Writes each output sample as a CSV row: step, then the bytes in hexadecimal.
pub struct CsvObserver<W: Write> {
    /// Whether a write failed; later rows are dropped.
    pub failed: bool,
    out: W,
}

/// A simulation for a task that needs no input.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoSim;

/// A packet with no content.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Empty;

/// One task run: what to run, how, against which simulation and observer.
pub struct TaskRun<'a, S, O> {
    /// Stop and setting requests from another thread.
    pub control: &'a RunControl,
    /// The task.
    pub id: ItemId,
    /// Receives the steps and the record.
    pub observer: &'a mut O,
    /// How to run.
    pub opts: TaskOptions,
    /// The host-side plant.
    pub sim: S,
}

/// State of one run loop.
struct Loop<'a> {
    control: &'a RunControl,
    expected_seq: u64,
    id: ItemId,
    last_event: Instant,
    last_heartbeat: Instant,
    opts: TaskOptions,
    record: TaskRecord,
    started: Instant,
    stop_sent: Option<Instant>,
}

/// An output sample: its step and its bytes.
type Sample<'a> = (u64, &'a [u8]);

/// The plant and the observer of a run.
type Rig<'a, S, O> = (&'a mut S, &'a mut O);

impl Default for TaskOptions {
    fn default() -> Self {
        Self {
            duration: None,
            max_steps: 0,
            mode: ExecMode::Lockstep,
            period_ns: 0,
            silence: Some(Duration::from_secs(10)),
        }
    }
}

impl RunControl {
    /// Asks the run to stop.
    pub fn stop(&self) {
        self.stop.store(true, Ordering::SeqCst);
    }

    /// Withdraws a stop request, so the next run starts fresh.
    pub fn clear(&self) {
        self.stop.store(false, Ordering::SeqCst);
    }

    /// Whether a stop was requested.
    #[must_use]
    pub fn stop_requested(&self) -> bool {
        self.stop.load(Ordering::SeqCst)
    }

    /// Queues a setting write for the next iteration of the run loop.
    pub fn write_setting(&self, suite: u16, name: &str, value: Param) {
        if let Ok(mut queue) = self.settings.lock() {
            queue.push((suite, name.to_string(), value));
        }
    }

    fn take_settings(&self) -> Vec<QueuedWrite> {
        self.settings
            .lock()
            .map(|mut queue| std::mem::take(&mut *queue))
            .unwrap_or_default()
    }
}

impl RunObserver for MemoryObserver {
    fn on_end(&mut self, record: &TaskRecord) {
        self.record = Some(record.clone());
    }

    fn on_state(&mut self, state: ItemState, message: Option<&str>) {
        self.states.push((state, message.map(str::to_string)));
    }

    fn on_step(&mut self, k: u64, y: &[u8]) {
        self.steps.push((k, y.to_vec()));
    }
}

impl<W: Write> CsvObserver<W> {
    /// Writes rows to `out`.
    pub const fn new(out: W) -> Self {
        Self { failed: false, out }
    }
}

impl<W: Write> RunObserver for CsvObserver<W> {
    fn on_step(&mut self, k: u64, y: &[u8]) {
        if self.failed {
            return;
        }
        let hex = y.iter().fold(String::new(), |mut acc, b| {
            let _ = write!(acc, "{b:02x}");
            acc
        });
        self.failed = writeln!(self.out, "{k},{hex}").is_err();
    }
}

impl Packet for Empty {
    const INFO: control_rs_ets::wire::PacketInfo<'static> =
        control_rs_ets::wire::PacketInfo {
            type_name: "",
            encoding: control_rs_ets::wire::PayloadEncoding::Postcard,
        };

    fn decode(_bytes: &[u8]) -> Option<Self> {
        Some(Self)
    }

    fn encode(&self, _out: &mut [u8]) -> Option<usize> {
        Some(0)
    }
}

impl TaskSim for NoSim {
    type Input = Empty;
    type Output = Empty;

    const DRIVES_INPUT: bool = false;

    fn initial(&mut self) -> Empty {
        Empty
    }

    fn advance(&mut self, _k: u64, _y: &Empty) -> Empty {
        Empty
    }
}

impl<L: HostLink<Error = HostError>> Session<L> {
    /// Runs a task against its simulation until it ends.
    ///
    /// # Errors
    /// Returns [`HostError::Refused`] when the item is not a task, the mode is not allowed, or
    /// `Triggered` is requested from a target without a release source;
    /// [`HostError::Simulation`] when a packet does not fit; or a transport error.
    pub fn run_task<S: TaskSim, O: RunObserver>(
        &mut self,
        run: TaskRun<'_, S, O>,
    ) -> SessionResult<TaskRecord> {
        let TaskRun {
            control,
            id,
            observer,
            opts,
            mut sim,
        } = run;
        self.admit_task(id, &opts)?;
        let period_ns = if opts.mode == ExecMode::Triggered {
            0
        } else {
            opts.period_ns
        };
        self.link.send(&Command::RunExecutable {
            suite: id.0,
            item: id.1,
            mode: opts.mode,
            max_steps: opts.max_steps,
            period_ns,
        })?;
        self.device = crate::session::DeviceState::Running(id);
        if S::DRIVES_INPUT {
            let first = sim.initial();
            self.send_input(id, 0, &first)?;
        }
        let now = Instant::now();
        let mut state = Loop {
            control,
            expected_seq: 0,
            id,
            last_event: now,
            last_heartbeat: now,
            opts,
            record: TaskRecord {
                warnings: packet_warnings::<S>(self, id),
                ..TaskRecord::default()
            },
            started: now,
            stop_sent: None,
        };
        let result = self.drive(&mut state, (&mut sim, &mut *observer));
        state.record.wall_time = state.started.elapsed();
        result?;
        observer.on_end(&state.record);
        Ok(state.record)
    }

    /// Checks that the task can run as requested.
    fn admit_task(&self, id: ItemId, opts: &TaskOptions) -> SessionResult<()> {
        let Some(exec) = self.catalog.exec(id).filter(|e| e.is_task()) else {
            return Err(refused("not a task of this target"));
        };
        if !exec.allows(opts.mode) {
            return Err(refused("mode not allowed for this task"));
        }
        if opts.mode == ExecMode::Triggered && !self.details.release_source {
            return Err(refused("target reports no release source"));
        }
        Ok(())
    }

    /// The event loop of a run.
    fn drive<S: TaskSim, O: RunObserver>(
        &mut self,
        state: &mut Loop<'_>,
        rig: Rig<'_, S, O>,
    ) -> SessionResult<()> {
        let (sim, observer) = rig;
        loop {
            if self.escalate(state)? {
                return Ok(());
            }
            for (suite, name, value) in state.control.take_settings() {
                self.link.send(&Command::Param {
                    suite,
                    key: ParamKey::Name(&name),
                    value: Some(value),
                })?;
            }
            let waited = self.wait_event(deadline(SLICE));
            for line in self.take_log() {
                observer.on_log(&line);
            }
            match waited {
                Wait::Closed => return Err(closed()),
                Wait::Timeout => {
                    if state
                        .opts
                        .silence
                        .is_some_and(|s| state.last_event.elapsed() > s)
                    {
                        state.record.status = RunStatus::UnreportedFault;
                        let _ = self.recover();
                        return Ok(());
                    }
                }
                Wait::Event(event) => {
                    state.last_event = Instant::now();
                    if self.on_event(
                        &event,
                        state,
                        (&mut *sim, &mut *observer),
                    )? {
                        return Ok(());
                    }
                }
            }
        }
    }

    /// Sends heartbeats and `StopNow`; resets an unresponsive target. Returns `true` when the
    /// run was abandoned.
    fn escalate(&mut self, state: &mut Loop<'_>) -> SessionResult<bool> {
        let due = state
            .opts
            .duration
            .is_some_and(|d| state.started.elapsed() >= d);
        if (state.control.stop_requested() || due) && state.stop_sent.is_none()
        {
            self.link.send(&Command::StopNow {
                suite: state.id.0,
                item: state.id.1,
            })?;
            state.stop_sent = Some(Instant::now());
        }
        if state
            .stop_sent
            .is_some_and(|t| t.elapsed() > self.opts.stop_timeout)
        {
            state.record.status = RunStatus::Unresponsive;
            let _ = self.recover();
            return Ok(true);
        }
        if state.last_heartbeat.elapsed() >= HEARTBEAT {
            self.link.send(&Command::Heartbeat)?;
            state.last_heartbeat = Instant::now();
        }
        Ok(false)
    }

    /// Applies `RunStats` to the record and ends the run.
    ///
    /// Stats without a prior Done leave `verdict` unset; that is silence, not
    /// a clean `Completed` run.
    const fn finish_with_stats(
        &mut self,
        event: &OwnedTelemetry,
        state: &mut Loop<'_>,
    ) -> bool {
        if let Some(stats) = stats_of(event) {
            state.record.jitter_max_ns = stats.jitter_max_ns;
            state.record.missed_releases = stats.missed_releases;
            state.record.step_max_ns = stats.step_max_ns;
            state.record.step_min_ns = stats.step_min_ns;
            state.record.steps = stats.steps;
        }
        if state.record.verdict.is_none() {
            state.record.status = RunStatus::UnreportedFault;
        }
        self.device = crate::session::DeviceState::Idle;
        true
    }

    /// Handles one telemetry message of a run. Returns `true` when the run ended.
    fn on_event<S: TaskSim, O: RunObserver>(
        &mut self,
        event: &OwnedTelemetry,
        state: &mut Loop<'_>,
        rig: Rig<'_, S, O>,
    ) -> SessionResult<bool> {
        let (sim, observer) = rig;
        let id = state.id;
        match event {
            OwnedTelemetry::TaskSample {
                payload,
                seq,
                suite_id,
                test_id,
            } if (*suite_id, *test_id) == id => {
                observer.on_step(*seq, payload);
                self.on_sample((*seq, payload), state, sim)?;
            }
            OwnedTelemetry::TestStateChange {
                suite_id,
                test_id,
                state: item_state,
                verdict,
                message,
            } if (*suite_id, *test_id) == id => {
                observer.on_state(
                    state_of(*item_state, *verdict),
                    message.as_deref(),
                );
                if verdict.is_some() {
                    state.record.verdict = *verdict;
                }
            }
            OwnedTelemetry::TeardownReport {
                ok,
                suite_id,
                test_id,
                ..
            } if (*suite_id, *test_id) == id => {
                state.record.teardown_ok = Some(*ok);
            }
            OwnedTelemetry::RunStats {
                suite_id, test_id, ..
            } if (*suite_id, *test_id) == id => {
                return Ok(self.finish_with_stats(event, state));
            }
            OwnedTelemetry::TargetPanic { .. } => {
                state.record.status = RunStatus::Faulted;
                let _ = self.recover();
                return Ok(true);
            }
            other => {
                let _ = self.note_setting(id.0, other);
            }
        }
        Ok(false)
    }

    /// Handles output sample `seq`: records gaps and answers with the next input.
    fn on_sample<S: TaskSim>(
        &mut self,
        (seq, payload): Sample<'_>,
        state: &mut Loop<'_>,
        sim: &mut S,
    ) -> SessionResult<()> {
        state.record.seq_gaps = state
            .record
            .seq_gaps
            .saturating_add(seq.saturating_sub(state.expected_seq));
        state.expected_seq = seq.saturating_add(1);
        if S::DRIVES_INPUT {
            let y = S::Output::decode(payload)
                .ok_or_else(|| simulation("an output packet did not decode"))?;
            let next = sim.advance(seq, &y);
            self.send_input(state.id, seq.saturating_add(1), &next)?;
        }
        Ok(())
    }

    /// Sends input packet `seq` of the task `id`.
    fn send_input<P: Packet>(
        &mut self,
        id: ItemId,
        seq: u64,
        packet: &P,
    ) -> SessionResult<()> {
        let mut buf = [0u8; MAX_MESSAGE_SIZE];
        let used = packet
            .encode(&mut buf)
            .ok_or_else(|| simulation("an input packet does not fit"))?;
        self.link.send(&Command::TaskInput {
            suite: id.0,
            item: id.1,
            seq,
            payload: buf.get(..used).unwrap_or(&[]),
        })
    }
}

/// Warnings for packet declarations that differ from the simulation's packets.
fn packet_warnings<S: TaskSim>(
    session: &Session<impl HostLink<Error = HostError>>,
    id: ItemId,
) -> Vec<String> {
    let Some(exec) = session.catalog.exec(id) else {
        return Vec::new();
    };
    let mut warnings = Vec::new();
    for (side, declared, expected) in [
        ("input", &exec.input, S::Input::INFO),
        ("output", &exec.output, S::Output::INFO),
    ] {
        let want = PacketDecl::from(expected);
        if S::DRIVES_INPUT && *declared != want {
            warnings.push(format!(
                "{side} packet is declared as {:?}, the simulation uses {:?}",
                declared.type_name, want.type_name
            ));
        }
    }
    warnings
}

/// A failure of the host-side simulation.
fn simulation(what: &str) -> HostError {
    HostError::Simulation {
        source: what.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::Session;
    use crate::testing::{Fake, Increment, ScriptLink, quick};

    const TASK: ItemId = (0, 2);

    macro_rules! go {
        ($session:expr, $id:expr, $opts:expr, $sim:expr, $observer:expr, $control:expr $(,)?) => {
            $session.run_task(TaskRun {
                control: $control,
                id: $id,
                observer: $observer,
                opts: *$opts,
                sim: $sim,
            })
        };
    }

    fn open(fake: Fake) -> Session<ScriptLink> {
        Session::open_with(fake.link(), quick()).unwrap()
    }

    fn bounded(steps: u64) -> TaskOptions {
        TaskOptions {
            max_steps: steps,
            ..TaskOptions::default()
        }
    }

    #[test]
    fn task_sim_exchange() {
        let mut s = open(Fake::default());
        let mut obs = MemoryObserver::default();
        let rec = go!(
            s,
            TASK,
            &bounded(3),
            Increment,
            &mut obs,
            &RunControl::default(),
        )
        .unwrap();
        // The target doubles its input and the simulation adds one: 1, 3, 7 in; 2, 6, 14 out.
        let bytes: Vec<u8> =
            obs.steps.iter().flat_map(|(_, y)| y.clone()).collect();
        assert_eq!(bytes, [2, 6, 14]);
        assert_eq!(rec.steps, 3);
        assert_eq!(rec.verdict, Some(Verdict::Bounded));
        assert_eq!(rec.teardown_ok, Some(true));
        assert_eq!(rec.seq_gaps, 0);
        assert_eq!(rec.warnings, Vec::<String>::new());
    }

    #[test]
    fn steps_and_final_record() {
        let mut s = open(Fake::default());
        let mut obs = MemoryObserver::default();
        let rec = go!(
            s,
            TASK,
            &bounded(2),
            Increment,
            &mut obs,
            &RunControl::default(),
        )
        .unwrap();
        assert_eq!(obs.steps.len(), 2);
        assert_eq!(obs.record.as_ref(), Some(&rec));
        assert!(
            obs.states
                .iter()
                .any(|(st, _)| matches!(st, ItemState::Done(Verdict::Bounded)))
        );
        let mut csv = CsvObserver::new(Vec::new());
        csv.on_step(4, &[0xAB, 1]);
        assert_eq!(String::from_utf8(csv.out).unwrap(), "4,ab01\n");
    }

    #[test]
    fn a_stop_request_aborts_a_run() {
        let mut s = open(Fake::default());
        let control = RunControl::default();
        control.stop();
        let rec = go!(
            s,
            TASK,
            &bounded(0),
            Increment,
            &mut MemoryObserver::default(),
            &control,
        )
        .unwrap();
        assert_eq!(rec.verdict, Some(Verdict::Aborted));
        assert_eq!(rec.status, RunStatus::Completed);
    }

    #[test]
    fn blocked_step_resets_externally_and_reports_unresponsive() {
        let mut s = open(Fake {
            ignore_stop: true,
            ..Fake::default()
        });
        let control = RunControl::default();
        control.stop();
        let rec = go!(
            s,
            TASK,
            &bounded(0),
            Increment,
            &mut MemoryObserver::default(),
            &control,
        )
        .unwrap();
        assert_eq!(rec.status, RunStatus::Unresponsive);
        assert_eq!(s.link.resets, 1);
        assert!(s.link.sent.iter().any(|c| c.contains("StopNow")));
        assert_eq!(s.device_state(), crate::session::DeviceState::Idle);
    }

    #[test]
    fn silent_task_is_an_unreported_fault() {
        let mut s = open(Fake {
            hang_on: Some(2),
            ..Fake::default()
        });
        let opts = TaskOptions {
            silence: Some(Duration::from_millis(30)),
            ..bounded(0)
        };
        let rec = go!(
            s,
            TASK,
            &opts,
            Increment,
            &mut MemoryObserver::default(),
            &RunControl::default(),
        )
        .unwrap();
        assert_eq!(rec.status, RunStatus::UnreportedFault);
        assert_eq!(s.results().resets, 1);
    }

    #[test]
    fn triggered_gated_by_release_source() {
        let opts = TaskOptions {
            mode: ExecMode::Triggered,
            ..bounded(1)
        };
        let mut without = open(Fake::default());
        let err = go!(
            without,
            TASK,
            &opts,
            NoSim,
            &mut MemoryObserver::default(),
            &RunControl::default(),
        )
        .unwrap_err();
        assert!(matches!(err, HostError::Refused { .. }));
        assert!(
            !without
                .link
                .sent
                .iter()
                .any(|c| c.contains("RunExecutable"))
        );

        let mut with = open(Fake {
            release_source: true,
            ..Fake::default()
        });
        let rec = go!(
            with,
            TASK,
            &opts,
            NoSim,
            &mut MemoryObserver::default(),
            &RunControl::default(),
        )
        .unwrap();
        assert_eq!(rec.verdict, Some(Verdict::Bounded));
        assert!(
            with.link.sent.iter().any(|c| c.contains("period_ns: 0")),
            "a triggered run starts without a period"
        );
    }

    #[test]
    fn missed_releases_recorded() {
        let mut s = open(Fake {
            missed: 2,
            release_source: true,
            ..Fake::default()
        });
        let opts = TaskOptions {
            mode: ExecMode::Triggered,
            ..bounded(1)
        };
        let rec = go!(
            s,
            TASK,
            &opts,
            NoSim,
            &mut MemoryObserver::default(),
            &RunControl::default(),
        )
        .unwrap();
        assert_eq!(rec.missed_releases, 2);
        assert_eq!(rec.jitter_max_ns, 500);
    }

    #[test]
    fn requests_the_host_refuses_are_not_sent() {
        let mut s = open(Fake::default());
        let obs = &mut MemoryObserver::default();
        let control = RunControl::default();
        assert!(go!(s, (0, 0), &bounded(1), NoSim, obs, &control).is_err());
        let atomic = TaskOptions {
            mode: ExecMode::Atomic,
            ..bounded(1)
        };
        assert!(go!(s, TASK, &atomic, NoSim, obs, &control).is_ok());
    }

    #[test]
    fn settings_queued_in_the_control_reach_the_target() {
        let mut s = open(Fake::default());
        let control = RunControl::default();
        control.write_setting(0, "gain", Param::U8(6));
        go!(
            s,
            TASK,
            &bounded(2),
            Increment,
            &mut MemoryObserver::default(),
            &control,
        )
        .unwrap();
        assert_eq!(
            s.catalog().setting(0, "gain").map(|e| e.value),
            Some(Param::U8(6))
        );
    }

    #[test]
    fn a_mismatched_packet_declaration_is_a_warning() {
        struct Wide;
        impl TaskSim for Wide {
            type Input = Empty;
            type Output = Empty;
            fn initial(&mut self) -> Empty {
                Empty
            }
            fn advance(&mut self, _k: u64, _y: &Empty) -> Empty {
                Empty
            }
        }
        let mut s = open(Fake::default());
        let result = go!(
            s,
            TASK,
            &bounded(1),
            Wide,
            &mut MemoryObserver::default(),
            &RunControl::default(),
        );
        // The target's one-byte reply does not decode as `Empty`... which accepts anything.
        let rec = result.unwrap();
        assert_eq!(rec.warnings.len(), 2);
    }

    #[test]
    fn orphan_run_stats_is_an_unreported_fault() {
        let mut s = open(Fake {
            orphan_stats: Some(2),
            ..Fake::default()
        });
        let rec = go!(
            s,
            TASK,
            &bounded(1),
            Increment,
            &mut MemoryObserver::default(),
            &RunControl::default(),
        )
        .unwrap();
        assert_eq!(rec.status, RunStatus::UnreportedFault);
        assert!(rec.verdict.is_none());
    }
}
