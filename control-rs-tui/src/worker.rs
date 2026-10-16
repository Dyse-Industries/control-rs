//! The worker thread that owns the session.
//!
//! The UI thread sends [`Request`]s and receives [`Update`]s. Stop requests go through the shared
//! [`RunControl`], so they reach a running task without waiting for the worker. The UI withdraws a
//! leftover stop when it issues a new run; this loop must not clear [`RunControl`] at dequeue, or a
//! Stop that arrived while the request was queued is lost. Rendering never blocks the session, and
//! a stalled terminal cannot back-pressure a run.

use std::sync::Arc;
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, channel};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use control_rs_ets::wire::{ExecMode, ItemState, Param};
use control_rs_ets_host::catalog::{
    Catalog, ItemId, SettingEntry, TargetDetails,
};
use control_rs_ets_host::session::{
    CaseOutcome, CaseStatus, Session, SessionOptions,
};
use control_rs_ets_host::task_run::{
    NoSim, RunControl, RunObserver, RunStatus, TaskOptions, TaskRecord, TaskRun,
};
use control_rs_ets_host::{ETSBridge, HostError};

/// How long the worker waits for a request before it checks the link.
const IDLE: Duration = Duration::from_millis(100);

/// Longest interval between samples shown to the operator (about 20 Hz).
const SAMPLE_INTERVAL: Duration = Duration::from_millis(50);

/// Join handle of the worker thread.
type Handle = Option<JoinHandle<()>>;

/// A request from the UI thread.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Request {
    /// Run one case.
    RunCase(ItemId),
    /// Run every case; tasks are never started.
    RunAll,
    /// Run a task in a mode that needs no plant.
    RunTask {
        /// The task.
        id: ItemId,
        /// Mode of the run.
        mode: ExecMode,
        /// Release period in nanoseconds for `Periodic`.
        period_ns: u32,
    },
    /// Write one setting by name.
    SetSetting {
        /// Suite that owns the setting.
        suite: u16,
        /// Setting name.
        name: String,
        /// New value, of the setting's type.
        value: Param,
    },
    /// Stop the worker.
    Quit,
    /// Stop the running task or run-all sequence.
    Stop,
}

/// An update from the worker thread.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Update {
    /// A case finished.
    Case(CaseOutcome),
    /// A case is starting.
    CaseStarted(ItemId),
    /// Discovery finished or was repeated after a reset.
    Catalog(Box<Catalog>, TargetDetails),
    /// A fault, an unreported fault or an unresponsive run.
    Fault(String),
    /// The link is gone; the worker has stopped.
    LinkLost(String),
    /// One log line.
    Log(String),
    /// The target speaks another wire revision; the worker waits for `Quit`.
    Mismatch {
        /// Revision of this host.
        host: u32,
        /// Revision of the target.
        target: u32,
    },
    /// Nothing is running.
    Idle,
    /// A decimated output sample of a running task.
    Sample {
        /// The task.
        id: ItemId,
        /// Step that produced the sample.
        k: u64,
        /// Encoded packet.
        bytes: Vec<u8>,
    },
    /// A setting value the target echoed.
    Setting(u16, Box<SettingEntry>),
    /// A task changed state.
    TaskState(ItemId, ItemState),
    /// A task run ended.
    TaskDone(ItemId, Box<TaskRecord>),
}

/// The worker thread and its channels.
pub struct Worker {
    /// Stop requests that reach a running task directly.
    pub control: Arc<RunControl>,
    handle: Handle,
    /// Requests to the worker.
    pub requests: Sender<Request>,
    /// Updates from the worker.
    pub updates: Receiver<Update>,
}

/// Forwards what a task run produces to the UI thread.
struct Feed<'a> {
    id: ItemId,
    last_sample: Option<Instant>,
    updates: &'a Sender<Update>,
}

impl Worker {
    /// Starts the worker over `bridge`.
    #[must_use]
    pub fn spawn(bridge: ETSBridge, opts: SessionOptions) -> Self {
        let (requests, request_rx) = channel();
        let (update_tx, updates) = channel();
        let control = Arc::new(RunControl::default());
        let shared = Arc::clone(&control);
        let handle = thread::spawn(move || {
            serve(Session::new(bridge, opts), &request_rx, &update_tx, &shared);
        });
        Self {
            control,
            handle: Some(handle),
            requests,
            updates,
        }
    }

    /// Stops the worker and waits for it.
    pub fn shutdown(&mut self) {
        let _ = self.requests.send(Request::Quit);
        self.control.stop();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.shutdown();
    }
}

impl RunObserver for Feed<'_> {
    fn on_end(&mut self, record: &TaskRecord) {
        let _ = self
            .updates
            .send(Update::TaskDone(self.id, Box::new(record.clone())));
    }

    fn on_log(&mut self, line: &str) {
        let _ = self.updates.send(Update::Log(line.to_string()));
    }

    fn on_state(&mut self, state: ItemState, _message: Option<&str>) {
        let _ = self.updates.send(Update::TaskState(self.id, state));
    }

    fn on_step(&mut self, k: u64, y: &[u8]) {
        let due = self
            .last_sample
            .is_none_or(|t| t.elapsed() >= SAMPLE_INTERVAL);
        if due {
            self.last_sample = Some(Instant::now());
            let _ = self.updates.send(Update::Sample {
                id: self.id,
                k,
                bytes: y.to_vec(),
            });
        }
    }
}

/// The worker loop: discover, then serve requests until `Quit` or a lost link.
fn serve(
    mut session: Session<ETSBridge>,
    requests: &Receiver<Request>,
    updates: &Sender<Update>,
    control: &RunControl,
) {
    if !discover(&mut session, requests, updates) {
        return;
    }
    announce(&session, updates);
    let _ = updates.send(Update::Idle);
    loop {
        match requests.recv_timeout(IDLE) {
            Ok(Request::Quit) | Err(RecvTimeoutError::Disconnected) => break,
            Ok(request) => {
                // Do not clear RunControl here: a Stop pressed after the UI queued
                // this request but before dequeue must still abort the run. The UI
                // clears when it issues a fresh run while idle.
                let resets = session.results().resets;
                if let Err(e) = handle(&mut session, request, updates, control)
                {
                    let _ = updates.send(Update::LinkLost(e.to_string()));
                    return;
                }
                flush_log(&mut session, updates);
                if session.results().resets != resets {
                    announce(&session, updates);
                }
                let _ = updates.send(Update::Idle);
            }
            Err(RecvTimeoutError::Timeout) => {
                flush_log(&mut session, updates);
                if session.exited().unwrap_or(false) {
                    let _ = updates
                        .send(Update::LinkLost("target exited".to_string()));
                    return;
                }
            }
        }
    }
    session.close();
}

/// Repeats discovery until it succeeds. Returns `false` when the worker should stop.
fn discover(
    session: &mut Session<ETSBridge>,
    requests: &Receiver<Request>,
    updates: &Sender<Update>,
) -> bool {
    loop {
        match session.discover() {
            Ok(()) => return true,
            Err(HostError::ProtocolMismatch { host, target }) => {
                let _ = updates.send(Update::Mismatch { host, target });
                while !matches!(requests.recv(), Ok(Request::Quit) | Err(_)) {}
                return false;
            }
            Err(HostError::Discovery) => {
                flush_log(session, updates);
                if session.exited().unwrap_or(false) {
                    let _ = updates
                        .send(Update::LinkLost("target exited".to_string()));
                    return false;
                }
                if matches!(requests.try_recv(), Ok(Request::Quit)) {
                    return false;
                }
            }
            Err(e) => {
                let _ = updates.send(Update::LinkLost(e.to_string()));
                return false;
            }
        }
    }
}

/// Handles one request; only a lost link is an error.
fn handle(
    session: &mut Session<ETSBridge>,
    request: Request,
    updates: &Sender<Update>,
    control: &RunControl,
) -> Result<(), HostError> {
    match request {
        Request::RunCase(id) => run_case(session, id, updates),
        Request::RunAll => {
            for id in session.catalog().case_ids() {
                if control.stop_requested() {
                    break;
                }
                run_case(session, id, updates)?;
            }
            Ok(())
        }
        Request::RunTask {
            id,
            mode,
            period_ns,
        } => {
            let mut feed = Feed {
                id,
                last_sample: None,
                updates,
            };
            let record = session.run_task(TaskRun {
                control,
                id,
                observer: &mut feed,
                opts: TaskOptions {
                    mode,
                    period_ns,
                    ..TaskOptions::default()
                },
                sim: NoSim,
            });
            match record {
                Ok(record) => report_task(updates, id, &record),
                Err(HostError::Refused { reason }) => {
                    let _ = updates.send(Update::Log(format!(
                        "[HOST] request refused: {reason}"
                    )));
                }
                Err(e) => return Err(e),
            }
            Ok(())
        }
        Request::SetSetting { suite, name, value } => {
            let entry = session.set_setting(suite, &name, value)?;
            let _ = updates.send(Update::Setting(suite, Box::new(entry)));
            Ok(())
        }
        Request::Quit | Request::Stop => Ok(()),
    }
}

/// Runs one case and reports it.
fn run_case(
    session: &mut Session<ETSBridge>,
    id: ItemId,
    updates: &Sender<Update>,
) -> Result<(), HostError> {
    let _ = updates.send(Update::CaseStarted(id));
    let outcome = session.run_case(id)?;
    match &outcome.status {
        CaseStatus::Faulted {
            file,
            line,
            message,
        } => {
            let _ = updates.send(Update::Fault(format!(
                "panic at {file}:{line}: {message}"
            )));
        }
        CaseStatus::UnreportedFault => {
            let _ = updates.send(Update::Fault(format!(
                "target went silent during {}/{}",
                id.0, id.1
            )));
        }
        CaseStatus::Done(_) => {}
    }
    let _ = updates.send(Update::Case(outcome));
    Ok(())
}

/// Reports the end of a task run, with a fault line when it did not complete.
fn report_task(updates: &Sender<Update>, id: ItemId, record: &TaskRecord) {
    let fault = match record.status {
        RunStatus::Completed => None,
        RunStatus::Faulted => Some("target panic during task"),
        RunStatus::Unresponsive => {
            Some("task ignored StopNow; target was reset")
        }
        RunStatus::UnreportedFault => Some("target went silent during task"),
    };
    if let Some(text) = fault {
        let _ =
            updates.send(Update::Fault(format!("{}/{}: {text}", id.0, id.1)));
    }
}

/// Publishes the catalog and the target details.
fn announce(session: &Session<ETSBridge>, updates: &Sender<Update>) {
    let _ = updates.send(Update::Catalog(
        Box::new(session.catalog().clone()),
        session.details(),
    ));
}

/// Forwards queued log lines.
fn flush_log(session: &mut Session<ETSBridge>, updates: &Sender<Update>) {
    for line in session.take_log() {
        let _ = updates.send(Update::Log(line));
    }
}

#[cfg(test)]
mod tests {
    use control_rs_ets::wire::PROTOCOL_VERSION;
    use control_rs_ets_host::{BridgeMessage, OwnedTelemetry};

    use super::*;

    fn quick() -> SessionOptions {
        SessionOptions {
            discovery_timeout: Duration::from_millis(40),
            ..SessionOptions::default()
        }
    }

    fn discovery(version: u8) -> Vec<OwnedTelemetry> {
        vec![
            OwnedTelemetry::TargetInfo {
                protocol_version: version,
                board_id: 1,
                core_clock_hz: 1_000_000,
                fpu_flags: 0,
                release_source: false,
            },
            OwnedTelemetry::SuiteInfo {
                description: String::new(),
                name: "s".to_string(),
                setting_count: 0,
                suite_id: 0,
                test_count: 0,
            },
            OwnedTelemetry::DiscoveryComplete,
        ]
    }

    fn next(worker: &Worker) -> Update {
        worker
            .updates
            .recv_timeout(Duration::from_secs(3))
            .unwrap_or_else(|_| Update::LinkLost("test timed out".to_string()))
    }

    #[test]
    fn rediscovery_with_fake_link() {
        let fake = ETSBridge::fake(None);
        let written = std::sync::Arc::clone(&fake.written);
        let worker = Worker::spawn(fake.bridge, quick());
        std::thread::sleep(Duration::from_millis(150));
        let attempts_before = written.lock().unwrap().len();
        assert!(attempts_before > 0, "discovery is requested repeatedly");
        for t in discovery(PROTOCOL_VERSION) {
            fake.tx.send(BridgeMessage::Telemetry(t)).unwrap();
        }
        let Update::Catalog(catalog, details) = next(&worker) else {
            panic!("expected the catalog");
        };
        assert_eq!(catalog.suites.len(), 1);
        assert_eq!(details.board_id, 1);
        assert_eq!(next(&worker), Update::Idle);
    }

    #[test]
    fn a_foreign_revision_is_reported_and_the_worker_waits() {
        let fake = ETSBridge::fake(None);
        for t in discovery(PROTOCOL_VERSION.wrapping_add(1)) {
            fake.tx.send(BridgeMessage::Telemetry(t)).unwrap();
        }
        let mut worker = Worker::spawn(fake.bridge, quick());
        assert!(matches!(next(&worker), Update::Mismatch { .. }));
        worker.shutdown();
    }

    #[test]
    fn an_exited_target_is_reported() {
        let fake = ETSBridge::fake(Some(1));
        let worker = Worker::spawn(fake.bridge, quick());
        assert!(matches!(next(&worker), Update::LinkLost(_)));
    }

    #[test]
    fn stop_queued_before_dequeue_survives_into_handle() {
        let fake = ETSBridge::fake(None);
        for t in discovery(PROTOCOL_VERSION) {
            fake.tx.send(BridgeMessage::Telemetry(t)).unwrap();
        }
        let worker = Worker::spawn(fake.bridge, quick());
        assert!(matches!(next(&worker), Update::Catalog(_, _)));
        assert_eq!(next(&worker), Update::Idle);
        // Stop wins the race against a request already on the channel (UI would
        // have cleared only when *sending* the run, not here).
        worker.control.stop();
        worker.requests.send(Request::RunAll).unwrap();
        assert_eq!(next(&worker), Update::Idle);
        assert!(
            worker.control.stop_requested(),
            "dequeue must not clear a Stop that arrived while a run was queued"
        );
    }
}
