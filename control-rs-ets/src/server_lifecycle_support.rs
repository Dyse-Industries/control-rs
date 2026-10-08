// Support for the loop-run tests: a scripted loop, a recording link and a
// profiler with a ticking clock. Included into `server::tests`.

use crate::comms::LoopRunState;
use crate::{LoopDescriptor, LoopIo, LoopOutcome};
use std::borrow::ToOwned;
use std::collections::VecDeque;
use std::string::String;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::vec::Vec;
use super::*;

static FIXTURE: Mutex<Fixture> = Mutex::new(Fixture::new());

/// The loops of an image with one loop without a timeout.
pub static LOOPS_PLAIN: &[&LoopDescriptor] = &[&LOOP_PLAIN];

/// The loops of an image with one loop with a timeout.
pub static LOOPS_TIMEOUT: &[&LoopDescriptor] = &[&LOOP_TIMEOUT];

/// Two loops for one suite.
pub static LOOPS_TWINS: &[&LoopDescriptor] = &[&LOOP_PLAIN, &LOOP_TWIN];

/// A loop without a link timeout.
pub static LOOP_PLAIN: LoopDescriptor = LoopDescriptor {
    suite: &SUITE_DESC,
    name: "scripted",
    description: "scripted loop",
    input_type: "()",
    output_type: "()",
    setup,
    step,
    reset,
    teardown,
    link_timeout_ms: 0,
};

/// The loop whose link timeout is 500 ms.
pub static LOOP_TIMEOUT: LoopDescriptor = LoopDescriptor {
    suite: &SUITE_DESC,
    name: "scripted",
    description: "scripted loop",
    input_type: "()",
    output_type: "()",
    setup,
    step,
    reset,
    teardown,
    link_timeout_ms: 500,
};

/// A second loop for the same suite, whose setup is counted apart.
pub static LOOP_TWIN: LoopDescriptor = LoopDescriptor {
    suite: &SUITE_DESC,
    name: "twin",
    description: "second loop",
    input_type: "()",
    output_type: "()",
    setup: twin_setup,
    step,
    reset,
    teardown,
    link_timeout_ms: 0,
};

/// The running state, which scripts repeat.
pub const RUN: LoopRunState = LoopRunState::Running;

/// The result of a loop's setup, reset or teardown.
pub type HookResult = Result<(), &'static str>;

/// The state of a run and its message.
pub type StateLog = (LoopRunState, Option<String>);

/// The result of a bounded server run and the server.
pub type Outcome = (ServerResult<&'static str>, LoopServer);

/// Telemetry frames as sent.
pub type Frames = Vec<Vec<u8>>;

/// An output packet and the step that produced it.
pub type Sample = (u64, Vec<u8>);

/// Picks events by a property.
pub type EventPick = fn(&Event) -> bool;

/// One scripted step: status, message and the output packet bytes.
pub type Scripted = (LoopRunState, Option<&'static str>, &'static [u8]);

/// Input bytes and index a step saw.
pub type SeenInput = (Option<Vec<u8>>, Option<u64>);

/// What the link did, in order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    Flush,
    Poll,
    Sample(u64),
    State(LoopRunState),
}

/// A link that replays commands, then idles, then fails to end the run.
pub struct LoopComms {
    commands: VecDeque<Command<'static>>,
    pub events: Vec<Event>,
    idle_polls_left: usize,
    payloads: Frames,
}

/// A profiler whose clock advances by a fixed tick on every read and which
/// counts interrupt masking.
pub struct TickProfiler {
    pub masked: AtomicUsize,
    now: AtomicU64,
    tick: u64,
}

/// A server over the loop link.
pub type LoopServer = Server<'static, LoopComms, TickProfiler>;

/// Settings of one loop-test run.
pub struct Run {
    pub commands: Vec<Command<'static>>,
    pub idle_polls: usize,
    pub tick_ns: u64,
}

/// How the scripted loop behaves.
#[derive(Clone, Copy)]
pub struct Config {
    pub reset_err: Option<&'static str>,
    pub script: &'static [Scripted],
    pub setup_err: Option<&'static str>,
    pub teardown_err: Option<&'static str>,
}

/// How often the scripted loop's functions ran.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Counts {
    pub reset: usize,
    pub reset_saw_setting: usize,
    pub setup: usize,
    pub setup_before_first_step: usize,
    pub steps: usize,
    pub teardown: usize,
    pub twin_setup: usize,
}

/// Shared state of the scripted loop.
struct Fixture {
    config: Config,
    counts: Counts,
    inputs: Vec<SeenInput>,
}

impl Run {
    pub fn new(commands: Vec<Command<'static>>) -> Self {
        Self {
            commands,
            idle_polls: 50,
            tick_ns: 1_000_000,
        }
    }
}

impl Config {
    pub const fn new(script: &'static [Scripted]) -> Self {
        Self {
            reset_err: None,
            script,
            setup_err: None,
            teardown_err: None,
        }
    }
}

impl Fixture {
    const fn new() -> Self {
        Self {
            config: Config::new(&[]),
            counts: Counts {
                reset: 0,
                reset_saw_setting: 0,
                setup: 0,
                setup_before_first_step: usize::MAX,
                steps: 0,
                teardown: 0,
                twin_setup: 0,
            },
            inputs: Vec::new(),
        }
    }
}

impl HostComms for LoopComms {
    type Error = &'static str;

    fn flush(&mut self) -> Result<(), Self::Error> {
        self.events.push(Event::Flush);
        Ok(())
    }

    fn poll_command(&mut self) -> Result<Option<Command<'static>>, Self::Error> {
        self.events.push(Event::Poll);
        if let Some(cmd) = self.commands.pop_front() {
            return Ok(Some(cmd));
        }
        self.idle_polls_left =
            self.idle_polls_left.checked_sub(1).ok_or("Exit loop")?;
        Ok(None)
    }

    fn send_telemetry(
        &mut self,
        telemetry: &Telemetry<'_>,
    ) -> Result<(), Self::Error> {
        match telemetry {
            Telemetry::LoopSample { seq, .. } => {
                self.events.push(Event::Sample(*seq));
            }
            Telemetry::LoopState { state, .. } => {
                self.events.push(Event::State(*state));
            }
            _ => {}
        }
        self.payloads
            .push(postcard::to_allocvec(telemetry).map_err(|_| "serialize")?);
        Ok(())
    }
}

impl CPUProfiler for TickProfiler {
    fn disable_interrupts<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        self.masked.fetch_add(1, Ordering::SeqCst);
        f()
    }

    fn get_cycles(&self) -> u64 {
        0
    }

    fn get_nanos(&self) -> u64 {
        self.now.fetch_add(self.tick, Ordering::SeqCst)
    }

    fn get_sp(&self) -> usize {
        0
    }

    fn get_stack_end(&self) -> usize {
        0
    }
}

fn fixture() -> MutexGuard<'static, Fixture> {
    FIXTURE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn setup() -> HookResult {
    let err = {
        let mut fx = fixture();
        fx.counts.setup = fx.counts.setup.saturating_add(1);
        fx.config.setup_err
    };
    err.map_or(Ok(()), Err)
}

fn twin_setup() -> HookResult {
    let err = {
        let mut fx = fixture();
        fx.counts.twin_setup = fx.counts.twin_setup.saturating_add(1);
        fx.config.setup_err
    };
    err.map_or(Ok(()), Err)
}

fn step(io: &mut LoopIo<'_>) -> LoopOutcome {
    let (status, message, bytes) = {
        let mut fx = fixture();
        if fx.counts.steps == 0 {
            fx.counts.setup_before_first_step = fx.counts.setup;
        }
        fx.counts.steps = fx.counts.steps.saturating_add(1);
        fx.inputs.push((io.input.map(<[u8]>::to_vec), io.input_seq));
        let last = fx.config.script.len().saturating_sub(1);
        let index = usize::try_from(io.step).unwrap_or(usize::MAX).min(last);
        fx.config.script.get(index).copied().unwrap_or((
            LoopRunState::Error,
            Some("empty script"),
            &[],
        ))
    };
    if let Some(dst) = io.output.get_mut(..bytes.len()) {
        dst.copy_from_slice(bytes);
    }
    io.output_len = bytes.len();
    LoopOutcome { message, status }
}

fn reset() -> HookResult {
    let err = {
        let mut fx = fixture();
        fx.counts.reset = fx.counts.reset.saturating_add(1);
        if let SettingValue::U8(v) = TEST_U8_SETTING.get() {
            fx.counts.reset_saw_setting = usize::from(v);
        }
        fx.config.reset_err
    };
    err.map_or(Ok(()), Err)
}

fn teardown() -> HookResult {
    let err = {
        let mut fx = fixture();
        fx.counts.teardown = fx.counts.teardown.saturating_add(1);
        fx.config.teardown_err
    };
    err.map_or(Ok(()), Err)
}

/// Resets the scripted loop to `config` and holds the run-state lock.
pub fn begin(config: Config) -> MutexGuard<'static, ()> {
    let guard = crate::server::test_lock::hold();
    let mut fx = fixture();
    *fx = Fixture {
        config,
        ..Fixture::new()
    };
    guard
}

/// How often the scripted loop's functions ran.
pub fn counts() -> Counts {
    fixture().counts
}

/// The inputs the scripted step saw.
pub fn seen_inputs() -> Vec<SeenInput> {
    fixture().inputs.clone()
}

/// Runs a server over `loops` until the link ends, returning the result and
/// the server.
pub fn run_loops(
    loops: &'static [&'static LoopDescriptor],
    run: Run,
) -> Outcome {
    let comms = LoopComms {
        commands: run.commands.into(),
        events: Vec::new(),
        idle_polls_left: run.idle_polls,
        payloads: Vec::new(),
    };
    let profiler = TickProfiler {
        masked: AtomicUsize::new(0),
        now: AtomicU64::new(0),
        tick: run.tick_ns,
    };
    let mut server =
        Server::new(Context::new(comms, profiler), SUITES).with_loops(loops);
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let result = server.run();
        let _ = tx.send((result, server));
    });
    rx.recv_timeout(std::time::Duration::from_secs(5))
        .expect("the server loop did not terminate")
}

/// Decodes every telemetry frame the server sent.
pub fn sent(server: &LoopServer) -> Vec<Telemetry<'_>> {
    server
        .context
        .comms
        .payloads
        .iter()
        .map(|p| postcard::from_bytes(p).expect("telemetry decodes"))
        .collect()
}

pub fn start(max_steps: u64, lockstep: bool) -> Command<'static> {
    Command::StartLoop {
        suite_id: 0,
        test_id: 1,
        max_steps,
        lockstep,
    }
}

pub fn input(seq: u64, payload: &'static [u8]) -> Command<'static> {
    Command::LoopInput {
        suite_id: 0,
        test_id: 1,
        seq,
        payload,
    }
}

pub fn stop() -> Command<'static> {
    Command::StopNow {
        suite_id: 0,
        test_id: 1,
    }
}

pub fn set_u8(v: u8) -> Command<'static> {
    Command::SetSetting {
        suite_id: 0,
        setting_id: 0,
        value: SettingValue::U8(v),
    }
}

/// `LoopState` frames in order, with owned messages.
pub fn states(server: &LoopServer) -> Vec<StateLog> {
    sent(server)
        .into_iter()
        .filter_map(|t| match t {
            Telemetry::LoopState { state, message, .. } => {
                Some((state, message.map(str::to_owned)))
            }
            _ => None,
        })
        .collect()
}

/// The final `LoopState` of the run.
pub fn final_state(server: &LoopServer) -> Option<StateLog> {
    states(server).pop()
}

/// Number of `Log` frames.
pub fn log_count(server: &LoopServer) -> usize {
    sent(server)
        .iter()
        .filter(|t| matches!(t, Telemetry::Log(_)))
        .count()
}

/// Number of frames for which `pick` holds.
pub fn count_frames(
    server: &LoopServer,
    pick: impl Fn(&Telemetry<'_>) -> bool,
) -> usize {
    sent(server).iter().filter(|t| pick(t)).count()
}

/// Output packets the server sent, in order.
pub fn samples(server: &LoopServer) -> Vec<Sample> {
    sent(server)
        .into_iter()
        .filter_map(|t| match t {
            Telemetry::LoopSample { seq, payload, .. } => {
                Some((seq, payload.to_vec()))
            }
            _ => None,
        })
        .collect()
}

/// Number of events for which `pick` holds.
pub fn count_events(events: &[Event], pick: EventPick) -> usize {
    events.iter().filter(|e| pick(e)).count()
}
