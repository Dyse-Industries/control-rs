//! Runner and execution modes.
//!
//! # Description
//!
//! A [`Runner`] executes one run of one executable and owns the only execution loop. A [`Mode`]
//! supplies the per-mode decisions: whether the step runs masked and profiled, whether it waits for
//! input and how releases are paced. The server maps the commanded mode to a mode type once per
//! run.
//!
//! # Core Concepts
//!
//! - **Boundary**: Between steps the runner receives at most one command, checks the host link
//!   deadline and feeds the board. The first verdict wins.
//! - **Modes**: `Atomic` runs one masked, profiled step. `Lockstep` runs step `k` only with input
//!   `k`. `FreeRunning` never waits. `Periodic` and `Triggered` run one step per release.
//! - **Verdicts**: `Pass`, `Fail` and `Error` come from the executable, `Aborted` from a stop,
//!   `TimedOut` from the link deadline, `Bounded` from the step bound and `Overrun` from a missed
//!   release.
//! - **Settle**: Teardown always runs after setup, then the run reports its verdict and statistics.
//!
//! # Usage
//!
//! ```
//! use control_rs_ets::run::{Atomic, Mode, Pacing};
//!
//! assert!(Atomic::ONE_STEP);
//! assert_eq!(Atomic::PACING, Pacing::None);
//! ```
//!
//! # Limitations
//!
//! - **Cooperative**: A step that does not return blocks stop, heartbeat and reset; a board
//!   watchdog recovers the target.

use core::marker::PhantomData;

use crate::MAX_PACKET_SIZE;
use crate::board::Board;
use crate::exec::{ExecDescriptor, Message, TaskIo, TaskOutcome};
use crate::fault::{self, Phase};
use crate::link::{Link, UnitOf};
use crate::params::ParamTable;
use crate::profiler::CPUProfiler;
use crate::server::{CURRENT_SUITE, CURRENT_TEST, Context};
use crate::wire::{
    Command, ExecMode, ItemState, MAX_MESSAGE_SIZE, Telemetry, Verdict,
};

/// Message of the `Error` verdict returned when an input index skips ahead.
pub const INPUT_GAP_ERROR: &str = "input sequence gap";

/// Nanoseconds per millisecond.
const NS_PER_MS: u64 = 1_000_000;

/// How a mode releases steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pacing {
    /// Steps are not paced.
    None,
    /// A step is released every period from the run start.
    Periodic,
    /// A step is released per event of the board release source.
    Triggered,
}

/// The per-mode decisions of a run.
pub trait Mode {
    /// The commanded mode this type runs.
    const MODE: ExecMode;
    /// The run is exactly one step, masked and profiled.
    const ONE_STEP: bool = false;
    /// Step `k` waits for input `k`.
    const NEEDS_INPUT: bool = false;
    /// How steps are released.
    const PACING: Pacing = Pacing::None;
}

/// One masked, profiled step.
#[derive(Debug, Clone, Copy, Default)]
pub struct Atomic;

/// Step `k` runs only after input `k`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Lockstep;

/// Steps run back to back; the newest input wins.
#[derive(Debug, Clone, Copy, Default)]
pub struct FreeRunning;

/// One step per period from the run start.
#[derive(Debug, Clone, Copy, Default)]
pub struct Periodic;

/// One step per board release event.
#[derive(Debug, Clone, Copy, Default)]
pub struct Triggered;

/// The server's link, profiler and lock, borrowed for one run.
pub type Ctx<'a, L, P> = &'a mut Context<L, P>;

/// The server's receive buffer, borrowed for one run.
pub type Rx<'a> = &'a mut [u8; MAX_MESSAGE_SIZE];

/// The verdict of a run and its message, once set.
type Ended = Option<(Verdict, Message)>;

/// Which executable to run and how.
#[derive(Clone, Copy)]
pub struct Job {
    /// The executable.
    pub exec: &'static ExecDescriptor,
    /// Index of the executable in its suite.
    pub item: u16,
    /// Step bound (`0` is unbounded).
    pub max_steps: u64,
    /// The parameter table of the suite.
    pub params: ParamTable,
    /// Release period in nanoseconds for a periodic run.
    pub period_ns: u32,
    /// Suite that owns the executable.
    pub suite: u16,
}

/// Executes one run in mode `M`.
pub struct Runner<'a, L, P, B, M> {
    board: &'a mut B,
    context: Ctx<'a, L, P>,
    job: Job,
    mode: PhantomData<M>,
    rx: Rx<'a>,
}

/// The mutable state of one run.
struct State {
    cycles: u64,
    deadline_ns: u64,
    /// Release schedule origin: set when the paced run arms (after setup).
    epoch_ns: u64,
    input: [u8; MAX_PACKET_SIZE],
    input_len: usize,
    input_seq: Option<u64>,
    jitter_max_ns: u32,
    k: u64,
    missed: u32,
    pending: u32,
    stack_peak: u32,
    started_ns: u64,
    step_max_ns: u32,
    step_min_ns: u32,
    verdict: Ended,
}

/// Whether the next step may run.
enum Ready {
    Overrun(u32),
    Step,
    Wait,
}

impl Mode for Atomic {
    const MODE: ExecMode = ExecMode::Atomic;
    const ONE_STEP: bool = true;
}

impl Mode for Lockstep {
    const MODE: ExecMode = ExecMode::Lockstep;
    const NEEDS_INPUT: bool = true;
}

impl Mode for FreeRunning {
    const MODE: ExecMode = ExecMode::FreeRunning;
}

impl Mode for Periodic {
    const MODE: ExecMode = ExecMode::Periodic;
    const PACING: Pacing = Pacing::Periodic;
}

impl Mode for Triggered {
    const MODE: ExecMode = ExecMode::Triggered;
    const PACING: Pacing = Pacing::Triggered;
}

impl State {
    const fn new() -> Self {
        Self {
            cycles: 0,
            deadline_ns: 0,
            epoch_ns: 0,
            input: [0; MAX_PACKET_SIZE],
            input_len: 0,
            input_seq: None,
            jitter_max_ns: 0,
            k: 0,
            missed: 0,
            pending: 0,
            stack_peak: 0,
            started_ns: 0,
            step_max_ns: 0,
            step_min_ns: u32::MAX,
            verdict: None,
        }
    }

    /// Records a verdict unless one is already set: the first verdict wins.
    const fn end(&mut self, verdict: Verdict, message: Message) {
        if self.verdict.is_none() {
            self.verdict = Some((verdict, message));
        }
    }

    /// Stores an input packet for step `seq`.
    fn store(&mut self, seq: u64, payload: &[u8], lockstep: bool) {
        let Some(dst) = self.input.get_mut(..payload.len()) else {
            return;
        };
        if lockstep {
            if seq > self.k {
                self.end(Verdict::Error, Some(INPUT_GAP_ERROR));
                return;
            }
            if seq < self.k {
                return;
            }
        } else if self.input_seq.is_some_and(|held| seq < held) {
            return;
        }
        dst.copy_from_slice(payload);
        self.input_len = payload.len();
        self.input_seq = Some(seq);
    }
}

impl<'a, L, P, B, M> Runner<'a, L, P, B, M> {
    /// Creates a runner for `job`.
    #[must_use]
    pub const fn new(
        board: &'a mut B,
        context: Ctx<'a, L, P>,
        rx: Rx<'a>,
        job: Job,
    ) -> Self {
        Self {
            board,
            context,
            job,
            mode: PhantomData,
            rx,
        }
    }
}

impl<L, P, B, M> Runner<'_, L, P, B, M>
where
    L: Link,
    P: CPUProfiler,
    B: Board,
    M: Mode,
{
    /// Executes the run: arms the fault path, runs setup, drives steps to a verdict, tears down
    /// and reports.
    ///
    /// # Errors
    /// Returns a transport error if the link fails; teardown and the fault-path cleanup still run.
    pub fn run(mut self) -> UnitOf<L> {
        let Job {
            exec, item, suite, ..
        } = self.job;
        fault::arm(exec);
        CURRENT_SUITE.set_active(usize::from(suite));
        CURRENT_TEST.set_active(usize::from(item));
        let mut state = State::new();
        let driven = self.drive(&mut state);
        let settled = self.settle(&state);
        fault::clear();
        CURRENT_SUITE.set_idle();
        CURRENT_TEST.set_idle();
        driven.and(settled)
    }

    fn now(&self) -> u64 {
        self.context.cpu_utils.get_nanos()
    }

    fn send(&mut self, telemetry: &Telemetry<'_>) -> UnitOf<L> {
        self.context.send_telemetry_locked(telemetry).map(|_| ())
    }

    fn drive(&mut self, state: &mut State) -> UnitOf<L> {
        let Job {
            exec,
            item,
            suite,
            period_ns,
            ..
        } = self.job;
        self.send(&Telemetry::ItemState {
            suite,
            item,
            state: ItemState::Running,
            message: None,
        })?;
        // Wall-clock origin for RunStats; release epoch and link deadline open after setup
        // (FR-5 scopes the heartbeat to waiting for input/release; FR-8 / the state diagram
        // arm the release source only after setup succeeds).
        state.started_ns = self.now();
        if let Err(message) = (exec.setup)() {
            state.end(Verdict::Error, Some(message));
            return Ok(());
        }
        if M::PACING != Pacing::None {
            let period = (M::PACING == Pacing::Periodic).then_some(period_ns);
            self.board.arm(period);
        }
        state.epoch_ns = self.now();
        state.deadline_ns = Self::deadline_from(state.epoch_ns, exec);
        while state.verdict.is_none() {
            // A masked single step has no boundary: nothing can interrupt it, and a link that
            // can only block would stall it.
            if !M::ONE_STEP {
                self.boundary(state)?;
            }
            if state.verdict.is_some() {
                break;
            }
            self.board.feed();
            match self.ready(state) {
                Ready::Step => self.step(state)?,
                Ready::Wait => self.board.wait(),
                Ready::Overrun(missed) => {
                    state.missed = missed;
                    state.end(Verdict::Overrun, None);
                }
            }
        }
        Ok(())
    }

    fn deadline_from(now: u64, exec: &ExecDescriptor) -> u64 {
        now.saturating_add(
            u64::from(exec.link_timeout_ms).saturating_mul(NS_PER_MS),
        )
    }

    /// Receives one command, applies it and checks the link deadline.
    fn boundary(&mut self, state: &mut State) -> UnitOf<L> {
        let exec = self.job.exec;
        let received = self.context.recv_locked(self.rx)?;
        let msg = received.and_then(|len| self.rx.get(..len));
        if let Some(Ok(command)) = msg.map(postcard::from_bytes::<Command<'_>>)
        {
            state.deadline_ns =
                Self::deadline_from(self.context.cpu_utils.get_nanos(), exec);
            Self::apply(&mut *self.context, self.job, command, state)?;
        }
        if exec.link_timeout_ms != 0 && self.now() > state.deadline_ns {
            state.end(Verdict::TimedOut, None);
        }
        Ok(())
    }

    /// Applies one command received while a run is active.
    fn apply(
        context: Ctx<'_, L, P>,
        job: Job,
        command: Command<'_>,
        state: &mut State,
    ) -> UnitOf<L> {
        let Job {
            exec,
            item,
            params,
            suite,
            ..
        } = job;
        match command {
            Command::StopNow { suite: s, item: i }
                if (s, i) == (suite, item) =>
            {
                state.end(Verdict::Aborted, None);
            }
            Command::TaskInput {
                suite: s,
                item: i,
                seq,
                payload,
            } if (s, i) == (suite, item) => {
                state.store(seq, payload, M::NEEDS_INPUT);
            }
            Command::Param {
                suite: s,
                key,
                value,
            } if s == suite => {
                let stored =
                    context.param_request((suite, params), key, value)?;
                if stored && let Err(message) = (exec.reset)() {
                    state.end(Verdict::Error, Some(message));
                }
            }
            Command::Heartbeat | Command::TryReset => {}
            _ => context
                .report_error_log("Error: command rejected during a run")?,
        }
        Ok(())
    }

    /// Decides whether the next step may run, and takes releases for paced modes.
    fn ready(&mut self, state: &mut State) -> Ready {
        if M::NEEDS_INPUT {
            return if state.input_seq == Some(state.k) {
                Ready::Step
            } else {
                Ready::Wait
            };
        }
        if M::PACING == Pacing::None {
            return Ready::Step;
        }
        let new = self.releases(state);
        state.pending = state.pending.saturating_add(new);
        match state.pending {
            0 => Ready::Wait,
            1 => {
                state.pending = 0;
                Ready::Step
            }
            more => Ready::Overrun(more.saturating_sub(1)),
        }
    }

    /// Releases since the previous call: the board's count, or the profiler clock for `Periodic`.
    fn releases(&mut self, state: &State) -> u32 {
        if self.board.has_release_source() {
            return self.board.releases();
        }
        if M::PACING != Pacing::Periodic || self.job.period_ns == 0 {
            return 0;
        }
        let elapsed = self.now().saturating_sub(state.epoch_ns);
        let due = elapsed
            .checked_div(u64::from(self.job.period_ns))
            .unwrap_or(0)
            .saturating_add(1);
        let released = state.k.saturating_add(u64::from(state.pending));
        u32::try_from(due.saturating_sub(released)).unwrap_or(u32::MAX)
    }

    /// Runs one step and reports its output.
    fn step(&mut self, state: &mut State) -> UnitOf<L> {
        let Job {
            exec,
            item,
            suite,
            period_ns,
            ..
        } = self.job;
        let mut output = [0u8; MAX_PACKET_SIZE];
        let held = state
            .input_seq
            .and_then(|_| state.input.get(..state.input_len));
        let mut io = TaskIo {
            input: held,
            input_seq: state.input_seq,
            output: &mut output,
            output_len: 0,
            step: state.k,
        };
        let started = self.now();
        let (outcome, took_ns) = if M::ONE_STEP {
            let mut outcome = TaskOutcome::Continue;
            let metrics =
                self.context.profile_test(|| outcome = (exec.step)(&mut io));
            state.cycles = metrics.cycles;
            state.stack_peak = metrics.stack_peak;
            (outcome, metrics.time_us.saturating_mul(1000))
        } else {
            let outcome = (exec.step)(&mut io);
            (outcome, self.now().saturating_sub(started))
        };
        let output_len = io.output_len;
        let took = u32::try_from(took_ns).unwrap_or(u32::MAX);
        state.step_min_ns = state.step_min_ns.min(took);
        state.step_max_ns = state.step_max_ns.max(took);
        if M::PACING == Pacing::Periodic && period_ns != 0 {
            let release = state
                .epoch_ns
                .saturating_add(state.k.saturating_mul(u64::from(period_ns)));
            let late = u32::try_from(started.saturating_sub(release))
                .unwrap_or(u32::MAX);
            state.jitter_max_ns = state.jitter_max_ns.max(late);
        }
        if output_len > 0 || M::NEEDS_INPUT {
            let payload = output.get(..output_len).unwrap_or(&[]);
            self.send(&Telemetry::TaskSample {
                suite,
                item,
                seq: state.k,
                payload,
            })?;
        }
        self.finish_step(state, outcome);
        Ok(())
    }

    /// Records the outcome of a step and applies the step bound.
    const fn finish_step(&self, state: &mut State, outcome: TaskOutcome) {
        if let TaskOutcome::Finish(verdict, message) = outcome {
            state.end(verdict, message);
        }
        if M::NEEDS_INPUT {
            state.input_seq = None;
        }
        state.k = state.k.saturating_add(1);
        if M::ONE_STEP {
            state.end(Verdict::Pass, None);
        } else if self.job.max_steps != 0 && state.k >= self.job.max_steps {
            state.end(Verdict::Bounded, None);
        }
    }

    /// Tears the run down and reports its verdict and statistics.
    fn settle(&mut self, state: &State) -> UnitOf<L> {
        let Job {
            exec, item, suite, ..
        } = self.job;
        if M::PACING != Pacing::None {
            self.board.disarm();
        }
        fault::set_phase(Phase::TearingDown);
        let teardown = (exec.teardown)();
        fault::set_phase(Phase::Done);
        let (verdict, message) = state
            .verdict
            .unwrap_or((Verdict::Error, Some("run ended without a verdict")));
        let ended = self.now();
        let report = if M::ONE_STEP {
            Ok(())
        } else {
            self.send(&Telemetry::TeardownReport {
                suite,
                item,
                ok: teardown.is_ok(),
                message: teardown.err(),
            })
        };
        let done = self.send(&Telemetry::ItemState {
            suite,
            item,
            state: ItemState::Done(verdict),
            message,
        });
        let usage = self.send(&Telemetry::RunStats {
            suite,
            item,
            steps: state.k,
            time_ns: if M::ONE_STEP {
                u64::from(state.step_max_ns)
            } else {
                ended.saturating_sub(state.started_ns)
            },
            step_min_ns: if state.k == 0 { 0 } else { state.step_min_ns },
            step_max_ns: state.step_max_ns,
            cycles: state.cycles,
            stack_peak: state.stack_peak,
            jitter_max_ns: state.jitter_max_ns,
            missed_releases: state.missed,
        });
        report.and(done).and(usage)
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use crate::SuiteDescriptor;
    use crate::exec::{HookOf, StepFn};
    use crate::link::{RecvOf, SendOf};
    use crate::params::ParamEntry;
    use crate::wire::{Param, ParamKey};
    use core::cell::Cell;
    use std::collections::VecDeque;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::vec::Vec;

    static BROKEN: ExecDescriptor = {
        let mut d = task(echo, 0);
        d.setup = bad_setup;
        d
    };
    static ECHO: ExecDescriptor = task(echo, 0);
    static GAIN: ParamEntry =
        ParamEntry::u8("gain", "", 1).bounded(Param::U8(0), Param::U8(9));
    static PARAMS: ParamTable = &[&GAIN];
    static RESETS: AtomicUsize = AtomicUsize::new(0);
    static SERIAL: Mutex<()> = Mutex::new(());
    static SETUPS: AtomicUsize = AtomicUsize::new(0);
    /// Nanoseconds queued by a slow setup; consumed on the next profiler clock read.
    static SETUP_ADVANCE_NS: core::sync::atomic::AtomicU64 =
        core::sync::atomic::AtomicU64::new(0);
    static SLOW: ExecDescriptor = {
        let mut d = task(echo, 1);
        d.setup = slow_setup;
        d
    };
    const STOP: Command<'static> = Command::StopNow { suite: 0, item: 0 };
    static TEARDOWNS: AtomicUsize = AtomicUsize::new(0);
    static TIMED: ExecDescriptor = task(echo, 1);

    /// Messages a link sent.
    type Sent = Vec<Vec<u8>>;

    /// A verdict count: steps and missed releases.
    type Counts = Option<(u64, u32)>;

    /// Sample sequence numbers with payloads.
    type Samples = Vec<(u64, Vec<u8>)>;

    /// The periods a board was armed with.
    type Arms = Vec<Option<u32>>;

    /// Shortest step, longest step and largest jitter in nanoseconds.
    type Timing = Option<(u32, u32, u32)>;

    /// What one run produced.
    struct Outcome {
        armed: Arms,
        disarms: usize,
        feeds: usize,
        result: HookOf,
        telemetry: Sent,
        waits: usize,
    }

    /// A link that replays commands and ends the loop after its idle budget.
    struct Script {
        commands: VecDeque<Command<'static>>,
        idle: usize,
        sent: Sent,
    }

    /// A board with scripted release counts.
    struct Timer {
        armed: Arms,
        disarms: usize,
        feeds: usize,
        releases: VecDeque<u32>,
        source: bool,
        waits: usize,
    }

    /// A profiler whose clock advances by a fixed tick per read.
    struct Clock {
        now: Cell<u64>,
        tick: u64,
    }

    impl Link for Script {
        type Error = &'static str;

        fn recv(&mut self, buf: &mut [u8]) -> RecvOf<Self> {
            if let Some(cmd) = self.commands.pop_front() {
                let len =
                    postcard::to_slice(&cmd, buf).map_err(|_| "encode")?.len();
                return Ok(Some(len));
            }
            self.idle = self.idle.checked_sub(1).ok_or("end")?;
            Ok(None)
        }

        fn send(&mut self, msg: &[u8]) -> SendOf<Self> {
            self.sent.push(msg.to_vec());
            Ok(true)
        }
    }

    impl Board for Timer {
        fn feed(&mut self) {
            self.feeds = self.feeds.saturating_add(1);
        }

        fn wait(&mut self) {
            self.waits = self.waits.saturating_add(1);
            assert!(self.waits < 100_000, "wait loop does not terminate");
        }

        fn arm(&mut self, period_ns: Option<u32>) {
            self.armed.push(period_ns);
        }

        fn disarm(&mut self) {
            self.disarms = self.disarms.saturating_add(1);
        }

        fn has_release_source(&self) -> bool {
            self.source
        }

        fn releases(&mut self) -> u32 {
            self.releases.pop_front().unwrap_or(0)
        }
    }

    impl CPUProfiler for Clock {
        fn get_cycles(&self) -> u64 {
            0
        }

        fn get_nanos(&self) -> u64 {
            let queued = SETUP_ADVANCE_NS.swap(0, Ordering::SeqCst);
            self.now.set(
                self.now
                    .get()
                    .saturating_add(self.tick)
                    .saturating_add(queued),
            );
            self.now.get()
        }

        fn get_sp(&self) -> usize {
            0
        }

        fn get_stack_end(&self) -> usize {
            0
        }
    }

    fn counted(counter: &AtomicUsize) -> HookOf {
        counter
            .fetch_add(1, Ordering::SeqCst)
            .checked_add(1)
            .map(drop)
            .ok_or("count overflow")
    }

    fn setup() -> HookOf {
        counted(&SETUPS)
    }

    fn bad_setup() -> HookOf {
        counted(&SETUPS).and(Err("no sensor"))
    }

    /// Setup that burns more than one millisecond of profiler time (and more than
    /// one 10 µs period when the clock tick is 1 µs).
    fn slow_setup() -> HookOf {
        SETUP_ADVANCE_NS.store(2_000_000, Ordering::SeqCst);
        counted(&SETUPS)
    }

    fn reset() -> HookOf {
        counted(&RESETS)
    }

    fn teardown() -> HookOf {
        counted(&TEARDOWNS)
    }

    /// Copies the held input to the output; ends with `Pass` at step 5.
    fn echo(io: &mut TaskIo<'_>) -> TaskOutcome {
        if let (Some(src), Some(dst)) = (
            io.input,
            io.output.get_mut(..io.input.map_or(0, <[u8]>::len)),
        ) {
            dst.copy_from_slice(src);
            io.output_len = src.len();
        }
        if io.step == 5 {
            TaskOutcome::Finish(Verdict::Pass, Some("five"))
        } else {
            TaskOutcome::Continue
        }
    }

    const fn task(step: StepFn, timeout_ms: u32) -> ExecDescriptor {
        let mut d = ExecDescriptor::case("t", "", step);
        d.modes = 0b1_1111;
        d.setup = setup;
        d.reset = reset;
        d.teardown = teardown;
        d.link_timeout_ms = timeout_ms;
        d
    }

    fn drive<M: Mode>(
        exec: &'static ExecDescriptor,
        commands: Vec<Command<'static>>,
        releases: &[u32],
        job: (u64, u32),
    ) -> Outcome {
        let mut context = Context::new(
            Script {
                commands: commands.into(),
                idle: 5000,
                sent: Vec::new(),
            },
            Clock {
                now: Cell::new(0),
                tick: 1_000,
            },
        );
        let mut board = Timer {
            armed: Vec::new(),
            disarms: 0,
            feeds: 0,
            releases: releases.iter().copied().collect(),
            source: !releases.is_empty(),
            waits: 0,
        };
        let mut rx = [0u8; MAX_MESSAGE_SIZE];
        let job = Job {
            exec,
            item: 0,
            max_steps: job.0,
            params: PARAMS,
            period_ns: job.1,
            suite: 0,
        };
        let result =
            Runner::<_, _, _, M>::new(&mut board, &mut context, &mut rx, job)
                .run();
        Outcome {
            armed: board.armed,
            disarms: board.disarms,
            feeds: board.feeds,
            result,
            telemetry: context.link.sent,
            waits: board.waits,
        }
    }

    fn frames(o: &Outcome) -> Vec<Telemetry<'_>> {
        o.telemetry
            .iter()
            .map(|p| postcard::from_bytes(p).unwrap())
            .collect()
    }

    fn verdict(o: &Outcome) -> Ended {
        frames(o).iter().find_map(|t| match *t {
            Telemetry::ItemState {
                state: ItemState::Done(v),
                message,
                ..
            } => Some((
                v,
                message.map(|m| if m == "five" { "five" } else { "" }),
            )),
            _ => None,
        })
    }

    fn stats(o: &Outcome) -> Counts {
        frames(o).iter().find_map(|t| match *t {
            Telemetry::RunStats {
                steps,
                missed_releases,
                ..
            } => Some((steps, missed_releases)),
            _ => None,
        })
    }

    fn serial() -> std::sync::MutexGuard<'static, ()> {
        SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// The message of the `Done` frame, or `None` when it carries none.
    fn done_message(o: &Outcome) -> Option<std::string::String> {
        frames(o).iter().find_map(|t| match *t {
            Telemetry::ItemState {
                state: ItemState::Done(_),
                message,
                ..
            } => message.map(std::string::String::from),
            _ => None,
        })
    }

    /// Shortest step, longest step and largest jitter of the `RunStats` frame.
    fn timing(o: &Outcome) -> Timing {
        frames(o).iter().find_map(|t| match *t {
            Telemetry::RunStats {
                step_min_ns,
                step_max_ns,
                jitter_max_ns,
                ..
            } => Some((step_min_ns, step_max_ns, jitter_max_ns)),
            _ => None,
        })
    }

    fn samples(o: &Outcome) -> Samples {
        frames(o)
            .iter()
            .filter_map(|t| match *t {
                Telemetry::TaskSample { seq, payload, .. } => {
                    Some((seq, payload.to_vec()))
                }
                _ => None,
            })
            .collect()
    }

    fn input(seq: u64, byte: &'static [u8]) -> Command<'static> {
        Command::TaskInput {
            suite: 0,
            item: 0,
            seq,
            payload: byte,
        }
    }

    fn reset_counters() {
        SETUPS.store(0, Ordering::SeqCst);
        RESETS.store(0, Ordering::SeqCst);
        TEARDOWNS.store(0, Ordering::SeqCst);
        SETUP_ADVANCE_NS.store(0, Ordering::SeqCst);
    }

    #[test]
    fn feed_at_boundaries_and_wait_only_when_nothing_is_runnable() {
        let _g = SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let busy = drive::<FreeRunning>(&ECHO, Vec::new(), &[], (3, 0));
        assert_eq!(busy.feeds, 3);
        assert_eq!(busy.waits, 0);
        let idle = drive::<Lockstep>(&TIMED, Vec::new(), &[], (0, 0));
        assert!(idle.waits > 0);
        assert_eq!(idle.feeds, idle.waits);
    }

    #[test]
    fn atomic_runs_one_step_masked() {
        let _g = SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset_counters();
        let o = drive::<Atomic>(&ECHO, Vec::new(), &[], (0, 0));
        assert_eq!(o.result, Ok(()));
        assert_eq!(verdict(&o), Some((Verdict::Pass, None)));
        assert_eq!(stats(&o), Some((1, 0)));
        assert_eq!(SETUPS.load(Ordering::SeqCst), 1);
        assert_eq!(TEARDOWNS.load(Ordering::SeqCst), 1);
        assert!(
            !frames(&o)
                .iter()
                .any(|t| matches!(t, Telemetry::TeardownReport { .. })),
            "a case reports no teardown"
        );
    }

    #[test]
    fn lockstep_waits_for_input_k() {
        let _g = SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset_counters();
        let o = drive::<Lockstep>(
            &ECHO,
            std::vec![input(0, &[7]), input(1, &[8]), STOP],
            &[],
            (0, 0),
        );
        assert_eq!(o.result, Ok(()));
        let samples: Samples = frames(&o)
            .iter()
            .filter_map(|t| match *t {
                Telemetry::TaskSample { seq, payload, .. } => {
                    Some((seq, payload.to_vec()))
                }
                _ => None,
            })
            .collect();
        assert_eq!(samples, std::vec![(0, std::vec![7]), (1, std::vec![8])]);
        assert_eq!(verdict(&o), Some((Verdict::Aborted, None)));
        assert_eq!(TEARDOWNS.load(Ordering::SeqCst), 1);
        assert!(
            frames(&o).iter().any(|t| matches!(
                t,
                Telemetry::TeardownReport { ok: true, .. }
            ))
        );
    }

    #[test]
    fn a_gap_in_lockstep_input_is_an_error() {
        let _g = SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let o =
            drive::<Lockstep>(&ECHO, std::vec![input(2, &[1])], &[], (0, 0));
        assert_eq!(verdict(&o), Some((Verdict::Error, Some(""))));
        assert_eq!(stats(&o), Some((0, 0)));
    }

    #[test]
    fn stop_and_timeout_while_waiting() {
        let _g = SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let stopped = drive::<Lockstep>(&ECHO, std::vec![STOP], &[], (0, 0));
        assert_eq!(verdict(&stopped), Some((Verdict::Aborted, None)));
        // The link deadline ends a run that waits for input that never comes.
        let timed = drive::<Lockstep>(&TIMED, Vec::new(), &[], (0, 0));
        assert_eq!(timed.result, Ok(()));
        assert_eq!(verdict(&timed), Some((Verdict::TimedOut, None)));
        assert!(timed.waits > 0);
    }

    #[test]
    fn free_running_is_bounded_and_the_first_verdict_wins() {
        let _g = SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let o = drive::<FreeRunning>(&ECHO, Vec::new(), &[], (3, 0));
        assert_eq!(verdict(&o), Some((Verdict::Bounded, None)));
        assert_eq!(stats(&o), Some((3, 0)));
        // The verdict of the executable at step 5 precedes the bound at step 9.
        let finished = drive::<FreeRunning>(&ECHO, Vec::new(), &[], (9, 0));
        assert_eq!(verdict(&finished), Some((Verdict::Pass, Some("five"))));
        assert_eq!(stats(&finished), Some((6, 0)));
    }

    #[test]
    fn triggered_runs_one_step_per_release() {
        let _g = SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let o = drive::<Triggered>(&ECHO, Vec::new(), &[0, 1, 0, 1], (2, 0));
        assert_eq!(verdict(&o), Some((Verdict::Bounded, None)));
        assert_eq!(stats(&o), Some((2, 0)));
        assert!(o.waits >= 2, "waits when no release is pending");
    }

    #[test]
    fn release_count_above_one_is_overrun() {
        let _g = SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let o = drive::<Triggered>(&ECHO, Vec::new(), &[1, 3], (0, 0));
        assert_eq!(verdict(&o), Some((Verdict::Overrun, None)));
        assert_eq!(stats(&o), Some((1, 2)));
    }

    #[test]
    fn periodic_release_is_absolute() {
        let _g = SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // No board source: releases come from the clock, which ticks 1000 nanoseconds per read.
        let o = drive::<Periodic>(&ECHO, Vec::new(), &[], (3, 10_000));
        assert_eq!(o.result, Ok(()));
        assert_eq!(verdict(&o), Some((Verdict::Bounded, None)));
        assert_eq!(stats(&o), Some((3, 0)));
    }

    #[test]
    fn a_failed_setup_ends_the_run_with_an_error_and_still_tears_down() {
        let _g = SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset_counters();
        let o = drive::<FreeRunning>(&BROKEN, Vec::new(), &[], (3, 0));
        assert_eq!(verdict(&o), Some((Verdict::Error, Some(""))));
        assert_eq!(stats(&o), Some((0, 0)));
        assert_eq!(TEARDOWNS.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn a_parameter_write_during_a_run_is_followed_by_one_reset() {
        let _g = SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset_counters();
        let write = |v| Command::Param {
            suite: 0,
            key: ParamKey::Name("gain"),
            value: Some(Param::U8(v)),
        };
        let o = drive::<FreeRunning>(
            &ECHO,
            std::vec![write(4), write(99)],
            &[],
            (4, 0),
        );
        assert_eq!(
            RESETS.load(Ordering::SeqCst),
            1,
            "the refused write does not reset"
        );
        assert_eq!(GAIN.get(), Param::U8(4));
        let echoes = frames(&o)
            .iter()
            .filter(|t| matches!(t, Telemetry::ParamValue { .. }))
            .count();
        assert_eq!(echoes, 2, "every request is answered");
        let _ = GAIN.set(Param::U8(1));
    }

    #[test]
    fn commands_for_other_items_are_rejected_with_a_log() {
        let _g = SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let o = drive::<FreeRunning>(
            &ECHO,
            std::vec![
                Command::StopNow { suite: 3, item: 3 },
                Command::ListSuites
            ],
            &[],
            (3, 0),
        );
        let logs = frames(&o)
            .iter()
            .filter(|t| matches!(t, Telemetry::Log(_)))
            .count();
        assert_eq!(logs, 2);
        assert_eq!(verdict(&o), Some((Verdict::Bounded, None)));
    }

    #[test]
    fn a_transport_error_still_tears_down() {
        let _g = SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset_counters();
        // Lockstep with no input and no deadline waits until the link ends.
        let o = drive::<Lockstep>(&ECHO, Vec::new(), &[], (0, 0));
        assert_eq!(o.result, Err("end"));
        assert_eq!(TEARDOWNS.load(Ordering::SeqCst), 1);
        assert_eq!(crate::fault::phase(), Phase::Idle);
    }

    #[test]
    fn a_lockstep_input_ahead_of_the_step_ends_the_run_with_the_gap_error() {
        let _g = serial();
        let o = drive::<Lockstep>(
            &ECHO,
            std::vec![input(1, &[1]), input(0, &[2])],
            &[],
            (0, 0),
        );
        assert_eq!(o.result, Ok(()));
        assert_eq!(verdict(&o).map(|v| v.0), Some(Verdict::Error));
        assert_eq!(done_message(&o).as_deref(), Some(INPUT_GAP_ERROR));
        assert_eq!(samples(&o), Samples::new(), "no step ran");
    }

    #[test]
    fn free_running_keeps_the_newest_input_and_ignores_older_ones() {
        let _g = serial();
        let stale = drive::<FreeRunning>(
            &ECHO,
            std::vec![input(5, &[1]), input(3, &[2])],
            &[],
            (3, 0),
        );
        let held: Samples = (0..3).map(|k| (k, std::vec![1])).collect();
        assert_eq!(samples(&stale), held, "an older input is dropped");
        let equal = drive::<FreeRunning>(
            &ECHO,
            std::vec![input(5, &[1]), input(5, &[2])],
            &[],
            (3, 0),
        );
        let replaced =
            std::vec![(0, std::vec![1]), (1, std::vec![2]), (2, std::vec![2])];
        assert_eq!(samples(&equal), replaced, "an equal input replaces");
    }

    #[test]
    fn a_lockstep_step_without_output_still_reports_its_sample() {
        let _g = serial();
        let o = drive::<Lockstep>(
            &ECHO,
            std::vec![input(0, &[]), STOP],
            &[],
            (0, 0),
        );
        assert_eq!(samples(&o), std::vec![(0, Vec::new())]);
    }

    #[test]
    fn the_board_is_armed_for_paced_modes_only() {
        let _g = serial();
        let periodic = drive::<Periodic>(&ECHO, Vec::new(), &[], (1, 10_000));
        assert_eq!(periodic.armed, std::vec![Some(10_000)]);
        assert_eq!(periodic.disarms, 1);
        let triggered =
            drive::<Triggered>(&ECHO, Vec::new(), &[1], (1, 10_000));
        assert_eq!(triggered.armed, std::vec![None]);
        assert_eq!(triggered.disarms, 1);
        let free = drive::<FreeRunning>(&ECHO, Vec::new(), &[], (1, 10_000));
        assert_eq!(free.armed, Arms::new());
        assert_eq!(free.disarms, 0);
    }

    #[test]
    fn the_link_deadline_fires_only_after_it_has_passed() {
        let _g = serial();
        // The clock ticks 1000 nanoseconds per read; the deadline is 1 ms after the first read, so the
        // read that lands exactly on it does not end the run.
        let o = drive::<Lockstep>(&TIMED, Vec::new(), &[], (0, 0));
        assert_eq!(verdict(&o), Some((Verdict::TimedOut, None)));
        assert_eq!(o.waits, 1000);
    }

    #[test]
    fn heartbeat_and_try_reset_during_a_run_are_silent() {
        let _g = serial();
        let o = drive::<FreeRunning>(
            &ECHO,
            std::vec![Command::Heartbeat, Command::TryReset],
            &[],
            (3, 0),
        );
        assert!(
            !frames(&o).iter().any(|t| matches!(t, Telemetry::Log(_))),
            "neither command is rejected"
        );
        assert_eq!(verdict(&o), Some((Verdict::Bounded, None)));
    }

    #[test]
    fn a_paced_mode_without_a_release_source_never_steps() {
        let _g = serial();
        // Triggered has no source here; Periodic has a clock but a zero period.
        let triggered =
            drive::<Triggered>(&TIMED, Vec::new(), &[], (0, 10_000));
        let periodic = drive::<Periodic>(&TIMED, Vec::new(), &[], (0, 0));
        for o in [triggered, periodic] {
            assert_eq!(verdict(&o), Some((Verdict::TimedOut, None)));
            assert_eq!(stats(&o), Some((0, 0)));
        }
    }

    #[test]
    fn step_timing_and_jitter_are_reported_for_periodic_runs_only() {
        let _g = serial();
        let periodic = drive::<Periodic>(&ECHO, Vec::new(), &[], (3, 10_000));
        let (min, max, jitter) = timing(&periodic).unwrap();
        assert_eq!((min, max), (1000, 1000));
        assert!(jitter > 0, "steps start after their release");
        let free = drive::<FreeRunning>(&ECHO, Vec::new(), &[], (3, 10_000));
        assert_eq!(timing(&free), Some((1000, 1000, 0)));
        let unpaced_zero = drive::<Periodic>(&ECHO, Vec::new(), &[], (3, 0));
        assert_eq!(timing(&unpaced_zero).map(|t| t.2), Some(0));
    }

    #[test]
    fn a_run_without_steps_reports_a_zero_minimum_step() {
        let _g = serial();
        let o = drive::<FreeRunning>(&BROKEN, Vec::new(), &[], (3, 0));
        assert_eq!(timing(&o), Some((0, 0, 0)));
    }

    #[test]
    fn setup_longer_than_one_period_does_not_overrun_before_step_zero() {
        let _g = serial();
        reset_counters();
        // Clock-paced Periodic: setup burns 2 ms while period_ns is 10 µs. The release
        // epoch must open after setup/arm, or the first ready() stacks due ≫ 1 → Overrun.
        let o = drive::<Periodic>(&SLOW, Vec::new(), &[], (3, 10_000));
        assert_eq!(o.result, Ok(()));
        assert_eq!(verdict(&o), Some((Verdict::Bounded, None)));
        assert_eq!(stats(&o), Some((3, 0)));
        assert_eq!(SETUPS.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn setup_longer_than_link_timeout_does_not_time_out_before_wait() {
        let _g = serial();
        reset_counters();
        // SLOW has link_timeout_ms = 1 and burns 2 ms of profiler time in setup. The
        // deadline must open after setup, or the first boundary ends with TimedOut
        // before any wait/step (FR-5).
        let o = drive::<Lockstep>(&SLOW, Vec::new(), &[], (0, 0));
        assert_eq!(o.result, Ok(()));
        assert_eq!(verdict(&o), Some((Verdict::TimedOut, None)));
        assert!(o.waits > 0, "timeout only after waiting for input");
        assert_eq!(stats(&o), Some((0, 0)));
    }

    #[test]
    fn a_suite_descriptor_can_hold_the_executable_and_table() {
        static EXECS: [ExecDescriptor; 1] = [task(echo, 0)];
        static SUITE: SuiteDescriptor = SuiteDescriptor {
            description: "",
            executables: &EXECS,
            name: "s",
            params: PARAMS,
        };
        assert_eq!(SUITE.executables.len(), 1);
    }
}
