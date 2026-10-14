//! Test doubles: a scripted link and a model of a target.

use std::collections::VecDeque;

use control_rs_ets::exec::Packet;
use control_rs_ets::wire::{
    Command, ExecMode, PROTOCOL_VERSION, PacketInfo, PayloadEncoding, Verdict,
};
use control_rs_ets::wire::{Param, ParamKey};

use crate::bridge::{OwnedTelemetry, PacketDecl, TestState};
use crate::error::HostError;
use crate::link::{FlagOf, HostLink, LinkEvent, Polled, UnitOf};
use crate::session::SessionOptions;
use crate::task_run::TaskSim;

/// A target model: how the fake answers each command.
pub type Behavior = Box<dyn FnMut(&Command<'_>) -> Vec<OwnedTelemetry>>;

/// A link that answers every command from a [`Behavior`].
pub struct ScriptLink {
    behavior: Behavior,
    inbox: VecDeque<LinkEvent>,
    /// Debug text of every command sent.
    pub sent: Vec<String>,
    /// Link resets requested.
    pub resets: u32,
    /// Whether to report the link closed once the inbox is empty.
    pub closed: bool,
    /// Whether the target answers nothing.
    pub mute: bool,
    /// Events delivered once, before any command is answered.
    pub preload: Vec<LinkEvent>,
}

/// The model of a target behind a [`ScriptLink`].
#[derive(Debug, Clone)]
pub struct Fake {
    /// Item that never answers while `silent` is set.
    pub hang_on: Option<u16>,
    /// A run that ignores `StopNow`.
    pub ignore_stop: bool,
    /// Releases missed, reported at the end of a triggered run.
    pub missed: u32,
    /// Item that panics the first time it runs.
    pub panic_item: Option<u16>,
    /// Setting index that discovery drops.
    pub drop_setting: Option<u16>,
    /// Release-source flag of `TargetInfo`.
    pub release_source: bool,
    /// Wire revision reported.
    pub version: u8,
    /// The running task: item, mode, step bound.
    pub active: Option<Running>,
    /// Whether the target is hung and answers nothing.
    pub dead: bool,
    /// Value of the `gain` setting.
    pub gain: u8,
}

/// The running task: item, mode, step bound.
pub type Running = (u16, ExecMode, u64);

/// One-byte packet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Byte(pub u8);

/// Doubles a byte on the target; the simulation adds one to each answer.
pub struct Increment;

impl Default for Fake {
    fn default() -> Self {
        Self {
            hang_on: None,
            ignore_stop: false,
            missed: 0,
            panic_item: None,
            drop_setting: None,
            release_source: false,
            version: PROTOCOL_VERSION,
            active: None,
            dead: false,
            gain: 1,
        }
    }
}

impl Fake {
    /// A link driving this model.
    pub fn link(self) -> ScriptLink {
        let mut model = self;
        ScriptLink {
            behavior: Box::new(move |cmd| model.answer(cmd)),
            inbox: VecDeque::new(),
            sent: Vec::new(),
            resets: 0,
            closed: false,
            mute: false,
            preload: Vec::new(),
        }
    }

    fn answer(&mut self, cmd: &Command<'_>) -> Vec<OwnedTelemetry> {
        match *cmd {
            Command::TryReset => {
                self.dead = false;
                self.active = None;
                Vec::new()
            }
            _ if self.dead => Vec::new(),
            Command::ListSuites => self.discovery(),
            Command::RunExecutable {
                item,
                mode,
                max_steps,
                ..
            } => self.start(item, mode, max_steps),
            Command::TaskInput { seq, payload, .. } => {
                self.step(seq, payload.first().copied().unwrap_or(0))
            }
            Command::StopNow { .. } if !self.ignore_stop => {
                self.finish(self.active.map_or(0, |a| a.2), Verdict::Aborted)
            }
            Command::Param { key, value, .. } => self.param(key, value),
            _ => Vec::new(),
        }
    }

    fn discovery(&self) -> Vec<OwnedTelemetry> {
        let mut out = vec![
            OwnedTelemetry::TargetInfo {
                protocol_version: self.version,
                board_id: 7,
                core_clock_hz: 1,
                fpu_flags: 0,
                release_source: self.release_source,
            },
            OwnedTelemetry::SuiteInfo {
                description: "suite".to_string(),
                name: "s".to_string(),
                setting_count: 2,
                suite_id: 0,
                test_count: 3,
            },
        ];
        let byte = PacketDecl {
            encoding: PayloadEncoding::Raw,
            type_name: "u8".to_string(),
        };
        for (item, name, modes, io) in [
            (0, "ok", ExecMode::Atomic.bit(), PacketDecl::default()),
            (1, "boom", ExecMode::Atomic.bit(), PacketDecl::default()),
            (2, "loop", 0b1_1111, byte),
        ] {
            out.push(OwnedTelemetry::TestInfo {
                description: String::new(),
                input: io.clone(),
                link_timeout_ms: 0,
                modes,
                name: name.to_string(),
                output: io,
                suite_id: 0,
                test_id: item,
            });
        }
        for index in 0..2_u16 {
            if self.drop_setting != Some(index) {
                out.push(self.setting(index));
            }
        }
        out.push(OwnedTelemetry::DiscoveryComplete);
        out
    }

    fn setting(&self, index: u16) -> OwnedTelemetry {
        OwnedTelemetry::SettingInfo {
            count: 2,
            description: String::new(),
            max: Param::U8(9),
            min: Param::U8(0),
            name: if index == 0 { "gain" } else { "bias" }.to_string(),
            setting_id: index,
            suite_id: 0,
            value: Param::U8(if index == 0 { self.gain } else { 3 }),
        }
    }

    fn param(
        &mut self,
        key: ParamKey<'_>,
        value: Option<Param>,
    ) -> Vec<OwnedTelemetry> {
        let index = match key {
            ParamKey::Name("gain") | ParamKey::Index(0) => 0,
            ParamKey::Name("bias") | ParamKey::Index(1) => 1,
            _ => return Vec::new(),
        };
        if let (0, Some(Param::U8(v))) = (index, value)
            && v <= 9
        {
            self.gain = v;
        }
        vec![self.setting(index)]
    }

    fn start(
        &mut self,
        item: u16,
        mode: ExecMode,
        max_steps: u64,
    ) -> Vec<OwnedTelemetry> {
        if self.hang_on == Some(item) {
            self.dead = true;
            return Vec::new();
        }
        if self.panic_item == Some(item) {
            self.panic_item = None;
            self.dead = true;
            return vec![OwnedTelemetry::TargetPanic {
                test_id: Some(item),
                fault_status: 0,
                file: "f.rs".to_string(),
                line: 3,
                message: "boom".to_string(),
                suite_id: Some(0),
            }];
        }
        let mut out = vec![state(item, TestState::Running, None)];
        match (item, mode) {
            (2, ExecMode::Triggered) => {
                out.push(sample(0, vec![9]));
                out.extend(self.finish(1, Verdict::Bounded));
            }
            (2, _) => self.active = Some((item, mode, max_steps)),
            _ => {
                out.push(state(item, TestState::Passed, Some(Verdict::Pass)));
                out.push(stats(item, 1, 0, 0));
            }
        }
        out
    }

    fn step(&mut self, seq: u64, input: u8) -> Vec<OwnedTelemetry> {
        let Some((_, _, max_steps)) = self.active else {
            return Vec::new();
        };
        let mut out = vec![sample(seq, vec![input.wrapping_mul(2)])];
        if max_steps != 0 && seq.saturating_add(1) >= max_steps {
            out.extend(self.finish(seq.saturating_add(1), Verdict::Bounded));
        }
        out
    }

    fn finish(&mut self, steps: u64, verdict: Verdict) -> Vec<OwnedTelemetry> {
        self.active = None;
        vec![
            OwnedTelemetry::TeardownReport {
                message: None,
                ok: true,
                suite_id: 0,
                test_id: 2,
            },
            state(2, TestState::Passed, Some(verdict)),
            stats(
                2,
                steps,
                self.missed,
                if self.missed == 0 { 0 } else { 500 },
            ),
        ]
    }
}

impl HostLink for ScriptLink {
    type Error = HostError;

    fn send(&mut self, cmd: &Command<'_>) -> UnitOf<Self> {
        self.sent.push(format!("{cmd:?}"));
        for event in std::mem::take(&mut self.preload) {
            self.inbox.push_back(event);
        }
        if self.mute {
            return Ok(());
        }
        for t in (self.behavior)(cmd) {
            self.inbox.push_back(LinkEvent::Telemetry(t));
        }
        Ok(())
    }

    fn poll(&mut self) -> Polled {
        match self.inbox.pop_front() {
            Some(event) => Polled::Event(event),
            None if self.closed => Polled::Closed,
            None => Polled::Idle,
        }
    }

    fn exited(&mut self) -> FlagOf<Self> {
        Ok(self.closed)
    }

    fn close(&mut self) {
        self.closed = true;
    }

    fn describe(&self) -> &'static str {
        "script"
    }

    fn reset(&mut self) -> UnitOf<Self> {
        self.resets = self.resets.saturating_add(1);
        Ok(())
    }
}

impl Packet for Byte {
    const INFO: PacketInfo<'static> = PacketInfo {
        type_name: "u8",
        encoding: PayloadEncoding::Raw,
    };

    fn decode(bytes: &[u8]) -> Option<Self> {
        match bytes {
            [b] => Some(Self(*b)),
            _ => None,
        }
    }

    fn encode(&self, out: &mut [u8]) -> Option<usize> {
        *out.first_mut()? = self.0;
        Some(1)
    }
}

impl TaskSim for Increment {
    type Input = Byte;
    type Output = Byte;

    fn initial(&mut self) -> Byte {
        Byte(1)
    }

    fn advance(&mut self, _k: u64, y: &Byte) -> Byte {
        Byte(y.0.wrapping_add(1))
    }
}

/// Short deadlines so failure paths finish quickly.
#[must_use]
pub const fn quick() -> SessionOptions {
    SessionOptions {
        case_timeout: std::time::Duration::from_millis(60),
        discovery_timeout: std::time::Duration::from_millis(100),
        reply_timeout: std::time::Duration::from_millis(100),
        stop_timeout: std::time::Duration::from_millis(40),
    }
}

fn state(
    item: u16,
    state: TestState,
    verdict: Option<Verdict>,
) -> OwnedTelemetry {
    OwnedTelemetry::TestStateChange {
        message: None,
        state,
        suite_id: 0,
        test_id: item,
        verdict,
    }
}

fn stats(item: u16, steps: u64, missed: u32, jitter: u32) -> OwnedTelemetry {
    OwnedTelemetry::RunStats {
        cycles: 10,
        jitter_max_ns: jitter,
        missed_releases: missed,
        stack_peak: 4,
        step_max_ns: 30,
        step_min_ns: 10,
        steps,
        suite_id: 0,
        test_id: item,
        time_ns: 100,
    }
}

fn sample(seq: u64, payload: Vec<u8>) -> OwnedTelemetry {
    OwnedTelemetry::TaskSample {
        payload,
        seq,
        suite_id: 0,
        test_id: 2,
    }
}
