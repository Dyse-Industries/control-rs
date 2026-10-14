//! Command and telemetry schema of the target-to-host wire protocol.
//!
//! # Description
//!
//! `Command` and `Telemetry` are externally tagged Postcard enums. The declaration order of their
//! variants is the wire discriminant, so variants are append-only after revision 2. A message is
//! encoded into one buffer of at most [`MAX_MESSAGE_SIZE`] bytes and moves over a
//! [`Link`](crate::link::Link) as a whole.
//!
//! # Core Concepts
//!
//! - **Revision check**: [`Telemetry::TargetInfo`] is the first message after boot and carries
//!   [`PROTOCOL_VERSION`]; a host refuses a target of another revision before it decodes anything
//!   else.
//! - **Items**: Cases and tasks are executables addressed by `(suite, item)`. `ItemState` and
//!   `Verdict` report their progress.
//! - **Parameters**: A suite's parameters travel as one `Command::Param` request and one
//!   `Telemetry::ParamValue` answer carrying a named, typed primitive ([`Param`]).
//!
//! # Usage
//!
//! ```
//! use control_rs_ets::wire::{Command, Telemetry};
//!
//! let mut buf = [0u8; 64];
//! let bytes = postcard::to_slice(&Command::ListSuites, &mut buf)?;
//! let back: Command<'_> = postcard::from_bytes(bytes)?;
//! assert!(matches!(back, Command::ListSuites));
//! let done = postcard::to_slice(&Telemetry::DiscoveryComplete, &mut buf)?;
//! assert_eq!(done.len(), 1);
//! # Ok::<(), postcard::Error>(())
//! ```
//!
//! # Limitations
//!
//! - **Message size**: An encoded message is at most [`MAX_MESSAGE_SIZE`] bytes.

/// Maximum bytes of one encoded message on every link.
pub const MAX_MESSAGE_SIZE: usize = 512;

/// Wire-contract revision carried in [`Telemetry::TargetInfo`].
///
/// Incremented whenever `Command` or `Telemetry` gains, loses or reorders a variant, or a
/// variant's payload changes shape. Variants are append-only.
pub const PROTOCOL_VERSION: u8 = 2;

/// Commands sent from the host to the target.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize,
)]
pub enum Command<'a> {
    /// Request the target to stream the list of all suites, executables and parameters.
    ListSuites,
    /// Start an executable in a mode.
    RunExecutable {
        /// Suite that owns the executable.
        suite: u16,
        /// Executable within the suite.
        item: u16,
        /// Execution mode.
        mode: ExecMode,
        /// Stop with [`Verdict::Bounded`] after this many steps (`0` is unbounded).
        max_steps: u64,
        /// Release period in nanoseconds for [`ExecMode::Periodic`].
        period_ns: u32,
    },
    /// Read a parameter, or write it when `value` is set.
    ///
    /// The target answers with [`Telemetry::ParamValue`] carrying the stored value, also when a
    /// write is refused.
    Param {
        /// Suite that owns the parameter.
        suite: u16,
        /// Name or index of the parameter.
        key: ParamKey<'a>,
        /// New value, of the parameter's type, or `None` to read.
        value: Option<Param>,
    },
    /// An input packet for a running task.
    TaskInput {
        /// Suite that owns the task.
        suite: u16,
        /// Task within the suite.
        item: u16,
        /// Step the input is for.
        seq: u64,
        /// Encoded input packet.
        payload: &'a [u8],
    },
    /// Refresh the target's host link deadline.
    Heartbeat,
    /// Stop the running executable at the next boundary.
    StopNow {
        /// Suite that owns the executable.
        suite: u16,
        /// Executable within the suite.
        item: u16,
    },
    /// Request the target to reset.
    TryReset,
}

/// Telemetry and logs sent from the target to the host.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize,
)]
pub enum Telemetry<'a> {
    /// Target identity and wire revision; the first message after boot.
    TargetInfo {
        /// The target's [`PROTOCOL_VERSION`].
        protocol_version: u8,
        /// Board identifier.
        board_id: u16,
        /// Core clock in hertz.
        core_clock_hz: u32,
        /// Floating-point unit flags.
        fpu_flags: u8,
        /// Whether the board binding supplies a release source.
        release_source: bool,
    },
    /// Metadata about a suite.
    SuiteInfo {
        /// Suite identifier.
        suite: u16,
        /// Suite name.
        name: &'a str,
        /// Suite description.
        description: &'a str,
        /// Number of executables in the suite.
        executable_count: u16,
        /// Number of parameters in the suite.
        param_count: u16,
    },
    /// Metadata about an executable.
    ExecutableInfo {
        /// Suite that owns the executable.
        suite: u16,
        /// Executable within the suite.
        item: u16,
        /// Executable name.
        name: &'a str,
        /// Executable description.
        description: &'a str,
        /// Bit `n` is set when [`ExecMode`] variant `n` is allowed.
        modes: u8,
        /// Encoding of the input packet.
        input: PacketInfo<'a>,
        /// Encoding of the output packet.
        output: PacketInfo<'a>,
        /// Host link timeout in milliseconds (`0` is none).
        link_timeout_ms: u32,
    },
    /// A parameter's description, bounds and stored value.
    ParamValue {
        /// Suite that owns the parameter.
        suite: u16,
        /// Index of the parameter within the suite.
        index: u16,
        /// Number of parameters in the suite.
        count: u16,
        /// Parameter name.
        name: &'a str,
        /// Parameter description.
        description: &'a str,
        /// Stored value.
        value: Param,
        /// Smallest accepted value.
        min: Param,
        /// Largest accepted value.
        max: Param,
    },
    /// Notification that the target has finished sending discovery information.
    DiscoveryComplete,
    /// State change of an executable.
    ItemState {
        /// Suite that owns the executable.
        suite: u16,
        /// Executable within the suite.
        item: u16,
        /// New state.
        state: ItemState,
        /// Optional explanation.
        message: Option<&'a str>,
    },
    /// Statistics of a finished run.
    RunStats {
        /// Suite that owns the executable.
        suite: u16,
        /// Executable within the suite.
        item: u16,
        /// Steps executed.
        steps: u64,
        /// Run time in nanoseconds.
        time_ns: u64,
        /// Shortest step in nanoseconds.
        step_min_ns: u32,
        /// Longest step in nanoseconds.
        step_max_ns: u32,
        /// CPU cycles consumed.
        cycles: u64,
        /// Peak stack usage in bytes.
        stack_peak: u32,
        /// Largest release jitter in nanoseconds.
        jitter_max_ns: u32,
        /// Releases that were missed.
        missed_releases: u32,
    },
    /// An output packet of a task step.
    TaskSample {
        /// Suite that owns the task.
        suite: u16,
        /// Task within the suite.
        item: u16,
        /// Step that produced the packet.
        seq: u64,
        /// Encoded output packet.
        payload: &'a [u8],
    },
    /// Result of a task's teardown.
    TeardownReport {
        /// Suite that owns the task.
        suite: u16,
        /// Task within the suite.
        item: u16,
        /// Whether teardown succeeded.
        ok: bool,
        /// Optional explanation.
        message: Option<&'a str>,
    },
    /// A log message.
    Log(LogMessage<'a>),
    /// Notification of a target panic or hardware fault.
    TargetPanic {
        /// File where the panic was raised.
        file: &'a str,
        /// Line where the panic was raised.
        line: u32,
        /// Panic message.
        message: &'a str,
        /// Architecture fault status (`0` when none was captured).
        fault_status: u32,
        /// Suite that was active, if any.
        suite: Option<u16>,
        /// Executable that was active, if any.
        item: Option<u16>,
    },
}

/// Mode in which an executable runs.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize,
)]
pub enum ExecMode {
    /// Run once with interrupts masked.
    Atomic,
    /// Call step `k` after input `k` arrives.
    Lockstep,
    /// Call steps without waiting for input.
    FreeRunning,
    /// Call steps at a fixed period.
    Periodic,
    /// Call a step per event of the board release source.
    Triggered,
}

/// State of an executable.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize,
)]
pub enum ItemState {
    /// Not running.
    Idle,
    /// Running.
    Running,
    /// Finished with a verdict.
    Done(Verdict),
    /// Ended by a fault.
    Faulted,
}

/// End state of a run.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize,
)]
pub enum Verdict {
    /// The run passed.
    Pass,
    /// The run failed.
    Fail,
    /// The run could not complete.
    Error,
    /// The host stopped the run.
    Aborted,
    /// The host link timed out.
    TimedOut,
    /// The step bound was reached.
    Bounded,
    /// A step overran its release.
    Overrun,
}

/// Typed primitive value of a parameter.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub enum Param {
    /// An 8-bit unsigned integer.
    U8(u8),
    /// A 16-bit unsigned integer.
    U16(u16),
    /// A 32-bit unsigned integer.
    U32(u32),
    /// A 32-bit signed integer.
    I32(i32),
    /// A 32-bit floating point value.
    F32(f32),
    /// A boolean.
    Bool(bool),
}

/// Selects a parameter by name or by index.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize,
)]
pub enum ParamKey<'a> {
    /// Look up by name.
    Name(&'a str),
    /// Look up by index within the suite.
    Index(u16),
}

/// Encoding declaration of a task packet.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize,
)]
pub struct PacketInfo<'a> {
    /// Name of the packet type (empty for none).
    pub type_name: &'a str,
    /// Encoding of the packet bytes.
    pub encoding: PayloadEncoding,
}

/// Encoding of a task packet.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize,
)]
pub enum PayloadEncoding {
    /// Postcard.
    Postcard,
    /// Packed little-endian bytes.
    Raw,
}

/// A log message produced by an executable or the server.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize,
)]
pub struct LogMessage<'a> {
    /// Severity: `0` error, `1` warning, `2` information.
    pub level: u8,
    /// Origin of the message.
    pub target: &'a str,
    /// Message text.
    pub message: &'a str,
}

impl PartialEq for Param {
    /// Compares two values: types must match, and `F32` compares by bit pattern.
    fn eq(&self, other: &Self) -> bool {
        match (*self, *other) {
            (Self::U8(l), Self::U8(r)) => l == r,
            (Self::U16(l), Self::U16(r)) => l == r,
            (Self::U32(l), Self::U32(r)) => l == r,
            (Self::I32(l), Self::I32(r)) => l == r,
            (Self::F32(l), Self::F32(r)) => l.to_bits() == r.to_bits(),
            (Self::Bool(l), Self::Bool(r)) => l == r,
            _ => false,
        }
    }
}

impl Eq for Param {}

impl Param {
    /// Returns the largest value of this value's type.
    #[must_use]
    pub const fn type_max(&self) -> Self {
        match self {
            Self::U8(_) => Self::U8(u8::MAX),
            Self::U16(_) => Self::U16(u16::MAX),
            Self::U32(_) => Self::U32(u32::MAX),
            Self::I32(_) => Self::I32(i32::MAX),
            Self::F32(_) => Self::F32(f32::MAX),
            Self::Bool(_) => Self::Bool(true),
        }
    }

    /// Returns the smallest value of this value's type.
    #[must_use]
    pub const fn type_min(&self) -> Self {
        match self {
            Self::U8(_) => Self::U8(u8::MIN),
            Self::U16(_) => Self::U16(u16::MIN),
            Self::U32(_) => Self::U32(u32::MIN),
            Self::I32(_) => Self::I32(i32::MIN),
            Self::F32(_) => Self::F32(f32::MIN),
            Self::Bool(_) => Self::Bool(false),
        }
    }
}

impl ExecMode {
    /// Returns the bit that marks this mode in `ExecutableInfo.modes`.
    #[must_use]
    pub const fn bit(self) -> u8 {
        1 << (self as u8)
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use std::vec::Vec;

    fn encode<T: serde::Serialize>(value: &T) -> Vec<u8> {
        let mut buf = [0u8; MAX_MESSAGE_SIZE];
        postcard::to_slice(value, &mut buf).unwrap().to_vec()
    }

    fn commands() -> [Command<'static>; 8] {
        [
            Command::ListSuites,
            Command::RunExecutable {
                suite: 1,
                item: 2,
                mode: ExecMode::Lockstep,
                max_steps: 300,
                period_ns: 1_000_000,
            },
            Command::Param {
                suite: 0,
                key: ParamKey::Name("kp"),
                value: Some(Param::F32(1.5)),
            },
            Command::TaskInput {
                suite: 0,
                item: 1,
                seq: 7,
                payload: &[1, 2, 3],
            },
            Command::Heartbeat,
            Command::StopNow { suite: 3, item: 4 },
            Command::TryReset,
            Command::Param {
                suite: 2,
                key: ParamKey::Index(5),
                value: None,
            },
        ]
    }

    fn telemetry_a() -> [Telemetry<'static>; 4] {
        [
            Telemetry::TargetInfo {
                protocol_version: PROTOCOL_VERSION,
                board_id: 0x0401,
                core_clock_hz: 600_000_000,
                fpu_flags: 1,
                release_source: true,
            },
            Telemetry::SuiteInfo {
                suite: 0,
                name: "s",
                description: "d",
                executable_count: 2,
                param_count: 1,
            },
            Telemetry::ExecutableInfo {
                suite: 0,
                item: 1,
                name: "n",
                description: "d",
                modes: ExecMode::Atomic.bit() | ExecMode::Lockstep.bit(),
                input: PacketInfo {
                    type_name: "f32",
                    encoding: PayloadEncoding::Postcard,
                },
                output: PacketInfo {
                    type_name: "",
                    encoding: PayloadEncoding::Raw,
                },
                link_timeout_ms: 500,
            },
            Telemetry::ParamValue {
                suite: 0,
                index: 0,
                count: 1,
                name: "kp",
                description: "gain",
                value: Param::U32(115_200),
                min: Param::U32(0),
                max: Param::U32(u32::MAX),
            },
        ]
    }

    fn telemetry_b() -> [Telemetry<'static>; 7] {
        [
            Telemetry::DiscoveryComplete,
            Telemetry::ItemState {
                suite: 0,
                item: 1,
                state: ItemState::Done(Verdict::Pass),
                message: Some("ok"),
            },
            Telemetry::RunStats {
                suite: 0,
                item: 1,
                steps: 1,
                time_ns: 2_000,
                step_min_ns: 2_000,
                step_max_ns: 2_000,
                cycles: 1_200,
                stack_peak: 64,
                jitter_max_ns: 0,
                missed_releases: 0,
            },
            Telemetry::TaskSample {
                suite: 0,
                item: 1,
                seq: 9,
                payload: &[4, 5],
            },
            Telemetry::TeardownReport {
                suite: 0,
                item: 1,
                ok: false,
                message: None,
            },
            Telemetry::Log(LogMessage {
                level: 1,
                target: "server",
                message: "m",
            }),
            Telemetry::TargetPanic {
                file: "f.rs",
                line: 50,
                message: "boom",
                fault_status: 0xDEAD,
                suite: Some(1),
                item: None,
            },
        ]
    }

    fn telemetry() -> Vec<Telemetry<'static>> {
        telemetry_a().into_iter().chain(telemetry_b()).collect()
    }

    #[test]
    fn command_and_telemetry_round_trip() {
        for cmd in commands() {
            let bytes = encode(&cmd);
            let back: Command<'_> = postcard::from_bytes(&bytes).unwrap();
            assert_eq!(back, cmd);
        }
        for tel in telemetry() {
            let bytes = encode(&tel);
            let back: Telemetry<'_> = postcard::from_bytes(&bytes).unwrap();
            assert_eq!(back, tel);
        }
    }

    /// Variant tags are the declaration order: a change here is a wire break.
    #[test]
    fn golden_vectors() {
        let tags: Vec<u8> = commands()
            .iter()
            .map(|c| encode(c).first().copied().unwrap())
            .collect();
        assert_eq!(tags, std::vec![0, 1, 2, 3, 4, 5, 6, 2]);
        let tags: Vec<u8> = telemetry()
            .iter()
            .map(|t| encode(t).first().copied().unwrap())
            .collect();
        assert_eq!(tags, std::vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        golden_commands();
        golden_telemetry();
    }

    fn golden_commands() {
        assert_eq!(encode(&Command::ListSuites), std::vec![0x00]);
        assert_eq!(encode(&Command::Heartbeat), std::vec![0x04]);
        assert_eq!(
            encode(&Command::StopNow { suite: 3, item: 4 }),
            std::vec![0x05, 3, 4]
        );
        assert_eq!(
            encode(&Command::RunExecutable {
                suite: 1,
                item: 2,
                mode: ExecMode::Periodic,
                max_steps: 300,
                period_ns: 1_000_000,
            }),
            std::vec![0x01, 1, 2, 3, 0xAC, 0x02, 0xC0, 0x84, 0x3D]
        );
        assert_eq!(
            encode(&Command::Param {
                suite: 0,
                key: ParamKey::Name("kp"),
                value: Some(Param::U32(115_200)),
            }),
            std::vec![0x02, 0, 0, 2, b'k', b'p', 1, 2, 0x80, 0x84, 0x07]
        );
        assert_eq!(
            encode(&Command::Param {
                suite: 2,
                key: ParamKey::Index(5),
                value: None,
            }),
            std::vec![0x02, 2, 1, 5, 0]
        );
    }

    fn golden_telemetry() {
        assert_eq!(
            encode(&Telemetry::TargetInfo {
                protocol_version: 2,
                board_id: 0x0401,
                core_clock_hz: 600_000_000,
                fpu_flags: 1,
                release_source: true,
            }),
            std::vec![0x00, 2, 0x81, 0x08, 0x80, 0x8C, 0x8D, 0x9E, 0x02, 1, 1]
        );
        assert_eq!(
            encode(&Telemetry::ItemState {
                suite: 0,
                item: 1,
                state: ItemState::Done(Verdict::Bounded),
                message: None,
            }),
            std::vec![0x05, 0, 1, 2, 5, 0]
        );
        assert_eq!(
            encode(&PacketInfo {
                type_name: "f32",
                encoding: PayloadEncoding::Raw,
            }),
            std::vec![3, b'f', b'3', b'2', 1]
        );
    }

    #[test]
    fn largest_variant_fits_512() {
        let text = "x".repeat(200);
        let big = Telemetry::TargetPanic {
            file: &text,
            line: u32::MAX,
            message: &text,
            fault_status: u32::MAX,
            suite: Some(u16::MAX),
            item: Some(u16::MAX),
        };
        assert!(encode(&big).len() <= MAX_MESSAGE_SIZE);
        let packet = [0xFFu8; crate::MAX_PACKET_SIZE];
        let sample = Telemetry::TaskSample {
            suite: u16::MAX,
            item: u16::MAX,
            seq: u64::MAX,
            payload: &packet,
        };
        assert!(encode(&sample).len() <= MAX_MESSAGE_SIZE);
        let input = Command::TaskInput {
            suite: u16::MAX,
            item: u16::MAX,
            seq: u64::MAX,
            payload: &packet,
        };
        assert!(encode(&input).len() <= MAX_MESSAGE_SIZE);
    }

    #[test]
    fn param_equality_is_type_strict_and_bitwise_for_floats() {
        assert_ne!(Param::U8(1), Param::U16(1));
        assert_eq!(Param::F32(f32::NAN), Param::F32(f32::NAN));
        assert_ne!(Param::F32(0.0), Param::F32(-0.0));
        assert_eq!(Param::U8(4).type_max(), Param::U8(255));
        assert_eq!(Param::I32(4).type_min(), Param::I32(i32::MIN));
        assert_eq!(Param::Bool(false).type_max(), Param::Bool(true));
    }

    #[test]
    fn mode_bits_follow_declaration_order() {
        assert_eq!(ExecMode::Atomic.bit(), 0b00001);
        assert_eq!(ExecMode::Triggered.bit(), 0b10000);
    }
}
