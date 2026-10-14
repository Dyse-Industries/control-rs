//! Executable descriptors and the typed lifecycle contract.
//!
//! # Description
//!
//! One descriptor kind, [`ExecDescriptor`], carries everything the executive needs to run a case
//! or a lifecycle task: name, description, allowed modes, packet encodings and five function
//! pointers (`setup`, `step`, `reset`, `teardown`, `safe_state`). A case is a one-step `Atomic`
//! executable with shared no-op hooks. [`LifecycleTask`] is the typed contract a user implements;
//! `#[ets_suite]` erases it into the descriptor through the generic wrappers in this module.
//!
//! # Core Concepts
//!
//! - **Descriptor**: Data in flash that names no server, transport or profiler type.
//! - **Typed contract**: `setup` returns the task, `teardown` consumes it, so no step precedes
//!   setup or follows teardown. `safe_state` is an associated function because the fault path
//!   cannot borrow the state a faulting step holds.
//! - **Slot**: [`TaskSlot`] holds the task between setup and teardown in an interior-mutable
//!   static.
//! - **Packet**: [`Packet`] converts a task input or output to and from bytes.
//!
//! # Usage
//!
//! ```
//! use control_rs_ets::exec::{ExecDescriptor, TaskIo, TaskOutcome};
//! use control_rs_ets::wire::{ExecMode, Verdict};
//!
//! fn pass(_: &mut TaskIo<'_>) -> TaskOutcome {
//!     TaskOutcome::Finish(Verdict::Pass, None)
//! }
//!
//! let case = ExecDescriptor::case("case", "A one-step case", pass);
//! assert_eq!(case.modes, ExecMode::Atomic.bit());
//! ```
//!
//! # Limitations
//!
//! - **Cooperative**: A step must return; the executive regains control only between steps.

use core::cell::UnsafeCell;

use crate::wire::{ExecMode, PacketInfo, PayloadEncoding, Verdict};

/// Message of the `Error` verdict returned when an input packet fails to decode.
pub const INPUT_DECODE_ERROR: &str = "input decode";
/// Message of the `Error` verdict returned when a step runs before setup.
pub const NOT_SET_UP_ERROR: &str = "task not set up";
const NO_PACKET: PacketInfo<'static> = PacketInfo {
    type_name: "",
    encoding: PayloadEncoding::Postcard,
};

/// Message of the `Error` verdict returned when an output packet does not fit.
pub const OUTPUT_OVERFLOW_ERROR: &str = "output overflow";

/// Result of a lifecycle hook: success or a static error message.
pub type HookOf = Result<(), &'static str>;

/// A lifecycle hook.
pub type Hook = fn() -> HookOf;

/// The safe-state hook.
pub type SafeFn = fn();

/// The five hooks of a task: setup, step, reset, teardown and safe state.
pub type Hooks = (Hook, StepFn, Hook, Hook, SafeFn);

/// An optional borrowed packet.
pub type Held<'a> = Option<&'a [u8]>;

/// An optional static message.
pub type Message = Option<&'static str>;

/// Result of a task setup.
pub type SetupOf<T> = Result<T, &'static str>;

/// An optional reference to the input packet of `T`.
pub type InputOf<'a, T> = Option<&'a <T as LifecycleTask>::Input>;

/// What a typed step of `T` reports.
pub type StepOf<T> = TaskStep<<T as LifecycleTask>::Output>;

/// Storage of a task between setup and teardown.
type Stored<T> = UnsafeCell<Option<T>>;

/// The step function of an executable.
pub type StepFn = fn(&mut TaskIo<'_>) -> TaskOutcome;

/// What one step reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskOutcome {
    /// The run goes on.
    Continue,
    /// The run ends with a verdict and an optional message.
    Finish(Verdict, Message),
}

/// What a typed task step reports; the output of a continuing step is encoded for the wire.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskStep<O> {
    /// The run goes on and this packet is the step's output.
    Continue(O),
    /// The run ends with a verdict and an optional message.
    Finish(Verdict, Message),
}

/// The buffers one step reads and writes.
pub struct TaskIo<'a> {
    /// The input packet of this step, if one is held.
    pub input: Held<'a>,
    /// The index of the held input, if any.
    pub input_seq: Option<u64>,
    /// Buffer for the output packet.
    pub output: &'a mut [u8],
    /// Bytes of `output` the step wrote.
    pub output_len: usize,
    /// The step index.
    pub step: u64,
}

/// Describes one executable: a case or a lifecycle task.
#[derive(Debug, Clone, Copy)]
pub struct ExecDescriptor {
    /// Description reported at discovery.
    pub description: &'static str,
    /// Encoding of the input packet.
    pub input: PacketInfo<'static>,
    /// Host link timeout in milliseconds (`0` is none).
    pub link_timeout_ms: u32,
    /// Allowed modes: bit `n` is set for [`ExecMode`] variant `n`.
    pub modes: u8,
    /// Name reported at discovery.
    pub name: &'static str,
    /// Encoding of the output packet.
    pub output: PacketInfo<'static>,
    /// Re-applies parameters after a write.
    pub reset: Hook,
    /// Drives the hardware to a safe state on a fault.
    pub safe_state: fn(),
    /// Prepares a run.
    pub setup: Hook,
    /// Runs one step.
    pub step: StepFn,
    /// Ends a run.
    pub teardown: Hook,
}

/// A typed packet that crosses the link as bytes.
pub trait Packet: Sized {
    /// Encoding declaration reported at discovery.
    const INFO: PacketInfo<'static>;

    /// Decodes a packet from `bytes`.
    fn decode(bytes: &[u8]) -> Option<Self>;

    /// Encodes the packet into `out` and returns the length.
    fn encode(&self, out: &mut [u8]) -> Option<usize>;
}

/// The typed contract of a lifecycle task.
///
/// The trait has no default bodies: omitting a function is a compile error.
pub trait LifecycleTask: Sized {
    /// Allowed modes: bit `n` is set for [`ExecMode`] variant `n`.
    const MODES: u8;
    /// Host link timeout in milliseconds (`0` is none).
    const LINK_TIMEOUT_MS: u32;
    /// The input packet type.
    type Input: Packet;
    /// The output packet type.
    type Output: Packet;

    /// Prepares the task for a run.
    ///
    /// # Errors
    /// Returns a static message to end the run with an `Error` verdict.
    fn setup() -> SetupOf<Self>;

    /// Runs step `step` with the input of that step, if one is held.
    fn step(&mut self, step: u64, input: InputOf<'_, Self>) -> StepOf<Self>;

    /// Re-applies parameters after a write.
    ///
    /// # Errors
    /// Returns a static message to end the run with an `Error` verdict.
    fn reset(&mut self) -> HookOf;

    /// Ends the run and consumes the task.
    ///
    /// # Errors
    /// Returns a static message that the teardown report carries.
    fn teardown(self) -> HookOf;

    /// Drives the hardware to a safe state; called by the fault path.
    fn safe_state();
}

/// Holds one task between setup and teardown.
pub struct TaskSlot<T>(Stored<T>);

// SAFETY: One run is active at a time, the server thread is the only one that touches the slot
// between arming and completion, and the fault path never touches it
// (`test-suite-design.md` §4.3).
unsafe impl<T> Sync for TaskSlot<T> {}

impl ExecDescriptor {
    /// Describes a plain case: one `Atomic` step with shared no-op hooks.
    #[must_use]
    pub const fn case(
        name: &'static str,
        description: &'static str,
        step: StepFn,
    ) -> Self {
        Self {
            description,
            input: NO_PACKET,
            link_timeout_ms: 0,
            modes: ExecMode::Atomic.bit(),
            name,
            output: NO_PACKET,
            reset: || Ok(()),
            safe_state: || (),
            setup: || Ok(()),
            step,
            teardown: || Ok(()),
        }
    }
}

impl<T> Default for TaskSlot<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> TaskSlot<T> {
    /// Creates an empty slot.
    #[must_use]
    pub const fn new() -> Self {
        Self(UnsafeCell::new(None))
    }

    /// Runs `f` on the slot contents.
    fn with<R>(&self, f: impl FnOnce(&mut Option<T>) -> R) -> R {
        // SAFETY: See the `Sync` implementation; the borrow ends when `f` returns.
        f(unsafe { &mut *self.0.get() })
    }
}

/// Decodes a Postcard packet; the codec behind `#[derive(Packet)]`.
#[must_use]
pub fn postcard_decode<T: serde::de::DeserializeOwned>(
    bytes: &[u8],
) -> Option<T> {
    postcard::from_bytes(bytes).ok()
}

/// Encodes a Postcard packet into `out` and returns the length; the codec behind
/// `#[derive(Packet)]`.
#[must_use]
pub fn postcard_encode<T: serde::Serialize>(
    value: &T,
    out: &mut [u8],
) -> Option<usize> {
    postcard::to_slice(value, out).ok().map(|used| used.len())
}

/// Sets a task up into `slot`; the wrapper behind `setup`.
///
/// # Errors
/// Returns the task's setup message.
pub fn task_setup<T: LifecycleTask>(slot: &TaskSlot<T>) -> HookOf {
    let task = T::setup()?;
    slot.with(|state| *state = Some(task));
    Ok(())
}

/// Runs one step of the task in `slot`; the wrapper behind `step`.
pub fn task_step<T: LifecycleTask>(
    slot: &TaskSlot<T>,
    io: &mut TaskIo<'_>,
) -> TaskOutcome {
    let input = match io.input.map(T::Input::decode) {
        Some(None) => {
            return TaskOutcome::Finish(
                Verdict::Error,
                Some(INPUT_DECODE_ERROR),
            );
        }
        Some(Some(value)) => Some(value),
        None => None,
    };
    let step = io.step;
    let stepped = slot.with(|state| {
        state.as_mut().map(|task| task.step(step, input.as_ref()))
    });
    match stepped {
        None => TaskOutcome::Finish(Verdict::Error, Some(NOT_SET_UP_ERROR)),
        Some(TaskStep::Finish(verdict, message)) => {
            TaskOutcome::Finish(verdict, message)
        }
        Some(TaskStep::Continue(output)) => match output.encode(io.output) {
            Some(len) => {
                io.output_len = len;
                TaskOutcome::Continue
            }
            None => {
                TaskOutcome::Finish(Verdict::Error, Some(OUTPUT_OVERFLOW_ERROR))
            }
        },
    }
}

/// Re-applies parameters to the task in `slot`; the wrapper behind `reset`.
///
/// # Errors
/// Returns the task's reset message.
pub fn task_reset<T: LifecycleTask>(slot: &TaskSlot<T>) -> HookOf {
    slot.with(|state| {
        state
            .as_mut()
            .map_or(Err(NOT_SET_UP_ERROR), LifecycleTask::reset)
    })
}

/// Ends the task in `slot`; the wrapper behind `teardown`.
///
/// # Errors
/// Returns the task's teardown message.
pub fn task_teardown<T: LifecycleTask>(slot: &TaskSlot<T>) -> HookOf {
    slot.with(|state| {
        state
            .take()
            .map_or(Err(NOT_SET_UP_ERROR), LifecycleTask::teardown)
    })
}

/// Describes a task: the packet info, modes and timeout of `T`; the five hooks come from the
/// caller, which has the slot.
#[must_use]
pub const fn task_descriptor<T: LifecycleTask>(
    name: &'static str,
    description: &'static str,
    hooks: Hooks,
) -> ExecDescriptor {
    ExecDescriptor {
        description,
        input: T::Input::INFO,
        link_timeout_ms: T::LINK_TIMEOUT_MS,
        modes: T::MODES,
        name,
        output: T::Output::INFO,
        reset: hooks.2,
        safe_state: hooks.4,
        setup: hooks.0,
        step: hooks.1,
        teardown: hooks.3,
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;

    static SLOT: TaskSlot<Sum> = TaskSlot::new();

    /// A task that adds its input to a running sum.
    struct Sum(u8);

    /// A packet of one byte.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct Byte(u8);

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

    impl LifecycleTask for Sum {
        const MODES: u8 = ExecMode::Lockstep.bit();
        const LINK_TIMEOUT_MS: u32 = 250;
        type Input = Byte;
        type Output = Byte;

        fn setup() -> Result<Self, &'static str> {
            Ok(Self(0))
        }

        fn step(&mut self, step: u64, input: Option<&Byte>) -> TaskStep<Byte> {
            if step == 3 {
                return TaskStep::Finish(Verdict::Pass, Some("done"));
            }
            self.0 = self.0.saturating_add(input.map_or(0, |b| b.0));
            TaskStep::Continue(Byte(self.0))
        }

        fn reset(&mut self) -> HookOf {
            self.0 = 0;
            Ok(())
        }

        fn teardown(self) -> HookOf {
            if self.0 > 100 {
                Err("too large")
            } else {
                Ok(())
            }
        }

        fn safe_state() {}
    }

    fn io<'a>(input: Held<'a>, out: &'a mut [u8], step: u64) -> TaskIo<'a> {
        TaskIo {
            input,
            input_seq: input.map(|_| step),
            output: out,
            output_len: 0,
            step,
        }
    }

    #[test]
    fn a_case_descriptor_is_a_one_step_atomic_executable() {
        fn step(_: &mut TaskIo<'_>) -> TaskOutcome {
            TaskOutcome::Finish(Verdict::Pass, None)
        }
        let case = ExecDescriptor::case("c", "d", step);
        assert_eq!((case.name, case.description), ("c", "d"));
        assert_eq!(case.modes, ExecMode::Atomic.bit());
        assert_eq!(case.link_timeout_ms, 0);
        assert_eq!(case.input.type_name, "");
        assert_eq!((case.setup)(), Ok(()));
        assert_eq!((case.reset)(), Ok(()));
        assert_eq!((case.teardown)(), Ok(()));
        (case.safe_state)();
    }

    #[test]
    fn a_task_runs_through_the_generic_wrappers() {
        assert_eq!(
            task_step(&SLOT, &mut io(None, &mut [0; 4], 0)),
            TaskOutcome::Finish(Verdict::Error, Some(NOT_SET_UP_ERROR))
        );
        assert_eq!(task_setup(&SLOT), Ok(()));
        let mut out = [0u8; 4];
        let mut first = io(Some(&[5]), &mut out, 0);
        assert_eq!(task_step(&SLOT, &mut first), TaskOutcome::Continue);
        assert_eq!((first.output_len, out.first().copied()), (1, Some(5)));
        let mut second = io(Some(&[7]), &mut out, 1);
        assert_eq!(task_step(&SLOT, &mut second), TaskOutcome::Continue);
        assert_eq!(out.first().copied(), Some(12));
        assert_eq!(task_reset(&SLOT), Ok(()));
        let mut third = io(Some(&[1]), &mut out, 2);
        assert_eq!(task_step(&SLOT, &mut third), TaskOutcome::Continue);
        assert_eq!(out.first().copied(), Some(1));
        let mut last = io(None, &mut out, 3);
        assert_eq!(
            task_step(&SLOT, &mut last),
            TaskOutcome::Finish(Verdict::Pass, Some("done"))
        );
        assert_eq!(task_teardown(&SLOT), Ok(()));
        assert_eq!(task_teardown(&SLOT), Err(NOT_SET_UP_ERROR));
        assert_eq!(task_reset(&SLOT), Err(NOT_SET_UP_ERROR));
    }

    #[test]
    fn bad_input_and_a_full_output_buffer_end_the_run_with_an_error() {
        static BAD: TaskSlot<Sum> = TaskSlot::new();
        assert_eq!(task_setup(&BAD), Ok(()));
        let mut out = [0u8; 4];
        assert_eq!(
            task_step(&BAD, &mut io(Some(&[1, 2]), &mut out, 0)),
            TaskOutcome::Finish(Verdict::Error, Some(INPUT_DECODE_ERROR))
        );
        assert_eq!(
            task_step(&BAD, &mut io(Some(&[1]), &mut [], 0)),
            TaskOutcome::Finish(Verdict::Error, Some(OUTPUT_OVERFLOW_ERROR))
        );
    }

    #[test]
    fn teardown_reports_the_task_failure() {
        static FAIL: TaskSlot<Sum> = TaskSlot::new();
        assert_eq!(task_setup(&FAIL), Ok(()));
        let mut out = [0u8; 4];
        let _ = task_step(&FAIL, &mut io(Some(&[200]), &mut out, 0));
        let _ = task_step(&FAIL, &mut io(Some(&[200]), &mut out, 1));
        assert_eq!(task_teardown(&FAIL), Err("too large"));
    }

    #[test]
    fn the_postcard_codec_round_trips() {
        let mut out = [0u8; 8];
        let used = postcard_encode(&300u16, &mut out).unwrap();
        assert_eq!(postcard_decode::<u16>(out.get(..used).unwrap()), Some(300));
        assert_eq!(postcard_decode::<u16>(&[]), None);
        assert_eq!(postcard_encode(&300u16, &mut out[..1]), None);
    }

    #[test]
    fn a_task_descriptor_takes_its_metadata_from_the_type() {
        static D: TaskSlot<Sum> = TaskSlot::new();
        fn setup() -> HookOf {
            task_setup(&D)
        }
        fn step(io: &mut TaskIo<'_>) -> TaskOutcome {
            task_step(&D, io)
        }
        fn reset() -> HookOf {
            task_reset(&D)
        }
        fn teardown() -> HookOf {
            task_teardown(&D)
        }
        let d = task_descriptor::<Sum>(
            "sum",
            "adds",
            (setup, step, reset, teardown, Sum::safe_state),
        );
        assert_eq!(d.modes, ExecMode::Lockstep.bit());
        assert_eq!(d.link_timeout_ms, 250);
        assert_eq!(d.input.type_name, "u8");
        assert_eq!(d.output.encoding, PayloadEncoding::Raw);
    }
}
