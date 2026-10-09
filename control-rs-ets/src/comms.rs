//! Target-to-Host communication protocol and traits.
//!
//! # Description
//!
//! This module defines the framing reader, serialization utilities and target-to-host messaging traits
//! used to exchange ETS test suite information, run tests, report performance metrics and handle telemetry logs.
//! The transport layer uses a robust packet framing scheme with CRC-16-IBM-SDLC checksum protection.
//!
//! # Core Concepts
//!
//! - **`HostComms` Trait**: Target-agnostic interface that serial-comms or USB-comms handlers implement.
//! - **Framed Reader**: `FrameReader` parses incoming stream bytes statefully into verified frame slices.
//! - **Command/Telemetry Enums**: Defines the protocol message schemas for control instructions and test reporting.
//! - **`CommsLock`**: An atomic flag lock preventing multiple execution paths from concurrently writing telemetry frames.
//!
//! # Usage
//!
//! ```
//! use control_rs_ets::comms::{FrameReader, Telemetry, frame_telemetry};
//!
//! let mut reader = FrameReader::new();
//! assert!(reader.is_idle());
//!
//! let mut buf = [0u8; 128];
//! let size = frame_telemetry(&Telemetry::DiscoveryComplete, &mut buf).unwrap();
//!
//! let mut decoded = false;
//! for &b in &buf[..size] {
//!     if reader.handle_byte(b).is_some() {
//!         decoded = true;
//!         break;
//!     }
//! }
//! assert!(decoded);
//! ```
//!
//! # Features
//!
//! - **Checksum Verification**: Integrates CRC-16 to check frame data integrity automatically.
//!
//! # Limitations
//!
//! - **Max Payload Size**: Individual payload lengths are limited to 512 bytes.
//!
//! > L. L. Peterson and B. S. Davie, "2.3 Framing," in Computer Networks: A Systems Approach, 2024.
//! >   \[Online\]. Available: <https://book.systemsapproach.org/>. [Accessed: June 27, 2026].

use core::sync::atomic::{AtomicBool, Ordering};

use crate::settings::SettingValue;

/// Sync (2) + length (2) + CRC-16 (2).
pub const FRAME_OVERHEAD: usize = 6;
/// Maximum encoded frame size (`MAX_PAYLOAD_SIZE` + `FRAME_OVERHEAD`).
pub const MAX_FRAME_SIZE: usize = 518;
/// Maximum bytes of a [`Telemetry::TaskState`] or [`Telemetry::TeardownReport`]
/// message; longer messages are truncated at a `char` boundary.
pub const MAX_MESSAGE_SIZE: usize = 256;
/// Maximum postcard payload bytes in one frame.
pub const MAX_PAYLOAD_SIZE: usize = 512;
/// Wire-contract revision carried in [`Telemetry::TargetInfo`].
///
/// Incremented whenever `Command` or `Telemetry` gains, loses or reorders a
/// variant, or a variant's payload changes shape. Variants are append-only.
/// `0` is reserved for firmware that predates the handshake and never sends
/// `TargetInfo`.
pub const PROTOCOL_VERSION: u8 = 2;
const START_BYTE_1: u8 = 0xAA;
const START_BYTE_2: u8 = 0x55;

/// The payload returned by `handle_byte` when a full frame is decoded.
///
/// Refers to the decoded raw byte slice inside the reader's internal buffer.
pub type DecodedFrame<'a> = &'a [u8];

/// Result of polling a command from the host.
///
/// Returns a command if successfully decoded or transport error `E`.
pub type PollResult<'a, E> = Result<Option<Command<'a>>, E>;

/// Result of sending telemetry or flushing.
///
/// Returns `Ok(())` or transport error `E`.
pub type SendResult<E> = Result<(), E>;

/// A trait for executing frame-based communication between target and host.
///
/// Handlers of this trait bridge the parsed commands and telemetry messages
/// onto concrete hardware peripherals (like UART, USB or RTT).
///
/// # Safety
/// Implementations must guarantee safe register/hardware access during transfer and framing.
/// This trait does not use `unsafe` code.
///
/// # Panics
/// Trait operations do not panic.
///
/// # Example
/// ```
/// use control_rs_ets::comms::{HostComms, Telemetry, Command, SendResult, PollResult};
///
/// struct MyComms;
/// impl HostComms for MyComms {
///     type Error = &'static str;
///     fn flush(&mut self) -> SendResult<Self::Error> { Ok(()) }
///     fn poll_command(&mut self) -> PollResult<'_, Self::Error> { Ok(None) }
///     fn send_telemetry(&mut self, _: &Telemetry<'_>) -> SendResult<Self::Error> { Ok(()) }
/// }
///
/// let mut comms = MyComms;
/// assert!(comms.flush().is_ok());
/// ```
#[allow(clippy::type_complexity)]
pub trait HostComms {
    /// The error type associated with transport failures.
    type Error;

    /// Closes the communication interface (for example, signaling semihosting exit).
    fn close(&mut self) {}

    /// Closes the communication interface with a failure/error status.
    fn close_on_failure(&mut self) {}

    /// Flush any pending buffered data out to the physical interface.
    ///
    /// # Returns
    /// * `SendResult<Self::Error>` - Success or transport error status.
    ///
    /// # Errors
    /// Returns a transport error if flushing the buffer to the physical interface fails.
    fn flush(&mut self) -> SendResult<Self::Error>;

    /// Read incoming bytes and try to parse a Command.
    ///
    /// This should be non-blocking.
    ///
    /// # Returns
    /// * `PollResult<'_, Self::Error>`
    ///     * `Ok(Some(Command))` when a full valid command frame is parsed.
    ///     * `Ok(None)` if no command is ready yet.
    ///     * `Err(Error)` on serial port or protocol errors.
    ///
    /// # Errors
    /// Returns a serial port or protocol error if reading or de-framing fails.
    fn poll_command(&mut self) -> PollResult<'_, Self::Error>;

    /// Send a telemetry message to the host.
    ///
    /// # Arguments
    /// * `telemetry` - Reference to the telemetry variant schema.
    ///
    /// # Returns
    /// * `SendResult<Self::Error>` - Success or transport error status.
    ///
    /// # Errors
    /// Returns a transport error if serialization or writing to the physical interface fails.
    fn send_telemetry(
        &mut self,
        telemetry: &Telemetry<'_>,
    ) -> SendResult<Self::Error>;
}

/// Commands sent from the Host TUI to the Target MCU.
///
/// Encapsulates all executable remote operations that can be instructed by the host.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum Command<'a> {
    /// Request the target to stream the list of all suites, tests and settings.
    ListSuites,
    /// Request execution of a specific test.
    RunExecutable {
        /// The ID of the test suite to execute.
        suite_id: u16,
        /// The ID of the test within the suite.
        test_id: u16,
    },
    /// Update a setting's value.
    SetSetting {
        /// The ID of the setting to update.
        setting_id: u16,
        /// The ID of the suite containing the setting.
        suite_id: u16,
        /// The new value of the setting.
        value: SettingValue,
    },
    /// Request the target to reset.
    TryReset,
    /// Start the task of a suite.
    StartTask {
        /// The ID of the suite containing the task.
        suite_id: u16,
        /// The task's identifier within the suite.
        test_id: u16,
        /// Stop with [`TaskRunState::Bounded`] after this many steps (`0` is unbounded).
        max_steps: u64,
        /// Call step `k` only after input `k` has arrived.
        lockstep: bool,
    },
    /// Stop the running task at the next step boundary.
    StopNow {
        /// The ID of the suite containing the task.
        suite_id: u16,
        /// The task's identifier within the suite.
        test_id: u16,
    },
    /// Refresh the target's host link deadline.
    Heartbeat,
    /// An input packet for a running task.
    TaskInput {
        /// The ID of the suite containing the task.
        suite_id: u16,
        /// The task's identifier within the suite.
        test_id: u16,
        /// The step index the input is for.
        seq: u64,
        /// The `postcard` encoding of the task's input type.
        payload: &'a [u8],
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReaderState {
    ReadingPayload { len: usize, read: usize },
    WaitChecksum1 { len: usize },
    WaitChecksum2 { len: usize, crc1: u8 },
    WaitLen1,
    WaitLen2,
    WaitStart1,
    WaitStart2,
}

/// Telemetry and logs sent from the Target MCU to the Host TUI.
///
/// Encapsulates all data updates, performance metrics, log outputs and crash reports.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum Telemetry<'a> {
    /// Notification that the target has finished sending discovery information.
    DiscoveryComplete,
    /// A log message.
    Log(LogMessage<'a>),
    /// Performance metrics for a completed test run.
    MetricReport {
        /// CPU cycles consumed during the test run.
        cycles: u64,
        /// Peak stack memory usage in bytes during the test run.
        stack_peak: u32,
        /// The ID of the suite containing the test.
        suite_id: u16,
        /// The ID of the test.
        test_id: u16,
        /// Elapsed time of the test run in microseconds.
        time_us: u64,
    },
    /// Metadata about a setting within a suite.
    SettingInfo {
        /// The doc comment description of the setting.
        description: &'a str,
        /// The name of the setting.
        name: &'a str,
        /// The ID assigned to this setting.
        setting_id: u16,
        /// The ID of the suite containing this setting.
        suite_id: u16,
        /// The current value of the setting.
        value: SettingValue,
    },
    /// Metadata about a discovered suite.
    SuiteInfo {
        /// The doc comment description of the suite.
        description: &'a str,
        /// The name of the suite.
        name: &'a str,
        /// The number of configurable settings in the suite.
        setting_count: u16,
        /// The ID assigned to this suite.
        suite_id: u16,
        /// The number of tests in the suite.
        test_count: u16,
    },
    /// Notification of a general target crash or panic.
    TargetPanic {
        /// The filename where the panic was triggered.
        file: &'a str,
        /// The line number where the panic was triggered.
        line: u32,
        /// The panic message description.
        message: &'a str,
    },
    /// Metadata about a test within a suite.
    TestInfo {
        /// The doc comment description of the test.
        description: &'a str,
        /// The name of the test.
        name: &'a str,
        /// The ID of the suite containing this test.
        suite_id: u16,
        /// The ID assigned to this test.
        test_id: u16,
    },
    /// Notification of a test state transition.
    TestStateChange {
        /// The new state of the test.
        state: TestState,
        /// The ID of the suite containing the test.
        suite_id: u16,
        /// The ID of the test.
        test_id: u16,
    },
    /// Wire-contract revision and target hardware metadata, sent first in
    /// every discovery stream. Appended last so its discriminant does not
    /// shift the existing variants.
    TargetInfo {
        /// The target's [`PROTOCOL_VERSION`].
        protocol_version: u8,
        /// Board identifier chosen by the profiler (`0` when unknown).
        board_id: u16,
        /// Core clock frequency in hertz (`0` when unknown).
        core_clock_hz: u32,
        /// FPU capability bits: bit 0 single precision, bit 1 double precision.
        fpu_flags: u8,
    },
    /// Marks a suite as a lifecycle suite, sent after the suite's records
    /// and before its [`Telemetry::TaskInfo`]. Suites without a task send none.
    LifecycleSuite {
        /// The ID of the suite.
        suite_id: u16,
        /// The suite's task count (always `1`).
        task_count: u8,
    },
    /// Metadata about the task of a suite.
    TaskInfo {
        /// The ID of the suite containing the task.
        suite_id: u16,
        /// The task's identifier within the suite.
        test_id: u16,
        /// The name of the task.
        name: &'a str,
        /// The doc comment description of the task.
        description: &'a str,
        /// The type name of the task's input packet.
        input_type: &'a str,
        /// The type name of the task's output packet.
        output_type: &'a str,
    },
    /// Notification of a task run state transition.
    TaskState {
        /// The ID of the suite containing the task.
        suite_id: u16,
        /// The task's identifier within the suite.
        test_id: u16,
        /// The new state of the run.
        state: TaskRunState,
        /// An optional message accompanying the state.
        message: Option<&'a str>,
    },
    /// The output packet of one task step.
    TaskSample {
        /// The ID of the suite containing the task.
        suite_id: u16,
        /// The task's identifier within the suite.
        test_id: u16,
        /// The step that produced the output.
        seq: u64,
        /// The `postcard` encoding of the task's output type.
        payload: &'a [u8],
    },
    /// The outcome of a task's teardown, independent of the run verdict.
    TeardownReport {
        /// The ID of the suite containing the task.
        suite_id: u16,
        /// The task's identifier within the suite.
        test_id: u16,
        /// Whether teardown succeeded.
        ok: bool,
        /// An optional message accompanying the result.
        message: Option<&'a str>,
    },
    /// Statistics of a finished task run.
    TaskStats {
        /// The ID of the suite containing the task.
        suite_id: u16,
        /// The task's identifier within the suite.
        test_id: u16,
        /// The number of steps called.
        steps: u64,
        /// Elapsed time from setup entry to teardown entry in microseconds.
        time_us: u64,
    },
}

/// The state of a task run. The order is append-only.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize,
)]
pub enum TaskRunState {
    /// The run is active, or the last step returned `Running`.
    Running,
    /// The last step returned `Warn`.
    Warn,
    /// A step returned `Pass`.
    Pass,
    /// A step returned `Fail`, or the run panicked.
    Fail,
    /// A step, setup or reset returned `Error`.
    Error,
    /// The host stopped the run.
    Aborted,
    /// The host link deadline passed.
    TimedOut,
    /// The run reached its step bound.
    Bounded,
}

/// The state of a test executable during a test session.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize,
)]
pub enum TestState {
    /// Test execution failed (for example, panic or assertion failure).
    Failed,
    /// Test completed successfully.
    Passed,
    /// Test is registered but has not run.
    Pending,
    /// Test is currently executing.
    Running,
}

/// A thread-safe lock using an atomic boolean flag to protect communication resources.
///
/// Prevents concurrent execution threads (such as main event loops and exception handlers)
/// from writing overlapping byte telemetry frames to the shared physical peripheral.
///
/// # Safety
/// This struct does not use `unsafe` code.
///
/// # Panics
/// Locking operations do not panic.
///
/// # Example
/// ```
/// use control_rs_ets::comms::CommsLock;
///
/// let lock = CommsLock::new();
/// assert!(lock.try_lock());
/// assert!(!lock.try_lock());
/// lock.unlock();
/// assert!(lock.try_lock());
/// ```
pub struct CommsLock {
    locked: AtomicBool,
}

/// State machine to de-frame a stream of incoming bytes into packets.
///
/// Statefully processes serial stream input byte-by-byte and performs
/// payload control-rs-verification via CRC-16 checks.
///
/// # Safety
/// This struct does not use `unsafe` code.
///
/// # Panics
/// Byte handling operations do not panic.
///
/// # Example
/// ```
/// use control_rs_ets::comms::FrameReader;
///
/// let mut reader = FrameReader::new();
/// assert!(reader.is_idle());
/// ```
pub struct FrameReader {
    payload_buffer: [u8; MAX_PAYLOAD_SIZE],
    state: ReaderState,
    temp_len: u16,
}

/// A log message produced by a test executable or the Server itself.
///
/// Bundles message payload text and ETS metadata tags together.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LogMessage<'a> {
    /// The log text payload.
    pub payload: &'a str,
    /// The ID of the test suite.
    pub suite_id: u16,
    /// The ID of the test executable.
    pub test_id: u16,
    /// Microseconds elapsed since boot / epoch.
    pub timestamp_us: u64,
}

/// Helper to serialize and frame payloads, commands, and telemetry messages into packet wire format.
///
/// # Wire Format
///
/// | Field          | Width    | Value                                 |
/// |:---------------|:---------|:--------------------------------------|
/// | Sync header    | 2 B      | `0xAA 0x55`                           |
/// | Payload length | 2 B      | big-endian (`u16`), payload ≤ 512 B   |
/// | Payload        | variable | `postcard` serialized payload         |
/// | Checksum       | 2 B      | big-endian CRC-16 (`CRC_16_IBM_SDLC`) |
pub struct FrameEncoder;

/// A [`FrameReader`] that keeps the bytes of a chunked transport read which
/// follow the first complete frame, so no command is lost when one read holds
/// more than one frame.
///
/// # Safety
/// This struct does not use `unsafe` code.
///
/// # Panics
/// Polling does not panic.
///
/// # Example
/// ```
/// use control_rs_ets::comms::{BufferedFrameReader, Command, FrameEncoder};
///
/// let mut wire = [0u8; 64];
/// let a = FrameEncoder::frame_command(&Command::Heartbeat, &mut wire).unwrap();
/// let b = FrameEncoder::frame_command(&Command::TryReset, &mut wire[a..]).unwrap();
/// let chunk = wire;
///
/// let mut rx = BufferedFrameReader::<64>::new();
/// let first = rx.poll(|buf| { buf.copy_from_slice(&chunk); Ok::<_, ()>(a + b) });
/// assert!(matches!(first, Ok(Some(Command::Heartbeat))));
/// let second = rx.poll(|_| Ok::<_, ()>(0));
/// assert!(matches!(second, Ok(Some(Command::TryReset))));
/// ```
pub struct BufferedFrameReader<const N: usize> {
    buf: [u8; N],
    len: usize,
    pos: usize,
    reader: FrameReader,
}

impl Default for CommsLock {
    fn default() -> Self {
        Self::new()
    }
}

impl CommsLock {
    /// Creates a new, unlocked `CommsLock`.
    ///
    /// # Returns
    /// * `Self` - An unlocked `CommsLock` instance.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            locked: AtomicBool::new(false),
        }
    }

    /// Attempts to acquire the lock. Returns `true` if successful or `false` if already locked.
    ///
    /// # Returns
    /// * `bool` - `true` if lock was acquired successfully, `false` otherwise.
    #[must_use]
    pub fn try_lock(&self) -> bool {
        self.locked
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
    }

    /// Releases the lock.
    pub fn unlock(&self) {
        self.locked.store(false, Ordering::Release);
    }
}

impl Default for FrameReader {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameReader {
    /// Process a single incoming byte. Returns `Some(&[u8])` when a complete,
    /// checksum-verified payload has been received.
    ///
    /// # Arguments
    /// * `byte` - Incoming serial stream byte.
    ///
    /// # Returns
    /// * `Option<DecodedFrame<'_>>`
    ///     * `Some(frame_payload)` - A byte slice reference to the verified frame payload buffer.
    ///     * `None` - If the frame is still incomplete or the CRC check failed.
    pub fn handle_byte(&mut self, byte: u8) -> Option<DecodedFrame<'_>> {
        match self.state {
            ReaderState::WaitStart1 => {
                if byte == START_BYTE_1 {
                    self.state = ReaderState::WaitStart2;
                }
            }
            ReaderState::WaitStart2 => {
                if byte == START_BYTE_2 {
                    self.state = ReaderState::WaitLen1;
                } else if byte == START_BYTE_1 {
                    // Orphan/noise 0xAA left us in WaitStart2; this 0xAA may
                    // start a real frame, so stay armed for 0x55.
                } else {
                    self.state = ReaderState::WaitStart1;
                }
            }
            ReaderState::WaitLen1 => {
                self.temp_len = u16::from(byte) << 8;
                self.state = ReaderState::WaitLen2;
            }
            ReaderState::WaitLen2 => {
                self.temp_len |= u16::from(byte);
                let len = self.temp_len as usize;
                if len == 0 || len > MAX_PAYLOAD_SIZE {
                    // Invalid length, reset state machine
                    self.state = ReaderState::WaitStart1;
                } else {
                    self.state = ReaderState::ReadingPayload { len, read: 0 };
                }
            }
            ReaderState::ReadingPayload { len, ref mut read } => {
                if let Some(slot) = self.payload_buffer.get_mut(*read) {
                    *slot = byte;
                }
                *read = (*read).saturating_add(1);
                if *read == len {
                    self.state = ReaderState::WaitChecksum1 { len };
                }
            }
            ReaderState::WaitChecksum1 { len } => {
                self.state = ReaderState::WaitChecksum2 { len, crc1: byte };
            }
            ReaderState::WaitChecksum2 { len, crc1 } => {
                let crc_value = (u16::from(crc1) << 8) | u16::from(byte);
                let crc = crc::Crc::<u16>::new(&crc::CRC_16_IBM_SDLC);
                let calculated =
                    crc.checksum(self.payload_buffer.get(..len).unwrap_or(&[]));

                self.state = ReaderState::WaitStart1;
                if calculated == crc_value {
                    return self.payload_buffer.get(..len);
                }
            }
        }
        None
    }

    /// The payload of the most recently completed frame, until the next
    /// frame's payload bytes arrive.
    fn completed_payload(&self, len: usize) -> &[u8] {
        self.payload_buffer.get(..len).unwrap_or(&[])
    }

    /// Returns true if the reader is currently idle (waiting for a new frame start).
    ///
    /// # Returns
    /// * `bool` - `true` if the frame reader state is in wait-for-start mode.
    #[must_use]
    pub const fn is_idle(&self) -> bool {
        matches!(self.state, ReaderState::WaitStart1)
    }

    /// Creates a new `FrameReader`.
    ///
    /// # Returns
    /// * `Self` - Initialized `FrameReader` instance.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            payload_buffer: [0u8; MAX_PAYLOAD_SIZE],
            state: ReaderState::WaitStart1,
            temp_len: 0,
        }
    }
}

impl<const N: usize> Default for BufferedFrameReader<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> BufferedFrameReader<N> {
    /// Creates an empty reader.
    ///
    /// # Returns
    /// * `Self` - A reader holding no bytes.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            buf: [0u8; N],
            len: 0,
            pos: 0,
            reader: FrameReader::new(),
        }
    }

    /// Feeds held bytes first, stopping after the byte that completes a frame.
    ///
    /// `read` is called only when no bytes are held, at most once per poll, and
    /// returns the number of bytes it wrote into the slice. A poll processes at
    /// most `N` bytes. A CRC-valid frame that does not decode as a [`Command`]
    /// is dropped and the poll returns `Ok(None)`.
    ///
    /// # Arguments
    /// * `read` - Fills the slice from the transport and returns the byte count.
    ///
    /// # Returns
    /// * `PollResult<'_, E>` - The next command, `None` when the held bytes and
    ///   the read complete no frame, or the transport error.
    ///
    /// # Errors
    /// Returns the transport error produced by `read`.
    pub fn poll<E>(
        &mut self,
        read: impl FnOnce(&mut [u8]) -> Result<usize, E>,
    ) -> PollResult<'_, E> {
        if self.pos >= self.len {
            self.pos = 0;
            self.len = 0;
            self.len = read(&mut self.buf)?.min(N);
        }

        let mut frame_len = None;
        while let Some(&byte) =
            self.buf.get(self.pos).filter(|_| self.pos < self.len)
        {
            self.pos = self.pos.saturating_add(1);
            if let Some(payload) = self.reader.handle_byte(byte) {
                frame_len = Some(payload.len());
                break;
            }
        }

        Ok(frame_len.and_then(|len| {
            postcard::from_bytes(self.reader.completed_payload(len)).ok()
        }))
    }
}

impl FrameEncoder {
    /// Writes sync, length, and CRC for a payload already sitting at `dest[4..]`.
    fn finish_frame(
        dest: &mut [u8],
        payload_len: usize,
    ) -> Result<usize, postcard::Error> {
        if payload_len > MAX_PAYLOAD_SIZE {
            return Err(postcard::Error::SerializeBufferFull);
        }
        let total_len = payload_len
            .checked_add(FRAME_OVERHEAD)
            .ok_or(postcard::Error::SerializeBufferFull)?;
        if dest.len() < total_len {
            return Err(postcard::Error::SerializeBufferFull);
        }

        if let Some(slot) = dest.get_mut(0) {
            *slot = START_BYTE_1;
        }
        if let Some(slot) = dest.get_mut(1) {
            *slot = START_BYTE_2;
        }

        let len_u16 = u16::try_from(payload_len)
            .map_err(|_| postcard::Error::SerializeBufferFull)?;
        if let Some(slot) = dest.get_mut(2) {
            *slot = (len_u16 >> 8) as u8;
        }
        if let Some(slot) = dest.get_mut(3) {
            *slot = (len_u16 & 0xFF) as u8;
        }

        let crc_value = {
            let payload = dest
                .get(4..payload_len.saturating_add(4))
                .ok_or(postcard::Error::SerializeBufferFull)?;
            crc::Crc::<u16>::new(&crc::CRC_16_IBM_SDLC).checksum(payload)
        };
        if let Some(slot) = dest.get_mut(payload_len.saturating_add(4)) {
            *slot = (crc_value >> 8) as u8;
        }
        if let Some(slot) =
            dest.get_mut(payload_len.saturating_add(4).saturating_add(1))
        {
            *slot = (crc_value & 0xFF) as u8;
        }
        Ok(total_len)
    }

    /// Serializes `value` into `dest[4..]` then writes the frame header and CRC.
    fn serialize_then_frame<T: serde::Serialize>(
        value: &T,
        dest: &mut [u8],
    ) -> Result<usize, postcard::Error> {
        if dest.len() < FRAME_OVERHEAD {
            return Err(postcard::Error::SerializeBufferFull);
        }
        let max_payload = dest
            .len()
            .saturating_sub(FRAME_OVERHEAD)
            .min(MAX_PAYLOAD_SIZE);
        let payload_len = {
            let end = 4usize
                .checked_add(max_payload)
                .ok_or(postcard::Error::SerializeBufferFull)?;
            let slice = dest
                .get_mut(4..end)
                .ok_or(postcard::Error::SerializeBufferFull)?;
            postcard::to_slice(value, slice)?.len()
        };
        Self::finish_frame(dest, payload_len)
    }

    /// Frames a raw payload byte slice into the destination buffer.
    ///
    /// # Errors
    /// Returns `postcard::Error::SerializeBufferFull` if `dest` is too small or `payload` exceeds `MAX_PAYLOAD_SIZE`.
    pub fn frame_payload(
        payload: &[u8],
        dest: &mut [u8],
    ) -> Result<usize, postcard::Error> {
        let payload_len = payload.len();
        if payload_len > MAX_PAYLOAD_SIZE {
            return Err(postcard::Error::SerializeBufferFull);
        }
        let total_len = payload_len
            .checked_add(FRAME_OVERHEAD)
            .ok_or(postcard::Error::SerializeBufferFull)?;
        if dest.len() < total_len {
            return Err(postcard::Error::SerializeBufferFull);
        }
        let end = 4usize
            .checked_add(payload_len)
            .ok_or(postcard::Error::SerializeBufferFull)?;
        if let Some(slice) = dest.get_mut(4..end) {
            slice.copy_from_slice(payload);
        }
        Self::finish_frame(dest, payload_len)
    }

    /// Serializes and frames a [`Command`] message into the destination buffer.
    ///
    /// # Errors
    /// Returns `postcard::Error` if serialization fails or `dest` is too small.
    pub fn frame_command(
        cmd: &Command<'_>,
        dest: &mut [u8],
    ) -> Result<usize, postcard::Error> {
        Self::serialize_then_frame(cmd, dest)
    }

    /// Serializes and frames a [`Telemetry`] message into the destination buffer.
    ///
    /// # Errors
    /// Returns `postcard::Error` if serialization fails or `dest` is too small.
    pub fn frame_telemetry(
        telemetry: &Telemetry<'_>,
        dest: &mut [u8],
    ) -> Result<usize, postcard::Error> {
        Self::serialize_then_frame(telemetry, dest)
    }
}

/// Helper to serialize and frame a telemetry message into a destination buffer.
///
/// Forwards to [`FrameEncoder::frame_telemetry`].
///
/// # Errors
/// Returns `postcard::Error` if serialization fails or destination buffer is too small.
#[inline]
pub fn frame_telemetry(
    telemetry: &Telemetry<'_>,
    dest: &mut [u8],
) -> Result<usize, postcard::Error> {
    FrameEncoder::frame_telemetry(telemetry, dest)
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;

    /// A decoded payload copied into a fixed buffer, with its length.
    type DecodedPayload<const N: usize> = ([u8; N], usize);

    /// Scratch frame buffer with a little slack beyond the largest frame.
    type FrameBuf = [u8; MAX_FRAME_SIZE + 8];

    /// A framed buffer and the frame length, or the framing error.
    type Framed = Result<(FrameBuf, usize), postcard::Error>;

    /// A command and its checked-in postcard bytes.
    type GoldenCommand = (Command<'static>, &'static [u8]);

    /// A telemetry frame and its checked-in postcard bytes.
    type GoldenTelemetry = (Telemetry<'static>, &'static [u8]);

    /// A command and the tag byte it encodes with.
    type TaggedCommand = (Command<'static>, u8);

    /// A telemetry frame and the tag byte it encodes with.
    type TaggedTelemetry = (Telemetry<'static>, u8);

    /// Commands framed back to back and the number of bytes used.
    type Wire = ([u8; 256], usize);

    /// Returns `buf[range]`, failing the test when the range leaves the
    /// buffer.
    fn span(buf: &[u8], range: core::ops::Range<usize>) -> &[u8] {
        buf.get(range)
            .expect("range must lie inside the frame buffer")
    }

    /// Returns the byte at `index`, failing the test when it is out of range.
    fn byte_at(buf: &[u8], index: usize) -> u8 {
        buf.get(index)
            .copied()
            .expect("index must lie inside the frame buffer")
    }

    /// Big-endian payload length carried in bytes 2 and 3 of a frame.
    fn header_payload_len(buf: &[u8]) -> usize {
        usize::from(
            u16::from(byte_at(buf, 2)) << 8 | u16::from(byte_at(buf, 3)),
        )
    }

    /// Feeds `frame` through a fresh `FrameReader` and reports whether any
    /// byte completed a frame.
    fn frame_decodes(frame: &[u8]) -> bool {
        let mut reader = FrameReader::new();
        frame.iter().any(|&b| reader.handle_byte(b).is_some())
    }

    /// Feeds `frame` through a fresh `FrameReader` and returns a copy of the
    /// first payload it yields together with the payload length.
    fn first_payload<const N: usize>(
        frame: &[u8],
    ) -> Option<DecodedPayload<N>> {
        let mut reader = FrameReader::new();
        for &b in frame {
            if let Some(payload) = reader.handle_byte(b) {
                let mut out = [0u8; N];
                out.get_mut(..payload.len())?.copy_from_slice(payload);
                return Some((out, payload.len()));
            }
        }
        None
    }

    /// Frames `cmd`, decodes it back through `FrameReader` and passes the
    /// deserialized command to `check`.
    fn command_round_trip<R>(
        cmd: &Command<'_>,
        check: impl FnOnce(Command<'_>) -> R,
    ) -> R {
        let mut buf = [0u8; 32];
        let len = FrameEncoder::frame_command(cmd, &mut buf)
            .expect("command framing");
        let (payload, payload_len) = first_payload::<32>(span(&buf, 0..len))
            .expect("a framed command must decode");
        check(
            postcard::from_bytes(span(&payload, 0..payload_len))
                .expect("postcard decode command"),
        )
    }

    #[test]
    fn test_frame_reader_idle() {
        let reader = FrameReader::new();
        assert!(reader.is_idle());
    }

    #[test]
    fn test_frame_reader_default() {
        let reader = FrameReader::default();
        assert!(reader.is_idle());
    }

    #[test]
    fn test_frame_reader_invalid_start_bytes() {
        let mut reader = FrameReader::new();
        // Send a byte that is not START_BYTE_1
        assert!(reader.handle_byte(0x00).is_none());
        assert!(reader.is_idle());

        // Send START_BYTE_1 followed by not START_BYTE_2
        assert!(reader.handle_byte(START_BYTE_1).is_none());
        assert!(!reader.is_idle()); // now in WaitStart2
        assert!(reader.handle_byte(0x00).is_none());
        assert!(reader.is_idle()); // reset to WaitStart1
    }

    #[test]
    fn test_frame_reader_resync_after_orphan_start_byte() {
        let mut buf = [0u8; 128];
        let framed_len =
            frame_telemetry(&Telemetry::DiscoveryComplete, &mut buf)
                .expect("framing");

        let mut reader = FrameReader::new();
        // Noise/orphan 0xAA must not consume the real frame's leading 0xAA.
        assert!(reader.handle_byte(START_BYTE_1).is_none());
        let decoded = span(&buf, 0..framed_len)
            .iter()
            .any(|&b| reader.handle_byte(b).is_some());
        assert!(
            decoded,
            "valid frame after an orphan 0xAA must still decode"
        );
    }

    #[test]
    fn test_frame_reader_invalid_lengths() {
        let mut reader = FrameReader::new();
        // Move to WaitLen1
        assert!(reader.handle_byte(START_BYTE_1).is_none());
        assert!(reader.handle_byte(START_BYTE_2).is_none());

        // Send length = 0 (MSB=0, LSB=0)
        assert!(reader.handle_byte(0x00).is_none());
        assert!(reader.handle_byte(0x00).is_none());
        assert!(reader.is_idle()); // should reset since len = 0 is invalid

        // Send length > MAX_PAYLOAD_SIZE (for example, 513 = MSB=2, LSB=1)
        assert!(reader.handle_byte(START_BYTE_1).is_none());
        assert!(reader.handle_byte(START_BYTE_2).is_none());
        assert!(reader.handle_byte(0x02).is_none());
        assert!(reader.handle_byte(0x01).is_none());
        assert!(reader.is_idle()); // should reset since len > MAX_PAYLOAD_SIZE
    }

    #[test]
    fn test_frame_reader_valid_frame() {
        let mut reader = FrameReader::new();
        let telemetry = Telemetry::DiscoveryComplete;
        let mut buf = [0u8; 128];
        let framed_len = frame_telemetry(&telemetry, &mut buf).unwrap();

        // Feed all bytes except the last one (checksum)
        for &item in buf.iter().take(framed_len - 1) {
            assert!(reader.handle_byte(item).is_none());
        }
        // Feed the checksum, it should complete and return the payload slice
        let payload = reader
            .handle_byte(buf.get(framed_len - 1).copied().unwrap())
            .unwrap();

        // Deserialize and check
        let decoded: Telemetry<'_> = postcard::from_bytes(payload).unwrap();
        assert!(matches!(decoded, Telemetry::DiscoveryComplete));
        assert!(reader.is_idle());
    }

    #[test]
    fn test_frame_reader_invalid_checksum() {
        let mut reader = FrameReader::new();
        let telemetry = Telemetry::DiscoveryComplete;
        let mut buf = [0u8; 128];
        let framed_len = frame_telemetry(&telemetry, &mut buf).unwrap();

        // Feed all bytes except the 2 checksum bytes
        for &item in buf.iter().take(framed_len - 2) {
            assert!(reader.handle_byte(item).is_none());
        }
        // Feed an invalid first checksum byte
        let bad_checksum = buf.get(framed_len - 2).copied().unwrap() ^ 0xFF;
        assert!(reader.handle_byte(bad_checksum).is_none());
        // Feed the second checksum byte
        assert!(
            reader
                .handle_byte(buf.get(framed_len - 1).copied().unwrap())
                .is_none()
        );
        assert!(reader.is_idle()); // reset
    }

    #[test]
    fn test_frame_telemetry_too_small_buffer() {
        let telemetry = Telemetry::DiscoveryComplete;
        {
            let mut buf = [0u8; 5];
            let res = frame_telemetry(&telemetry, &mut buf);
            assert!(matches!(res, Err(postcard::Error::SerializeBufferFull)));
        }

        // Buffer large enough for header but too small for payload
        {
            let mut buf = [0u8; 6];
            let res = frame_telemetry(&telemetry, &mut buf);
            assert!(res.is_err());
        }
    }

    #[test]
    fn test_comms_lock_default() {
        let lock = CommsLock::default();
        assert!(lock.try_lock());
    }

    #[test]
    fn test_host_comms_defaults() {
        struct TestComms;
        impl HostComms for TestComms {
            type Error = ();
            fn flush(&mut self) -> Result<(), Self::Error> {
                Ok(())
            }
            fn poll_command(
                &mut self,
            ) -> Result<Option<Command<'static>>, Self::Error> {
                Ok(None)
            }
            fn send_telemetry(
                &mut self,
                _telemetry: &Telemetry<'_>,
            ) -> Result<(), Self::Error> {
                Ok(())
            }
        }
        let mut comms = TestComms;
        comms.close();
        comms.close_on_failure();
    }

    /// Byte-level layout of a framed telemetry packet:
    /// `[0xAA, 0x55, len_hi, len_lo, payload.., crc_hi, crc_lo]`, with the
    /// CRC-16/IBM-SDLC taken over the payload only.
    ///
    /// Every field is asserted against an independently computed value, so a
    /// swapped header byte, a little-endian length, a CRC over the wrong span
    /// or an off-by-one placement all fail here.
    #[test]
    fn test_frame_telemetry_byte_layout() {
        let mut buf = [0u8; 128];
        let n = frame_telemetry(&Telemetry::DiscoveryComplete, &mut buf)
            .expect("framing must succeed into a 128 byte buffer");

        // Header.
        assert_eq!(buf[0], 0xAA);
        assert_eq!(buf[1], 0x55);

        // Length is big-endian and counts the payload only.
        let payload_len = header_payload_len(&buf);
        assert_eq!(n, payload_len + 6);

        // The payload is exactly what postcard produces on its own.
        let mut direct = [0u8; 64];
        let encoded =
            postcard::to_slice(&Telemetry::DiscoveryComplete, &mut direct)
                .expect("payload must serialize");
        assert_eq!(payload_len, encoded.len());
        assert_eq!(span(&buf, 4..4 + payload_len), encoded);

        // CRC over the payload, big-endian, in the last two bytes.
        let crc = crc::Crc::<u16>::new(&crc::CRC_16_IBM_SDLC);
        let want = crc.checksum(encoded);
        let got = u16::from(byte_at(&buf, 4 + payload_len)) << 8
            | u16::from(byte_at(&buf, 5 + payload_len));
        assert_eq!(got, want);

        // Nothing is written past the frame.
        assert!(span(&buf, n..buf.len()).iter().all(|&b| b == 0));
    }

    /// A buffer smaller than the six framing bytes is rejected before
    /// anything is written, and the boundary is exactly six.
    #[test]
    fn test_frame_telemetry_rejects_short_buffers() {
        for len in 0..6usize {
            let mut small = [0u8; 8];
            let target = small
                .get_mut(..len)
                .expect("len must lie inside the scratch buffer");
            let res = frame_telemetry(&Telemetry::DiscoveryComplete, target);
            assert!(res.is_err(), "len {len} must be rejected");
            assert!(
                small.iter().all(|&b| b == 0),
                "len {len} must not write into the buffer"
            );
        }

        // A buffer that clears the header check but cannot hold the payload
        // still fails rather than truncating.
        let mut tight = [0u8; 6];
        assert!(
            frame_telemetry(&Telemetry::DiscoveryComplete, &mut tight).is_err()
        );
    }

    /// A framed packet decodes back through `FrameReader`, and corrupting any
    /// single byte of the payload or the CRC makes the reader reject it.
    #[test]
    fn test_frame_telemetry_round_trip_and_corruption() {
        let mut buf = [0u8; 128];
        let n = frame_telemetry(&Telemetry::DiscoveryComplete, &mut buf)
            .expect("framing must succeed");

        assert!(frame_decodes(span(&buf, 0..n)), "a clean frame must decode");

        // Flipping a bit anywhere after the header must break the frame.
        for corrupt_at in 4..n {
            let mut bad = buf;
            let target = bad
                .get_mut(corrupt_at)
                .expect("corrupt_at must lie inside the frame");
            *target ^= 0xFF;
            assert!(
                !frame_decodes(span(&bad, 0..n)),
                "corrupting byte {corrupt_at} must fail the CRC check"
            );
        }
    }

    /// The returned length is `6 + payload_len` for every telemetry variant,
    /// and the frame always starts with the two header bytes.
    #[test]
    fn test_frame_telemetry_length_accounting_across_variants() {
        let variants = [
            Telemetry::DiscoveryComplete,
            Telemetry::MetricReport {
                cycles: 123_456,
                stack_peak: 2048,
                suite_id: 7,
                test_id: 9,
                time_us: 654_321,
            },
        ];
        for v in &variants {
            let mut buf = [0u8; 128];
            let n = frame_telemetry(v, &mut buf).expect("framing");
            assert_eq!(n, header_payload_len(&buf) + 6);
            assert_eq!(buf[0], 0xAA);
            assert_eq!(buf[1], 0x55);
            assert!(n >= 6);
        }
    }

    /// `CommsLock` is a single-holder flag: the first `try_lock` wins, every
    /// later one fails until `unlock`, and `unlock` is idempotent.
    #[test]
    fn test_comms_lock_excludes_second_holder() {
        let lock = CommsLock::new();

        assert!(lock.try_lock(), "an unlocked lock must be acquirable");
        for _ in 0..4 {
            assert!(!lock.try_lock(), "a held lock must stay held");
        }

        lock.unlock();
        assert!(lock.try_lock(), "unlock must release the flag");

        lock.unlock();
        lock.unlock();
        assert!(lock.try_lock(), "a repeated unlock must not wedge the lock");
        lock.unlock();
    }

    /// A fresh lock starts unlocked, and `Default` agrees with `new`.
    #[test]
    fn test_comms_lock_starts_unlocked() {
        let a = CommsLock::new();
        assert!(a.try_lock());

        let b = CommsLock::default();
        assert!(b.try_lock());
        assert!(!b.try_lock());
    }

    #[test]
    fn test_frame_encoder_payload_and_command() {
        let payload = [0x12, 0x34, 0x56, 0x78];
        let mut buf = [0u8; 32];
        let len = FrameEncoder::frame_payload(&payload, &mut buf)
            .expect("payload framing");
        assert_eq!(len, 10);
        assert_eq!(buf[0], 0xAA);
        assert_eq!(buf[1], 0x55);
        assert_eq!(buf[2], 0x00);
        assert_eq!(buf[3], 0x04);
        assert_eq!(&buf[4..8], &payload);

        let (decoded, decoded_len) = first_payload::<32>(span(&buf, 0..len))
            .expect("a framed payload must decode");
        assert_eq!(span(&decoded, 0..decoded_len), &payload);

        command_round_trip(
            &Command::RunExecutable {
                suite_id: 1,
                test_id: 2,
            },
            |decoded_cmd| match decoded_cmd {
                Command::RunExecutable { suite_id, test_id } => {
                    assert_eq!(suite_id, 1);
                    assert_eq!(test_id, 2);
                }
                _ => panic!("unexpected command variant"),
            },
        );
    }

    #[test]
    fn test_golden_wire_vector_list_suites() {
        let mut buf = [0u8; 32];
        let len = FrameEncoder::frame_command(&Command::ListSuites, &mut buf)
            .expect("framing ListSuites");
        let crc = crc::Crc::<u16>::new(&crc::CRC_16_IBM_SDLC);
        let exp_crc = crc.checksum(&[0x00]);
        let crc_hi = u8::try_from(exp_crc >> 8).expect("high byte fits in u8");
        let crc_lo = u8::try_from(exp_crc & 0xFF).expect("low byte fits in u8");
        let expected = [0xAA, 0x55, 0x00, 0x01, 0x00, crc_hi, crc_lo];
        assert_eq!(span(&buf, 0..len), &expected);
    }

    #[test]
    fn test_golden_wire_vector_run_executable() {
        command_round_trip(
            &Command::RunExecutable {
                suite_id: 1,
                test_id: 2,
            },
            |decoded| {
                assert!(matches!(
                    decoded,
                    Command::RunExecutable {
                        suite_id: 1,
                        test_id: 2
                    }
                ));
            },
        );
    }

    #[test]
    fn test_golden_wire_vector_try_reset() {
        command_round_trip(&Command::TryReset, |decoded| {
            assert!(matches!(decoded, Command::TryReset));
        });
    }

    /// One instance of every telemetry variant, with payload fields chosen
    /// to exercise each string and integer encoding.
    fn golden_telemetry_variants() -> [Telemetry<'static>; 8] {
        [
            Telemetry::DiscoveryComplete,
            Telemetry::Log(LogMessage {
                payload: "test log",
                suite_id: 1,
                test_id: 2,
                timestamp_us: 1000,
            }),
            Telemetry::MetricReport {
                cycles: 5000,
                stack_peak: 256,
                suite_id: 1,
                test_id: 2,
                time_us: 420,
            },
            Telemetry::SettingInfo {
                description: "Setting desc",
                name: "baud_rate",
                setting_id: 1,
                suite_id: 2,
                value: SettingValue::U32(115_200),
            },
            Telemetry::SuiteInfo {
                description: "Suite desc",
                name: "TestSuite1",
                setting_count: 1,
                suite_id: 0,
                test_count: 4,
            },
            Telemetry::TargetPanic {
                file: "src/main.rs",
                line: 50,
                message: "Target panic test",
            },
            Telemetry::TestInfo {
                description: "Test desc",
                name: "test_addition",
                suite_id: 0,
                test_id: 1,
            },
            Telemetry::TargetInfo {
                protocol_version: PROTOCOL_VERSION,
                board_id: 0x0401,
                core_clock_hz: 600_000_000,
                fpu_flags: 1,
            },
        ]
    }

    /// Checked-in postcard bytes for `TargetInfo`: variant index 8, then
    /// the version byte, then varint `board_id`, `core_clock_hz` and the
    /// flags byte. A change here is a wire break and must bump
    /// `PROTOCOL_VERSION`.
    #[test]
    fn test_golden_wire_vector_target_info() {
        let info = Telemetry::TargetInfo {
            protocol_version: 1,
            board_id: 0x0401,
            core_clock_hz: 600_000_000,
            fpu_flags: 1,
        };
        let mut buf = [0u8; 32];
        let bytes = postcard::to_slice(&info, &mut buf).expect("encode");
        assert_eq!(
            bytes,
            &[0x08, 0x01, 0x81, 0x08, 0x80, 0x8C, 0x8D, 0x9E, 0x02, 0x01]
        );
    }

    #[test]
    fn test_golden_wire_vectors_telemetry() {
        for t in &golden_telemetry_variants() {
            let mut buf = [0u8; 256];
            let len = FrameEncoder::frame_telemetry(t, &mut buf)
                .expect("telemetry framing");
            assert_eq!(buf[0], 0xAA);
            assert_eq!(buf[1], 0x55);
            assert_eq!(len, header_payload_len(&buf) + 6);

            let (decoded, decoded_len) =
                first_payload::<256>(span(&buf, 0..len))
                    .expect("a framed telemetry packet must decode");
            let _dec_t: Telemetry<'_> =
                postcard::from_bytes(span(&decoded, 0..decoded_len))
                    .expect("postcard decode telemetry");
        }
    }

    /// Frames `payload` into a buffer of exactly `dest_len` bytes.
    fn frame_into(payload: &[u8], dest_len: usize) -> Framed {
        let mut dest = [0u8; MAX_FRAME_SIZE + 8];
        let target = dest
            .get_mut(..dest_len)
            .expect("dest_len must lie inside the scratch buffer");
        let len = FrameEncoder::frame_payload(payload, target)?;
        Ok((dest, len))
    }

    #[test]
    fn a_maximum_size_payload_frames_and_decodes() {
        let payload = [0x5A_u8; MAX_PAYLOAD_SIZE];
        let (frame, len) =
            frame_into(&payload, MAX_FRAME_SIZE).expect("a full payload fits");
        assert_eq!(len, MAX_FRAME_SIZE);
        assert_eq!(header_payload_len(&frame), MAX_PAYLOAD_SIZE);
        let mut reader = FrameReader::new();
        let decoded = span(&frame, 0..len)
            .iter()
            .find_map(|&b| reader.handle_byte(b).map(<[u8]>::len));
        assert_eq!(decoded, Some(MAX_PAYLOAD_SIZE));
    }

    #[test]
    fn an_oversized_payload_is_refused() {
        let payload = [0u8; MAX_PAYLOAD_SIZE + 1];
        assert!(frame_into(&payload, MAX_FRAME_SIZE + 8).is_err());
    }

    #[test]
    fn framing_needs_room_for_the_whole_frame() {
        let payload = [7u8; 10];
        let total = payload.len() + FRAME_OVERHEAD;
        let (frame, len) = frame_into(&payload, total).expect("an exact fit");
        assert_eq!(len, total);
        assert!(frame_decodes(span(&frame, 0..len)));
        assert!(matches!(
            frame_into(&payload, total - 1),
            Err(postcard::Error::SerializeBufferFull)
        ));
    }

    #[test]
    fn the_length_field_is_big_endian_across_both_bytes() {
        let payload = [1u8; 300];
        let (frame, len) =
            frame_into(&payload, MAX_FRAME_SIZE).expect("framing");
        assert_eq!(byte_at(&frame, 2), 1, "high byte of 300");
        assert_eq!(byte_at(&frame, 3), 44, "low byte of 300");
        assert!(frame_decodes(span(&frame, 0..len)));
    }

    /// Postcard bytes of every pre-existing variant at protocol revision 1.
    #[test]
    fn existing_variant_encoding_unchanged() {
        let mut buf = [0u8; 64];
        for (cmd, expected) in golden_commands() {
            assert_eq!(postcard::to_slice(&cmd, &mut buf).unwrap(), expected);
        }
        let telemetry =
            golden_telemetry_a().into_iter().chain(golden_telemetry_b());
        for (t, expected) in telemetry {
            assert_eq!(postcard::to_slice(&t, &mut buf).unwrap(), expected);
        }

        // Task variants are appended after the last pre-existing one.
        for (cmd, tag) in appended_command_tags() {
            let bytes = postcard::to_slice(&cmd, &mut buf).unwrap();
            assert_eq!(bytes.first(), Some(&tag));
        }
        for (t, tag) in appended_telemetry_tags() {
            let bytes = postcard::to_slice(&t, &mut buf).unwrap();
            assert_eq!(bytes.first(), Some(&tag));
        }

        // `TaskRunState` encodes as its declared-order discriminant.
        for (i, state) in run_states().iter().enumerate() {
            let bytes = postcard::to_slice(state, &mut buf).unwrap();
            assert_eq!(bytes, [u8::try_from(i).unwrap()]);
        }
    }

    fn golden_commands() -> [GoldenCommand; 4] {
        [
            (Command::ListSuites, &[0x00]),
            (
                Command::RunExecutable {
                    suite_id: 1,
                    test_id: 2,
                },
                &[0x01, 0x01, 0x02],
            ),
            (
                Command::SetSetting {
                    setting_id: 3,
                    suite_id: 4,
                    value: SettingValue::U32(115_200),
                },
                &[0x02, 0x03, 0x04, 0x05, 0x80, 0x84, 0x07],
            ),
            (Command::TryReset, &[0x03]),
        ]
    }

    fn golden_telemetry_a() -> [GoldenTelemetry; 5] {
        let log = LogMessage {
            payload: "log",
            suite_id: 1,
            test_id: 2,
            timestamp_us: 1000,
        };
        let setting = Telemetry::SettingInfo {
            description: "d",
            name: "n",
            setting_id: 1,
            suite_id: 2,
            value: SettingValue::U32(115_200),
        };
        [
            (Telemetry::DiscoveryComplete, &[0x00]),
            (
                Telemetry::Log(log),
                &[0x01, 0x03, 0x6C, 0x6F, 0x67, 0x01, 0x02, 0xE8, 0x07],
            ),
            (
                Telemetry::MetricReport {
                    cycles: 5000,
                    stack_peak: 256,
                    suite_id: 1,
                    test_id: 2,
                    time_us: 420,
                },
                &[0x02, 0x88, 0x27, 0x80, 0x02, 0x01, 0x02, 0xA4, 0x03],
            ),
            (
                setting,
                &[
                    0x03, 0x01, 0x64, 0x01, 0x6E, 0x01, 0x02, 0x05, 0x80, 0x84,
                    0x07,
                ],
            ),
            (
                Telemetry::SuiteInfo {
                    description: "d",
                    name: "n",
                    setting_count: 1,
                    suite_id: 0,
                    test_count: 4,
                },
                &[0x04, 0x01, 0x64, 0x01, 0x6E, 0x01, 0x00, 0x04],
            ),
        ]
    }

    fn golden_telemetry_b() -> [GoldenTelemetry; 4] {
        [
            (
                Telemetry::TargetPanic {
                    file: "f",
                    line: 50,
                    message: "m",
                },
                &[0x05, 0x01, 0x66, 0x32, 0x01, 0x6D],
            ),
            (
                Telemetry::TestInfo {
                    description: "d",
                    name: "n",
                    suite_id: 0,
                    test_id: 1,
                },
                &[0x06, 0x01, 0x64, 0x01, 0x6E, 0x00, 0x01],
            ),
            (
                Telemetry::TestStateChange {
                    state: TestState::Failed,
                    suite_id: 1,
                    test_id: 2,
                },
                &[0x07, 0x00, 0x01, 0x02],
            ),
            (
                Telemetry::TargetInfo {
                    protocol_version: 1,
                    board_id: 0x0401,
                    core_clock_hz: 600_000_000,
                    fpu_flags: 1,
                },
                &[0x08, 0x01, 0x81, 0x08, 0x80, 0x8C, 0x8D, 0x9E, 0x02, 0x01],
            ),
        ]
    }

    fn appended_command_tags() -> [TaggedCommand; 4] {
        [
            (
                Command::StartTask {
                    suite_id: 0,
                    test_id: 0,
                    max_steps: 0,
                    lockstep: false,
                },
                4,
            ),
            (
                Command::StopNow {
                    suite_id: 0,
                    test_id: 0,
                },
                5,
            ),
            (Command::Heartbeat, 6),
            (
                Command::TaskInput {
                    suite_id: 0,
                    test_id: 0,
                    seq: 0,
                    payload: &[],
                },
                7,
            ),
        ]
    }

    fn appended_telemetry_tags() -> [TaggedTelemetry; 6] {
        let (suite_id, test_id) = (0, 0);
        [
            (
                Telemetry::LifecycleSuite {
                    suite_id,
                    task_count: 1,
                },
                9,
            ),
            (
                Telemetry::TaskInfo {
                    suite_id,
                    test_id,
                    name: "",
                    description: "",
                    input_type: "",
                    output_type: "",
                },
                10,
            ),
            (
                Telemetry::TaskState {
                    suite_id,
                    test_id,
                    state: TaskRunState::Running,
                    message: None,
                },
                11,
            ),
            (
                Telemetry::TaskSample {
                    suite_id,
                    test_id,
                    seq: 0,
                    payload: &[],
                },
                12,
            ),
            (
                Telemetry::TeardownReport {
                    suite_id,
                    test_id,
                    ok: true,
                    message: None,
                },
                13,
            ),
            (
                Telemetry::TaskStats {
                    suite_id,
                    test_id,
                    steps: 0,
                    time_us: 0,
                },
                14,
            ),
        ]
    }

    fn run_states() -> [TaskRunState; 8] {
        [
            TaskRunState::Running,
            TaskRunState::Warn,
            TaskRunState::Pass,
            TaskRunState::Fail,
            TaskRunState::Error,
            TaskRunState::Aborted,
            TaskRunState::TimedOut,
            TaskRunState::Bounded,
        ]
    }

    #[test]
    fn task_frames_fit_one_payload() {
        let packet = [0xA5_u8; crate::MAX_PACKET_SIZE];
        let text = "m".repeat(MAX_MESSAGE_SIZE);
        let ids = (u16::MAX, u16::MAX);
        let sample = Telemetry::TaskSample {
            suite_id: ids.0,
            test_id: ids.1,
            seq: u64::MAX,
            payload: &packet,
        };
        let state = Telemetry::TaskState {
            suite_id: ids.0,
            test_id: ids.1,
            state: TaskRunState::Bounded,
            message: Some(&text),
        };
        let report = Telemetry::TeardownReport {
            suite_id: ids.0,
            test_id: ids.1,
            ok: false,
            message: Some(&text),
        };
        let mut buf = [0u8; MAX_PAYLOAD_SIZE];
        for t in [&sample, &state, &report] {
            assert!(
                postcard::to_slice(t, &mut buf).is_ok(),
                "fits one payload"
            );
        }
        let exact = postcard::to_slice(&sample, &mut buf).unwrap().len();
        assert_eq!(
            exact, MAX_PAYLOAD_SIZE,
            "MAX_PACKET_SIZE is the exact bound"
        );
        let input = Command::TaskInput {
            suite_id: ids.0,
            test_id: ids.1,
            seq: u64::MAX,
            payload: &packet,
        };
        let exact = postcard::to_slice(&input, &mut buf).unwrap().len();
        assert_eq!(exact, MAX_PAYLOAD_SIZE);
    }

    #[test]
    fn a_task_input_decodes_borrowing_its_payload() {
        let packet = [0xA5_u8; crate::MAX_PACKET_SIZE];
        let input = Command::TaskInput {
            suite_id: u16::MAX,
            test_id: u16::MAX,
            seq: u64::MAX,
            payload: &packet,
        };
        let mut wire = [0u8; MAX_FRAME_SIZE];
        let len = FrameEncoder::frame_command(&input, &mut wire).unwrap();
        let mut reader = FrameReader::new();
        let mut frame_len = None;
        for &b in span(&wire, 0..len) {
            if let Some(payload) = reader.handle_byte(b) {
                frame_len = Some(payload.len());
            }
        }
        let decoded = frame_len.and_then(|n| {
            postcard::from_bytes::<Command<'_>>(reader.completed_payload(n))
                .ok()
        });
        match decoded {
            Some(Command::TaskInput { seq, payload, .. }) => {
                assert_eq!(seq, u64::MAX);
                assert_eq!(payload, &packet[..]);
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    /// Frames each command back to back.
    fn wire_of(cmds: &[Command<'_>]) -> Wire {
        let mut wire = [0u8; 256];
        let mut at = 0_usize;
        for cmd in cmds {
            let dest = wire.get_mut(at..).unwrap();
            at = at.saturating_add(
                FrameEncoder::frame_command(cmd, dest).unwrap(),
            );
        }
        (wire, at)
    }

    #[test]
    fn buffered_reader_delivers_every_frame_of_a_chunk() {
        let (wire, len) = wire_of(&[
            Command::Heartbeat,
            Command::StopNow {
                suite_id: 1,
                test_id: 2,
            },
            Command::TryReset,
        ]);
        let mut rx = BufferedFrameReader::<64>::new();
        let mut reads = 0_u32;
        let mut chunk = wire.get(..len);
        let mut got = std::vec::Vec::new();
        for _ in 0..4 {
            let cmd = rx
                .poll(|buf| {
                    reads = reads.saturating_add(1);
                    let data = chunk.take().unwrap_or(&[]);
                    buf.get_mut(..data.len()).unwrap().copy_from_slice(data);
                    Ok::<_, ()>(data.len())
                })
                .unwrap();
            if let Some(cmd) = cmd {
                got.push(std::format!("{cmd:?}"));
            }
        }
        assert_eq!(got.len(), 3, "no command is lost");
        for (g, want) in got.iter().zip(["Heartbeat", "StopNow", "TryReset"]) {
            assert!(g.contains(want), "{g} vs {want}");
        }
        assert_eq!(reads, 2, "read is not called while bytes are held");
    }

    #[test]
    fn buffered_reader_handles_split_frames_and_noise() {
        let (wire, len) = wire_of(&[Command::Heartbeat, Command::TryReset]);
        let mut noisy = std::vec![0x13_u8, 0xAA, 0x00];
        noisy.extend_from_slice(wire.get(..len).unwrap());
        let mut rx = BufferedFrameReader::<4>::new();
        let mut at = 0_usize;
        let mut got = std::vec::Vec::new();
        for _ in 0..40 {
            let cmd = rx
                .poll(|buf| {
                    let rest = noisy.get(at..).unwrap_or(&[]);
                    let n = rest.len().min(buf.len());
                    let (src, dst) =
                        (rest.get(..n).unwrap(), buf.get_mut(..n).unwrap());
                    dst.copy_from_slice(src);
                    at = at.saturating_add(n);
                    Ok::<_, ()>(n)
                })
                .unwrap();
            if let Some(cmd) = cmd {
                got.push(std::format!("{cmd:?}"));
            }
        }
        assert_eq!(got.len(), 2);
        for (g, want) in got.iter().zip(["Heartbeat", "TryReset"]) {
            assert!(g.contains(want), "{g} vs {want}");
        }
    }

    #[test]
    fn buffered_reader_keeps_a_frame_split_across_short_reads() {
        let (wire, len) = wire_of(&[Command::StopNow {
            suite_id: 1,
            test_id: 2,
        }]);
        let half = len.checked_div(2).unwrap();
        let head = wire.get(..half).unwrap();
        let tail = wire.get(half..len).unwrap();
        let mut chunks = [head, tail].into_iter();
        let mut rx = BufferedFrameReader::<64>::new();
        let mut next_chunk = |buf: &mut [u8]| {
            let chunk = chunks.next().unwrap_or(&[]);
            buf.get_mut(..chunk.len()).unwrap().copy_from_slice(chunk);
            Ok::<_, ()>(chunk.len())
        };
        assert!(matches!(rx.poll(&mut next_chunk), Ok(None)));
        assert!(matches!(
            rx.poll(&mut next_chunk),
            Ok(Some(Command::StopNow {
                suite_id: 1,
                test_id: 2
            }))
        ));
    }

    #[test]
    fn buffered_reader_drops_an_undecodable_frame_and_continues() {
        let mut wire = [0u8; 64];
        let bad =
            FrameEncoder::frame_payload(&[0x7F, 0x00], &mut wire).unwrap();
        let good = FrameEncoder::frame_command(
            &Command::Heartbeat,
            wire.get_mut(bad..).unwrap(),
        )
        .unwrap();
        let total = bad.saturating_add(good);
        let mut rx = BufferedFrameReader::<64>::new();
        let first = rx.poll(|buf| {
            buf.get_mut(..total)
                .unwrap()
                .copy_from_slice(wire.get(..total).unwrap());
            Ok::<_, ()>(total)
        });
        assert!(
            matches!(first, Ok(None)),
            "the undecodable frame is dropped"
        );
        let second = rx.poll(|_| Ok::<_, ()>(0));
        assert!(matches!(second, Ok(Some(Command::Heartbeat))));
    }

    #[test]
    fn buffered_reader_propagates_transport_errors() {
        let mut rx = BufferedFrameReader::<8>::new();
        let res = rx.poll(|_| Err::<usize, _>("link down"));
        assert_eq!(res.unwrap_err(), "link down");
    }
}
