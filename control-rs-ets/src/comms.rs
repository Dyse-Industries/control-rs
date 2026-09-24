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
/// Maximum postcard payload bytes in one frame.
pub const MAX_PAYLOAD_SIZE: usize = 512;
const START_BYTE_1: u8 = 0xAA;
const START_BYTE_2: u8 = 0x55;

/// The payload returned by `handle_byte` when a full frame is decoded.
///
/// Refers to the decoded raw byte slice inside the reader's internal buffer.
pub type DecodedFrame<'a> = &'a [u8];

/// Result of polling a command from the host.
///
/// Returns a command if successfully decoded or transport error `E`.
pub type PollResult<E> = Result<Option<Command>, E>;

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
///     fn poll_command(&mut self) -> PollResult<Self::Error> { Ok(None) }
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
    /// * `PollResult<Self::Error>`
    ///     * `Ok(Some(Command))` when a full valid command frame is parsed.
    ///     * `Ok(None)` if no command is ready yet.
    ///     * `Err(Error)` on serial port or protocol errors.
    ///
    /// # Errors
    /// Returns a serial port or protocol error if reading or de-framing fails.
    fn poll_command(&mut self) -> PollResult<Self::Error>;

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
pub enum Command {
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
    #[allow(clippy::arithmetic_side_effects)]
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
                *read += 1;
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

impl FrameEncoder {
    /// Writes sync, length, and CRC for a payload already sitting at `dest[4..]`.
    #[allow(clippy::arithmetic_side_effects)]
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
                .get(4..4 + payload_len)
                .ok_or(postcard::Error::SerializeBufferFull)?;
            crc::Crc::<u16>::new(&crc::CRC_16_IBM_SDLC).checksum(payload)
        };
        if let Some(slot) = dest.get_mut(4 + payload_len) {
            *slot = (crc_value >> 8) as u8;
        }
        if let Some(slot) = dest.get_mut(4 + payload_len + 1) {
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
        cmd: &Command,
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
    use super::*;

    /// A decoded payload copied into a fixed buffer, with its length.
    type DecodedPayload<const N: usize> = ([u8; N], usize);

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

    /// Frames `cmd`, decodes it back through `FrameReader` and returns the
    /// deserialized command.
    fn command_round_trip(cmd: &Command) -> Command {
        let mut buf = [0u8; 32];
        let len = FrameEncoder::frame_command(cmd, &mut buf)
            .expect("command framing");
        let (payload, payload_len) = first_payload::<32>(span(&buf, 0..len))
            .expect("a framed command must decode");
        postcard::from_bytes(span(&payload, 0..payload_len))
            .expect("postcard decode command")
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
            fn poll_command(&mut self) -> Result<Option<Command>, Self::Error> {
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

        let decoded_cmd = command_round_trip(&Command::RunExecutable {
            suite_id: 1,
            test_id: 2,
        });
        match decoded_cmd {
            Command::RunExecutable { suite_id, test_id } => {
                assert_eq!(suite_id, 1);
                assert_eq!(test_id, 2);
            }
            _ => panic!("unexpected command variant"),
        }
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
        let decoded = command_round_trip(&Command::RunExecutable {
            suite_id: 1,
            test_id: 2,
        });
        assert!(matches!(
            decoded,
            Command::RunExecutable {
                suite_id: 1,
                test_id: 2
            }
        ));
    }

    #[test]
    fn test_golden_wire_vector_try_reset() {
        let decoded = command_round_trip(&Command::TryReset);
        assert!(matches!(decoded, Command::TryReset));
    }

    /// One instance of every telemetry variant, with payload fields chosen
    /// to exercise each string and integer encoding.
    fn golden_telemetry_variants() -> [Telemetry<'static>; 7] {
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
        ]
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
}
