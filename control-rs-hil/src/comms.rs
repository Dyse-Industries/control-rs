//! Target-to-Host communication protocol and traits.
//!
//! > L. L. Peterson and B. S. Davie, "2.3 Framing," in Computer Networks: A Systems Approach, 2024.
//! >   [Online]. Available: [URL]. [Accessed: [Date of Access, e.g., June 27, 2026]].

use core::sync::atomic::{AtomicBool, Ordering};

use crate::settings::SettingValue;

const MAX_PAYLOAD_SIZE: usize = 512;
const START_BYTE_1: u8 = 0xAA;
const START_BYTE_2: u8 = 0x55;

/// The payload returned by `handle_byte` when a full frame is decoded.
pub type DecodedFrame<'a> = &'a [u8];

/// Result of polling a command from the host.
pub type PollResult<E> = Result<Option<Command>, E>;

/// Result of sending telemetry or flushing.
pub type SendResult<E> = Result<(), E>;

/// A trait for executing frame-based communication between target and host.
///
/// Handlers of this trait bridge the parsed commands and telemetry messages
/// onto concrete hardware peripherals (like UART, USB, or RTT).
#[allow(clippy::type_complexity)]
pub trait HostComms {
    /// The error type associated with transport failures.
    type Error;

    /// Closes the communication interface (e.g. signaling semihosting exit).
    fn close(&mut self) {}

    /// Closes the communication interface with a failure/error status.
    fn close_on_failure(&mut self) {}

    /// Flush any pending buffered data out to the physical interface.
    ///
    /// # Errors
    ///
    /// Returns a transport error if flushing the buffer to the physical interface fails.
    fn flush(&mut self) -> SendResult<Self::Error>;

    /// Read incoming bytes and try to parse a Command.
    ///
    /// This should be non-blocking. It returns:
    /// - `Ok(Some(Command))` when a full valid command frame is parsed.
    /// - `Ok(None)` if no command is ready yet.
    /// - `Err(Error)` on serial port or protocol errors.
    ///
    /// # Errors
    ///
    /// Returns a serial port or protocol error if reading or de-framing fails.
    fn poll_command(&mut self) -> PollResult<Self::Error>;

    /// Send a telemetry message to the host.
    ///
    /// # Errors
    ///
    /// Returns a transport error if serialization or writing to the physical interface fails.
    fn send_telemetry(
        &mut self,
        telemetry: &Telemetry<'_>,
    ) -> SendResult<Self::Error>;
}

/// Commands sent from the Host TUI to the Target MCU.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum Command {
    /// Request the target to stream the list of all suites, tests, and settings.
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
    /// Test execution failed (e.g. panic or assertion failure).
    Failed,
    /// Test completed successfully.
    Passed,
    /// Test is registered but has not run.
    Pending,
    /// Test is currently executing.
    Running,
}

/// A thread-safe lock using an atomic boolean flag to protect communication resources.
pub struct CommsLock {
    locked: AtomicBool,
}

/// State machine to de-frame a stream of incoming bytes into packets.
pub struct FrameReader {
    payload_buffer: [u8; MAX_PAYLOAD_SIZE],
    state: ReaderState,
    temp_len: u16,
}

/// A log message produced by a test executable or the runner itself.
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

impl Default for CommsLock {
    fn default() -> Self {
        Self::new()
    }
}

impl CommsLock {
    /// Creates a new, unlocked `CommsLock`.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            locked: AtomicBool::new(false),
        }
    }

    /// Attempts to acquire the lock. Returns `true` if successful, or `false` if already locked.
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
    #[must_use]
    pub const fn is_idle(&self) -> bool {
        matches!(self.state, ReaderState::WaitStart1)
    }

    /// Creates a new `FrameReader`.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            payload_buffer: [0u8; MAX_PAYLOAD_SIZE],
            state: ReaderState::WaitStart1,
            temp_len: 0,
        }
    }
}

/// Helper to serialize and frame a telemetry message into a destination buffer.
/// Returns the number of bytes written.
///
/// # Errors
/// Returns `postcard::Error` if serialization fails.
#[allow(clippy::arithmetic_side_effects)]
pub fn frame_telemetry(
    telemetry: &Telemetry<'_>,
    dest: &mut [u8],
) -> Result<usize, postcard::Error> {
    if dest.len() < 6 {
        return Err(postcard::Error::SerializeBufferFull);
    }

    // Set the headers safely using get_mut to avoid index bounds checks
    if let Some(slot) = dest.get_mut(0) {
        *slot = START_BYTE_1;
    }
    if let Some(slot) = dest.get_mut(1) {
        *slot = START_BYTE_2;
    }

    let dest_len = dest.len();
    // Serialize payload into the dest starting at index 4 (leaving space for length and leaving 2 bytes at the end for CRC-16)
    let (payload_len, crc_value) = {
        let slice = dest
            .get_mut(4..dest_len - 2)
            .ok_or(postcard::Error::SerializeBufferFull)?;
        let payload_slice = postcard::to_slice(telemetry, slice)?;
        let crc = crc::Crc::<u16>::new(&crc::CRC_16_IBM_SDLC);
        let crc_value = crc.checksum(payload_slice);
        (payload_slice.len(), crc_value)
    };

    // Write length (big-endian)
    let len_u16 = u16::try_from(payload_len).unwrap_or(0);
    if let Some(slot) = dest.get_mut(2) {
        *slot = (len_u16 >> 8) as u8;
    }
    if let Some(slot) = dest.get_mut(3) {
        *slot = (len_u16 & 0xFF) as u8;
    }

    // Write checksum (2 bytes, big-endian)
    if let Some(slot) = dest.get_mut(4 + payload_len) {
        *slot = (crc_value >> 8) as u8;
    }
    if let Some(slot) = dest.get_mut(4 + payload_len + 1) {
        *slot = (crc_value & 0xFF) as u8;
    }

    Ok(6 + payload_len)
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_frame_reader_invalid_lengths() {
        let mut reader = FrameReader::new();
        // Move to WaitLen1
        assert!(reader.handle_byte(START_BYTE_1).is_none());
        assert!(reader.handle_byte(START_BYTE_2).is_none());

        // Send length = 0 (MSB=0, LSB=0)
        assert!(reader.handle_byte(0x00).is_none());
        assert!(reader.handle_byte(0x00).is_none());
        assert!(reader.is_idle()); // should reset since len = 0 is invalid

        // Send length > MAX_PAYLOAD_SIZE (e.g. 513 = MSB=2, LSB=1)
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
}
