//! Target-to-Host communication protocol and traits.

use crate::settings::SettingValue;

/// Commands sent from the Host TUI to the Target MCU.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum Command {
    /// Request the target to stream the list of all suites, tests, and settings.
    ListSuites,
    /// Request execution of a specific test.
    RunExecutable { suite_id: u16, test_id: u16 },
    /// Update a setting's value.
    SetSetting {
        suite_id: u16,
        setting_id: u16,
        value: SettingValue,
    },
    /// Request the target to reset.
    OkToReset,
}

/// The state of a test executable during a test session.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize,
)]
pub enum TestState {
    /// Test is registered but has not run.
    Pending,
    /// Test is currently executing.
    Running,
    /// Test completed successfully.
    Passed,
    /// Test execution failed (e.g. panic or assertion failure).
    Failed,
}

/// A log message produced by a test executable or the runner itself.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LogMessage<'a> {
    /// Microseconds elapsed since boot / epoch.
    pub timestamp_us: u64,
    /// The ID of the test suite.
    pub suite_id: u16,
    /// The ID of the test executable.
    pub test_id: u16,
    /// The log text payload.
    pub payload: &'a str,
}

/// Telemetry and logs sent from the Target MCU to the Host TUI.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum Telemetry<'a> {
    /// Metadata about a discovered suite.
    SuiteInfo {
        suite_id: u16,
        name: &'a str,
        test_count: u16,
        setting_count: u16,
    },
    /// Metadata about a test within a suite.
    TestInfo {
        suite_id: u16,
        test_id: u16,
        name: &'a str,
    },
    /// Metadata about a setting within a suite.
    SettingInfo {
        suite_id: u16,
        setting_id: u16,
        name: &'a str,
        value: SettingValue,
    },
    /// Notification that the target has finished sending discovery information.
    DiscoveryComplete,
    /// Notification of a test state transition.
    TestStateChange {
        suite_id: u16,
        test_id: u16,
        state: TestState,
    },
    /// Performance metrics for a completed test run.
    MetricReport {
        suite_id: u16,
        test_id: u16,
        cycles: u64,
        time_us: u64,
    },
    /// A log message.
    Log(LogMessage<'a>),
    /// Notification of a general target crash or panic.
    TargetPanic {
        message: &'a str,
        file: &'a str,
        line: u32,
    },
}

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

    /// Read incoming bytes and try to parse a Command.
    ///
    /// This should be non-blocking. It returns:
    /// - `Ok(Some(Command))` when a full valid command frame is parsed.
    /// - `Ok(None)` if no command is ready yet.
    /// - `Err(Error)` on serial port or protocol errors.
    fn poll_command(&mut self) -> PollResult<Self::Error>;

    /// Send a telemetry message to the host.
    fn send_telemetry(
        &mut self,
        telemetry: &Telemetry<'_>,
    ) -> SendResult<Self::Error>;

    /// Flush any pending buffered data out to the physical interface.
    fn flush(&mut self) -> SendResult<Self::Error>;
}

// --- Frame Reader / Writer Implementation helpers ---

const START_BYTE_1: u8 = 0xAA;
const START_BYTE_2: u8 = 0x55;
const MAX_PAYLOAD_SIZE: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReaderState {
    WaitStart1,
    WaitStart2,
    WaitLen1,
    WaitLen2,
    ReadingPayload { len: usize, read: usize },
    WaitChecksum { len: usize },
}

/// The payload returned by handle_byte when a full frame is decoded.
pub type DecodedFrame<'a> = &'a [u8];

/// State machine to deframe a stream of incoming bytes into packets.
pub struct FrameReader {
    state: ReaderState,
    payload_buffer: [u8; MAX_PAYLOAD_SIZE],
    temp_len: u16,
}

impl Default for FrameReader {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameReader {
    /// Creates a new `FrameReader`.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            state: ReaderState::WaitStart1,
            payload_buffer: [0u8; MAX_PAYLOAD_SIZE],
            temp_len: 0,
        }
    }

    /// Returns true if the reader is currently idle (waiting for a new frame start).
    #[must_use]
    pub const fn is_idle(&self) -> bool {
        matches!(self.state, ReaderState::WaitStart1)
    }

    /// Process a single incoming byte. Returns `Some(&[u8])` when a complete,
    /// checksum-verified payload has been received.
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
                    self.state = ReaderState::WaitChecksum { len };
                }
            }
            ReaderState::WaitChecksum { len } => {
                // Calculate simple XOR checksum over the payload
                let mut checksum: u8 = 0;
                for i in 0..len {
                    checksum ^= self.payload_buffer[i];
                }

                self.state = ReaderState::WaitStart1;
                if checksum == byte {
                    return Some(&self.payload_buffer[..len]);
                }
            }
        }
        None
    }
}

/// Helper to serialize and frame a telemetry message into a destination buffer.
/// Returns the number of bytes written.
///
/// # Errors
/// Returns `postcard::Error` if serialization fails.
pub fn frame_telemetry(
    telemetry: &Telemetry<'_>,
    dest: &mut [u8],
) -> Result<usize, postcard::Error> {
    if dest.len() < 5 {
        return Err(postcard::Error::SerializeBufferFull);
    }

    // Set the headers
    dest[0] = START_BYTE_1;
    dest[1] = START_BYTE_2;

    let dest_len = dest.len();
    // Serialize payload into the dest starting at index 4 (leaving space for length)
    let (payload_len, checksum) = {
        let payload_slice =
            postcard::to_slice(telemetry, &mut dest[4..dest_len - 1])?;
        let mut sum = 0u8;
        for &b in payload_slice.iter() {
            sum ^= b;
        }
        (payload_slice.len(), sum)
    };

    // Write length (big-endian)
    let len_u16 = payload_len as u16;
    dest[2] = (len_u16 >> 8) as u8;
    dest[3] = (len_u16 & 0xFF) as u8;

    // Write checksum
    dest[4 + payload_len] = checksum;

    Ok(5 + payload_len)
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

        // Send length > MAX_PAYLOAD_SIZE (e.g. 257 = MSB=1, LSB=1)
        assert!(reader.handle_byte(START_BYTE_1).is_none());
        assert!(reader.handle_byte(START_BYTE_2).is_none());
        assert!(reader.handle_byte(0x01).is_none());
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
        let payload = reader.handle_byte(buf[framed_len - 1]).unwrap();

        // Deserialize and check
        let decoded: Telemetry = postcard::from_bytes(payload).unwrap();
        assert!(matches!(decoded, Telemetry::DiscoveryComplete));
        assert!(reader.is_idle());
    }

    #[test]
    fn test_frame_reader_invalid_checksum() {
        let mut reader = FrameReader::new();
        let telemetry = Telemetry::DiscoveryComplete;
        let mut buf = [0u8; 128];
        let framed_len = frame_telemetry(&telemetry, &mut buf).unwrap();

        // Feed all bytes except the checksum
        for &item in buf.iter().take(framed_len - 1) {
            assert!(reader.handle_byte(item).is_none());
        }
        // Feed an invalid checksum byte
        let bad_checksum = buf[framed_len - 1] ^ 0xFF;
        assert!(reader.handle_byte(bad_checksum).is_none());
        assert!(reader.is_idle()); // reset
    }

    #[test]
    fn test_frame_telemetry_too_small_buffer() {
        let telemetry = Telemetry::DiscoveryComplete;
        let mut buf = [0u8; 4];
        let res = frame_telemetry(&telemetry, &mut buf);
        assert!(matches!(res, Err(postcard::Error::SerializeBufferFull)));

        // Buffer large enough for header but too small for payload
        let mut buf = [0u8; 5];
        let res = frame_telemetry(&telemetry, &mut buf);
        assert!(res.is_err());
    }
}
