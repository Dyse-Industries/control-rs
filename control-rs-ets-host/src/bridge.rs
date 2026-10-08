//! Bridge module to interface host computer with the target device.
//! Manages spawning and monitoring the execution environments (QEMU or Serial).
//!
//! # Examples
//!
//! ```
//! use control_rs_ets::comms::{Command, FrameEncoder, FrameReader};
//!
//! let cmd = Command::ListSuites;
//! let mut buf = [0u8; 64];
//! let len = FrameEncoder::frame_command(&cmd, &mut buf).unwrap();
//!
//! let mut reader = FrameReader::new();
//! let mut decoded = false;
//! for &b in &buf[..len] {
//!     if reader.handle_byte(b).is_some() {
//!         decoded = true;
//!         break;
//!     }
//! }
//! assert!(decoded);
//! ```
//!
//! ```no_run
//! use control_rs_ets_host::ETSBridge;
//! use control_rs_ets_host::target::Target;
//!
//! let target = Target::qemu_arm();
//! let bridge = ETSBridge::new(target, false);
//! assert!(bridge.is_ok());
//! ```

use std::io::{Read, Write as IoWrite};
use std::process::{Child, ChildStdout, Command as StdCommand, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use control_rs_ets::comms::{
    Command, FrameEncoder, FrameReader, MAX_FRAME_SIZE, Telemetry,
};
use control_rs_ets::settings::SettingValue;

use crate::error::HostError;
use crate::target::{SubprocessTarget, Target};

/// Bytes written to an in-memory test link, shared with the test.
#[cfg(all(any(test, feature = "fake-link"), unix))]
pub type SharedBytes = Arc<std::sync::Mutex<Vec<u8>>>;

type WaitResult = Result<Option<std::process::ExitStatus>, std::io::Error>;

/// Join handles for the threads that read target output.
type ReaderHandles = Vec<JoinHandle<()>>;

/// Host driver (`ETSBridge`) for virtual ETS (QEMU) and ETS (board).
pub struct ETSBridge {
    inner: BridgeInner,
    link_info: String,
    rx_from_target: Receiver<BridgeMessage>,
    target_info: String,
    shutdown: Arc<AtomicBool>,
    readers: ReaderHandles,
}

/// Inner bridge enum representing active connection variant.
enum BridgeInner {
    /// Connection to QEMU via child process.
    Qemu {
        /// The child process handle.
        child: Child,
        /// Stdin of the child process.
        stdin: std::process::ChildStdin,
    },
    /// Connection to hardware via serial port.
    Serial {
        /// Serial port interface.
        port: serial2::SerialPort,
    },
    /// In-memory link recording written frames (tests only).
    #[cfg(all(any(test, feature = "fake-link"), unix))]
    Fake(FakeLink),
}

/// An [`ETSBridge`] over an in-memory link with its test handles.
#[cfg(all(any(test, feature = "fake-link"), unix))]
pub struct FakeBridge {
    /// The bridge under test.
    pub bridge: ETSBridge,
    /// Feeds messages to the bridge as if the target sent them.
    pub tx: Sender<BridgeMessage>,
    /// Bytes the bridge has written to the target.
    pub written: SharedBytes,
}

/// State of the in-memory test link.
#[cfg(all(any(test, feature = "fake-link"), unix))]
pub(crate) struct FakeLink {
    /// Every byte written to the link.
    pub(crate) written: SharedBytes,
    /// Polls of `try_wait` before the fake target reports an exit (`None` never).
    pub(crate) exit_after_polls: Option<usize>,
    /// Polls of `try_wait` so far.
    pub(crate) polls: usize,
}

/// Host-owned telemetry. String fields are copied out of the decode buffer
/// so the reader thread can reuse it without leaking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OwnedTelemetry {
    /// Discovery finished.
    DiscoveryComplete,
    /// Log line from the target.
    Log {
        /// Microseconds since boot.
        timestamp_us: u64,
        /// Suite that emitted the log.
        suite_id: u16,
        /// Test that emitted the log.
        test_id: u16,
        /// Log text.
        payload: String,
    },
    /// Pass metrics for a completed test.
    MetricReport {
        /// CPU cycles.
        cycles: u64,
        /// Peak stack bytes.
        stack_peak: u32,
        /// Suite id.
        suite_id: u16,
        /// Test id.
        test_id: u16,
        /// Duration microseconds.
        time_us: u64,
    },
    /// Discovered setting metadata.
    SettingInfo {
        /// Doc comment.
        description: String,
        /// Setting name.
        name: String,
        /// Setting id.
        setting_id: u16,
        /// Parent suite id.
        suite_id: u16,
        /// Current value.
        value: SettingValue,
    },
    /// Discovered suite metadata.
    SuiteInfo {
        /// Doc comment.
        description: String,
        /// Suite name.
        name: String,
        /// Number of settings.
        setting_count: u16,
        /// Suite id.
        suite_id: u16,
        /// Number of tests.
        test_count: u16,
    },
    /// Target panic / exception report.
    TargetPanic {
        /// Source file.
        file: String,
        /// Source line.
        line: u32,
        /// Panic message.
        message: String,
    },
    /// Discovered test metadata.
    TestInfo {
        /// Doc comment.
        description: String,
        /// Test name.
        name: String,
        /// Parent suite id.
        suite_id: u16,
        /// Test id.
        test_id: u16,
    },
    /// Test state transition.
    TestStateChange {
        /// New state.
        state: control_rs_ets::comms::TestState,
        /// Parent suite id.
        suite_id: u16,
        /// Test id.
        test_id: u16,
    },
    /// Wire-contract revision and target metadata (FR-8).
    TargetInfo {
        /// Target's `PROTOCOL_VERSION`.
        protocol_version: u8,
        /// Board identifier (`0` when unknown).
        board_id: u16,
        /// Core clock in hertz (`0` when unknown).
        core_clock_hz: u32,
        /// FPU bits: 0 single, 1 double precision.
        fpu_flags: u8,
    },
    /// A suite provides a loop; its `LoopInfo` follows.
    LifecycleSuite {
        /// Suite id.
        suite_id: u16,
        /// Loop count (always `1`).
        loop_count: u8,
    },
    /// Discovered loop metadata.
    LoopInfo {
        /// Parent suite id.
        suite_id: u16,
        /// Loop identifier within the suite.
        test_id: u16,
        /// Loop name.
        name: String,
        /// Doc comment.
        description: String,
        /// Input packet type name.
        input_type: String,
        /// Output packet type name.
        output_type: String,
    },
    /// Loop run state transition.
    LoopState {
        /// Parent suite id.
        suite_id: u16,
        /// Loop identifier within the suite.
        test_id: u16,
        /// New run state.
        state: control_rs_ets::comms::LoopRunState,
        /// Optional message.
        message: Option<String>,
    },
    /// Output packet of one loop step.
    LoopSample {
        /// Parent suite id.
        suite_id: u16,
        /// Loop identifier within the suite.
        test_id: u16,
        /// Step that produced the output.
        seq: u64,
        /// Encoded output packet.
        payload: Vec<u8>,
    },
    /// Teardown outcome of a loop run.
    TeardownReport {
        /// Parent suite id.
        suite_id: u16,
        /// Loop identifier within the suite.
        test_id: u16,
        /// Whether teardown succeeded.
        ok: bool,
        /// Optional message.
        message: Option<String>,
    },
    /// Statistics of a finished loop run.
    LoopStats {
        /// Parent suite id.
        suite_id: u16,
        /// Loop identifier within the suite.
        test_id: u16,
        /// Steps called.
        steps: u64,
        /// Elapsed microseconds from setup to teardown entry.
        time_us: u64,
    },
}

/// Message type sent from the background reader thread to the host controller or UI.
pub enum BridgeMessage {
    /// Raw console output (stdout/stderr) from the target/QEMU.
    RawConsole(String),
    /// Telemetry parsed from target, with owned strings.
    Telemetry(OwnedTelemetry),
}

impl OwnedTelemetry {
    /// Copies string fields out of a borrowed [`Telemetry`] frame.
    #[must_use]
    pub fn from_telemetry(tel: &Telemetry<'_>) -> Self {
        match *tel {
            Telemetry::DiscoveryComplete => Self::DiscoveryComplete,
            Telemetry::Log(ref msg) => Self::Log {
                timestamp_us: msg.timestamp_us,
                suite_id: msg.suite_id,
                test_id: msg.test_id,
                payload: msg.payload.to_string(),
            },
            Telemetry::MetricReport {
                cycles,
                stack_peak,
                suite_id,
                test_id,
                time_us,
            } => Self::MetricReport {
                cycles,
                stack_peak,
                suite_id,
                test_id,
                time_us,
            },
            Telemetry::TargetPanic {
                file,
                line,
                message,
            } => Self::TargetPanic {
                file: file.to_string(),
                line,
                message: message.to_string(),
            },
            Telemetry::TestStateChange {
                state,
                suite_id,
                test_id,
            } => Self::TestStateChange {
                state,
                suite_id,
                test_id,
            },
            Telemetry::TargetInfo {
                protocol_version,
                board_id,
                core_clock_hz,
                fpu_flags,
            } => Self::TargetInfo {
                protocol_version,
                board_id,
                core_clock_hz,
                fpu_flags,
            },
            Telemetry::SettingInfo { .. }
            | Telemetry::SuiteInfo { .. }
            | Telemetry::TestInfo { .. } => Self::from_catalog_entry(tel),
            Telemetry::LifecycleSuite { .. }
            | Telemetry::LoopInfo { .. }
            | Telemetry::LoopState { .. }
            | Telemetry::LoopSample { .. }
            | Telemetry::TeardownReport { .. }
            | Telemetry::LoopStats { .. } => Self::from_loop_frame(tel),
        }
    }

    /// Copies a loop frame. [`Self::from_telemetry`] routes exactly the six
    /// loop variants here; any other variant is handed back to it.
    fn from_loop_frame(tel: &Telemetry<'_>) -> Self {
        match *tel {
            Telemetry::LifecycleSuite { .. } | Telemetry::LoopInfo { .. } => {
                Self::from_loop_catalog(tel)
            }
            Telemetry::LoopState {
                suite_id,
                test_id,
                state,
                message,
            } => Self::LoopState {
                suite_id,
                test_id,
                state,
                message: message.map(str::to_string),
            },
            Telemetry::LoopSample {
                suite_id,
                test_id,
                seq,
                payload,
            } => Self::LoopSample {
                suite_id,
                test_id,
                seq,
                payload: payload.to_vec(),
            },
            Telemetry::TeardownReport {
                suite_id,
                test_id,
                ok,
                message,
            } => Self::TeardownReport {
                suite_id,
                test_id,
                ok,
                message: message.map(str::to_string),
            },
            Telemetry::LoopStats {
                suite_id,
                test_id,
                steps,
                time_us,
            } => Self::LoopStats {
                suite_id,
                test_id,
                steps,
                time_us,
            },
            ref other => Self::from_telemetry(other),
        }
    }

    /// Copies the loop announcement of a suite: its marker and its metadata.
    fn from_loop_catalog(tel: &Telemetry<'_>) -> Self {
        match *tel {
            Telemetry::LifecycleSuite {
                suite_id,
                loop_count,
            } => Self::LifecycleSuite {
                suite_id,
                loop_count,
            },
            Telemetry::LoopInfo {
                suite_id,
                test_id,
                name,
                description,
                input_type,
                output_type,
            } => Self::LoopInfo {
                suite_id,
                test_id,
                name: name.to_string(),
                description: description.to_string(),
                input_type: input_type.to_string(),
                output_type: output_type.to_string(),
            },
            ref other => Self::from_telemetry(other),
        }
    }

    /// Copies a discovery catalog entry (suite, test or setting metadata).
    ///
    /// [`Self::from_telemetry`] routes exactly the three catalog variants
    /// here; any other variant is handed back to it, so the conversion stays
    /// total without a panic path.
    fn from_catalog_entry(tel: &Telemetry<'_>) -> Self {
        match *tel {
            Telemetry::SettingInfo {
                description,
                name,
                setting_id,
                suite_id,
                value,
            } => Self::SettingInfo {
                description: description.to_string(),
                name: name.to_string(),
                setting_id,
                suite_id,
                value,
            },
            Telemetry::SuiteInfo {
                description,
                name,
                setting_count,
                suite_id,
                test_count,
            } => Self::SuiteInfo {
                description: description.to_string(),
                name: name.to_string(),
                setting_count,
                suite_id,
                test_count,
            },
            Telemetry::TestInfo {
                description,
                name,
                suite_id,
                test_id,
            } => Self::TestInfo {
                description: description.to_string(),
                name: name.to_string(),
                suite_id,
                test_id,
            },
            ref runtime => Self::from_telemetry(runtime),
        }
    }
}

impl BridgeMessage {
    /// Copies a borrowed telemetry frame into an owned [`BridgeMessage`].
    #[must_use]
    pub fn telemetry(tel: &Telemetry<'_>) -> Self {
        Self::Telemetry(OwnedTelemetry::from_telemetry(tel))
    }
}

impl BridgeInner {
    fn write_frame(&mut self, frame: &[u8]) -> std::io::Result<()> {
        match self {
            Self::Qemu { stdin, .. } => {
                stdin.write_all(frame).and_then(|()| stdin.flush())
            }
            Self::Serial { port } => {
                port.write_all(frame).and_then(|()| port.flush())
            }
            #[cfg(all(any(test, feature = "fake-link"), unix))]
            Self::Fake(link) => {
                link.written
                    .lock()
                    .map_err(|_| std::io::Error::other("poisoned"))?
                    .extend_from_slice(frame);
                Ok(())
            }
        }
    }
}

impl Drop for ETSBridge {
    fn drop(&mut self) {
        self.terminate();
    }
}

impl ETSBridge {
    /// Stop reader threads and kill a QEMU child. Serial has no process to
    /// kill; the reader still joins after the shutdown flag and read timeout.
    pub fn terminate(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        match &mut self.inner {
            BridgeInner::Qemu { child, .. } => {
                let _ = child.kill();
            }
            BridgeInner::Serial { .. } => {}
            #[cfg(all(any(test, feature = "fake-link"), unix))]
            BridgeInner::Fake(_) => {}
        }
        for handle in self.readers.drain(..) {
            let _ = handle.join();
        }
    }

    /// Gets description of the communication link.
    #[must_use]
    pub fn link_info(&self) -> &str {
        &self.link_info
    }

    /// Spawns a new QEMU process or connects to a serial device.
    ///
    /// When `inherit_stderr` is true, QEMU `cargo run` stderr is left on the
    /// host terminal so cargo status lines are not captured as `RawConsole`.
    ///
    /// # Errors
    ///
    /// Returns `HostError::SerialOpen` if opening serial port fails after retry budget,
    /// `HostError::SerialClone` if serial port cannot be cloned, or
    /// `HostError::Spawn` if the subprocess cannot be launched.
    pub fn new(
        target: Target,
        inherit_stderr: bool,
    ) -> Result<Self, HostError> {
        let (tx, rx) = channel();

        match target {
            Target::Serial {
                port: port_path,
                baud,
            } => Self::new_serial(&port_path, baud, tx, rx),
            Target::Subprocess(sub) => {
                Self::new_subprocess(&sub, tx, rx, inherit_stderr)
            }
        }
    }

    fn new_serial(
        port_path: &str,
        baud: u32,
        tx: Sender<BridgeMessage>,
        rx: Receiver<BridgeMessage>,
    ) -> Result<Self, HostError> {
        let port = open_serial_with_retry(port_path, baud)?;

        let mut port_clone =
            port.try_clone().map_err(|e| HostError::SerialClone {
                source: e.to_string().into(),
            })?;
        port_clone
            .set_read_timeout(Duration::from_millis(100))
            .map_err(|e| HostError::Transport {
                source: format!("serial read timeout: {e}").into(),
            })?;

        let shutdown = Arc::new(AtomicBool::new(false));
        let reader = spawn_serial_reader(port_clone, tx, Arc::clone(&shutdown));

        Ok(Self {
            inner: BridgeInner::Serial { port },
            rx_from_target: rx,
            target_info: "Teensy 4.0 (Cortex-M7)".to_string(),
            link_info: format!("USB CDC ({port_path})"),
            shutdown,
            readers: vec![reader],
        })
    }

    fn new_subprocess(
        target: &SubprocessTarget,
        tx: Sender<BridgeMessage>,
        rx: Receiver<BridgeMessage>,
        inherit_stderr: bool,
    ) -> Result<Self, HostError> {
        let stderr_stdio = if inherit_stderr {
            Stdio::inherit()
        } else {
            Stdio::piped()
        };

        let crate_dir = target.crate_dir();
        let mut child = cargo_run_command(target)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(stderr_stdio)
            .spawn()
            .map_err(|e| HostError::Spawn {
                source: format!(
                    "Failed to spawn cargo run process in '{}': {e}",
                    crate_dir.display()
                )
                .into(),
            })?;

        let stdin = child.stdin.take().ok_or_else(|| HostError::Spawn {
            source: "Failed to open stdin".into(),
        })?;
        let stdout = child.stdout.take().ok_or_else(|| HostError::Spawn {
            source: "Failed to open stdout".into(),
        })?;

        let shutdown = Arc::new(AtomicBool::new(false));
        let mut readers = ReaderHandles::new();

        if inherit_stderr {
            readers.push(spawn_qemu_stdout_reader(
                stdout,
                tx,
                Arc::clone(&shutdown),
            ));
        } else {
            readers.push(spawn_qemu_stdout_reader(
                stdout,
                tx.clone(),
                Arc::clone(&shutdown),
            ));
            let stderr =
                child.stderr.take().ok_or_else(|| HostError::Spawn {
                    source: "Failed to open stderr".into(),
                })?;
            readers.push(spawn_stderr_reader(
                stderr,
                tx,
                Arc::clone(&shutdown),
            ));
        }

        Ok(Self {
            inner: BridgeInner::Qemu { child, stdin },
            rx_from_target: rx,
            target_info: target.display_name(),
            link_info: subprocess_link_info(target),
            shutdown,
            readers,
        })
    }

    /// Gets the channel receiver to poll messages from the target.
    #[must_use]
    pub const fn receiver(&self) -> &Receiver<BridgeMessage> {
        &self.rx_from_target
    }

    /// Sends a command to the target using the packet framing protocol.
    ///
    /// # Errors
    ///
    /// Returns `HostError::Transport` if serializing or writing to the target stream fails.
    pub fn send_command(&mut self, cmd: &Command<'_>) -> Result<(), HostError> {
        let mut buf = [0u8; MAX_FRAME_SIZE];
        let len = FrameEncoder::frame_command(cmd, &mut buf).map_err(|e| {
            HostError::Transport {
                source: format!("Failed to serialize command: {e}").into(),
            }
        })?;

        let frame = buf.get(..len).ok_or_else(|| HostError::Transport {
            source: format!(
                "framed command length {len} exceeds the {MAX_FRAME_SIZE} byte buffer"
            )
            .into(),
        })?;
        self.inner
            .write_frame(frame)
            .map_err(|e| HostError::Transport {
                source: format!("I/O failure sending command: {e}").into(),
            })
    }

    /// Gets description of the target platform.
    #[must_use]
    pub fn target_info(&self) -> &str {
        &self.target_info
    }

    /// Checks if the child process has exited (returns Ok(None) for serial).
    ///
    /// # Errors
    ///
    /// Returns an error if querying the child process status fails.
    pub fn try_wait(&mut self) -> WaitResult {
        match &mut self.inner {
            BridgeInner::Qemu { child, .. } => child.try_wait(),
            BridgeInner::Serial { .. } => Ok(None),
            #[cfg(all(any(test, feature = "fake-link"), unix))]
            BridgeInner::Fake(link) => {
                use std::os::unix::process::ExitStatusExt;
                link.polls = link.polls.saturating_add(1);
                Ok(link
                    .exit_after_polls
                    .filter(|limit| link.polls >= *limit)
                    .map(|_| std::process::ExitStatus::from_raw(0)))
            }
        }
    }

    /// A bridge over an in-memory link, with the sender that feeds it messages
    /// and the frames written to it (tests only).
    #[cfg(all(any(test, feature = "fake-link"), unix))]
    #[must_use]
    pub fn fake(exit_after_polls: Option<usize>) -> FakeBridge {
        let (tx, rx) = channel();
        let written = Arc::new(std::sync::Mutex::new(Vec::new()));
        let bridge = Self {
            inner: BridgeInner::Fake(FakeLink {
                written: Arc::clone(&written),
                exit_after_polls,
                polls: 0,
            }),
            link_info: "fake".to_string(),
            rx_from_target: rx,
            target_info: "fake".to_string(),
            shutdown: Arc::new(AtomicBool::new(false)),
            readers: Vec::new(),
        };
        FakeBridge {
            bridge,
            tx,
            written,
        }
    }

    /// Whether the shutdown flag is set (tests only).
    #[cfg(all(any(test, feature = "fake-link"), unix))]
    #[must_use]
    pub fn is_shut_down(&self) -> bool {
        self.shutdown.load(Ordering::SeqCst)
    }
}

/// Builds the `cargo run` invocation for a subprocess target.
fn cargo_run_command(target: &SubprocessTarget) -> StdCommand {
    let mut cmd = StdCommand::new("cargo");
    let crate_dir = target.crate_dir();
    if !crate_dir.as_os_str().is_empty()
        && crate_dir != std::path::Path::new(".")
    {
        cmd.current_dir(&crate_dir);
    }
    cmd.arg("run");
    if let Some(bin) = &target.bin {
        cmd.args(["--bin", bin]);
    }
    if let Some(triple) = &target.target {
        cmd.args(["--target", triple]);
    }
    for arg in &target.args {
        cmd.arg(arg);
    }
    cmd
}

/// Describes the link to a subprocess target for the TUI header.
fn subprocess_link_info(target: &SubprocessTarget) -> String {
    if target.path.is_empty() || target.path == "." {
        "Subprocess (cargo run)".to_string()
    } else {
        format!("Subprocess ({})", target.path)
    }
}

/// Opens the serial port, retrying once per second for up to five attempts.
fn open_serial_with_retry(
    port_path: &str,
    baud: u32,
) -> Result<serial2::SerialPort, HostError> {
    let mut attempts = 0u32;
    loop {
        match serial2::SerialPort::open(port_path, baud) {
            Ok(port) => return Ok(port),
            Err(e) => {
                attempts = attempts.saturating_add(1);
                if attempts >= 5 {
                    return Err(HostError::SerialOpen {
                        port: port_path.to_string(),
                        attempts,
                        source: e.to_string().into(),
                    });
                }
                thread::sleep(Duration::from_secs(1));
            }
        }
    }
}

/// Reads the serial link byte by byte until `shutdown` is set, forwarding
/// framed telemetry plus raw lines.
fn spawn_serial_reader(
    port: serial2::SerialPort,
    tx: Sender<BridgeMessage>,
    shutdown: Arc<AtomicBool>,
) -> JoinHandle<()> {
    thread::spawn(move || pump_serial(|buf| port.read(buf), &tx, &shutdown))
}

/// Forwards bytes from `read` until `shutdown` is set. A read that yields
/// nothing (a timeout) is retried.
fn pump_serial(
    mut read: impl FnMut(&mut [u8]) -> std::io::Result<usize>,
    tx: &Sender<BridgeMessage>,
    shutdown: &AtomicBool,
) {
    let mut reader = FrameReader::new();
    let mut raw_line_buf = Vec::new();
    let mut byte_buf = [0u8; 1];

    while !shutdown.load(Ordering::Relaxed) {
        if matches!(read(&mut byte_buf), Ok(1)) {
            let [b] = byte_buf;
            process_incoming_byte(b, &mut reader, &mut raw_line_buf, tx);
        }
    }
}

/// Forwards each stderr line of the `cargo run` child as raw console output.
fn spawn_stderr_reader(
    stderr: std::process::ChildStderr,
    tx: Sender<BridgeMessage>,
    shutdown: Arc<AtomicBool>,
) -> JoinHandle<()> {
    thread::spawn(move || {
        pump_lines(std::io::BufReader::new(stderr), &tx, &shutdown);
    })
}

/// Forwards each line of `reader` as raw console output until it ends or
/// `shutdown` is set.
fn pump_lines(
    mut reader: impl std::io::BufRead,
    tx: &Sender<BridgeMessage>,
    shutdown: &AtomicBool,
) {
    let mut line = String::new();
    while !shutdown.load(Ordering::Relaxed) {
        match reader.read_line(&mut line) {
            Ok(0) | Err(_) => break,
            Ok(_) => {
                let trimmed = line.trim_end().to_string();
                let _ = tx.send(BridgeMessage::RawConsole(trimmed));
                line.clear();
            }
        }
    }
}

/// Reads QEMU `cargo run` stdout and forwards framed telemetry plus raw lines.
fn spawn_qemu_stdout_reader(
    mut stdout: ChildStdout,
    tx_stdout: Sender<BridgeMessage>,
    shutdown: Arc<AtomicBool>,
) -> JoinHandle<()> {
    thread::spawn(move || {
        pump_stream(|buf| stdout.read(buf), &tx_stdout, &shutdown);
    })
}

/// Forwards bytes from `read` until it stops yielding one or `shutdown` is set.
fn pump_stream(
    mut read: impl FnMut(&mut [u8]) -> std::io::Result<usize>,
    tx: &Sender<BridgeMessage>,
    shutdown: &AtomicBool,
) {
    let mut reader = FrameReader::new();
    let mut raw_line_buf = Vec::new();
    let mut byte_buf = [0u8; 1];

    while !shutdown.load(Ordering::Relaxed)
        && matches!(read(&mut byte_buf), Ok(1))
    {
        let [b] = byte_buf;
        process_incoming_byte(b, &mut reader, &mut raw_line_buf, tx);
    }
}

/// Processes a single byte received from the target device.
pub fn process_incoming_byte(
    b: u8,
    reader: &mut FrameReader,
    raw_line_buf: &mut Vec<u8>,
    tx: &Sender<BridgeMessage>,
) {
    if let Some(payload) = reader.handle_byte(b) {
        match postcard::from_bytes::<Telemetry<'_>>(payload) {
            Ok(telemetry) => {
                let _ = tx.send(BridgeMessage::telemetry(&telemetry));
            }
            Err(e) => {
                let _ = tx.send(BridgeMessage::RawConsole(format!(
                    "[Host Error] Postcard decode failed: {e:?}"
                )));
            }
        }
        raw_line_buf.clear();
    } else if reader.is_idle()
        && (b.is_ascii_graphic()
            || b == b' '
            || b == b'\n'
            || b == b'\r'
            || b == b'\t')
    {
        if b == b'\n' {
            if !raw_line_buf.is_empty() {
                let line = String::from_utf8_lossy(raw_line_buf).into_owned();
                let _ = tx.send(BridgeMessage::RawConsole(line));
                raw_line_buf.clear();
            }
        } else if b != b'\r' {
            raw_line_buf.push(b);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use control_rs_ets::comms::LogMessage;

    /// One scripted read: a byte, or nothing (a timeout).
    type Step = Option<u8>;

    /// A reader over `steps` that sets `shutdown` when the script ends (or
    /// reports end of stream when `eof`) and fails the test if
    /// it is polled far beyond that.
    struct Script<'a> {
        steps: &'a [Step],
        calls: usize,
        shutdown: &'a AtomicBool,
        eof: bool,
    }

    impl<'a> Script<'a> {
        const fn new(
            steps: &'a [Step],
            shutdown: &'a AtomicBool,
            eof: bool,
        ) -> Self {
            Self {
                steps,
                calls: 0,
                shutdown,
                eof,
            }
        }

        fn next(&mut self, buf: &mut [u8]) -> usize {
            assert!(
                self.calls < self.steps.len().saturating_add(8),
                "reader polled after its end"
            );
            let step = self.steps.get(self.calls).copied();
            self.calls = self.calls.saturating_add(1);
            match step {
                Some(Some(b)) => {
                    if let Some(slot) = buf.first_mut() {
                        *slot = b;
                    }
                    1
                }
                Some(None) => 0,
                None => {
                    if !self.eof {
                        self.shutdown.store(true, Ordering::SeqCst);
                    }
                    0
                }
            }
        }
    }

    #[test]
    fn test_owned_telemetry_metadata() {
        let s = Telemetry::SuiteInfo {
            suite_id: 1,
            name: "suite1",
            description: "desc1",
            test_count: 5,
            setting_count: 2,
        };
        let owned = OwnedTelemetry::from_telemetry(&s);
        if let OwnedTelemetry::SuiteInfo {
            suite_id,
            name,
            description,
            test_count,
            setting_count,
        } = owned
        {
            assert_eq!(suite_id, 1);
            assert_eq!(name, "suite1");
            assert_eq!(description, "desc1");
            assert_eq!(test_count, 5);
            assert_eq!(setting_count, 2);
        } else {
            panic!("Expected SuiteInfo");
        }

        let t = Telemetry::TestInfo {
            suite_id: 1,
            test_id: 2,
            name: "test1",
            description: "tdesc",
        };
        let owned_t = OwnedTelemetry::from_telemetry(&t);
        if let OwnedTelemetry::TestInfo {
            suite_id,
            test_id,
            name,
            description,
        } = owned_t
        {
            assert_eq!(suite_id, 1);
            assert_eq!(test_id, 2);
            assert_eq!(name, "test1");
            assert_eq!(description, "tdesc");
        } else {
            panic!("Expected TestInfo");
        }
    }

    #[test]
    fn test_owned_telemetry_setting_info() {
        let set = Telemetry::SettingInfo {
            suite_id: 1,
            setting_id: 3,
            name: "set1",
            description: "sdesc",
            value: SettingValue::U8(10),
        };
        let owned_set = OwnedTelemetry::from_telemetry(&set);
        if let OwnedTelemetry::SettingInfo {
            suite_id,
            setting_id,
            name,
            description,
            value,
        } = owned_set
        {
            assert_eq!(suite_id, 1);
            assert_eq!(setting_id, 3);
            assert_eq!(name, "set1");
            assert_eq!(description, "sdesc");
            assert!(matches!(value, SettingValue::U8(10)));
        } else {
            panic!("Expected SettingInfo");
        }
    }

    #[test]
    fn test_owned_telemetry_simple() {
        assert!(matches!(
            OwnedTelemetry::from_telemetry(&Telemetry::DiscoveryComplete),
            OwnedTelemetry::DiscoveryComplete
        ));
        assert!(matches!(
            OwnedTelemetry::from_telemetry(&Telemetry::TestStateChange {
                suite_id: 1,
                test_id: 2,
                state: control_rs_ets::comms::TestState::Passed
            }),
            OwnedTelemetry::TestStateChange {
                suite_id: 1,
                test_id: 2,
                state: control_rs_ets::comms::TestState::Passed
            }
        ));
        assert!(matches!(
            OwnedTelemetry::from_telemetry(&Telemetry::MetricReport {
                suite_id: 1,
                test_id: 2,
                cycles: 10,
                time_us: 20,
                stack_peak: 30
            }),
            OwnedTelemetry::MetricReport {
                suite_id: 1,
                test_id: 2,
                cycles: 10,
                time_us: 20,
                stack_peak: 30
            }
        ));
    }

    #[test]
    fn test_owned_telemetry_log() {
        let log = Telemetry::Log(LogMessage {
            timestamp_us: 100,
            suite_id: 1,
            test_id: 2,
            payload: "hello",
        });
        let owned_log = OwnedTelemetry::from_telemetry(&log);
        if let OwnedTelemetry::Log {
            timestamp_us,
            suite_id,
            test_id,
            payload,
        } = owned_log
        {
            assert_eq!(timestamp_us, 100);
            assert_eq!(suite_id, 1);
            assert_eq!(test_id, 2);
            assert_eq!(payload, "hello");
        } else {
            panic!("Expected Log");
        }
    }

    #[test]
    fn test_owned_telemetry_panic() {
        let panic_tel = Telemetry::TargetPanic {
            message: "panic message",
            file: "main.rs",
            line: 5,
        };
        let owned_panic = OwnedTelemetry::from_telemetry(&panic_tel);
        if let OwnedTelemetry::TargetPanic {
            message,
            file,
            line,
        } = owned_panic
        {
            assert_eq!(message, "panic message");
            assert_eq!(file, "main.rs");
            assert_eq!(line, 5);
        } else {
            panic!("Expected TargetPanic");
        }
    }

    #[test]
    fn test_process_incoming_byte() {
        let (tx, rx) = std::sync::mpsc::channel();
        let mut reader = FrameReader::new();
        let mut raw_line_buf = Vec::new();

        process_incoming_byte(b'h', &mut reader, &mut raw_line_buf, &tx);
        process_incoming_byte(b'i', &mut reader, &mut raw_line_buf, &tx);
        process_incoming_byte(b'\n', &mut reader, &mut raw_line_buf, &tx);

        let msg = rx.try_recv().unwrap();
        if let BridgeMessage::RawConsole(s) = msg {
            assert_eq!(s, "hi");
        } else {
            panic!("Expected RawConsole");
        }

        let mut buf = [0u8; 128];
        let size = control_rs_ets::comms::frame_telemetry(
            &Telemetry::DiscoveryComplete,
            &mut buf,
        )
        .unwrap();
        for &b in buf.get(..size).unwrap() {
            process_incoming_byte(b, &mut reader, &mut raw_line_buf, &tx);
        }

        let msg2 = rx.try_recv().unwrap();
        assert!(matches!(
            msg2,
            BridgeMessage::Telemetry(OwnedTelemetry::DiscoveryComplete)
        ));
    }

    #[test]
    fn test_process_incoming_byte_corrupted_payload_and_special_chars() {
        let (tx, rx) = channel();
        let mut reader = FrameReader::new();
        let mut raw_buf = Vec::new();

        // 1. Send invalid postcard payload inside a valid frame
        let invalid_payload = [0xFF, 0xFF, 0xFF];
        let mut frame_buf = [0u8; 32];
        let frame_len = control_rs_ets::comms::FrameEncoder::frame_payload(
            &invalid_payload,
            &mut frame_buf,
        )
        .unwrap();

        for &b in frame_buf.get(..frame_len).unwrap() {
            process_incoming_byte(b, &mut reader, &mut raw_buf, &tx);
        }

        let msg = rx.try_recv().unwrap();
        if let BridgeMessage::RawConsole(err) = msg {
            assert!(err.contains("Postcard decode failed"));
        } else {
            panic!("expected raw console error");
        }

        // 2. Send \r and \t and newline
        process_incoming_byte(b'a', &mut reader, &mut raw_buf, &tx);
        process_incoming_byte(b'\r', &mut reader, &mut raw_buf, &tx);
        process_incoming_byte(b'\t', &mut reader, &mut raw_buf, &tx);
        process_incoming_byte(b'\n', &mut reader, &mut raw_buf, &tx);
        let msg2 = rx.try_recv().unwrap();
        if let BridgeMessage::RawConsole(line) = msg2 {
            assert_eq!(line, "a\t");
        } else {
            panic!("expected raw console line");
        }
    }

    fn console_lines(rx: &Receiver<BridgeMessage>) -> Vec<String> {
        let mut lines = Vec::new();
        while let Ok(msg) = rx.try_recv() {
            match msg {
                BridgeMessage::RawConsole(line) => lines.push(line),
                BridgeMessage::Telemetry(_) => panic!("unexpected telemetry"),
            }
        }
        lines
    }

    #[test]
    fn stray_control_bytes_and_carriage_returns_stay_out_of_console_lines() {
        let (tx, rx) = channel();
        let mut reader = FrameReader::new();
        let mut raw = Vec::new();
        for &b in b"a b\x01c\r\td\n" {
            process_incoming_byte(b, &mut reader, &mut raw, &tx);
        }
        assert_eq!(console_lines(&rx), ["a bc\td"]);
    }

    #[test]
    fn cargo_run_commands_run_in_the_crate_directory_only_when_it_differs() {
        let dir = |path: &str| {
            cargo_run_command(&SubprocessTarget::new(path))
                .get_current_dir()
                .map(std::path::Path::to_path_buf)
        };
        assert_eq!(
            dir("examples/qemu"),
            Some(std::path::PathBuf::from("examples/qemu"))
        );
        assert_eq!(dir("."), None);
        assert_eq!(dir(""), None);

        let sub = SubprocessTarget::new(".")
            .with_bin("b")
            .with_target("t")
            .with_arg("--release");
        let cmd = cargo_run_command(&sub);
        let args: Vec<_> = cmd
            .get_args()
            .map(|a| a.to_string_lossy().into_owned())
            .collect();
        assert_eq!(args, ["run", "--bin", "b", "--target", "t", "--release"]);
    }

    #[test]
    fn link_info_names_the_subprocess_path() {
        assert_eq!(
            subprocess_link_info(&SubprocessTarget::new(".")),
            "Subprocess (cargo run)"
        );
        assert_eq!(
            subprocess_link_info(&SubprocessTarget::new("")),
            "Subprocess (cargo run)"
        );
        assert_eq!(
            subprocess_link_info(&SubprocessTarget::new("crates/x")),
            "Subprocess (crates/x)"
        );
    }

    #[test]
    fn opening_a_missing_serial_port_gives_up_after_five_attempts() {
        let Err(err) =
            open_serial_with_retry("/dev/control-rs-no-such-port", 115_200)
        else {
            panic!("the port cannot exist");
        };
        assert!(
            matches!(err, HostError::SerialOpen { attempts: 5, .. }),
            "{err}"
        );
    }

    fn steps(text: &str) -> Vec<Step> {
        text.bytes().map(Some).collect()
    }

    #[test]
    fn serial_bytes_become_console_lines_and_timeouts_are_retried() {
        let (tx, rx) = channel();
        let shutdown = AtomicBool::new(false);
        let mut script = steps("h");
        script.push(None);
        script.extend(steps("i\n"));
        let mut reader = Script::new(&script, &shutdown, false);
        pump_serial(|buf| Ok(reader.next(buf)), &tx, &shutdown);
        assert_eq!(console_lines(&rx), ["hi"]);
    }

    #[test]
    fn a_serial_reader_that_is_already_shut_down_reads_nothing() {
        let (tx, rx) = channel();
        let shutdown = AtomicBool::new(true);
        pump_serial(
            |_: &mut [u8]| panic!("read after shutdown"),
            &tx,
            &shutdown,
        );
        assert_eq!(console_lines(&rx), Vec::<String>::new());
    }

    #[test]
    fn stream_bytes_become_console_lines_until_the_stream_ends() {
        let (tx, rx) = channel();
        let shutdown = AtomicBool::new(false);
        let script = steps("ok\nnext\n");
        let mut reader = Script::new(&script, &shutdown, true);
        let done = AtomicBool::new(false);
        let stuck = AtomicBool::new(false);
        // A pump that ignores the end of the stream never returns, and a
        // `||` in its loop test would never even poll the reader, so a
        // watchdog turns that hang into a failure.
        thread::scope(|scope| {
            scope.spawn(|| {
                pump_stream(|buf| Ok(reader.next(buf)), &tx, &shutdown);
                done.store(true, Ordering::SeqCst);
            });
            scope.spawn(|| {
                let deadline = std::time::Instant::now()
                    + std::time::Duration::from_secs(5);
                while !done.load(Ordering::SeqCst)
                    && std::time::Instant::now() < deadline
                {
                    thread::sleep(std::time::Duration::from_millis(10));
                }
                if !done.load(Ordering::SeqCst) {
                    stuck.store(true, Ordering::SeqCst);
                    shutdown.store(true, Ordering::SeqCst);
                }
            });
        });
        assert!(
            !stuck.load(Ordering::SeqCst),
            "the pump kept running after the stream ended"
        );
        assert_eq!(console_lines(&rx), ["ok", "next"]);
    }

    #[test]
    fn a_stream_reader_that_is_already_shut_down_reads_nothing() {
        let (tx, rx) = channel();
        let shutdown = AtomicBool::new(true);
        pump_stream(
            |_: &mut [u8]| panic!("read after shutdown"),
            &tx,
            &shutdown,
        );
        assert_eq!(console_lines(&rx), Vec::<String>::new());
    }

    #[test]
    fn stderr_lines_are_forwarded_trimmed_until_shutdown() {
        let (tx, rx) = channel();
        let shutdown = AtomicBool::new(false);
        pump_lines(std::io::Cursor::new("a\nb  \n"), &tx, &shutdown);
        assert_eq!(console_lines(&rx), ["a", "b"]);

        let (tx, rx) = channel();
        let shutdown = AtomicBool::new(true);
        pump_lines(std::io::Cursor::new("ignored\n"), &tx, &shutdown);
        assert_eq!(console_lines(&rx), Vec::<String>::new());
    }

    #[cfg(unix)]
    #[test]
    fn dropping_a_bridge_stops_its_readers() {
        let bridge = ETSBridge::fake(None).bridge;
        let shutdown = Arc::clone(&bridge.shutdown);
        assert!(!shutdown.load(Ordering::SeqCst));
        assert!(!bridge.is_shut_down());
        drop(bridge);
        assert!(shutdown.load(Ordering::SeqCst));
    }

    #[test]
    fn revision_mismatch_refused() {
        use crate::session::SessionState;
        let mut state = SessionState::new();
        let old = Telemetry::TargetInfo {
            protocol_version: 1,
            board_id: 0,
            core_clock_hz: 0,
            fpu_flags: 0,
        };
        let _ = state.handle_message(BridgeMessage::telemetry(&old));
        assert_eq!(state.protocol_mismatch, Some(1));
        assert!(state.exit_loop, "a revision-1 target ends the session");
        assert_eq!(control_rs_ets::comms::PROTOCOL_VERSION, 2);
    }

    #[test]
    fn loop_frames_convert_to_owned_telemetry() {
        let info = Telemetry::LoopInfo {
            suite_id: 1,
            test_id: 2,
            name: "n",
            description: "d",
            input_type: "f32",
            output_type: "u8",
        };
        assert!(matches!(
            OwnedTelemetry::from_telemetry(&info),
            OwnedTelemetry::LoopInfo { input_type, .. } if input_type == "f32"
        ));
        let sample = Telemetry::LoopSample {
            suite_id: 1,
            test_id: 2,
            seq: 9,
            payload: &[1, 2, 3],
        };
        assert!(matches!(
            OwnedTelemetry::from_telemetry(&sample),
            OwnedTelemetry::LoopSample { seq: 9, payload, .. } if payload == [1, 2, 3]
        ));
    }

    #[test]
    fn run_frames_convert_to_owned_telemetry() {
        let report = Telemetry::TeardownReport {
            suite_id: 1,
            test_id: 2,
            ok: false,
            message: None,
        };
        assert!(matches!(
            OwnedTelemetry::from_telemetry(&report),
            OwnedTelemetry::TeardownReport {
                ok: false,
                message: None,
                ..
            }
        ));
        let stats = Telemetry::LoopStats {
            suite_id: 1,
            test_id: 2,
            steps: 4,
            time_us: 5,
        };
        assert!(matches!(
            OwnedTelemetry::from_telemetry(&stats),
            OwnedTelemetry::LoopStats {
                steps: 4,
                time_us: 5,
                ..
            }
        ));
        let bounded = Telemetry::LoopState {
            suite_id: 1,
            test_id: 2,
            state: control_rs_ets::comms::LoopRunState::Bounded,
            message: Some("m"),
        };
        assert!(matches!(
            OwnedTelemetry::from_telemetry(&bounded),
            OwnedTelemetry::LoopState { message: Some(m), .. } if m == "m"
        ));
        let suite = Telemetry::LifecycleSuite {
            suite_id: 1,
            loop_count: 1,
        };
        assert!(matches!(
            OwnedTelemetry::from_telemetry(&suite),
            OwnedTelemetry::LifecycleSuite { loop_count: 1, .. }
        ));
    }
}
