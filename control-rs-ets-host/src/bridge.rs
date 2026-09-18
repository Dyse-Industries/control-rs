//! Bridge module to interface host computer with the target device.
//! Manages spawning and monitoring the execution environments (QEMU or Serial).

use std::io::{Read, Write as IoWrite};
use std::process::{Child, ChildStdout, Command as StdCommand, Stdio};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread;

use control_rs_ets::comms::{Command, FrameReader, LogMessage, Telemetry};
use control_rs_ets::settings::SettingValue;

use crate::error::HostError;
use crate::target::{SubprocessTarget, Target};

type WaitResult = Result<Option<std::process::ExitStatus>, std::io::Error>;

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
}

/// Message type sent from the background reader thread to the host controller or UI.
pub enum BridgeMessage {
    /// Raw console output (stdout/stderr) from the target/QEMU.
    RawConsole(String),
    /// Telemetry parsed from target.
    Telemetry(Telemetry<'static>),
}

/// Host driver (`ServerBridge`) for virtual ETS (QEMU) and ETS (board).
pub struct ServerBridge {
    inner: BridgeInner,
    link_info: String,
    rx_from_target: Receiver<BridgeMessage>,
    target_info: String,
}

impl ServerBridge {
    /// Terminate QEMU (no-op for serial).
    pub fn kill(&mut self) {
        match &mut self.inner {
            BridgeInner::Qemu { child, .. } => {
                let _ = child.kill();
            }
            BridgeInner::Serial { .. } => {}
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
        _elf_path: Option<&str>,
        inherit_stderr: bool,
    ) -> Result<Self, HostError> {
        let (tx, rx) = channel();

        match target {
            Target::Serial {
                port: port_path,
                baud,
            } => {
                let mut port = None;
                let mut attempts = 0u32;
                let mut last_err = String::new();
                while port.is_none() {
                    match serial2::SerialPort::open(&port_path, baud) {
                        Ok(p) => port = Some(p),
                        Err(e) => {
                            attempts = attempts.saturating_add(1);
                            last_err = e.to_string();
                            if attempts >= 5 {
                                return Err(HostError::SerialOpen {
                                    port: port_path,
                                    attempts,
                                    source: last_err.into(),
                                });
                            }
                            thread::sleep(std::time::Duration::from_secs(1));
                        }
                    }
                }
                let port = match port {
                    Some(p) => p,
                    None => {
                        return Err(HostError::SerialOpen {
                            port: port_path,
                            attempts,
                            source: last_err.into(),
                        });
                    }
                };

                let port_clone =
                    port.try_clone().map_err(|e| HostError::SerialClone {
                        source: e.to_string().into(),
                    })?;

                // Spawn serial reader thread
                thread::spawn(move || {
                    let mut reader = FrameReader::new();
                    let mut raw_line_buf = Vec::new();
                    let mut byte_buf = [0u8; 1];

                    loop {
                        match port_clone.read(&mut byte_buf) {
                            Ok(1) => {
                                let b = byte_buf[0];
                                process_incoming_byte(
                                    b,
                                    &mut reader,
                                    &mut raw_line_buf,
                                    &tx,
                                );
                            }
                            _ => {
                                thread::sleep(
                                    std::time::Duration::from_millis(1),
                                );
                            }
                        }
                    }
                });

                Ok(Self {
                    inner: BridgeInner::Serial { port },
                    rx_from_target: rx,
                    target_info: "Teensy 4.0 (Cortex-M7)".to_string(),
                    link_info: format!("USB CDC ({port_path})"),
                })
            }
            Target::Subprocess(sub) => {
                Self::new_subprocess_inner(&sub, tx, rx, inherit_stderr)
            }
        }
    }

    fn new_subprocess_inner(
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

        let mut child = cmd
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

        if inherit_stderr {
            spawn_qemu_stdout_reader(stdout, tx);
        } else {
            spawn_qemu_stdout_reader(stdout, tx.clone());
            let stderr =
                child.stderr.take().ok_or_else(|| HostError::Spawn {
                    source: "Failed to open stderr".into(),
                })?;
            thread::spawn(move || {
                let mut reader = std::io::BufReader::new(stderr);
                let mut line = String::new();
                while let Ok(n) =
                    std::io::BufRead::read_line(&mut reader, &mut line)
                {
                    if n == 0 {
                        break;
                    }
                    let trimmed = line.trim_end().to_string();
                    let _ = tx.send(BridgeMessage::RawConsole(trimmed));
                    line.clear();
                }
            });
        }

        let target_desc = target.display_name();
        let link_desc = if target.path.is_empty() || target.path == "." {
            "Subprocess (cargo run)".to_string()
        } else {
            format!("Subprocess ({})", target.path)
        };

        Ok(Self {
            inner: BridgeInner::Qemu { child, stdin },
            rx_from_target: rx,
            target_info: target_desc,
            link_info: link_desc,
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
    pub fn send_command(&mut self, cmd: &Command) -> Result<(), HostError> {
        let mut payload =
            postcard::to_allocvec(cmd).map_err(|e| HostError::Transport {
                source: format!("Failed to serialize command: {e}").into(),
            })?;
        let mut frame = Vec::new();
        frame.push(0xAA);
        frame.push(0x55);
        let len =
            u16::try_from(payload.len()).map_err(|e| HostError::Transport {
                source: format!("Payload too large: {e}").into(),
            })?;
        frame.push((len >> 8) as u8);
        frame.push((len & 0xFF) as u8);

        let crc = crc::Crc::<u16>::new(&crc::CRC_16_IBM_SDLC);
        let crc_value = crc.checksum(&payload);
        frame.append(&mut payload);
        frame.push((crc_value >> 8) as u8);
        frame.push((crc_value & 0xFF) as u8);

        let res = match &mut self.inner {
            BridgeInner::Qemu { stdin, .. } => {
                stdin.write_all(&frame).and_then(|()| stdin.flush())
            }
            BridgeInner::Serial { port } => {
                port.write_all(&frame).and_then(|()| port.flush())
            }
        };

        res.map_err(|e| HostError::Transport {
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
        }
    }
}

fn leak_str(s: &str) -> &'static str {
    Box::leak(s.to_string().into_boxed_str())
}

fn make_suite_info(
    suite_id: u16,
    name: &str,
    description: &str,
    test_count: u16,
    setting_count: u16,
) -> Telemetry<'static> {
    Telemetry::SuiteInfo {
        suite_id,
        name: leak_str(name),
        description: leak_str(description),
        test_count,
        setting_count,
    }
}

fn make_test_info(
    suite_id: u16,
    test_id: u16,
    name: &str,
    description: &str,
) -> Telemetry<'static> {
    Telemetry::TestInfo {
        suite_id,
        test_id,
        name: leak_str(name),
        description: leak_str(description),
    }
}

fn make_setting_info(
    suite_id: u16,
    setting_id: u16,
    name: &str,
    description: &str,
    value: SettingValue,
) -> Telemetry<'static> {
    Telemetry::SettingInfo {
        suite_id,
        setting_id,
        name: leak_str(name),
        description: leak_str(description),
        value,
    }
}

fn make_log_info(msg: &LogMessage<'_>) -> Telemetry<'static> {
    Telemetry::Log(LogMessage {
        timestamp_us: msg.timestamp_us,
        suite_id: msg.suite_id,
        test_id: msg.test_id,
        payload: leak_str(msg.payload),
    })
}

fn make_target_panic(
    message: &str,
    file: &str,
    line: u32,
) -> Telemetry<'static> {
    Telemetry::TargetPanic {
        message: leak_str(message),
        file: leak_str(file),
        line,
    }
}

/// Converts a Telemetry object references into static owned equivalents.
#[must_use]
pub fn make_telemetry_owned(tel: &Telemetry<'_>) -> Telemetry<'static> {
    match *tel {
        Telemetry::SuiteInfo {
            suite_id,
            name,
            description,
            test_count,
            setting_count,
        } => make_suite_info(
            suite_id,
            name,
            description,
            test_count,
            setting_count,
        ),
        Telemetry::TestInfo {
            suite_id,
            test_id,
            name,
            description,
        } => make_test_info(suite_id, test_id, name, description),
        Telemetry::SettingInfo {
            suite_id,
            setting_id,
            name,
            description,
            value,
        } => make_setting_info(suite_id, setting_id, name, description, value),
        Telemetry::DiscoveryComplete => Telemetry::DiscoveryComplete,
        Telemetry::TestStateChange {
            suite_id,
            test_id,
            state,
        } => Telemetry::TestStateChange {
            suite_id,
            test_id,
            state,
        },
        Telemetry::MetricReport {
            suite_id,
            test_id,
            cycles,
            time_us,
            stack_peak,
        } => Telemetry::MetricReport {
            suite_id,
            test_id,
            cycles,
            time_us,
            stack_peak,
        },
        Telemetry::Log(ref msg) => make_log_info(msg),
        Telemetry::TargetPanic {
            message,
            file,
            line,
        } => make_target_panic(message, file, line),
    }
}

/// Reads QEMU `cargo run` stdout and forwards framed telemetry plus raw lines.
fn spawn_qemu_stdout_reader(
    mut stdout: ChildStdout,
    tx_stdout: Sender<BridgeMessage>,
) {
    thread::spawn(move || {
        let mut reader = FrameReader::new();
        let mut raw_line_buf = Vec::new();
        let mut byte_buf = [0u8; 1];

        while matches!(stdout.read(&mut byte_buf), Ok(1)) {
            let b = byte_buf[0];
            process_incoming_byte(
                b,
                &mut reader,
                &mut raw_line_buf,
                &tx_stdout,
            );
        }
    });
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
                let owned_telemetry = make_telemetry_owned(&telemetry);
                let _ = tx.send(BridgeMessage::Telemetry(owned_telemetry));
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

    #[test]
    fn test_make_telemetry_owned_metadata() {
        let s = Telemetry::SuiteInfo {
            suite_id: 1,
            name: "suite1",
            description: "desc1",
            test_count: 5,
            setting_count: 2,
        };
        let owned = make_telemetry_owned(&s);
        if let Telemetry::SuiteInfo {
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
        let owned_t = make_telemetry_owned(&t);
        if let Telemetry::TestInfo {
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
    fn test_make_telemetry_owned_setting_info() {
        let set = Telemetry::SettingInfo {
            suite_id: 1,
            setting_id: 3,
            name: "set1",
            description: "sdesc",
            value: SettingValue::U8(10),
        };
        let owned_set = make_telemetry_owned(&set);
        if let Telemetry::SettingInfo {
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
    fn test_make_telemetry_owned_simple() {
        assert!(matches!(
            make_telemetry_owned(&Telemetry::DiscoveryComplete),
            Telemetry::DiscoveryComplete
        ));
        assert!(matches!(
            make_telemetry_owned(&Telemetry::TestStateChange {
                suite_id: 1,
                test_id: 2,
                state: control_rs_ets::comms::TestState::Passed
            }),
            Telemetry::TestStateChange {
                suite_id: 1,
                test_id: 2,
                state: control_rs_ets::comms::TestState::Passed
            }
        ));
        assert!(matches!(
            make_telemetry_owned(&Telemetry::MetricReport {
                suite_id: 1,
                test_id: 2,
                cycles: 10,
                time_us: 20,
                stack_peak: 30
            }),
            Telemetry::MetricReport {
                suite_id: 1,
                test_id: 2,
                cycles: 10,
                time_us: 20,
                stack_peak: 30
            }
        ));
    }

    #[test]
    fn test_make_telemetry_owned_log() {
        let log = Telemetry::Log(LogMessage {
            timestamp_us: 100,
            suite_id: 1,
            test_id: 2,
            payload: "hello",
        });
        let owned_log = make_telemetry_owned(&log);
        if let Telemetry::Log(LogMessage {
            timestamp_us,
            suite_id,
            test_id,
            payload,
        }) = owned_log
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
    fn test_make_telemetry_owned_panic() {
        let panic_tel = Telemetry::TargetPanic {
            message: "panic message",
            file: "main.rs",
            line: 5,
        };
        let owned_panic = make_telemetry_owned(&panic_tel);
        if let Telemetry::TargetPanic {
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
            BridgeMessage::Telemetry(Telemetry::DiscoveryComplete)
        ));
    }

    #[test]
    fn test_process_incoming_byte_corrupted_payload_and_special_chars() {
        let (tx, rx) = channel();
        let mut reader = FrameReader::new();
        let mut raw_buf = Vec::new();

        // 1. Send invalid postcard payload inside a valid frame
        let invalid_payload = [0xFF, 0xFF, 0xFF];
        let len = u16::try_from(invalid_payload.len()).unwrap();
        let crc = crc::Crc::<u16>::new(&crc::CRC_16_IBM_SDLC);
        let crc_val = crc.checksum(&invalid_payload);
        let mut frame = vec![0xAA, 0x55, (len >> 8) as u8, (len & 0xFF) as u8];
        frame.extend_from_slice(&invalid_payload);
        frame.push((crc_val >> 8) as u8);
        frame.push((crc_val & 0xFF) as u8);

        for b in frame {
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
}
