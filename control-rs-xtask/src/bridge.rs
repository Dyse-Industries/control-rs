//! Bridge module to interface host computer with the target device.
//! Manages spawning and monitoring the execution environments (QEMU or Serial).

use std::io::{Read, Write as IoWrite};
use std::process::{Child, Command as StdCommand, Stdio};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread;

use control_rs_hil::comms::{Command, FrameReader, LogMessage, Telemetry};
use control_rs_hil::settings::SettingValue;

type BridgeResult<T> = Result<T, Box<dyn std::error::Error>>;
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

/// Message type sent from the background reader thread to the TUI.
pub enum BridgeMessage {
    /// Raw console output (stdout/stderr) from the target/QEMU.
    RawConsole(String),
    /// Telemetry parsed from target.
    Telemetry(Telemetry<'static>),
}

/// Target QEMU architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QemuArch {
    /// ARM architecture.
    Arm,
    /// RISC-V architecture.
    Riscv,
}

/// Target details for QEMU.
pub struct QemuTargetDetails {
    /// Binary name of the example.
    pub binary_name: &'static str,
    /// Human readable description of the target.
    pub description: &'static str,
    /// Human readable description of the execution environment.
    pub execution_env: &'static str,
    /// Target triple used by rustc/cargo.
    pub target_triple: &'static str,
}

/// Target execution platform.
#[derive(Debug, Clone)]
pub enum Target {
    /// QEMU emulator target.
    QemuSemihosting {
        /// Target architecture.
        arch: QemuArch,
    },
    /// Serial connection target.
    Serial {
        /// Serial port path (e.g. `/dev/ttyACM0`).
        port: String,
        /// Baud rate (e.g. `115200`).
        baud: u32,
    },
}

/// Host bridge to manage the target execution environment (QEMU or Serial).
pub struct ServerBridge {
    inner: BridgeInner,
    link_info: String,
    rx_from_target: Receiver<BridgeMessage>,
    target_info: String,
}

impl QemuArch {
    /// Gets the configuration and target details for this architecture.
    #[must_use]
    pub const fn details(&self) -> QemuTargetDetails {
        match self {
            Self::Arm => QemuTargetDetails {
                binary_name: "control-rs-qemu-arm",
                description: "QEMU (cortex-m7)",
                execution_env: "Semihosting (mps2-an500)",
                target_triple: "thumbv7em-none-eabihf",
            },
            Self::Riscv => QemuTargetDetails {
                binary_name: "control-rs-qemu-risc-v",
                description: "QEMU (risc-v32)",
                execution_env: "Semihosting (virt)",
                target_triple: "riscv32imac-unknown-none-elf",
            },
        }
    }
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
    /// # Errors
    ///
    /// Returns an error if opening the serial port fails after 5 attempts, or if QEMU fails to start.
    ///
    /// # Panics
    ///
    /// Panics if the serial port reference is unexpectedly missing after loop execution.
    #[allow(clippy::too_many_lines)]
    pub fn new(target: Target, elf_path: Option<&str>) -> BridgeResult<Self> {
        let (tx, rx) = channel();

        match target {
            Target::Serial {
                port: port_path,
                baud,
            } => {
                let mut port = None;
                let mut attempts = 0u32;
                while port.is_none() {
                    match serial2::SerialPort::open(&port_path, baud) {
                        Ok(p) => port = Some(p),
                        Err(e) => {
                            attempts = attempts.saturating_add(1);
                            if attempts >= 5 {
                                return Err(format!(
                                    "Failed to open serial port '{port_path}' after 5 attempts (5 seconds): {e}"
                                ).into());
                            }
                            thread::sleep(std::time::Duration::from_secs(1));
                        }
                    }
                }
                let port = port.unwrap();

                let port_clone = port
                    .try_clone()
                    .map_err(|e| format!("Failed to clone serial port: {e}"))?;

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
            Target::QemuSemihosting { arch } => {
                let elf =
                    elf_path.ok_or("ELF path is required for QEMU target")?;
                Self::new_qemu_inner(elf, arch, tx, rx)
            }
        }
    }
    fn new_qemu_inner(
        _elf_path: &str,
        arch: QemuArch,
        tx: Sender<BridgeMessage>,
        rx: Receiver<BridgeMessage>,
    ) -> BridgeResult<Self> {
        let details = arch.details();

        let mut child = StdCommand::new("cargo")
            .current_dir("examples/qemu")
            .args([
                "run",
                "--bin",
                details.binary_name,
                "--target",
                details.target_triple,
                "--release",
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to spawn cargo run process: {e}"))?;

        let stdin = child.stdin.take().ok_or("Failed to open stdin")?;
        let stdout = child.stdout.take().ok_or("Failed to open stdout")?;
        let stderr = child.stderr.take().ok_or("Failed to open stderr")?;

        // Spawn stdout reader thread
        let tx_stdout = tx.clone();
        thread::spawn(move || {
            let mut reader = FrameReader::new();
            let mut raw_line_buf = Vec::new();
            let mut byte_buf = [0u8; 1];
            let mut stdout = stdout;

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

        // Spawn stderr reader thread
        let tx_stderr = tx;
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
                let _ = tx_stderr.send(BridgeMessage::RawConsole(trimmed));
                line.clear();
            }
        });

        Ok(Self {
            inner: BridgeInner::Qemu { child, stdin },
            rx_from_target: rx,
            target_info: details.description.to_string(),
            link_info: details.execution_env.to_string(),
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
    /// Returns an error if writing to or flushing the underlying writer fails.
    ///
    /// # Panics
    ///
    /// Panics if serialization of the command fails.
    pub fn send_command(
        &mut self,
        cmd: &Command,
    ) -> Result<(), std::io::Error> {
        let mut payload = postcard::to_allocvec(cmd).unwrap();
        let mut frame = Vec::new();
        frame.push(0xAA);
        frame.push(0x55);
        let len = u16::try_from(payload.len()).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, e)
        })?;
        frame.push((len >> 8) as u8);
        frame.push((len & 0xFF) as u8);

        let crc = crc::Crc::<u16>::new(&crc::CRC_16_IBM_SDLC);
        let crc_value = crc.checksum(&payload);
        frame.append(&mut payload);
        frame.push((crc_value >> 8) as u8);
        frame.push((crc_value & 0xFF) as u8);

        match &mut self.inner {
            BridgeInner::Qemu { stdin, .. } => {
                stdin.write_all(&frame)?;
                stdin.flush()?;
            }
            BridgeInner::Serial { port } => {
                port.write_all(&frame)?;
                port.flush()?;
            }
        }
        Ok(())
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

impl Target {
    /// Parses target parameters from CLI arguments.
    ///
    /// # Errors
    ///
    /// Returns an error if the target string is unknown or the QEMU architecture is unknown.
    #[allow(clippy::type_complexity)]
    pub fn parse(
        args: &[String],
        default_qemu_arch: &str,
        default_teensy_port: &str,
    ) -> Result<Option<Self>, String> {
        let target_str = args.get(2).map_or("qemu", String::as_str);
        let arch_or_port = args.get(3).map(String::as_str);
        let baud_str = args.get(4).map(String::as_str);

        match target_str {
            "qemu" => {
                let arch = arch_or_port.unwrap_or(default_qemu_arch);
                match arch {
                    "arm" => Ok(Some(Self::QemuSemihosting {
                        arch: QemuArch::Arm,
                    })),
                    "riscv" | "risc-v" => Ok(Some(Self::QemuSemihosting {
                        arch: QemuArch::Riscv,
                    })),
                    "all" => Ok(None),
                    _ => Err(format!("Unknown QEMU architecture: {arch}")),
                }
            }
            "teensy" => {
                let port = arch_or_port.map_or_else(
                    || default_teensy_port.to_string(),
                    String::from,
                );
                let baud =
                    baud_str.and_then(|b| b.parse().ok()).unwrap_or(115_200);
                Ok(Some(Self::Serial { port, baud }))
            }
            _ => Err(format!("Unknown target: {target_str}")),
        }
    }

    /// Helper to create a QEMU ARM target.
    #[must_use]
    pub const fn qemu_arm() -> Self {
        Self::QemuSemihosting {
            arch: QemuArch::Arm,
        }
    }

    /// Helper to create a QEMU RISC-V target.
    #[must_use]
    pub const fn qemu_riscv() -> Self {
        Self::QemuSemihosting {
            arch: QemuArch::Riscv,
        }
    }

    /// Helper to create a Serial target.
    #[must_use]
    pub const fn serial(port: String, baud: u32) -> Self {
        Self::Serial { port, baud }
    }
}

fn leak_str(s: &str) -> &'static str {
    Box::leak(s.to_string().into_boxed_str())
}

#[allow(clippy::too_many_arguments)]
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

#[allow(clippy::too_many_arguments)]
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
fn make_telemetry_owned(tel: &Telemetry<'_>) -> Telemetry<'static> {
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

/// Processes a single byte received from the target device.
fn process_incoming_byte(
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
