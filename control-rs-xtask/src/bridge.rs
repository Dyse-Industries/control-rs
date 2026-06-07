use std::io::{Read, Write as IoWrite};
use std::process::{Child, Command as StdCommand, Stdio};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;

use control_rs_hil::comms::{Command, FrameReader, LogMessage, Telemetry};

/// Message type sent from the background reader thread to the TUI.
pub enum BridgeMessage {
    /// Telemetry parsed from target.
    Telemetry(Telemetry<'static>),
    /// Raw console output (stdout/stderr) from the target/QEMU.
    RawConsole(String),
}

#[derive(Debug, Clone)]
pub enum Target {
    Qemu,
    Serial { port: String, baud: u32 },
}

enum BridgeInner {
    Qemu {
        child: Child,
        stdin: std::process::ChildStdin,
    },
    Serial {
        port: Box<dyn serialport::SerialPort>,
    },
}

/// Host bridge to manage the target execution environment (QEMU or Serial).
pub struct QemuBridge {
    inner: BridgeInner,
    rx_from_target: Receiver<BridgeMessage>,
    target_info: String,
    link_info: String,
}

impl QemuBridge {
    /// Spawns a new QEMU process or connects to a serial device.
    pub fn new(
        elf_path: &str,
        target: Target,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let (tx, rx) = channel();

        match target {
            Target::Serial {
                port: port_path,
                baud,
            } => {
                let mut port = None;
                let mut attempts = 0;
                while port.is_none() {
                    match serialport::new(&port_path, baud)
                        .timeout(std::time::Duration::from_millis(10))
                        .open()
                    {
                        Ok(p) => port = Some(p),
                        Err(e) => {
                            attempts += 1;
                            if attempts >= 5 {
                                return Err(format!(
                                    "Failed to open serial port '{}' after 5 attempts (5 seconds): {}",
                                    port_path, e
                                ).into());
                            }
                            thread::sleep(std::time::Duration::from_millis(
                                1000,
                            ));
                        }
                    }
                }
                let port = port.unwrap();

                let mut port_clone = port.try_clone().map_err(|e| {
                    format!("Failed to clone serial port: {}", e)
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
                                if let Some(payload) = reader.handle_byte(b) {
                                    if let Ok(telemetry) =
                                        postcard::from_bytes::<Telemetry<'_>>(
                                            payload,
                                        )
                                    {
                                        let owned_telemetry =
                                            make_telemetry_owned(&telemetry);
                                        let _ =
                                            tx.send(BridgeMessage::Telemetry(
                                                owned_telemetry,
                                            ));
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
                                            let line = String::from_utf8_lossy(
                                                &raw_line_buf,
                                            )
                                            .into_owned();
                                            let _ = tx.send(
                                                BridgeMessage::RawConsole(line),
                                            );
                                            raw_line_buf.clear();
                                        }
                                    } else if b != b'\r' {
                                        raw_line_buf.push(b);
                                    }
                                }
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
                    link_info: format!("USB CDC ({})", port_path),
                })
            }
            Target::Qemu => Self::new_qemu_inner(elf_path, tx, rx),
        }
    }

    fn new_qemu_inner(
        elf_path: &str,
        tx: Sender<BridgeMessage>,
        rx: Receiver<BridgeMessage>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut child = StdCommand::new("qemu-system-arm")
            .args([
                "-cpu",
                "cortex-m7",
                "-machine",
                "mps2-an500",
                "-nographic",
                "-serial",
                "none",
                "-monitor",
                "none",
                "-chardev",
                "stdio,id=con0",
                "-semihosting-config",
                "enable=on,chardev=con0",
                "-kernel",
                elf_path,
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to spawn QEMU process: {}", e))?;

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

            while let Ok(1) = stdout.read(&mut byte_buf) {
                let b = byte_buf[0];
                if let Some(payload) = reader.handle_byte(b) {
                    if let Ok(telemetry) =
                        postcard::from_bytes::<Telemetry<'_>>(payload)
                    {
                        let owned_telemetry = make_telemetry_owned(&telemetry);
                        let _ = tx_stdout
                            .send(BridgeMessage::Telemetry(owned_telemetry));
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
                            let line = String::from_utf8_lossy(&raw_line_buf)
                                .into_owned();
                            let _ =
                                tx_stdout.send(BridgeMessage::RawConsole(line));
                            raw_line_buf.clear();
                        }
                    } else if b != b'\r' {
                        raw_line_buf.push(b);
                    }
                }
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
            target_info: "QEMU (cortex-m7)".to_string(),
            link_info: "Semihosting (mps2-an500)".to_string(),
        })
    }

    pub fn target_info(&self) -> &str {
        &self.target_info
    }

    pub fn link_info(&self) -> &str {
        &self.link_info
    }

    /// Sends a command to the target using the packet framing protocol.
    pub fn send_command(
        &mut self,
        cmd: &Command,
    ) -> Result<(), std::io::Error> {
        let mut payload = postcard::to_allocvec(cmd).unwrap();
        let mut frame = Vec::new();
        frame.push(0xAA);
        frame.push(0x55);
        let len = payload.len() as u16;
        frame.push((len >> 8) as u8);
        frame.push((len & 0xFF) as u8);

        let mut checksum = 0u8;
        for &b in &payload {
            checksum ^= b;
        }
        frame.append(&mut payload);
        frame.push(checksum);

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

    /// Checks if the child process has exited (returns Ok(None) for serial).
    pub fn try_wait(
        &mut self,
    ) -> Result<Option<std::process::ExitStatus>, std::io::Error> {
        match &mut self.inner {
            BridgeInner::Qemu { child, .. } => child.try_wait(),
            BridgeInner::Serial { .. } => Ok(None),
        }
    }

    /// Gets the channel receiver to poll messages from the target.
    pub fn receiver(&self) -> &Receiver<BridgeMessage> {
        &self.rx_from_target
    }

    /// Terminate QEMU (no-op for serial).
    pub fn kill(&mut self) {
        match &mut self.inner {
            BridgeInner::Qemu { child, .. } => {
                let _ = child.kill();
            }
            BridgeInner::Serial { .. } => {}
        }
    }
}

fn make_telemetry_owned(tel: &Telemetry<'_>) -> Telemetry<'static> {
    match *tel {
        Telemetry::SuiteInfo {
            suite_id,
            name,
            test_count,
            setting_count,
        } => Telemetry::SuiteInfo {
            suite_id,
            name: Box::leak(name.to_string().into_boxed_str()),
            test_count,
            setting_count,
        },
        Telemetry::TestInfo {
            suite_id,
            test_id,
            name,
        } => Telemetry::TestInfo {
            suite_id,
            test_id,
            name: Box::leak(name.to_string().into_boxed_str()),
        },
        Telemetry::SettingInfo {
            suite_id,
            setting_id,
            name,
            value,
        } => Telemetry::SettingInfo {
            suite_id,
            setting_id,
            name: Box::leak(name.to_string().into_boxed_str()),
            value,
        },
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
        } => Telemetry::MetricReport {
            suite_id,
            test_id,
            cycles,
            time_us,
        },
        Telemetry::Log(ref msg) => Telemetry::Log(LogMessage {
            timestamp_us: msg.timestamp_us,
            suite_id: msg.suite_id,
            test_id: msg.test_id,
            payload: Box::leak(msg.payload.to_string().into_boxed_str()),
        }),
        Telemetry::TargetPanic {
            message,
            file,
            line,
        } => Telemetry::TargetPanic {
            message: Box::leak(message.to_string().into_boxed_str()),
            file: Box::leak(file.to_string().into_boxed_str()),
            line,
        },
    }
}