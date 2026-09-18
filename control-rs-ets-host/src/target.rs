//! Target descriptors and target ELF compilation utilities for host-side ETS.

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::error::HostError;

/// Target details for QEMU.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QemuTargetDetails {
    /// Binary name of the example.
    pub binary_name: &'static str,
    /// Human-readable description of the target.
    pub description: &'static str,
    /// Human readable description of the execution environment.
    pub execution_env: &'static str,
    /// Target triple used by rustc/cargo.
    pub target_triple: &'static str,
}

/// Configuration for running an ETS binary via a subprocess (e.g. `cargo run`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubprocessTarget {
    /// Working directory or crate path (defaults to current directory ".").
    pub path: String,
    /// Target triple (e.g. "thumbv7em-none-eabihf").
    pub target: Option<String>,
    /// Binary name (e.g. "control-rs-qemu-thumbv7em-none-eabihf").
    pub bin: Option<String>,
    /// Additional arguments passed to the runner / cargo (e.g. `["--release"]`).
    pub args: Vec<String>,
    /// Optional display name (e.g. "ARM HF (thumbv7em-none-eabihf)").
    pub name: Option<String>,
}

/// Target QEMU architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QemuArch {
    /// RISC-V 32-bit architecture.
    Riscv32imacUnknownNoneElf,
    /// RISC-V 64-bit architecture.
    Riscv64gcUnknownNoneElf,
    /// ARM Soft-Float architecture.
    Thumbv7emNoneEabi,
    /// ARM Hard-Float architecture.
    Thumbv7emNoneEabihf,
}

/// Target execution platform.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Target {
    /// Subprocess / virtual ETS runner (cargo run, QEMU, simulator).
    Subprocess(SubprocessTarget),
    /// ETS (physical board) target over serial port.
    Serial {
        /// Serial port path (e.g. `/dev/ttyACM0`).
        port: String,
        /// Baud rate (e.g. `115200`).
        baud: u32,
    },
}

impl QemuArch {
    /// Gets the configuration and target details for this architecture.
    #[must_use]
    pub const fn details(&self) -> QemuTargetDetails {
        match self {
            Self::Riscv32imacUnknownNoneElf => QemuTargetDetails {
                binary_name: "control-rs-qemu-riscv32imac-unknown-none-elf",
                description: "QEMU (risc-v32)",
                execution_env: "Semihosting (virt)",
                target_triple: "riscv32imac-unknown-none-elf",
            },
            Self::Riscv64gcUnknownNoneElf => QemuTargetDetails {
                binary_name: "control-rs-qemu-riscv64gc-unknown-none-elf",
                description: "QEMU (risc-v64)",
                execution_env: "Semihosting (virt)",
                target_triple: "riscv64gc-unknown-none-elf",
            },
            Self::Thumbv7emNoneEabi => QemuTargetDetails {
                binary_name: "control-rs-qemu-thumbv7em-none-eabi",
                description: "QEMU (cortex-m7 soft-float)",
                execution_env: "Semihosting (mps2-an500)",
                target_triple: "thumbv7em-none-eabi",
            },
            Self::Thumbv7emNoneEabihf => QemuTargetDetails {
                binary_name: "control-rs-qemu-thumbv7em-none-eabihf",
                description: "QEMU (cortex-m7 hard-float)",
                execution_env: "Semihosting (mps2-an500)",
                target_triple: "thumbv7em-none-eabihf",
            },
        }
    }
}

impl SubprocessTarget {
    /// Creates a new subprocess target for the given crate/directory path.
    #[must_use]
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            target: None,
            bin: None,
            args: Vec::new(),
            name: None,
        }
    }

    /// Sets the target triple.
    #[must_use]
    pub fn with_target(mut self, target: impl Into<String>) -> Self {
        self.target = Some(target.into());
        self
    }

    /// Sets the binary name.
    #[must_use]
    pub fn with_bin(mut self, bin: impl Into<String>) -> Self {
        self.bin = Some(bin.into());
        self
    }

    /// Appends extra arguments.
    #[must_use]
    pub fn with_arg(mut self, arg: impl Into<String>) -> Self {
        self.args.push(arg.into());
        self
    }

    /// Sets the display name.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Returns the directory containing the manifest or crate.
    #[must_use]
    pub fn crate_dir(&self) -> PathBuf {
        let p = Path::new(&self.path);
        if p.is_file() || p.file_name().is_some_and(|n| n == "Cargo.toml") {
            p.parent().unwrap_or_else(|| Path::new(".")).to_path_buf()
        } else {
            p.to_path_buf()
        }
    }

    /// Returns the manifest path (pointing to Cargo.toml).
    #[must_use]
    pub fn manifest_file(&self) -> PathBuf {
        let p = Path::new(&self.path);
        if p.is_file() || p.file_name().is_some_and(|n| n == "Cargo.toml") {
            p.to_path_buf()
        } else {
            p.join("Cargo.toml")
        }
    }

    /// Gets a human-readable display name for this target.
    #[must_use]
    pub fn display_name(&self) -> String {
        self.name.as_ref().map_or_else(
            || match (&self.bin, &self.target) {
                (Some(bin), Some(triple)) => format!("{bin} ({triple})"),
                (Some(bin), None) => bin.clone(),
                (None, Some(triple)) => triple.clone(),
                (None, None) => {
                    if self.path.is_empty() || self.path == "." {
                        "Subprocess (cargo run)".to_string()
                    } else {
                        format!("Subprocess ({})", self.path)
                    }
                }
            },
            Clone::clone,
        )
    }
}

impl Target {
    /// Default QEMU ARM target.
    #[must_use]
    pub fn qemu_arm() -> Self {
        Self::Subprocess(
            SubprocessTarget::new("examples/qemu")
                .with_target("thumbv7em-none-eabihf")
                .with_bin("control-rs-qemu-thumbv7em-none-eabihf")
                .with_name("ARM HF (thumbv7em-none-eabihf)")
                .with_arg("--release"),
        )
    }

    /// Gets a human-readable display name for this target.
    #[must_use]
    pub fn display_name(&self) -> String {
        match self {
            Self::Subprocess(sub) => sub.display_name(),
            Self::Serial { port, .. } => format!("Serial ({port})"),
        }
    }

    /// Parses target parameters from CLI arguments.
    ///
    /// # Errors
    ///
    /// Returns an error if argument parsing fails or target strings are invalid.
    pub fn parse(
        args: &[String],
        default_qemu_arch: &str,
        default_teensy_port: &str,
    ) -> Result<Option<Self>, String> {
        let mut targets = parse_targets(args)?;
        if targets.is_empty() {
            if default_qemu_arch == "all" {
                return Ok(None);
            }
            return Ok(map_shorthand(default_qemu_arch)
                .map(|(t, n)| Self::Subprocess(qemu_example_target(t, n))));
        }
        if targets.len() > 1 {
            return Ok(None);
        }
        let mut target = targets.remove(0);
        if let Self::Serial { ref mut port, .. } = target
            && port == "/dev/teensy"
            && default_teensy_port != "/dev/teensy"
        {
            port.clone_from(&default_teensy_port.to_string());
        }
        Ok(Some(target))
    }
}

/// Maps common target architecture shorthand strings to target triple and description.
#[must_use]
pub fn map_shorthand(s: &str) -> Option<(&'static str, &'static str)> {
    match s {
        "arm" | "arm-hf" | "thumbv7em-none-eabihf" => {
            Some(("thumbv7em-none-eabihf", "ARM HF (thumbv7em-none-eabihf)"))
        }
        "arm-sf" | "arm-soft" | "thumbv7em-none-eabi" => {
            Some(("thumbv7em-none-eabi", "ARM SF (thumbv7em-none-eabi)"))
        }
        "riscv" | "riscv32" | "risc-v" | "riscv32imac-unknown-none-elf" => {
            Some((
                "riscv32imac-unknown-none-elf",
                "RISC-V 32 (riscv32imac-unknown-none-elf)",
            ))
        }
        "riscv64" | "risc-v64" | "riscv64gc-unknown-none-elf" => Some((
            "riscv64gc-unknown-none-elf",
            "RISC-V 64 (riscv64gc-unknown-none-elf)",
        )),
        _ => None,
    }
}

/// Maps a QEMU target triple onto the example firmware binary name.
fn qemu_bin_for_triple(triple: &str) -> Option<&'static str> {
    [
        QemuArch::Riscv32imacUnknownNoneElf,
        QemuArch::Riscv64gcUnknownNoneElf,
        QemuArch::Thumbv7emNoneEabi,
        QemuArch::Thumbv7emNoneEabihf,
    ]
    .iter()
    .map(QemuArch::details)
    .find(|details| details.target_triple == triple)
    .map(|details| details.binary_name)
}

/// Example-firmware subprocess for a QEMU shorthand triple.
fn qemu_example_target(triple: &str, name: &str) -> SubprocessTarget {
    let mut sub = SubprocessTarget::new("examples/qemu")
        .with_target(triple)
        .with_name(name);
    if let Some(bin) = qemu_bin_for_triple(triple) {
        sub = sub.with_bin(bin);
    }
    sub
}

/// Parses CLI arguments into a list of targets.
///
/// # Errors
///
/// Returns an error if argument values are missing or unrecognized.
pub fn parse_targets(args: &[String]) -> Result<Vec<Target>, String> {
    let mut iter = args.iter().peekable();
    if iter.peek().is_some_and(|a| !a.starts_with('-')) {
        iter.next();
    }
    if iter.peek().is_some_and(|a| *a == "ci" || *a == "tui") {
        iter.next();
    }

    let mut path = String::from(".");
    let mut targets = Vec::new();
    let mut extra_args = Vec::new();
    let mut is_serial = false;
    let mut port = None;
    let mut baud = 115_200;
    let mut qemu_mode = false;

    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--" => {
                extra_args.extend(iter.map(Clone::clone));
                break;
            }
            "--manifest-path" | "--path" | "-p" => {
                path.clone_from(
                    iter.next().ok_or("Missing value for --manifest-path")?,
                );
            }
            "--target" | "-t" => {
                let val = iter.next().ok_or("Missing value for --target")?;
                let (triple, bin) = val.split_once(':').map_or_else(
                    || (val.clone(), None),
                    |(t, b)| (t.to_string(), Some(b.to_string())),
                );
                let mut sub = SubprocessTarget::new(".").with_target(triple);
                if let Some(b) = bin {
                    sub = sub.with_bin(b);
                }
                targets.push(sub);
            }
            "--bin" | "-b" => {
                let b = iter.next().ok_or("Missing value for --bin")?.clone();
                if let Some(last) = targets.last_mut() {
                    last.bin = Some(b);
                } else {
                    targets.push(SubprocessTarget::new(".").with_bin(b));
                }
            }
            "--args" => {
                extra_args.push(
                    iter.next().ok_or("Missing value for --args")?.clone(),
                );
            }
            "--release" => {
                if !extra_args.iter().any(|a| a == "--release") {
                    extra_args.push("--release".to_string());
                }
            }
            "--serial" | "teensy" => is_serial = true,
            "--port" => {
                port = Some(
                    iter.next().ok_or("Missing value for --port")?.clone(),
                );
                is_serial = true;
            }
            "--baud" => {
                baud = iter
                    .next()
                    .ok_or("Missing value for --baud")?
                    .parse()
                    .map_err(|e| format!("Invalid baud: {e}"))?;
                is_serial = true;
            }
            "qemu" => qemu_mode = true,
            "all" => {
                for sh in ["arm", "arm-sf", "riscv32", "riscv64"] {
                    if let Some((t, n)) = map_shorthand(sh) {
                        targets.push(qemu_example_target(t, n));
                    }
                }
            }
            p if p.starts_with("/dev/") => {
                port = Some(p.to_string());
                is_serial = true;
            }
            b if is_serial && b.parse::<u32>().is_ok() => {
                if let Ok(val) = b.parse() {
                    baud = val;
                }
            }
            sh if map_shorthand(sh).is_some() => {
                let (t, n) = map_shorthand(sh).unwrap_or(("", ""));
                targets.push(qemu_example_target(t, n));
            }
            other => return Err(format!("Unknown argument: {other}")),
        }
    }

    if is_serial {
        let p = port.unwrap_or_else(|| "/dev/teensy".to_string());
        return Ok(vec![Target::Serial { port: p, baud }]);
    }

    let uses_known_qemu_triple = targets
        .iter()
        .any(|t| t.target.as_deref().and_then(qemu_bin_for_triple).is_some());
    if (qemu_mode || uses_known_qemu_triple) && (path.is_empty() || path == ".")
    {
        path = String::from("examples/qemu");
    }

    for t in &mut targets {
        if t.path.is_empty() || t.path == "." {
            t.path.clone_from(&path);
        }
        if t.bin.is_none()
            && let Some(triple) = t.target.as_deref()
            && let Some(bin) = qemu_bin_for_triple(triple)
        {
            t.bin = Some(bin.to_string());
        }
        for a in &extra_args {
            if !t.args.contains(a) {
                t.args.push(a.clone());
            }
        }
    }

    Ok(targets.into_iter().map(Target::Subprocess).collect())
}

/// Derives the expected ELF path for a subprocess target if binary is known.
#[must_use]
pub fn target_elf_path(target: &SubprocessTarget) -> String {
    let profile = if target.args.iter().any(|a| a == "--release") {
        "release"
    } else {
        "debug"
    };

    let crate_dir = target.crate_dir();
    let dir_str = crate_dir.to_string_lossy();

    match (&target.target, &target.bin) {
        (Some(triple), Some(bin)) => {
            if dir_str.is_empty() || dir_str == "." {
                format!("target/{triple}/{profile}/{bin}")
            } else {
                format!(
                    "{}/target/{triple}/{profile}/{bin}",
                    dir_str.trim_end_matches('/')
                )
            }
        }
        (None, Some(bin)) => {
            if dir_str.is_empty() || dir_str == "." {
                format!("target/{profile}/{bin}")
            } else {
                format!(
                    "{}/target/{profile}/{bin}",
                    dir_str.trim_end_matches('/')
                )
            }
        }
        _ => String::new(),
    }
}

/// Helper function to build the target binary before running ETS or virtual ETS.
///
/// # Errors
///
/// Returns `HostError::Build` if `cargo build` fails or returns a non-zero exit code.
pub fn build_target_elf(
    target: &SubprocessTarget,
) -> Result<String, HostError> {
    let mut cmd = Command::new("cargo");
    let crate_dir = target.crate_dir();
    if !crate_dir.as_os_str().is_empty() && crate_dir != Path::new(".") {
        cmd.current_dir(&crate_dir);
    }
    cmd.arg("build");
    if let Some(bin) = &target.bin {
        cmd.args(["--bin", bin]);
    }
    if let Some(triple) = &target.target {
        cmd.args(["--target", triple]);
    }
    for arg in &target.args {
        cmd.arg(arg);
    }

    let status = cmd.status().map_err(|e| HostError::Build {
        target: target.display_name(),
        source: format!(
            "Failed to spawn cargo build in '{}': {e}",
            crate_dir.display()
        )
        .into(),
    })?;

    if !status.success() {
        return Err(HostError::Build {
            target: target.display_name(),
            source: format!("cargo build exited with status {status}").into(),
        });
    }

    Ok(target_elf_path(target))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qemu_arch_details() {
        let arm_details = QemuArch::Thumbv7emNoneEabihf.details();
        assert_eq!(
            arm_details.binary_name,
            "control-rs-qemu-thumbv7em-none-eabihf"
        );
        let riscv_details = QemuArch::Riscv32imacUnknownNoneElf.details();
        assert_eq!(
            riscv_details.binary_name,
            "control-rs-qemu-riscv32imac-unknown-none-elf"
        );
        let arm_sf = QemuArch::Thumbv7emNoneEabi.details();
        assert_eq!(arm_sf.binary_name, "control-rs-qemu-thumbv7em-none-eabi");
        let riscv64 = QemuArch::Riscv64gcUnknownNoneElf.details();
        assert_eq!(
            riscv64.binary_name,
            "control-rs-qemu-riscv64gc-unknown-none-elf"
        );
    }

    #[test]
    fn test_target_elf_path() {
        let sub = SubprocessTarget::new("examples/qemu")
            .with_target("thumbv7em-none-eabihf")
            .with_bin("control-rs-qemu-thumbv7em-none-eabihf")
            .with_arg("--release");
        assert_eq!(
            target_elf_path(&sub),
            "examples/qemu/target/thumbv7em-none-eabihf/release/control-rs-qemu-thumbv7em-none-eabihf"
        );

        let sub_debug = SubprocessTarget::new(".").with_bin("my-bin");
        assert_eq!(target_elf_path(&sub_debug), "target/debug/my-bin");

        let sub_manifest = SubprocessTarget::new("examples/qemu/Cargo.toml")
            .with_target("thumbv7em-none-eabihf")
            .with_bin("control-rs-qemu-thumbv7em-none-eabihf")
            .with_arg("--release");
        assert_eq!(
            target_elf_path(&sub_manifest),
            "examples/qemu/target/thumbv7em-none-eabihf/release/control-rs-qemu-thumbv7em-none-eabihf"
        );
    }

    #[test]
    fn test_parse_targets_manifest_path() {
        let targets = parse_targets(&[
            "--manifest-path".to_string(),
            "examples/qemu/Cargo.toml".to_string(),
            "--target".to_string(),
            "thumbv7em-none-eabihf".to_string(),
            "--bin".to_string(),
            "control-rs-qemu-thumbv7em-none-eabihf".to_string(),
        ])
        .unwrap();
        assert_eq!(targets.len(), 1);
        if let Target::Subprocess(ref sub) = targets[0] {
            assert_eq!(sub.path, "examples/qemu/Cargo.toml");
            assert_eq!(sub.crate_dir(), Path::new("examples/qemu"));
        } else {
            panic!("Expected Subprocess target");
        }
    }

    #[test]
    fn test_target_parse_none() {
        let t1 = Target::parse(
            &["bin".to_string(), "ci".to_string()],
            "all",
            "/dev/ttyACM0",
        )
        .unwrap();
        assert!(t1.is_none());
    }

    #[test]
    fn test_target_parse_qemu() {
        let t2 = Target::parse(
            &[
                "bin".to_string(),
                "ci".to_string(),
                "qemu".to_string(),
                "arm".to_string(),
            ],
            "arm",
            "/dev/ttyACM0",
        )
        .unwrap()
        .unwrap();
        assert!(matches!(
            t2,
            Target::Subprocess(SubprocessTarget {
                target: Some(ref t),
                ..
            }) if t == "thumbv7em-none-eabihf"
        ));
    }

    #[test]
    /// Shorthand QEMU names must spawn the example firmware, not a
    /// workspace-root `cargo run`.
    ///
    /// # Verification
    /// Trace: ets-host#FR-9
    /// Method: Requirements-based test
    fn test_qemu_shorthand_uses_example_crate() {
        let target = Target::parse(
            &["tui".to_string(), "qemu".to_string(), "risc-v".to_string()],
            "arm",
            "/dev/ttyACM0",
        )
        .unwrap()
        .unwrap();
        if let Target::Subprocess(sub) = target {
            assert!(
                sub.path.contains("examples/qemu"),
                "path = {}, expected examples/qemu",
                sub.path
            );
            assert_eq!(
                sub.bin.as_deref(),
                Some("control-rs-qemu-riscv32imac-unknown-none-elf")
            );
        } else {
            panic!("expected subprocess target");
        }
    }

    #[test]
    /// Architecture shorthand without a `qemu` token still names the example
    /// firmware crate.
    ///
    /// # Verification
    /// Trace: ets-host#FR-9
    /// Method: Requirements-based test
    fn test_parse_targets_shorthand_without_qemu_token() {
        let targets =
            parse_targets(&["tui".to_string(), "arm".to_string()]).unwrap();
        assert_eq!(targets.len(), 1);
        if let Target::Subprocess(sub) = &targets[0] {
            assert!(
                sub.path.contains("examples/qemu"),
                "path = {}, expected examples/qemu",
                sub.path
            );
            assert_eq!(sub.target.as_deref(), Some("thumbv7em-none-eabihf"));
            assert_eq!(
                sub.bin.as_deref(),
                Some("control-rs-qemu-thumbv7em-none-eabihf")
            );
        } else {
            panic!("expected subprocess target");
        }
    }

    #[test]
    /// `--target` with a known QEMU triple and default path `.` uses the
    /// example crate, not workspace-root `cargo run`.
    ///
    /// # Verification
    /// Trace: ets-host#FR-9
    /// Method: Requirements-based test
    fn test_parse_targets_known_triple_defaults_to_example_crate() {
        let targets = parse_targets(&[
            "tui".to_string(),
            "--target".to_string(),
            "thumbv7em-none-eabihf".to_string(),
        ])
        .unwrap();
        assert_eq!(targets.len(), 1);
        if let Target::Subprocess(sub) = &targets[0] {
            assert!(
                sub.path.contains("examples/qemu"),
                "path = {}, expected examples/qemu",
                sub.path
            );
            assert_eq!(
                sub.bin.as_deref(),
                Some("control-rs-qemu-thumbv7em-none-eabihf")
            );
        } else {
            panic!("expected subprocess target");
        }
    }

    #[test]
    fn test_target_parse_teensy() {
        let t3 = Target::parse(
            &[
                "bin".to_string(),
                "ci".to_string(),
                "teensy".to_string(),
                "/dev/ttyUSB0".to_string(),
                "9600".to_string(),
            ],
            "arm",
            "/dev/ttyACM0",
        )
        .unwrap()
        .unwrap();
        if let Target::Serial { port, baud } = t3 {
            assert_eq!(port, "/dev/ttyUSB0");
            assert_eq!(baud, 9600);
        } else {
            panic!("Expected Target::Serial");
        }
    }
}
