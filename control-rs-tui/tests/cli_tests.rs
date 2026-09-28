//! Command-line behavior of the `control-rs-tui` binary.

#![cfg(unix)]

#[cfg(test)]
mod cli {
    use std::fs;
    use std::os::unix::fs::PermissionsExt;
    use std::path::PathBuf;
    use std::process::{Command, Output};

    const BIN: &str = env!("CARGO_BIN_EXE_control-rs-tui");

    /// Runs the binary with a `cargo` that always fails, so a target that
    /// would build and run firmware cannot stall the test.
    fn run(name: &str, args: &[&str]) -> Output {
        let dir: PathBuf = std::env::temp_dir()
            .join(format!("control_rs_tui_cli_{name}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let cargo = dir.join("cargo");
        fs::write(&cargo, "#!/bin/sh\nexit 1\n").unwrap();
        fs::set_permissions(&cargo, fs::Permissions::from_mode(0o755)).unwrap();
        let out = Command::new(BIN)
            .args(args)
            .current_dir(&dir)
            .env("PATH", &dir)
            .output()
            .unwrap();
        let _ = fs::remove_dir_all(&dir);
        out
    }

    fn text(out: &Output) -> String {
        format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        )
    }

    #[test]
    fn help_flags_print_usage_and_exit_cleanly() {
        for flag in ["--help", "-h"] {
            let out = run("help", &[flag]);
            assert_eq!(out.status.code(), Some(0), "{flag}");
            assert!(text(&out).contains("Usage: control-rs-tui"), "{flag}");
        }
    }

    #[test]
    fn other_arguments_do_not_print_the_usage() {
        let out = run("unknown", &["--bogus"]);
        assert_eq!(out.status.code(), Some(1));
        let text = text(&out);
        assert!(text.contains("error:"), "{text}");
        assert!(!text.contains("Usage: control-rs-tui"), "{text}");
    }

    #[test]
    fn several_targets_are_rejected() {
        let out = run("several", &["arm", "riscv32"]);
        assert_eq!(out.status.code(), Some(1));
        assert!(text(&out).contains("Multiple targets"), "{}", text(&out));
    }

    #[test]
    fn a_single_target_is_started() {
        let out = run(
            "single",
            &["teensy", "--port", "/dev/control-rs-tui-no-such-port"],
        );
        assert_eq!(out.status.code(), Some(1));
        let text = text(&out);
        assert!(text.contains("failed to start bridge"), "{text}");
        assert!(!text.contains("Multiple targets"), "{text}");
    }
}
